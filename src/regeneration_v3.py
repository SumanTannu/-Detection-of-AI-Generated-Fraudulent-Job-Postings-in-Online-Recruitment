"""Versioned 12-source regeneration; v2 files and implementation remain read only."""

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.generation import CORE_FIELDS, JOB_SCHEMA, GenerationError, GroqProvider, parse_structured_response, source_fields
from src.groq_pilot import atomic_json, file_hash, safe_output
from src.validation import combine_fields
from src.validation_v3 import validate_v3

TARGET_IDS = [11662, 12066, 13604, 14350, 14484, 15885, 3203, 4581, 5581, 6199, 7192, 9208]
MODEL = "openai/gpt-oss-120b"


def retry_delay(headers, attempt, now=None):
    """Honor Retry-After seconds/HTTP-date; otherwise 65, 130 seconds backoff."""
    value = headers.get("retry-after") if headers else None
    if value:
        try:
            return max(1.0, float(value)), "retry-after"
        except ValueError:
            try:
                delta = (parsedate_to_datetime(value) - (now or datetime.now(UTC))).total_seconds()
                return max(1.0, delta), "retry-after-date"
            except (TypeError, ValueError, OverflowError):
                pass
    return 65.0 * 2 ** (attempt - 1), "exponential-backoff"


def paced_wait(seconds):
    """Use short chunks so a long backoff remains observable and interruptible."""
    while seconds > 0:
        chunk = min(seconds, 30.0)
        print(f"Rate pacing: waiting {chunk:.1f}s ({seconds:.1f}s remaining).", flush=True)
        time.sleep(chunk)
        seconds -= chunk


class GroqV3Provider(GroqProvider):
    """Retain structured output and expose HTTP rate metadata to the v3 runner."""

    def generate_json(self, prompt, schema):
        self.last_response = {}
        try:
            raw = self.client.chat.completions.with_raw_response.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "recruitment_record", "strict": True, "schema": schema}},
                max_completion_tokens=4096, temperature=0.5, seed=42,
            )
            response = raw.parse()
            choice = response.choices[0]
            self.last_response = {
                "response_id": response.id, "response_model": response.model,
                "finish_reason": choice.finish_reason,
                "raw_response": choice.message.content or "",
                "usage": response.usage.model_dump() if response.usage else None,
                "rate_headers": {k: v for k, v in raw.headers.items() if k.startswith("x-ratelimit")},
            }
            if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
                raise GenerationError(f"Incomplete/refused response: {choice.finish_reason}")
            return parse_structured_response(choice.message.content or "")
        except Exception as error:
            failure = GenerationError(self.error_message(error))
            failure.status_code = getattr(error, "status_code", None)
            headers = getattr(getattr(error, "response", None), "headers", {})
            failure.rate_headers = {k.lower(): v for k, v in headers.items()
                                    if k.lower() == "retry-after" or k.lower().startswith("x-ratelimit")}
            failure.transient = failure.status_code in {429, 500, 502, 503, 504}
            raise failure from None


class RegenerationV3:
    def __init__(self, root, provider=None, wait=paced_wait):
        self.root = Path(root).resolve()
        load_dotenv(self.root / ".env", override=False)
        self.output = self.root / "data/synthetic/pilot/groq_gpt_v3"
        self.output.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.wait = wait
        self.protocol_path = self.root / "prompts/gpt_regeneration_v3_protocol.json"
        self.prompt_path = self.root / "prompts/ai_fraud_generation_v3.txt"
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.assignments = self.protocol["assignments"]
        if [p["source_row_id"] for p in self.assignments] != TARGET_IDS:
            raise ValueError("Protocol must contain exactly the 12 approved source IDs in frozen order.")
        self.v2 = self.root / "data/synthetic/pilot/groq_gpt/openai_gpt-oss-120b"
        old = [json.loads(line) for line in (self.v2 / "ai_fraud_pilot_raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.parents = {r["source_row_id"]: r for r in old}
        clean = pd.read_csv(self.root / "data/processed/emscad_clean.csv", dtype=str, keep_default_na=False)
        self.sources = clean.set_index("source_row_id").loc[list(map(str, TARGET_IDS))]
        for sid in TARGET_IDS:
            if self.sources.loc[str(sid), "fraudulent"] != "f" or self.parents[sid]["generation_status"] != "generated":
                raise ValueError("Sources must be legitimate EMSCAD rows with successful v2 parent candidates.")
            if self.sources.loc[str(sid), "text_group_id"] != self.parents[sid]["source_text_group_id"]:
                raise ValueError("Source group provenance mismatch.")
        if not self.sources["text_group_id"].is_unique:
            raise ValueError("Duplicate source groups in regeneration request.")
        protected_paths = [self.root / "data/raw/emscad.csv", self.root / "data/processed/emscad_clean.csv"]
        protected_paths += [p for p in self.v2.rglob("*") if p.is_file()]
        protected_paths += [p for p in (self.root / "data/synthetic/pilot").glob("*.*") if p.is_file()]
        self.protected = {str(p.relative_to(self.root)): file_hash(p) for p in protected_paths}
        self.manifest = {
            "provider": "groq", "model": MODEL, "prompt_version": "v3", "validation_version": "v3",
            "source_ids": TARGET_IDS, "smoke_source_id": TARGET_IDS[0], "excluded_source_id": 10308,
            "assignments": self.assignments, "temperature": 0.5, "seed": 42,
            "max_completion_tokens": 4096, "request_spacing_seconds": 65,
            "max_attempts_per_source": 3, "consecutive_rate_limit_stop": 3,
            "prompt_hash": file_hash(self.prompt_path), "protocol_hash": file_hash(self.protocol_path),
            "code_hashes": {f: file_hash(self.root / "src" / f) for f in
                            ["generation.py", "validation.py", "groq_pilot.py", "regeneration_v3.py", "validation_v3.py"]},
            "protected_hashes": self.protected,
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != self.manifest:
            raise ValueError("Frozen v3 run changed; refusing to mix versions or overwrite outputs.")
        if not manifest_path.exists():
            atomic_json(manifest_path, self.manifest)
        self.state = {"records": [], "smoke_status": "not_run", "stop_reason": None,
                      "finished": False, "next_request_at": 0.0, "events": [], "consecutive_429": 0}
        state_path = self.output / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(r["generation_status"] == "request_in_flight" for r in self.state["records"]):
            self.state["stop_reason"] = "Interrupted request has unknown outcome; no automatic reissue."
        self.checkpoint()

    def checkpoint(self):
        atomic_json(self.output / "state.json", safe_output(self.state))

    def configure(self):
        if self.state["stop_reason"] or self.state["finished"]:
            return False
        try:
            self.provider = self.provider or GroqV3Provider(MODEL)
            if self.provider.model != MODEL:
                raise GenerationError("Only the approved GPT model can run this regeneration.")
            models = self.provider.available_models()
            atomic_json(self.output / "model_availability.json", {"checked_at": datetime.now(UTC).isoformat(), "models": models})
            if MODEL not in models:
                raise GenerationError("Approved GPT model is unavailable.")
            return True
        except GenerationError as error:
            self.state["stop_reason"] = safe_output(str(error))
            self.checkpoint()
            return False

    def source(self, sid):
        source = safe_output(source_fields(self.sources.loc[str(sid)]))
        return {f: re.sub(r"https?://\S+|www\.\S+|#URL_[^#]+#", "[REDACTED_URL]", s) for f, s in source.items()}

    def new_record(self, policy):
        sid = policy["source_row_id"]
        source = self.source(sid)
        instruction = {**policy, "mechanism_description": self.protocol["mechanisms"][policy["fraud_mechanism"]]}
        prompt = self.prompt.replace("{{PROTOCOL_JSON}}", json.dumps(instruction, ensure_ascii=True)).replace(
            "{{SOURCE_ADVERTISEMENT_JSON}}", json.dumps(source, ensure_ascii=True))
        return {
            "synthetic_id": "v3_" + uuid.uuid5(uuid.NAMESPACE_URL, f"groq:{MODEL}:v3:{sid}").hex,
            "source_row_id": sid, "source_text_group_id": self.sources.loc[str(sid), "text_group_id"],
            "source_dataset": "EMSCAD", "source_label": 0, "generated_label": 2,
            "parent_synthetic_id": self.parents[sid]["synthetic_id"], "provider": "groq", "model": MODEL,
            "generation_mode": "api", "generation_timestamp": None, "prompt_version": "v3",
            "validation_version": "v3", "fraud_mechanism": policy["fraud_mechanism"],
            "validation_status": "not_run", "human_review_status": "not_reviewed",
            "source": source, "generated": None, "generation_status": "not_attempted",
            "request_prompt": prompt, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "validation": {}, "attempts": [],
        }

    def generate_one(self, policy):
        sid = policy["source_row_id"]
        previous = next((r for r in self.state["records"] if r["source_row_id"] == sid), None)
        if previous:
            return previous
        if self.state["stop_reason"]:
            return None
        record = self.new_record(policy)
        self.state["records"].append(record)
        for attempt in range(1, 4):
            self.wait(max(0.0, self.state["next_request_at"] - time.time()))
            record["generation_status"] = "request_in_flight"
            record["generation_timestamp"] = datetime.now(UTC).isoformat()
            self.checkpoint()
            error_message, code, retry = None, None, False
            headers = {}
            self.state["next_request_at"] = time.time() + 65
            try:
                record["generated"] = parse_structured_response(self.provider.generate_json(record["request_prompt"], JOB_SCHEMA))
                record["generation_status"] = "generated"
                self.state["consecutive_429"] = 0
            except Exception as error:
                record["generation_status"] = "failed"
                error_message = safe_output(str(error))
                code = getattr(error, "status_code", None)
                headers = getattr(error, "rate_headers", {})
                retry = getattr(error, "transient", False)
                self.state["consecutive_429"] = self.state["consecutive_429"] + 1 if code == 429 else 0
            event = {"attempt": attempt, "timestamp": record["generation_timestamp"], "http_status": code,
                     "error": error_message, "rate_headers": headers,
                     "response": safe_output(getattr(self.provider, "last_response", {}))}
            if retry:
                delay, basis = retry_delay(headers, attempt)
                event.update(retry_delay_seconds=delay, retry_delay_basis=basis)
                self.state["next_request_at"] = max(self.state["next_request_at"], time.time() + delay)
                if delay > 300:
                    self.state["stop_reason"] = "Server requests a wait above 300s; stopped without early retry."
            record["attempts"].append(event)
            if self.state["consecutive_429"] >= 3:
                self.state["stop_reason"] = "Stopped after three consecutive HTTP 429 responses."
            self.checkpoint()
            if record["generation_status"] == "generated" or not retry or attempt == 3 or self.state["stop_reason"]:
                break
        record["validation"] = validate_v3(record["source"], record["generated"], policy)
        record["validation_status"] = record["validation"]["validation_status"]
        self.checkpoint()
        return record

    def smoke_test(self):
        if self.state["smoke_status"] != "not_run":
            return self.state["smoke_status"] == "passed"
        if self.state["stop_reason"]:
            return False
        if self.provider is None:
            raise RuntimeError("Configure the provider before generation.")
        record = self.generate_one(self.assignments[0])
        passed = bool(record and record["validation"].get("automatic_passed"))
        self.state["smoke_status"] = "passed" if passed else "failed"
        if not passed:
            self.state["stop_reason"] = self.state["stop_reason"] or "Smoke candidate failed full v3 validation; remaining 11 not generated."
        self.checkpoint()
        self.export()
        return passed

    def remaining(self):
        if self.state["smoke_status"] != "passed" or self.state["stop_reason"] or self.state["finished"]:
            return
        for policy in self.assignments[1:]:
            if self.state["stop_reason"]:
                break
            record = self.generate_one(policy)
            self.export()
            if record:
                print(f"Source {record['source_row_id']}: {record['validation_status']}", flush=True)
        self.state["finished"] = len(self.state["records"]) == 12
        self.checkpoint()

    def export(self):
        records = self.state["records"]
        paths = self.output
        temporary = paths / "regenerated_records.jsonl.tmp"
        temporary.write_text("".join(json.dumps(safe_output(r), ensure_ascii=True) + "\n" for r in records), encoding="utf-8")
        temporary.replace(paths / "regenerated_records.jsonl")
        rows, reviews, validations = [], [], []
        for policy in self.assignments:
            record = next((r for r in records if r["source_row_id"] == policy["source_row_id"]), self.new_record(policy))
            row = {k: v for k, v in record.items() if k not in {"source", "generated", "validation", "attempts", "request_prompt"}}
            row.update({f: (record["generated"] or {}).get(f, "") for f in CORE_FIELDS})
            row["text"] = combine_fields(record["generated"] or {})
            row["attempts"] = len(record["attempts"])
            if record["generation_status"] != "not_attempted":
                rows.append(row)
                validations.append({"source_row_id": record["source_row_id"], "synthetic_id": record["synthetic_id"],
                                    **{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in record["validation"].items()}})
            reviews.append({"synthetic_id": record["synthetic_id"], "source_row_id": record["source_row_id"],
                            "source_text_group_id": record["source_text_group_id"], "source_text": combine_fields(record["source"]),
                            "generated_title": row["title"], "generated_text": row["text"],
                            "fraud_mechanism": row["fraud_mechanism"], "validation_status": row["validation_status"],
                            "generation_status": row["generation_status"], "reasons": json.dumps(record["validation"].get("reasons", [])),
                            "review_context": "", "review_fraud_intent": "", "review_coherence": "", "review_fidelity": "",
                            "review_identity_artifacts": "", "review_decision": "", "reviewer_notes": ""})
        pd.DataFrame(rows, columns=list(row)).to_csv(paths / "regenerated_records.csv", index=False)
        pd.DataFrame(validations, columns=list(validations[0]) if validations else ["source_row_id", "validation_status", "reasons"]).to_csv(paths / "validation_results.csv", index=False)
        review_frame = pd.DataFrame(reviews)
        review_path = paths / "review_template.csv"
        if review_path.exists():
            old = pd.read_csv(review_path, dtype=str, keep_default_na=False).set_index("synthetic_id")
            for field in [c for c in review_frame.columns if c.startswith("review_") or c == "reviewer_notes"]:
                review_frame[field] = review_frame["synthetic_id"].map(old[field]).fillna("")
        review_frame.to_csv(review_path, index=False)
        successful = [r for r in records if r["generation_status"] == "generated"]
        passed = [r for r in records if r["validation"].get("automatic_passed")]
        summary = {
            "provider": "groq", "model": MODEL, "prompt_version": "v3", "validation_version": "v3",
            "selected": 12, "attempted": len(records), "generated": len(successful),
            "generation_failures": sum(r["generation_status"] == "failed" for r in records),
            "not_attempted": 12 - len(records), "automatic_passes": len(passed),
            "automatic_rejections": sum(r["validation_status"] == "auto_reject" for r in records),
            "human_accepted": 0, "human_review_status": "not_reviewed",
            "rate_limit_responses": sum(a["http_status"] == 429 for r in records for a in r["attempts"]),
            "generation_requests": sum(len(r["attempts"]) for r in records),
            "smoke_status": self.state["smoke_status"], "stop_reason": self.state["stop_reason"],
            "completed_all_12": self.state["finished"],
            "assigned_mechanism_distribution": dict(Counter(p["fraud_mechanism"] for p in self.assignments)),
            "generated_mechanism_distribution": dict(Counter(r["fraud_mechanism"] for r in successful)),
            "passing_mechanism_distribution": dict(Counter(r["fraud_mechanism"] for r in passed)),
        }
        atomic_json(paths / "generation_summary.json", summary)
        for path, digest in self.protected.items():
            if file_hash(self.root / path) != digest:
                raise RuntimeError(f"Frozen artifact changed: {path}")
        return summary


if __name__ == "__main__":
    run = RegenerationV3(Path(__file__).resolve().parents[1])
    if run.configure() and run.smoke_test():
        run.remaining()
    print(json.dumps(run.export(), indent=2))
