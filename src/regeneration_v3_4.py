"""Rate-limited v3.4 regeneration for the five audit-approved revision sources."""

import hashlib
import json
import re
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.generation import CORE_FIELDS, JOB_SCHEMA, GenerationError, parse_structured_response, source_fields
from src.groq_pilot import atomic_json, file_hash, safe_output
from src.regeneration_v3 import MODEL, GroqV3Provider, paced_wait, retry_delay
from src.validation import combine_fields
from src.validation_v3_4 import VERSION, validate_v3_4

TARGET_IDS = [14350, 3203, 4581, 5581, 7192]


class RegenerationV3_4:
    """Creates a separate, resumable five-record follow-up without touching earlier runs."""

    def __init__(self, root, provider=None, wait=paced_wait):
        self.root = Path(root).resolve()
        load_dotenv(self.root / ".env", override=False)
        self.output = self.root / "data/synthetic/pilot/groq_gpt_v3_4"
        self.output.mkdir(parents=True, exist_ok=True)
        self.provider, self.wait = provider, wait
        self.protocol_path = self.root / "prompts/gpt_regeneration_v3_protocol.json"
        self.followup_path = self.root / "prompts/gpt_regeneration_v3_4_followup.json"
        self.prompt_path = self.root / "prompts/ai_fraud_generation_v3_4.txt"
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.followup = json.loads(self.followup_path.read_text(encoding="utf-8"))
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        all_policies = {item["source_row_id"]: item for item in self.protocol["assignments"]}
        if self.followup["source_ids"] != TARGET_IDS:
            raise ValueError("The v3.4 follow-up must retain exactly the five audit-approved source IDs.")
        self.assignments = [all_policies[source_id] for source_id in TARGET_IDS]

        self.v2 = self.root / "data/synthetic/pilot/groq_gpt/openai_gpt-oss-120b"
        self.v3_3 = self.root / "data/synthetic/pilot/groq_gpt_v3_3"
        parents = [json.loads(line) for line in (self.v2 / "ai_fraud_pilot_raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.parents = {record["source_row_id"]: record for record in parents}
        previous = [json.loads(line) for line in (self.v3_3 / "regenerated_records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.previous_v3_3 = {record["source_row_id"]: record for record in previous}
        clean = pd.read_csv(self.root / "data/processed/emscad_clean.csv", dtype=str, keep_default_na=False)
        self.sources = clean.set_index("source_row_id").loc[list(map(str, TARGET_IDS))]
        for source_id in TARGET_IDS:
            if self.sources.loc[str(source_id), "fraudulent"] != "f":
                raise ValueError("Every v3.4 source must remain a legitimate EMSCAD record.")
            if self.parents[source_id]["generation_status"] != "generated":
                raise ValueError("Each v3.4 source requires a successful frozen v2 parent candidate.")
            if self.sources.loc[str(source_id), "text_group_id"] != self.parents[source_id]["source_text_group_id"]:
                raise ValueError("Source text-group provenance mismatch.")
        if not self.sources["text_group_id"].is_unique:
            raise ValueError("Related source records may not share a text group.")

        protected_paths = [self.root / "data/raw/emscad.csv", self.root / "data/processed/emscad_clean.csv"]
        for directory in [
            self.v2,
            self.root / "data/synthetic/pilot/groq_gpt_v3",
            self.root / "data/synthetic/pilot/groq_gpt_v3_1",
            self.root / "data/synthetic/pilot/groq_gpt_v3_2",
            self.v3_3,
            self.root / "data/synthetic/pilot/groq_gpt_v3_3_audit",
            self.root / "data/synthetic/pilot/groq_gpt_v3_4_validation",
        ]:
            protected_paths.extend(path for path in directory.rglob("*") if path.is_file())
        self.protected = {str(path.relative_to(self.root)): file_hash(path) for path in protected_paths}
        self.manifest = {
            "provider": "groq", "model": MODEL, "prompt_version": "v3.4", "validation_version": VERSION,
            "source_ids": TARGET_IDS, "retained_for_human_review": self.followup["retained_for_human_review"],
            "assignments": self.assignments, "temperature": 0.5, "seed": 42, "max_completion_tokens": 4096,
            "request_spacing_seconds": 65, "max_attempts_per_source": 3, "consecutive_rate_limit_stop": 3,
            "prompt_hash": file_hash(self.prompt_path), "protocol_hash": file_hash(self.protocol_path),
            "followup_protocol_hash": file_hash(self.followup_path),
            "code_hashes": {name: file_hash(self.root / "src" / name) for name in [
                "generation.py", "validation.py", "groq_pilot.py", "regeneration_v3.py", "validation_v3.py",
                "validation_v3_1.py", "validation_v3_2.py", "validation_v3_3.py", "validation_v3_4.py",
                "regeneration_v3_4.py"]},
            "protected_hashes": self.protected,
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != self.manifest:
            raise ValueError("Existing v3.4 run configuration changed; refusing to overwrite or mix artifacts.")
        if not manifest_path.exists():
            atomic_json(manifest_path, self.manifest)
        self.state = {"records": [], "stop_reason": None, "finished": False, "next_request_at": 0.0,
                      "events": [], "consecutive_429": 0}
        state_path = self.output / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(record["generation_status"] == "request_in_flight" for record in self.state["records"]):
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
                raise GenerationError("Only openai/gpt-oss-120b is approved for this regeneration.")
            models = self.provider.available_models()
            atomic_json(self.output / "model_availability.json", {
                "checked_at": datetime.now(UTC).isoformat(), "models": models, "selected_model": MODEL,
            })
            if MODEL not in models:
                raise GenerationError("The approved GPT model is not available through this Groq account.")
            return True
        except GenerationError as error:
            self.state["stop_reason"] = safe_output(str(error))
            self.checkpoint()
            return False

    def source(self, source_id):
        source = safe_output(source_fields(self.sources.loc[str(source_id)]))
        return {field: re.sub(r"https?://\S+|www\.\S+|#URL_[^#]+#", "[REDACTED_URL]", value)
                for field, value in source.items()}

    def new_record(self, policy):
        source_id = policy["source_row_id"]
        source = self.source(source_id)
        instruction = {**policy, "mechanism_description": self.protocol["mechanisms"][policy["fraud_mechanism"]]}
        prompt = self.prompt.replace("{{PROTOCOL_JSON}}", json.dumps(instruction, ensure_ascii=True)).replace(
            "{{SOURCE_ADVERTISEMENT_JSON}}", json.dumps(source, ensure_ascii=True))
        return {
            "synthetic_id": "v3_4_" + uuid.uuid5(uuid.NAMESPACE_URL, f"groq:{MODEL}:v3.4:{source_id}").hex,
            "source_row_id": source_id, "source_text_group_id": self.sources.loc[str(source_id), "text_group_id"],
            "source_dataset": "EMSCAD", "source_label": 0, "generated_label": 2,
            "parent_synthetic_id": self.parents[source_id]["synthetic_id"],
            "previous_v3_3_synthetic_id": self.previous_v3_3[source_id]["synthetic_id"],
            "provider": "groq", "model": MODEL, "generation_mode": "api", "generation_timestamp": None,
            "prompt_version": "v3.4", "validation_version": VERSION, "fraud_mechanism": policy["fraud_mechanism"],
            "validation_status": "not_run", "human_review_status": "pending_human_review", "source": source,
            "generated": None, "generation_status": "not_attempted", "request_prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(), "validation": {}, "attempts": [],
        }

    def generate_one(self, policy):
        source_id = policy["source_row_id"]
        existing = next((record for record in self.state["records"] if record["source_row_id"] == source_id), None)
        if existing or self.state["stop_reason"]:
            return existing
        record = self.new_record(policy)
        self.state["records"].append(record)
        for attempt in range(1, 4):
            self.wait(max(0.0, self.state["next_request_at"] - time.time()))
            record["generation_status"] = "request_in_flight"
            record["generation_timestamp"] = datetime.now(UTC).isoformat()
            self.checkpoint()
            error_message, status_code, transient, headers = None, None, False, {}
            self.state["next_request_at"] = time.time() + 65
            try:
                record["generated"] = parse_structured_response(self.provider.generate_json(record["request_prompt"], JOB_SCHEMA))
                record["generation_status"] = "generated"
                self.state["consecutive_429"] = 0
            except Exception as error:
                record["generation_status"] = "failed"
                error_message = safe_output(str(error))
                status_code, headers = getattr(error, "status_code", None), getattr(error, "rate_headers", {})
                transient = getattr(error, "transient", False)
                self.state["consecutive_429"] = self.state["consecutive_429"] + 1 if status_code == 429 else 0
            event = {"attempt": attempt, "timestamp": record["generation_timestamp"], "http_status": status_code,
                     "error": error_message, "rate_headers": headers,
                     "response": safe_output(getattr(self.provider, "last_response", {}))}
            if transient:
                delay, basis = retry_delay(headers, attempt)
                event.update(retry_delay_seconds=delay, retry_delay_basis=basis)
                self.state["next_request_at"] = max(self.state["next_request_at"], time.time() + delay)
                if delay > 300:
                    self.state["stop_reason"] = "Server requested a wait above 300 seconds; stopped safely."
            record["attempts"].append(event)
            if self.state["consecutive_429"] >= 3:
                self.state["stop_reason"] = "Stopped after three consecutive HTTP 429 responses."
            self.checkpoint()
            if record["generation_status"] == "generated" or not transient or attempt == 3 or self.state["stop_reason"]:
                break
        record["validation"] = validate_v3_4(record["source"], record["generated"], policy)
        record["validation_status"] = record["validation"]["validation_status"]
        record["human_review_status"] = "pending_human_review"
        self.checkpoint()
        return record

    def run_all(self):
        for policy in self.assignments:
            if self.state["stop_reason"]:
                break
            record = self.generate_one(policy)
            self.export()
            if record:
                print(f"Source {record['source_row_id']}: {record['generation_status']} / {record['validation_status']}", flush=True)
        self.state["finished"] = len(self.state["records"]) == len(TARGET_IDS) and not self.state["stop_reason"]
        self.checkpoint()

    def export(self):
        records = self.state["records"]
        jsonl = "".join(json.dumps(safe_output(record), ensure_ascii=True) + "\n" for record in records)
        temporary = self.output / "regenerated_records.jsonl.tmp"
        temporary.write_text(jsonl, encoding="utf-8")
        temporary.replace(self.output / "regenerated_records.jsonl")
        rows, validations, reviews = [], [], []
        for record in records:
            row = {key: value for key, value in record.items()
                   if key not in {"source", "generated", "validation", "attempts", "request_prompt"}}
            row.update({field: (record["generated"] or {}).get(field, "") for field in CORE_FIELDS})
            row["text"] = combine_fields(record["generated"] or {})
            row["attempts"] = len(record["attempts"])
            rows.append(row)
            validations.append({"source_row_id": record["source_row_id"], "synthetic_id": record["synthetic_id"],
                                **{key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value
                                   for key, value in record["validation"].items()}})
            reviews.append({
                "synthetic_id": record["synthetic_id"], "source_row_id": record["source_row_id"],
                "source_text_group_id": record["source_text_group_id"], "generated_title": row["title"],
                "generated_text": row["text"], "fraud_mechanism": record["fraud_mechanism"],
                "automatic_validation_result": record["validation_status"], "review_context_preservation": "",
                "review_fraud_intent": "", "review_realism": "", "review_unsupported_facts": "",
                "review_research_meta_artifacts": "", "review_ai_artifacts": "", "review_decision": "",
                "reviewer_notes": "", "review_guidance": "Yes/Partial/No; decision: Accept/Revise/Reject",
            })
        pd.DataFrame(rows).to_csv(self.output / "regenerated_records.csv", index=False)
        pd.DataFrame(validations).to_csv(self.output / "validation_results.csv", index=False)
        pd.DataFrame(reviews).to_csv(self.output / "review_template.csv", index=False)
        generated = [record for record in records if record["generation_status"] == "generated"]
        statuses = Counter(record["validation_status"] for record in generated)
        summary = {
            "provider": "groq", "model": MODEL, "prompt_version": "v3.4", "validation_version": VERSION,
            "selected": len(TARGET_IDS), "attempted": len(records), "generated": len(generated),
            "generation_failures": sum(record["generation_status"] == "failed" for record in records),
            "not_attempted": len(TARGET_IDS) - len(records), "automatic_passes": statuses["auto_pass"],
            "automatic_rejections": statuses["auto_reject"], "needs_human_review": statuses["needs_human_review"],
            "records_requiring_human_review": len(generated), "human_review_status": "pending_human_review",
            "rate_limit_responses": sum(event["http_status"] == 429 for record in records for event in record["attempts"]),
            "generation_requests": sum(len(record["attempts"]) for record in records),
            "stop_reason": self.state["stop_reason"], "completed_all_five": self.state["finished"],
            "fraud_mechanism_distribution": dict(Counter(record["fraud_mechanism"] for record in generated)),
            "structural_validation_failures": sum(not record["validation"].get("structural_valid", False) for record in generated),
            "unsupported_fact_failures": sum(not record["validation"].get("unsupported_facts_clear", True) for record in generated),
            "employment_arrangement_failures": sum(any("employment arrangement" in reason for reason in record["validation"].get("reasons", [])) for record in generated),
            "research_meta_artifact_failures": sum(not record["validation"].get("research_artifacts_clear", True) for record in generated),
            "ai_artifact_failures": sum(not record["validation"].get("ai_artifacts_clear", True) for record in generated),
            "contextual_integration_failures": sum(not record["validation"].get("fraud_integrated", True) for record in generated),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        atomic_json(self.output / "generation_summary.json", summary)
        for path, digest in self.protected.items():
            if file_hash(self.root / path) != digest:
                raise RuntimeError(f"Frozen artifact changed: {path}")
        return summary


if __name__ == "__main__":
    run = RegenerationV3_4(Path(__file__).resolve().parents[1])
    if run.configure():
        run.run_all()
    print(json.dumps(run.export(), indent=2))
