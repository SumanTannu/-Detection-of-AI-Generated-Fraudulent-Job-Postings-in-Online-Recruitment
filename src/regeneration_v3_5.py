"""Controlled v3.5 regeneration with attempt-level validation and review simulation."""

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
from src.review_v3_5 import VERSION as REVIEW_VERSION, failure_signature, review_candidate
from src.validation import combine_fields
from src.validation_v3_4 import VERSION as VALIDATION_VERSION, validate_v3_4

TARGET_IDS = [14350, 3203, 4581, 7192]


class RegenerationV3_5:
    """Run no more than ten auditable attempts per source, without manual edits."""

    def __init__(self, root, provider=None, wait=paced_wait):
        self.root = Path(root).resolve()
        load_dotenv(self.root / ".env", override=False)
        self.output = self.root / "data/synthetic/pilot/groq_gpt_v3_5"
        self.output.mkdir(parents=True, exist_ok=True)
        self.provider, self.wait = provider, wait
        self.protocol_path = self.root / "prompts/gpt_regeneration_v3_protocol.json"
        self.followup_path = self.root / "prompts/gpt_regeneration_v3_5_followup.json"
        self.prompt_path = self.root / "prompts/ai_fraud_generation_v3_5.txt"
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.followup = json.loads(self.followup_path.read_text(encoding="utf-8"))
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        if self.followup["source_ids"] != TARGET_IDS:
            raise ValueError("v3.5 may regenerate only the four reviewed revision sources.")
        self.max_attempts = int(self.followup["max_attempts"])
        policies = {item["source_row_id"]: item for item in self.protocol["assignments"]}
        self.assignments = [policies[source_id] for source_id in TARGET_IDS]
        self.v2 = self.root / "data/synthetic/pilot/groq_gpt/openai_gpt-oss-120b"
        self.v3_4 = self.root / "data/synthetic/pilot/groq_gpt_v3_4"
        parents = [json.loads(line) for line in (self.v2 / "ai_fraud_pilot_raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.parents = {record["source_row_id"]: record for record in parents}
        previous = [json.loads(line) for line in (self.v3_4 / "regenerated_records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.previous_v3_4 = {record["source_row_id"]: record for record in previous}
        clean = pd.read_csv(self.root / "data/processed/emscad_clean.csv", dtype=str, keep_default_na=False)
        self.sources = clean.set_index("source_row_id").loc[list(map(str, TARGET_IDS))]
        for source_id in TARGET_IDS:
            if self.sources.loc[str(source_id), "fraudulent"] != "f":
                raise ValueError("v3.5 may use legitimate EMSCAD records only.")
            if self.parents[source_id]["generation_status"] != "generated":
                raise ValueError("Each source must retain a successful v2 parent candidate.")
            if self.sources.loc[str(source_id), "text_group_id"] != self.parents[source_id]["source_text_group_id"]:
                raise ValueError("Source text-group provenance mismatch.")
        if not self.sources["text_group_id"].is_unique:
            raise ValueError("Related source records must not share a text group.")
        self.protected = self._protected_hashes()
        existing_state_path = self.output / "state.json"
        prior_attempt_count = 0
        if existing_state_path.exists():
            prior_attempt_count = len(json.loads(existing_state_path.read_text(encoding="utf-8")).get("attempts", []))
        self.manifest = {
            "provider": "groq", "model": MODEL, "prompt_version": "v3.5", "validation_version": VALIDATION_VERSION,
            "review_version": REVIEW_VERSION, "source_ids": TARGET_IDS, "max_attempts": self.max_attempts,
            "assignments": self.assignments, "temperature": 0.5, "seed": 42, "max_completion_tokens": 4096,
            "request_spacing_seconds": 65, "max_attempts_per_request": 3, "consecutive_rate_limit_stop": 3,
            "prompt_hash": file_hash(self.prompt_path), "protocol_hash": file_hash(self.protocol_path),
            "followup_protocol_hash": file_hash(self.followup_path), "protected_hashes": self.protected,
            "code_hashes": {name: file_hash(self.root / "src" / name) for name in [
                "generation.py", "validation.py", "validation_v3.py", "validation_v3_1.py", "validation_v3_2.py",
                "validation_v3_3.py", "validation_v3_4.py", "review_v3_5.py", "regeneration_v3.py", "regeneration_v3_5.py"]},
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest != self.manifest:
                # Guidance implementations may evolve between failed attempts. Preserve
                # the prior hash and allow only this runner's code hash to change.
                old_core = {key: value for key, value in existing_manifest.items()
                            if key not in {"code_hashes", "implementation_revisions"}}
                new_core = {key: value for key, value in self.manifest.items() if key != "code_hashes"}
                old_hashes = existing_manifest.get("code_hashes", {})
                new_hashes = self.manifest["code_hashes"]
                changed = {key for key in set(old_hashes) | set(new_hashes) if old_hashes.get(key) != new_hashes.get(key)}
                allowed_guidance_files = {"regeneration_v3_5.py", "review_v3_5.py"}
                if old_core != new_core or (changed and not changed.issubset(allowed_guidance_files)):
                    raise ValueError("Existing v3.5 configuration changed beyond the versioned guidance implementation.")
                if not changed:
                    self.manifest = existing_manifest
                    existing_manifest = None
                if existing_manifest is None:
                    pass
                else:
                    revisions = existing_manifest.setdefault("implementation_revisions", [])
                    revisions.append({"timestamp": datetime.now(UTC).isoformat(), "changed_files": sorted(changed),
                                      "previous_sha256": {name: old_hashes.get(name) for name in sorted(changed)},
                                      "current_sha256": {name: new_hashes[name] for name in sorted(changed)},
                                      "applies_after_attempt_count": prior_attempt_count,
                                      "reason": "Versioned diagnostic guidance or review-gate correction after a recorded attempt."})
                    existing_manifest["code_hashes"] = new_hashes
                    atomic_json(manifest_path, existing_manifest)
                    self.manifest = existing_manifest
            else:
                self.manifest = existing_manifest
        else:
            atomic_json(manifest_path, self.manifest)
        self.state = {"attempts": [], "final": {}, "stop_reason": None, "finished": False,
                      "next_request_at": 0.0, "consecutive_429": 0, "events": []}
        if (self.output / "state.json").exists():
            self.state = json.loads((self.output / "state.json").read_text(encoding="utf-8"))
        if any(attempt["generation_status"] == "request_in_flight" for attempt in self.state["attempts"]):
            self.state["stop_reason"] = "Interrupted request has unknown outcome; no automatic reissue."
        self.checkpoint()

    def _protected_hashes(self):
        files = [self.root / "data/raw/emscad.csv", self.root / "data/processed/emscad_clean.csv"]
        for directory in [
            self.v2, self.root / "data/synthetic/pilot/groq_gpt_v3",
            self.root / "data/synthetic/pilot/groq_gpt_v3_1", self.root / "data/synthetic/pilot/groq_gpt_v3_2",
            self.root / "data/synthetic/pilot/groq_gpt_v3_3", self.root / "data/synthetic/pilot/groq_gpt_v3_3_audit",
            self.v3_4, self.root / "data/synthetic/pilot/groq_gpt_v3_4_validation",
        ]:
            files.extend(path for path in directory.rglob("*") if path.is_file())
        return {str(path.relative_to(self.root)): file_hash(path) for path in files}

    def _verify_protected(self):
        mismatches = [path for path, digest in self.protected.items() if file_hash(self.root / path) != digest]
        if mismatches:
            raise RuntimeError("Protected artifact changed: " + ", ".join(mismatches))
        return mismatches

    def checkpoint(self):
        atomic_json(self.output / "state.json", safe_output(self.state))

    def configure(self):
        if self.state["stop_reason"] or self.state["finished"]:
            return False
        try:
            self.provider = self.provider or GroqV3Provider(MODEL)
            if self.provider.model != MODEL:
                raise GenerationError("Only openai/gpt-oss-120b is approved for this controlled run.")
            models = self.provider.available_models()
            atomic_json(self.output / "model_availability.json", {"checked_at": datetime.now(UTC).isoformat(),
                        "models": models, "selected_model": MODEL})
            if MODEL not in models:
                raise GenerationError("The approved GPT model is unavailable on this Groq account.")
            return True
        except GenerationError as error:
            self.state["stop_reason"] = safe_output(str(error))
            self.checkpoint()
            return False

    def source(self, source_id):
        source = safe_output(source_fields(self.sources.loc[str(source_id)]))
        return {field: re.sub(r"https?://\S+|www\.\S+|#URL_[^#]+#", "[REDACTED_URL]", value)
                for field, value in source.items()}

    def attempts_for(self, source_id):
        return [attempt for attempt in self.state["attempts"] if attempt["source_row_id"] == source_id]

    def _guidance(self, source_id):
        prior = self.attempts_for(source_id)
        policy = next(item for item in self.assignments if item["source_row_id"] == source_id)
        if not prior:
            return "Use the assigned mechanism inside an existing recruitment workflow sentence; preserve all source facts."
        signature = prior[-1].get("failure_signature", ["other"])
        guidance = ["Prior failure categories: " + ", ".join(signature) + "."]
        if "warning_append" in signature or "integration" in signature:
            guidance.append("Put the mechanism within the main application, screening, interview, verification, scheduling or onboarding narrative, never after a warning marker or as the last add-on.")
        if "unsupported_fact" in signature:
            guidance.append("Omit distinctive employer/institution names and do not add facts absent from the source.")
        if "similarity" in signature:
            guidance.append("Rephrase source material at sentence level while retaining the factual anchors.")
        if "compensation" in signature:
            guidance.append("Preserve compensation statements separately and do not make a salary range.")
        if "employment" in signature:
            guidance.append("Keep every employment arrangement and hour claim exactly as the source supports.")
        if "fraud_intent" in signature:
            guidance.append("Make the assigned mechanism unmistakable in the recruitment workflow without adding another mechanism.")
        if "fraud_intent" in signature:
            if policy["fraud_mechanism"] == "urgency_pressure":
                guidance.append("For urgency_pressure, the main recruitment narrative must explicitly state that the applicant confirms interest within a short deadline BEFORE written terms are provided or normal screening begins. Preserve that ordering exactly; do not say terms are withheld until confirmation. Do not invent a calendar date.")
            elif policy["fraud_mechanism"] == "payment_related":
                guidance.append("For payment_related, place both the processing-charge condition and the withheld interview/screening progression within the application-processing narrative, without an amount or destination.")
            elif policy["fraud_mechanism"] == "unusual_financial_arrangements":
                guidance.append("For unusual_financial_arrangements, link both a personal-account condition and company-funds wording to an onboarding verification step before standard employment checks; give no transfer instructions or destination.")
        missing_anchors = [reason for reason in prior[-1].get("validation", {}).get("reasons", [])
                           if reason.startswith("Missing source")]
        if missing_anchors:
            guidance.append("Restore every required source anchor explicitly. The source facts to retain include: " + " ".join(policy["facts"]))
        return " ".join(guidance)

    def new_attempt(self, policy):
        source_id = policy["source_row_id"]
        number = len(self.attempts_for(source_id)) + 1
        source = self.source(source_id)
        instruction = {**policy, "mechanism_description": self.protocol["mechanisms"][policy["fraud_mechanism"]]}
        prompt = self.prompt.replace("{{ATTEMPT_NUMBER}}", str(number)).replace("{{ATTEMPT_GUIDANCE}}", self._guidance(source_id))
        prompt = prompt.replace("{{PROTOCOL_JSON}}", json.dumps(instruction, ensure_ascii=True)).replace(
            "{{SOURCE_ADVERTISEMENT_JSON}}", json.dumps(source, ensure_ascii=True))
        return {
            "synthetic_id": "v3_5_" + uuid.uuid5(uuid.NAMESPACE_URL, f"groq:{MODEL}:v3.5:{source_id}:{number}").hex,
            "source_row_id": source_id, "source_text_group_id": self.sources.loc[str(source_id), "text_group_id"],
            "source_dataset": "EMSCAD", "source_label": 0, "generated_label": 2, "provider": "groq", "model": MODEL,
            "generation_version": "v3.5", "prompt_version": "v3.5", "strategy_revision": "diagnostic-guidance-r2",
            "validation_version": VALIDATION_VERSION,
            "review_version": REVIEW_VERSION, "attempt_number": number, "fraud_mechanism": policy["fraud_mechanism"],
            "generation_timestamp": None, "generation_status": "not_attempted", "final_decision": "pending",
            "human_review_status": "pending", "parent_v3_4_synthetic_id": self.previous_v3_4[source_id]["synthetic_id"],
            "source": source, "generated": None, "request_prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(), "attempts": [], "validation": {}, "review": {},
            "failure_signature": [], "blocking_reason": None,
        }

    def generate_attempt(self, policy):
        source_id = policy["source_row_id"]
        record = self.new_attempt(policy)
        self.state["attempts"].append(record)
        for request_attempt in range(1, 4):
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
            event = {"request_attempt": request_attempt, "timestamp": record["generation_timestamp"],
                     "http_status": status_code, "error": error_message, "rate_headers": headers,
                     "response": safe_output(getattr(self.provider, "last_response", {}))}
            if transient:
                delay, basis = retry_delay(headers, request_attempt)
                event.update(retry_delay_seconds=delay, retry_delay_basis=basis)
                self.state["next_request_at"] = max(self.state["next_request_at"], time.time() + delay)
                if delay > 300:
                    self.state["stop_reason"] = "Server requested a delay above 300 seconds; stopped safely."
            record["attempts"].append(event)
            if self.state["consecutive_429"] >= 3:
                self.state["stop_reason"] = "Stopped after three consecutive HTTP 429 responses."
            self.checkpoint()
            if record["generation_status"] == "generated" or not transient or request_attempt == 3 or self.state["stop_reason"]:
                break
        if record["generation_status"] != "generated":
            record["final_decision"] = "needs_human_review"
            record["blocking_reason"] = "generation_failure"
            self.checkpoint()
            return record
        record["validation"] = validate_v3_4(record["source"], record["generated"], policy)
        record["review"] = review_candidate(record["source"], record["generated"], policy, record["validation"])
        record["failure_signature"] = list(failure_signature(record["validation"], record["review"]))
        prior = self.attempts_for(source_id)[:-1]
        repeated = any(item.get("failure_signature") == record["failure_signature"] for item in prior)
        if record["review"]["accepted"]:
            record["final_decision"] = "ACCEPT"
            self.state["final"][str(source_id)] = {"status": "ACCEPT", "attempt_number": record["attempt_number"],
                "blocking_reason": None, "synthetic_id": record["synthetic_id"]}
        elif record["review"].get("checks", {}).get("source_ambiguity"):
            record["final_decision"] = "needs_human_review"
            record["blocking_reason"] = "source_ambiguity"
            self.state["final"][str(source_id)] = {"status": "needs_human_review", "attempt_number": record["attempt_number"],
                "blocking_reason": "source_ambiguity", "synthetic_id": record["synthetic_id"]}
        elif repeated:
            record["final_decision"] = "retry"
            record["blocking_reason"] = "repeated_failure_pattern"
        else:
            record["final_decision"] = "retry"
            record["blocking_reason"] = "validation_or_review_failure"
        self.checkpoint()
        return record

    def run_source(self, policy):
        source_id = policy["source_row_id"]
        if str(source_id) in self.state["final"] or self.state["stop_reason"]:
            return self.state["final"].get(str(source_id))
        while len(self.attempts_for(source_id)) < self.max_attempts and not self.state["stop_reason"]:
            record = self.generate_attempt(policy)
            self.export()
            if record["final_decision"] in {"ACCEPT", "needs_human_review"}:
                self.state["final"][str(source_id)] = {"status": record["final_decision"],
                    "attempt_number": record["attempt_number"], "blocking_reason": record["blocking_reason"],
                    "synthetic_id": record["synthetic_id"]}
                self.checkpoint()
                return record
        attempts = self.attempts_for(source_id)
        last = attempts[-1]
        self.state["final"][str(source_id)] = {"status": "needs_human_review", "attempt_number": last["attempt_number"],
            "blocking_reason": "max_attempts_exhausted", "synthetic_id": last["synthetic_id"]}
        self.checkpoint()
        return last

    def run_all(self):
        for policy in self.assignments:
            if self.state["stop_reason"]:
                break
            result = self.run_source(policy)
            self.export()
            if result:
                print(f"Source {policy['source_row_id']}: {result['final_decision']} after attempt {result['attempt_number']}", flush=True)
        self.state["finished"] = len(self.state["final"]) == len(TARGET_IDS) and not self.state["stop_reason"]
        self.checkpoint()

    def finalize_existing_statuses(self):
        """Mark terminal only when every approved source already has a recorded final status.

        This method performs no generation or provider access.
        """
        missing = [source_id for source_id in TARGET_IDS if str(source_id) not in self.state["final"]]
        if missing:
            raise ValueError(f"Cannot finalize unresolved sources: {missing}")
        self.state["finished"] = self.state["stop_reason"] is None
        self.checkpoint()

    def reevaluate_reviews(self):
        """Versioned deterministic re-evaluation; original review outputs remain in state history."""
        for record in self.state["attempts"]:
            if record["generation_status"] != "generated":
                continue
            policy = next(item for item in self.assignments if item["source_row_id"] == record["source_row_id"])
            revised = review_candidate(record["source"], record["generated"], policy, record["validation"])
            if revised != record.get("review"):
                record.setdefault("review_revisions", []).append({"superseded_at": datetime.now(UTC).isoformat(),
                    "previous": record.get("review"), "replacement": revised})
                record["review"] = revised
                record["failure_signature"] = list(failure_signature(record["validation"], revised))
            if record["review"].get("accepted"):
                self.state["final"][str(record["source_row_id"])] = {
                    "status": "ACCEPT", "attempt_number": record["attempt_number"],
                    "blocking_reason": None, "synthetic_id": record["synthetic_id"],
                }
            elif record["review"].get("checks", {}).get("source_ambiguity"):
                self.state["final"][str(record["source_row_id"])] = {
                    "status": "needs_human_review", "attempt_number": record["attempt_number"],
                    "blocking_reason": "source_ambiguity", "synthetic_id": record["synthetic_id"],
                }
        self.checkpoint()

    def export(self):
        rows, validation_rows, review_rows, diagnosis_rows = [], [], [], []
        for record in self.state["attempts"]:
            row = {key: value for key, value in record.items() if key not in {"source", "generated", "validation", "review", "attempts", "request_prompt"}}
            row.update({field: (record["generated"] or {}).get(field, "") for field in CORE_FIELDS})
            row["generated_text"] = combine_fields(record["generated"] or {})
            row["request_count"] = len(record["attempts"])
            rows.append(row)
            validation_rows.append({"synthetic_id": record["synthetic_id"], "source_row_id": record["source_row_id"],
                "attempt_number": record["attempt_number"], **{key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value for key, value in record["validation"].items()}})
            review_rows.append({"synthetic_id": record["synthetic_id"], "source_row_id": record["source_row_id"],
                "attempt_number": record["attempt_number"], **{key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value for key, value in record["review"].items()}})
            diagnosis_rows.append({"source_row_id": record["source_row_id"], "attempt_number": record["attempt_number"],
                "synthetic_id": record["synthetic_id"], "generation_status": record["generation_status"],
                "validation_status": record["validation"].get("validation_status"), "review_decision": record["review"].get("decision"),
                "failure_signature": json.dumps(record["failure_signature"]), "blocking_reason": record["blocking_reason"],
                "validator_reasons": json.dumps(record["validation"].get("reasons", [])), "review_reasons": json.dumps(record["review"].get("reasons", []))})
        pd.DataFrame(rows).to_csv(self.output / "generated_records.csv", index=False)
        pd.DataFrame(rows).to_csv(self.output / "generation_attempts.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(self.output / "validation_results.csv", index=False)
        pd.DataFrame(review_rows).to_csv(self.output / "human_review_results.csv", index=False)
        pd.DataFrame(diagnosis_rows).to_csv(self.output / "failure_diagnosis.csv", index=False)
        final_rows = [{"source_row_id": source_id, **result} for source_id, result in self.state["final"].items()]
        pd.DataFrame(final_rows).to_csv(self.output / "final_summary.csv", index=False)
        mismatch = self._verify_protected()
        integrity = {"protected_file_count": len(self.protected), "hash_mismatches": mismatch,
                     "verified_at": datetime.now(UTC).isoformat(), "status": "PASS" if not mismatch else "FAIL"}
        atomic_json(self.output / "integrity_report.json", integrity)
        generated = [record for record in self.state["attempts"] if record["generation_status"] == "generated"]
        summary = {"records_requested": len(TARGET_IDS), "attempts_total": len(self.state["attempts"]),
            "records_with_generation": len({record["source_row_id"] for record in generated}),
            "generation_failures": sum(record["generation_status"] == "failed" for record in self.state["attempts"]),
            "http_429_failures": sum(event["http_status"] == 429 for record in self.state["attempts"] for event in record["attempts"]),
            "final_status": self.state["final"], "validation_status_counts": dict(Counter(record["validation"].get("validation_status") for record in generated)),
            "review_decision_counts": dict(Counter(record["review"].get("decision") for record in generated)),
            "mechanism_distribution": dict(Counter(record["fraud_mechanism"] for record in generated)),
            "stop_reason": self.state["stop_reason"], "finished": self.state["finished"],
            "provider": "groq", "model": MODEL, "prompt_version": "v3.5", "validation_version": VALIDATION_VERSION,
            "review_version": REVIEW_VERSION, "updated_at": datetime.now(UTC).isoformat()}
        atomic_json(self.output / "generation_summary.json", summary)
        return summary


if __name__ == "__main__":
    run = RegenerationV3_5(Path(__file__).resolve().parents[1])
    if run.configure():
        run.run_all()
    print(json.dumps(run.export(), indent=2))
