"""Isolated v3.3 follow-up for the twelve approved GPT regeneration sources."""

import hashlib
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.generation import JOB_SCHEMA, parse_structured_response
from src.groq_pilot import atomic_json, file_hash, safe_output
from src.regeneration_v3 import MODEL, TARGET_IDS, RegenerationV3, paced_wait, retry_delay
from src.regeneration_v3_2 import RegenerationV3_2
from src.validation_v3_3 import VERSION, validate_v3_3


class RegenerationV3_3(RegenerationV3_2):
    """Reuse v3.2 rate safeguards with a versioned prompt and validator."""

    def __init__(self, root, provider=None, wait=paced_wait):
        self.root = Path(root).resolve()
        load_dotenv(self.root / ".env", override=False)
        self.output = self.root / "data/synthetic/pilot/groq_gpt_v3_3"
        self.output.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.wait = wait
        self.protocol_path = self.root / "prompts/gpt_regeneration_v3_protocol.json"
        self.followup_path = self.root / "prompts/gpt_regeneration_v3_3_followup.json"
        self.prompt_path = self.root / "prompts/ai_fraud_generation_v3_3.txt"
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.followup = json.loads(self.followup_path.read_text(encoding="utf-8"))
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.assignments = self.protocol["assignments"]
        if [policy["source_row_id"] for policy in self.assignments] != TARGET_IDS:
            raise ValueError("Protocol must contain the approved twelve source IDs in frozen order.")
        if self.followup["source_ids"] != TARGET_IDS or self.followup["excluded_source_id"] != 10308:
            raise ValueError("Follow-up source declaration does not match the approved scope.")

        self.v2 = self.root / "data/synthetic/pilot/groq_gpt/openai_gpt-oss-120b"
        parents = [json.loads(line) for line in
                   (self.v2 / "ai_fraud_pilot_raw.jsonl").read_text(encoding="utf-8").splitlines()]
        self.parents = {record["source_row_id"]: record for record in parents}
        clean = pd.read_csv(self.root / "data/processed/emscad_clean.csv", dtype=str, keep_default_na=False)
        self.sources = clean.set_index("source_row_id").loc[list(map(str, TARGET_IDS))]
        for source_id in TARGET_IDS:
            if self.sources.loc[str(source_id), "fraudulent"] != "f":
                raise ValueError("All follow-up sources must remain legitimate EMSCAD rows.")
            if self.parents[source_id]["generation_status"] != "generated":
                raise ValueError("Each follow-up source must have a successful frozen v2 parent candidate.")
            if self.sources.loc[str(source_id), "text_group_id"] != self.parents[source_id]["source_text_group_id"]:
                raise ValueError("Source text-group provenance mismatch.")
        if not self.sources["text_group_id"].is_unique:
            raise ValueError("Duplicate source groups are not allowed in the follow-up selection.")

        protected_paths = [self.root / "data/raw/emscad.csv", self.root / "data/processed/emscad_clean.csv"]
        for directory in [
            self.v2,
            self.root / "data/synthetic/pilot/groq_gpt_v3",
            self.root / "data/synthetic/pilot/groq_gpt_v3_1",
            self.root / "data/synthetic/pilot/groq_gpt_v3_2",
        ]:
            protected_paths.extend(path for path in directory.rglob("*") if path.is_file())
        self.protected = {str(path.relative_to(self.root)): file_hash(path) for path in protected_paths}
        self.manifest = {
            "provider": "groq", "model": MODEL, "prompt_version": "v3.3", "validation_version": VERSION,
            "validation_lineage": "v3.2-groq-pilot", "source_ids": TARGET_IDS,
            "smoke_source_id": 11662, "excluded_source_id": 10308, "assignments": self.assignments,
            "temperature": 0.5, "seed": 42, "max_completion_tokens": 4096,
            "request_spacing_seconds": 65, "max_attempts_per_source": 3, "consecutive_rate_limit_stop": 3,
            "prompt_hash": file_hash(self.prompt_path), "protocol_hash": file_hash(self.protocol_path),
            "followup_protocol_hash": file_hash(self.followup_path),
            "code_hashes": {name: file_hash(self.root / "src" / name) for name in [
                "generation.py", "validation.py", "groq_pilot.py", "regeneration_v3.py", "validation_v3.py",
                "validation_v3_1.py", "regeneration_v3_1.py", "validation_v3_2.py", "regeneration_v3_2.py",
                "validation_v3_3.py", "regeneration_v3_3.py"]},
            "protected_hashes": self.protected,
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != self.manifest:
            raise ValueError("Frozen v3.3 run changed; refusing to mix versions or overwrite outputs.")
        if not manifest_path.exists():
            atomic_json(manifest_path, self.manifest)
        self.state = {"records": [], "smoke_status": "not_run", "stop_reason": None,
                      "finished": False, "next_request_at": 0.0, "events": [], "consecutive_429": 0}
        state_path = self.output / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(record["generation_status"] == "request_in_flight" for record in self.state["records"]):
            self.state["stop_reason"] = "Interrupted request has unknown outcome; no automatic reissue."
        self.checkpoint()

    def new_record(self, policy):
        source_id = policy["source_row_id"]
        source = self.source(source_id)
        instruction = {**policy, "mechanism_description": self.protocol["mechanisms"][policy["fraud_mechanism"]]}
        prompt = self.prompt.replace("{{PROTOCOL_JSON}}", json.dumps(instruction, ensure_ascii=True)).replace(
            "{{SOURCE_ADVERTISEMENT_JSON}}", json.dumps(source, ensure_ascii=True))
        return {
            "synthetic_id": "v3_3_" + uuid.uuid5(uuid.NAMESPACE_URL, f"groq:{MODEL}:v3.3:{source_id}").hex,
            "source_row_id": source_id, "source_text_group_id": self.sources.loc[str(source_id), "text_group_id"],
            "source_dataset": "EMSCAD", "source_label": 0, "generated_label": 2,
            "parent_synthetic_id": self.parents[source_id]["synthetic_id"], "provider": "groq", "model": MODEL,
            "generation_mode": "api", "generation_timestamp": None, "prompt_version": "v3.3",
            "validation_version": VERSION, "fraud_mechanism": policy["fraud_mechanism"],
            "validation_status": "not_run", "human_review_status": "pending_human_review",
            "source": source, "generated": None, "generation_status": "not_attempted", "request_prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(), "validation": {}, "attempts": [],
        }

    def generate_one(self, policy):
        source_id = policy["source_row_id"]
        previous = next((record for record in self.state["records"] if record["source_row_id"] == source_id), None)
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
            error_message, status_code, transient, headers = None, None, False, {}
            self.state["next_request_at"] = time.time() + 65
            try:
                record["generated"] = parse_structured_response(self.provider.generate_json(record["request_prompt"], JOB_SCHEMA))
                record["generation_status"] = "generated"
                self.state["consecutive_429"] = 0
            except Exception as error:
                record["generation_status"] = "failed"
                error_message = safe_output(str(error))
                status_code = getattr(error, "status_code", None)
                headers = getattr(error, "rate_headers", {})
                transient = getattr(error, "transient", False)
                self.state["consecutive_429"] = self.state["consecutive_429"] + 1 if status_code == 429 else 0
            event = {
                "attempt": attempt, "timestamp": record["generation_timestamp"], "http_status": status_code,
                "error": error_message, "rate_headers": headers,
                "response": safe_output(getattr(self.provider, "last_response", {})),
            }
            if transient:
                delay, basis = retry_delay(headers, attempt)
                event.update(retry_delay_seconds=delay, retry_delay_basis=basis)
                self.state["next_request_at"] = max(self.state["next_request_at"], time.time() + delay)
                if delay > 300:
                    self.state["stop_reason"] = "Server requests a wait above 300s; stopped without early retry."
            record["attempts"].append(event)
            if self.state["consecutive_429"] >= 3:
                self.state["stop_reason"] = "Stopped after three consecutive HTTP 429 responses."
            self.checkpoint()
            if record["generation_status"] == "generated" or not transient or attempt == 3 or self.state["stop_reason"]:
                break
        record["validation"] = validate_v3_3(record["source"], record["generated"], policy)
        record["validation_status"] = record["validation"]["validation_status"]
        record["human_review_status"] = "pending_human_review"
        self.checkpoint()
        return record

    def export(self):
        summary = RegenerationV3.export(self)
        review_path = self.output / "review_template.csv"
        review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        review_fields = [
            "review_context_preservation", "review_fraud_intent", "review_realism",
            "review_unsupported_facts", "review_research_meta_artifacts", "review_ai_artifacts",
            "review_overall_suitability",
        ]
        for field in review_fields:
            if field not in review:
                review[field] = ""
        review["review_guidance"] = "Yes/Partial/No; overall suitability: Accept/Revise/Reject"
        review.to_csv(review_path, index=False)
        records = self.state["records"]
        summary.update({
            "prompt_version": "v3.3", "validation_version": VERSION,
            "validation_lineage": "v3.2-groq-pilot",
            "structural_validation_failures": sum(
                not record["validation"].get("structural_valid", False) for record in records),
            "records_requiring_human_review": sum(record["generation_status"] == "generated" for record in records),
            "rate_limit_failures": sum(
                record["generation_status"] == "failed" and any(a["http_status"] == 429 for a in record["attempts"])
                for record in records),
            "contextual_integration_failures": sum(
                record["generation_status"] == "generated" and not record["validation"].get("fraud_integrated", False)
                for record in records),
            "research_meta_artifact_failures": sum(
                record["generation_status"] == "generated" and not record["validation"].get("research_artifacts_clear", True)
                for record in records),
            "ai_artifact_failures": sum(
                record["generation_status"] == "generated" and not record["validation"].get("ai_artifacts_clear", True)
                for record in records),
            "unsupported_fact_failures": sum(
                record["generation_status"] == "generated" and not record["validation"].get("unsupported_facts_clear", True)
                for record in records),
            "employment_arrangement_failures": sum(
                record["generation_status"] == "generated" and any(
                    "employment arrangement" in reason for reason in record["validation"].get("reasons", [])
                ) for record in records),
            "timestamp": datetime.now(UTC).isoformat(), "human_review_status": "pending_human_review",
        })
        atomic_json(self.output / "generation_summary.json", summary)
        for path, digest in self.protected.items():
            if file_hash(self.root / path) != digest:
                raise RuntimeError(f"Frozen artifact changed: {path}")
        return summary


if __name__ == "__main__":
    run = RegenerationV3_3(Path(__file__).resolve().parents[1])
    if run.configure() and run.smoke_test():
        run.remaining()
    print(json.dumps(run.export(), indent=2))
