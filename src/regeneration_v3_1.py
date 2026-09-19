"""Isolated v3.1 follow-up for the twelve approved GPT regeneration sources."""

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.groq_pilot import atomic_json, file_hash, safe_output
from src.regeneration_v3 import MODEL, TARGET_IDS, GroqV3Provider, RegenerationV3, source_fields
from src.validation_v3_1 import VERSION, validate_v3_1


class RegenerationV3_1(RegenerationV3):
    """Use the frozen v3 mechanics with a new directory, prompt and validator."""

    def __init__(self, root, provider=None, wait=None):
        self.root = Path(root).resolve()
        from dotenv import load_dotenv
        import pandas as pd
        load_dotenv(self.root / ".env", override=False)
        self.output = self.root / "data/synthetic/pilot/groq_gpt_v3_1"
        self.output.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.wait = wait or __import__("src.regeneration_v3", fromlist=["paced_wait"]).paced_wait
        self.protocol_path = self.root / "prompts/gpt_regeneration_v3_protocol.json"
        self.followup_path = self.root / "prompts/gpt_regeneration_v3_1_followup.json"
        self.prompt_path = self.root / "prompts/ai_fraud_generation_v3_1.txt"
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.followup = json.loads(self.followup_path.read_text(encoding="utf-8"))
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.assignments = self.protocol["assignments"]
        if [p["source_row_id"] for p in self.assignments] != TARGET_IDS:
            raise ValueError("Protocol must contain the approved twelve source IDs in frozen order.")
        if self.followup["source_ids"] != TARGET_IDS or self.followup["excluded_source_id"] != 10308:
            raise ValueError("Follow-up source declaration does not match the approved scope.")
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
        protected_paths += [p for p in (self.root / "data/synthetic/pilot/groq_gpt_v3").rglob("*") if p.is_file()]
        self.protected = {str(p.relative_to(self.root)): file_hash(p) for p in protected_paths}
        self.manifest = {
            "provider": "groq", "model": MODEL, "prompt_version": "v3.1", "validation_version": VERSION,
            "validation_lineage": "v3-groq-pilot", "source_ids": TARGET_IDS,
            "smoke_source_id": 11662, "excluded_source_id": 10308, "assignments": self.assignments,
            "temperature": 0.5, "seed": 42, "max_completion_tokens": 4096,
            "request_spacing_seconds": 65, "max_attempts_per_source": 3, "consecutive_rate_limit_stop": 3,
            "prompt_hash": file_hash(self.prompt_path), "protocol_hash": file_hash(self.protocol_path),
            "followup_protocol_hash": file_hash(self.followup_path),
            "code_hashes": {name: file_hash(self.root / "src" / name) for name in [
                "generation.py", "validation.py", "groq_pilot.py", "regeneration_v3.py", "validation_v3.py",
                "validation_v3_1.py", "regeneration_v3_1.py"]},
            "protected_hashes": self.protected,
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != self.manifest:
            raise ValueError("Frozen v3.1 run changed; refusing to mix versions or overwrite outputs.")
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

    def source(self, sid):
        source = safe_output(source_fields(self.sources.loc[str(sid)]))
        return {field: re.sub(r"https?://\S+|www\.\S+|#URL_[^#]+#", "[REDACTED_URL]", value)
                for field, value in source.items()}

    def new_record(self, policy):
        source = self.source(policy["source_row_id"])
        instruction = {**policy, "mechanism_description": self.protocol["mechanisms"][policy["fraud_mechanism"]]}
        prompt = self.prompt.replace("{{PROTOCOL_JSON}}", json.dumps(instruction, ensure_ascii=True)).replace(
            "{{SOURCE_ADVERTISEMENT_JSON}}", json.dumps(source, ensure_ascii=True))
        sid = policy["source_row_id"]
        return {
            "synthetic_id": "v3_1_" + uuid.uuid5(uuid.NAMESPACE_URL, f"groq:{MODEL}:v3.1:{sid}").hex,
            "source_row_id": sid, "source_text_group_id": self.sources.loc[str(sid), "text_group_id"],
            "source_dataset": "EMSCAD", "source_label": 0, "generated_label": 2,
            "parent_synthetic_id": self.parents[sid]["synthetic_id"], "provider": "groq", "model": MODEL,
            "generation_mode": "api", "generation_timestamp": None, "prompt_version": "v3.1",
            "validation_version": VERSION, "fraud_mechanism": policy["fraud_mechanism"],
            "validation_status": "not_run", "human_review_status": "pending_human_review",
            "source": source, "generated": None, "generation_status": "not_attempted", "request_prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(), "validation": {}, "attempts": [],
        }

    def generate_one(self, policy):
        record = super().generate_one(policy)
        if record is not None:
            record["validation"] = validate_v3_1(record["source"], record["generated"], policy)
            record["validation_status"] = record["validation"]["validation_status"]
            record["human_review_status"] = "pending_human_review"
            self.checkpoint()
        return record

    def export(self):
        summary = super().export()
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
        summary.update({
            "prompt_version": "v3.1", "validation_version": VERSION,
            "validation_lineage": "v3-groq-pilot",
            "structural_validation_failures": sum(
                not record["validation"].get("structural_valid", False) for record in self.state["records"]),
            "records_requiring_human_review": sum(
                record["generation_status"] == "generated" for record in self.state["records"]),
            "rate_limit_failures": sum(
                record["generation_status"] == "failed" and any(a["http_status"] == 429 for a in record["attempts"])
                for record in self.state["records"]),
            "timestamp": datetime.now(UTC).isoformat(),
            "human_review_status": "pending_human_review",
        })
        atomic_json(self.output / "generation_summary.json", summary)
        return summary


if __name__ == "__main__":
    run = RegenerationV3_1(Path(__file__).resolve().parents[1])
    if run.configure() and run.smoke_test():
        run.remaining()
    print(json.dumps(run.export(), indent=2))
