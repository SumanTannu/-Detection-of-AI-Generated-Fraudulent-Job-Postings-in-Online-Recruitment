"""Resumable, sequential 20-source pilot. Existing mock artifacts are read only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.generation import (
    CORE_FIELDS, JOB_SCHEMA, GenerationError, GroqProvider,
    build_generation_prompt, parse_structured_response, select_pilot_sources, source_fields,
)
from src.validation import (
    VALIDATION_VERSION, combine_fields, flatten_record, validate_pilot_candidate,
)

PILOT_SIZE = 20
SEED = 42
PROMPT_VERSION = "v2"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_output(value):
    """Remove credential-shaped values even if a provider echoes one in an error."""
    if isinstance(value, dict):
        return {key: safe_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_output(item) for item in value]
    if isinstance(value, str):
        key = os.getenv("GROQ_API_KEY", "")
        if key:
            value = value.replace(key, "[REDACTED_KEY]")
        return re.sub(r"(?:gsk_|sk-)[A-Za-z0-9_-]+", "[REDACTED_KEY]", value)
    return value


class GroqPilot:
    """Notebook stages share durable state; reruns never regenerate attempted sources."""

    def __init__(self, root: Path, provider=None, pause_seconds: float = 3.0):
        self.root = root.resolve()
        load_dotenv(self.root / ".env", override=False)
        self.model = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b").strip()
        if os.getenv("GENERATION_PROVIDER", "groq").strip().lower() != "groq":
            raise ValueError("This notebook run requires GENERATION_PROVIDER=groq.")
        self.provider = provider
        self.pause_seconds = pause_seconds
        self.problem = None
        self.available_models = []
        self.records = []
        self.events = []
        self.smoke_status = "not_run"
        self.finished = False
        self.prompt_path = self.root / "prompts/ai_fraud_generation_v2.txt"
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.raw_path = self.root / "data/raw/emscad.csv"
        self.clean_path = self.root / "data/processed/emscad_clean.csv"
        self.pilot_parent = self.root / "data/synthetic/pilot"
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", self.model)
        if not slug:
            raise ValueError("GENERATION_MODEL must be a non-empty model ID.")
        family = "groq_gpt" if "gpt" in self.model.lower() else "groq_other"
        self.output = self.pilot_parent / family / slug
        self.output.mkdir(parents=True, exist_ok=True)
        protected = [self.raw_path, self.clean_path, *self.pilot_parent.glob("*.*")]
        protected += list((self.root / "results/tables").glob("ai_fraud_pilot_*.csv"))
        self.protected = {str(p.relative_to(self.root)): file_hash(p) for p in protected if p.is_file()}

        clean = pd.read_csv(self.clean_path, dtype=str, keep_default_na=False)
        raw = pd.read_csv(self.raw_path, dtype=str, keep_default_na=True)
        if set(clean["fraudulent"].unique()) != {"f", "t"}:
            raise ValueError("Unexpected EMSCAD targets; expected previously verified f/t encoding.")
        self.sources = select_pilot_sources(clean, raw, PILOT_SIZE, SEED, "f")
        if len(self.sources) != PILOT_SIZE or not self.sources["text_group_id"].is_unique:
            raise ValueError("Exactly 20 distinct source text groups are required.")
        if not self.sources["fraudulent"].eq("f").all():
            raise ValueError("All generation sources must have fraudulent=f.")
        mock_selection = self.root / "results/tables/ai_fraud_pilot_source_selection.csv"
        if mock_selection.exists():
            previous = pd.read_csv(mock_selection, dtype=str)
            if previous["source_row_id"].tolist() != self.sources["source_row_id"].tolist():
                raise ValueError("Selection differs from the preserved mock pilot; investigate before generation.")

        self.manifest = {
            "provider": "groq", "model": self.model, "generation_mode": "api",
            "seed": SEED, "pilot_size": PILOT_SIZE, "prompt_version": PROMPT_VERSION,
            "prompt_sha256": file_hash(self.prompt_path),
            "validation_version": VALIDATION_VERSION,
            "code_hashes": {name: file_hash(self.root / "src" / name)
                            for name in ("generation.py", "validation.py", "groq_pilot.py")},
            "protected_input_sha256": self.protected,
            "source_row_ids": self.sources["source_row_id"].tolist(),
            "max_completion_tokens": 4096, "temperature": 0.7,
            "smoke_max_attempts": 1, "remaining_max_attempts": 2,
            "maximum_generation_requests": 39,
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != self.manifest:
                raise ValueError("Existing run inputs/configuration changed. Refusing to overwrite or resume it.")
        else:
            atomic_json(manifest_path, self.manifest)
        state = self.output / "state.json"
        if state.exists():
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.records = saved["records"]
            self.events = saved["events"]
            self.smoke_status = saved["smoke_status"]
            self.finished = saved["finished"]
            if any(r["generation_status"] == "request_in_flight" for r in self.records):
                self.problem = "Interrupted request has unknown server outcome; no automatic reissue is allowed."
        self.sources[["source_row_id", "text_group_id", "title", "selection_context",
                      "selection_length_bucket"]].to_csv(self.output / "source_selection.csv", index=False)
        snapshot = self.output / "prompt.txt"
        if snapshot.exists() and snapshot.read_text(encoding="utf-8") != self.prompt:
            raise ValueError("Saved prompt differs from the current version.")
        if not snapshot.exists():
            snapshot.write_text(self.prompt, encoding="utf-8")
        self.checkpoint()

    def log(self, stage, message):
        self.events.append({"timestamp": datetime.now(UTC).isoformat(), "stage": stage,
                            "message": safe_output(message)})
        self.checkpoint()

    def checkpoint(self):
        atomic_json(self.output / "state.json", safe_output({
            "records": self.records, "events": self.events,
            "smoke_status": self.smoke_status, "finished": self.finished,
        }))

    def check_models(self):
        if self.problem or self.finished:
            return not self.problem
        # A failed first generation is a durable stopping condition, including reruns.
        if self.smoke_status == "failed":
            self.problem = "The first API generation failed; this run is stopped. See errors.json."
            return False
        try:
            if self.provider is None:
                self.provider = GroqProvider(self.model)
            if self.provider.model != self.model:
                raise GenerationError("Provider model does not match the frozen run configuration.")
            self.available_models = self.provider.available_models()
            atomic_json(self.output / "model_availability.json", {
                "checked_at": datetime.now(UTC).isoformat(),
                "available_models": self.available_models, "selected_model": self.model,
            })
            if self.model not in self.available_models:
                raise GenerationError(f"Configured model {self.model} is unavailable on this Groq account.")
            return True
        except GenerationError as error:
            self.problem = safe_output(str(error))
            self.log("preflight_failed", self.problem)
            return False

    def new_record(self, row):
        source = source_fields(row)
        source = {field: re.sub(r"https?://\S+|www\.\S+|#URL_[^#]*#", "[REDACTED_URL]", value)
                  for field, value in source.items()}
        source = safe_output(source)
        prompt = build_generation_prompt(self.prompt, source)
        signature = f"groq:{self.model}:{PROMPT_VERSION}:{row['source_row_id']}:{row['text_group_id']}"
        return {
            "synthetic_id": "pilot_" + uuid.uuid5(uuid.NAMESPACE_URL, signature).hex,
            "source_row_id": int(row["source_row_id"]),
            "source_text_group_id": row["text_group_id"],
            "source_dataset": "EMSCAD", "source_label": 0, "generated_label": 2,
            "provider": "groq", "model": self.model, "generation_mode": "api",
            "generation_timestamp": None, "prompt_version": PROMPT_VERSION,
            "prompt_sha256": self.manifest["prompt_sha256"],
            "validation_version": VALIDATION_VERSION, "validation_status": "pending",
            "source": source, "source_redaction": "contacts, URLs and credentials redacted",
            "request_prompt": prompt,
            "request_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "generated": None, "generation_status": "pending", "generation_attempts": 0,
            "generation_error": None, "attempts": [], "validation": {},
        }

    def generate_one(self, row, smoke=False):
        existing = next((r for r in self.records if r["source_row_id"] == int(row["source_row_id"])), None)
        if existing:
            return existing
        record = self.new_record(row)
        self.records.append(record)
        for attempt in range(1, (1 if smoke else 2) + 1):
            record["generation_attempts"] = attempt
            record["generation_status"] = "request_in_flight"
            record["generation_timestamp"] = datetime.now(UTC).isoformat()
            self.checkpoint()  # Write before sending; crashes must not cause duplicate API requests.
            transient = False
            try:
                generated = parse_structured_response(self.provider.generate_json(record["request_prompt"], JOB_SCHEMA))
                if not combine_fields(generated):
                    raise GenerationError("Structured response contains an entirely empty advertisement.")
                record["generated"] = safe_output(generated)
                record["generation_status"] = "generated"
                record["generation_error"] = None
            except GenerationError as error:
                record["generation_status"] = "failed"
                record["generation_error"] = safe_output(str(error))
                transient = getattr(error, "transient", False)
            except Exception as error:
                record["generation_status"] = "failed"
                record["generation_error"] = safe_output(f"{type(error).__name__}: {error}")
            record["attempts"].append({
                "attempt": attempt, "timestamp": record["generation_timestamp"],
                "status": record["generation_status"], "error": record["generation_error"],
                "provider_response": safe_output(getattr(self.provider, "last_response", {})),
            })
            validate_pilot_candidate(record)
            self.checkpoint()
            if record["generation_status"] == "generated" or smoke or not transient or attempt == 2:
                break
            time.sleep(max(self.pause_seconds, 2.0))
        return record

    def smoke_test(self):
        if self.problem or self.finished or self.smoke_status == "succeeded":
            return self.smoke_status == "succeeded"
        if self.provider is None or not self.available_models:
            raise RuntimeError("Check model availability before the smoke test.")
        first = self.generate_one(self.sources.iloc[0], smoke=True)
        self.smoke_status = "succeeded" if first["generation_status"] == "generated" else "failed"
        if self.smoke_status == "failed":
            self.problem = first["generation_error"]
        self.log("smoke_test", self.smoke_status)
        self.export()
        return self.smoke_status == "succeeded"

    def generate_remaining(self):
        if self.problem or self.smoke_status != "succeeded" or self.finished:
            return
        for _, row in self.sources.iloc[1:].iterrows():
            if any(r["source_row_id"] == int(row["source_row_id"]) for r in self.records):
                continue
            time.sleep(self.pause_seconds)
            self.generate_one(row)
            self.export()
            print(f"Completed source {len(self.records)}/{PILOT_SIZE}.", flush=True)
        self.finished = True
        self.checkpoint()

    def export(self):
        """Materialize tables from state while preserving completed reviewer fields."""
        successful = [r for r in self.records if r["generation_status"] == "generated"]
        rows = []
        review_rows = []
        for _, source in self.sources.iterrows():
            record = next((r for r in self.records if r["source_row_id"] == int(source["source_row_id"])), None)
            if record is None:
                record = self.new_record(source)
            generated = record["generated"] or {}
            validation = record["validation"]
            review_rows.append({
                "synthetic_id": record["synthetic_id"], "source_row_id": record["source_row_id"],
                "source_text_group_id": record["source_text_group_id"],
                "source_title": record["source"]["title"], "source_text": combine_fields(record["source"]),
                "generated_title": generated.get("title", ""), "generated_text": combine_fields(generated),
                "automatic_result": validation.get("automatic_result", "not_run"),
                "validation_status": record["validation_status"],
                "generation_status": record["generation_status"],
                "reviewer_decision": "", "fraud_intent_decision": "",
                "context_preservation_decision": "", "reviewer_notes": "",
            })
            if record["generation_status"] != "pending":
                flat = flatten_record(record)
                flat.update({"automatic_result": validation.get("automatic_result"),
                             "validation_version": VALIDATION_VERSION,
                             "generated_text": combine_fields(generated)})
                rows.append(flat)
        columns = list(flatten_record({}).keys()) + ["automatic_result", "validation_version", "generated_text"]
        table = pd.DataFrame(rows, columns=columns)
        table.to_csv(self.output / "validation_results.csv", index=False)
        table.loc[table["automatic_passed"].eq(True)].to_csv(self.output / "ai_fraud_pilot_validated.csv", index=False)
        review = pd.DataFrame(review_rows)
        review_path = self.output / "human_review_template.csv"
        if review_path.exists():
            old = pd.read_csv(review_path, dtype=str, keep_default_na=False).set_index("synthetic_id")
            for field in ["reviewer_decision", "fraud_intent_decision", "context_preservation_decision", "reviewer_notes"]:
                review[field] = review["synthetic_id"].map(old[field]).fillna("")
        review.to_csv(review_path, index=False)
        raw_path = self.output / "ai_fraud_pilot_raw.jsonl"
        temporary = raw_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(safe_output(r), ensure_ascii=True) + "\n" for r in self.records), encoding="utf-8")
        temporary.replace(raw_path)
        source_lengths = [len(combine_fields(source_fields(r))) for _, r in self.sources.iterrows()]
        generated_lengths = [len(combine_fields(r["generated"])) for r in successful]
        summary = {
            "number_selected": len(self.sources), "number_attempted": len(self.records),
            "number_successfully_generated": len(successful),
            "number_generation_failures": sum(r["generation_status"] == "failed" for r in self.records),
            "number_not_attempted": PILOT_SIZE - len(self.records),
            "number_structurally_valid": sum(r["validation"].get("structural_valid", False) for r in self.records),
            "number_automatically_passing": sum(r["validation"].get("automatic_passed", False) for r in self.records),
            "number_automatically_rejected": sum(r["validation"].get("automatic_result") == "auto_reject" for r in self.records),
            "number_requiring_human_review": sum(r["validation_status"] == "needs_human_review" for r in self.records),
            "number_with_completed_human_review": 0,
            "average_source_text_characters": round(sum(source_lengths) / len(source_lengths), 2),
            "average_generated_text_characters": round(sum(generated_lengths) / len(generated_lengths), 2) if generated_lengths else None,
            "generation_requests_attempted": sum(r["generation_attempts"] for r in self.records),
            "provider": "groq", "model": self.model, "generation_mode": "api",
            "prompt_version": PROMPT_VERSION, "validation_version": VALIDATION_VERSION,
            "smoke_test": self.smoke_status, "pilot_complete": self.finished,
            "error": self.problem,
        }
        pd.DataFrame(summary.items(), columns=["metric", "value"]).to_csv(self.output / "summary.csv", index=False)
        errors = [{"source_row_id": r["source_row_id"], "attempt": a["attempt"], "error": a["error"]}
                  for r in self.records for a in r["attempts"] if a["error"]]
        atomic_json(self.output / "errors.json", {"run_error": self.problem, "generation_errors": errors, "events": self.events})
        for path, expected in self.protected.items():
            if file_hash(self.root / path) != expected:
                raise RuntimeError(f"Protected input changed during run: {path}")
        return summary


if __name__ == "__main__":
    pilot = GroqPilot(Path(__file__).resolve().parents[1])
    if pilot.check_models() and pilot.smoke_test():
        pilot.generate_remaining()
    print(json.dumps(pilot.export(), indent=2))
