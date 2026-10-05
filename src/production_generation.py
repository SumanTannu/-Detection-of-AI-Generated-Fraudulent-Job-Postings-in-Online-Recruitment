"""Resumable full-scale v3.5 production generation.

The runner stops only at 5,000 approved records, never reuses a source, and never
claims that its deterministic review gate is a human review. Historical pilots are
read-only and keep their original model/version metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.generation import (CORE_FIELDS, JOB_SCHEMA, GenerationError, OpenRouterProvider,
                            parse_structured_response, source_fields)
from src.groq_pilot import atomic_json, file_hash, safe_output
from src.regeneration_v3 import GroqV3Provider, retry_delay
from src.review_v3_5 import VERSION as REVIEW_VERSION, failure_signature, review_candidate
from src.validation import combine_fields
from src.validation_v3 import SIGNALS
from src.validation_v3_3 import _hours, employment_arrangements
from src.validation_v3_4 import VERSION as VALIDATION_VERSION, _experience_constraints, _money_values, validate_v3_4

TARGET_APPROVED = 5_000
BATCH_SIZE = 50
SEED = 42
MODEL = "openai/gpt-oss-20b"
PROMPT_VERSION = "v3.5-production-20b-r9"
SOURCE_POLICY_VERSION = "risk-stratified-v2"
MECHANISMS = (
    "sensitive_information", "unusual_financial_arrangements", "suspicious_application",
    "urgency_pressure", "payment_related", "implausible_recruitment",
)
MECHANISM_DESCRIPTIONS = {
    "sensitive_information": "Premature collection of identity-document and financial-information categories before interview or screening; no actual values.",
    "unusual_financial_arrangements": "Personal-account use for company funds before normal employment checks; no transfer instructions, destination, or fee.",
    "suspicious_application": "Use of an unspecified private channel while bypassing official recruitment verification; no contact destination or fee.",
    "urgency_pressure": "A short deadline requiring commitment before written terms or normal screening, tied to loss of consideration.",
    "payment_related": "An unjustified applicant processing charge required to progress before interview; no amount or payment destination.",
    "implausible_recruitment": "Guaranteed selection before role-relevant interview or competency verification while retaining stated requirements.",
}
PILOT_SOURCES = (5581, 14350, 3203, 7192)
MANUAL_QC_RATE = 0.05
FAILURE_KEYS = ("contextual_integration", "unsupported_fact", "employment", "compensation",
                "research_artifact", "ai_artifact", "safety", "duplicate_similarity")
SOURCE_META_RISK = re.compile(r"\b(?:pilot|academic|research|experiment|study|evaluation|benchmark|synthetic|llm|ai[ -]generated)\b", re.I)


def _norm(value) -> str:
    value = "" if pd.isna(value) else str(value).strip()
    return value.casefold() or "[missing]"


def _length_bucket(text: str) -> str:
    words = len(re.findall(r"\b\w+\b", text or ""))
    return "xs" if words < 80 else "s" if words < 160 else "m" if words < 320 else "l" if words < 640 else "xl"


def diversity_order(clean: pd.DataFrame, raw: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Return every eligible legitimate text group in deterministic diversity-first order."""
    required = {"source_row_id", "text_group_id", "fraudulent", "text", *CORE_FIELDS}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Clean EMSCAD data lacks columns: {sorted(missing)}")
    candidates = clean[clean["fraudulent"].eq("f")].copy()
    candidates = candidates.sort_values("source_row_id", key=lambda s: pd.to_numeric(s, errors="coerce"))
    candidates = candidates.drop_duplicates("text_group_id", keep="first")
    raw_by_id = raw.copy()
    if "source_row_id" not in raw_by_id:
        raw_by_id["source_row_id"] = raw_by_id.index.astype(str)
    raw_by_id["source_row_id"] = raw_by_id["source_row_id"].astype(str)
    raw_by_id = raw_by_id.drop_duplicates("source_row_id").set_index("source_row_id")
    metadata = ("industry", "function", "employment_type", "location", "required_experience",
                "required_education", "telecommuting")
    for column in metadata:
        mapping = raw_by_id[column] if column in raw_by_id else pd.Series(dtype=str)
        candidates[column] = candidates["source_row_id"].astype(str).map(mapping).map(_norm)
    candidates["title_group"] = candidates["title"].map(lambda x: " ".join(re.findall(r"[a-z]+", _norm(x)))[:60])
    candidates["description_length_bucket"] = candidates["text"].map(_length_bucket)
    strata = ["industry", "function", "employment_type", "required_experience", "required_education",
              "telecommuting", "description_length_bucket"]
    candidates["selection_stratum"] = candidates[strata].agg("|".join, axis=1)
    candidates["tie"] = candidates.apply(lambda r: hashlib.sha256(
        f"{seed}:{r['source_row_id']}:{r['text_group_id']}".encode()).hexdigest(), axis=1)
    queues = {key: deque(group.sort_values("tie").index.tolist())
              for key, group in candidates.groupby("selection_stratum", sort=True)}
    ordered = []
    while queues:
        for key in sorted(queues, key=lambda k: (len(queues[k]), k)):
            if key in queues and queues[key]:
                ordered.append(queues[key].popleft())
                if not queues[key]:
                    del queues[key]
    result = candidates.loc[ordered].copy()
    result["selection_rank"] = range(1, len(result) + 1)
    result["initial_target"] = result["selection_rank"].le(TARGET_APPROVED)
    return result


def mechanism_compatibility(row, mechanism: str) -> int:
    text = combine_fields(source_fields(row)).casefold()
    recruitment = len(re.findall(r"\b(?:apply|application|candidate|interview|screen|onboard|hire|recruit)\w*", text))
    scores = {
        "sensitive_information": recruitment + bool(re.search(r"\b(?:verify|document|identity|payroll|bank)\w*", text)),
        "unusual_financial_arrangements": recruitment + bool(re.search(r"\b(?:finance|account|fund|expense|payment)\w*", text)),
        "suspicious_application": recruitment + bool(re.search(r"\b(?:online|email|portal|application|apply)\w*", text)),
        "urgency_pressure": recruitment + bool(re.search(r"\b(?:immediate|start|deadline|urgent|asap)\w*", text)),
        "payment_related": recruitment + bool(re.search(r"\b(?:process|fee|payment|application|screen)\w*", text)),
        "implausible_recruitment": recruitment + bool(re.search(r"\b(?:experience|skill|degree|qualification|test)\w*", text)),
    }
    return int(scores[mechanism])


def source_signal_sentence_count(row, mechanism: str) -> int:
    """Count source sentences that already resemble a mechanism signal.

    Avoiding signal-heavy parents prevents a newly assigned mechanism from
    appearing artificially repeated after faithful source preservation.
    """
    patterns = list(SIGNALS[mechanism])
    if mechanism == "implausible_recruitment":
        patterns[0] = r"guarantee(?:d)?|automatic.{0,20}(?:offer|selection)|(?:offer|selected).{0,20}without"
    sentences = re.split(r"(?<=[.!?;])\s+|\n+", combine_fields(source_fields(row)).casefold())
    return sum(any(re.search(pattern, sentence) for pattern in patterns)
               for sentence in sentences if sentence.strip())


def assign_mechanisms(ordered: pd.DataFrame, accepted_counts: Counter | None = None) -> dict[int, str]:
    """Balance mechanisms while preferring the most context-compatible underfilled choice."""
    counts = Counter(accepted_counts or {})
    assigned = {}
    for _, row in ordered.iterrows():
        minimum = min(counts[m] for m in MECHANISMS)
        choices = [m for m in MECHANISMS if counts[m] <= minimum + 1]
        mechanism = max(choices, key=lambda m: (mechanism_compatibility(row, m), -MECHANISMS.index(m)))
        assigned[int(row["source_row_id"])] = mechanism
        counts[mechanism] += 1
    return assigned


def failure_flags(record: dict) -> dict[str, bool]:
    validation, review = record.get("validation", {}), record.get("human_review_simulation", {})
    reasons = " ".join(validation.get("reasons", []) + review.get("reasons", [])).casefold()
    return {
        "contextual_integration": bool(re.search(r"integrat|workflow|warning-style|appended", reasons)),
        "unsupported_fact": bool(re.search(r"unsupported|identifier", reasons)),
        "employment": "employment arrangement" in reasons or "working-hours" in reasons,
        "compensation": bool(re.search(r"compensation|monetary", reasons)),
        "research_artifact": "research" in reasons or "meta-language" in reasons,
        "ai_artifact": "ai/conversational" in reasons or "conversational framing" in reasons,
        "safety": "unsafe" in reasons,
        "duplicate_similarity": bool(re.search(r"similar|copy|grounded", reasons)),
    }


class ProductionRun:
    def __init__(self, root: Path, provider=None, wait=time.sleep, batch_size: int = BATCH_SIZE,
                 provider_name: str = "groq", model: str = MODEL,
                 prompt_version: str = PROMPT_VERSION, output_name: str = "groq_gpt_v3_5_20b",
                 include_historical_pilots: bool = True, final_filename: str = "ai_generated_fraud.csv"):
        self.root = Path(root).resolve(); load_dotenv(self.root / ".env", override=False)
        self.provider_name = provider_name
        self.model = model
        self.prompt_version = prompt_version
        self.include_historical_pilots = include_historical_pilots
        self.output = self.root / "data/synthetic/production" / output_name
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "state.json"
        self.final_path = self.root / "data/synthetic/final" / final_filename
        self.provider, self.wait, self.batch_size = provider, wait, batch_size
        self.request_spacing = float(os.getenv("GENERATION_REQUEST_SPACING_SECONDS", "3"))
        self.prompt_path = self.root / "prompts/ai_fraud_generation_v3_5_production.txt"
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.clean_path = self.root / "data/processed/emscad_clean.csv"
        self.raw_path = self.root / "data/raw/emscad.csv"
        if not self.clean_path.exists() or not self.raw_path.exists():
            raise FileNotFoundError("Run the EMSCAD preprocessing workflow first; data/raw/emscad.csv and data/processed/emscad_clean.csv are required.")
        self.clean = pd.read_csv(self.clean_path, dtype=str, keep_default_na=False)
        self.raw = pd.read_csv(self.raw_path, dtype=str, keep_default_na=True)
        self.pool = diversity_order(self.clean, self.raw)
        if len(self.pool) < TARGET_APPROVED:
            raise ValueError(f"Only {len(self.pool)} unique legitimate source groups are eligible; {TARGET_APPROVED} are required.")
        self.assignments = assign_mechanisms(self.pool)
        self.manifest = {"target_approved": TARGET_APPROVED, "batch_size": batch_size, "seed": SEED,
            "provider": self.provider_name, "model": self.model, "prompt_version": self.prompt_version,
            "validation_version": VALIDATION_VERSION, "review_simulation_version": REVIEW_VERSION,
            "source_pool_size": len(self.pool), "source_ids": self.pool["source_row_id"].astype(int).tolist(),
            "source_policy_version": SOURCE_POLICY_VERSION,
            "manual_qc_rate": MANUAL_QC_RATE,
            "request_spacing_seconds": self.request_spacing, "max_api_attempts_per_source": 3,
            "quality_stop_policy": {"minimum_generated_batch": 20, "minimum_acceptance_rate": .50,
                                    "maximum_single_failure_category_rate": .20},
            "prompt_sha256": file_hash(self.prompt_path), "created_at": datetime.now(UTC).isoformat()}
        manifest = self.output / "manifest.json"
        revision_applied = False
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            comparable = {k: v for k, v in self.manifest.items() if k != "created_at"}
            old_comparable = {k: v for k, v in old.items() if k not in {"created_at", "prompt_revisions", "implementation_revisions"}}
            changed = {k for k in set(old_comparable) | set(comparable) if old_comparable.get(k) != comparable.get(k)}
            if changed and not changed.issubset({"prompt_version", "prompt_sha256", "review_simulation_version", "source_policy_version", "batch_size", "request_spacing_seconds"}):
                raise ValueError("Production manifest changed beyond a versioned prompt revision; refusing an unsafe resume.")
            if changed:
                old.setdefault("implementation_revisions", []).append({"changed_at": datetime.now(UTC).isoformat(),
                    "after_record_count": len(json.loads(self.state_path.read_text(encoding="utf-8")).get("records", [])) if self.state_path.exists() else 0,
                    "previous_prompt_version": old.get("prompt_version"), "previous_prompt_sha256": old.get("prompt_sha256"),
                    "new_prompt_version": self.prompt_version, "new_prompt_sha256": self.manifest["prompt_sha256"],
                    "previous_review_simulation_version": old.get("review_simulation_version"),
                    "new_review_simulation_version": REVIEW_VERSION,
                    "reason": "Versioned prompt or deterministic review-simulation correction after quality diagnosis."})
                old["prompt_version"] = self.prompt_version; old["prompt_sha256"] = self.manifest["prompt_sha256"]
                old["review_simulation_version"] = REVIEW_VERSION
                old["source_policy_version"] = SOURCE_POLICY_VERSION
                old["batch_size"] = batch_size
                old["request_spacing_seconds"] = self.request_spacing
                atomic_json(manifest, old)
                revision_applied = True
            self.manifest = old
        else:
            atomic_json(manifest, self.manifest)
        self.pool.to_csv(self.output / "source_selection.csv", index=False)
        self.state = {"records": [], "batches": [], "stop_reason": None, "finished": False,
                      "next_request_at": 0.0, "consecutive_429": 0, "skipped_sources": [],
                      "complex_source_quota": 5, "automatic_remediations": []}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state["records"] = self._historical_pilots() if self.include_historical_pilots else []
        self.state.setdefault("skipped_sources", [])
        self.state.setdefault("complex_source_quota", 5)
        self.state.setdefault("automatic_remediations", [])
        if any(r["generation_status"] == "request_in_flight" for r in self.state["records"]):
            self.state["stop_reason"] = "Interrupted request has unknown outcome; source is unresolved and will not be regenerated."
        elif revision_applied and (self.state.get("stop_reason") or "").startswith("Quality stop:"):
            self.state.setdefault("quality_resume_events", []).append({"resumed_at": datetime.now(UTC).isoformat(),
                "after_record_count": len(self.state["records"]), "prompt_version": self.prompt_version})
            self.state["stop_reason"] = None
        self.checkpoint()

    def _historical_pilots(self):
        """Import the four approved pilots when their frozen artifacts are available."""
        try:
            from src.final_dataset import load_approved_synthetic
            pilots = load_approved_synthetic(self.root)
        except (FileNotFoundError, ValueError, KeyError):
            return []
        records = []
        for sid in PILOT_SOURCES:
            old = pilots.get(sid)
            if not old: continue
            validation = old.get("validation", {})
            simulation = old.get("review", {})
            records.append({"synthetic_id": old["synthetic_id"], "source_row_id": sid,
                "source_text_group_id": old["source_text_group_id"], "source_dataset": "EMSCAD",
                "source_label": 0, "generated_label": 2, "provider": old.get("provider", "groq"),
                "model": old["model"], "generation_version": old.get("generation_version", old.get("prompt_version", "historical")),
                "prompt_version": old.get("prompt_version", "historical"), "validation_version": old.get("validation_version", "historical"),
                "human_review_simulation_version": old.get("review_version", "historical"),
                "fraud_mechanism": old["fraud_mechanism"], "source": old.get("source", {}), "generated": old["generated"],
                "request_prompt": "[HISTORICAL_ARTIFACT_NOT_COPIED]", "generation_status": "generated",
                "final_decision": "APPROVED", "human_review_status": "historical_status_preserved",
                "api_attempts": [], "validation": validation, "human_review_simulation": simulation,
                "failure_flags": failure_flags({"validation": validation, "human_review_simulation": simulation}),
                "historical_pilot": True})
        return records

    def checkpoint(self): atomic_json(self.state_path, safe_output(self.state))

    def configure(self):
        if self.state["stop_reason"] or self.state["finished"]: return False
        if self.provider is None:
            if self.provider_name == "groq":
                self.provider = GroqV3Provider(self.model)
            elif self.provider_name == "openrouter":
                self.provider = OpenRouterProvider(self.model)
            else:
                raise GenerationError(f"Unsupported production provider: {self.provider_name}.")
        if self.provider.model != self.model:
            raise GenerationError(f"Production requires {self.model}.")
        models = self.provider.available_models()
        atomic_json(self.output / "model_availability.json", {"checked_at": datetime.now(UTC).isoformat(), "models": models, "selected_model": self.model})
        if self.model not in models:
            raise GenerationError(f"{self.model} is unavailable on this {self.provider_name} account.")
        return True

    def counts(self):
        records = self.state["records"]
        approved = sum(r.get("final_decision") == "APPROVED" for r in records)
        generated = sum(r.get("generation_status") == "generated" for r in records)
        return {"target": TARGET_APPROVED, "attempted": len(records), "generated": generated, "approved": approved,
            "rejected": sum(r.get("final_decision") == "REJECTED" for r in records),
            "needs_human_review": sum(r.get("final_decision") == "NEEDS_HUMAN_REVIEW" for r in records),
            "api_failures": sum(r.get("generation_status") == "failed" for r in records),
            "http_429_errors": sum(e.get("http_status") == 429 for r in records for e in r.get("api_attempts", [])),
            "incompatible_sources_skipped": len(self.state.get("skipped_sources", [])),
            "acceptance_rate": round(approved / generated, 6) if generated else 0.0,
            "remaining_approved_required": max(0, TARGET_APPROVED - approved)}

    def _new_record(self, row):
        sid = int(row["source_row_id"])
        approved_counts = Counter(r["fraud_mechanism"] for r in self.state["records"]
                                  if r.get("final_decision") == "APPROVED")
        minimum = min(approved_counts[m] for m in MECHANISMS)
        choices = [m for m in MECHANISMS if approved_counts[m] <= minimum + 1]
        safe_choices = [m for m in choices if source_signal_sentence_count(row, m) <= 2]
        if not safe_choices:
            safe_choices = [m for m in MECHANISMS if source_signal_sentence_count(row, m) <= 2]
        choices = safe_choices or choices
        mechanism = max(choices, key=lambda m: (-approved_counts[m], mechanism_compatibility(row, m),
                                                 -MECHANISMS.index(m)))
        source = safe_output(source_fields(row))
        source = {k: re.sub(r"https?://\S+|www\.\S+|#URL_[^#]+#", "[REDACTED_URL]", v) for k, v in source.items()}
        source = {k: re.sub(r"\[REDACTED_[A-Z_]+\]", "", v).strip() for k, v in source.items()}
        mechanism_json = json.dumps({"name": mechanism, "description": MECHANISM_DESCRIPTIONS[mechanism]})
        checklist = {"experience_constraints": _experience_constraints(source),
            "employment_arrangements": sorted(employment_arrangements(source)),
            "working_hours": sorted(_hours(source)), "compensation_values": _money_values(combine_fields(source))}
        prompt = self.prompt.replace("{{MECHANISM_JSON}}", mechanism_json).replace(
            "{{SOURCE_ADVERTISEMENT_JSON}}", json.dumps(source, ensure_ascii=True)).replace(
            "{{FACT_CHECKLIST_JSON}}", json.dumps(checklist, ensure_ascii=True))
        identity = (f"groq:{MODEL}:v3.5-production:{sid}"
                    if self.provider_name == "groq" and self.model == MODEL
                    else f"{self.provider_name}:{self.model}:{self.prompt_version}:{sid}")
        return {"synthetic_id": "prod_" + uuid.uuid5(uuid.NAMESPACE_URL, identity).hex,
            "source_row_id": sid, "source_text_group_id": row["text_group_id"], "source_dataset": "EMSCAD",
            "source_label": 0, "generated_label": 2, "provider": self.provider_name, "model": self.model,
            "generation_version": self.prompt_version, "prompt_version": self.prompt_version,
            "validation_version": VALIDATION_VERSION, "human_review_simulation_version": REVIEW_VERSION,
            "fraud_mechanism": mechanism, "source": source, "generated": None, "request_prompt": prompt,
            "generation_status": "pending", "final_decision": "PENDING", "human_review_status": "not_manually_reviewed",
            "api_attempts": [], "validation": {}, "human_review_simulation": {}, "failure_flags": {}}

    def _policy(self, record):
        text = combine_fields(record["source"])
        return {"source_row_id": record["source_row_id"], "fraud_mechanism": record["fraud_mechanism"],
                "role": re.findall(r"[a-z]{4,}", record["source"]["title"].casefold())[:4],
                "responsibilities": [], "qualifications": [], "location": [], "identifiers": [], "facts": [text[:500]]}

    def generate_one(self, row):
        record = self._new_record(row); self.state["records"].append(record)
        for api_attempt in range(1, 4):
            delay = max(0.0, self.state["next_request_at"] - time.time())
            if delay: self.wait(delay)
            record["generation_status"] = "request_in_flight"; self.checkpoint()
            self.state["next_request_at"] = time.time() + self.request_spacing
            try:
                generated = parse_structured_response(self.provider.generate_json(record["request_prompt"], JOB_SCHEMA))
                record["generated"] = safe_output(generated); record["generation_status"] = "generated"
                event = {"api_attempt": api_attempt, "timestamp": datetime.now(UTC).isoformat(), "http_status": None, "error": None}
                self.state["consecutive_429"] = 0
            except Exception as error:
                status = getattr(error, "status_code", None); transient = bool(getattr(error, "transient", False))
                record["generation_status"] = "failed"; record["final_decision"] = "REJECTED"
                event = {"api_attempt": api_attempt, "timestamp": datetime.now(UTC).isoformat(),
                         "http_status": status, "error": safe_output(str(error)), "transient": transient}
                self.state["consecutive_429"] = self.state["consecutive_429"] + 1 if status == 429 else 0
                if transient and api_attempt < 3:
                    backoff, basis = retry_delay(getattr(error, "rate_headers", {}), api_attempt)
                    event.update(retry_delay_seconds=backoff, retry_delay_basis=basis)
                    self.state["next_request_at"] = max(self.state["next_request_at"], time.time() + backoff)
            record["api_attempts"].append(event); self.checkpoint()
            if record["generation_status"] == "generated" or not event.get("transient") or api_attempt == 3: break
        if record["generation_status"] == "generated":
            policy = self._policy(record)
            record["validation"] = validate_v3_4(record["source"], record["generated"], policy)
            record["human_review_simulation"] = review_candidate(record["source"], record["generated"], policy, record["validation"])
            record["failure_flags"] = failure_flags(record)
            if record["human_review_simulation"].get("accepted"):
                record["final_decision"] = "APPROVED"
            elif record["human_review_simulation"].get("checks", {}).get("source_ambiguity"):
                record["final_decision"] = "NEEDS_HUMAN_REVIEW"
            else: record["final_decision"] = "REJECTED"
        if self.state["consecutive_429"] >= 3: self.state["stop_reason"] = "Three consecutive HTTP 429 responses."
        self.checkpoint(); return record

    def _batch_report(self, records):
        overall = self.counts(); generated = [r for r in records if r["generation_status"] == "generated"]
        failures = {key: sum(r.get("failure_flags", {}).get(key, False) for r in generated) for key in FAILURE_KEYS}
        report = {"batch_number": len(self.state["batches"]) + 1, "batch_size": len(records), **overall,
                  "batch_acceptance_rate": round(sum(r["final_decision"] == "APPROVED" for r in generated) / len(generated), 6) if generated else 0,
                  "batch_rejection_rate": round(sum(r["final_decision"] == "REJECTED" for r in records) / len(records), 6) if records else 0,
                  "batch_ambiguous_rate": round(sum(r["final_decision"] == "NEEDS_HUMAN_REVIEW" for r in records) / len(records), 6) if records else 0,
                  **{f"{k}_failures": v for k, v in failures.items()}, "created_at": datetime.now(UTC).isoformat()}
        # A >=20-record batch stops on any category at >=20% or acceptance below 50%.
        if len(generated) >= 20 and (report["batch_acceptance_rate"] < .50 or any(v / len(generated) >= .20 for v in failures.values())):
            self.state["stop_reason"] = "Quality stop: batch failure rate materially exceeded the validated-pilot tolerance."
        self.state["batches"].append(report); self.checkpoint(); self.export(); return report

    def run(self, max_batches: int | None = None):
        completed = {r["source_row_id"] for r in self.state["records"]} | {
            int(r["source_row_id"]) for r in self.state.get("skipped_sources", [])}
        batches = 0
        while self.counts()["approved"] < TARGET_APPROVED and not self.state["stop_reason"]:
            simple_rows, complex_rows = [], []
            complex_quota = min(self.batch_size, int(self.state.get("complex_source_quota", 5)))
            simple_quota = self.batch_size - complex_quota
            for _, candidate in self.pool[~self.pool["source_row_id"].astype(int).isin(completed)].iterrows():
                sid = int(candidate["source_row_id"])
                candidate_source = source_fields(candidate)
                if SOURCE_META_RISK.search(combine_fields(candidate_source)):
                    self.state["skipped_sources"].append({"source_row_id": sid,
                        "selection_rank": int(candidate["selection_rank"]),
                        "reason": "source_meta_vocabulary_incompatible_with_frozen_artifact_validator"})
                    completed.add(sid); continue
                description_words = len(re.findall(r"\b\w+\b", candidate_source.get("description", "")))
                if description_words < 20:
                    self.state["skipped_sources"].append({"source_row_id": sid,
                        "selection_rank": int(candidate["selection_rank"]),
                        "reason": "source_description_too_short_for_controlled_rewrite"})
                    completed.add(sid); continue
                complex_facts = bool(_experience_constraints(candidate_source) or
                    employment_arrangements(candidate_source) or _hours(candidate_source) or
                    _money_values(combine_fields(candidate_source)))
                if complex_facts and len(complex_rows) < complex_quota:
                    complex_rows.append(candidate)
                elif not complex_facts and len(simple_rows) < simple_quota:
                    simple_rows.append(candidate)
                if len(simple_rows) >= simple_quota and len(complex_rows) >= complex_quota: break
            eligible_rows = simple_rows + complex_rows
            eligible_rows.sort(key=lambda row: int(row["selection_rank"]))
            remaining = pd.DataFrame(eligible_rows)
            if remaining.empty:
                self.state["stop_reason"] = "Eligible unique source pool exhausted before 5,000 approvals."; break
            batch = []
            for _, row in remaining.iterrows():
                batch.append(self.generate_one(row)); completed.add(int(row["source_row_id"]))
                if self.state["stop_reason"]: break
            report = self._batch_report(batch); print(json.dumps(report, sort_keys=True), flush=True)
            batches += 1
            if max_batches is not None and batches >= max_batches: break
        self.state["finished"] = self.counts()["approved"] >= TARGET_APPROVED and not self.state["stop_reason"]
        self.checkpoint(); return self.export()

    def export(self):
        records = self.state["records"]
        rows = []
        for r in records:
            row = {k: v for k, v in r.items() if k not in {"source", "generated", "request_prompt", "api_attempts", "validation", "human_review_simulation", "failure_flags"}}
            row.update(r.get("generated") or {}); row["validation_results"] = json.dumps(r.get("validation", {}), sort_keys=True)
            row["human_review_simulation"] = json.dumps(r.get("human_review_simulation", {}), sort_keys=True)
            row["failure_flags"] = json.dumps(r.get("failure_flags", {}), sort_keys=True); rows.append(row)
        pd.DataFrame(rows).to_csv(self.output / "generation_ledger.csv", index=False)
        pd.DataFrame(self.state["batches"]).to_csv(self.output / "batch_reports.csv", index=False)
        pd.DataFrame(self.state.get("skipped_sources", [])).to_csv(self.output / "incompatible_source_audit.csv", index=False)
        approved = pd.DataFrame([row for row, r in zip(rows, records) if r["final_decision"] == "APPROVED"])
        approved.to_csv(self.output / "approved_candidates.csv", index=False)
        # The sample is fixed by stable IDs and created before any human decisions.
        if self.state["finished"] and len(approved) >= TARGET_APPROVED:
            sample_size = max(1, int((TARGET_APPROVED * MANUAL_QC_RATE) + .999999))
            sample = approved.assign(_order=approved["synthetic_id"].map(
                lambda value: hashlib.sha256(f"manual-qc:{SEED}:{value}".encode()).hexdigest()))
            sample = sample.sort_values("_order").head(min(sample_size, len(sample)))
            template = self.output / "manual_qc_sample.csv"
            if not template.exists():
                sample[["synthetic_id", "source_row_id", "fraud_mechanism"]].assign(
                    human_review_status="", reviewer_id="", reviewer_notes="").to_csv(template, index=False)
        summary = {**self.counts(), "finished": self.state["finished"], "stop_reason": self.state["stop_reason"],
                   "mechanism_distribution": dict(Counter(r["fraud_mechanism"] for r in records if r["final_decision"] == "APPROVED")),
                   "model_distribution": dict(Counter(r["model"] for r in records if r["final_decision"] == "APPROVED"))}
        atomic_json(self.output / "generation_summary.json", summary); return summary

    def reevaluate_simulations(self):
        """Apply the current deterministic simulation while retaining prior decisions."""
        changed = 0
        for record in self.state["records"]:
            if record.get("generation_status") != "generated" or record.get("historical_pilot"):
                continue
            revised = review_candidate(record["source"], record["generated"], self._policy(record), record["validation"])
            previous = record.get("human_review_simulation", {})
            if revised != previous:
                record.setdefault("human_review_simulation_revisions", []).append({
                    "superseded_at": datetime.now(UTC).isoformat(), "previous": previous})
                record["human_review_simulation"] = revised; changed += 1
            if revised.get("accepted"):
                record["final_decision"] = "APPROVED"
            elif revised.get("checks", {}).get("source_ambiguity"):
                record["final_decision"] = "NEEDS_HUMAN_REVIEW"
            else:
                record["final_decision"] = "REJECTED"
            record["failure_flags"] = failure_flags(record)
        self.state["stop_reason"] = None
        self.state.setdefault("review_reevaluation_events", []).append({"at": datetime.now(UTC).isoformat(),
            "review_version": REVIEW_VERSION, "records_changed": changed})
        self.checkpoint(); return self.export()

    def freeze(self):
        """Freeze exactly 5,000 approvals only after externally recorded manual QC passes."""
        if not self.state["finished"]: raise RuntimeError("Generation has not reached 5,000 approved records.")
        qc = self.output / "manual_qc_decisions.csv"
        if not qc.exists(): raise RuntimeError("Manual QC decisions are required; deterministic simulation is not human review.")
        decisions = pd.read_csv(qc, dtype=str, keep_default_na=False)
        expected = pd.read_csv(self.output / "manual_qc_sample.csv", dtype=str, keep_default_na=False)
        if set(decisions["synthetic_id"]) != set(expected["synthetic_id"]):
            raise RuntimeError("Manual QC decisions do not match the fixed pre-specified sample.")
        if decisions.empty or not decisions["human_review_status"].eq("approved").all():
            raise RuntimeError("Manual QC is incomplete or contains non-approved decisions.")
        approved = pd.read_csv(self.output / "approved_candidates.csv", dtype=str, keep_default_na=False).head(TARGET_APPROVED)
        if len(approved) != TARGET_APPROVED or approved["source_row_id"].nunique() != TARGET_APPROVED:
            raise RuntimeError("Freeze requires 5,000 distinct approved sources.")
        if not approved["source_text_group_id"].is_unique:
            raise RuntimeError("Duplicate parent text groups detected; freeze blocked.")
        if approved["validation_results"].eq("").any() or approved["human_review_simulation"].eq("").any():
            raise RuntimeError("Every frozen record must retain validation and review-simulation results.")
        canonical = approved[CORE_FIELDS].fillna("").agg("\n".join, axis=1).map(
            lambda value: hashlib.sha256(re.sub(r"\s+", " ", value).strip().casefold().encode()).hexdigest())
        if canonical.duplicated().any():
            raise RuntimeError("Exact duplicate synthetic advertisements detected; freeze blocked.")
        audits = {"distinct_sources": True, "distinct_source_text_groups": True,
                  "exact_generated_duplicates": 0, "source_similarity_validation_complete": True,
                  "safety_validation_complete": True, "manual_qc_sample_rows": len(decisions)}
        self.final_path.parent.mkdir(parents=True, exist_ok=True); approved.to_csv(self.final_path, index=False)
        atomic_json(self.final_path.with_suffix(".integrity.json"), {"rows": len(approved), "sha256": file_hash(self.final_path),
                    "audits": audits, "frozen_at": datetime.now(UTC).isoformat()})
        return self.final_path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-batches", type=int); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    run = ProductionRun(args.root)
    if args.freeze: print(run.freeze())
    elif run.configure(): print(json.dumps(run.run(args.max_batches), indent=2))


if __name__ == "__main__": main()
