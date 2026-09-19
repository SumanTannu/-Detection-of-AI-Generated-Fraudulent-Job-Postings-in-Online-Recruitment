"""Transparent validation for the Phase 2B-A generation pilot.

Automatic validation is a triage mechanism, not a substitute for human review.
Records that pass it are still marked needs_human_review by default.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from src.generation import (
    CORE_FIELDS,
    GenerationError,
    GenerationProvider,
    build_validation_prompt,
)


COMMENTARY_PATTERNS = (
    "ai-generated scam",
    "ai generated scam",
    "here is the fraudulent version",
    "modified version",
    "class 2",
    "mock - not real generated data",
)
PLACEHOLDER_PATTERNS = ("lorem ipsum", "[insert", "todo", "tbd", "xxx")
CONTACT_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|(?<!\w)(?:\+?\d[\d .()/-]{6,}\d)"
)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")

# Multiple categories are required; no individual term can decide acceptance.
FRAUD_SIGNAL_PATTERNS = {
    "upfront_payment": (
        r"(?:application|registration|training|processing|background)\s+(?:fee|charge|payment)",
        r"pay\s+(?:a|an|the)?\s*(?:fee|charge|deposit)",
    ),
    "financial_request": (
        r"(?:bank|account|payment)\s+(?:details|information)",
        r"(?:advance|wire|transfer)\s+(?:payment|funds|money)",
    ),
    "sensitive_information": (
        r"(?:passport|identity|government id|social security)\s+(?:copy|scan|number|details)",
        r"submit\s+(?:identity|personal)\s+(?:documents|information)",
    ),
    "unrealistic_compensation": (
        r"(?:guaranteed|instant)\s+(?:income|earnings|payment)",
        r"(?:no experience|no interview).{0,60}(?:high|large|exceptional)\s+(?:salary|income|pay)",
    ),
    "suspicious_urgency": (
        r"(?:apply|respond|act)\s+(?:immediately|today|within \d+ hours)",
        r"limited\s+(?:slots|time).{0,50}(?:payment|fee|deposit)",
    ),
    "suspicious_process": (
        r"(?:no|skip)\s+(?:interview|verification|screening)",
        r"(?:offer|hired|selected)\s+(?:immediately|before)\s+(?:interview|verification)",
    ),
}


def combine_fields(advertisement: dict[str, Any]) -> str:
    """Join known fields for validation without changing the stored record."""
    return "\n".join(
        str(advertisement.get(field, "") or "").strip() for field in CORE_FIELDS
    ).strip()


def tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def validate_structure(generated: Any) -> dict[str, Any]:
    """Require an object with all five string fields and meaningful content."""
    reasons: list[str] = []
    if not isinstance(generated, dict):
        return {"valid": False, "reasons": ["Generated output is not a JSON object."]}
    missing = [field for field in CORE_FIELDS if field not in generated]
    if missing:
        reasons.append(f"Missing required fields: {', '.join(missing)}.")
    if set(generated) - set(CORE_FIELDS):
        reasons.append("Unexpected output fields.")
    non_string = [field for field in CORE_FIELDS if field in generated and not isinstance(generated[field], str)]
    if non_string:
        reasons.append(f"Non-string fields: {', '.join(non_string)}.")
    if not reasons and not combine_fields(generated):
        reasons.append("Generated advertisement is completely empty.")
    return {"valid": not reasons, "reasons": reasons}


def validate_context(source: dict[str, str], generated: dict[str, str]) -> dict[str, Any]:
    """Check role continuity and non-trivial source-context overlap."""
    source_title = source.get("title", "")
    generated_title = generated.get("title", "")
    title_similarity = jaccard_similarity(source_title, generated_title)
    title_anchor_tokens = {
        token for token in tokens(source_title) if len(token) >= 4
    }
    title_anchor_preserved = bool(title_anchor_tokens & tokens(generated_title))
    source_text = combine_fields(source)
    generated_text = combine_fields(generated)
    context_overlap = jaccard_similarity(source_text, generated_text)
    source_length = max(len(tokens(source_text)), 1)
    generated_length = len(tokens(generated_text))
    length_ratio = generated_length / source_length
    valid = (title_similarity >= 0.35 or title_anchor_preserved) and context_overlap >= 0.03 and 0.25 <= length_ratio <= 3.5
    reasons: list[str] = []
    if not (title_similarity >= 0.35 or title_anchor_preserved):
        reasons.append("Job-title similarity is too low for reliable role preservation.")
    if context_overlap < 0.03:
        reasons.append("Too little lexical context overlaps with the source advertisement.")
    if not 0.25 <= length_ratio <= 3.5:
        reasons.append("Generated length is outside the permitted source-relative range.")
    return {
        "valid": valid,
        "title_similarity": round(title_similarity, 4),
        "title_anchor_preserved": title_anchor_preserved,
        "context_overlap": round(context_overlap, 4),
        "length_ratio": round(length_ratio, 4),
        "reasons": reasons,
    }


def validate_fraud_intent(generated: dict[str, str]) -> dict[str, Any]:
    """Require multiple explicit fraud-signal categories, not a single keyword."""
    text = combine_fields(generated).lower()
    matched_categories: list[str] = []
    matched_patterns: dict[str, list[str]] = {}
    for category, patterns in FRAUD_SIGNAL_PATTERNS.items():
        hits = [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)]
        if hits:
            matched_categories.append(category)
            matched_patterns[category] = hits
    core_categories = {"upfront_payment", "financial_request", "sensitive_information", "suspicious_process"}
    valid = len(matched_categories) >= 2 and bool(core_categories.intersection(matched_categories))
    reason = "" if valid else "At least two fraud-signal categories, including a core recruitment-risk category, are required."
    return {
        "valid": valid,
        "matched_categories": matched_categories,
        "matched_patterns": matched_patterns,
        "reasons": [reason] if reason else [],
    }


def has_repeated_phrase(text: str, ngram_size: int = 4, repetition_limit: int = 3) -> bool:
    words = TOKEN_PATTERN.findall(text.lower())
    if len(words) < ngram_size:
        return False
    ngrams = Counter(tuple(words[index : index + ngram_size]) for index in range(len(words) - ngram_size + 1))
    return any(count >= repetition_limit for count in ngrams.values())


def validate_generation_quality(source: dict[str, str], generated: dict[str, str]) -> dict[str, Any]:
    """Reject commentary, placeholders, contacts, boilerplate, and trivial copies."""
    source_text = combine_fields(source)
    generated_text = combine_fields(generated)
    normalized = re.sub(r"\s+", " ", generated_text).strip().lower()
    source_normalized = re.sub(r"\s+", " ", source_text).strip().lower()
    reasons: list[str] = []
    if any(pattern in normalized for pattern in COMMENTARY_PATTERNS):
        reasons.append("Contains generation commentary or a mock marker.")
    if any(pattern in normalized for pattern in PLACEHOLDER_PATTERNS):
        reasons.append("Contains placeholder text.")
    if CONTACT_PATTERN.search(generated_text):
        reasons.append("Contains an email address or phone-like contact channel.")
    if has_repeated_phrase(generated_text):
        reasons.append("Contains excessive repeated boilerplate.")
    sequence_ratio = SequenceMatcher(None, source_normalized, normalized).ratio()
    if sequence_ratio >= 0.90 and len(normalized) >= max(80, len(source_normalized) * 0.85):
        reasons.append("Appears to be a trivial near-copy of the source advertisement.")
    return {
        "valid": not reasons,
        "sequence_similarity": round(sequence_ratio, 4),
        "reasons": reasons,
    }


def run_automatic_validation(record: dict[str, Any]) -> dict[str, Any]:
    """Run structural, context, fraud, and quality checks for one generated record."""
    if record.get("generation_status") != "generated":
        return {
            "structural": {"valid": False, "reasons": ["Generation did not complete."]},
            "context": {"valid": False, "reasons": ["No generated output to compare."]},
            "fraud_intent": {"valid": False, "reasons": ["No generated output to inspect."]},
            "quality": {"valid": False, "reasons": ["No generated output to inspect."]},
            "automatic_passed": False,
            "validation_status": "rejected",
        }

    generated = record.get("generated")
    source = record.get("source", {})
    structural = validate_structure(generated)
    if not structural["valid"]:
        context = {"valid": False, "reasons": ["Skipped because structure is invalid."]}
        fraud_intent = {"valid": False, "reasons": ["Skipped because structure is invalid."]}
        quality = {"valid": False, "reasons": ["Skipped because structure is invalid."]}
    else:
        context = validate_context(source, generated)
        fraud_intent = validate_fraud_intent(generated)
        quality = validate_generation_quality(source, generated)
    automatic_passed = all(
        check["valid"] for check in (structural, context, fraud_intent, quality)
    )
    # Human review is mandatory for the pilot; automatic passage is not final acceptance.
    status = "needs_human_review" if automatic_passed else "rejected"
    return {
        "structural": structural,
        "context": context,
        "fraud_intent": fraud_intent,
        "quality": quality,
        "automatic_passed": automatic_passed,
        "validation_status": status,
    }


def run_llm_validator(
    source: dict[str, str],
    generated: dict[str, str],
    provider: GenerationProvider | None,
    validation_prompt_template: str,
    enabled: bool = False,
) -> dict[str, Any] | None:
    """Optionally collect secondary LLM judgement without treating it as ground truth."""
    if not enabled or provider is None or provider.generation_mode != "api":
        return None
    prompt = build_validation_prompt(validation_prompt_template, source, generated)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "context_preserved", "fraud_intent_present", "coherent",
            "substantially_transformed", "generation_artifacts", "quality_score",
            "accepted", "reason",
        ],
        "properties": {
            "context_preserved": {"type": "boolean"},
            "fraud_intent_present": {"type": "boolean"},
            "coherent": {"type": "boolean"},
            "substantially_transformed": {"type": "boolean"},
            "generation_artifacts": {"type": "boolean"},
            "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "accepted": {"type": "boolean"},
            "reason": {"type": "string"},
        },
    }
    response = provider.generate_json(prompt, schema)
    required = schema["required"]
    if not isinstance(response, dict) or set(response) != set(required):
        raise GenerationError("LLM validator returned an unexpected JSON structure.")
    if not isinstance(response["quality_score"], int) or not 1 <= response["quality_score"] <= 5:
        raise GenerationError("LLM validator quality_score must be an integer from 1 to 5.")
    bool_fields = [field for field in required if field not in {"quality_score", "reason"}]
    if not all(isinstance(response[field], bool) for field in bool_fields):
        raise GenerationError("LLM validator boolean fields have invalid types.")
    if not isinstance(response["reason"], str):
        raise GenerationError("LLM validator reason must be a string.")
    return json.loads(json.dumps(response))


def attach_validation(
    record: dict[str, Any], llm_validation: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Attach the automatic audit record in the required provenance shape."""
    automatic = run_automatic_validation(record)
    record["validation"] = {
        "structural_valid": automatic["structural"]["valid"],
        "context_preserved": automatic["context"]["valid"],
        "fraud_intent_present": automatic["fraud_intent"]["valid"],
        "quality_ok": automatic["quality"]["valid"],
        "automatic_passed": automatic["automatic_passed"],
        "validation_status": automatic["validation_status"],
        "validation_details": automatic,
        "llm_validation": llm_validation,
    }
    return record


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Produce one CSV-ready audit row without dropping provenance information."""
    validation = record.get("validation", {})
    generated = record.get("generated") or {}
    source = record.get("source") or {}
    return {
        "synthetic_id": record.get("synthetic_id"),
        "source_row_id": record.get("source_row_id"),
        "source_text_group_id": record.get("source_text_group_id"),
        "source_dataset": record.get("source_dataset"),
        "source_label": record.get("source_label"),
        "generated_label": record.get("generated_label"),
        "provider": record.get("provider"),
        "model": record.get("model"),
        "generation_mode": record.get("generation_mode"),
        "generation_timestamp": record.get("generation_timestamp"),
        "prompt_version": record.get("prompt_version"),
        "generation_status": record.get("generation_status"),
        "generation_attempts": record.get("generation_attempts"),
        "generation_error": record.get("generation_error"),
        "validation_status": validation.get("validation_status"),
        "structural_valid": validation.get("structural_valid"),
        "context_preserved": validation.get("context_preserved"),
        "fraud_intent_present": validation.get("fraud_intent_present"),
        "quality_ok": validation.get("quality_ok"),
        "automatic_passed": validation.get("automatic_passed"),
        "source_title": source.get("title", ""),
        "generated_title": generated.get("title", ""),
        "generated_company_profile": generated.get("company_profile", ""),
        "generated_description": generated.get("description", ""),
        "generated_requirements": generated.get("requirements", ""),
        "generated_benefits": generated.get("benefits", ""),
        "fraud_signal_categories": "; ".join(
            validation.get("validation_details", {})
            .get("fraud_intent", {})
            .get("matched_categories", [])
        ),
        "validation_reasons": " | ".join(
            reason
            for check in ("structural", "context", "fraud_intent", "quality")
            for reason in validation.get("validation_details", {}).get(check, {}).get("reasons", [])
        ),
        "llm_validation": json.dumps(validation.get("llm_validation"), ensure_ascii=False),
    }


def save_validated_records(records: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Save only automatic-pass records; final acceptance remains a human decision."""
    rows = [flatten_record(record) for record in records
            if record.get("generation_mode") == "api"
            and record.get("validation", {}).get("automatic_passed")]
    columns = list(flatten_record({
        "validation": {"validation_details": {}, "llm_validation": None},
        "source": {}, "generated": {},
    }).keys())
    frame = pd.DataFrame(rows, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame


VALIDATION_VERSION = "v2-groq-pilot"


def validate_pilot_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Reuse triage checks but reserve semantic judgements for human review."""
    attach_validation(record)
    validation = record["validation"]
    details = validation["validation_details"]
    generated = record.get("generated") or {}
    text = combine_fields(generated)
    endpoints = re.findall(r"https?://\S+|www\.\S+|\b[A-Za-z0-9.-]+\.(?:com|org|net|io)\b", text)
    secret_patterns = re.findall(r"(?:gsk_|sk-)[A-Za-z0-9_-]+|\b(?:\d[ -]?){13,19}\b", text)
    if endpoints or secret_patterns:
        details["quality"]["valid"] = False
        details["quality"]["reasons"].append("Contact endpoint or credential/account-like value detected.")
    if generated:
        source_normal = re.sub(r"\s+", " ", combine_fields(record["source"])).strip().lower()
        generated_normal = re.sub(r"\s+", " ", text).strip().lower()
        if len(source_normal) > 80 and source_normal in generated_normal:
            details["quality"]["valid"] = False
            details["quality"]["reasons"].append("Full source retained verbatim with added material.")

    # Lexical similarity is evidence for review, never proof of semantic equivalence.
    hard_failure = not details["structural"]["valid"] or not details["quality"]["valid"]
    all_pass = all(details[c]["valid"] for c in ("structural", "context", "fraud_intent", "quality"))
    if record.get("generation_mode") != "api":
        hard_failure, all_pass = True, False
    result = "auto_reject" if hard_failure else "auto_pass" if all_pass else "needs_human_review"
    validation.update({
        "quality_ok": details["quality"]["valid"],
        "automatic_passed": all_pass,
        "automatic_result": result,
        "validation_status": "auto_reject" if hard_failure else "needs_human_review",
        "human_review_completed": False,
        "semantic_context_verified": None,
        "validation_version": VALIDATION_VERSION,
    })
    details["automatic_passed"] = all_pass
    details["validation_status"] = validation["validation_status"]
    record["validation_status"] = validation["validation_status"]
    record["validation_version"] = VALIDATION_VERSION
    return record


def create_human_review_template(records: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Create blank review fields; no row is presented as human-approved."""
    rows = []
    for record in records:
        generated = record.get("generated") or {}
        rows.append({
            "synthetic_id": record.get("synthetic_id"),
            "source_row_id": record.get("source_row_id"),
            "generated_title": generated.get("title", ""),
            "generated_description": generated.get("description", ""),
            "review_context_preserved": "",
            "review_fraudulent_intent": "",
            "review_coherence": "",
            "review_acceptable": "",
            "reviewer_notes": "",
            "generation_mode": record.get("generation_mode"),
            "automatic_validation_status": record.get("validation", {}).get("validation_status"),
        })
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame
