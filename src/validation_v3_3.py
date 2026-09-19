"""v3.3 screening with explicit employment-arrangement preservation."""

import re

from src import validation_v3 as v3
from src.validation_v3_1 import unsupported_facts_v3_1
from src.validation_v3_2 import validate_v3_2

VERSION = "v3.3-groq-pilot"
EMPLOYMENT_PATTERNS = {
    "full-time": r"\bfull[- ]time\b",
    "part-time": r"\bpart[- ]time\b",
    "temporary": r"\btemporary\b",
    "permanent": r"\bpermanent\b",
    "contract": r"\bcontract(?:or|ual)?\b",
    "internship": r"\binternship\b|\bintern\b",
    "traineeship": r"\btraineeship\b|\btrainee program\b",
    "apprenticeship": r"\bapprenticeship\b|\bapprentice\b",
    "volunteer": r"\bvolunteer(?:ing)?\b",
    "freelance": r"\bfreelanc(?:e|er|ing)\b",
    "seasonal": r"\bseasonal\b",
    "hybrid": r"\bhybrid\b",
    "remote": r"\bremote\b|\bwork from home\b",
}
HOURS_PATTERN = r"\b\d{1,2}\s*(?:hours?|hrs?)\s*(?:per|a)\s*week\b"


def _employment_text(record):
    return v3.normalize(" ".join(record.get(field, "") for field in ["title", "description", "requirements"]))


def employment_arrangements(record):
    text = _employment_text(record)
    return {name for name, pattern in EMPLOYMENT_PATTERNS.items() if re.search(pattern, text)}


def _hours(record):
    return set(re.findall(HOURS_PATTERN, _employment_text(record)))


def unsupported_facts_v3_3(source, generated, policy):
    """Keep v3.1 fact checks and add source-to-output arrangement preservation."""
    findings = [reason for reason in unsupported_facts_v3_1(source, generated, policy)
                if not reason.startswith("Unsupported employment arrangement:")]
    source_arrangements = employment_arrangements(source)
    generated_arrangements = employment_arrangements(generated)
    for arrangement in sorted(generated_arrangements - source_arrangements):
        findings.append(f"Unsupported employment arrangement: {arrangement}")
    for arrangement in sorted(source_arrangements - generated_arrangements):
        findings.append(f"Source employment arrangement omitted or changed: {arrangement}")
    source_hours, generated_hours = _hours(source), _hours(generated)
    for hours in sorted(generated_hours - source_hours):
        findings.append(f"Unsupported working-hours claim: {hours}")
    for hours in sorted(source_hours - generated_hours):
        findings.append(f"Source working-hours claim omitted or changed: {hours}")
    return findings


def validate_v3_3(source, generated, policy):
    """Retain v3.2 validation, strengthening only fact preservation."""
    result = validate_v3_2(source, generated, policy)
    result["validation_version"] = VERSION
    if not result.get("structural_valid"):
        result["human_review_status"] = "pending_human_review"
        return result

    prior_facts = unsupported_facts_v3_1(source, generated, policy)
    current_facts = unsupported_facts_v3_3(source, generated, policy)
    result["reasons"] = [reason for reason in result["reasons"] if reason not in prior_facts]
    result["reasons"].extend(reason for reason in current_facts if reason not in result["reasons"])
    result["unsupported_facts_clear"] = not current_facts
    result["employment_arrangement_comparison"] = {
        "source": sorted(employment_arrangements(source)),
        "generated": sorted(employment_arrangements(generated)),
        "new": sorted(employment_arrangements(generated) - employment_arrangements(source)),
        "missing": sorted(employment_arrangements(source) - employment_arrangements(generated)),
    }
    result["automatic_passed"] = all(
        value for key, value in result.items()
        if key.endswith("_clear") or key.endswith("_preserved") or key in {
            "structural_valid", "fraud_intent_present", "fraud_integrated", "source_similarity_ok"
        }
    )
    result["validation_status"] = "auto_pass" if result["automatic_passed"] else "auto_reject"
    result["human_review_status"] = "pending_human_review"
    result["limitations"] = (
        "Source anchors and fact patterns are screening evidence, not exhaustive semantic verification. "
        "v3.3 compares explicit employment-arrangement and weekly-hours wording between source and output."
    )
    return result
