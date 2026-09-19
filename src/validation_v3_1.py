"""Follow-up v3.1 screening rules.

This module retains the frozen v3 checks and narrows one false-positive: a
source-supported age eligibility range is not an experience requirement.
"""

import re

from src import validation_v3 as v3
from src.validation import combine_fields

VERSION = "v3.1-groq-pilot"


def _age_ranges(text):
    """Return explicit age ranges, not arbitrary numeric durations."""
    pattern = r"\b(?:ages?\s*)?(\d{1,2})\s*[-\u2013\u2014]\s*(\d{1,2})\s*(?:years?\s*old|year[- ]?olds?)\b"
    return {f"{start}-{end}" for start, end in re.findall(pattern, v3.normalize(text))}


def unsupported_facts_v3_1(source, generated, policy):
    """Keep v3 checks, exempting only source-supported explicit age eligibility."""
    findings = v3.unsupported_facts(source, generated, policy)
    source_ages = _age_ranges(combine_fields(source))
    generated_ages = _age_ranges(combine_fields(generated))
    supported_ages = source_ages & generated_ages

    filtered = []
    for finding in findings:
        match = re.fullmatch(r"Unsupported experience duration: (\d{1,2})\s*[-\u2013\u2014]\s*(\d{1,2})\s*years?", finding)
        if match and f"{match.group(1)}-{match.group(2)}" in supported_ages:
            continue
        filtered.append(finding)
    return filtered


def validate_v3_1(source, generated, policy):
    """Run frozen v3 validation, replacing only the age-eligibility fact check."""
    result = v3.validate_v3(source, generated, policy)
    result["validation_version"] = VERSION
    if not result.get("structural_valid"):
        result["human_review_status"] = "pending_human_review"
        return result

    original_findings = v3.unsupported_facts(source, generated, policy)
    corrected_findings = unsupported_facts_v3_1(source, generated, policy)
    if corrected_findings != original_findings:
        removed = set(original_findings) - set(corrected_findings)
        result["reasons"] = [reason for reason in result["reasons"] if reason not in removed]
        result["age_eligibility_exemption"] = sorted(removed)

    result["unsupported_facts_clear"] = not corrected_findings
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
        "v3.1 exempts only matching explicit source-supported age eligibility ranges."
    )
    return result
