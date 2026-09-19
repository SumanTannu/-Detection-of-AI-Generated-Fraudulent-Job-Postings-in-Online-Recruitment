"""v3.2 screening: contextual fraud integration without a sentence-count threshold."""

import re

from src import validation_v3 as v3
from src.validation import combine_fields
from src.validation_v3_1 import validate_v3_1

VERSION = "v3.2-groq-pilot"
RECRUITMENT_CONTEXT = re.compile(
    r"\b(?:application|applicant|apply|candidate|recruitment|pre[- ]?screening|screening|"
    r"interview|selection|enrollment|onboarding|open day|hiring)\b"
)
APPENDED_MARKER = re.compile(r"^(?:please note|important|warning|caution|disclaimer)\b")


def _sentences(generated):
    return [sentence.strip() for sentence in re.split(
        r"(?<=[.!?;])\s+|\n+", v3.normalize(combine_fields(generated))
    ) if sentence.strip()]


def contextual_integration(generated, mechanism):
    """Screen for a coherent recruitment step rather than repeated signals."""
    sentences = _sentences(generated)
    patterns = v3.SIGNALS[mechanism]
    linked = [(index, sentence) for index, sentence in enumerate(sentences)
              if any(re.search(pattern, sentence) for pattern in patterns)]
    recruitment_linked = [(index, sentence) for index, sentence in linked if RECRUITMENT_CONTEXT.search(sentence)]
    appended = [sentence for index, sentence in linked
                if APPENDED_MARKER.search(sentence) or (
                    index == len(sentences) - 1 and not RECRUITMENT_CONTEXT.search(sentence)
                )]
    excessive = len(linked) > 3
    accepted = bool(recruitment_linked) and not appended and not excessive
    return {
        "accepted": accepted,
        "linked_sentence_count": len(linked),
        "recruitment_linked_sentence_count": len(recruitment_linked),
        "appended_signal_count": len(appended),
        "excessive_repetition": excessive,
    }


def validate_v3_2(source, generated, policy):
    """Retain v3.1 checks; replace only its two-sentence integration threshold."""
    if not isinstance(generated, dict):
        return {
            "validation_version": VERSION, "structural_valid": False, "automatic_passed": False,
            "validation_status": "auto_reject", "reasons": ["No parseable generated record available."],
            "human_review_status": "pending_human_review",
        }
    result = validate_v3_1(source, generated, policy)
    result["validation_version"] = VERSION
    if not result.get("structural_valid"):
        result["human_review_status"] = "pending_human_review"
        return result

    integration = contextual_integration(generated, policy["fraud_mechanism"])
    result["fraud_integrated"] = integration["accepted"]
    result["contextual_integration"] = integration
    result["reasons"] = [reason for reason in result["reasons"]
                         if reason != "Mechanism is confined to one sentence or lacks recruitment integration."]
    if not integration["accepted"]:
        if integration["appended_signal_count"]:
            result["reasons"].append("Fraud signal appears as a warning-style or disconnected appended statement.")
        elif integration["excessive_repetition"]:
            result["reasons"].append("Fraud mechanism is repeated excessively rather than integrated naturally.")
        else:
            result["reasons"].append("Fraud signal is not connected to a recruitment or application step.")

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
        "v3.2 treats integration as a contextual recruitment-step check, not a sentence-count test."
    )
    return result
