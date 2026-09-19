"""Deterministic human-review-style gate for v3.5 candidates.

This is deliberately not a claim of human approval. It records conservative,
inspectable screening criteria that are separate from the automatic validator.
"""

import re
from difflib import SequenceMatcher

from src import validation_v3 as v3
from src.validation import combine_fields, jaccard_similarity

VERSION = "v3.5-human-review-simulation"
META = re.compile(r"\b(?:pilot|research|academic|experiment|study|evaluation|dataset|benchmark|synthetic|generated|"
                  r"ai(?:-generated)?|llm|validator|prompt|fraud detection)\b", re.I)
AI_ARTIFACT = re.compile(r"\b(?:here is|below is|as requested|i have created|this job posting|the following)\b", re.I)
UNSAFE = re.compile(r"https?://|www\.|\b\S+@\S+\.\S+|\b(?:gsk_|sk-)[\w-]+|\b(?:\d[ -]?){13,19}\b", re.I)
APPENDED = re.compile(r"^(?:please note|important|warning|caution|disclaimer)\b", re.I)
RECRUITMENT = re.compile(r"\b(?:application|apply|applicant|candidate|screening|interview|selection|"
                          r"recruitment|onboarding|verification|hiring|considered|schedule)\b", re.I)


def _sentences(record):
    entries = []
    for field, text in record.items():
        for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text or ""):
            if sentence.strip():
                entries.append((field, sentence.strip()))
    return entries


def _mechanism_sentences(generated, mechanism):
    patterns = list(v3.SIGNALS[mechanism])
    if mechanism == "implausible_recruitment":
        patterns[0] = r"guarantee(?:d)?|automatic.{0,20}(?:offer|selection)|(?:offer|selected).{0,20}without"
    return [(field, sentence) for field, sentence in _sentences(generated)
            if any(re.search(pattern, sentence, re.I) for pattern in patterns)]


def review_candidate(source, generated, policy, validation):
    """Return an auditable simulation result; never calls an LLM or overrides validation."""
    source_text, generated_text = combine_fields(source), combine_fields(generated)
    reasons, checks = [], {}
    title_ok = bool(generated.get("title", "").strip()) and generated["title"].casefold() == source.get("title", "").casefold()
    checks["job_identity_clear"] = title_ok
    if not title_ok:
        reasons.append("The original job title is not preserved clearly.")
    linked = _mechanism_sentences(generated, policy["fraud_mechanism"])
    checks["fraud_mechanism_identifiable"] = bool(linked)
    if not linked:
        reasons.append("The assigned fraudulent mechanism is not clearly identifiable.")
    recruitment_linked = [(field, sentence) for field, sentence in linked if RECRUITMENT.search(sentence)]
    checks["fraud_workflow_integrated"] = bool(recruitment_linked) and not any(
        field == "benefits" or APPENDED.search(sentence) for field, sentence in linked)
    if not checks["fraud_workflow_integrated"]:
        reasons.append("The fraud signal is not naturally embedded in a recruitment workflow.")
    checks["no_warning_style_append"] = not any(APPENDED.search(sentence) for _, sentence in linked)
    if not checks["no_warning_style_append"]:
        reasons.append("The fraud signal is presented as a warning-style appendage.")
    checks["employment_conditions_preserved"] = not any(
        "employment arrangement" in item or "working-hours" in item for item in validation.get("reasons", []))
    if not checks["employment_conditions_preserved"]:
        reasons.append("A source-supported employment condition changed.")
    checks["no_unsupported_facts"] = validation.get("unsupported_facts_clear", False)
    if not checks["no_unsupported_facts"]:
        reasons.append("The candidate contains unsupported or retained identifying facts.")
    compensation = validation.get("compensation_analysis", {})
    checks["compensation_not_reinterpreted"] = not compensation.get("reasons")
    checks["source_ambiguity"] = compensation.get("review_required", False)
    if checks["source_ambiguity"]:
        reasons.append("Source compensation is intrinsically ambiguous and requires human review.")
    elif not checks["compensation_not_reinterpreted"]:
        reasons.append("Compensation was changed or reinterpreted.")
    checks["coherent_standalone_posting"] = len(generated_text.split()) >= 25 and not AI_ARTIFACT.search(generated_text)
    if not checks["coherent_standalone_posting"]:
        reasons.append("The text does not read as a coherent standalone job advertisement.")
    checks["no_research_meta"] = not bool(META.search(generated_text))
    if not checks["no_research_meta"]:
        reasons.append("Research or generation meta-language is present.")
    checks["no_ai_artifact"] = not bool(AI_ARTIFACT.search(generated_text))
    if not checks["no_ai_artifact"]:
        reasons.append("Obvious AI/conversational framing is present.")
    checks["safe_representation"] = not bool(UNSAFE.search(generated_text))
    if not checks["safe_representation"]:
        reasons.append("Unsafe operational contact, account, credential, or identity-like content is present.")
    similarity = jaccard_similarity(source_text, generated_text)
    description_ratio = SequenceMatcher(None, v3.normalize(source.get("description", "")),
                                       v3.normalize(generated.get("description", "")), autojunk=False).ratio()
    checks["source_output_coherent"] = similarity >= 0.12 and description_ratio < 0.93
    if not checks["source_output_coherent"]:
        reasons.append("The candidate is insufficiently grounded in the source or too close to a copy.")
    checks["automated_validation_passed"] = validation.get("validation_status") == "auto_pass"
    if not checks["automated_validation_passed"]:
        reasons.append("Automatic validation did not pass; this simulation cannot override it.")
    # Source ambiguity is informational: False is the normal case, not a failed gate.
    required_checks = {key: value for key, value in checks.items() if key != "source_ambiguity"}
    accepted = all(required_checks.values()) and not checks["source_ambiguity"]
    status = "ACCEPT" if accepted else ("NEEDS_HUMAN_REVIEW" if checks["source_ambiguity"] else "REVISE")
    return {
        "review_version": VERSION, "decision": status, "accepted": accepted, "checks": checks,
        "reasons": reasons, "source_jaccard": round(similarity, 4),
        "description_sequence_ratio": round(description_ratio, 4),
        "mechanism_sentence_count": len(linked), "recruitment_linked_sentence_count": len(recruitment_linked),
        "disclaimer": "Deterministic simulation only; final acceptance requires independent human review.",
    }


def failure_signature(validation, review):
    """Stable failure categories identify repeated structural problems across attempts."""
    reasons = " ".join(validation.get("reasons", []) + review.get("reasons", [])).casefold()
    categories = []
    for name, pattern in [
        ("warning_append", r"warning-style|appended"),
        ("integration", r"integrat|recruitment workflow"),
        ("unsupported_fact", r"unsupported|distinctive source identifier"),
        ("employment", r"employment arrangement|working-hours"),
        ("compensation", r"compensation|monetary"),
        ("artifact", r"artifact|meta-language|conversational"),
        ("safety", r"unsafe operational"),
        ("similarity", r"lexical|copy|grounded"),
        ("fraud_intent", r"mechanism lacks|not clearly identifiable"),
    ]:
        if re.search(pattern, reasons):
            categories.append(name)
    return tuple(categories or ["other"])
