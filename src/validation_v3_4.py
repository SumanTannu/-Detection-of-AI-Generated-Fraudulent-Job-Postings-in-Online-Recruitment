"""v3.4 validator: semantic fact matching and field-aware fraud integration."""

import re
from difflib import SequenceMatcher

from src import validation_v3 as v3
from src.generation import CORE_FIELDS
from src.validation import combine_fields, jaccard_similarity, validate_structure
from src.validation_v3_3 import _hours, employment_arrangements

VERSION = "v3.4-groq-pilot"

RECRUITMENT_CONTEXT = re.compile(
    r"\b(?:to be considered|application|applicant|apply|candidate|recruitment|pre[- ]?screening|"
    r"screening|interview|selection|enrollment|onboarding|open day|hiring|proceed)\b"
)
APPENDED_MARKER = re.compile(r"^(?:please note|important|warning|caution|disclaimer)\b")
META_PATTERNS = [
    r"\bas an ai\b",
    r"\b(?:here is|here's)\s+(?:the\s+)?(?:generated|modified|fraudulent|json|version|output)\b",
    r"\b(?:prompt|instruction)\s+(?:says|requires|requested)\b",
    r"\b(?:generated|synthetic)\s+(?:record|advertisement|dataset|output)\b",
    r"\[(?:redacted|placeholder)[^\]]*\]",
]
RESEARCH_TERMS = ["pilot", "academic", "research", "experiment", "study", "evaluation", "dataset",
                  "benchmark", "synthetic", "generated", "model", "llm", "ai generated", "fraud detection"]
ORDINARY_TEST_CONTEXT = re.compile(r"\b(?:test[- ]driven|testing|test code|test cases|quality assurance)\b")
RESEARCH_CONTEXT = re.compile(
    r"\b(?:test|pilot|academic|research|experiment|study|evaluation|dataset|benchmark|synthetic|"
    r"generated|model|llm|ai generated|fraud detection)\b.{0,50}\b(?:recruitment|job|advertisement|"
    r"dataset|study|experiment|project|classification|detection)\b|\b(?:recruitment|job|advertisement|"
    r"dataset|study|experiment|project|classification|detection)\b.{0,50}\b(?:test|pilot|academic|"
    r"research|experiment|study|evaluation|dataset|benchmark|synthetic|generated|model|llm|ai generated|"
    r"fraud detection)\b"
)
MONEY_PATTERN = re.compile(r"(?:(?:GBP)|[\u00a3$\u20ac])\s*(\d[\d,]*(?:\.\d+)?)([kK])?", re.I)
EXPLICIT_RANGE = re.compile(
    r"(?:(?:GBP)|[\u00a3$\u20ac])\s*\d[\d,]*(?:\.\d+)?[kK]?\s*(?:-|to|\u2013|\u2014)\s*"
    r"(?:(?:GBP)|[\u00a3$\u20ac])\s*\d[\d,]*(?:\.\d+)?[kK]?", re.I
)


def _normalized(text):
    return v3.normalize(text or "")


def _experience_constraints(record):
    """Extract source/generation experience lower bounds and ranges."""
    text = _normalized(" ".join(record.get(field, "") for field in ["description", "requirements"]))
    constraints = []
    for match in re.finditer(r"\b(\d+)\s*(?:-|\u2013|\u2014|to)\s*(\d+)\s*years?\b", text):
        constraints.append(("range", int(match.group(1)), int(match.group(2))))
    for match in re.finditer(r"\b(\d+)\s*\+\s*years?\b", text):
        constraints.append(("lower", int(match.group(1))))
    for match in re.finditer(r"\b(?:at least|minimum(?: of)?|more than|over)\s+(\d+)\s*years?\b", text):
        constraints.append(("lower", int(match.group(1))))
    for match in re.finditer(r"\b(\d+)\s*years?\s*(?:or more|minimum)\b", text):
        constraints.append(("lower", int(match.group(1))))
    for match in re.finditer(r"\b(\d+)\s*years?(?:\s+of)?(?:\s+[a-z-]+){0,3}\s+experience\b", text):
        constraints.append(("lower", int(match.group(1))))
    return sorted(set(constraints))


def _experience_supported(source_constraint, generated_constraint):
    if source_constraint == generated_constraint:
        return True
    return source_constraint[0] == generated_constraint[0] == "lower" and source_constraint[1] == generated_constraint[1]


def _money_values(text):
    values = []
    for match in MONEY_PATTERN.finditer(text or ""):
        value = float(match.group(1).replace(",", ""))
        if match.group(2):
            value *= 1000
        values.append(int(value) if value.is_integer() else value)
    return values


def _compensation_analysis(source, generated):
    source_text = combine_fields(source)
    generated_text = combine_fields(generated)
    source_values = _money_values(source_text)
    generated_values = _money_values(generated_text)
    source_set, generated_set = set(source_values), set(generated_values)
    source_has_range = bool(EXPLICIT_RANGE.search(source_text))
    generated_has_range = bool(EXPLICIT_RANGE.search(generated_text))
    ambiguous_source = len(source_set) > 1 and not source_has_range
    reasons = []
    if generated_set - source_set:
        reasons.append("New or changed monetary amount after canonical normalization.")
    if source_set - generated_set:
        reasons.append("Source compensation amount omitted or changed after canonical normalization.")
    # A source containing several compensation figures without an explicit range is
    # ambiguous. The validator preserves that ambiguity for a human reviewer.
    review_required = ambiguous_source
    if ambiguous_source and generated_has_range:
        reasons.append("Ambiguous source compensation was converted into a range; human review required.")
    return {
        "source_values": sorted(source_set), "generated_values": sorted(generated_set),
        "source_has_explicit_range": source_has_range, "generated_has_explicit_range": generated_has_range,
        "ambiguous_source": ambiguous_source, "review_required": review_required, "reasons": reasons,
    }


def _artifact_findings(source, generated):
    source_text = _normalized(combine_fields(source))
    output = _normalized(combine_fields(generated))
    findings, exemptions = [], []
    for pattern in META_PATTERNS:
        for match in re.finditer(pattern, output):
            findings.append(f"AI-generation meta-commentary: {match.group()}")
    for term in RESEARCH_TERMS:
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"[ -]+") + r"\b"
        for match in re.finditer(pattern, output):
            context = output[max(0, match.start() - 70):match.end() + 70]
            if term == "model" and re.search(r"\b(?:business|operating|delivery) model\b", context):
                exemptions.append({"term": term, "context": context, "basis": "ordinary job language"})
                continue
            if term == "test" and ORDINARY_TEST_CONTEXT.search(context):
                exemptions.append({"term": term, "context": context, "basis": "ordinary software/testing language"})
                continue
            source_match = re.search(pattern, source_text)
            if source_match and jaccard_similarity(context, source_text[max(0, source_match.start()-70):source_match.end()+70]) >= 0.25:
                exemptions.append({"term": term, "context": context, "basis": "source-supported local context"})
                continue
            if RESEARCH_CONTEXT.search(context) or term != "test":
                findings.append(f"Research/meta artifact: {term}")
    return findings, exemptions


def _field_sentences(generated):
    sentences = []
    for field in CORE_FIELDS:
        text = _normalized(generated.get(field, ""))
        for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
            if sentence.strip():
                sentences.append((field, sentence.strip()))
    return sentences


def _integration(generated, mechanism):
    patterns = list(v3.SIGNALS[mechanism])
    if mechanism == "implausible_recruitment":
        patterns[0] = r"guarantee(?:d)?|automatic.{0,20}(?:offer|selection)|(?:offer|selected).{0,20}without"
    linked = [(field, sentence) for field, sentence in _field_sentences(generated)
              if any(re.search(pattern, sentence) for pattern in patterns)]
    recruitment_linked = [(field, sentence) for field, sentence in linked if RECRUITMENT_CONTEXT.search(sentence)]
    benefits_linked = [(field, sentence) for field, sentence in linked if field == "benefits"]
    appended = [(field, sentence) for field, sentence in linked if APPENDED_MARKER.search(sentence)]
    excessive = len(linked) > 3
    accepted = bool(recruitment_linked) and not benefits_linked and not appended and not excessive
    return {
        "accepted": accepted, "linked_sentence_count": len(linked),
        "recruitment_linked_sentence_count": len(recruitment_linked),
        "benefits_field_signal_count": len(benefits_linked),
        "appended_signal_count": len(appended), "excessive_repetition": excessive,
    }


def _semantic_qualifications(text, anchor):
    if anchor == "experience":
        return bool(re.search(r"\b(?:experience|track record|background|proven sales)\b", text))
    return bool(re.search(anchor, text))


def _unsupported_facts(source, generated, policy):
    raw = v3.unsupported_facts(source, generated, policy)
    facts = [reason for reason in raw if not (
        reason.startswith("Unsupported experience duration:")
        or reason.startswith("New or changed monetary amount")
        or reason.startswith("Source compensation omitted")
        or reason.startswith("Unsupported employment arrangement:")
    )]
    source_experience = _experience_constraints(source)
    generated_experience = _experience_constraints(generated)
    for constraint in generated_experience:
        if not any(_experience_supported(source_constraint, constraint) for source_constraint in source_experience):
            facts.append(f"Unsupported experience requirement: {constraint}")
    for constraint in source_experience:
        if not any(_experience_supported(constraint, generated_constraint) for generated_constraint in generated_experience):
            facts.append(f"Source experience requirement omitted or changed: {constraint}")
    compensation = _compensation_analysis(source, generated)
    facts.extend(reason for reason in compensation["reasons"] if "human review required" not in reason)
    source_arrangements, generated_arrangements = employment_arrangements(source), employment_arrangements(generated)
    for arrangement in sorted(generated_arrangements - source_arrangements):
        facts.append(f"Unsupported employment arrangement: {arrangement}")
    for arrangement in sorted(source_arrangements - generated_arrangements):
        facts.append(f"Source employment arrangement omitted or changed: {arrangement}")
    source_hours, generated_hours = _hours(source), _hours(generated)
    for hours in sorted(generated_hours - source_hours):
        facts.append(f"Unsupported working-hours claim: {hours}")
    for hours in sorted(source_hours - generated_hours):
        facts.append(f"Source working-hours claim omitted or changed: {hours}")
    return facts, compensation, source_experience, generated_experience


def validate_v3_4(source, generated, policy):
    if not isinstance(generated, dict):
        return {"validation_version": VERSION, "structural_valid": False, "automatic_passed": False,
                "validation_status": "auto_reject", "reasons": ["No parseable generated record available."],
                "human_review_status": "pending_human_review"}
    structure = validate_structure(generated)
    if structure["valid"] and (not generated["title"].strip() or not generated["description"].strip()):
        structure = {"valid": False, "reasons": ["Title and description must both be non-empty."]}
    if not structure["valid"]:
        return {"validation_version": VERSION, "structural_valid": False, "automatic_passed": False,
                "validation_status": "auto_reject", "reasons": structure["reasons"],
                "human_review_status": "pending_human_review"}

    reasons, checks = [], {"structural_valid": True}
    output = _normalized(combine_fields(generated))
    title = _normalized(generated["title"])
    for category in ["role", "responsibilities", "qualifications", "location"]:
        text = title if category == "role" else output
        missing = [anchor for anchor in policy[category] if not _semantic_qualifications(text, anchor)]
        checks[f"{category}_preserved"] = not missing
        reasons.extend(f"Missing source {category} evidence: {anchor}" for anchor in missing)

    unsupported, compensation, source_experience, generated_experience = _unsupported_facts(source, generated, policy)
    checks["unsupported_facts_clear"] = not unsupported
    reasons.extend(unsupported)
    artifacts, exemptions = _artifact_findings(source, generated)
    checks["research_artifacts_clear"] = not any(reason.startswith("Research/meta") for reason in artifacts)
    checks["ai_artifacts_clear"] = not any(reason.startswith("AI-generation") for reason in artifacts)
    reasons.extend(artifacts)
    unsafe = re.search(r"https?://|www\.|\b\S+@\S+\.\S+|\b(?:gsk_|sk-)[\w-]+|\b(?:\d[ -]?){13,19}\b", output)
    checks["sensitive_values_clear"] = unsafe is None
    if unsafe:
        reasons.append("Contact endpoint, credential or account-like value found.")

    mechanism = policy["fraud_mechanism"]
    signal_patterns = list(v3.SIGNALS[mechanism])
    if mechanism == "implausible_recruitment":
        signal_patterns[0] = r"guarantee(?:d)?|automatic.{0,20}(?:offer|selection)|(?:offer|selected).{0,20}without"
    positive = [sentence for _, sentence in _field_sentences(generated)
                if not re.search(r"do not (?:pay|provide|submit)|never (?:pay|provide|submit)|no fees|no charge|not required to", sentence)]
    intent_text = " ".join(positive)
    evidence = [re.search(pattern, intent_text) for pattern in signal_patterns]
    obligation = re.search(r"must|required|asked|condition|only|cannot|will need|need to|contingent|agree|submit|provide|confirm|guarantee", intent_text)
    checks["fraud_intent_present"] = all(evidence) and bool(obligation)
    if not checks["fraud_intent_present"]:
        reasons.append("Assigned mechanism lacks two contextual indicators and an applicant obligation.")
    if mechanism != "payment_related" and re.search(r"\bfee\b|\bdeposit\b|processing charge", intent_text):
        checks["fraud_intent_present"] = False
        reasons.append("Unassigned fee mechanism introduced; violates diversity protocol.")
    integration = _integration(generated, mechanism)
    checks["fraud_integrated"] = integration["accepted"]
    if not integration["accepted"]:
        if integration["benefits_field_signal_count"]:
            reasons.append("Fraud mechanism is placed in benefits rather than a recruitment/application field.")
        elif integration["appended_signal_count"]:
            reasons.append("Fraud signal appears as a warning-style appended statement.")
        elif integration["excessive_repetition"]:
            reasons.append("Fraud mechanism is repeated excessively rather than integrated naturally.")
        else:
            reasons.append("Fraud signal is not connected to a recruitment or application step.")
    similarity = jaccard_similarity(combine_fields(source), combine_fields(generated))
    description_ratio = SequenceMatcher(None, _normalized(source["description"]), _normalized(generated["description"]), autojunk=False).ratio()
    copied = bool(source["description"].strip()) and _normalized(source["description"]) in _normalized(generated["description"])
    checks["source_similarity_ok"] = similarity >= 0.12 and description_ratio < 0.93 and not copied
    if not checks["source_similarity_ok"]:
        reasons.append("Insufficient source lexical evidence or description too close to a verbatim copy.")

    core_passed = all(checks.values())
    review_required = compensation["review_required"]
    if core_passed and review_required:
        status = "needs_human_review"
        reasons.append("Ambiguous source compensation requires human review.")
    else:
        status = "auto_pass" if core_passed else "auto_reject"
    return {
        "validation_version": VERSION, **checks, "automatic_passed": core_passed and not review_required,
        "validation_status": status, "reasons": reasons, "artifact_source_exemptions": exemptions,
        "source_jaccard": round(similarity, 4), "description_sequence_ratio": round(description_ratio, 4),
        "fraud_evidence": [match.group() for match in evidence if match],
        "contextual_integration": integration, "compensation_analysis": compensation,
        "source_experience_constraints": source_experience,
        "generated_experience_constraints": generated_experience,
        "employment_arrangement_comparison": {
            "source": sorted(employment_arrangements(source)), "generated": sorted(employment_arrangements(generated)),
            "new": sorted(employment_arrangements(generated) - employment_arrangements(source)),
            "missing": sorted(employment_arrangements(source) - employment_arrangements(generated)),
        },
        "human_review_status": "pending_human_review", "semantic_fidelity_verified_by_human": False,
        "limitations": "Structured matching improves normalization but does not replace independent human semantic review."
    }
