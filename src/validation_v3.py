"""Frozen v3 screening rules. Heuristics do not establish semantic ground truth."""

import re
import unicodedata
from difflib import SequenceMatcher

from src.generation import CORE_FIELDS
from src.validation import combine_fields, jaccard_similarity, validate_structure

VERSION = "v3"
ARTIFACTS = ["pilot", "academic", "research", "experiment", "study", "evaluation",
             "test", "fraud detection", "controlled", "dataset", "benchmark",
             "synthetic", "generated", "model", "llm", "ai generated"]
NUMBER_WORDS = dict(zip("one two three four five six seven eight nine ten".split(), map(str, range(1, 11))))
SIGNALS = {
    "sensitive_information": [r"identity|identification|passport|government.{0,12}id",
                              r"bank|financial.{0,15}(?:details|information)|account details"],
    "unusual_financial_arrangements": [r"personal.{0,25}(?:account|bank)",
                                        r"company.{0,30}(?:funds|money)|(?:receive|transfer|process).{0,40}(?:funds|payments)"],
    "suspicious_application": [r"private.{0,25}(?:channel|chat|messag)|off.platform|personal messag",
                               r"(?:bypass|outside|avoid|without|not use|rather than).{0,70}(?:official|verification|portal|hr|recruitment)"],
    "urgency_pressure": [r"within.{0,15}(?:hours|day)|deadline|immediately",
                         r"before.{0,45}(?:written|terms|screening|interview)|without.{0,35}(?:written|terms|screening|interview)"],
    "payment_related": [r"processing (?:fee|charge)|application (?:fee|charge)|recruitment (?:fee|charge)",
                        r"before.{0,55}(?:interview|screening)|(?:until|unless).{0,40}(?:paid|payment)|(?:paid|payment).{0,60}(?:before|interview)"],
    "implausible_recruitment": [r"guaranteed|automatic.{0,20}(?:offer|selection)|(?:offer|selected).{0,20}without",
                                r"(?:without|before|no|skip|waiv).{0,45}(?:interview|assessment|verification|check)"],
}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(r"\b" + word + r"\b", digit, text)
    return re.sub(r"\s+", " ", text).strip()


def artifact_findings(source, generated):
    """Source vocabulary alone is not an exemption: local job context must overlap."""
    findings, exemptions = [], []
    original = normalize(combine_fields(source))
    output = normalize(combine_fields(generated))
    for term in ARTIFACTS:
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"[ -]+") + r"\b"
        src = list(re.finditer(pattern, original))
        for match in re.finditer(pattern, output):
            context = output[max(0, match.start()-55):match.end()+55]
            allowed = any(jaccard_similarity(context, original[max(0, x.start()-55):x.end()+55]) >= 0.4 for x in src)
            if allowed:
                exemptions.append({"term": term, "context": context, "basis": "source term and local lexical context"})
            else:
                findings.append(f"Unjustified origin/artifact term: {term}")
    return findings, exemptions


def money_values(text):
    return {normalize(m.group()) for m in re.finditer(r"[\u00a3$\u20ac]\s*\d[\d,]*(?:\.\d+)?(?:[kK])?", text)}


def unsupported_facts(source, generated, policy):
    findings = []
    original, output = normalize(combine_fields(source)), normalize(combine_fields(generated))
    for field in CORE_FIELDS:
        if not source[field].strip() and generated[field].strip():
            findings.append(f"Invented content in empty source field: {field}")
    for pattern in [r"\b\d+(?:\s*[-+]\s*\d*)?\s*years?\b"]:
        existing = set(re.findall(pattern, original))
        for value in set(re.findall(pattern, output)) - existing:
            findings.append(f"Unsupported experience duration: {value}")
    if money_values(combine_fields(generated)) - money_values(combine_fields(source)):
        findings.append("New or changed monetary amount; preserve source notation for audit.")
    for value in money_values(source["benefits"]):
        if value not in money_values(generated["benefits"]):
            findings.append(f"Source compensation omitted or changed: {value}")
    employment_source = normalize(source["description"] + " " + source["requirements"])
    employment_output = normalize(generated["title"] + " " + generated["description"] + " " + generated["requirements"])
    for term in ["part-time", "full-time", "temporary", "permanent", "contract", "hybrid", "work from home"]:
        pattern = r"\b" + term.replace("-", "[ -]") + r"\b"
        if re.search(pattern, employment_output) and not re.search(pattern, employment_source):
            findings.append(f"Unsupported employment arrangement: {term}")
    if "remote" in normalize(generated["title"]) and "remote" not in normalize(source["title"]):
        findings.append("New remote-work claim in title; remote infrastructure is not remote employment.")
    for concept in [r"health insurance", r"paid time off", r"pension", r"company car", r"flexible working", r"life insurance"]:
        if re.search(concept, normalize(generated["benefits"])) and not re.search(concept, original):
            # Common semantic variants are handled explicitly, never inferred from model claims.
            equivalents = {"paid time off": "pto|holiday", "health insurance": "medical|health care", "pension": "pension|401"}
            if not re.search(equivalents.get(concept, concept), original):
                findings.append(f"Unsupported benefit: {concept}")
    for name in policy["identifiers"]:
        if re.search(r"\b" + re.escape(normalize(name)) + r"\b", output):
            findings.append(f"Distinctive source identifier retained: {name}")
    return findings


def validate_v3(source, generated, policy):
    reasons = []
    structure = validate_structure(generated)
    if structure["valid"] and (not generated["title"].strip() or not generated["description"].strip()):
        structure = {"valid": False, "reasons": ["Title and description must both be non-empty."]}
    if not structure["valid"]:
        return {"validation_version": VERSION, "structural_valid": False, "automatic_passed": False,
                "validation_status": "auto_reject", "reasons": structure["reasons"],
                "human_review_status": "not_reviewed"}
    output = normalize(combine_fields(generated))
    title = normalize(generated["title"])
    checks = {"structural_valid": True}
    for category in ["role", "responsibilities", "qualifications", "location"]:
        text = title if category == "role" else output
        missing = [anchor for anchor in policy[category] if not re.search(anchor, text)]
        checks[category + "_preserved"] = not missing
        reasons.extend(f"Missing source {category} evidence: {anchor}" for anchor in missing)
    unsupported = unsupported_facts(source, generated, policy)
    checks["unsupported_facts_clear"] = not unsupported
    reasons.extend(unsupported)
    artifacts, exemptions = artifact_findings(source, generated)
    checks["research_artifacts_clear"] = not artifacts
    reasons.extend(artifacts)
    meta = re.search(r"please note|as an ai|here is|prompt|modified version|creating a pressure point|(?:provided|following) link|\[redacted", output)
    checks["ai_artifacts_clear"] = meta is None
    if meta:
        reasons.append(f"Meta-commentary, appended disclaimer or unresolved placeholder: {meta.group()}")
    unsafe = re.search(r"https?://|www\.|\b\S+@\S+\.\S+|\b(?:gsk_|sk-)[\w-]+|\b(?:\d[ -]?){13,19}\b", output)
    checks["sensitive_values_clear"] = unsafe is None
    if unsafe:
        reasons.append("Contact endpoint, credential or account-like value found.")

    mechanism = policy["fraud_mechanism"]
    sentences = [s for s in re.split(r"(?<=[.!?;])\s+|\n+", normalize(combine_fields(generated))) if s]
    positive = [s for s in sentences if not re.search(r"do not (?:pay|provide|submit)|never (?:pay|provide|submit)|no fees|no charge|not required to", s)]
    intent_text = " ".join(positive)
    evidence = [re.search(pattern, intent_text) for pattern in SIGNALS[mechanism]]
    obligation = re.search(r"must|required|asked|condition|only|cannot|will need|need to|contingent|agree|submit|provide", intent_text)
    checks["fraud_intent_present"] = all(evidence) and bool(obligation)
    if not checks["fraud_intent_present"]:
        reasons.append("Assigned mechanism lacks two contextual indicators and an applicant obligation.")
    if mechanism != "payment_related" and re.search(r"\bfee\b|\bdeposit\b|processing charge", intent_text):
        checks["fraud_intent_present"] = False
        reasons.append("Unassigned fee mechanism introduced; violates diversity protocol.")
    linked_sentences = [s for s in positive if any(re.search(pattern, s) for pattern in SIGNALS[mechanism])]
    checks["fraud_integrated"] = len(linked_sentences) >= 2 and bool(re.search(r"interview|application|applicant|onboarding|candidate|selection", intent_text))
    if not checks["fraud_integrated"]:
        reasons.append("Mechanism is confined to one sentence or lacks recruitment integration.")
    similarity = jaccard_similarity(combine_fields(source), combine_fields(generated))
    description_ratio = SequenceMatcher(None, normalize(source["description"]), normalize(generated["description"]), autojunk=False).ratio()
    copied = bool(source["description"].strip()) and normalize(source["description"]) in normalize(generated["description"])
    checks["source_similarity_ok"] = similarity >= 0.12 and description_ratio < 0.93 and not copied
    if not checks["source_similarity_ok"]:
        reasons.append("Insufficient source lexical evidence or description too close to a verbatim copy.")
    passed = all(checks.values())
    return {"validation_version": VERSION, **checks, "automatic_passed": passed,
            "validation_status": "auto_pass" if passed else "auto_reject",
            "reasons": reasons, "artifact_source_exemptions": exemptions,
            "source_jaccard": round(similarity, 4), "description_sequence_ratio": round(description_ratio, 4),
            "fraud_evidence": [m.group() for m in evidence if m],
            "fraud_sentence_count": len(linked_sentences), "human_review_status": "not_reviewed",
            "semantic_fidelity_verified_by_human": False,
            "limitations": "Source anchors and fact patterns are screening evidence, not exhaustive semantic verification."}
