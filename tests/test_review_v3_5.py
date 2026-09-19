import unittest

from src.generation import CORE_FIELDS
from src.review_v3_5 import failure_signature, review_candidate
from src.validation_v3_4 import _compensation_analysis, validate_v3_4


class V35ReviewTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"identifiers": [], "role": ["customer"], "responsibilities": ["support"],
                       "qualifications": [], "location": [], "fraud_mechanism": "sensitive_information"}
        self.source = dict.fromkeys(CORE_FIELDS, "")
        self.source.update(title="Customer Support", description="Provide customer support during applications.",
                           requirements="Two years of experience.")

    def candidate(self, description):
        return {**self.source, "description": description}

    def test_appended_fraud_sentence_is_rejected(self):
        output = self.candidate("Provide customer support. Please note candidates must provide identity and bank details.")
        result = review_candidate(self.source, output, self.policy, {"validation_status": "auto_reject", "unsupported_facts_clear": True, "compensation_analysis": {}, "reasons": []})
        self.assertIn("warning-style", " ".join(result["reasons"]))

    def test_warning_style_fraud_insertion_is_rejected(self):
        output = self.candidate("Warning: candidates must provide identity and bank details during application.")
        result = review_candidate(self.source, output, self.policy, {"validation_status": "auto_reject", "unsupported_facts_clear": True, "compensation_analysis": {}, "reasons": []})
        self.assertFalse(result["checks"]["no_warning_style_append"])

    def test_employment_alteration_is_rejected(self):
        output = self.candidate("Full-time customer support; candidates must provide identity and bank details during application.")
        validation = validate_v3_4(self.source, output, self.policy)
        self.assertFalse(validation["unsupported_facts_clear"])

    def test_unsupported_experience_is_rejected(self):
        output = self.candidate("Five years of customer support experience; candidates must provide identity and bank details during application.")
        validation = validate_v3_4(self.source, output, self.policy)
        self.assertFalse(validation["unsupported_facts_clear"])

    def test_compensation_reinterpretation_and_range_are_detected(self):
        source = {**self.source, "benefits": "GBP35k OTE. GBP40,000 achievable in year one."}
        output = {**source, "benefits": "GBP35,000-GBP40,000 OTE."}
        analysis = _compensation_analysis(source, output)
        self.assertTrue(analysis["ambiguous_source"])
        self.assertTrue(analysis["generated_has_explicit_range"])

    def test_source_ambiguity_requires_review(self):
        source = {**self.source, "benefits": "GBP35k OTE. GBP40,000 achievable in year one."}
        output = {**source, "description": "During application, candidates must provide identity and bank details."}
        validation = validate_v3_4(source, output, self.policy)
        result = review_candidate(source, output, self.policy, validation)
        self.assertEqual(result["decision"], "NEEDS_HUMAN_REVIEW")

    def test_research_and_ai_artifacts_are_rejected(self):
        output = self.candidate("Here is a research pilot job. Candidates must provide identity and bank details during application.")
        result = review_candidate(self.source, output, self.policy, {"validation_status": "auto_reject", "unsupported_facts_clear": True, "compensation_analysis": {}, "reasons": []})
        self.assertFalse(result["checks"]["no_research_meta"])
        self.assertFalse(result["checks"]["no_ai_artifact"])

    def test_legitimate_technical_test_phrase_is_not_research_language(self):
        output = self.candidate("Write test-driven support tooling. During application candidates must provide identity and bank details.")
        result = review_candidate(self.source, output, self.policy, {"validation_status": "auto_reject", "unsupported_facts_clear": True, "compensation_analysis": {}, "reasons": []})
        self.assertTrue(result["checks"]["no_research_meta"])

    def test_contextual_integration_is_recognized(self):
        output = self.candidate("During application screening, candidates must provide identity and bank details before their application can proceed.")
        validation = validate_v3_4(self.source, output, self.policy)
        result = review_candidate(self.source, output, self.policy, validation)
        self.assertTrue(result["checks"]["fraud_workflow_integrated"])

    def test_repeated_failure_signature_is_stable(self):
        validation = {"reasons": ["Fraud signal appears as a warning-style appended statement."]}
        review = {"reasons": ["The fraud signal is presented as a warning-style appendage."]}
        self.assertEqual(failure_signature(validation, review), failure_signature(validation, review))


if __name__ == "__main__":
    unittest.main()
