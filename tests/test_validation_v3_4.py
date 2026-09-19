import unittest

from src.generation import CORE_FIELDS
from src.validation_v3_4 import _artifact_findings, _compensation_analysis, _integration, _unsupported_facts


class V34ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "identifiers": [], "role": ["customer"], "responsibilities": ["support"],
            "qualifications": ["experience"], "location": [], "fraud_mechanism": "sensitive_information",
        }
        self.base = dict.fromkeys(CORE_FIELDS, "")
        self.base.update(title="Customer Support", description="Provide customer support and application assistance.",
                         requirements="Relevant experience required.")

    def records(self, source_description, generated_description):
        return ({**self.base, "description": source_description}, {**self.base, "description": generated_description})

    def test_two_plus_years_equals_minimum_two_years(self):
        source, generated = self.records("2+ years of support experience.", "Minimum 2 years of support experience.")
        facts, *_ = _unsupported_facts(source, generated, self.policy)
        self.assertFalse(any("experience" in item for item in facts))

    def test_ten_plus_years_equals_at_least_ten_years(self):
        source, generated = self.records("10+ years as a Windows administrator.", "At least 10 years of Windows administration experience.")
        facts, *_ = _unsupported_facts(source, generated, self.policy)
        self.assertFalse(any("experience" in item for item in facts))

    def test_gbp_35k_equals_gbp_35000(self):
        source, generated = self.records("Salary GBP35k OTE.", "Salary GBP35,000 OTE.")
        analysis = _compensation_analysis(source, generated)
        self.assertEqual(analysis["reasons"], [])

    def test_conflicting_compensation_requires_review(self):
        source, generated = self.records("GBP35k OTE. GBP40,000 in year one.", "GBP35k OTE. GBP40,000 in year one.")
        analysis = _compensation_analysis(source, generated)
        self.assertTrue(analysis["review_required"])

    def test_test_driven_is_not_a_research_artifact(self):
        source, generated = self.records("Write test-driven code.", "Write test-driven code and support customers.")
        findings, _ = _artifact_findings(source, generated)
        self.assertFalse(findings)

    def test_there_is_is_not_ai_artifact(self):
        source, generated = self.records("There is no private healthcare.", "There is no private healthcare; support customers.")
        findings, _ = _artifact_findings(source, generated)
        self.assertFalse(findings)

    def test_genuine_research_language_is_detected(self):
        source, generated = self.records("Support customers.", "This pilot is part of a research recruitment study.")
        findings, _ = _artifact_findings(source, generated)
        self.assertTrue(any("Research/meta" in item for item in findings))

    def test_one_sentence_recruitment_fraud_integration_passes(self):
        generated = {**self.base, "description": "To be considered, candidates must provide identity and bank details before screening."}
        result = _integration(generated, "sensitive_information")
        self.assertTrue(result["accepted"])

    def test_appended_warning_fails_integration(self):
        generated = {**self.base, "description": "Support customers. Please note candidates must provide identity and bank details."}
        result = _integration(generated, "sensitive_information")
        self.assertFalse(result["accepted"])

    def test_benefits_field_fraud_signal_is_flagged(self):
        generated = {**self.base, "benefits": "Candidates must provide identity and bank details during application."}
        result = _integration(generated, "sensitive_information")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["benefits_field_signal_count"], 1)

    def test_unsupported_experience_still_fails(self):
        source, generated = self.records("Customer support role.", "Five years of support experience required.")
        facts, *_ = _unsupported_facts(source, generated, self.policy)
        self.assertTrue(any("Unsupported experience" in item for item in facts))

    def test_unsupported_employment_type_still_fails(self):
        source, generated = self.records("Customer support role.", "Full-time customer support role.")
        facts, *_ = _unsupported_facts(source, generated, self.policy)
        self.assertIn("Unsupported employment arrangement: full-time", facts)


if __name__ == "__main__":
    unittest.main()
