import unittest

from src.generation import CORE_FIELDS
from src.validation_v3_2 import contextual_integration, validate_v3_2


class V32IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.source = dict.fromkeys(CORE_FIELDS, "")
        self.source.update(title="Customer Service Traineeship", description="Complete a customer service traineeship.")
        self.policy = {
            "role": ["customer service", "traineeship"], "responsibilities": ["traineeship"],
            "qualifications": [], "location": [], "identifiers": [], "fraud_mechanism": "sensitive_information",
        }

    def test_one_coherent_recruitment_sentence_is_accepted(self):
        candidate = {**self.source, "description": (
            "Complete a customer service traineeship. To progress through pre-screening before interview, "
            "applicants must submit identity documents and bank details for eligibility verification."
        )}
        result = contextual_integration(candidate, "sensitive_information")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["linked_sentence_count"], 1)

    def test_disconnected_final_warning_is_rejected(self):
        candidate = {**self.source, "description": (
            "Complete a customer service traineeship. Please note: provide identity documents and bank details."
        )}
        self.assertFalse(contextual_integration(candidate, "sensitive_information")["accepted"])

    def test_excessive_mechanism_repetition_is_rejected(self):
        candidate = {**self.source, "description": (
            "During the application submit identity and bank details. During pre-screening submit identity and bank details. "
            "Before interview submit identity and bank details. During onboarding submit identity and bank details."
        )}
        self.assertFalse(contextual_integration(candidate, "sensitive_information")["accepted"])

    def test_existing_v31_checks_remain_active(self):
        candidate = {**self.source, "description": (
            "Complete a customer service traineeship. To progress through pre-screening before interview, "
            "applicants must submit identity documents and bank details for eligibility verification."
        ), "benefits": "Health insurance."}
        self.assertFalse(validate_v3_2(self.source, candidate, self.policy)["automatic_passed"])


if __name__ == "__main__":
    unittest.main()
