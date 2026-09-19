import unittest

from src.generation import CORE_FIELDS
from src.validation_v3_1 import unsupported_facts_v3_1, validate_v3_1


class V31ValidationTests(unittest.TestCase):
    def setUp(self):
        self.source = dict.fromkeys(CORE_FIELDS, "")
        self.source.update(
            title="Customer Service Traineeship",
            description="Government funding is only available for 16-18 year olds. Complete a traineeship.",
            requirements="16-18 year olds only due to funding.",
            benefits="Career prospects.",
        )
        self.policy = {
            "role": ["customer service", "traineeship"], "responsibilities": ["traineeship"],
            "qualifications": ["16", "18"], "location": [], "identifiers": [],
            "fraud_mechanism": "sensitive_information",
        }

    def test_matching_source_age_eligibility_is_not_experience(self):
        generated = {**self.source, "requirements": "Applicants must be 16-18 years old due to funding."}
        self.assertFalse(any("experience duration" in reason for reason in
                             unsupported_facts_v3_1(self.source, generated, self.policy)))

    def test_actual_new_experience_remains_rejected(self):
        generated = {**self.source, "requirements": "Applicants need 5 years of customer-service experience."}
        self.assertTrue(any("experience duration" in reason for reason in
                            unsupported_facts_v3_1(self.source, generated, self.policy)))

    def test_full_validation_still_rejects_unsupported_benefit(self):
        generated = {**self.source,
            "description": self.source["description"] + " Provide bank account details before interview and identity documents before selection.",
            "benefits": "Health insurance."}
        self.assertFalse(validate_v3_1(self.source, generated, self.policy)["automatic_passed"])


if __name__ == "__main__":
    unittest.main()
