import unittest

from src.generation import CORE_FIELDS
from src.validation_v3_3 import employment_arrangements, unsupported_facts_v3_3


class V33EmploymentTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"identifiers": []}
        self.base = dict.fromkeys(CORE_FIELDS, "")
        self.base.update(title="Customer Service Role", description="Support customers through applications.")

    def reasons(self, source_description, generated_description):
        source = {**self.base, "description": source_description}
        generated = {**self.base, "description": generated_description}
        return unsupported_facts_v3_3(source, generated, self.policy)

    def test_traineeship_must_not_become_full_time(self):
        reasons = self.reasons("Complete a traineeship.", "Complete a full-time traineeship.")
        self.assertIn("Unsupported employment arrangement: full-time", reasons)

    def test_internship_must_not_become_permanent(self):
        reasons = self.reasons("This is an internship.", "This is a permanent internship.")
        self.assertIn("Unsupported employment arrangement: permanent", reasons)

    def test_part_time_must_not_become_full_time(self):
        reasons = self.reasons("This is a part-time role.", "This is a full-time role.")
        self.assertIn("Unsupported employment arrangement: full-time", reasons)
        self.assertIn("Source employment arrangement omitted or changed: part-time", reasons)

    def test_temporary_must_not_become_permanent(self):
        reasons = self.reasons("This is a temporary role.", "This is a permanent role.")
        self.assertIn("Unsupported employment arrangement: permanent", reasons)

    def test_contract_must_not_become_permanent(self):
        reasons = self.reasons("This is a contract role.", "This is a permanent role.")
        self.assertIn("Unsupported employment arrangement: permanent", reasons)
        self.assertIn("Source employment arrangement omitted or changed: contract", reasons)

    def test_unspecified_employment_type_must_not_be_invented(self):
        reasons = self.reasons("Support customers through applications.", "This is a full-time customer support role.")
        self.assertIn("Unsupported employment arrangement: full-time", reasons)

    def test_source_supported_employment_type_is_accepted(self):
        source = {**self.base, "description": "This is a part-time contract role."}
        generated = {**self.base, "description": "This is a part-time contract role with customer support duties."}
        reasons = unsupported_facts_v3_3(source, generated, self.policy)
        self.assertFalse(any("employment arrangement" in reason for reason in reasons))
        self.assertEqual(employment_arrangements(source), {"part-time", "contract"})


if __name__ == "__main__":
    unittest.main()
