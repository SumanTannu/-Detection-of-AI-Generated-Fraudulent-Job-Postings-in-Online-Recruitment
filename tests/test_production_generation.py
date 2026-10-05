import unittest
import pandas as pd

from src.production_generation import MODEL, TARGET_APPROVED, diversity_order, assign_mechanisms, MECHANISMS
from src.openrouter_production_generation import (
    FINAL_FILENAME as OPENROUTER_FINAL_FILENAME,
    MODEL as OPENROUTER_MODEL,
    OUTPUT_NAME as OPENROUTER_OUTPUT_NAME,
    PROMPT_VERSION as OPENROUTER_PROMPT_VERSION,
)
from src.groq_120b_production_generation import MODEL as GROQ_120B_MODEL, OUTPUT_NAME as GROQ_120B_OUTPUT_NAME


class ProductionPlanningTests(unittest.TestCase):
    def fixture(self, count=60):
        rows = []
        for i in range(count):
            rows.append({"source_row_id": str(i), "text_group_id": f"g{i}", "fraudulent": "f", "text": "apply interview " + "word " * (i + 2),
                         "title": f"Role {i}", "company_profile": "", "description": "Apply for interview", "requirements": "Skill", "benefits": ""})
        clean = pd.DataFrame(rows)
        raw = pd.DataFrame({"source_row_id": [str(i) for i in range(count)], "industry": [f"I{i%5}" for i in range(count)],
            "function": [f"F{i%7}" for i in range(count)], "employment_type": ["Full-time"] * count,
            "location": [f"L{i%3}" for i in range(count)], "required_experience": [f"E{i%4}" for i in range(count)],
            "required_education": [f"D{i%3}" for i in range(count)], "telecommuting": [str(i%2)] * count})
        return clean, raw

    def test_locked_target_and_model(self):
        self.assertEqual(TARGET_APPROVED, 5000); self.assertEqual(MODEL, "openai/gpt-oss-20b")

    def test_openrouter_profile_is_separate_and_uses_qwen(self):
        self.assertEqual(OPENROUTER_MODEL, "qwen/qwen3.8-flash")
        self.assertEqual(OPENROUTER_OUTPUT_NAME, "openrouter_qwen3_8_flash")
        self.assertIn("qwen3.8-flash", OPENROUTER_PROMPT_VERSION)
        self.assertTrue(OPENROUTER_PROMPT_VERSION.endswith("-r4"))

    def test_groq_120b_profile_is_separate(self):
        self.assertEqual(GROQ_120B_MODEL, "openai/gpt-oss-120b")
        self.assertEqual(GROQ_120B_OUTPUT_NAME, "groq_gpt_oss_120b")
        self.assertNotEqual(OPENROUTER_FINAL_FILENAME, "ai_generated_fraud.csv")

    def test_diversity_order_is_reproducible_and_unique(self):
        clean, raw = self.fixture(); one = diversity_order(clean, raw); two = diversity_order(clean, raw)
        self.assertEqual(one.source_row_id.tolist(), two.source_row_id.tolist())
        self.assertTrue(one.text_group_id.is_unique); self.assertEqual(len(one), len(clean))

    def test_mechanisms_are_balanced(self):
        clean, raw = self.fixture(); ordered = diversity_order(clean, raw)
        counts = pd.Series(assign_mechanisms(ordered)).value_counts()
        self.assertEqual(set(counts.index), set(MECHANISMS)); self.assertLessEqual(counts.max() - counts.min(), 2)


if __name__ == "__main__": unittest.main()
