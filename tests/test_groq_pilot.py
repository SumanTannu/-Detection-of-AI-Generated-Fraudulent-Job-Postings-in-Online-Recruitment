"""Offline tests for request limits, resumption, isolation, and schema failures."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.generation import CORE_FIELDS, GenerationError, parse_structured_response
from src.groq_pilot import GroqPilot
from src.validation import validate_pilot_candidate, validate_structure


class FakeProvider:
    model = "openai/gpt-oss-120b"
    name = "groq"
    generation_mode = "api"
    last_response = {}

    def __init__(self, failure=None):
        self.calls = 0
        self.failure = failure

    def available_models(self):
        return [self.model]

    def generate_json(self, prompt, schema):
        self.calls += 1
        if self.failure == "transient":
            error = GenerationError("HTTP 429: offline test fixture")
            error.transient = True
            raise error
        if self.failure == "malformed":
            return {"title": "Only one field"}
        return dict(zip(CORE_FIELDS, ["Engineer", "", "Research fixture for an engineering role.", "", ""]))


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"GENERATION_PROVIDER": "groq", "GENERATION_MODEL": FakeProvider.model})
        self.env.start()
        for folder in ["data/raw", "data/processed", "data/synthetic/pilot", "prompts", "src"]:
            (self.root / folder).mkdir(parents=True)
        project = Path(__file__).resolve().parents[1]
        for name in ["generation.py", "validation.py", "groq_pilot.py"]:
            (self.root / "src" / name).write_bytes((project / "src" / name).read_bytes())
        (self.root / "prompts/ai_fraud_generation_v2.txt").write_bytes(
            (project / "prompts/ai_fraud_generation_v2.txt").read_bytes())
        rows = []
        for index in range(21):
            row = {field: "" for field in CORE_FIELDS}
            row.update(title=f"Engineer {index}", description=f"Design instruments for project {index}.",
                       text=f"Engineer {index} " + "design " * (index + 1),
                       source_row_id=str(index), text_group_id=f"group_{index}",
                       fraudulent="f" if index < 20 else "t")
            rows.append(row)
        data = pd.DataFrame(rows)
        data.to_csv(self.root / "data/processed/emscad_clean.csv", index=False)
        data.to_csv(self.root / "data/raw/emscad.csv", index=False)
        self.mock = self.root / "data/synthetic/pilot/ai_fraud_pilot_raw.jsonl"
        self.mock.write_text('{"generation_mode":"mock"}\n', encoding="utf-8")

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_failed_smoke_never_continues_or_retries(self):
        provider = FakeProvider("transient")
        pilot = GroqPilot(self.root, provider, pause_seconds=0)
        self.assertTrue(pilot.check_models())
        self.assertFalse(pilot.smoke_test())
        pilot.generate_remaining()
        self.assertEqual(provider.calls, 1)
        self.assertEqual(pilot.export()["number_generation_failures"], 1)
        rerun = GroqPilot(self.root, provider, pause_seconds=0)
        self.assertFalse(rerun.check_models())
        self.assertEqual(provider.calls, 1)

    def test_malformed_smoke_stops(self):
        provider = FakeProvider("malformed")
        pilot = GroqPilot(self.root, provider, pause_seconds=0)
        pilot.check_models()
        self.assertFalse(pilot.smoke_test())
        self.assertEqual(provider.calls, 1)

    def test_twenty_limit_resume_provenance_and_preserved_review(self):
        provider = FakeProvider()
        pilot = GroqPilot(self.root, provider, pause_seconds=0)
        original = self.mock.read_bytes()
        pilot.check_models()
        self.assertTrue(pilot.smoke_test())
        pilot.generate_remaining()
        result = pilot.export()
        self.assertEqual(provider.calls, 20)
        self.assertEqual(result["number_selected"], 20)
        self.assertEqual(result["number_successfully_generated"], 20)
        for record in pilot.records:
            self.assertEqual(record["source_label"], 0)
            self.assertEqual(record["generated_label"], 2)
            self.assertIn("validation_version", record)
            self.assertEqual(record["model"], provider.model)
            self.assertEqual(record["validation_status"], "needs_human_review")
        review_path = pilot.output / "human_review_template.csv"
        review = pd.read_csv(review_path, keep_default_na=False)
        review.at[0, "reviewer_notes"] = "Keep this reviewer note."
        review.to_csv(review_path, index=False)
        resumed = GroqPilot(self.root, provider, pause_seconds=0)
        resumed.check_models()
        resumed.smoke_test()
        resumed.generate_remaining()
        resumed.export()
        self.assertEqual(provider.calls, 20)
        self.assertEqual(self.mock.read_bytes(), original)
        self.assertEqual(pd.read_csv(review_path).at[0, "reviewer_notes"], "Keep this reviewer note.")

    def test_strict_structure_and_low_overlap_triage(self):
        valid = {field: "" for field in CORE_FIELDS}
        valid.update(title="Engineer", description="Design instruments.")
        with self.assertRaises(GenerationError):
            parse_structured_response("```json\n{}\n```")
        self.assertFalse(validate_structure({**valid, "extra": "unexpected"})["valid"])
        candidate = {"source": valid, "generated": {**valid, "title": "Developer",
                     "description": "Build reliable applications with colleagues."},
                     "generation_mode": "api", "generation_status": "generated"}
        validate_pilot_candidate(candidate)
        self.assertEqual(candidate["validation_status"], "needs_human_review")
        candidate["generation_mode"] = "mock"
        validate_pilot_candidate(candidate)
        self.assertEqual(candidate["validation_status"], "auto_reject")


if __name__ == "__main__":
    unittest.main()
