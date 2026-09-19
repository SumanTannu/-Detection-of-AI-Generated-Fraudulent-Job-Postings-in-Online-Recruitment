"""Offline v3 regression tests. No provider requests are made."""

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
import tempfile
import unittest

from src.generation import CORE_FIELDS, GenerationError
from src.regeneration_v3 import RegenerationV3, retry_delay
from src.validation_v3 import artifact_findings, unsupported_facts, validate_v3


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.source = dict.fromkeys(CORE_FIELDS, "")
        self.source.update(title="Software Engineer", description="Develop cloud software using test-driven code.",
                           requirements="2 years of JavaScript experience.")
        self.policy = {"role": ["engineer"], "responsibilities": ["software"],
                       "qualifications": ["javascript"], "location": [], "identifiers": [],
                       "fraud_mechanism": "sensitive_information"}

    def test_empty_title_description_and_extra_fields_rejected(self):
        for generated in [{**self.source, "title": ""}, {**self.source, "description": ""},
                          {**self.source, "extra": "text"}]:
            self.assertFalse(validate_v3(self.source, generated, self.policy)["structural_valid"])

    def test_genuine_test_language_allowed_but_research_artifacts_rejected(self):
        issues, allowed = artifact_findings(self.source, self.source)
        self.assertEqual(issues, [])
        self.assertTrue(allowed)
        for title in ["Engineer - Controlled Academic Recruitment-Fraud Detection Pilot",
                      "Engineer (Recruitment Pilot)", "Engineer - dataset benchmark"]:
            issues, _ = artifact_findings(self.source, {**self.source, "title": title})
            self.assertTrue(issues)

    def test_unsupported_facts_and_real_identifier(self):
        candidate = {**self.source, "benefits": "Health insurance.",
                     "title": "Remote Software Engineer", "requirements": "5 years of JavaScript experience."}
        reasons = unsupported_facts(self.source, candidate, self.policy)
        self.assertTrue(any("empty source field" in r for r in reasons))
        self.assertTrue(any("duration" in r for r in reasons))
        self.assertTrue(any("remote" in r for r in reasons))
        self.policy["identifiers"] = ["Example Institution"]
        self.assertTrue(unsupported_facts(self.source, {**self.source, "description": "Example Institution"}, self.policy))

    def test_negated_signals_and_append_only_fail(self):
        candidate = {**self.source, "description": self.source["description"] +
                     " Never provide passport or bank details before interviews."}
        result = validate_v3(self.source, candidate, self.policy)
        self.assertFalse(result["fraud_intent_present"])
        self.assertFalse(result["source_similarity_ok"])

    @unittest.expectedFailure
    def test_known_frozen_v3_age_eligibility_is_not_experience(self):
        # Exposed by the real smoke test; keep executed v3 unchanged for audit.
        source = {**self.source, "requirements": "16-18 year olds only due to government funding."}
        candidate = {**source, "requirements": "Applicants must be 16-18 years old due to government funding."}
        self.assertEqual(unsupported_facts(source, candidate, self.policy), [])

    def test_rate_retry_after_seconds_date_and_backoff(self):
        now = datetime.now(UTC).replace(microsecond=0)
        self.assertEqual(retry_delay({"retry-after": "80"}, 1)[0], 80)
        self.assertEqual(retry_delay({"retry-after": format_datetime(now + timedelta(seconds=120))}, 1, now)[0], 120)
        self.assertEqual(retry_delay({}, 1)[0], 65)
        self.assertEqual(retry_delay({}, 2)[0], 130)


class RunnerTests(unittest.TestCase):
    def runner(self, folder, provider):
        run = object.__new__(RegenerationV3)
        run.output = Path(folder)
        run.provider = provider
        run.wait = lambda seconds: None
        run.state = {"records": [], "stop_reason": None, "next_request_at": 0,
                     "consecutive_429": 0, "smoke_status": "not_run", "finished": False}
        source = dict.fromkeys(CORE_FIELDS, "")
        source.update(title="Engineer", description="Develop software.")
        policy = {"source_row_id": 11662, "role": ["engineer"], "responsibilities": ["software"],
                  "qualifications": [], "location": [], "identifiers": [], "fraud_mechanism": "sensitive_information"}
        run.assignments = [policy]
        run.new_record = lambda p: {"source_row_id": p["source_row_id"], "source": source,
                                  "generated": None, "request_prompt": "Offline test fixture.",
                                  "attempts": [], "generation_status": "not_attempted"}
        run.export = lambda: {}
        return run

    def test_full_validation_failure_stops_remaining_not_just_json_failure(self):
        class Provider:
            calls = 0
            def generate_json(self, prompt, schema):
                self.calls += 1
                return {f: "Engineer Recruitment Pilot" for f in CORE_FIELDS}
        with tempfile.TemporaryDirectory() as folder:
            provider = Provider()
            run = self.runner(folder, provider)
            self.assertFalse(run.smoke_test())
            run.remaining()
            self.assertEqual(provider.calls, 1)
            self.assertEqual(run.state["smoke_status"], "failed")
            self.assertIsNotNone(run.state["stop_reason"])
            run.smoke_test()
            self.assertEqual(provider.calls, 1)

    def test_three_429_stop_and_do_not_regenerate_attempted_record(self):
        class Provider:
            calls = 0
            def generate_json(self, prompt, schema):
                self.calls += 1
                failure = GenerationError("Offline HTTP 429 fixture")
                failure.status_code = 429
                failure.transient = True
                failure.rate_headers = {"retry-after": "90"}
                raise failure
        with tempfile.TemporaryDirectory() as folder:
            provider = Provider()
            run = self.runner(folder, provider)
            run.generate_one(run.assignments[0])
            self.assertEqual(provider.calls, 3)
            self.assertIn("three consecutive", run.state["stop_reason"])
            self.assertEqual(run.state["records"][0]["attempts"][0]["retry_delay_seconds"], 90)
            run.generate_one(run.assignments[0])
            self.assertEqual(provider.calls, 3)


if __name__ == "__main__":
    unittest.main()
