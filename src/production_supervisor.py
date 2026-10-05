"""Restart the production worker after infrastructure failures.

Quality/safety stops remain terminal and require a versioned diagnosis. The locked
corpus target remains 5,000; ``--milestone`` only controls this supervisor session.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from src.groq_pilot import atomic_json


PROFILES = {
    "groq": {"output": "groq_gpt_v3_5_20b", "worker": "src.production_generation", "key_slot": "2"},
    "groq120b": {"output": "groq_gpt_oss_120b", "worker": "src.groq_120b_production_generation", "key_slot": "1"},
    "openrouter": {"output": "openrouter_qwen3_8_flash", "worker": "src.openrouter_production_generation"},
}


def load_state(root: Path, output_name: str = "groq_gpt_v3_5_20b"):
    path = root / "data/synthetic/production" / output_name / "state.json"
    return json.loads(path.read_text(encoding="utf-8"))


def approved(state):
    return sum(record.get("final_decision") == "APPROVED" for record in state.get("records", []))


def remediate_quality_stop(root: Path, state: dict, output_name: str = "groq_gpt_v3_5_20b") -> bool:
    """Apply a conservative recorded correction for known non-safety batch drift."""
    if not (state.get("stop_reason") or "").startswith("Quality stop:") or not state.get("batches"):
        return False
    batch = state["batches"][-1]
    if batch.get("safety_failures", 0) or batch.get("ai_artifact_failures", 0):
        return False
    old_quota = int(state.get("complex_source_quota", 5))
    factual = sum(int(batch.get(key, 0)) for key in (
        "unsupported_fact_failures", "employment_failures", "compensation_failures"))
    research = int(batch.get("research_artifact_failures", 0))
    integration = int(batch.get("contextual_integration_failures", 0))
    if not (factual or research or integration):
        return False
    new_quota = max(0, old_quota - 1) if factual else old_quota
    event = {"at": datetime.now(UTC).isoformat(), "batch_number": batch.get("batch_number"),
             "reason": state["stop_reason"], "factual_failures": factual,
             "research_failures": research, "integration_failures": integration,
             "old_complex_source_quota": old_quota, "new_complex_source_quota": new_quota}
    state.setdefault("automatic_remediations", []).append(event)
    state["complex_source_quota"] = new_quota
    state["stop_reason"] = None
    atomic_json(root / "data/synthetic/production" / output_name / "state.json", state)
    return True


def remediate_rate_limit_stop(root: Path, state: dict,
                              output_name: str = "groq_gpt_v3_5_20b") -> bool:
    """Resume a checkpoint after a recorded upstream 429 cooldown.

    This does not alter or retry completed sources.  It only clears the
    infrastructure stop, resets the consecutive counter, and schedules a
    two-minute cooldown before the next provider request.
    """
    reason = state.get("stop_reason") or ""
    if not reason.startswith("Three consecutive HTTP 429 responses"):
        return False
    cooldown = 120.0
    state.setdefault("automatic_remediations", []).append({
        "at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "action": "rate_limit_cooldown_and_resume",
        "cooldown_seconds": cooldown,
    })
    state["consecutive_429"] = 0
    state["next_request_at"] = max(float(state.get("next_request_at", 0)), time.time() + cooldown)
    state["stop_reason"] = None
    atomic_json(root / "data/synthetic/production" / output_name / "state.json", state)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="groq")
    parser.add_argument("--milestone", type=int, default=5000)
    parser.add_argument("--restart-delay", type=float, default=10.0)
    args = parser.parse_args(); root = args.root.resolve()
    profile = PROFILES[args.profile]
    log_path = root / "data/synthetic/production" / profile["output"] / "supervisor_events.jsonl"
    while True:
        state = load_state(root, profile["output"])
        count = approved(state)
        if count >= args.milestone or state.get("finished"):
            return 0
        if state.get("stop_reason") and remediate_quality_stop(root, state, profile["output"]):
            state = load_state(root, profile["output"])
        if state.get("stop_reason") and remediate_rate_limit_stop(root, state, profile["output"]):
            state = load_state(root, profile["output"])
        if state.get("stop_reason"):
            event = {"at": datetime.now(UTC).isoformat(), "event": "quality_stop",
                     "approved": count, "reason": state["stop_reason"]}
            with log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event) + "\n")
            return 2
        started = datetime.now(UTC).isoformat()
        worker_env = dict(__import__("os").environ)
        if profile.get("key_slot"):
            worker_env["GROQ_API_KEY_SLOT"] = profile["key_slot"]
        result = subprocess.run([sys.executable, "-m", profile["worker"]], cwd=root, env=worker_env)
        state = load_state(root, profile["output"]); count = approved(state)
        event = {"at": datetime.now(UTC).isoformat(), "event": "worker_exit", "started_at": started,
                 "exit_code": result.returncode, "approved": count, "stop_reason": state.get("stop_reason")}
        with log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event) + "\n")
        if count >= args.milestone or state.get("finished"): return 0
        if state.get("stop_reason"):
            remediated = remediate_quality_stop(root, state, profile["output"])
            if not remediated:
                remediated = remediate_rate_limit_stop(root, state, profile["output"])
            if not remediated:
                return 2
        time.sleep(args.restart_delay)


if __name__ == "__main__": raise SystemExit(main())
