"""OpenRouter/Qwen3.8 Flash entry point for the 5,000-approved-record run.

This profile deliberately uses its own output directory and final filename.  It
does not read, mutate, or merge the existing Groq production checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.production_generation import ProductionRun


MODEL = "qwen/qwen3.8-flash"
PROMPT_VERSION = "v3.5-production-qwen3.8-flash-r4"
OUTPUT_NAME = "openrouter_qwen3_8_flash"
FINAL_FILENAME = "ai_generated_fraud_openrouter_qwen3_8_flash.csv"


class OpenRouterProductionRun(ProductionRun):
    def __init__(self, root: Path, provider=None, wait=None, batch_size: int = 50):
        kwargs = {}
        if wait is not None:
            kwargs["wait"] = wait
        super().__init__(
            root=root,
            provider=provider,
            batch_size=batch_size,
            provider_name="openrouter",
            model=MODEL,
            prompt_version=PROMPT_VERSION,
            output_name=OUTPUT_NAME,
            include_historical_pilots=False,
            final_filename=FINAL_FILENAME,
            **kwargs,
        )
        interrupted = [r for r in self.state["records"]
                       if r.get("generation_status") == "request_in_flight"]
        if interrupted:
            for record in interrupted:
                record["generation_status"] = "failed"
                record["final_decision"] = "REJECTED"
                record.setdefault("api_attempts", []).append({
                    "api_attempt": len(record.get("api_attempts", [])) + 1,
                    "timestamp": None,
                    "http_status": None,
                    "error": "Client process interrupted; outcome unknown and source will not be regenerated.",
                    "transient": False,
                })
            self.state.setdefault("interruption_recovery_events", []).append({
                "records_resolved_as_failed": [r["source_row_id"] for r in interrupted],
                "policy": "conservative_no-regeneration",
            })
            self.state["stop_reason"] = None
            self.checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = OpenRouterProductionRun(args.root, batch_size=args.batch_size)
    if args.freeze:
        print(run.freeze())
    elif run.configure():
        print(json.dumps(run.run(args.max_batches), indent=2))


if __name__ == "__main__":
    main()
