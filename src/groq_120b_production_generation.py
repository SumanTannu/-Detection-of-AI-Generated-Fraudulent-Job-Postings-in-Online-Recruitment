"""Dedicated Groq GPT-OSS 120B production profile.

The 120B run is isolated from the existing Groq-20B and OpenRouter/Qwen
checkpoints and keeps the locked 5,000-approved-record target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.production_generation import ProductionRun


MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "v3.5-production-gpt-oss-120b-r1"
OUTPUT_NAME = "groq_gpt_oss_120b"
FINAL_FILENAME = "ai_generated_fraud_groq_gpt_oss_120b.csv"


class Groq120BProductionRun(ProductionRun):
    def __init__(self, root: Path, provider=None, wait=None, batch_size: int = 50):
        kwargs = {}
        if wait is not None:
            kwargs["wait"] = wait
        super().__init__(
            root=root,
            provider=provider,
            batch_size=batch_size,
            provider_name="groq",
            model=MODEL,
            prompt_version=PROMPT_VERSION,
            output_name=OUTPUT_NAME,
            include_historical_pilots=True,
            final_filename=FINAL_FILENAME,
            **kwargs,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = Groq120BProductionRun(args.root, batch_size=args.batch_size)
    if args.freeze:
        print(run.freeze())
    elif run.configure():
        print(json.dumps(run.run(args.max_batches), indent=2))


if __name__ == "__main__":
    main()
