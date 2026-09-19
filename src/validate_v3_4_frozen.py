"""Run v3.4 validation against frozen v3.3 candidates without editing them."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.groq_pilot import atomic_json, file_hash
from src.validation_v3_4 import VERSION, validate_v3_4


def _flatten(value):
    return json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value


def validate_frozen_v3_3(root):
    """Write a versioned comparison only; v3.3 files are never opened for writing."""
    root = Path(root).resolve()
    frozen = root / "data/synthetic/pilot/groq_gpt_v3_3"
    output = root / "data/synthetic/pilot/groq_gpt_v3_4_validation"
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "prompts/gpt_regeneration_v3_protocol.json").read_text(encoding="utf-8"))
    policies = {item["source_row_id"]: item for item in protocol["assignments"]}
    records_path = frozen / "regenerated_records.jsonl"
    frozen_hash = file_hash(records_path)
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    validations, comparisons = [], []
    for record in records:
        source_id = record["source_row_id"]
        result = validate_v3_4(record["source"], record["generated"], policies[source_id])
        old = record.get("validation", {})
        validations.append({
            "source_row_id": source_id,
            "synthetic_id": record["synthetic_id"],
            "fraud_mechanism": record["fraud_mechanism"],
            **{key: _flatten(value) for key, value in result.items()},
        })
        comparisons.append({
            "source_row_id": source_id,
            "v3_3_status": old.get("validation_status"),
            "v3_4_status": result["validation_status"],
            "decision_changed": old.get("validation_status") != result["validation_status"],
            "v3_3_reasons": json.dumps(old.get("reasons", []), ensure_ascii=True),
            "v3_4_reasons": json.dumps(result["reasons"], ensure_ascii=True),
        })

    pd.DataFrame(validations).to_csv(output / "validation_results.csv", index=False)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output / "validator_comparison.csv", index=False)
    counts = pd.Series([row["v3_4_status"] for row in comparisons]).value_counts().to_dict()
    changed = comparison_frame[comparison_frame["decision_changed"]]["source_row_id"].tolist()
    summary = {
        "validation_version": VERSION,
        "source_run": "groq_gpt_v3_3",
        "frozen_records": len(records),
        "v3_3_records_sha256": frozen_hash,
        "v3_4_status_counts": counts,
        "decision_changes": changed,
        "generated_at": datetime.now(UTC).isoformat(),
        "read_only_guarantee": "The frozen v3.3 JSONL was read and hash-verified after validation.",
    }
    atomic_json(output / "validation_summary.json", summary)
    changes = [
        "# Validator v3.4 Changes",
        "",
        "- Experience durations use structured lower-bound and range comparison.",
        "- Compensation values normalize currency notation; ambiguous multi-value sources require human review.",
        "- Artifact checks use word boundaries and local context, preserving ordinary job language such as test-driven.",
        "- Fraud integration is field-aware: one coherent recruitment-context sentence can pass, while a signal in benefits or a detached warning fails.",
        "",
        "## Frozen Comparison",
        "",
        f"- Records revalidated: {len(records)}",
        f"- Decision changes: {', '.join(map(str, changed)) if changed else 'none'}",
        f"- Frozen v3.3 JSONL SHA-256: `{frozen_hash}`",
    ]
    (output / "validator_changes.md").write_text("\n".join(changes) + "\n", encoding="utf-8")
    if file_hash(records_path) != frozen_hash:
        raise RuntimeError("Frozen v3.3 records changed during the read-only validation run.")
    return summary


if __name__ == "__main__":
    print(json.dumps(validate_frozen_v3_3(Path(__file__).resolve().parents[1]), indent=2))
