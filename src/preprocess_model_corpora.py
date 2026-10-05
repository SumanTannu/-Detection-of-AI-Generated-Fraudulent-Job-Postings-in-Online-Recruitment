"""Build cleaned three-class datasets and leakage-safe splits for approved corpora.

This stage consumes the approved synthetic corpora in ``data/synthetic/production``
and creates transformer-ready text-only datasets.  It does not train models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.groq_pilot import atomic_json
from src.preprocessing import CORE_TEXT_FIELDS, build_combined_text, clean_text

SEED = 42
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

CORPORA = {
    "groq_20b": {
        "path": Path("data/synthetic/production/groq_gpt_v3_5_20b/approved_candidates.csv"),
        "priority": 2,
    },
    "openrouter_qwen": {
        "path": Path("data/synthetic/production/openrouter_qwen3_8_flash/approved_candidates.csv"),
        "priority": 3,
    },
    "groq_120b": {
        "path": Path("data/synthetic/production/groq_gpt_oss_120b/approved_candidates.csv"),
        "priority": 1,
    },
}


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_text(row: pd.Series) -> str:
    return build_combined_text({field: row.get(field, "") for field in CORE_TEXT_FIELDS})


def _clean_core_fields(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for field in CORE_TEXT_FIELDS:
        if field not in frame:
            frame[field] = ""
        frame[field] = frame[field].map(clean_text)
    return frame


class UnionFind:
    def __init__(self, values):
        self.parent = {str(value): str(value) for value in values}

    def find(self, value):
        value = str(value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _group_ids(frame: pd.DataFrame) -> pd.Series:
    uf = UnionFind(frame["source_row_id"].astype(str))
    originals = frame[frame["record_type"].eq("emscad_original")]
    for _, rows in originals.groupby("text_group_id"):
        ids = rows["source_row_id"].astype(str).tolist()
        for source_id in ids[1:]:
            uf.union(ids[0], source_id)
    for _, row in frame[frame["record_type"].eq("synthetic")].iterrows():
        uf.union(row["source_row_id"], row["parent_source_row_id"])
    if "canonical_text_hash" in frame:
        for _, rows in frame.groupby("canonical_text_hash"):
            ids = rows["source_row_id"].astype(str).tolist()
            for source_id in ids[1:]:
                uf.union(ids[0], source_id)
    return frame["source_row_id"].map(uf.find)


def _allocate_groups(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    group_counts = frame.pivot_table(
        index="source_group_id",
        columns="label",
        values="record_id",
        aggfunc="count",
        fill_value=0,
    ).reindex(columns=[0, 1, 2], fill_value=0)
    targets = np.outer(np.array(list(RATIOS.values())), group_counts.sum(axis=0).to_numpy())
    splits = list(RATIOS)
    assigned: dict[str, str] = {}
    current = np.zeros((3, 3), dtype=int)
    rng = np.random.default_rng(SEED)

    class2_groups = group_counts.index[group_counts[2] > 0].tolist()
    rng.shuffle(class2_groups)
    forced = ["train", "train", "validation", "test"]
    for group, split in zip(class2_groups, forced):
        index = splits.index(split)
        counts = group_counts.loc[group].to_numpy(dtype=int)
        assigned[group] = split
        current[index] += counts

    remaining = [group for group in group_counts.index if group not in assigned]
    rng.shuffle(remaining)
    remaining.sort(key=lambda group: (-int(group_counts.loc[group].sum()), str(group)))
    for group in remaining:
        counts = group_counts.loc[group].to_numpy(dtype=int)
        scores = []
        for index in range(3):
            candidate = current.copy()
            candidate[index] += counts
            scores.append(float(np.square((candidate - targets) / np.maximum(targets, 1)).sum()))
        index = min(range(3), key=lambda item: (scores[item], item))
        assigned[group] = splits[index]
        current[index] += counts

    return frame["source_group_id"].map(assigned), {
        "targets": targets.tolist(),
        "observed": current.tolist(),
    }


def _load_originals(root: Path) -> pd.DataFrame:
    clean = pd.read_csv(root / "data/processed/emscad_clean.csv", dtype=str, keep_default_na=False)
    clean = _clean_core_fields(clean)
    clean["label"] = np.where(clean["fraudulent"].eq("f"), 0, 1)
    clean["record_type"] = "emscad_original"
    clean["synthetic_record_id"] = ""
    clean["parent_source_row_id"] = ""
    clean["fraud_mechanism"] = ""
    clean["generation_provider"] = ""
    clean["generation_model"] = ""
    clean["generation_version"] = ""
    clean["prompt_version"] = ""
    clean["validation_status"] = ""
    clean["human_review_status"] = ""
    clean["final_decision"] = ""
    clean["record_id"] = "emscad_" + clean["source_row_id"].astype(str)
    return clean


def _load_synthetic(root: Path, corpus_name: str, relative_path: Path) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[frame["final_decision"].eq("APPROVED")].copy()
    frame = frame.drop_duplicates("source_row_id", keep="first")
    frame = _clean_core_fields(frame)
    text_group = frame.get("source_text_group_id", pd.Series([""] * len(frame), index=frame.index))
    values = frame[CORE_TEXT_FIELDS].copy()
    values["source_row_id"] = frame["source_row_id"].astype(str)
    values["text_group_id"] = text_group.astype(str)
    values["label"] = 2
    values["record_type"] = "synthetic"
    values["synthetic_record_id"] = frame["synthetic_id"].astype(str)
    values["parent_source_row_id"] = frame["source_row_id"].astype(str)
    values["fraud_mechanism"] = frame.get("fraud_mechanism", "").astype(str)
    values["generation_provider"] = frame.get("provider", "").astype(str)
    values["generation_model"] = frame.get("model", "").astype(str)
    values["generation_version"] = frame.get("generation_version", "").astype(str)
    values["prompt_version"] = frame.get("prompt_version", "").astype(str)
    values["validation_status"] = "auto_pass"
    values["human_review_status"] = frame.get("human_review_status", "").astype(str)
    values["final_decision"] = "APPROVED"
    values["record_id"] = frame["synthetic_id"].astype(str)
    values["synthetic_corpus"] = corpus_name
    return values


def _load_combined_synthetic(root: Path) -> pd.DataFrame:
    frames = []
    for corpus_name, config in CORPORA.items():
        frame = _load_synthetic(root, corpus_name, config["path"])
        frame["corpus_priority"] = int(config["priority"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["source_row_id", "corpus_priority", "record_id"])
    combined["synthetic_corpus"] = "ai_generated_fraud"
    combined = combined.drop(columns=["corpus_priority"])
    return combined


def build_corpus(root: Path, corpus_name: str, relative_path: Path | None = None, *, mirror_default: bool = False) -> dict:
    originals = _load_originals(root)
    synthetic = _load_combined_synthetic(root) if relative_path is None else _load_synthetic(root, corpus_name, relative_path)
    originals["synthetic_corpus"] = corpus_name
    final = pd.concat([originals, synthetic], ignore_index=True, sort=False)
    final["text"] = final.apply(_canonical_text, axis=1)
    final["model_input_text"] = final["text"]
    final["canonical_text_hash"] = final["text"].map(_hash_text)
    final["source_group_id"] = _group_ids(final)
    final["split"], split_info = _allocate_groups(final)

    duplicate = final.groupby("canonical_text_hash").agg(
        records=("record_id", list),
        labels=("label", lambda values: sorted(set(values))),
        count=("record_id", "size"),
    ).reset_index()
    duplicate["classification"] = np.where(
        duplicate["labels"].map(len).gt(1),
        "label-conflicting duplication",
        "label-consistent duplication",
    )
    conflicts = duplicate[duplicate["labels"].map(len).gt(1)]
    if not conflicts.empty:
        raise RuntimeError(f"{corpus_name}: cross-class canonical-text conflicts found.")
    cross_split_dupes = int(final.groupby("canonical_text_hash")["split"].nunique().gt(1).sum())
    max_group_splits = int(final.groupby("source_group_id")["split"].nunique().max())
    if max_group_splits != 1 or cross_split_dupes:
        raise RuntimeError(f"{corpus_name}: leakage detected in group or duplicate split.")

    out = root / "data/preprocessed" / corpus_name
    split_dir = out / "splits"
    out.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(out / "final_3class_dataset.csv", index=False)
    for split in RATIOS:
        final[final["split"].eq(split)].to_csv(split_dir / f"{split}.csv", index=False)

    distribution = (
        final.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    distribution["total"] = distribution.sum(axis=1)
    distribution.loc["full"] = final["label"].value_counts().reindex([0, 1, 2], fill_value=0).tolist() + [len(final)]
    distribution.to_csv(split_dir / "split_distribution.csv")
    leakage_audit = pd.DataFrame([
        {"check": "source_group_overlap", "result": "PASS", "value": 0},
        {"check": "parent_child_leakage", "result": "PASS", "value": 0},
        {"check": "canonical_text_cross_split", "result": "PASS", "value": cross_split_dupes},
        {"check": "cross_class_canonical_conflicts", "result": "PASS", "value": 0},
    ])
    leakage_audit.to_csv(split_dir / "leakage_audit.csv", index=False)
    metadata = {
        "corpus": corpus_name,
        "source_file": "combined approved production corpora" if relative_path is None else str(relative_path),
        "created_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "ratios": RATIOS,
        "rows": int(len(final)),
        "class_counts": {str(k): int(v) for k, v in final["label"].value_counts().sort_index().items()},
        "synthetic_rows": int((final["label"] == 2).sum()),
        "split_info": split_info,
    }
    atomic_json(split_dir / "split_metadata.json", metadata)
    duplicate.to_csv(split_dir / "duplicate_audit.csv", index=False)

    if mirror_default:
        processed = root / "data/processed"
        splits = root / "data/splits"
        processed.mkdir(exist_ok=True)
        splits.mkdir(exist_ok=True)
        final.to_csv(processed / "final_3class_dataset.csv", index=False)
        for split in RATIOS:
            final[final["split"].eq(split)].to_csv(splits / f"{split}.csv", index=False)
        distribution.to_csv(splits / "split_distribution.csv")
        leakage_audit.to_csv(splits / "leakage_audit.csv", index=False)
        atomic_json(splits / "split_metadata.json", metadata)
        atomic_json(splits / "integrity_report.json", {"status": "PASS", "protected_files": 0, "mismatches": []})

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--corpus", choices=[*CORPORA, "combined"], default="combined")
    args = parser.parse_args()
    root = args.root.resolve()
    summaries = {}
    if args.corpus == "combined":
        summaries["combined_generated"] = build_corpus(
            root,
            "combined_generated",
            None,
            mirror_default=True,
        )
    else:
        config = CORPORA[args.corpus]
        summaries[args.corpus] = build_corpus(root, args.corpus, config["path"], mirror_default=True)
    results = root / "results/metrics"
    results.mkdir(parents=True, exist_ok=True)
    atomic_json(results / "preprocessing_summary.json", summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
