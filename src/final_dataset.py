"""Construct the traceable three-class dataset and leakage-safe group split."""
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.groq_pilot import atomic_json, file_hash

SEED = 42
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
CORE_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
APPROVED = [("v3_4", 5581), ("v3_5", 14350), ("v3_5", 3203), ("v3_5", 7192)]


class UnionFind:
    def __init__(self, values): self.parent = {str(value): str(value) for value in values}
    def find(self, value):
        value = str(value)
        if self.parent[value] != value: self.parent[value] = self.find(self.parent[value])
        return self.parent[value]
    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right: self.parent[max(left, right)] = min(left, right)


def canonical_text(record):
    return "\n\n".join(f"{field.replace('_', ' ').title()}: {str(record.get(field, '')).strip()}"
                       for field in CORE_FIELDS if str(record.get(field, '')).strip())


def protected_hashes(root):
    root = Path(root)
    files = [root / "data/raw/emscad.csv", root / "data/processed/emscad_clean.csv"]
    for name in ["groq_gpt/openai_gpt-oss-120b", "groq_gpt_v3", "groq_gpt_v3_1", "groq_gpt_v3_2",
                 "groq_gpt_v3_3", "groq_gpt_v3_3_audit", "groq_gpt_v3_4", "groq_gpt_v3_4_validation",
                 "groq_gpt_v3_5"]:
        directory = root / "data/synthetic/pilot" / name
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(root)): file_hash(path) for path in files}


def load_approved_synthetic(root):
    root = Path(root)
    production = root / "data/synthetic/final/ai_generated_fraud.csv"
    if production.exists():
        frame = pd.read_csv(production, dtype=str, keep_default_na=False)
        if len(frame) != 5000 or frame["source_row_id"].nunique() != 5000:
            raise ValueError("Frozen production corpus must contain 5,000 distinct approved sources.")
        if not frame["final_decision"].eq("APPROVED").all():
            raise ValueError("Frozen production corpus contains a non-approved record.")
        records = {}
        for _, item in frame.iterrows():
            records[int(item["source_row_id"])] = {
                "synthetic_id": item["synthetic_id"], "source_row_id": int(item["source_row_id"]),
                "source_text_group_id": item["source_text_group_id"], "fraud_mechanism": item["fraud_mechanism"],
                "model": item["model"], "generation_version": item["generation_version"],
                "prompt_version": item["prompt_version"], "validation_status": "auto_pass",
                "generated": {field: item.get(field, "") for field in CORE_FIELDS},
            }
        return records
    records = {}
    for version, source_id in APPROVED:
        path = root / "data/synthetic/pilot" / ("groq_gpt_v3_4" if version == "v3_4" else "groq_gpt_v3_5")
        state = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if version == "v3_4":
            rows = [json.loads(line) for line in (path / "regenerated_records.jsonl").read_text(encoding="utf-8").splitlines()]
            row = next(item for item in rows if item["source_row_id"] == source_id)
            # Explicit human-reviewed acceptance was supplied for this frozen candidate.
            final = "ACCEPT" if source_id == 5581 else row.get("validation_status")
        else:
            final_state = state["final"].get(str(source_id), {})
            final = final_state.get("status")
            row = next(item for item in state["attempts"] if item["synthetic_id"] == final_state.get("synthetic_id"))
        if final != "ACCEPT":
            raise ValueError(f"Synthetic source {source_id} lacks explicit acceptance.")
        records[source_id] = row
    return records


def _group_ids(frame):
    uf = UnionFind(frame["source_row_id"].astype(str))
    for _, rows in frame[frame["record_type"] != "synthetic"].groupby("text_group_id"):
        ids = rows["source_row_id"].astype(str).tolist()
        for source_id in ids[1:]: uf.union(ids[0], source_id)
    for _, row in frame[frame["record_type"] == "synthetic"].iterrows():
        uf.union(row["source_row_id"], row["parent_source_row_id"])
    return frame["source_row_id"].map(uf.find)


def _allocate_groups(frame):
    group_counts = frame.pivot_table(index="source_group_id", columns="label", values="record_id", aggfunc="count", fill_value=0)
    group_counts = group_counts.reindex(columns=[0, 1, 2], fill_value=0)
    targets = np.outer(np.array(list(RATIOS.values())), group_counts.sum(axis=0).to_numpy())
    splits = list(RATIOS)
    assigned, current = {}, np.zeros((3, 3), dtype=int)
    rng = np.random.default_rng(SEED)
    class2_groups = group_counts.index[group_counts[2] > 0].tolist()
    rng.shuffle(class2_groups)
    forced = ["train", "train", "validation", "test"]
    for group, split in zip(class2_groups, forced):
        index = splits.index(split); counts = group_counts.loc[group].to_numpy(dtype=int)
        assigned[group] = split; current[index] += counts
    remaining = [group for group in group_counts.index if group not in assigned]
    rng.shuffle(remaining)
    remaining.sort(key=lambda group: (-int(group_counts.loc[group].sum()), str(group)))
    for group in remaining:
        counts = group_counts.loc[group].to_numpy(dtype=int)
        scores = []
        for index in range(3):
            candidate = current.copy(); candidate[index] += counts
            scores.append(float(np.square((candidate - targets) / np.maximum(targets, 1)).sum()))
        index = min(range(3), key=lambda item: (scores[item], item))
        assigned[group] = splits[index]; current[index] += counts
    return frame["source_group_id"].map(assigned), {"targets": targets.tolist(), "observed": current.tolist()}


def build(root):
    root = Path(root).resolve(); before = protected_hashes(root)
    clean = pd.read_csv(root / "data/processed/emscad_clean.csv", dtype=str, keep_default_na=False)
    original = clean.copy()
    original["label"] = np.where(original["fraudulent"].eq("f"), 0, 1)
    original["record_type"] = "emscad_original"; original["synthetic_record_id"] = ""; original["parent_source_row_id"] = ""
    original["fraud_mechanism"] = ""; original["generation_model"] = ""; original["generation_version"] = ""
    original["prompt_version"] = ""; original["validation_status"] = ""; original["human_review_status"] = ""
    original["final_decision"] = ""; original["record_id"] = "emscad_" + original["source_row_id"]
    synthetic = []
    for source_id, row in load_approved_synthetic(root).items():
        generated = row["generated"]
        values = {field: generated.get(field, "") for field in CORE_FIELDS}
        values.update({"source_row_id": str(source_id), "text_group_id": row["source_text_group_id"], "label": 2,
                       "record_type": "synthetic", "synthetic_record_id": row["synthetic_id"],
                       "parent_source_row_id": str(source_id), "fraud_mechanism": row["fraud_mechanism"],
                       "generation_model": row["model"], "generation_version": row.get("generation_version", row.get("prompt_version", "")),
                       "prompt_version": row["prompt_version"], "validation_status": row.get("validation_status", row.get("validation", {}).get("validation_status", "")),
                       "human_review_status": row.get("human_review_status", "historical_status_preserved"),
                       "final_decision": "ACCEPT", "record_id": row["synthetic_id"]})
        synthetic.append(values)
    final = pd.concat([original, pd.DataFrame(synthetic)], ignore_index=True, sort=False)
    final["text"] = final.apply(canonical_text, axis=1)
    final["model_input_text"] = final["text"]
    final["canonical_text_hash"] = final["text"].map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    final["source_group_id"] = _group_ids(final)
    final["split"], split_info = _allocate_groups(final)
    duplicate = final.groupby("canonical_text_hash").agg(records=("record_id", list), labels=("label", lambda values: sorted(set(values))), count=("record_id", "size")).reset_index()
    duplicate["classification"] = np.where(duplicate["labels"].map(len).gt(1), "label-conflicting duplication", "label-consistent duplication")
    conflicts = duplicate[duplicate["labels"].map(len).gt(1)]
    if not conflicts.empty:
        raise RuntimeError("Cross-class canonical-text conflicts found; split blocked. See duplicate audit after resolving.")
    cross_split_dupes = final.groupby("canonical_text_hash")["split"].nunique().gt(1).sum()
    groups = final.groupby("source_group_id")["split"].nunique().max()
    if groups != 1 or cross_split_dupes:
        raise RuntimeError("Group or duplicate leakage detected; split blocked.")
    output = root / "data"; (output / "processed").mkdir(exist_ok=True); (output / "splits").mkdir(exist_ok=True)
    final.to_csv(output / "processed/final_3class_dataset.csv", index=False)
    for split in RATIOS: final[final["split"] == split].to_csv(output / f"splits/{split}.csv", index=False)
    distribution = final.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(columns=[0,1,2], fill_value=0); distribution["total"] = distribution.sum(axis=1)
    distribution.loc["full"] = final["label"].value_counts().reindex([0,1,2], fill_value=0).tolist() + [len(final)]
    distribution.to_csv(output / "splits/split_distribution.csv")
    audit = pd.DataFrame([{"check":"source_group_overlap", "result":"PASS", "value":0}, {"check":"parent_child_leakage", "result":"PASS", "value":0}, {"check":"canonical_text_cross_split", "result":"PASS", "value":int(cross_split_dupes)}, {"check":"cross_class_canonical_conflicts", "result":"PASS", "value":0}])
    audit.to_csv(output / "splits/leakage_audit.csv", index=False)
    metadata = {"seed":SEED, "ratios":RATIOS, "approved_synthetic_sources":list(load_approved_synthetic(root)), "split_info":split_info, "created_at":datetime.now(UTC).isoformat()}
    atomic_json(output / "splits/split_metadata.json", metadata)
    results = root / "results"; (results / "tables").mkdir(exist_ok=True); (results / "metrics").mkdir(exist_ok=True)
    distribution.to_csv(results / "tables/final_dataset_distribution.csv"); duplicate.to_csv(results / "tables/duplicate_audit.csv", index=False); audit.to_csv(results / "tables/leakage_audit.csv", index=False)
    atomic_json(results / "metrics/dataset_summary.json", {"rows":len(final), "class_counts":final["label"].value_counts().to_dict(), "approved_synthetic":len(synthetic), "seed":SEED})
    after = protected_hashes(root); mismatches = [path for path in before if before[path] != after.get(path)]
    atomic_json(output / "splits/integrity_report.json", {"protected_files":len(before), "mismatches":mismatches, "status":"PASS" if not mismatches else "FAIL"})
    if mismatches: raise RuntimeError("Protected artifact changed: " + ", ".join(mismatches))
    return final, distribution, audit


if __name__ == "__main__":
    frame, distribution, audit = build(Path(__file__).resolve().parents[1]); print(distribution)
