import json
from pathlib import Path
import unittest
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/processed/final_3class_dataset.csv"


class FinalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(FINAL, dtype=str, keep_default_na=False)
        cls.splits = {name: pd.read_csv(ROOT / f"data/splits/{name}.csv", dtype=str, keep_default_na=False)
                      for name in ("train", "validation", "test")}

    def test_exactly_three_labels(self): self.assertEqual(set(self.frame.label), {"0", "1", "2"})
    def test_ai_records_have_parents_and_label_two(self):
        ai = self.frame[self.frame.label == "2"]; self.assertTrue((ai.parent_source_row_id != "").all()); self.assertTrue((ai.synthetic_record_id != "").all())
    def test_original_labels_match_emscad_flag(self):
        original = self.frame[self.frame.record_type == "emscad_original"]
        self.assertTrue(((original.fraudulent == "f") == (original.label == "0")).all())
    def test_no_group_crosses_splits(self):
        self.assertTrue((self.frame.groupby("source_group_id").split.nunique() == 1).all())
    def test_no_parent_child_split(self):
        ai = self.frame[self.frame.label == "2"]
        for _, row in ai.iterrows():
            parent = self.frame[(self.frame.record_type == "emscad_original") & (self.frame.source_row_id == row.parent_source_row_id)].iloc[0]
            self.assertEqual(row.split, parent.split)
    def test_no_canonical_duplicate_crosses_splits(self): self.assertTrue((self.frame.groupby("canonical_text_hash").split.nunique() == 1).all())
    def test_model_input_has_no_metadata_headers(self):
        forbidden = ["source_row_id:", "generation_model:", "human_review_status:", "fraud_mechanism:"]
        self.assertFalse(self.frame.model_input_text.str.lower().str.contains("|".join(forbidden), regex=True).any())
    def test_split_reconciles(self): self.assertEqual(sum(len(frame) for frame in self.splits.values()), len(self.frame))
    def test_split_metadata_is_reproducible(self): self.assertEqual(json.loads((ROOT / "data/splits/split_metadata.json").read_text())["seed"], 42)
    def test_final_count_reconciles(self): self.assertEqual(len(self.frame), 17880 + int((self.frame.label == "2").sum()))


if __name__ == "__main__": unittest.main()
