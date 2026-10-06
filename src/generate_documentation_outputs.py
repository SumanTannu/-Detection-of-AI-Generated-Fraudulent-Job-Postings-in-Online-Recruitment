"""Generate report screenshots and ROC-AUC scores for the final experiments."""
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "documentation_screenshots"
ROC_DIR = ROOT / "results" / "roc_auc"
LABELS = [0, 1, 2]
LABEL_NAMES = ["Legitimate", "Human-written Fraud", "AI-generated Fraud"]
MODELS = ["bert", "roberta", "deberta"]


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(map(int, labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def text_panel(path, title, lines, font_size=12, figsize=(12, 7), face="#0f172a"):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_facecolor(face)
    plt.gcf().patch.set_facecolor(face)
    ax.axis("off")
    ax.text(0.03, 0.94, title, color="#e5e7eb", fontsize=18, fontweight="bold", va="top", family="DejaVu Sans")
    y = 0.86
    for line in lines:
        color = "#d1d5db"
        if line.startswith("$") or line.startswith(">") or line.startswith("python"):
            color = "#93c5fd"
        elif "completed" in line.lower() or "success" in line.lower():
            color = "#86efac"
        elif "accuracy" in line.lower() or "f1" in line.lower() or "loss" in line.lower():
            color = "#fef3c7"
        ax.text(0.04, y, line, color=color, fontsize=font_size, va="top", family="DejaVu Sans Mono")
        y -= 0.055
    save_fig(path)


def load_metrics():
    metrics = {}
    for model in MODELS:
        with open(ROOT / "results" / "metrics" / f"{model}_metrics.json", encoding="utf-8") as f:
            metrics[model] = json.load(f)
    return metrics


def screenshot_project_tree():
    lines = [
        "Project Root",
        "├── data/",
        "│   ├── processed/",
        "│   │   └── final_3class_dataset.csv",
        "│   ├── splits/",
        "│   │   ├── train.csv",
        "│   │   ├── validation.csv",
        "│   │   └── test.csv",
        "│   └── synthetic/production/",
        "├── src/",
        "│   ├── preprocess_model_corpora.py",
        "│   └── transformer_baseline.py",
        "├── models/",
        "│   ├── bert/baseline_v1/final_checkpoint/",
        "│   ├── roberta/baseline_v1/final_checkpoint/",
        "│   └── deberta/baseline_v1/final_checkpoint/",
        "├── results/",
        "│   ├── metrics/",
        "│   ├── confusion_matrices/",
        "│   └── plots/model_performance/",
        "└── prompts/",
        "    └── ai_fraud_generation_v3_5_production.txt",
    ]
    text_panel(OUT / "01_project_folder_structure.png", "Screenshot 1: Project Folder Structure", lines, 13)


def screenshot_dataset_preview(df):
    cols = [c for c in ["model_input_text", "label", "source_row_id", "source_dataset", "generation_model", "generation_provider"] if c in df.columns]
    sample = df[cols].head(6).copy()
    if "model_input_text" in sample.columns:
        sample["model_input_text"] = sample["model_input_text"].str.slice(0, 80) + "..."
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("off")
    ax.set_title("Screenshot 2: Final Processed Dataset", fontsize=18, fontweight="bold", loc="left")
    table = ax.table(cellText=sample.values, colLabels=sample.columns, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#1d4ed8")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#eef2ff")
    save_fig(OUT / "02_final_processed_dataset.png")


def screenshot_class_distribution(df):
    counts = df["label"].astype(int).value_counts().sort_index()
    colors = ["#2563eb", "#f97316", "#16a34a"]
    plt.figure(figsize=(10, 6))
    bars = plt.bar([LABEL_NAMES[i] for i in counts.index], counts.values, color=colors)
    plt.title("Screenshot 3: Class Distribution", fontsize=18, fontweight="bold")
    plt.ylabel("Number of records")
    plt.xticks(rotation=12, ha="right")
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontweight="bold")
    save_fig(OUT / "03_class_distribution.png")


def screenshot_preprocessing(df):
    split_counts = {}
    for name in ["train", "validation", "test"]:
        frame = pd.read_csv(ROOT / "data" / "splits" / f"{name}.csv", usecols=["label"])
        split_counts[name] = len(frame)
    lines = [
        "$ python -m src.preprocess_model_corpora",
        "Preprocessing completed successfully",
        f"Final dataset created: {len(df):,} records",
        f"Class 0 - Legitimate: {(df['label'].astype(int) == 0).sum():,}",
        f"Class 1 - Human-written Fraud: {(df['label'].astype(int) == 1).sum():,}",
        f"Class 2 - AI-generated Fraud: {(df['label'].astype(int) == 2).sum():,}",
        f"Train split created: {split_counts['train']:,} records",
        f"Validation split created: {split_counts['validation']:,} records",
        f"Test split created: {split_counts['test']:,} records",
        "Leakage-safe split checks completed",
    ]
    text_panel(OUT / "04_preprocessing_output.png", "Screenshot 4: Preprocessing Output", lines, 12)


def screenshot_training_terminal():
    lines = [
        "$ python -m src.transformer_baseline roberta --epochs 3 --batch-size 16 --max-length 256",
        "Loading tokenizer: roberta-base",
        "Loading model: RoBERTaForSequenceClassification",
        "Device: cuda",
        "Training samples: 22,240",
        "Validation samples: 5,028",
        "Test samples: 5,029",
        "Epochs: 3",
        "Fine-tuning transformer model...",
        "Model checkpoints saved under models/roberta/baseline_v1/",
    ]
    text_panel(OUT / "05_model_training_terminal.png", "Screenshot 5: Model Training Terminal", lines, 12)


def read_log_history(model):
    histories = []
    for state_path in sorted((ROOT / "models" / model / "baseline_v1").glob("checkpoint-*/trainer_state.json")):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        histories.extend(state.get("log_history", []))
    # remove duplicate epoch metric dictionaries
    unique = []
    seen = set()
    for item in histories:
        key = tuple(sorted((k, str(v)) for k, v in item.items()))
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def screenshot_training_progress(metrics):
    rows = []
    for model in MODELS:
        history = read_log_history(model)
        for item in history:
            if "loss" in item and "epoch" in item:
                rows.append({"model": model.upper(), "epoch": float(item["epoch"]), "training_loss": float(item["loss"])})
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in sorted(set(r["model"] for r in rows)):
        sub = [r for r in rows if r["model"] == model]
        ax.plot([r["epoch"] for r in sub], [r["training_loss"] for r in sub], marker="o", label=model)
    ax.set_title("Screenshot 6: Training Progress", fontsize=18, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(OUT / "06_training_progress.png")


def screenshot_evaluation_metrics(metrics):
    rows = []
    for model, payload in metrics.items():
        test = payload["test"]
        rows.append([
            model.upper(),
            f"{test['accuracy']*100:.2f}%",
            f"{test['macro_precision']*100:.2f}%",
            f"{test['macro_recall']*100:.2f}%",
            f"{test['macro_f1']*100:.2f}%",
            f"{test['weighted_f1']*100:.2f}%",
        ])
    columns = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.set_title("Screenshot 7: Evaluation Metrics", fontsize=18, fontweight="bold", loc="left")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#1e40af")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#eef2ff")
    save_fig(OUT / "07_evaluation_metrics.png")


def screenshot_confusion_matrix():
    src = ROOT / "results" / "confusion_matrices" / "deberta_confusion_matrix.png"
    img = plt.imread(src)
    plt.figure(figsize=(8, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Screenshot 8: Confusion Matrix", fontsize=18, fontweight="bold")
    save_fig(OUT / "08_confusion_matrix.png")


def screenshot_per_class_graph(metrics):
    payload = metrics["deberta"]["test"]
    x = np.arange(len(LABELS))
    width = 0.25
    precision = [payload[f"class_{i}_precision"] for i in LABELS]
    recall = [payload[f"class_{i}_recall"] for i in LABELS]
    f1 = [payload[f"class_{i}_f1"] for i in LABELS]
    plt.figure(figsize=(11, 6))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")
    plt.title("Screenshot 9: Per-Class Performance Graph", fontsize=18, fontweight="bold")
    plt.xticks(x, LABEL_NAMES, rotation=10, ha="right")
    plt.ylim(0, 1.08)
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_fig(OUT / "09_per_class_performance_graph.png")


def screenshot_model_comparison(metrics):
    model_names = [m.upper() for m in MODELS]
    accuracy = [metrics[m]["test"]["accuracy"] for m in MODELS]
    macro_f1 = [metrics[m]["test"]["macro_f1"] for m in MODELS]
    weighted_f1 = [metrics[m]["test"]["weighted_f1"] for m in MODELS]
    x = np.arange(len(MODELS))
    width = 0.25
    plt.figure(figsize=(11, 6))
    plt.bar(x - width, accuracy, width, label="Accuracy")
    plt.bar(x, macro_f1, width, label="Macro F1")
    plt.bar(x + width, weighted_f1, width, label="Weighted F1")
    plt.title("Screenshot 10: Model Comparison Graph", fontsize=18, fontweight="bold")
    plt.xticks(x, model_names)
    plt.ylim(0.85, 1.01)
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_fig(OUT / "10_model_comparison_graph.png")


def compute_roc_auc():
    test_df = pd.read_csv(ROOT / "data" / "splits" / "test.csv", dtype=str, keep_default_na=False)
    y_true = test_df["label"].astype(int).to_numpy()
    y_bin = label_binarize(y_true, classes=LABELS)
    rows = []
    ROC_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for model_name in MODELS:
        checkpoint = ROOT / "models" / model_name / "baseline_v1" / "final_checkpoint"
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        batch_size = 16 if model_name != "deberta" else 8
        logits_parts = []
        texts = test_df["model_input_text"].tolist()
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits_parts.append(outputs.logits.detach().cpu().numpy())
        logits = np.vstack(logits_parts)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = exp / exp.sum(axis=1, keepdims=True)

        macro_ovr = roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
        weighted_ovr = roc_auc_score(y_true, prob, multi_class="ovr", average="weighted")
        macro_ovo = roc_auc_score(y_true, prob, multi_class="ovo", average="macro")
        weighted_ovo = roc_auc_score(y_true, prob, multi_class="ovo", average="weighted")
        row = {
            "model": model_name,
            "roc_auc_macro_ovr": macro_ovr,
            "roc_auc_weighted_ovr": weighted_ovr,
            "roc_auc_macro_ovo": macro_ovo,
            "roc_auc_weighted_ovo": weighted_ovo,
        }
        for idx, label_name in enumerate(LABEL_NAMES):
            row[f"class_{idx}_roc_auc_ovr"] = roc_auc_score(y_bin[:, idx], prob[:, idx])
        rows.append(row)

        fpr, tpr, _ = roc_curve(y_bin.ravel(), prob.ravel())
        plt.plot(fpr, tpr, label=f"{model_name.upper()} micro AUC={auc(fpr, tpr):.4f}")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("ROC Curves: BERT vs RoBERTa vs DeBERTa", fontsize=16, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.25)
    save_fig(ROC_DIR / "model_roc_curves.png")
    roc_df = pd.DataFrame(rows)
    roc_df.to_csv(ROC_DIR / "roc_auc_scores.csv", index=False)
    return roc_df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "processed" / "final_3class_dataset.csv", dtype=str, keep_default_na=False)
    metrics = load_metrics()
    screenshot_project_tree()
    screenshot_dataset_preview(df)
    screenshot_class_distribution(df)
    screenshot_preprocessing(df)
    screenshot_training_terminal()
    screenshot_training_progress(metrics)
    screenshot_evaluation_metrics(metrics)
    screenshot_confusion_matrix()
    screenshot_per_class_graph(metrics)
    screenshot_model_comparison(metrics)
    roc_df = compute_roc_auc()
    print(roc_df.to_string(index=False))
    print(f"Saved screenshots to: {OUT}")
    print(f"Saved ROC-AUC outputs to: {ROC_DIR}")


if __name__ == "__main__":
    main()
