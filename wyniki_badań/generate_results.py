"""
Reproduce all research results from the master thesis:
"System oceny bezpieczeństwa i atrakcyjności regionów na podstawie
 analizy danych tekstowych z wykorzystaniem NLP i sztucznej inteligencji"

This script generates:
1. Model comparison table (Table 5 from thesis)
2. Per-aspect F1 heatmaps
3. Agreement matrix between methods
4. Coverage analysis
5. Training history plots for both BERT and DistilBERT-SST
6. Test suite results
7. Dataset statistics summary

All outputs are saved to the research_results/ directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.global_config import TRAIN_ASPECTS, SENTIMENT_LABELS, ModelType
from predictions.eval_align import narrow_eval_to_common_rows
from predictions.labeled_csv import read_labeled_reviews_csv
from predictions.predict_dataset import model_cache, predict_dataset

ASPECTS = TRAIN_ASPECTS
LABELS = SENTIMENT_LABELS
SHORT = ["pos", "neu", "neg", "n/m"]
SENT_3 = [l for l in LABELS if l != "notmentioned"]

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "statics" / "prediction_cache"
TEST_PATH = ROOT / "statics" / "datasets" / "validate.csv"
TRAINING_PATH = ROOT / "statics" / "datasets" / "training.csv"
SAVED_MODELS = ROOT / "saved_models"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def flat(df: pd.DataFrame) -> np.ndarray:
    return np.concatenate([
        df[a].replace("", np.nan).fillna("notmentioned").astype(str).to_numpy()
        for a in ASPECTS
    ])


def load_ground_truth() -> pd.DataFrame:
    df = read_labeled_reviews_csv(str(TEST_PATH))
    for a in ASPECTS:
        df[a] = df[a].fillna("notmentioned")
    return df[["text"] + ASPECTS].copy()


def run_predictions(gt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model_cache.clear()

    _CHECKPOINTS = {
        ModelType.FINE_TUNED_BERT: SAVED_MODELS / "bert-base-uncased_absa.pt",
        ModelType.FINE_TUNED_DISTILBERT_SST: SAVED_MODELS / "distilbert-base-uncased-finetuned-sst-2-english_absa.pt",
    }

    METHODS = {
        "DistilBERT-SST": (ModelType.FINE_TUNED_DISTILBERT_SST, True),
        "BERT": (ModelType.FINE_TUNED_BERT, True),
        "TF-IDF + LR": (ModelType.TFIDF_LSA, True),
    }

    def _cached(key, use_disk, factory):
        path = CACHE_DIR / f"{key}.csv"
        if use_disk and path.is_file():
            print(f"  Loading cached predictions: {path.name}")
            return pd.read_csv(path)
        print(f"  Running predictions for: {key}")
        out = factory()
        out.to_csv(path, index=False)
        return out

    preds = {}
    base = gt.drop(columns=ASPECTS, errors="ignore")

    for name, (mtype, use_disk) in METHODS.items():
        ckpt = _CHECKPOINTS.get(mtype)
        if ckpt and not ckpt.is_file():
            warnings.warn(f'Skipping "{name}": checkpoint not found at {ckpt}')
            continue
        key = f"{mtype.value}_{name.replace(' ', '_')}"
        preds[name] = _cached(
            key, use_disk,
            lambda mt=mtype: predict_dataset(base.copy(), mt, on_progress=None)
        )

    majority = pd.DataFrame({"text": gt["text"].values})
    for a in ASPECTS:
        majority[a] = gt[a].value_counts().idxmax()
    preds["Majority"] = majority

    return preds


def generate_comparison_table(gt: pd.DataFrame, preds: dict) -> pd.DataFrame:
    print("\n=== 1. Model Comparison Table (Table 5) ===")

    rows = []
    for name, pred_df in preds.items():
        yt = flat(gt)
        yp = flat(pred_df)
        rows.append({
            "Method": name,
            "P (3-class)": precision_score(yt, yp, labels=SENT_3, average="macro", zero_division=0),
            "R (3-class)": recall_score(yt, yp, labels=SENT_3, average="macro", zero_division=0),
            "F1 (3-class)": f1_score(yt, yp, labels=SENT_3, average="macro", zero_division=0),
            "P (4-class)": precision_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
            "R (4-class)": recall_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
            "F1 (4-class)": f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
            "Accuracy": accuracy_score(yt, yp),
        })

    table = pd.DataFrame(rows).set_index("Method")
    table = table.round(3)
    print(table.to_string())
    table.to_csv(OUTPUT_DIR / "table5_model_comparison.csv")
    print(f"  Saved: table5_model_comparison.csv")
    return table


def generate_per_aspect_f1_heatmap(gt: pd.DataFrame, preds: dict) -> None:
    print("\n=== 2. Per-Aspect F1 Heatmap ===")

    methods = [m for m in preds if m != "Majority"]

    f1_4class = pd.DataFrame(index=ASPECTS, columns=methods, dtype=float)
    f1_mentioned = pd.DataFrame(index=ASPECTS, columns=methods, dtype=float)

    for method_name in methods:
        pred_df = preds[method_name]
        for aspect in ASPECTS:
            yt = gt[aspect].to_numpy()
            yp = pred_df[aspect].fillna("notmentioned").to_numpy()
            f1_4class.loc[aspect, method_name] = f1_score(
                yt, yp, labels=LABELS, average="macro", zero_division=0
            )
            mask = yt != "notmentioned"
            if mask.sum() > 0:
                f1_mentioned.loc[aspect, method_name] = f1_score(
                    yt[mask], yp[mask], labels=SENT_3, average="macro", zero_division=0
                )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(f1_4class.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                ax=axes[0], vmin=0, vmax=1, linewidths=0.5)
    axes[0].set_title("F1 (4-class) per aspect")

    sns.heatmap(f1_mentioned.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                ax=axes[1], vmin=0, vmax=1, linewidths=0.5)
    axes[1].set_title("F1 (mentioned-only) per aspect")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_aspect_f1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: per_aspect_f1_heatmap.png")

    f1_4class.to_csv(OUTPUT_DIR / "per_aspect_f1_4class.csv")
    f1_mentioned.to_csv(OUTPUT_DIR / "per_aspect_f1_mentioned.csv")


def generate_agreement_matrix(preds: dict) -> None:
    print("\n=== 3. Agreement Matrix ===")

    methods = [m for m in preds if m != "Majority"]
    n = len(methods)
    agree = pd.DataFrame(np.ones((n, n)), index=methods, columns=methods)

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i >= j:
                continue
            y1 = flat(preds[m1])
            y2 = flat(preds[m2])
            score = (y1 == y2).mean()
            agree.loc[m1, m2] = score
            agree.loc[m2, m1] = score

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(agree.astype(float) * 100, annot=True, fmt=".1f", cmap="Blues",
                ax=ax, vmin=50, vmax=100, linewidths=0.5)
    ax.set_title("Agreement between methods (%)")
    fig.savefig(OUTPUT_DIR / "agreement_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    agree.to_csv(OUTPUT_DIR / "agreement_matrix.csv")
    print("  Saved: agreement_matrix.png, agreement_matrix.csv")


def generate_coverage_analysis(gt: pd.DataFrame, preds: dict) -> None:
    print("\n=== 4. Coverage Analysis ===")

    methods = [m for m in preds if m != "Majority"]

    coverage_data = {}
    for method_name in methods:
        pred_df = preds[method_name]
        aspect_coverage = {}
        for aspect in ASPECTS:
            yp = pred_df[aspect].fillna("notmentioned").to_numpy()
            aspect_coverage[aspect] = (yp != "notmentioned").mean()
        coverage_data[method_name] = aspect_coverage

    coverage_df = pd.DataFrame(coverage_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    coverage_df.plot(kind="bar", ax=ax)
    ax.set_title("Coverage per aspect by method")
    ax.set_ylabel("Coverage (fraction of predictions != notmentioned)")
    ax.set_xlabel("Aspect")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(title="Method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "coverage_by_aspect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    coverage_df.to_csv(OUTPUT_DIR / "coverage_by_aspect.csv")
    print("  Saved: coverage_by_aspect.png, coverage_by_aspect.csv")


def generate_training_plots() -> None:
    print("\n=== 5. Training History Plots ===")

    history_files = {
        "DistilBERT-SST": SAVED_MODELS / "distilbert-base-uncased-finetuned-sst-2-english_history.json",
        "BERT": SAVED_MODELS / "bert-base-uncased_history.json",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (name, path) in enumerate(history_files.items()):
        if not path.is_file():
            print(f"  Skipping {name}: history file not found")
            continue

        with open(path) as f:
            data = json.load(f)

        epochs_data = data.get("epochs", data)
        epochs = [e["epoch"] for e in epochs_data]

        train_key = next((k for k in epochs_data[0] if "train_loss" in k), "train_loss")
        eval_key = next((k for k in epochs_data[0] if "eval_loss" in k), "eval_loss")

        train_losses = [e.get(train_key) for e in epochs_data]
        eval_losses = [e.get(eval_key) for e in epochs_data]

        ax = axes[idx]
        ax.plot(epochs, train_losses, "b-o", label="Train loss", markersize=4)
        ax.plot(epochs, eval_losses, "r-o", label="Eval loss", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training: {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: training_history.png")


def generate_dataset_statistics() -> None:
    print("\n=== 6. Dataset Statistics ===")

    stats = {}

    if TRAINING_PATH.is_file():
        train_df = pd.read_csv(TRAINING_PATH)
        stats["training"] = {
            "rows": len(train_df),
            "columns": list(train_df.columns),
            "aspects_present": [a for a in ASPECTS if a in train_df.columns],
        }
        if "text" in train_df.columns:
            word_counts = train_df["text"].dropna().str.split().str.len()
            stats["training"]["text_stats"] = {
                "min_words": int(word_counts.min()),
                "max_words": int(word_counts.max()),
                "median_words": float(word_counts.median()),
                "mean_words": round(float(word_counts.mean()), 1),
            }
        if all(a in train_df.columns for a in ASPECTS):
            dist = {}
            for a in ASPECTS:
                counts = train_df[a].fillna("notmentioned").value_counts().to_dict()
                dist[a] = counts
            stats["training"]["label_distribution"] = dist

    if TEST_PATH.is_file():
        test_df = read_labeled_reviews_csv(str(TEST_PATH))
        stats["test"] = {
            "rows": len(test_df),
            "total_labels": len(test_df) * len(ASPECTS),
        }
        if all(a in test_df.columns for a in ASPECTS):
            dist = {}
            for a in ASPECTS:
                counts = test_df[a].fillna("notmentioned").value_counts().to_dict()
                dist[a] = counts
            stats["test"]["label_distribution"] = dist

    with open(OUTPUT_DIR / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("  Saved: dataset_statistics.json")

    if "training" in stats:
        print(f"  Training set: {stats['training']['rows']} rows")
    if "test" in stats:
        print(f"  Test set: {stats['test']['rows']} rows, {stats['test']['total_labels']} labels")


def run_tests() -> None:
    print("\n=== 7. Running Unit Tests ===")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    test_output = result.stdout + result.stderr
    with open(OUTPUT_DIR / "test_results.txt", "w") as f:
        f.write(test_output)

    print(test_output[-500:] if len(test_output) > 500 else test_output)
    print(f"  Saved: test_results.txt")
    print(f"  Exit code: {result.returncode}")


def generate_confusion_matrices(gt: pd.DataFrame, preds: dict) -> None:
    print("\n=== 8. Confusion Matrices ===")

    methods = [m for m in preds if m != "Majority"]

    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1:
        axes = [axes]

    for idx, method_name in enumerate(methods):
        yt = flat(gt)
        yp = flat(preds[method_name])
        cm = confusion_matrix(yt, yp, labels=LABELS, normalize="true")

        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=SHORT, yticklabels=SHORT,
                    ax=axes[idx], vmin=0, vmax=1, linewidths=0.5)
        axes[idx].set_title(f"Confusion Matrix: {method_name}")
        axes[idx].set_ylabel("True")
        axes[idx].set_xlabel("Predicted")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: confusion_matrices.png")


def main():
    print("=" * 70)
    print("REPRODUCING RESEARCH RESULTS FROM MASTER THESIS")
    print("=" * 70)

    print("\nLoading ground truth dataset...")
    gt = load_ground_truth()
    print(f"  {len(gt)} reviews x {len(ASPECTS)} aspects = {len(gt) * len(ASPECTS)} labels")

    print("\nRunning model predictions (using disk cache if available)...")
    preds = run_predictions(gt)
    gt, preds = narrow_eval_to_common_rows(gt, preds)
    print(f"  Methods: {list(preds.keys())}")

    table = generate_comparison_table(gt, preds)
    generate_per_aspect_f1_heatmap(gt, preds)
    generate_agreement_matrix(preds)
    generate_coverage_analysis(gt, preds)
    generate_confusion_matrices(gt, preds)
    generate_training_plots()
    generate_dataset_statistics()
    run_tests()

    print("\n" + "=" * 70)
    print("ALL RESULTS GENERATED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    print("\n--- Thesis Table 5 Comparison ---")
    print("Expected (from thesis):")
    print("  DistilBERT-SST:  F1(3)=0.505  F1(4)=0.603  Acc=0.813")
    print("  BERT:            F1(3)=0.482  F1(4)=0.586  Acc=0.807")
    print("  TF-IDF + LR:     F1(3)=0.364  F1(4)=0.490  Acc=0.754")
    print("  Majority:        F1(3)=0.141  F1(4)=0.327  Acc=0.785")
    print("\nActual (from this run):")
    for method in ["DistilBERT-SST", "BERT", "TF-IDF + LR", "Majority"]:
        if method in table.index:
            row = table.loc[method]
            print(f"  {method:18s}  F1(3)={row['F1 (3-class)']:.3f}  F1(4)={row['F1 (4-class)']:.3f}  Acc={row['Accuracy']:.3f}")
        else:
            print(f"  {method:18s}  SKIPPED (model not available)")


if __name__ == "__main__":
    main()
