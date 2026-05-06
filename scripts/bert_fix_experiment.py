"""Quick BERT overfitting diagnosis and fix experiment.

Tests DistilBERT with conservative settings to reduce overfitting and
bring coverage closer to ground truth (~0.24).
Runs on a 500-row training sample first, then validates on the full val set.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.global_config import (
    MAX_LENGTH,
    SENTIMENT_3,
    SENTIMENT_LABELS,
    SentimentLabel,
    TRAIN_ASPECTS,
)
from model.choose_architecture import get_device
from model.model import ABSAModel
from model.prepare_dataset import ABSADataset, stratified_split
from model.threshold_tuning import tune_mention_threshold
from model.train import (
    compute_mention_class_weights,
    compute_sentiment_class_weights,
    train_model,
)

try:
    from transformers import AutoTokenizer, set_seed
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    raise SystemExit("transformers not installed")

SEED = 25
set_seed(SEED)
DATA_PATH = "statics/datasets/training.csv"
VAL_CSV = "statics/datasets/validate.csv"

NM = SentimentLabel.NOTMENTIONED.value
SENT_3_LIST = [l for l in SENTIMENT_LABELS if l != NM]
NM_IDX = SENTIMENT_LABELS.index(NM)
SENT_IDXS = [SENTIMENT_LABELS.index(s) for s in SENT_3_LIST]


def evaluate_on_val(model, tokenizer, val_csv, device, threshold=0.5):
    """Predict on the validation set and compute key metrics."""
    import torch.nn.functional as F

    df = pd.read_csv(val_csv)
    for a in TRAIN_ASPECTS:
        df[a] = df[a].fillna(NM)

    lmap = {s: i for i, s in enumerate(SENTIMENT_LABELS)}
    gt = {a: df[a].map(lmap).fillna(NM_IDX).astype(int).values for a in TRAIN_ASPECTS}

    model.eval()
    model.to(device)
    preds = {a: [] for a in TRAIN_ASPECTS}

    with torch.no_grad():
        for _, row in df.iterrows():
            text = str(row["text"]) if pd.notna(row["text"]) else " "
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            m_probs = torch.softmax(out["mention_logits"], dim=-1)[0, :, 1].cpu().numpy()
            s_preds = out["sentiment_logits"].argmax(-1)[0].cpu().numpy()

            for j, a in enumerate(TRAIN_ASPECTS):
                if m_probs[j] >= threshold:
                    preds[a].append(SENTIMENT_LABELS.index(SENTIMENT_3[s_preds[j]]))
                else:
                    preds[a].append(NM_IDX)

    y_pred = {a: np.array(preds[a]) for a in TRAIN_ASPECTS}
    at = np.concatenate([gt[a] for a in TRAIN_ASPECTS])
    ap = np.concatenate([y_pred[a] for a in TRAIN_ASPECTS])
    det_t = (at != NM_IDX).astype(int)
    det_p = (ap != NM_IDX).astype(int)
    det_f1 = f1_score(det_t, det_p, average="binary", zero_division=0)
    mask = at != NM_IDX
    sent_f1 = f1_score(at[mask], ap[mask], labels=SENT_IDXS, average="macro", zero_division=0) if mask.sum() else 0.0
    f1_4 = f1_score(at, ap, labels=list(range(len(SENTIMENT_LABELS))), average="macro", zero_division=0)
    coverage = (ap != NM_IDX).mean()
    return {"det_f1": det_f1, "sent_f1": sent_f1, "f1_4class": f1_4, "coverage": coverage}


EXPERIMENTS = [
    {
        "name": "DistilBERT_conservative",
        "model_name": "distilbert-base-uncased",
        "max_rows": 2000,
        "epochs": 3,
        "batch_size": 16,
        "lr": 3e-5,
        "grad_accum": 1,
        "oversample": 0.0,
        "focal_gamma": 2.0,
        "label_smooth": 0.1,
        "dropout": 0.3,
        "mention_loss_w": 1.0,
        "sentiment_loss_w": 0.8,
        "patience": 1,
        "threshold": 0.5,
    },
    {
        "name": "DistilBERT_high_threshold",
        "model_name": "distilbert-base-uncased",
        "max_rows": 2000,
        "epochs": 3,
        "batch_size": 16,
        "lr": 2e-5,
        "grad_accum": 1,
        "oversample": 0.5,
        "focal_gamma": 2.0,
        "label_smooth": 0.1,
        "dropout": 0.3,
        "mention_loss_w": 1.2,
        "sentiment_loss_w": 0.8,
        "patience": 1,
        "threshold": 0.6,
    },
    {
        "name": "DistilBERT_SST_lowlr",
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "max_rows": 2000,
        "epochs": 3,
        "batch_size": 16,
        "lr": 1e-5,
        "grad_accum": 1,
        "oversample": 0.5,
        "focal_gamma": 2.0,
        "label_smooth": 0.1,
        "dropout": 0.3,
        "mention_loss_w": 1.0,
        "sentiment_loss_w": 1.0,
        "patience": 1,
        "threshold": 0.5,
    },
]


def run_experiment(cfg):
    print(f"\n{'='*60}")
    print(f"Experiment: {cfg['name']}")
    print(f"  model={cfg['model_name']}  rows={cfg['max_rows']}  epochs={cfg['epochs']}")
    print(f"{'='*60}")

    df = pd.read_csv(DATA_PATH)
    df[TRAIN_ASPECTS] = df[TRAIN_ASPECTS].fillna("notmentioned")
    if cfg["max_rows"]:
        df = df.head(cfg["max_rows"]).reset_index(drop=True)

    train_df, val_df = stratified_split(df, val_ratio=0.2, seed=SEED)

    if cfg["oversample"] > 0:
        from model.prepare_dataset import oversample_mentions
        train_df = oversample_mentions(train_df, factor=cfg["oversample"], seed=SEED)

    print(f"  Train: {len(train_df)}, Val (split): {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_ds = ABSADataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_ds = ABSADataset(val_df, tokenizer, max_length=MAX_LENGTH)

    sent_w = compute_sentiment_class_weights(train_ds.get_sentiment_labels_numpy())
    ment_w = compute_mention_class_weights(train_ds.get_mention_labels_numpy())

    model = ABSAModel(
        cfg["model_name"],
        num_aspects=len(TRAIN_ASPECTS),
        num_sentiment_classes=len(SENTIMENT_3),
        sentiment_class_weights=sent_w,
        mention_pos_weight=ment_w,
        focal_gamma=cfg["focal_gamma"],
        dropout_rate=cfg["dropout"],
        mention_loss_weight=cfg["mention_loss_w"],
        sentiment_loss_weight=cfg["sentiment_loss_w"],
    )

    output_dir = f"./bert_fix_results/{cfg['name']}"
    trainer = train_model(
        model, train_ds, val_ds,
        tokenizer=tokenizer,
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        label_smoothing_factor=cfg["label_smooth"],
        early_stopping_patience=cfg["patience"],
        seed=SEED,
    )

    device = get_device()

    threshold, stats = tune_mention_threshold(
        model, tokenizer, val_ds, device=device, batch_size=32,
    )
    print(f"  Tuned threshold: {threshold:.3f}  (stats: {stats})")

    use_threshold = max(threshold, cfg["threshold"])
    print(f"  Using threshold: {use_threshold:.3f}")

    metrics = evaluate_on_val(model, tokenizer, VAL_CSV, device, threshold=use_threshold)
    print(f"  Val results: det={metrics['det_f1']:.3f}  sent={metrics['sent_f1']:.3f}  4c={metrics['f1_4class']:.3f}  cov={metrics['coverage']:.3f}")

    return {**cfg, "tuned_threshold": threshold, "used_threshold": use_threshold, **metrics}


def main():
    results = []
    for cfg in EXPERIMENTS:
        t0 = time.time()
        try:
            r = run_experiment(cfg)
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No experiments completed.")
        return

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x.get("f1_4class", 0), reverse=True):
        print(f"  {r['name']:35s}  det={r['det_f1']:.3f}  sent={r['sent_f1']:.3f}  4c={r['f1_4class']:.3f}  cov={r['coverage']:.3f}  thr={r['used_threshold']:.2f}  ({r['time_s']:.0f}s)")

    best = max(results, key=lambda x: x.get("f1_4class", 0))
    print(f"\nBEST: {best['name']}  (4-class F1={best['f1_4class']:.4f}, coverage={best['coverage']:.4f})")


if __name__ == "__main__":
    main()
