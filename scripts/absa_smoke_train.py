#!/usr/bin/env python3
"""Minimal ABSA training smoke test (1 epoch, small subset). Run from repo root: uv run python scripts/absa_smoke_train.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, set_seed

from config.global_config import MAX_LENGTH, TRAIN_ASPECTS, SENTIMENT_3
from model.model import ABSAModel, ARCHITECTURE_VERSION
from model.prepare_dataset import ABSADataset, stratified_split
from model.train import (
    compute_mention_class_weights,
    compute_sentiment_class_weights,
    train_model,
)


def main() -> None:
    transformers.logging.set_verbosity_error()
    set_seed(25)

    data_path = ROOT / "statics" / "datasets" / "training.csv"
    if not data_path.is_file():
        print("SKIP: no training.csv at", data_path)
        return

    df = pd.read_csv(data_path)
    df[TRAIN_ASPECTS] = df[TRAIN_ASPECTS].fillna("notmentioned")
    df = df.head(256).reset_index(drop=True)

    train_df, val_df = stratified_split(df, val_ratio=0.2, seed=25)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ABSADataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_ds = ABSADataset(val_df, tokenizer, max_length=MAX_LENGTH)

    sw = compute_sentiment_class_weights(train_ds.get_sentiment_labels_numpy())
    mw = compute_mention_class_weights(train_ds.get_mention_labels_numpy())

    model = ABSAModel(
        model_name,
        num_aspects=len(TRAIN_ASPECTS),
        num_sentiment_classes=len(SENTIMENT_3),
        sentiment_class_weights=sw,
        mention_pos_weight=mw,
        focal_gamma=2.0,
    )

    out_dir = ROOT / "absa_results_smoke"
    out_dir.mkdir(exist_ok=True)

    trainer = train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        tokenizer=tokenizer,
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        early_stopping_patience=1,
        seed=25,
    )

    last_eval = None
    first_loss = None
    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry and first_loss is None:
            first_loss = entry["loss"]
        if "eval_loss" in entry:
            last_eval = entry

    print("first_train_log_loss:", first_loss)
    print("last_eval:", {k: v for k, v in (last_eval or {}).items() if not str(k).startswith("_")})

    model.eval()
    dev = next(model.parameters()).device
    batch = next(iter(trainer.get_eval_dataloader()))
    batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
    assert "loss" in out or "mention_logits" in out
    print("forward_ok logits", out["mention_logits"].shape, out["sentiment_logits"].shape)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
