from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from config.global_config import MAX_LENGTH, SENTIMENT_3, TRAIN_ASPECTS

NOTMENTIONED = "notmentioned"
SENTIMENT_IGNORE_INDEX = -100


def coerce_text(value) -> str:
    """Turn a dataframe cell into a non-empty string safe for the tokenizer."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " "
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none", "<na>") else " "


class ABSADataset(Dataset):
    """Dual-label ABSA dataset (mention + sentiment) with deferred padding.

    - mention_labels: 0=notmentioned, 1=mentioned
    - sentiment_labels: index into SENTIMENT_3, or -100 for notmentioned
    """

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.n = len(df)

        texts = [coerce_text(v) for v in df["text"].values]
        enc = tokenizer(texts, add_special_tokens=True, truncation=True, max_length=max_length, padding=False)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids")

        s2i = {s: i for i, s in enumerate(SENTIMENT_3)}
        m_rows, s_rows = [], []
        for _, row in df.iterrows():
            m, s = [], []
            for a in TRAIN_ASPECTS:
                lab = str(row[a]).strip().lower()
                if lab == NOTMENTIONED or lab not in s2i:
                    m.append(0)
                    s.append(SENTIMENT_IGNORE_INDEX)
                else:
                    m.append(1)
                    s.append(s2i[lab])
            m_rows.append(m)
            s_rows.append(s)

        self.mention_labels = torch.tensor(m_rows, dtype=torch.long)
        self.sentiment_labels = torch.tensor(s_rows, dtype=torch.long)
        print(f"ABSADataset: {self.n} examples, aspects={len(TRAIN_ASPECTS)}, sentiments={len(SENTIMENT_3)} (+ ignore)")

    def __len__(self):
        return self.n

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "mention_labels": self.mention_labels[i],
            "sentiment_labels": self.sentiment_labels[i],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[i]
        return item

    def get_sentiment_labels_numpy(self) -> np.ndarray:
        return self.sentiment_labels.numpy()

    def get_mention_labels_numpy(self) -> np.ndarray:
        return self.mention_labels.numpy()


class ABSADataCollator:
    """Pads token sequences dynamically; stacks fixed-size label tensors."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        tok_feats = [{k: f[k] for k in ("input_ids", "attention_mask") if k in f} for f in features]
        if "token_type_ids" in features[0]:
            for i, f in enumerate(features):
                tok_feats[i]["token_type_ids"] = f["token_type_ids"]

        batch = self.tokenizer.pad(tok_feats, return_tensors="pt")
        batch["mention_labels"] = torch.stack([f["mention_labels"] for f in features])
        batch["sentiment_labels"] = torch.stack([f["sentiment_labels"] for f in features])
        return batch


def stratified_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 25):
    """Split preserving distribution of mention-count buckets."""
    df = df.reset_index(drop=True)

    def _mention_bin(row):
        n = sum(str(row[a]).strip().lower() != NOTMENTIONED for a in TRAIN_ASPECTS)
        return min(n, 3) if n > 1 else n

    bins = df.apply(_mention_bin, axis=1).to_numpy()
    _, counts = np.unique(bins, return_counts=True)
    stratify = bins if (counts >= 2).all() and len(set(bins)) > 1 else None

    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, stratify=stratify, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def oversample_mentions(df: pd.DataFrame, factor: float = 2.0, seed: int = 25):
    """Duplicate rows with at least one mentioned aspect to reduce imbalance."""
    if factor <= 0:
        return df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    has = df[TRAIN_ASPECTS].apply(lambda col: col.str.lower() != NOTMENTIONED).any(axis=1)
    mentioned = df[has]
    if mentioned.empty:
        return df
    n_extra = int(round(len(mentioned) * factor))
    if n_extra <= 0:
        return df
    rng = np.random.default_rng(seed)
    extra = mentioned.iloc[rng.integers(0, len(mentioned), size=n_extra)]
    return pd.concat([df, extra], ignore_index=True)


def mention_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Per-aspect count of mentioned labels."""
    return {a: int((df[a].str.lower() != NOTMENTIONED).sum()) for a in TRAIN_ASPECTS if a in df}
