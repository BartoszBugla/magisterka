"""Search mention_threshold on a validation set for dual-head ABSA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config.global_config import SENTIMENT_3, SentimentLabel
from model.prepare_dataset import ABSADataCollator

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

NM = SentimentLabel.NOTMENTIONED.value
SENT_LABELS = list(SENTIMENT_3)


def _collect(model, loader, device):
    """Run model on all batches, return stacked logits and labels."""
    model.eval()
    ml, sl, mlb, slb = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            lm = batch.pop("mention_labels").numpy()
            ls = batch.pop("sentiment_labels").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            ml.append(out["mention_logits"].cpu().numpy())
            sl.append(out["sentiment_logits"].cpu().numpy())
            mlb.append(lm)
            slb.append(ls)
    return np.concatenate(ml), np.concatenate(sl), np.concatenate(mlb), np.concatenate(slb)


def _to_labels(m_logits, s_logits, threshold):
    """Convert logits + threshold to flat string labels."""
    probs = F.softmax(torch.tensor(m_logits), dim=-1).numpy()[:, :, 1]
    s_pred = s_logits.argmax(-1)
    mentioned = probs >= threshold

    flat = []
    for i in range(m_logits.shape[0]):
        for j in range(m_logits.shape[1]):
            flat.append(SENT_LABELS[s_pred[i, j]] if mentioned[i, j] else NM)
    return np.array(flat)


def _true_labels(m_labels, s_labels):
    """Convert GT arrays to flat string labels."""
    flat = []
    for i in range(m_labels.shape[0]):
        for j in range(m_labels.shape[1]):
            if m_labels[i, j] == 0:
                flat.append(NM)
            else:
                flat.append(SENT_LABELS[s_labels[i, j]] if s_labels[i, j] >= 0 else NM)
    return np.array(flat)


def tune_mention_threshold(
    model: torch.nn.Module,
    tokenizer: "PreTrainedTokenizerBase",
    val_dataset,
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
    thresholds: np.ndarray | None = None,
    sentiment_weight: float = 0.65,
    detection_weight: float = 0.35,
) -> tuple[float, dict[str, float]]:
    """Pick threshold maximizing weighted combo of sentiment + detection F1."""
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ABSADataCollator(tokenizer), num_workers=0)
    model.to(device)
    m_logits, s_logits, m_labels, s_labels = _collect(model, loader, device)

    m_true_flat = m_labels.reshape(-1)
    y_true = _true_labels(m_labels, s_labels)

    if thresholds is None:
        thresholds = np.linspace(0.30, 0.70, 17)

    best_t, best_score, best_stats = 0.5, -1.0, {}

    for t in thresholds:
        y_pred = _to_labels(m_logits, s_logits, float(t))
        m_pred = (y_pred != NM).astype(np.int64)

        det = f1_score(m_true_flat, m_pred, average="binary", zero_division=0)
        mask = m_true_flat == 1
        sent = f1_score(y_true[mask], y_pred[mask], labels=SENT_LABELS, average="macro", zero_division=0) if mask.sum() else 0.0

        score = sentiment_weight * sent + detection_weight * det
        if score > best_score:
            best_t, best_score = float(t), score
            best_stats = {"sentiment_macro_f1_mentioned": sent, "detection_f1": det, "combined_score": score}

    return best_t, best_stats
