"""Single-text inference for the dual-head ABSA model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from config.global_config import MAX_LENGTH, SENTIMENT_3, TRAIN_ASPECTS
from model.choose_architecture import get_device
from model.prepare_dataset import coerce_text

NM = "notmentioned"


def predict(
    text: str,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    *,
    mention_threshold: float = 0.5,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Return (labels, probabilities) for every aspect."""
    device = get_device()
    model.to(device).eval()

    enc = tokenizer(coerce_text(text), add_special_tokens=True, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    m_probs = F.softmax(out["mention_logits"][0], dim=-1).cpu().numpy()
    s_probs = F.softmax(out["sentiment_logits"][0], dim=-1).cpu().numpy()

    labels, probs = {}, {}
    for i, aspect in enumerate(TRAIN_ASPECTS):
        pm = float(m_probs[i, 1])
        sp = s_probs[i]
        labels[aspect] = SENTIMENT_3[int(sp.argmax())] if pm >= mention_threshold else NM
        probs[aspect] = {NM: round(1.0 - pm, 4), **{SENTIMENT_3[j]: round(float(sp[j]) * pm, 4) for j in range(len(SENTIMENT_3))}}

    return labels, probs
