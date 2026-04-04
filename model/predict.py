import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from model.choose_architecture import choose_architecture
from config.global_config import SENTIMENT_LABELS, TRAIN_ASPECTS

device = choose_architecture()


def predict(
    text: str,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """
    Generates sentiment predictions and probabilities for a single text.
    Uses the MultiHead architecture (softmax per aspect).
    """
    model = model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)  # (1, num_aspects, num_sentiments)
        # Softmax across the sentiment dimension
        probs = (
            F.softmax(logits, dim=-1).cpu().numpy()[0]
        )  # (num_aspects, num_sentiments)

    results = {}
    probabilities = {}

    for i, aspect in enumerate(TRAIN_ASPECTS):
        aspect_probs = probs[i]  # (num_sentiments,)
        sentiment_idx = int(aspect_probs.argmax())

        results[aspect] = SENTIMENT_LABELS[sentiment_idx]
        probabilities[aspect] = {
            SENTIMENT_LABELS[j]: round(float(aspect_probs[j]), 4)
            for j in range(len(SENTIMENT_LABELS))
        }

    return results, probabilities


def predict_batch(
    texts: list[str],
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> list[dict[str, str]]:
    model = model.to(device)
    model.eval()

    all_results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs)  # (
