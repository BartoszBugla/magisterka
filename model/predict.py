import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from model.choose_architecture import choose_architecture
from model.prepare_dataset import coerce_text_for_tokenizer
from config.global_config import SENTIMENT_LABELS, TRAIN_ASPECTS

device = choose_architecture()


def predict(
    text: str,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    model = model.to(device)
    model.eval()

    text = coerce_text_for_tokenizer(text)

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
