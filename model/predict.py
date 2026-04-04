from torch import nn
import torch
from transformers import BertTokenizer
from model.choose_architecture import choose_architecture
from config.global_config import SENTIMENT_LABELS, TRAIN_ASPECTS

device = choose_architecture()


def predict(
    text: str,
    model: nn.Module,
    tokenizer: BertTokenizer,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
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
        logits = model(**inputs)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    n = len(SENTIMENT_LABELS)
    results = {}
    probabilities = {}
    for i, aspect in enumerate(TRAIN_ASPECTS):
        aspect_probs = probs[i * n : (i + 1) * n]
        sentiment_idx = int(aspect_probs.argmax())
        results[aspect] = SENTIMENT_LABELS[sentiment_idx]
        probabilities[aspect] = {
            SENTIMENT_LABELS[j]: round(float(aspect_probs[j]), 4)
            for j in range(n)
        }

    return results, probabilities
