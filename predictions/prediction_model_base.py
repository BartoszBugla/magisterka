from __future__ import annotations

from config.global_config import SentimentLabel


class PredictionModel:
    aspects: list[str]

    def __init__(self, aspects: list[str]):
        self.aspects = aspects

    def predict(self, text: str) -> dict[str, str]:
        label = SentimentLabel.POSITIVE.value
        return {a: label for a in self.aspects}
