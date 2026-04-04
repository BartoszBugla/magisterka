from __future__ import annotations

from config.global_config import ModelType, SentimentLabel


class PredictionModel:
    type: ModelType
    aspects: list[str]

    def __init__(self, type: ModelType, aspects: list[str]):
        self.type = type
        self.aspects = aspects

    def predict(self, text: str) -> dict[str, str]:
        label = SentimentLabel.POSITIVE.value
        return {a: label for a in self.aspects}
