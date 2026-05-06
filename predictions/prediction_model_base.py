"""Abstract base for all ABSA prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PredictionModel(ABC):
    aspects: list[str]

    def __init__(self, aspects: list[str]):
        self.aspects = aspects

    @abstractmethod
    def predict(self, text: str) -> dict[str, str]:
        ...
