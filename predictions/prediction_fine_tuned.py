"""Inference wrapper for fine-tuned dual-head ABSA checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config.global_config import MODEL_DIR, TRAIN_ASPECTS, SentimentLabel
from model.model import ABSAModel
from model.predict import predict as dual_head_predict
from predictions.prediction_model_base import PredictionModel

DEFAULT_BASE_MODEL = "bert-base-uncased"
NOTMENTIONED = SentimentLabel.NOTMENTIONED.value


class FineTunedModel(PredictionModel):
    """Load a ``model.pt`` checkpoint and run dual-head ABSA inference."""

    def __init__(
        self,
        local_model_path: str | None = None,
        aspects: list[str] | None = None,
    ):
        super().__init__(aspects if aspects is not None else [])
        path = Path(local_model_path or str(MODEL_DIR))
        self._model, self._tokenizer, self._mention_threshold = _load_checkpoint(path)

    def predict(self, text: str) -> dict[str, str]:
        labels, _ = dual_head_predict(
            text,
            self._model,
            self._tokenizer,
            mention_threshold=self._mention_threshold,
        )
        return {a: labels.get(a, NOTMENTIONED) for a in self.aspects}


def _load_checkpoint(
    path: Path,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase, float]:
    ckpt_file = path if path.is_file() else path / "model.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state = loaded["model_state_dict"]
        base_model = loaded.get("base_model_name", DEFAULT_BASE_MODEL)
        threshold = float(loaded.get("mention_threshold", 0.5))
    else:
        state = loaded
        base_model = DEFAULT_BASE_MODEL
        threshold = 0.5

    model = ABSAModel(base_model, num_aspects=len(TRAIN_ASPECTS))
    model.load_state_dict(state, strict=False)
    model.cpu().eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer, threshold
