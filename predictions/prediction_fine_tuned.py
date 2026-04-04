from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config.global_config import MODEL_DIR, ModelType, SentimentLabel
from model.model import ABSAModel
from model.predict import predict as absa_predict
from predictions.prediction_model_base import PredictionModel

DEFAULT_BASE_MODEL = "bert-base-uncased"


class FineTunedModel(PredictionModel):
    def __init__(
        self,
        local_model_path: str | None = None,
        aspects: list[str] | None = None,
    ):
        aspects = aspects if aspects is not None else []
        super().__init__(ModelType.FINE_TUNED, aspects)
        self.model_dir = Path(local_model_path or str(MODEL_DIR))
        self._model, self._tokenizer = self._load_checkpoint(self.model_dir)

    @staticmethod
    def _load_checkpoint(path: Path) -> tuple[ABSAModel, PreTrainedTokenizerBase]:
        path = Path(path)
        ckpt_file = path if path.is_file() else path / "model.pt"

        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

        loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        base_model_name = (
            loaded.get("base_model_name", DEFAULT_BASE_MODEL)
            if isinstance(loaded, dict)
            else DEFAULT_BASE_MODEL
        )
        state = (
            loaded["model_state_dict"]
            if isinstance(loaded, dict) and "model_state_dict" in loaded
            else loaded
        )

        model = ABSAModel(base_model_name)
        model.load_state_dict(state, strict=False)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        return model, tokenizer

    def predict(self, text: str) -> dict[str, str]:
        results_dict, _ = absa_predict(
            text,
            self._model,
            self._tokenizer,
        )

        return {
            aspect: results_dict.get(aspect, SentimentLabel.NOTMENTIONED.value)
            for aspect in self.aspects
        }
