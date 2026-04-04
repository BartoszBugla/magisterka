from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer

from config.global_config import BASE_BERT_MODEL, MODEL_DIR, ModelType, SentimentLabel
from model.model import BertForABSA
from model.predict import predict as absa_predict
from predictions.prediction_model_base import PredictionModel


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
    def _load_checkpoint(path: Path) -> tuple[BertForABSA, BertTokenizer]:
        path = Path(path)
        ckpt_file = path if path.is_file() else path / "model.pt"

        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

        tokenizer = BertTokenizer.from_pretrained(BASE_BERT_MODEL)
        bert = BertModel.from_pretrained(BASE_BERT_MODEL)
        loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        model = BertForABSA(bert)
        state = (
            loaded["model_state_dict"]
            if isinstance(loaded, dict) and "model_state_dict" in loaded
            else loaded
        )
        model.load_state_dict(state)
        model.eval()

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
