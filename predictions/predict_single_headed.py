import torch
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config.global_config import (
    MODEL_DIR,
    TRAIN_ASPECTS,
    SentimentLabel,
)
from model.model import ABSAModel  # Upewnij się, że to zaimportuje nową wersję modelu
from predictions.prediction_model_base import PredictionModel

DEFAULT_BASE_MODEL = "bert-base-uncased"


class FineTunedModelSingleHeaded(PredictionModel):
    def __init__(
        self,
        local_model_path: str | None = None,
        aspects: list[str] | None = None,
    ):
        aspects = aspects if aspects is not None else TRAIN_ASPECTS
        super().__init__(aspects)
        self.model_dir = Path(local_model_path or str(MODEL_DIR))

        # Order must match ``SENTIMENT_LABELS[:3]`` and ``_aspect_labels_to_multihot``.
        self.sentiment_names = ["positive", "neutral", "negative"]

        self._model, self._tokenizer = self._load_checkpoint(self.model_dir)

    def _load_checkpoint(
        self, path: Path
    ) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
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

        model = ABSAModel(
            base_model_name,
            num_aspects=len(TRAIN_ASPECTS),
            num_sentiments=3,  # ZMIANA: Model wyrzuca teraz 3 sentymenty na aspekt, nie 4
        )
        model.load_state_dict(state, strict=False)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        return model, tokenizer

    def predict(self, text: str) -> dict[str, str]:
        # Tokenizacja
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )

        # Przenosimy na to samo urządzenie co model (GPU/CPU)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Model zwraca płaski wektor logitów np. [1, 24]
            logits = self._model(**inputs)

        # Zmieniamy kształt z powrotem na [Liczba Aspektów, 3 Sentymenty]
        logits = logits.view(len(TRAIN_ASPECTS), 3)

        # UWAGA KRYTYCZNA: Używamy Sigmoid dla klasyfikacji binarnej/multilabel, a nie Softmax!
        probs = torch.sigmoid(logits)

        results_dict = {}

        for i, aspect in enumerate(TRAIN_ASPECTS):
            aspect_probs = probs[i]  # Wektor 3 wartości np. [0.85, 0.12, 0.05]

            # Pobieramy najwyższe prawdopodobieństwo i jego indeks dla tego aspektu
            max_prob, max_idx = torch.max(aspect_probs, dim=0)

            # Jeśli najwyższe prawdopodobieństwo przekracza 0.5 - mamy sentyment
            if max_prob > 0.5:
                results_dict[aspect] = self.sentiment_names[max_idx.item()]
            else:
                # Jeśli wszystkie są poniżej 0.5, traktujemy to jako brak wzmianki (Zamiast osobnej klasy)
                results_dict[aspect] = SentimentLabel.NOTMENTIONED.value

        return {
            aspect: results_dict.get(aspect, SentimentLabel.NOTMENTIONED.value)
            for aspect in self.aspects
        }
