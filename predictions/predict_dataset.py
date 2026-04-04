from __future__ import annotations

import os
from collections.abc import Callable

import pandas as pd
from config.global_config import MODEL_DIR, TRAIN_ASPECTS, ModelType

from predictions.prediction_llm import LLMPredictionModel
from predictions.prediction_model_base import PredictionModel
from predictions.prediction_zero_shot import ZeroShotPredictionModel
from predictions.prediction_fine_tuned import FineTunedModel


API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, "absa_1000", "model.pt")

PROGRESS_BAR_STEP = 10

_model_cache: dict[ModelType, PredictionModel] = {}


def get_prediction_model(model_type: ModelType) -> PredictionModel:
    if model_type not in _model_cache:
        if model_type == ModelType.ZERO_SHOT:
            _model_cache[model_type] = ZeroShotPredictionModel(aspects=TRAIN_ASPECTS)
        elif model_type == ModelType.LLM:
            _model_cache[model_type] = LLMPredictionModel(
                aspects=TRAIN_ASPECTS, openai_model=OPENAI_MODEL, api_key=API_KEY
            )
        elif model_type == ModelType.FINE_TUNED:
            _model_cache[model_type] = FineTunedModel(
                aspects=TRAIN_ASPECTS, local_model_path=FINE_TUNED_MODEL_PATH
            )
        else:
            raise KeyError(model_type)

    return _model_cache[model_type]


def predict_dataset(
    dataset: pd.DataFrame,
    model_type: ModelType,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    model = get_prediction_model(model_type)
    total_rows = len(dataset)

    if total_rows == 0:
        return dataset
    if on_progress is not None:
        on_progress(0, total_rows)

    done = 0

    for index, row in dataset.iterrows():
        text = row["text"]

        prediction = model.predict(text)

        for aspect in model.aspects:
            dataset.at[index, aspect] = prediction[aspect]

        done += 1
        if done % PROGRESS_BAR_STEP == 0 or done == total_rows:
            if on_progress is not None:
                on_progress(done, total_rows)

    return dataset
