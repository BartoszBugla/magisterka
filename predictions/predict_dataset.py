from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from config.global_config import SentimentLabel, TRAIN_ASPECTS, ModelType

from predictions.prediction_fine_tuned import FineTunedModel
from predictions.prediction_tfidf_lsa import TfidfLsaModel
from predictions.prediction_tfidf_lsa_rf import TfidfLsaRfModel


PROGRESS_BAR_STEP = 10


models = {
    ModelType.FINE_TUNED_BERT: lambda: FineTunedModel(
        aspects=TRAIN_ASPECTS,
        local_model_path="saved_models/bert-base-uncased_absa.pt",
    ),
    ModelType.FINE_TUNED_DISTILBERT: lambda: FineTunedModel(
        aspects=TRAIN_ASPECTS,
        local_model_path="saved_models/distilbert-base-uncased_absa.pt",
    ),
    ModelType.TEST_BERT_BASE_UNCASED_ABSA: lambda: FineTunedModel(
        aspects=TRAIN_ASPECTS,
        local_model_path="saved_models/distilbert-base-uncased-finetuned-sst-2-english_test_absa.pt",
    ),
    ModelType.TFIDF_LSA: lambda: TfidfLsaModel(aspects=TRAIN_ASPECTS),
    ModelType.FINE_TUNED_DISTILBERT_SST: lambda: FineTunedModel(
        aspects=TRAIN_ASPECTS,
        local_model_path="saved_models/distilbert-base-uncased-finetuned-sst-2-english_absa.pt",
    ),
    ModelType.TFIDF_LSA_RF: lambda: TfidfLsaRfModel(aspects=TRAIN_ASPECTS),
}

model_cache = {}


def predict_dataset(
    dataset: pd.DataFrame,
    model_type: ModelType,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not found")

    if model_type not in model_cache:
        model_cache[model_type] = models[model_type]()

    model = model_cache[model_type]

    total_rows = len(dataset)

    if total_rows == 0:
        return dataset
    if on_progress is not None:
        on_progress(0, total_rows)

    done = 0

    for index, row in dataset.iterrows():
        text = row["text"]

        if not isinstance(text, str) or not text.strip():
            nm = SentimentLabel.NOTMENTIONED.value
            for aspect in model.aspects:
                dataset.at[index, aspect] = nm
            done += 1
            continue

        prediction = model.predict(text)

        for aspect in model.aspects:
            dataset.at[index, aspect] = prediction[aspect]

        done += 1
        if done % PROGRESS_BAR_STEP == 0 or done == total_rows:
            if on_progress is not None:
                on_progress(done, total_rows)

    return dataset
