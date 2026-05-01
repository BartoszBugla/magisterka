"""Klasyczny baseline ABSA: TF-IDF → LSA (SVD) → osobna regresja losowa (Random Forest) na aspekt.

Trenowany jednorazowo przy pierwszym użyciu na `statics/datasets/training.csv`
(ta sama idea co w `ml_course_project/absa_sentiment_analysis.ipynb`).
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from config.global_config import (
    REPO_ROOT,
    SentimentLabel,
    SENTIMENT_LABELS,
    TRAIN_ASPECTS,
)
from predictions.prediction_model_base import PredictionModel

_TRAINING_CSV = f"{REPO_ROOT}/statics/datasets/training.csv"
_SEED = 25
_N_COMPONENTS = 100


class TfidfLsaRfModel(PredictionModel):
    def __init__(self, aspects: list[str] | None = None):
        aspects = aspects if aspects is not None else list(TRAIN_ASPECTS)
        super().__init__(aspects)
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._clf: dict[str, RandomForestClassifier] = {}

    def _fit_if_needed(self) -> None:
        if self._tfidf is not None:
            return

        df = pd.read_csv(_TRAINING_CSV)
        for a in self.aspects:
            df[a] = df[a].fillna(SentimentLabel.NOTMENTIONED.value)

        texts = df["text"].fillna("").astype(str).to_numpy()
        label_to_idx = {s: i for i, s in enumerate(SENTIMENT_LABELS)}
        not_mentioned_idx = label_to_idx[SentimentLabel.NOTMENTIONED.value]
        y = {
            a: df[a].map(label_to_idx).fillna(not_mentioned_idx).astype(int).values
            for a in self.aspects
        }

        self._tfidf = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        X = self._tfidf.fit_transform(texts)

        self._svd = TruncatedSVD(n_components=_N_COMPONENTS, random_state=_SEED)
        X_lsa = self._svd.fit_transform(X)

        for a in self.aspects:
            clf = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=_SEED,
                n_jobs=-1,
            )
            clf.fit(X_lsa, y[a])
            self._clf[a] = clf

    def predict(self, text: str) -> dict[str, str]:
        if not text or not str(text).strip():
            return {a: SentimentLabel.NOTMENTIONED.value for a in self.aspects}

        self._fit_if_needed()
        assert self._tfidf is not None and self._svd is not None

        x = self._tfidf.transform([str(text)])
        x_lsa = self._svd.transform(x)

        out: dict[str, str] = {}
        for a in self.aspects:
            i = int(self._clf[a].predict(x_lsa)[0])
            out[a] = SENTIMENT_LABELS[i]
        return out
