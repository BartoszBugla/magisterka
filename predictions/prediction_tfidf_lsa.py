"""TF-IDF + LSA baselines for ABSA.

Two variants share the same feature pipeline (TF-IDF -> TruncatedSVD) but use
different classifiers: Logistic Regression and Random Forest.  Both train
lazily on first use from ``statics/datasets/training.csv``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config.global_config import (
    REPO_ROOT,
    SENTIMENT_LABELS,
    SentimentLabel,
    TRAIN_ASPECTS,
)
from predictions.prediction_model_base import PredictionModel

_SEED = 25
_N_COMPONENTS = 300


class _TfidfLsaBase(PredictionModel):
    """Shared TF-IDF -> LSA feature pipeline with per-aspect classifiers."""

    _training_csv: str = f"{REPO_ROOT}/statics/datasets/training.csv"

    def __init__(self, aspects: list[str] | None = None):
        super().__init__(aspects if aspects is not None else list(TRAIN_ASPECTS))
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._clf: dict[str, ClassifierMixin] = {}

    def _make_classifier(self) -> ClassifierMixin:
        raise NotImplementedError

    def _fit_if_needed(self) -> None:
        if self._tfidf is not None:
            return

        df = pd.read_csv(self._training_csv)
        for a in self.aspects:
            df[a] = df[a].fillna(SentimentLabel.NOTMENTIONED.value)

        texts = df["text"].fillna("").astype(str).to_numpy()
        label_to_idx = {s: i for i, s in enumerate(SENTIMENT_LABELS)}
        nm_idx = label_to_idx[SentimentLabel.NOTMENTIONED.value]
        y = {
            a: df[a].map(label_to_idx).fillna(nm_idx).astype(int).values
            for a in self.aspects
        }

        self._tfidf = TfidfVectorizer(
            max_features=20_000, ngram_range=(1, 3), sublinear_tf=True,
            min_df=2, stop_words="english",
        )
        X = self._tfidf.fit_transform(texts)

        self._svd = TruncatedSVD(n_components=_N_COMPONENTS, random_state=_SEED)
        X_lsa = self._svd.fit_transform(X)

        for a in self.aspects:
            clf = self._make_classifier()
            clf.fit(X_lsa, y[a])
            self._clf[a] = clf

    def predict(self, text: str) -> dict[str, str]:
        if not text or not str(text).strip():
            return {a: SentimentLabel.NOTMENTIONED.value for a in self.aspects}

        self._fit_if_needed()
        assert self._tfidf is not None and self._svd is not None

        x_lsa = self._svd.transform(self._tfidf.transform([str(text)]))
        return {
            a: SENTIMENT_LABELS[int(self._clf[a].predict(x_lsa)[0])]
            for a in self.aspects
        }


class TfidfLsaModel(_TfidfLsaBase):
    def _make_classifier(self) -> Any:
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0,
            random_state=_SEED, solver="lbfgs",
        )


class TfidfLsaRfModel(_TfidfLsaBase):
    _training_csv: str = f"{REPO_ROOT}/statics/datasets/tfidf.csv"

    def _make_classifier(self) -> Any:
        return RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=_SEED, n_jobs=-1
        )
