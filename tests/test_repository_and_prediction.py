"""Zbiór scenariuszy A: repozytorium plików + predykcja danych.

Repozytorium (zarządzanie jak w aplikacji):
    A1. Zapis zbioru i spójny odczyt `get_csv_path` + `get_metadata` przygotowuje dane „pod obróbkę”.
    A2. `list_entries` zawiera zapisane wpisy; po `delete` wpis nie występuje na liście.
Predykcja:
    A3. Zbiór o zerowej liczbie wierszy zwraca się bez wgrywania modelu (`total_rows == 0`).
    A4. Tekst opinii jest pusty → aspekty dla tego wiersza ustawione na ``None``, model się nie wywołuje.
    A5. Poprawny tekst → wartości z mockowanego ``predict`` oraz opcjonalny callback ``on_progress``.
"""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from application.dataset_types import DatasetType
from application.results_repository import EntryMetadata, ResultsRepository
from config.global_config import TRAIN_ASPECTS, ModelType
from predictions import predict_dataset as pred_pkg


class _FakeAbsaModel:
    aspects = TRAIN_ASPECTS

    def predict(self, text: str) -> dict[str, str]:
        return {aspect: ("positive" if "good" in text else "neutral") for aspect in self.aspects}


class TestRepositoryWorkflow(unittest.TestCase):
    """Scenariusze związane z życiem wpisu w repozytorium przed/po obróbce danych."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = ResultsRepository(root=self._tmp.name)

    def test_save_roundtrip_matches_bytes_and_metadata(self) -> None:
        name = "for_predict.csv"
        raw = "text,name,latitude,longitude\nHello,Cafe,50.1,19.9\n".encode("utf-8")
        meta = EntryMetadata(
            csv_filename=name,
            dataset_type=DatasetType.LABELLED_AI,
            notes="generated",
        )
        self.repo.save(name, raw, meta)
        csv_path, loaded_meta = self.repo.get(name)
        self.assertEqual(csv_path.read_bytes(), raw)
        self.assertEqual(loaded_meta.dataset_type, DatasetType.LABELLED_AI)
        self.assertEqual(loaded_meta.notes, "generated")

    def test_list_then_delete_updates_entries(self) -> None:
        for n in ("a.csv", "b.csv"):
            self.repo.save(
                n,
                b"x\n",
                EntryMetadata(csv_filename=n, dataset_type=DatasetType.CLEANED),
            )
        self.assertEqual(len(self.repo.list_entries()), 2)
        self.repo.delete("a.csv")
        remaining = self.repo.list_entries()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].csv_filename, "b.csv")


class TestPredictionWorkflow(unittest.TestCase):
    """Scenariusze predykcji bez wgrywania prawdziwych wag modelu."""

    def setUp(self) -> None:
        pred_pkg.model_cache.clear()

    def tearDown(self) -> None:
        pred_pkg.model_cache.clear()

    def test_empty_dataframe_returns_immediately(self) -> None:
        df = pd.DataFrame(columns=["text"])
        out = pred_pkg.predict_dataset(df, ModelType.TFIDF_LSA)
        self.assertTrue(out.empty)

    def test_blank_text_sets_aspects_none_without_predict(self) -> None:
        texts_seen: list[str] = []

        class SpyModel(_FakeAbsaModel):
            def predict(self, text: str) -> dict[str, str]:
                texts_seen.append(text)
                return super().predict(text)

        df = pd.DataFrame({"text": ["", " "]})
        patched = {
            ModelType.TFIDF_LSA: lambda: SpyModel(),
        }

        with patch.object(pred_pkg, "models", patched):
            pred_pkg.predict_dataset(df, ModelType.TFIDF_LSA)

        self.assertEqual(texts_seen, [])
        for idx in df.index:
            for aspect in TRAIN_ASPECTS:
                self.assertIsNone(df.at[idx, aspect])

    def test_predict_fills_labels_and_reports_progress(self) -> None:
        df = pd.DataFrame(
            {"text": ["This is good", "average place"], "noise": [1, 2]}
        )
        patched = {
            ModelType.TFIDF_LSA: lambda: _FakeAbsaModel(),
        }
        progress_snapshots: list[tuple[int, int]] = []

        with patch.object(pred_pkg, "models", patched):
            out = pred_pkg.predict_dataset(
                df,
                ModelType.TFIDF_LSA,
                on_progress=lambda cur, total: progress_snapshots.append((cur, total)),
            )

        self.assertEqual(out.loc[0, "safety"], "positive")
        self.assertEqual(out.loc[1, "safety"], "neutral")
        self.assertEqual(progress_snapshots[0], (0, 2))
        self.assertEqual(progress_snapshots[-1], (2, 2))


if __name__ == "__main__":
    unittest.main()
