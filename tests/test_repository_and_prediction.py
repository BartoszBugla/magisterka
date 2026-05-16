from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from application.dataset_types import DatasetType
from application.dataset_upload_validation import validate
from application.results_repository import EntryMetadata, ResultsRepository
from config.global_config import TRAIN_ASPECTS, ModelType, SentimentLabel
from predictions import predict_dataset as pred_pkg
from predictions.prediction_model_base import PredictionModel


class _FakeAbsaModel:
    aspects = TRAIN_ASPECTS

    def predict(self, text: str) -> dict[str, str]:
        return {
            aspect: ("positive" if "good" in text else "neutral")
            for aspect in self.aspects
        }


class MockAbsaModel(PredictionModel):
    """Model zaślepkowy (mock) zwracający stałe wartości — weryfikuje polimorfizm."""

    def predict(self, text: str) -> dict[str, str]:
        return {a: SentimentLabel.NEUTRAL.value for a in self.aspects}


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

    def test_blank_text_sets_notmentioned_without_predict(self) -> None:
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
        from config.global_config import SentimentLabel

        for idx in df.index:
            for aspect in TRAIN_ASPECTS:
                self.assertEqual(df.at[idx, aspect], SentimentLabel.NOTMENTIONED.value)

    def test_predict_fills_labels_and_reports_progress(self) -> None:
        df = pd.DataFrame({"text": ["This is good", "average place"], "noise": [1, 2]})
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


class TestMockModelPolymorphism(unittest.TestCase):
    """Weryfikacja polimorfizmu PredictionModel za pomocą modelu zaślepkowego (mock)."""

    def setUp(self) -> None:
        pred_pkg.model_cache.clear()
        self.mock_model = MockAbsaModel(aspects=TRAIN_ASPECTS)

    def tearDown(self) -> None:
        pred_pkg.model_cache.clear()

    def test_mock_model_conforms_to_base_interface(self) -> None:
        self.assertIsInstance(self.mock_model, PredictionModel)
        self.assertTrue(hasattr(self.mock_model, "aspects"))
        self.assertTrue(callable(getattr(self.mock_model, "predict", None)))

    def test_mock_model_predict_returns_all_aspects(self) -> None:
        result = self.mock_model.predict("sample review text")
        for aspect in TRAIN_ASPECTS:
            self.assertIn(aspect, result)
            self.assertIn(result[aspect], [s.value for s in SentimentLabel])

    def test_mock_model_injected_via_models_dict(self) -> None:
        patched = {ModelType.TFIDF_LSA: lambda: MockAbsaModel(aspects=TRAIN_ASPECTS)}
        df = pd.DataFrame({"text": ["test sentence"]})

        with patch.object(pred_pkg, "models", patched):
            out = pred_pkg.predict_dataset(df, ModelType.TFIDF_LSA)

        for aspect in TRAIN_ASPECTS:
            self.assertIn(aspect, out.columns)

    def test_predict_dataset_with_mock_fills_all_aspect_columns(self) -> None:
        patched = {ModelType.TFIDF_LSA: lambda: MockAbsaModel(aspects=TRAIN_ASPECTS)}
        df = pd.DataFrame({
            "text": ["Great park", "Terrible road", "Average food"],
        })

        with patch.object(pred_pkg, "models", patched):
            out = pred_pkg.predict_dataset(df, ModelType.TFIDF_LSA)

        for aspect in TRAIN_ASPECTS:
            self.assertIn(aspect, out.columns)
            for idx in out.index:
                self.assertEqual(out.at[idx, aspect], SentimentLabel.NEUTRAL.value)

    def test_progress_callback_invoked_at_boundaries(self) -> None:
        patched = {ModelType.TFIDF_LSA: lambda: MockAbsaModel(aspects=TRAIN_ASPECTS)}
        df = pd.DataFrame({"text": ["review one", "review two", "review three"]})
        snapshots: list[tuple[int, int]] = []

        with patch.object(pred_pkg, "models", patched):
            pred_pkg.predict_dataset(
                df,
                ModelType.TFIDF_LSA,
                on_progress=lambda cur, total: snapshots.append((cur, total)),
            )

        self.assertEqual(snapshots[0], (0, 3))
        self.assertEqual(snapshots[-1], (3, 3))


class TestPredictionToRepositoryIntegration(unittest.TestCase):
    """Test integracyjny: predykcja mock modelem → walidacja → zapis do repozytorium."""

    def setUp(self) -> None:
        pred_pkg.model_cache.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = ResultsRepository(root=self._tmp.name)

    def tearDown(self) -> None:
        pred_pkg.model_cache.clear()

    def test_prediction_result_saved_as_labelled_ai(self) -> None:
        patched = {ModelType.TFIDF_LSA: lambda: MockAbsaModel(aspects=TRAIN_ASPECTS)}
        df = pd.DataFrame({
            "name": ["Muzeum", "Park Jordana"],
            "latitude": [50.06, 50.07],
            "longitude": [19.94, 19.92],
            "text": ["Wspaniałe eksponaty", "Dużo zieleni"],
            "time": ["2024-05-10", "2024-05-11"],
            "rating": [5, 4],
        })

        with patch.object(pred_pkg, "models", patched):
            result = pred_pkg.predict_dataset(df, ModelType.TFIDF_LSA)

        raw = result.to_csv(index=False).encode("utf-8")

        validation_error = validate(raw, DatasetType.LABELLED_AI)
        self.assertIsNone(
            validation_error,
            f"Wynik predykcji nie przeszedł walidacji LABELLED_AI: {validation_error}",
        )

        name = "predicted_output.csv"
        meta = EntryMetadata(
            csv_filename=name,
            dataset_type=DatasetType.LABELLED_AI,
            model_type=ModelType.TFIDF_LSA,
            notes="Etykietowane z mock modelu",
        )
        self.repo.save(name, raw, meta)

        self.assertTrue(self.repo.exists(name))
        csv_path, loaded_meta = self.repo.get(name)
        self.assertEqual(loaded_meta.dataset_type, DatasetType.LABELLED_AI)
        self.assertEqual(loaded_meta.model_type, ModelType.TFIDF_LSA)

        saved_df = pd.read_csv(csv_path)
        for aspect in TRAIN_ASPECTS:
            self.assertIn(aspect, saved_df.columns)


if __name__ == "__main__":
    unittest.main()
