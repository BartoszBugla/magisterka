from __future__ import annotations

import unittest

from application.dataset_types import DatasetType, REQUIRED_COLUMNS
from application.dataset_upload_validation import (
    is_csv_valid,
    is_in_limit,
    validate,
)
from config.global_config import FILE_LIMIT


def _build_csv_bytes(columns: list[str], rows: list[list[str]] | None = None) -> bytes:
    header = ",".join(columns)
    if rows is None:
        rows = [["x"] * len(columns)]
    body = "\n".join(",".join(r) for r in rows)
    return f"{header}\n{body}\n".encode("utf-8")


_ALL_REQUIRED = list(REQUIRED_COLUMNS)


class TestCsvFormatValidation(unittest.TestCase):
    """walidacja struktury pliku CSV."""

    def test_empty_file_rejected(self) -> None:
        result = is_csv_valid(b"")
        self.assertIsNotNone(result)
        self.assertIn("No file uploaded", result)

    def test_malformed_csv_rejected(self) -> None:
        broken = b'"unclosed,field\n\x00\x01\x02'
        result = validate(broken, DatasetType.RAW_REVIEWS)
        self.assertIsNotNone(result)

    def test_valid_csv_passes(self) -> None:
        data = _build_csv_bytes(_ALL_REQUIRED)
        result = is_csv_valid(data)
        self.assertIsNone(result)


class TestFileSizeValidation(unittest.TestCase):
    """plik przekraczający limit 200 MB."""

    def test_within_limit_passes(self) -> None:
        small = b"a,b\n1,2\n"
        self.assertIsNone(is_in_limit(small))

    def test_over_limit_rejected(self) -> None:
        oversized = b"x" * (FILE_LIMIT + 1)
        result = is_in_limit(oversized)
        self.assertIsNotNone(result)
        self.assertIn("too large", result)


class TestRequiredColumnsValidation(unittest.TestCase):
    """weryfikacja wymaganych kolumn dla każdego typu zbioru."""

    def test_raw_reviews_with_all_columns_passes(self) -> None:
        data = _build_csv_bytes(_ALL_REQUIRED)
        result = validate(data, DatasetType.RAW_REVIEWS)
        self.assertIsNone(result)

    def test_cleaned_with_all_columns_passes(self) -> None:
        data = _build_csv_bytes(_ALL_REQUIRED)
        result = validate(data, DatasetType.CLEANED)
        self.assertIsNone(result)

    def test_missing_single_required_column_rejected(self) -> None:
        for col in REQUIRED_COLUMNS:
            cols = [c for c in _ALL_REQUIRED if c != col]
            data = _build_csv_bytes(cols)
            with self.subTest(missing_column=col):
                result = validate(data, DatasetType.RAW_REVIEWS)
                self.assertIsNotNone(result, f"Should reject when '{col}' is missing")
                self.assertIn(col, result)


class TestAspectColumnValidation(unittest.TestCase):
    """weryfikacja kolumn aspektowych w zbiorach etykietowanych."""

    def test_labelled_ai_without_aspect_columns_rejected(self) -> None:
        data = _build_csv_bytes(_ALL_REQUIRED)
        result = validate(data, DatasetType.LABELLED_AI)
        self.assertIsNotNone(result)
        self.assertIn("No aspect columns", result)

    def test_labelled_ai_with_valid_sentiments_passes(self) -> None:
        cols = _ALL_REQUIRED + ["safety", "cleanliness"]
        rows = [
            [
                "Cafe",
                "50.0",
                "19.0",
                "Great food",
                "2024-01-01",
                "5",
                "positive",
                "neutral",
            ],
        ]
        data = _build_csv_bytes(cols, rows)
        result = validate(data, DatasetType.LABELLED_AI)
        self.assertIsNone(result)

    def test_labelled_human_with_valid_sentiments_passes(self) -> None:
        cols = _ALL_REQUIRED + ["infrastructure"]
        rows = [
            ["Park", "51.0", "17.0", "Nice park", "2024-02-01", "4", "negative"],
        ]
        data = _build_csv_bytes(cols, rows)
        result = validate(data, DatasetType.LABELLED_HUMAN)
        self.assertIsNone(result)

    def test_labelled_ai_with_nan_aspects_tolerated(self) -> None:
        cols = _ALL_REQUIRED + ["costs"]
        rows = [
            ["Shop", "52.0", "21.0", "Cheap", "2024-03-01", "3", "positive"],
            ["Market", "52.1", "21.1", "Expensive", "2024-03-02", "2", ""],
        ]
        data = _build_csv_bytes(cols, rows)
        result = validate(data, DatasetType.LABELLED_AI)
        self.assertIsNone(result)

    def test_labelled_with_all_sentiment_labels_passes(self) -> None:
        cols = _ALL_REQUIRED + ["nature"]
        rows = [
            ["Forest", "50.0", "19.0", "Beautiful", "2024-01-01", "5", "positive"],
            ["Lake", "50.1", "19.1", "Clean water", "2024-01-02", "4", "neutral"],
            ["River", "50.2", "19.2", "Polluted", "2024-01-03", "2", "negative"],
            ["Field", "50.3", "19.3", "Empty", "2024-01-04", "3", "notmentioned"],
        ]
        data = _build_csv_bytes(cols, rows)
        result = validate(data, DatasetType.LABELLED_AI)
        self.assertIsNone(result)

    def test_labelled_with_multiple_aspect_columns_passes(self) -> None:
        cols = _ALL_REQUIRED + ["safety", "cleanliness", "infrastructure", "costs"]
        rows = [
            [
                "Place",
                "50.0",
                "19.0",
                "Text",
                "2024-01-01",
                "5",
                "positive",
                "neutral",
                "negative",
                "notmentioned",
            ],
        ]
        data = _build_csv_bytes(cols, rows)
        result = validate(data, DatasetType.LABELLED_AI)
        self.assertIsNone(result)

    def test_labelled_ai_with_invalid_sentiment_triggers_nan_fill(self) -> None:
        cols = _ALL_REQUIRED + ["safety"]
        rows = [
            ["Cafe", "50.0", "19.0", "Good food", "2024-01-01", "5", "invalid"],
        ]
        data = _build_csv_bytes(cols, rows)
        result = validate(data, DatasetType.LABELLED_AI)
        self.assertIsNone(result)


class TestEndToEndValidation(unittest.TestCase):
    """Pełna walidacja end-to-end: format + limit + kolumny."""

    def test_complete_raw_reviews_file(self) -> None:
        cols = _ALL_REQUIRED
        rows = [
            [
                "Muzeum Narodowe",
                "50.06",
                "19.94",
                "Wspaniałe eksponaty",
                "2024-05-10",
                "5",
            ],
            ["Rynek Główny", "50.06", "19.94", "Piękne miejsce", "2024-05-11", "4"],
        ]
        data = _build_csv_bytes(cols, rows)
        self.assertIsNone(validate(data, DatasetType.RAW_REVIEWS))

    def test_complete_labelled_ai_file(self) -> None:
        cols = _ALL_REQUIRED + ["safety", "heritage"]
        rows = [
            [
                "Zamek Królewski",
                "50.05",
                "19.93",
                "Bezpieczna okolica",
                "2024-06-01",
                "5",
                "positive",
                "neutral",
            ],
            [
                "Park Jordana",
                "50.06",
                "19.92",
                "Dużo zieleni",
                "2024-06-02",
                "4",
                "notmentioned",
                "positive",
            ],
        ]
        data = _build_csv_bytes(cols, rows)
        self.assertIsNone(validate(data, DatasetType.LABELLED_AI))

    def test_empty_then_valid_are_distinguishable(self) -> None:
        self.assertIsNotNone(validate(b"", DatasetType.RAW_REVIEWS))
        valid = _build_csv_bytes(_ALL_REQUIRED)
        self.assertIsNone(validate(valid, DatasetType.RAW_REVIEWS))


if __name__ == "__main__":
    unittest.main()
