"""Unit tests for the file repository (ResultsRepository).

Scenariusze:
1. Zapis wpisu tworzy katalog z CSV i metadata.json oraz exists() jest True.
2. get_metadata zwraca spójne metadane (typ zbioru, notatki) po zapisie.
3. Ponowny save tej samej nazwy nadpisuje zawartość pliku CSV.
4. list_entries zwraca wpisy w kolejności alfabetycznej po nazwie pliku.
5. Brak wpisu / po delete: exists False oraz odczyt metadanych i CSV rzuca FileNotFoundError.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from application.dataset_types import DatasetType
from application.results_repository import (
    METADATA_FILENAME,
    EntryMetadata,
    ResultsRepository,
)


class TestResultsRepository(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = ResultsRepository(root=self._tmp.name)

    def test_save_creates_csv_metadata_and_exists(self) -> None:
        name = "data.csv"
        raw = b"a,b,c\n1,2,3\n"
        meta = EntryMetadata(
            csv_filename=name,
            dataset_type=DatasetType.CLEANED,
            notes="draft",
        )
        entry_dir = self.repo.save(name, raw, meta)

        self.assertTrue(entry_dir.is_dir())
        self.assertTrue(self.repo.exists(name))
        self.assertEqual((entry_dir / name).read_bytes(), raw)
        self.assertTrue((entry_dir / METADATA_FILENAME).is_file())

    def test_get_metadata_matches_saved_fields(self) -> None:
        name = "labels.csv"
        meta = EntryMetadata(
            csv_filename=name,
            dataset_type=DatasetType.LABELLED_HUMAN,
            notes="annotated batch 1",
        )
        self.repo.save(name, b"x\n", meta)
        loaded = self.repo.get_metadata(name)

        self.assertEqual(loaded.csv_filename, name)
        self.assertEqual(loaded.dataset_type, DatasetType.LABELLED_HUMAN)
        self.assertEqual(loaded.notes, "annotated batch 1")
        self.assertTrue(loaded.created_at)

    def test_save_overwrites_csv_bytes(self) -> None:
        name = "dup.csv"
        self.repo.save(
            name,
            b"v1\n",
            EntryMetadata(csv_filename=name, dataset_type=DatasetType.RAW_REVIEWS),
        )
        self.repo.save(
            name,
            b"v2\n",
            EntryMetadata(csv_filename=name, dataset_type=DatasetType.RAW_REVIEWS),
        )
        path, _meta = self.repo.get(name)
        self.assertEqual(path.read_bytes(), b"v2\n")

    def test_list_entries_sorted_by_filename(self) -> None:
        for fn, letter in [("zebra.csv", b"z\n"), ("alpha.csv", b"a\n")]:
            self.repo.save(
                fn,
                letter,
                EntryMetadata(csv_filename=fn, dataset_type=DatasetType.CLEANED),
            )
        names = [e.csv_filename for e in self.repo.list_entries()]
        self.assertEqual(names, ["alpha.csv", "zebra.csv"])

    def test_delete_and_missing_entries_raise_on_read(self) -> None:
        name = "gone.csv"
        with self.assertRaises(FileNotFoundError):
            self.repo.get_metadata(name)
        with self.assertRaises(FileNotFoundError):
            self.repo.get_csv_path(name)

        self.repo.save(
            name,
            b"1\n",
            EntryMetadata(csv_filename=name, dataset_type=DatasetType.LABELLED_AI),
        )
        self.assertTrue(self.repo.exists(name))
        self.repo.delete(name)

        self.assertFalse(self.repo.exists(name))
        with self.assertRaises(FileNotFoundError):
            self.repo.get_metadata(name)
        with self.assertRaises(FileNotFoundError):
            self.repo.get_csv_path(name)


if __name__ == "__main__":
    unittest.main()
