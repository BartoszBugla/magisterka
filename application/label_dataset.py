from __future__ import annotations

from pathlib import Path

import pandas as pd

from application.dataset_types import DatasetType
from application.results_repository import EntryMetadata, ResultsRepository, repository

LABEL_INPUT_TYPES = frozenset({DatasetType.CLEANED, DatasetType.RAW_REVIEWS})


def list_labelable_entries(
    repo: ResultsRepository | None = None,
) -> list[EntryMetadata]:
    r = repo or repository
    return sorted(
        (m for m in r.list_entries() if m.dataset_type in LABEL_INPUT_TYPES),
        key=lambda m: m.csv_filename,
    )


def default_labelled_filename(csv_filename: str) -> str:
    p = Path(csv_filename)
    return f"{p.stem}_labelled{p.suffix}"


def load_source_dataframe(repo: ResultsRepository, csv_filename: str) -> pd.DataFrame:
    return pd.read_csv(repo.get_csv_path(csv_filename))
