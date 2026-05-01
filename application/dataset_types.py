from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DatasetType(str, Enum):
    RAW_REVIEWS = "raw_reviews"
    CLEANED = "cleaned"
    LABELLED_AI = "labelled_ai"
    LABELLED_HUMAN = "labelled_human"

    @property
    def label_pl(self) -> str:
        return _SCHEMAS[self].label_pl

    @property
    def description(self) -> str:
        return _SCHEMAS[self].description

    @property
    def uploadable(self) -> bool:
        return _SCHEMAS[self].uploadable

    @property
    def schema(self) -> DatasetSchema:
        return _SCHEMAS[self]


@dataclass(frozen=True)
class DatasetSchema:
    dataset_type: DatasetType
    label_pl: str
    description: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    requires_aspects: bool = False
    uploadable: bool = True

    @property
    def all_known_columns(self) -> set[str]:
        return set(self.required_columns) | set(self.optional_columns)


REQUIRED_COLUMNS = (
    "name",
    "latitude",
    "longitude",
    "text",
    "time",
    "rating",
)

_SCHEMAS: dict[DatasetType, DatasetSchema] = {
    DatasetType.RAW_REVIEWS: DatasetSchema(
        dataset_type=DatasetType.RAW_REVIEWS,
        label_pl="Surowe — opinie",
        description="Surowe opinie: tekst, identyfikator miejsca, czas, ocena.",
        required_columns=REQUIRED_COLUMNS,
    ),
    DatasetType.CLEANED: DatasetSchema(
        dataset_type=DatasetType.CLEANED,
        label_pl="Przetworzone — oczyszczone opinie",
        description=(
            "Opinie po czyszczeniu (bez emotek, min. liczba słów, "
            "deduplikacja). Bez labelek aspektowych."
        ),
        required_columns=REQUIRED_COLUMNS,
    ),
    DatasetType.LABELLED_AI: DatasetSchema(
        dataset_type=DatasetType.LABELLED_AI,
        label_pl="Oetykowane — silnik AI",
        description=(
            "Opinie z etykietami aspektów nadanymi przez model AI. "
            "Kolumny aspektów są dynamiczne."
        ),
        required_columns=REQUIRED_COLUMNS,
        requires_aspects=True,
    ),
    DatasetType.LABELLED_HUMAN: DatasetSchema(
        dataset_type=DatasetType.LABELLED_HUMAN,
        label_pl="Oetykowane — człowiek",
        description=(
            "Opinie z etykietami aspektów nadanymi ręcznie przez annotatora. "
            "Kolumny aspektów są dynamiczne."
        ),
        required_columns=REQUIRED_COLUMNS,
        requires_aspects=True,
    ),
}
