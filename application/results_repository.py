"""Filesystem-backed CSV results repository."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from application.dataset_types import DatasetType
from config.global_config import RESULTS_REPOSITORY_DIR, ModelType

METADATA_FILENAME = "metadata.json"


def _model_type_from_stored(raw: str | None) -> ModelType | None:
    if not raw:
        return None
    try:
        return ModelType(raw)
    except ValueError:
        pass
    if raw.startswith("ModelType.") and len(raw) > len("ModelType."):
        return ModelType[raw.removeprefix("ModelType.")]
    raise ValueError(f"Invalid model_type in metadata: {raw!r}")


@dataclass
class EntryMetadata:
    csv_filename: str
    dataset_type: DatasetType
    created_at: str = ""
    notes: str = ""
    model_type: ModelType | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        d = asdict(self)
        d["dataset_type"] = self.dataset_type.value
        d["model_type"] = self.model_type.value if self.model_type else None
        return json.dumps(d, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> EntryMetadata:
        d = json.loads(text)
        d["dataset_type"] = DatasetType(d["dataset_type"])
        d["model_type"] = _model_type_from_stored(d.get("model_type"))
        return cls(**d)


class ResultsRepository:
    def __init__(self, root: Path | str = RESULTS_REPOSITORY_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, csv_filename: str) -> Path:
        return self.root / csv_filename

    def save(
        self, csv_filename: str, csv_bytes: bytes, metadata: EntryMetadata | None = None
    ) -> Path:
        entry_dir = self._entry_dir(csv_filename)

        entry_dir.mkdir(parents=True, exist_ok=True)

        (entry_dir / csv_filename).write_bytes(csv_bytes)

        meta = metadata or EntryMetadata(csv_filename=csv_filename)

        (entry_dir / METADATA_FILENAME).write_text(meta.to_json(), encoding="utf-8")

        return entry_dir

    def get_metadata(self, csv_filename: str) -> EntryMetadata:
        path = self._entry_dir(csv_filename) / METADATA_FILENAME

        if not path.exists():
            raise FileNotFoundError(f"No entry for '{csv_filename}'")

        return EntryMetadata.from_json(path.read_text(encoding="utf-8"))

    def get_csv_path(self, csv_filename: str) -> Path:
        path = self._entry_dir(csv_filename) / csv_filename
        if not path.exists():
            raise FileNotFoundError(f"CSV not found for '{csv_filename}'")
        return path

    def get(self, csv_filename: str) -> tuple[Path, EntryMetadata]:
        return self.get_csv_path(csv_filename), self.get_metadata(csv_filename)

    def delete(self, csv_filename: str) -> None:
        entry_dir = self._entry_dir(csv_filename)

        if entry_dir.is_dir():
            shutil.rmtree(entry_dir)

    def list_entries(self) -> list[EntryMetadata]:
        entries = []

        for meta_file in sorted(self.root.glob(f"*/{METADATA_FILENAME}")):
            entries.append(
                EntryMetadata.from_json(meta_file.read_text(encoding="utf-8"))
            )

        return entries

    def exists(self, csv_filename: str) -> bool:
        return (self._entry_dir(csv_filename) / METADATA_FILENAME).exists()


repository = ResultsRepository()
