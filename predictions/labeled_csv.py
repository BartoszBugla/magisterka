"""Robust CSV reader for ABSA labeled review exports (15-column schema).

Exports sometimes append trailing fields (e.g. ``created_at``, ``id``,
``lead_time``, ``updated_at``), which breaks ``pandas.read_csv`` when the
header has 15 columns. This reader truncates or pads rows to match the header.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def read_labeled_reviews_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    n = len(header)
    body: list[list[str]] = []
    for row in rows[1:]:
        if not row or (len(row) == 1 and not str(row[0]).strip()):
            continue
        if len(row) > n:
            row = row[:n]
        elif len(row) < n:
            row = row + [""] * (n - len(row))
        body.append(row)
    return pd.DataFrame(body, columns=header)
