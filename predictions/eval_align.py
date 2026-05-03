"""Align ground-truth and prediction DataFrames for evaluation (stale cache vs CSV)."""

from __future__ import annotations

import warnings

import pandas as pd


def narrow_eval_to_common_rows(
    gt: pd.DataFrame,
    preds: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Trim ``gt`` and every frame in ``preds`` to ``min(len(...))`` rows.

    Cached predictions often keep an old row count after ``validate.csv`` shrinks
    (or grows), which makes ``sklearn`` raise *inconsistent numbers of samples*.
    """
    if not preds:
        return gt, preds
    lengths = [len(gt)] + [len(p) for p in preds.values()]
    m = min(lengths)
    if m != max(lengths):
        warnings.warn(
            "Niejednakowa liczba wierszy GT vs predykcje (cache): "
            f"{lengths}. Metryki liczone na pierwszych "
            f"{m} wierszach. Wyczyść `statics/prediction_cache/` lub "
            "przetwórz ponownie z USE_CACHE=False po zmianie CSV.",
            UserWarning,
            stacklevel=2,
        )
    gt_n = gt.iloc[:m].reset_index(drop=True)
    preds_n = {k: v.iloc[:m].reset_index(drop=True) for k, v in preds.items()}
    return gt_n, preds_n
