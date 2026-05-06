"""Align ground-truth and prediction DataFrames for evaluation."""

from __future__ import annotations

import warnings

import pandas as pd


def narrow_eval_to_common_rows(
    gt: pd.DataFrame,
    preds: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Trim ``gt`` and every frame in ``preds`` to ``min(len(...))`` rows.

    Cached predictions may keep an old row count after the validation CSV
    changes, which makes sklearn raise *inconsistent numbers of samples*.
    """
    if not preds:
        return gt, preds
    lengths = [len(gt)] + [len(p) for p in preds.values()]
    m = min(lengths)
    if m != max(lengths):
        warnings.warn(
            f"Row counts differ across GT and predictions: {lengths}. "
            f"Metrics computed on first {m} rows. Clear statics/prediction_cache/ "
            "or re-run with USE_CACHE=False after changing CSV.",
            UserWarning,
            stacklevel=2,
        )
    gt_n = gt.iloc[:m].reset_index(drop=True)
    preds_n = {k: v.iloc[:m].reset_index(drop=True) for k, v in preds.items()}
    return gt_n, preds_n
