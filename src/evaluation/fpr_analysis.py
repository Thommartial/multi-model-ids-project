"""False-positive-rate views for the RQ2 analysis.

Small helpers on top of class_analysis.per_class_metrics: per-class
one-vs-rest FPR for a model, and the lowest-FPR model per class across a
set of models. FPR matters because a high false-positive rate makes an
IDS unusable in practice even when recall looks fine.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.class_analysis import per_class_metrics


def per_class_fpr(y_true, y_pred, classes: list[str]) -> pd.Series:
    """One-vs-rest false-positive rate for each class."""
    return per_class_metrics(y_true, y_pred, classes)["fpr"]


def lowest_fpr_per_class(fpr_by_model: pd.DataFrame) -> pd.DataFrame:
    """For each class, the model with the lowest FPR.

    fpr_by_model is a (models x classes) frame. Returns a frame indexed
    by class with [best_model, fpr].
    """
    out = {}
    for c in fpr_by_model.columns:
        col = fpr_by_model[c].dropna()
        if col.empty:
            continue
        out[c] = {"best_model": col.idxmin(), "fpr": float(col.min())}
    return pd.DataFrame(out).T
