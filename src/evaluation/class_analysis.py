"""Per-class difficulty metrics for the RQ2 analysis.

Per-class metrics computed straight from saved (y_true, y_pred) pairs, so
the analysis reproduces from the run artifacts without reloading models:

- per_class_metrics: recall, precision, F1, one-vs-rest FPR per class.
- difficulty_ranking: order classes by mean F1 across models.
- most_confused_pairs: largest off-diagonal confusions, normalised by
  true-class support.

Together these answer RQ2: which classes stay hard across models, and
which classes they get confused with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_class_metrics(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    """Recall, precision, F1 and one-vs-rest FPR per class.

    Returns a frame indexed by class with columns
    [recall, precision, f1, fpr, support].
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    rows = {}
    for c in classes:
        tp = int(((yt == c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        tn = int(((yt != c) & (yp != c)).sum())
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision and recall and precision + recall > 0)
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        rows[c] = dict(recall=recall, precision=precision, f1=f1, fpr=fpr, support=tp + fn)
    return pd.DataFrame(rows).T[["recall", "precision", "f1", "fpr", "support"]]


def difficulty_ranking(f1_by_model: pd.DataFrame) -> pd.DataFrame:
    """Rank classes easiest-to-hardest by mean F1 across models.

    f1_by_model is a (models x classes) frame. Returns a frame indexed
    by class (ascending mean F1) with [mean_f1, std_f1].
    """
    mean_f1 = f1_by_model.mean(axis=0).sort_values()
    std_f1 = f1_by_model.std(axis=0)
    return pd.DataFrame({"mean_f1": mean_f1, "std_f1": std_f1.loc[mean_f1.index]})


def most_confused_pairs(y_true, y_pred, classes: list[str], top: int = 12) -> pd.DataFrame:
    """Largest (true -> predicted) confusions, normalised by true support."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    conf = np.zeros((n, n))
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    for t, p in zip(yt, yp):
        conf[idx[t], idx[p]] += 1
    row_sum = conf.sum(axis=1, keepdims=True)
    norm = np.divide(conf, row_sum, out=np.zeros_like(conf), where=row_sum > 0)
    pairs = [
        (classes[i], classes[j], float(norm[i, j])) for i in range(n) for j in range(n) if i != j
    ]
    pairs.sort(key=lambda x: -x[2])
    return pd.DataFrame(pairs[:top], columns=["true", "predicted_as", "frac_of_true_class"])
