"""Flow-pair ("bigram") features for the Multi-Model IDS project.

An adaptation of the Hamed, Dara & Kremer (2018) payload-bigram
technique to flow-summary records. Settled with Prof. Dara on
2026-05-22 (docs/rfa_bigram_spec.md section 7;
experimental_protocol.md section 10.1 condition 5).

The original bigram technique counts character 2-grams in raw packet
payloads (text-like). UNSW-NB15's partitioned benchmark contains no
payload data, only flow-summary statistics; the technique therefore
cannot be applied as-is. The adaptation pairs each flow record with its
predecessor (lag = 1 by default) and derives per-feature
differences, ratios, and concatenated (previous-row) values
-- a *flow-temporal* construction. Cited as *inspired by*, not identical
to, the Hamed et al. payload-bigram technique.

Functions
* create_sequences - wrapper exposing the project's existing
  3-D sliding-window builder; useful when a downstream model wants the
  bigram-style pairs as 3-D (n_windows, window_size, n_features)
  rather than flat per-row derived columns.
* compute_bigram_differences - diff_<f>[i] = f[i] - f[i-lag]
  for each numeric feature f.
* compute_bigram_ratios - ratio_<f>[i] = f[i] / max(|f[i-lag]|, eps).
* compute_bigram_concatenated - prev_<f>[i] = f[i-lag]
  (the predecessor's value, as a new column).
* extract_all_bigram_features - one-stop call that returns the
  original frame with whichever family or families are requested
  appended.

Leakage note
These functions are pure data transformations - nothing is fitted on
the input. If applied *before* the stratified train / val / test split,
a row's derived features depend on its predecessor in the file's
original order, which may end up in a different fold. That is not
classical train -> test leakage (no model fitting on test data), but it
does mean a row's derived columns can use information from rows in
another fold. Apply *after* splitting (within each fold) for the
strictest leakage-clean behaviour; apply before splitting for the
temporally-adjacent-predecessor semantics. The choice is recorded in
the experimental protocol.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data.preprocessor import make_sliding_windows

EPS_DEFAULT = 1e-6


# ---------------------------------------------------------------------------
# Sequence builder (WBS 5.3.1) - wrapper around make_sliding_windows
# ---------------------------------------------------------------------------


def create_sequences(
    x,
    y,
    window_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 3-D (n_windows, window_size, n_features) sequences.

    Thin wrapper around src.data.preprocessor.make_sliding_windows
    for callers who want the bigram-style pairs as a 3-D array rather
    than flat per-row derived columns. With window_size=2 (the
    default) each sequence is a consecutive pair of records.

    Parameters
    x
        2-D feature matrix or DataFrame (rows in pairing order).
    y
        1-D label vector aligned with x.
    window_size
        Pair size (default 2 = consecutive pair).
    """
    if isinstance(x, pd.DataFrame):
        x_arr = x.to_numpy()
    else:
        x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return make_sliding_windows(x_arr, y_arr, window_size=window_size)


# ---------------------------------------------------------------------------
# Per-row derived features (WBS 5.3.2)
# ---------------------------------------------------------------------------


def _numeric_columns(x: pd.DataFrame, columns: Iterable[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    return x.select_dtypes(include=np.number).columns.tolist()


def compute_bigram_differences(
    x: pd.DataFrame,
    lag: int = 1,
    columns: Iterable[str] | None = None,
    fill: float = 0.0,
) -> pd.DataFrame:
    """Per-row differences with the row lag positions earlier.

    For each numeric column f:

        diff_<f>[i] = f[i] - f[i - lag]

    The first lag rows have no predecessor; their derived values are
    filled with fill (default 0).

    Returns a DataFrame with one column per input column, renamed
    diff_<original>. The input DataFrame is not modified.
    """
    cols = _numeric_columns(x, columns)
    diffs = x[cols].diff(periods=lag)
    diffs = diffs.fillna(fill)
    diffs.columns = [f"diff_{c}" for c in cols]
    return diffs.reset_index(drop=True)


def compute_bigram_ratios(
    x: pd.DataFrame,
    lag: int = 1,
    columns: Iterable[str] | None = None,
    fill: float = 0.0,
    eps: float = EPS_DEFAULT,
) -> pd.DataFrame:
    """Per-row ratios with the row lag positions earlier.

    For each numeric column f:

        ratio_<f>[i] = f[i] / max(|f[i - lag]|, eps)

    The first lag rows have no predecessor and rows where the
    predecessor is exactly zero are stabilised by eps; any
    division producing inf is replaced by fill.

    Returns a DataFrame with columns renamed ratio_<original>.
    """
    cols = _numeric_columns(x, columns)
    prev = x[cols].shift(lag)
    denom = prev.abs().clip(lower=eps)
    ratios = x[cols] / denom
    ratios = ratios.replace([np.inf, -np.inf], fill).fillna(fill)
    ratios.columns = [f"ratio_{c}" for c in cols]
    return ratios.reset_index(drop=True)


def compute_bigram_concatenated(
    x: pd.DataFrame,
    lag: int = 1,
    columns: Iterable[str] | None = None,
    fill: float = 0.0,
) -> pd.DataFrame:
    """Per-row predecessor values, as new columns.

    For each numeric column f:

        prev_<f>[i] = f[i - lag]

    This is the "concatenation" path - it widens the feature space by
    exposing each row's predecessor as side-by-side columns. The
    current row's columns are *not* duplicated; the caller can
    concatenate with the original DataFrame if both are wanted (or use
    extract_all_bigram_features with keep_original=True).
    First lag rows are filled with fill.
    """
    cols = _numeric_columns(x, columns)
    shifted = x[cols].shift(lag).fillna(fill)
    shifted.columns = [f"prev_{c}" for c in cols]
    return shifted.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combined extraction (WBS 5.3.3)
# ---------------------------------------------------------------------------


def extract_all_bigram_features(
    x: pd.DataFrame,
    lag: int = 1,
    include: Iterable[str] | None = None,
    columns: Iterable[str] | None = None,
    keep_original: bool = True,
    fill: float = 0.0,
    eps: float = EPS_DEFAULT,
) -> pd.DataFrame:
    """Combine the requested flow-pair feature families into one DataFrame.

    Parameters
    x
        Input DataFrame. Rows are assumed to be in the order in which
        pairing should occur (typically the original file order).
    lag
        How many rows back the predecessor sits (default 1).
    include
        Which families to compute - any subset of
        {"difference", "ratio", "concat"}. Default: all three.
    columns
        Numeric columns to derive features from. Default: every numeric
        column in x.
    keep_original
        If True (default), the returned DataFrame contains the
        original columns followed by the derived ones.
    fill
        Value used to fill the first lag rows (where there is no
        predecessor) and any divisions by zero in the ratio family.
    eps
        Lower bound on |f[i - lag]| when forming the ratio denominator,
        for numerical stability.
    """
    families = set(include) if include is not None else {"difference", "ratio", "concat"}
    unknown = families - {"difference", "ratio", "concat"}
    if unknown:
        raise ValueError(
            f"unknown bigram feature family/families: {sorted(unknown)} "
            f"(expected subset of {{'difference', 'ratio', 'concat'}})"
        )

    parts: list[pd.DataFrame] = []
    if keep_original:
        parts.append(x.reset_index(drop=True))
    if "difference" in families:
        parts.append(compute_bigram_differences(x, lag=lag, columns=columns, fill=fill))
    if "ratio" in families:
        parts.append(compute_bigram_ratios(x, lag=lag, columns=columns, fill=fill, eps=eps))
    if "concat" in families:
        parts.append(compute_bigram_concatenated(x, lag=lag, columns=columns, fill=fill))
    return pd.concat(parts, axis=1)
