"""Unit tests for src.data.bigram_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.bigram_features import (
    compute_bigram_concatenated,
    compute_bigram_differences,
    compute_bigram_ratios,
    create_sequences,
    extract_all_bigram_features,
)


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 4.0, 8.0, 16.0],
            "b": [10.0, 20.0, 0.0, 5.0, 5.0],
            "label": ["x", "y", "x", "y", "y"],  # non-numeric -> should be ignored
        }
    )


# ---------- create_sequences ----------


def test_create_sequences_default_pair() -> None:
    x = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    windows, targets = create_sequences(x, y, window_size=2)
    # 10 records with window 2 -> 9 windows of shape (2, 2)
    assert windows.shape == (9, 2, 2)
    # target of window i is y[i + window_size - 1]
    assert list(targets) == list(range(1, 10))


def test_create_sequences_accepts_dataframe() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    y = np.array([0, 1, 0, 1])
    windows, _ = create_sequences(df, y, window_size=2)
    assert windows.shape == (3, 2, 2)


# ---------- compute_bigram_differences ----------


def test_differences_lag_1_values() -> None:
    df = _toy_frame()
    out = compute_bigram_differences(df, lag=1, fill=0.0)
    # 'label' is non-numeric and should be dropped
    assert list(out.columns) == ["diff_a", "diff_b"]
    # first row filled with 0
    assert out.loc[0, "diff_a"] == 0.0
    assert out.loc[1, "diff_a"] == 1.0  # 2 - 1
    assert out.loc[2, "diff_a"] == 2.0  # 4 - 2
    assert out.loc[4, "diff_b"] == 0.0  # 5 - 5
    assert out.loc[2, "diff_b"] == -20.0  # 0 - 20


def test_differences_respects_columns_argument() -> None:
    df = _toy_frame()
    out = compute_bigram_differences(df, lag=1, columns=["a"])
    assert list(out.columns) == ["diff_a"]


# ---------- compute_bigram_ratios ----------


def test_ratios_lag_1_values() -> None:
    df = _toy_frame()
    out = compute_bigram_ratios(df, lag=1, fill=0.0, eps=1e-6)
    assert list(out.columns) == ["ratio_a", "ratio_b"]
    # first row -> filled
    assert out.loc[0, "ratio_a"] == 0.0
    assert out.loc[1, "ratio_a"] == pytest.approx(2.0)  # 2 / 1
    assert out.loc[2, "ratio_a"] == pytest.approx(2.0)  # 4 / 2
    # division by 0 is stabilised, not infinite
    assert np.isfinite(out.loc[3, "ratio_b"])


def test_ratios_replace_inf_and_nan() -> None:
    df = pd.DataFrame({"a": [0.0, 10.0, 20.0]})  # second row: 10/0 -> inf without eps
    out = compute_bigram_ratios(df, lag=1, fill=-1.0, eps=1e-12)
    # the value 10/eps is finite but huge; with our small eps it's 1e13
    # which is finite. fill_value applies only where inf/NaN.
    assert np.isfinite(out.loc[1, "ratio_a"])


# ---------- compute_bigram_concatenated ----------


def test_concatenated_values_are_predecessor() -> None:
    df = _toy_frame()
    out = compute_bigram_concatenated(df, lag=1, fill=0.0)
    assert list(out.columns) == ["prev_a", "prev_b"]
    assert out.loc[0, "prev_a"] == 0.0  # no predecessor
    assert out.loc[1, "prev_a"] == 1.0
    assert out.loc[4, "prev_b"] == 5.0


# ---------- extract_all_bigram_features ----------


def test_extract_all_default_includes_everything() -> None:
    df = _toy_frame()
    out = extract_all_bigram_features(df, lag=1)
    # Original (a, b, label) + diff_a, diff_b + ratio_a, ratio_b + prev_a, prev_b
    assert {"a", "b", "label"}.issubset(out.columns)
    assert {"diff_a", "diff_b", "ratio_a", "ratio_b", "prev_a", "prev_b"}.issubset(out.columns)
    assert len(out) == len(df)


def test_extract_all_can_skip_original() -> None:
    df = _toy_frame()
    out = extract_all_bigram_features(df, lag=1, keep_original=False)
    assert "a" not in out.columns
    assert "b" not in out.columns
    assert "diff_a" in out.columns


def test_extract_all_subset_of_families() -> None:
    df = _toy_frame()
    out = extract_all_bigram_features(df, lag=1, include=["difference"], keep_original=False)
    assert list(out.columns) == ["diff_a", "diff_b"]


def test_extract_all_rejects_unknown_family() -> None:
    df = _toy_frame()
    with pytest.raises(ValueError, match="unknown bigram feature family"):
        extract_all_bigram_features(df, lag=1, include=["bogus"])


def test_extract_all_preserves_row_count() -> None:
    df = _toy_frame()
    out = extract_all_bigram_features(df, lag=2)
    # lag=2 should still keep all rows (first 2 filled, not dropped)
    assert len(out) == len(df)
