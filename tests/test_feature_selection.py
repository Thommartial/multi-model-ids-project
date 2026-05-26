"""Unit tests for :mod:`src.data.feature_selection`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_selection import (
    FilterResult,
    consensus_selection,
    extra_trees_selection,
    mutual_information_selection,
)


def _toy_dataset(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Two informative features, three pure-noise features."""
    rng = np.random.RandomState(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y = ((f1 + 0.6 * f2 + rng.normal(scale=0.3, size=n)) > 0).astype(int)
    x = pd.DataFrame(
        {
            "informative1": f1,
            "informative2": f2,
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
            "noise3": rng.normal(size=n),
        }
    )
    return x, y


# ---------- extra_trees_selection ----------


def test_extra_trees_returns_one_score_per_feature() -> None:
    x, y = _toy_dataset()
    s = extra_trees_selection(x, y, seed=42)
    assert isinstance(s, pd.Series)
    assert len(s) == x.shape[1]
    assert set(s.index) == set(x.columns)
    assert (s >= 0).all()


def test_extra_trees_ranks_informative_above_noise() -> None:
    x, y = _toy_dataset(n=600)
    s = extra_trees_selection(x, y, seed=42)
    top2 = set(s.sort_values(ascending=False).head(2).index)
    # at least one of the two informative features must make the top 2
    assert top2 & {"informative1", "informative2"}


# ---------- mutual_information_selection ----------


def test_mutual_information_returns_one_score_per_feature() -> None:
    x, y = _toy_dataset()
    s = mutual_information_selection(x, y, seed=42, sample_size=None)
    assert isinstance(s, pd.Series)
    assert len(s) == x.shape[1]
    assert set(s.index) == set(x.columns)


def test_mutual_information_subsample_path() -> None:
    """When sample_size < n, the subsampling code path runs without error."""
    x, y = _toy_dataset(n=400)
    s = mutual_information_selection(x, y, seed=42, sample_size=200)
    assert len(s) == x.shape[1]


# ---------- consensus_selection ----------


def test_consensus_rank_mean_returns_top_k() -> None:
    x, y = _toy_dataset(n=400)
    result = consensus_selection(x, y, k=3, mode="rank_mean", seed=42)
    assert isinstance(result, FilterResult)
    assert len(result.selected) == 3
    expected_cols = {
        "feature",
        "et_importance",
        "et_rank",
        "mi_score",
        "mi_rank",
        "consensus_rank",
    }
    assert expected_cols.issubset(set(result.ranking.columns))
    # ranking covers every feature exactly once and is sorted ascending
    assert len(result.ranking) == x.shape[1]
    assert result.ranking["consensus_rank"].is_monotonic_increasing


def test_consensus_intersection_returns_at_most_k() -> None:
    x, y = _toy_dataset(n=400)
    result = consensus_selection(x, y, k=2, mode="intersection", seed=42)
    assert len(result.selected) <= 2


def test_consensus_rejects_unknown_mode() -> None:
    x, y = _toy_dataset()
    with pytest.raises(ValueError, match="unknown mode"):
        consensus_selection(x, y, k=2, mode="bogus", seed=42)


def test_consensus_intersection_requires_k() -> None:
    x, y = _toy_dataset()
    with pytest.raises(ValueError, match="requires k"):
        consensus_selection(x, y, mode="intersection", seed=42)


def test_consensus_reproducible_with_same_seed() -> None:
    x, y = _toy_dataset(n=400)
    r1 = consensus_selection(x, y, k=3, seed=42)
    r2 = consensus_selection(x, y, k=3, seed=42)
    assert r1.selected == r2.selected
    pd.testing.assert_frame_equal(r1.ranking, r2.ranking)
