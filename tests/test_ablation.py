"""Unit tests for src.evaluation.ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation import (
    AblationResult,
    ConditionResult,
    bootstrap_ci,
    compute_metrics,
    holm_bonferroni,
    pairwise_comparisons,
    run_ablation,
    run_condition,
)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def _three_class_split(n_per_split: int = 200, n_features: int = 6, seed: int = 0):
    """Three-class problem split into train / val / test."""
    rng = np.random.RandomState(seed)
    n_total = 3 * n_per_split
    f_info = rng.normal(size=n_total)
    y = np.full(n_total, "B", dtype=object)
    y[f_info < -0.5] = "A"
    y[f_info > 0.5] = "C"
    cols = [f"f{i}" for i in range(n_features)]
    data = rng.normal(size=(n_total, n_features))
    data[:, 0] = f_info  # the first feature is informative
    x = pd.DataFrame(data, columns=cols)
    return (
        x.iloc[:n_per_split].reset_index(drop=True),
        x.iloc[n_per_split : 2 * n_per_split].reset_index(drop=True),
        x.iloc[2 * n_per_split :].reset_index(drop=True),
        y[:n_per_split],
        y[n_per_split : 2 * n_per_split],
        y[2 * n_per_split :],
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_multiclass_returns_expected_keys() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    m = compute_metrics(y_true, y_pred, task="multiclass")
    assert set(m.keys()) == {"macro_f1", "balanced_accuracy"}
    assert 0 <= m["macro_f1"] <= 1


def test_compute_metrics_binary_includes_roc_pr() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.2, 0.6, 0.8, 0.9])
    m = compute_metrics(y_true, y_pred, y_proba=y_proba, task="binary")
    assert {"macro_f1", "balanced_accuracy", "auroc", "pr_auc"}.issubset(m.keys())


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = np.array([0.80, 0.81, 0.82, 0.83, 0.84])
    lo, hi = bootstrap_ci(values, n_boot=2_000, seed=42)
    assert lo <= values.mean() <= hi
    assert hi - lo > 0


def test_bootstrap_ci_empty_input_returns_nan() -> None:
    lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


def test_holm_bonferroni_smallest_multiplied_by_n() -> None:
    p = [0.01, 0.02, 0.03]
    out = holm_bonferroni(p)
    # Smallest sorted p (0.01) gets multiplied by 3 -> 0.03
    assert out[0] == pytest.approx(0.03)


def test_holm_bonferroni_caps_at_one() -> None:
    p = [0.5, 0.5, 0.5]
    out = holm_bonferroni(p)
    assert (out <= 1.0).all()


def test_holm_bonferroni_preserves_order() -> None:
    p = [0.04, 0.005, 0.5]
    out = holm_bonferroni(p)
    assert len(out) == 3
    # The originally-smallest p (0.005) at index 1 should get the largest multiplier.
    # sorted: 0.005 -> *3 = 0.015; 0.04 -> *2 = 0.08; 0.5 -> *1 = 0.5
    # monotonic: 0.015, 0.08, 0.5
    # restored to original order: 0.08, 0.015, 0.5
    assert out[1] == pytest.approx(0.015)
    assert out[0] == pytest.approx(0.08)
    assert out[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# run_condition / run_ablation
# ---------------------------------------------------------------------------


def test_run_condition_returns_one_row_per_seed() -> None:
    x_tr, x_val, x_te, y_tr, y_val, y_te = _three_class_split()
    result = run_condition(
        name="all_features",
        feature_set=list(x_tr.columns),
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        x_test=x_te, y_test=y_te,
        seeds=[42, 43, 44],
        rf_kwargs={"n_estimators": 20, "max_depth": 5, "n_jobs": 1},
    )
    assert isinstance(result, ConditionResult)
    assert len(result.per_seed_metrics) == 3
    assert {"val_macro_f1", "test_macro_f1", "val_balanced_accuracy",
            "test_balanced_accuracy"}.issubset(result.per_seed_metrics.columns)


def test_run_ablation_runs_multiple_conditions() -> None:
    x_tr, x_val, x_te, y_tr, y_val, y_te = _three_class_split(n_per_split=150)
    feature_sets = {
        "all": list(x_tr.columns),
        "informative_only": ["f0"],  # we constructed f0 to carry the signal
        "noise_only": ["f1", "f2"],
    }
    result = run_ablation(
        x_tr, y_tr, x_val, y_val, x_te, y_te,
        feature_sets=feature_sets,
        seeds=[42, 43, 44],
        rf_kwargs={"n_estimators": 20, "max_depth": 5, "n_jobs": 1},
    )
    assert isinstance(result, AblationResult)
    assert set(result.conditions.keys()) == set(feature_sets.keys())
    summary = result.summary_table()
    assert len(summary) == 3
    assert "test_macro_f1_mean" in summary.columns


def test_run_ablation_informative_beats_noise() -> None:
    x_tr, x_val, x_te, y_tr, y_val, y_te = _three_class_split(n_per_split=250)
    feature_sets = {
        "informative": ["f0"],
        "noise": ["f1", "f2", "f3"],
    }
    result = run_ablation(
        x_tr, y_tr, x_val, y_val, x_te, y_te,
        feature_sets=feature_sets,
        seeds=[42, 43, 44, 45, 46],
        rf_kwargs={"n_estimators": 40, "max_depth": 6, "n_jobs": 1},
    )
    info_mean = result.conditions["informative"].per_seed_metrics["test_macro_f1"].mean()
    noise_mean = result.conditions["noise"].per_seed_metrics["test_macro_f1"].mean()
    assert info_mean > noise_mean


def test_pairwise_comparisons_has_expected_columns() -> None:
    x_tr, x_val, x_te, y_tr, y_val, y_te = _three_class_split(n_per_split=150)
    feature_sets = {
        "all": list(x_tr.columns),
        "subset": ["f0", "f1"],
    }
    result = run_ablation(
        x_tr, y_tr, x_val, y_val, x_te, y_te,
        feature_sets=feature_sets,
        seeds=[42, 43, 44],
        rf_kwargs={"n_estimators": 20, "max_depth": 5, "n_jobs": 1},
    )
    cmp_df = result.pairwise_comparisons
    expected = {"cond_a", "cond_b", "mean_a", "mean_b",
                "delta_a_minus_b", "p_wilcoxon", "p_holm", "claim"}
    assert expected.issubset(cmp_df.columns)
    assert len(cmp_df) == 1  # 2 conditions -> 1 pair
