"""Tests for the shared model harness :mod:`src.evaluation.model_runner`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.model_runner import ModelRunResult, run_model_seeds
from src.models.rule_based import make_rule_based


def _three_class_split(n_per_split: int = 200, seed: int = 0):
    rng = np.random.RandomState(seed)
    n_total = 3 * n_per_split
    f_info = rng.normal(size=n_total)
    y = np.full(n_total, "B", dtype=object)
    y[f_info < -0.5] = "A"
    y[f_info > 0.5] = "C"
    x = pd.DataFrame(
        {
            "f_info": f_info,
            "noise1": rng.normal(size=n_total),
            "noise2": rng.normal(size=n_total),
        }
    )
    return (
        x.iloc[:n_per_split].reset_index(drop=True),
        x.iloc[n_per_split : 2 * n_per_split].reset_index(drop=True),
        x.iloc[2 * n_per_split :].reset_index(drop=True),
        y[:n_per_split],
        y[n_per_split : 2 * n_per_split],
        y[2 * n_per_split :],
    )


def test_run_model_seeds_returns_one_row_per_seed() -> None:
    x_tr, x_val, x_te, y_tr, y_val, y_te = _three_class_split()
    result = run_model_seeds(
        name="rule_based",
        model_factory=lambda s: make_rule_based(seed=s, max_depth=3),
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        x_test=x_te, y_test=y_te,
        seeds=[42, 43, 44],
    )
    assert isinstance(result, ModelRunResult)
    assert len(result.per_seed_metrics) == 3
    assert {"val_macro_f1", "test_macro_f1"}.issubset(result.per_seed_metrics.columns)


def test_run_model_seeds_writes_artefacts(tmp_path: Path) -> None:
    x_tr, x_val, x_te, y_tr, y_val, y_te = _three_class_split(n_per_split=100)
    run_model_seeds(
        name="rule_based",
        model_factory=lambda s: make_rule_based(seed=s, max_depth=3),
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        x_test=x_te, y_test=y_te,
        seeds=[42, 43],
        save_dir=tmp_path,
    )
    # Per-seed folder is created with all the expected files.
    for seed in (42, 43):
        seed_dir = tmp_path / "rule_based" / f"seed_{seed}"
        assert seed_dir.is_dir()
        assert (seed_dir / "metrics.json").exists()
        assert (seed_dir / "metadata.json").exists()
        assert (seed_dir / "predictions_val.parquet").exists()
        assert (seed_dir / "predictions_test.parquet").exists()
        assert (seed_dir / "confusion_matrix_val.csv").exists()
        assert (seed_dir / "confusion_matrix_test.csv").exists()
        # model file: either model.joblib (sklearn) or model.<ext>
        assert any(seed_dir.glob("model*"))

        with open(seed_dir / "metadata.json") as fh:
            meta = json.load(fh)
        assert meta["model"] == "rule_based"
        assert meta["seed"] == seed
        assert meta["n_features"] == x_tr.shape[1]


def test_summary_returns_mean_std_and_ci() -> None:
    x_tr, x_val, x_te, y_tr, y_val, y_te = _three_class_split()
    result = run_model_seeds(
        name="rule_based",
        model_factory=lambda s: make_rule_based(seed=s, max_depth=3),
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        x_test=x_te, y_test=y_te,
        seeds=[42, 43, 44, 45, 46],
    )
    s = result.summary("test_macro_f1")
    assert set(s.keys()) == {"mean", "std", "ci95_lo", "ci95_hi"}
    assert 0.0 <= s["mean"] <= 1.0
    assert s["ci95_lo"] <= s["mean"] <= s["ci95_hi"]


def test_feature_set_subsets_columns() -> None:
    x_tr, x_val, x_te, y_tr, y_val, y_te = _three_class_split()
    result = run_model_seeds(
        name="rule_based",
        model_factory=lambda s: make_rule_based(seed=s, max_depth=3),
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        x_test=x_te, y_test=y_te,
        seeds=[42],
        feature_set=["f_info"],
    )
    assert result.n_features == 1
    assert result.feature_set == ["f_info"]
