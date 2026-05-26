"""Unit tests for :mod:`src.models.classical_ml`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.classical_ml import (
    RandomForestModel,
    SVMModel,
    XGBoostModel,
    get_random_forest,
    get_svm,
    get_xgboost,
    load_hparam_space,
    load_model,
    save_model,
    _XGBOOST_AVAILABLE,
)


def _toy_dataset(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y = ((f1 + f2) > 0).astype(int)
    x = pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
        }
    )
    return x, y


# ---------- load_hparam_space ----------


def test_load_hparam_space_returns_dict_for_known_model() -> None:
    space = load_hparam_space("random_forest")
    assert isinstance(space, dict)
    assert "n_estimators" in space


def test_load_hparam_space_raises_for_unknown_model() -> None:
    with pytest.raises(FileNotFoundError):
        load_hparam_space("not_a_model")


# ---------- RandomForestModel ----------


def test_random_forest_fit_predict_shape() -> None:
    x, y = _toy_dataset()
    rf = get_random_forest(seed=42, n_estimators=20, max_depth=4)
    rf.fit(x, y)
    preds = rf.predict(x)
    assert preds.shape == (len(x),)
    proba = rf.predict_proba(x)
    assert proba.shape == (len(x), 2)


def test_random_forest_predict_before_fit_raises() -> None:
    rf = RandomForestModel()
    with pytest.raises(RuntimeError, match="before fit"):
        rf.predict(np.zeros((3, 4)))


def test_random_forest_class_weights_kwarg_forwards() -> None:
    x, y = _toy_dataset()
    rf = get_random_forest(seed=42, n_estimators=20, max_depth=4)
    rf.fit(x, y, class_weights={0: 1.0, 1: 5.0})
    assert rf.predict(x).shape == (len(x),)


# ---------- SVMModel ----------


def test_svm_fit_predict_with_internal_scaling() -> None:
    x, y = _toy_dataset(n=200)
    # 200 < default subsample 30k -> no subsampling, scaler still fitted
    svm = get_svm(seed=42, subsample=None, probability=True)
    svm.fit(x, y)
    preds = svm.predict(x)
    assert preds.shape == (len(x),)
    proba = svm.predict_proba(x)
    assert proba.shape == (len(x), 2)


def test_svm_subsample_path_triggers_when_n_exceeds_cap() -> None:
    x, y = _toy_dataset(n=400)
    svm = get_svm(seed=42, subsample=100, probability=True)
    svm.fit(x, y)
    # smoke: predicts after the subsample-fit
    assert svm.predict(x).shape == (len(x),)


# ---------- XGBoostModel ----------


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="xgboost not installed in test env")
def test_xgboost_fit_predict() -> None:
    x, y = _toy_dataset()
    xgb = get_xgboost(seed=42, n_estimators=30, max_depth=4)
    xgb.fit(x, y)
    preds = xgb.predict(x)
    assert preds.shape == (len(x),)


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="xgboost not installed in test env")
def test_xgboost_handles_class_weights_as_sample_weights() -> None:
    x, y = _toy_dataset()
    xgb = get_xgboost(seed=42, n_estimators=20, max_depth=3)
    xgb.fit(x, y, class_weights={0: 1.0, 1: 4.0})
    assert xgb.predict(x).shape == (len(x),)


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="xgboost not installed in test env")
def test_xgboost_string_labels_roundtrip_through_label_encoder() -> None:
    x, y = _toy_dataset()
    y_str = np.where(y == 1, "attack", "normal")
    xgb = get_xgboost(seed=42, n_estimators=20, max_depth=3)
    xgb.fit(x, y_str)
    preds = xgb.predict(x)
    assert set(preds.tolist()) <= {"attack", "normal"}


# ---------- save_model / load_model roundtrip ----------


def test_random_forest_save_load_roundtrip(tmp_path: Path) -> None:
    x, y = _toy_dataset()
    rf = get_random_forest(seed=42, n_estimators=20, max_depth=4).fit(x, y)
    pred_before = rf.predict(x)
    save_model(rf, tmp_path / "rf_model")
    loaded = load_model(tmp_path / "rf_model.joblib")
    assert np.array_equal(pred_before, loaded.predict(x))


def test_svm_save_load_roundtrip(tmp_path: Path) -> None:
    x, y = _toy_dataset(n=150)
    svm = get_svm(seed=42, subsample=None, probability=True).fit(x, y)
    pred_before = svm.predict(x)
    save_model(svm, tmp_path / "svm_model")
    loaded = load_model(tmp_path / "svm_model.joblib")
    assert np.array_equal(pred_before, loaded.predict(x))
