"""Smoke tests for src.models.deep_learning.

These tests train tiny models for very few epochs to verify the
fit/predict/save/load contract. They are still slow relative to the
sklearn tests (~30-60 seconds) but still in the project's budget.
The tests are skipped automatically if TensorFlow is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import tensorflow as tf  # noqa: F401

    _TF_AVAILABLE = True
except Exception:  # pragma: no cover
    _TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not available")


def _toy_two_class(n: int = 200, n_features: int = 8, seed: int = 0):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, n_features)).astype("float32")
    y = (x.sum(axis=1) > 0).astype(int)
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(n_features)]), y


# - 1-D CNN ---------------------------------------------------------------


def test_onedcnn_fit_predict_returns_one_per_row() -> None:
    from src.models.deep_learning import get_onedcnn

    x, y = _toy_two_class()
    model = get_onedcnn(
        seed=42,
        conv_blocks=1,
        filters_per_block=8,
        kernel_size=3,
        dense_units=16,
        max_epochs=2,
        batch_size=32,
    )
    model.fit(x.iloc[:150], y[:150], x_val=x.iloc[150:], y_val=y[150:])
    preds = model.predict(x)
    assert preds.shape == (len(x),)
    proba = model.predict_proba(x)
    assert proba.shape == (len(x), 2)


def test_onedcnn_predict_before_fit_raises() -> None:
    from src.models.deep_learning import get_onedcnn

    with pytest.raises(RuntimeError, match="before fit"):
        get_onedcnn(seed=0).predict_proba(np.zeros((3, 5)))


def test_onedcnn_save_load_roundtrip(tmp_path: Path) -> None:
    from src.models.deep_learning import OneDCNN, get_onedcnn

    x, y = _toy_two_class(n=120)
    model = get_onedcnn(
        seed=42,
        conv_blocks=1,
        filters_per_block=8,
        kernel_size=3,
        dense_units=16,
        max_epochs=1,
        batch_size=32,
    )
    model.fit(x, y)
    pred_before = model.predict(x)
    model.save(tmp_path / "cnn")
    loaded = OneDCNN.load(tmp_path / "cnn")
    pred_after = loaded.predict(x)
    assert np.array_equal(pred_before, pred_after)


# - LSTM ------------------------------------------------------------------


def test_lstm_fit_predict_pads_to_input_length() -> None:
    from src.models.deep_learning import get_lstm

    x, y = _toy_two_class(n=200)
    model = get_lstm(
        seed=42,
        window_size=4,
        lstm_layers=1,
        hidden_size=16,
        max_epochs=2,
        batch_size=32,
    )
    model.fit(x.iloc[:150], y[:150], x_val=x.iloc[150:], y_val=y[150:])
    preds = model.predict(x)
    # output length == input length (LSTM pads the first window_size-1 rows)
    assert preds.shape == (len(x),)


def test_lstm_raises_on_too_few_rows() -> None:
    from src.models.deep_learning import get_lstm

    x, y = _toy_two_class(n=5)
    model = get_lstm(
        seed=42, window_size=10, lstm_layers=1, hidden_size=8, max_epochs=1, batch_size=8
    )
    with pytest.raises(ValueError, match="at least"):
        model.fit(x, y)
