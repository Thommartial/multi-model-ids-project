"""Unit tests for src.models.rule_based."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.rule_based import RuleBasedIDS, make_rule_based


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


def test_fit_predict_returns_array_of_correct_length() -> None:
    x, y = _toy_dataset()
    rb = RuleBasedIDS(max_depth=4, seed=42).fit(x, y)
    preds = rb.predict(x)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(x)


def test_predict_before_fit_raises() -> None:
    rb = RuleBasedIDS()
    with pytest.raises(RuntimeError, match="before fit"):
        rb.predict(np.zeros((3, 4)))


def test_predict_proba_shape() -> None:
    x, y = _toy_dataset()
    rb = RuleBasedIDS(max_depth=4, seed=42).fit(x, y)
    proba = rb.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_get_rules_returns_string_after_fit() -> None:
    x, y = _toy_dataset()
    rb = RuleBasedIDS(max_depth=3, seed=42).fit(x, y)
    rules = rb.get_rules()
    assert isinstance(rules, str)
    # the trained tree should reference at least one feature by name
    assert any(col in rules for col in x.columns)


def test_n_leaves_at_most_2_to_depth() -> None:
    x, y = _toy_dataset(n=400)
    rb = RuleBasedIDS(max_depth=4, seed=42).fit(x, y)
    assert rb.n_leaves() <= 2**4


def test_make_rule_based_factory_takes_seed() -> None:
    rb = make_rule_based(seed=7, max_depth=3)
    assert rb.seed == 7
    assert rb.max_depth == 3


def test_fit_returns_self_and_classifies_informative_data() -> None:
    x, y = _toy_dataset(n=600)
    rb = RuleBasedIDS(max_depth=5, seed=42)
    out = rb.fit(x, y)
    assert out is rb
    # Should beat random on a dataset where y depends on two of the features.
    preds = rb.predict(x)
    acc = (preds == y).mean()
    assert acc > 0.7


def test_save_and_load_roundtrip(tmp_path) -> None:
    x, y = _toy_dataset()
    rb = RuleBasedIDS(max_depth=3, seed=42).fit(x, y)
    pred_before = rb.predict(x)
    rb.save(tmp_path / "rb_model")
    loaded = RuleBasedIDS.load(tmp_path / "rb_model.joblib")
    pred_after = loaded.predict(x)
    assert np.array_equal(pred_before, pred_after)


def test_class_weights_kwarg_forwards_to_tree() -> None:
    """If class_weights dict is passed to fit, it should be used instead of the constructor default."""
    x, y = _toy_dataset()
    rb = RuleBasedIDS(max_depth=3, seed=42)
    rb.fit(x, y, class_weights={0: 1.0, 1: 5.0})
    # smoke: it should not error and should still predict
    preds = rb.predict(x)
    assert len(preds) == len(x)
