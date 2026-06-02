"""Rule-based IDS baseline (WBS Part 6.1; protocol section 6.1 row).

A shallow decision tree *is* a rule-based classifier: every leaf is a
conjunction of feature-threshold rules of the form
feature_a <= t_a AND feature_b > t_b AND .... The tree at the
project's default depth of 5 yields at most 32 leaves - a compact,
interpretable rule set the supervisor and a reader can audit
directly.

Why this is the project's "rule-based" baseline:

* It is interpretable - get_rules() returns each leaf as a
  human-readable conjunction with its predicted class.
* It is defensible - decision-tree leaves are the canonical
  representation of rule-based classifiers in the IDS literature.
* It is fair to compare - it learns thresholds from the training
  fold the same way every other Phase-1 model does, so it slots into
  the protocol's evaluation harness without ceremony.

Optional hand-crafted thresholds (e.g. specific to one of the 9
UNSW-NB15 attack types) can be layered on top by extending
RuleBasedIDS.predict - this is left for the refinement pass
after the headline run is in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


class RuleBasedIDS:
    """Sklearn-compatible rule-based IDS wrapper around a shallow tree.

    Parameters
    max_depth
        Tree depth. Default 5 (≤ 32 leaves - still inspectable).
    min_samples_leaf
        Minimum samples per leaf, in *fraction* of training set if a
        float, in absolute counts if int. Default 0.001 = 0.1% of train.
    seed
        Random seed forwarded to the underlying tree.
    class_weight
        "balanced" (default) handles UNSW-NB15's class imbalance.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: float | int = 0.001,
        seed: int = 42,
        class_weight: str | dict = "balanced",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.class_weight = class_weight
        self._tree: DecisionTreeClassifier | None = None
        self._feature_names: list[str] | None = None
        self._classes: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Required BaseModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        class_weights=None,
    ) -> "RuleBasedIDS":
        """Fit the shallow decision tree on the training fold."""
        cw: Any = class_weights if class_weights is not None else self.class_weight
        self._tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=cw,
            random_state=self.seed,
        )
        if isinstance(x_train, pd.DataFrame):
            self._feature_names = list(x_train.columns)
            self._tree.fit(x_train.to_numpy(), np.asarray(y_train))
        else:
            self._feature_names = [f"f{i}" for i in range(np.asarray(x_train).shape[1])]
            self._tree.fit(np.asarray(x_train), np.asarray(y_train))
        self._classes = self._tree.classes_
        return self

    def predict(self, x) -> np.ndarray:
        if self._tree is None:
            raise RuntimeError("RuleBasedIDS.predict called before fit().")
        x_arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        return self._tree.predict(x_arr)

    def predict_proba(self, x) -> np.ndarray:
        if self._tree is None:
            raise RuntimeError("RuleBasedIDS.predict_proba called before fit().")
        x_arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        return self._tree.predict_proba(x_arr)

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def get_rules(self) -> str:
        """Return the trained tree as a human-readable rule listing."""
        if self._tree is None or self._feature_names is None:
            raise RuntimeError("RuleBasedIDS.get_rules called before fit().")
        return export_text(self._tree, feature_names=self._feature_names)

    def n_leaves(self) -> int:
        if self._tree is None:
            raise RuntimeError("RuleBasedIDS.n_leaves called before fit().")
        return int(self._tree.get_n_leaves())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the wrapper (and the trained tree inside) via joblib."""
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".joblib")
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: str | Path) -> "RuleBasedIDS":
        return joblib.load(Path(path))


def make_rule_based(seed: int = 42, **hparams) -> RuleBasedIDS:
    """Factory that returns an unfit RuleBasedIDS instance.

    Used by src.evaluation.model_runner.run_model_seeds - one
    instance per seed for the multi-seed protocol section 8 evaluation.
    """
    return RuleBasedIDS(seed=seed, **hparams)
