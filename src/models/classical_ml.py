"""Classical ML model wrappers for Phase 1 (BaseModel protocol).

RandomForestModel: sklearn RF, class-weighted, unscaled inputs.
SVMModel: sklearn SVC (RBF), class-weighted, stratified subsample plus an
internal StandardScaler fitted on the subsample.
XGBoostModel: XGBClassifier with sample weights from the class weights.

Hyperparameter spaces load from configs/hparam_spaces/<model>.yaml.
optimize_hyperparameters runs GridSearchCV; Optuna lives in src/optimization.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:  # XGBoost is in requirements.txt but keep the import optional for tests
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    XGBClassifier = None  # type: ignore[assignment, misc]
    _XGBOOST_AVAILABLE = False

from src.utils.reproducibility import DEFAULT_SEED

CONFIGS_DIR = Path("configs/hparam_spaces")


def load_hparam_space(name: str) -> dict:
    """Load configs/hparam_spaces/<name>.yaml as a dict."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"hyperparameter spec not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


class RandomForestModel:
    """Random Forest wrapper (BaseModel protocol)."""

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",
        seed: int = DEFAULT_SEED,
        n_jobs: int = -1,
        class_weight: str | dict = "balanced",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self._rf: RandomForestClassifier | None = None

    def fit(self, x_train, y_train, x_val=None, y_val=None, class_weights=None):
        cw: Any = class_weights if class_weights is not None else self.class_weight
        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            class_weight=cw,
        )
        x = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else np.asarray(x_train)
        self._rf.fit(x, np.asarray(y_train))
        return self

    def predict(self, x):
        if self._rf is None:
            raise RuntimeError("predict before fit")
        return self._rf.predict(x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x))

    def predict_proba(self, x):
        if self._rf is None:
            raise RuntimeError("predict_proba before fit")
        return self._rf.predict_proba(
            x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".joblib")
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestModel":
        return joblib.load(Path(path))


def get_random_forest(seed: int = DEFAULT_SEED, **hparams) -> RandomForestModel:
    """Factory for the harness. Hparams override the defaults."""
    return RandomForestModel(seed=seed, **hparams)


class SVMModel:
    """SVC (RBF kernel) wrapper.

    Train folds larger than subsample rows (default 30,000) are stratified
    down before fitting; validation and test are never subsampled. Features
    are standardised with a StandardScaler fitted on the subsample only (RBF
    distance is scale-sensitive) and the same scaler is applied at predict.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        subsample: int | None = 30_000,
        seed: int = DEFAULT_SEED,
        class_weight: str | dict = "balanced",
        probability: bool = True,
    ) -> None:
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.subsample = subsample
        self.seed = seed
        self.class_weight = class_weight
        self.probability = probability
        self._svm: SVC | None = None
        self._scaler: StandardScaler | None = None

    def fit(self, x_train, y_train, x_val=None, y_val=None, class_weights=None):
        x = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else np.asarray(x_train)
        y = np.asarray(y_train)

        if self.subsample is not None and len(x) > self.subsample:
            _, x, _, y = train_test_split(
                x,
                y,
                test_size=self.subsample,
                stratify=y,
                random_state=self.seed,
            )

        self._scaler = StandardScaler().fit(x)
        x_scaled = self._scaler.transform(x)

        cw: Any = class_weights if class_weights is not None else self.class_weight
        self._svm = SVC(
            C=self.C,
            gamma=self.gamma,
            kernel=self.kernel,
            class_weight=cw,
            random_state=self.seed,
            probability=self.probability,
        )
        self._svm.fit(x_scaled, y)
        return self

    def _transform(self, x):
        arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        if self._scaler is None:
            raise RuntimeError("transform before fit")
        return self._scaler.transform(arr)

    def predict(self, x):
        if self._svm is None:
            raise RuntimeError("predict before fit")
        return self._svm.predict(self._transform(x))

    def predict_proba(self, x):
        if self._svm is None:
            raise RuntimeError("predict_proba before fit")
        return self._svm.predict_proba(self._transform(x))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".joblib")
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: str | Path) -> "SVMModel":
        return joblib.load(Path(path))


def get_svm(seed: int = DEFAULT_SEED, **hparams) -> SVMModel:
    return SVMModel(seed=seed, **hparams)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class XGBoostModel:
    """XGBoost wrapper with class-weighted sample weights."""

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        seed: int = DEFAULT_SEED,
        n_jobs: int = -1,
        eval_metric: str = "mlogloss",
        tree_method: str = "hist",
    ) -> None:
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. `pip install xgboost` (or use the " "project conda env)."
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.seed = seed
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self._xgb: Any = None
        self._label_encoder: dict | None = None

    def _encode(self, y):
        """Map y to integer ids; xgboost wants 0..K-1 for multiclass."""
        y_arr = np.asarray(y)
        if self._label_encoder is None:
            classes = sorted(set(y_arr.tolist()))
            self._label_encoder = {c: i for i, c in enumerate(classes)}
        return np.array([self._label_encoder[c] for c in y_arr])

    def _decode(self, y_int):
        if self._label_encoder is None:
            return y_int
        inv = {i: c for c, i in self._label_encoder.items()}
        return np.array([inv[i] for i in y_int])

    def fit(self, x_train, y_train, x_val=None, y_val=None, class_weights=None):
        x = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else np.asarray(x_train)
        y_enc = self._encode(y_train)

        sample_weight = None
        if class_weights is not None:
            sample_weight = np.array(
                [class_weights[c] for c in np.asarray(y_train)],
                dtype=float,
            )

        self._xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
        )
        self._xgb.fit(x, y_enc, sample_weight=sample_weight)
        return self

    def predict(self, x):
        if self._xgb is None:
            raise RuntimeError("predict before fit")
        arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        return self._decode(self._xgb.predict(arr))

    def predict_proba(self, x):
        if self._xgb is None:
            raise RuntimeError("predict_proba before fit")
        arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        return self._xgb.predict_proba(arr)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".joblib")
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        return joblib.load(Path(path))


def get_xgboost(seed: int = DEFAULT_SEED, **hparams) -> XGBoostModel:
    return XGBoostModel(seed=seed, **hparams)


# ---------------------------------------------------------------------------
# Universal save / load
# ---------------------------------------------------------------------------


def save_model(model, path: str | Path) -> None:
    """Save any BaseModel wrapper; delegates to its save or to joblib."""
    if hasattr(model, "save"):
        model.save(path)
    else:
        joblib.dump(model, Path(path))


def load_model(path: str | Path):
    """Load a wrapper saved with save_model."""
    return joblib.load(Path(path))


# ---------------------------------------------------------------------------
# Hyperparameter optimisation (WBS 6.2.4)
# ---------------------------------------------------------------------------


def _grid_from_yaml_space(space: dict) -> dict[str, list]:
    """Coerce a YAML hyperparameter space into a sklearn-style grid.

    A YAML entry that is a plain list ([a, b, c]) is used as-is.
    Entries that are dicts with type/choices/low/high are
    not appropriate for grid search and are ignored here (they are
    intended for Optuna in the optimization module).
    """
    grid: dict[str, list] = {}
    for k, v in space.items():
        if isinstance(v, list):
            grid[k] = v
        elif isinstance(v, dict) and "choices" in v:
            grid[k] = list(v["choices"])
    return grid


def optimize_hyperparameters(
    estimator,
    space: dict,
    x_train,
    y_train,
    *,
    cv: int = 3,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    verbose: int = 0,
):
    """Grid-search the sklearn estimator over space.

    Returns the fitted GridSearchCV - the caller can read
    best_estimator_, best_params_, and cv_results_. Suitable
    for the Random Forest and SVM families (WBS 6.2.4 / protocol section 9).
    """
    grid = _grid_from_yaml_space(space)
    if not grid:
        raise ValueError(
            "no list-valued hyperparameters in the space -- not suitable for "
            "grid search. Use the Optuna-style optimiser for this space."
        )
    gs = GridSearchCV(
        estimator,
        grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    gs.fit(x_train, y_train)
    return gs


def train_with_cv(
    model_class,
    space: dict | None,
    x_train,
    y_train,
    *,
    cv: int = 3,
    scoring: str = "f1_macro",
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
):
    """Convenience: build an unfit estimator + grid-search.

    If space is None or list-empty, the estimator is fitted with its
    defaults and returned as-is. Otherwise grid search is run and the
    refit best estimator is returned.
    """
    estimator = model_class(seed=seed)
    if not space:
        estimator.fit(x_train, y_train)
        return estimator, None
    inner_grid = _grid_from_yaml_space(space)
    if not inner_grid:
        estimator.fit(x_train, y_train)
        return estimator, None
    gs = optimize_hyperparameters(
        estimator,
        space,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    return gs.best_estimator_, gs
