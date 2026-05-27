"""Shared per-model training harness for Phase-1 models (Part 6).

A single training-and-evaluation loop that every Part-6 model uses, so the
seeds, the metrics, the confusion matrices, and the on-disk artefact
layout are identical across model families. This is the protocol §8 +
§12 surface for the project's six Phase-1 models (rule-based, RF, SVM,
XGBoost, 1D-CNN, LSTM).

Convention
----------
Each model is exposed as a *factory* function ``seed -> model`` where
``model`` follows :class:`BaseModel`:

* ``fit(X_train, y_train, X_val=None, y_val=None, class_weights=None)``
* ``predict(X) -> array``
* ``predict_proba(X) -> array`` (optional; used for binary AUROC / PR-AUC)
* ``save(path)`` (optional; falls back to ``joblib.dump``)

For sklearn-style models the wrapper is a thin shim that ignores
``X_val``; for Keras models the validation fold flows into early
stopping and best-on-val checkpointing.

Artefacts per (model × seed)
----------------------------
Written when ``save_dir`` is set, per protocol §12.2:

* ``model.joblib`` (or whatever the wrapper's own ``save`` writes)
* ``predictions_val.parquet`` and ``predictions_test.parquet``
* ``metrics.json``
* ``confusion_matrix_val.csv`` and ``confusion_matrix_test.csv``
* ``metadata.json`` (seed, n_features, runtime, timestamp, task)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.evaluation.ablation import bootstrap_ci, compute_metrics
from src.utils.reproducibility import DEFAULT_SEED

DEFAULT_SEEDS: list[int] = [42, 43, 44, 45, 46]


class BaseModel(Protocol):
    """Common interface every Phase-1 model wrapper implements."""

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        class_weights=None,
    ) -> None: ...

    def predict(self, x) -> np.ndarray: ...


@dataclass
class ModelRunResult:
    """Output of running one model across the protocol's seed list."""

    name: str
    per_seed_metrics: pd.DataFrame
    confusion_val: dict[int, np.ndarray]
    confusion_test: dict[int, np.ndarray]
    feature_set: list[str]
    n_features: int
    runtime_seconds: float
    task: str = "multiclass"

    def summary(self, metric: str = "test_macro_f1") -> dict[str, float]:
        """Mean, std, and 95% bootstrap CI for ``metric`` across seeds."""
        values = self.per_seed_metrics[metric].to_numpy()
        lo, hi = bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=DEFAULT_SEED)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
        }


def _balanced_class_weights(y) -> dict:
    """Inverse-frequency class weights, matching ``imbalance_handling``."""
    from src.data.imbalance_handling import compute_class_weights

    return compute_class_weights(y)


def _save_seed_artefacts(
    seed_dir: Path,
    name: str,
    seed: int,
    task: str,
    model,
    y_val,
    y_test,
    pred_val,
    pred_test,
    metrics_row: dict,
    cm_val: np.ndarray,
    cm_test: np.ndarray,
    classes: list,
    seed_runtime: float,
    n_features: int,
) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)

    # model -- delegate to wrapper if it knows how to save itself
    if hasattr(model, "save") and callable(model.save):
        try:
            model.save(str(seed_dir / "model"))
        except Exception:
            joblib.dump(model, seed_dir / "model.joblib")
    else:
        joblib.dump(model, seed_dir / "model.joblib")

    pd.DataFrame({"y_true": np.asarray(y_val), "y_pred": np.asarray(pred_val)}).to_parquet(
        seed_dir / "predictions_val.parquet"
    )
    pd.DataFrame({"y_true": np.asarray(y_test), "y_pred": np.asarray(pred_test)}).to_parquet(
        seed_dir / "predictions_test.parquet"
    )

    with open(seed_dir / "metrics.json", "w") as fh:
        json.dump({k: float(v) for k, v in metrics_row.items() if k != "seed"}, fh, indent=2)

    pd.DataFrame(cm_val, index=classes, columns=classes).to_csv(seed_dir / "confusion_matrix_val.csv")
    pd.DataFrame(cm_test, index=classes, columns=classes).to_csv(seed_dir / "confusion_matrix_test.csv")

    with open(seed_dir / "metadata.json", "w") as fh:
        json.dump(
            {
                "model": name,
                "seed": int(seed),
                "task": task,
                "n_features": int(n_features),
                "runtime_seconds": float(seed_runtime),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            fh,
            indent=2,
        )


def run_model_seeds(
    name: str,
    model_factory: Callable[[int], BaseModel],
    x_train: pd.DataFrame,
    y_train,
    x_val: pd.DataFrame,
    y_val,
    x_test: pd.DataFrame,
    y_test,
    *,
    seeds: list[int] = DEFAULT_SEEDS,
    task: str = "multiclass",
    feature_set: list[str] | None = None,
    save_dir: Path | str | None = None,
    use_class_weights: bool = True,
    verbose: bool = False,
) -> ModelRunResult:
    """Train one model factory across ``seeds`` and write the per-seed artefacts.

    Parameters
    ----------
    name
        Folder name under ``save_dir`` (e.g. ``"rule_based"``).
    model_factory
        Callable ``(seed) -> unfit model implementing :class:`BaseModel```.
    x_train, y_train, x_val, y_val, x_test, y_test
        The three folds. ``y`` may be strings (multiclass) or ints (binary).
    seeds
        Protocol §8 default list ``[42, 43, 44, 45, 46]``.
    task
        ``"multiclass"`` or ``"binary"``.
    feature_set
        Subset of columns to train on. ``None`` = all of ``x_train``.
    save_dir
        Root directory for per-seed artefacts (``reports/runs/`` is the
        protocol's canonical location). ``None`` = do not save.
    use_class_weights
        Compute inverse-frequency class weights from ``y_train`` and pass
        to ``model.fit`` via ``class_weights=``.
    verbose
        Print a per-seed progress line.
    """
    if feature_set is None:
        feature_set = list(x_train.columns)

    class_weights = _balanced_class_weights(y_train) if use_class_weights else None
    classes = sorted({*np.asarray(y_train).tolist()})

    save_dir_path = Path(save_dir) if save_dir is not None else None

    per_seed_rows: list[dict] = []
    confusion_val: dict[int, np.ndarray] = {}
    confusion_test: dict[int, np.ndarray] = {}

    t0 = time.time()
    for seed in seeds:
        model = model_factory(seed)
        # Per-seed root: includes task so a model trained on multiclass
        # cannot overwrite the same model trained on binary (or vice versa).
        seed_root = save_dir_path / name / task / f"seed_{seed}" if save_dir_path is not None else None
        # If the wrapper supports TensorBoard logging and we have a save_dir,
        # write logs alongside the rest of this seed's artefacts so launching
        # `tensorboard --logdir reports/runs` shows every model and seed.
        if seed_root is not None and hasattr(model, "set_tensorboard_dir"):
            model.set_tensorboard_dir(str(seed_root / "tensorboard"))
        seed_t0 = time.time()
        model.fit(
            x_train[feature_set],
            y_train,
            x_val=x_val[feature_set],
            y_val=y_val,
            class_weights=class_weights,
        )
        seed_runtime = time.time() - seed_t0

        pred_val = model.predict(x_val[feature_set])
        pred_test = model.predict(x_test[feature_set])

        proba_val, proba_test = None, None
        if task == "binary" and hasattr(model, "predict_proba"):
            try:
                pv = model.predict_proba(x_val[feature_set])
                pt = model.predict_proba(x_test[feature_set])
                if pv is not None and pv.ndim == 2 and pv.shape[1] == 2:
                    proba_val = pv[:, 1]
                    proba_test = pt[:, 1]
            except Exception:
                pass

        row: dict[str, Any] = {"seed": int(seed), "runtime_s": seed_runtime}
        for split_name, y_true, y_pred, y_proba in (
            ("val", y_val, pred_val, proba_val),
            ("test", y_test, pred_test, proba_test),
        ):
            m = compute_metrics(y_true, y_pred, y_proba=y_proba, task=task)
            for k, v in m.items():
                row[f"{split_name}_{k}"] = float(v)

        confusion_val[seed] = confusion_matrix(y_val, pred_val, labels=classes)
        confusion_test[seed] = confusion_matrix(y_test, pred_test, labels=classes)
        per_seed_rows.append(row)

        if verbose:
            print(
                f"[{name}] seed={seed:>3d}  "
                f"val_macro_f1={row['val_macro_f1']:.4f}  "
                f"test_macro_f1={row['test_macro_f1']:.4f}  "
                f"({seed_runtime:.1f}s)"
            )

        if seed_root is not None:
            _save_seed_artefacts(
                seed_dir=seed_root,
                name=name,
                seed=int(seed),
                task=task,
                model=model,
                y_val=y_val,
                y_test=y_test,
                pred_val=pred_val,
                pred_test=pred_test,
                metrics_row=row,
                cm_val=confusion_val[seed],
                cm_test=confusion_test[seed],
                classes=classes,
                seed_runtime=seed_runtime,
                n_features=len(feature_set),
            )

    return ModelRunResult(
        name=name,
        per_seed_metrics=pd.DataFrame(per_seed_rows).set_index("seed").sort_index(),
        confusion_val=confusion_val,
        confusion_test=confusion_test,
        feature_set=list(feature_set),
        n_features=len(feature_set),
        runtime_seconds=time.time() - t0,
        task=task,
    )
