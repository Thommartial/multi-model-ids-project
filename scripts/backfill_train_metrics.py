"""Backfill train-set metrics into existing run artefacts (Subtask 3b).

The harness now records train/val/test metrics for every run, but the
runs produced before that change only have val/test. Rather than retrain
everything, this script loads each saved model, scores the TRAIN fold,
and writes train_macro_f1 / train_balanced_accuracy (and, for
binary, train_auroc / train_pr_auc) into that run's
metrics.json.

This gives us the train-vs-test generalisation gap for every model
family - the overfitting check for the non-iterative classical models,
which (unlike the CNN/LSTM) have no epoch curve.

Usage
:

    python scripts/backfill_train_metrics.py            # fill only missing
    python scripts/backfill_train_metrics.py --force    # recompute all
    python scripts/backfill_train_metrics.py --runs-dir reports/runs

Loads the deep-learning wrappers lazily, so it only imports TensorFlow
if a CNN/LSTM run is actually encountered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.ablation import compute_metrics  # noqa: E402

PROCESSED = Path("data/processed")
LABEL_COLS = ("label", "attack_cat")
DL_MODELS = {"cnn_1d", "lstm"}


def _load_fold(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def _load_model(model_name: str, seed_dir: Path):
    """Reload a saved model wrapper from a seed directory."""
    joblib_path = seed_dir / "model.joblib"
    model_dir = seed_dir / "model"
    if joblib_path.exists():
        return joblib.load(joblib_path)
    if model_dir.is_dir():
        if model_name == "cnn_1d":
            from src.models.deep_learning import OneDCNN

            return OneDCNN.load(model_dir)
        if model_name == "lstm":
            from src.models.deep_learning import LSTMClassifier

            return LSTMClassifier.load(model_dir)
        # classical wrappers that saved into a directory
        from src.models.classical_ml import load_model

        return load_model(model_dir)
    raise FileNotFoundError(f"no saved model under {seed_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="reports/runs")
    ap.add_argument("--force", action="store_true", help="recompute even if train_* already present")
    ap.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict to these model folders (e.g. --models xgboost). Default: all.",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_dir)
    # Cache the train fold per task so we load each parquet once.
    train_cache: dict[str, pd.DataFrame] = {}

    n_done, n_skip, n_err = 0, 0, 0
    for metrics_file in sorted(runs_root.rglob("metrics.json")):
        seed_dir = metrics_file.parent
        rel = seed_dir.relative_to(runs_root).parts
        model_name, task = rel[0], rel[1]
        if args.models is not None and model_name not in args.models:
            continue
        target = "attack_cat" if task == "multiclass" else "label"

        with open(metrics_file) as fh:
            metrics = json.load(fh)
        if "train_macro_f1" in metrics and not args.force:
            n_skip += 1
            continue

        try:
            if task not in train_cache:
                train_cache[task] = _load_fold("train")
            train = train_cache[task]
            x_train = train.drop(columns=list(LABEL_COLS))
            y_train = train[target].to_numpy()

            model = _load_model(model_name, seed_dir)
            pred_train = model.predict(x_train)

            proba_train = None
            if task == "binary" and hasattr(model, "predict_proba"):
                try:
                    p = model.predict_proba(x_train)
                    if p is not None and p.ndim == 2 and p.shape[1] == 2:
                        proba_train = p[:, 1]
                except Exception:
                    proba_train = None

            m = compute_metrics(y_train, pred_train, y_proba=proba_train, task=task)
            for k, v in m.items():
                metrics[f"train_{k}"] = float(v)
            with open(metrics_file, "w") as fh:
                json.dump(metrics, fh, indent=2)
            n_done += 1
            print(f"[ok]   {'/'.join(rel)}  train_macro_f1={metrics['train_macro_f1']:.4f}")
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            print(f"[err]  {'/'.join(rel)}: {exc}")

    print(f"\nbackfilled={n_done}  skipped(existing)={n_skip}  errors={n_err}")


if __name__ == "__main__":
    main()
