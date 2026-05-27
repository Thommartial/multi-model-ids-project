"""Universal runner for Phase-1 models (Part 6).

Trains one model family across the protocol's seed list, computes
val + test metrics, and writes the §12.2 artefacts under
``reports/runs/<model>/seed_<n>/``.

The same command shape works for every model. Pick the model with
``--model``; pick the task with ``--task``.

Examples
--------
::

    # Interpretable rule-based baseline -- fastest model, fastest result.
    python scripts/run_model.py --model rule_based

    # Random Forest, multiclass.
    python scripts/run_model.py --model random_forest

    # SVM (uses the protocol §9 30k subsample internally).
    python scripts/run_model.py --model svm

    # XGBoost on the binary task.
    python scripts/run_model.py --model xgboost --task binary

    # 1-D CNN -- needs a GPU to be reasonable; works on CPU but slow.
    python scripts/run_model.py --model cnn_1d

    # LSTM (uses an internal sliding window of length 8 by default).
    python scripts/run_model.py --model lstm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# allow running as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.model_runner import DEFAULT_SEEDS, run_model_seeds  # noqa: E402
from src.models.classical_ml import get_random_forest, get_svm, get_xgboost  # noqa: E402
from src.models.rule_based import make_rule_based  # noqa: E402

PROCESSED = Path("data/processed")
RUNS_DIR = Path("reports/runs")
LABEL_COLS = ("label", "attack_cat")


def _load_fold(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def _get_factory(name: str, task: str):
    """Return the factory function for the requested model.

    Imports of the deep-learning module are lazy because they pull in
    TensorFlow.
    """
    if name == "rule_based":
        return lambda seed: make_rule_based(seed=seed, max_depth=5)
    if name == "random_forest":
        return lambda seed: get_random_forest(seed=seed)
    if name == "svm":
        return lambda seed: get_svm(seed=seed)
    if name == "xgboost":
        return lambda seed: get_xgboost(seed=seed)
    if name == "cnn_1d":
        from src.models.deep_learning import get_onedcnn

        return lambda seed: get_onedcnn(seed=seed)
    if name == "lstm":
        from src.models.deep_learning import get_lstm

        return lambda seed: get_lstm(seed=seed)
    raise ValueError(f"unknown model: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=["rule_based", "random_forest", "svm", "xgboost", "cnn_1d", "lstm"],
    )
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Override the protocol's default seed list.",
    )
    parser.add_argument(
        "--save-dir", default=str(RUNS_DIR),
        help="Root directory for per-seed artefacts.",
    )
    parser.add_argument(
        "--no-class-weights", action="store_true",
        help="Train without inverse-frequency class weights.",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Configure the GPU (memory growth) before training the DL models.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    target = "attack_cat" if args.task == "multiclass" else "label"
    print(f"[run_model] model={args.model}  task={args.task}  target={target}")

    if args.gpu and args.model in {"cnn_1d", "lstm"}:
        from src.utils.gpu_management import configure_gpu, gpu_summary

        configure_gpu(memory_growth=True)
        print(f"[run_model] GPU: {gpu_summary()}")

    print(f"[run_model] loading data/processed/{{train, val, test}}.parquet")
    train = _load_fold("train")
    val = _load_fold("val")
    test = _load_fold("test")
    x_train = train.drop(columns=list(LABEL_COLS))
    x_val = val.drop(columns=list(LABEL_COLS))
    x_test = test.drop(columns=list(LABEL_COLS))
    y_train = train[target].to_numpy()
    y_val = val[target].to_numpy()
    y_test = test[target].to_numpy()
    print(f"[run_model] X_train={x_train.shape}  X_val={x_val.shape}  X_test={x_test.shape}")

    factory = _get_factory(args.model, args.task)
    result = run_model_seeds(
        name=args.model,
        model_factory=factory,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        seeds=args.seeds,
        task=args.task,
        save_dir=args.save_dir,
        use_class_weights=not args.no_class_weights,
        verbose=not args.quiet,
    )

    summary = result.summary("test_macro_f1")
    print()
    print(f"[run_model] {args.model} / test macro-F1: "
          f"{summary['mean']:.4f} ± {summary['std']:.4f}  "
          f"(95% CI [{summary['ci95_lo']:.4f}, {summary['ci95_hi']:.4f}])")
    print(f"[run_model] per-seed artefacts saved under {args.save_dir}/{args.model}/{args.task}/")


if __name__ == "__main__":
    main()
