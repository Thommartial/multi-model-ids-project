"""Run Recursive Feature Addition on the train fold (Part 5.2).

Two variants, settled with the supervisor on 2026-05-22 (see
docs/rfa_bigram_spec.md section 7):

* --variant rf   - the proposal's RF / val-macro-F1 forward selection.
* --variant svm  - the original Hamed, Dara & Kremer (2018) SVM
  cost-function RFA (binary target).
* --variant both - run both and save them side by side.

Outputs go under reports/feature_selection/ - ranking CSV, selection-path
CSV, hyperparameters JSON, and the selection-path plot.

Examples
:

    # Default: full RF/F1 RFA on attack_cat, plus the SVM RFA on label
    python scripts/run_rfa.py

    # Just the RF variant, capped at 30 features (faster, useful for a demo)
    python scripts/run_rfa.py --variant rf --max-features 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# allow running as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rfa import (  # noqa: E402
    plot_selection_path,
    rfa_random_forest,
    rfa_svm_cost_function,
)

PROCESSED = Path("data/processed")
OUT_DIR = Path("reports/feature_selection")
LABEL_COLS = ("label", "attack_cat")


def _load_fold(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def _save_result(result, tag: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_path = OUT_DIR / f"rfa_ranking_{tag}.csv"
    path_path = OUT_DIR / f"rfa_selection_path_{tag}.csv"
    hp_path = OUT_DIR / f"rfa_hyperparams_{tag}.json"
    plot_path = OUT_DIR / f"rfa_selection_path_{tag}.png"

    result.ranking.to_csv(ranking_path, index=False)
    result.selection_path.to_csv(path_path, index=False)
    with open(hp_path, "w") as fh:
        json.dump(
            {
                "method": result.method,
                "hyperparams": result.hyperparams,
                "runtime_seconds": result.runtime_seconds,
                "n_selected": len(result.selected),
                "selected": result.selected,
            },
            fh,
            indent=2,
        )
    plot_selection_path(result, save_path=str(plot_path))
    print(f"[rfa] saved ranking      -> {ranking_path}")
    print(f"[rfa] saved path         -> {path_path}")
    print(f"[rfa] saved hyperparams  -> {hp_path}")
    print(f"[rfa] saved plot         -> {plot_path}")


def _run_rf(max_features: int | None, seed: int, verbose: bool) -> None:
    print("=" * 72)
    print("RFA -- proposal variant (Random Forest + val macro-F1)")
    print("=" * 72)
    train = _load_fold("train")
    val = _load_fold("val")
    x_tr = train.drop(columns=list(LABEL_COLS))
    x_val = val.drop(columns=list(LABEL_COLS))
    y_tr = train["attack_cat"].to_numpy()
    y_val = val["attack_cat"].to_numpy()
    print(f"[rfa-rf] X_train={x_tr.shape}, X_val={x_val.shape}, "
          f"target=attack_cat ({pd.Series(y_tr).nunique()} classes)")
    result = rfa_random_forest(
        x_tr, y_tr, x_val, y_val,
        max_features=max_features,
        seed=seed,
        verbose=verbose,
    )
    print()
    print(f"[rfa-rf] done in {result.runtime_seconds:.1f}s; selected {len(result.selected)} features")
    print(f"[rfa-rf] top 20: {result.selected[:20]}")
    _save_result(result, "rf_macro_f1_attack_cat")


def _run_svm(max_features: int | None, subsample: int, seed: int, verbose: bool) -> None:
    print("=" * 72)
    print("RFA -- original (Hamed et al. 2018; SVM cost-function approximation)")
    print("=" * 72)
    train = _load_fold("train")
    x_tr = train.drop(columns=list(LABEL_COLS))
    y_tr = train["label"].to_numpy()
    print(f"[rfa-svm] X_train={x_tr.shape}, target=label (binary), "
          f"subsample={subsample}")
    result = rfa_svm_cost_function(
        x_tr, y_tr,
        max_features=max_features,
        subsample=subsample,
        seed=seed,
        verbose=verbose,
    )
    print()
    print(f"[rfa-svm] done in {result.runtime_seconds:.1f}s; "
          f"ranked {len(result.selected)} features")
    print(f"[rfa-svm] top 20: {result.selected[:20]}")
    _save_result(result, "svm_cost_function_label")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["rf", "svm", "both"],
        default="both",
        help="Which RFA variant(s) to run.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Hard cap on features added. Default: no cap (rank everything for SVM; "
             "use patience for RF).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=5_000,
        help="SVM stratified subsample. Smaller = faster but noisier.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration progress.")
    args = parser.parse_args()

    verbose = not args.quiet
    if args.variant in ("rf", "both"):
        _run_rf(args.max_features, args.seed, verbose)
        print()
    if args.variant in ("svm", "both"):
        _run_svm(args.max_features, args.subsample, args.seed, verbose)


if __name__ == "__main__":
    main()
