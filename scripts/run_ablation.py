"""Run the five-condition feature-engineering ablation (Part 5.4).

Compares baseline, filter-only, the two RFA variants, and RFA + flow-pair
features under a fixed Random Forest, with the protocol's statistical
rigour (5 seeds, paired Wilcoxon, Holm--Bonferroni, 95% bootstrap CIs).

Dependencies on prior steps
* Filter ranking JSON (Part 5.1) - produced by run_filter_selection.py.
* RFA rankings (Part 5.2) - produced by run_rfa.py; if absent, the
  corresponding ablation conditions are skipped and a warning is printed.
* Flow-pair features (Part 5.3) - computed on the fly here from the
  numeric columns of the processed parquets.

Outputs go under reports/ablation/:

* ablation_per_seed.csv - one row per (condition, seed) with all metrics.
* ablation_summary.csv - one row per condition with mean ± std + 95% CI.
* ablation_comparisons.csv - pairwise Wilcoxon + Holm--Bonferroni.
* ablation_summary.md - short human-readable summary.

Usage
:

    python scripts/run_ablation.py                 # multiclass, all available conditions
    python scripts/run_ablation.py --task binary   # binary task (uses label)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# allow running as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bigram_features import extract_all_bigram_features  # noqa: E402
from src.evaluation.ablation import DEFAULT_SEEDS, run_ablation  # noqa: E402

PROCESSED = Path("data/processed")
FS_DIR = Path("reports/feature_selection")
OUT_DIR = Path("reports/ablation")
LABEL_COLS = ("label", "attack_cat")


def _load_fold(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def _df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Plain-markdown table renderer that does not depend on tabulate."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v) if pd.notna(v) else "")
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _try_load_filter_selected(target: str, top_k: int = 30) -> list[str] | None:
    path = FS_DIR / f"filter_selected_top{top_k}_{target}_rank_mean.json"
    if not path.exists():
        print(f"[ablation] filter selection not found at {path}")
        return None
    with open(path) as fh:
        data = json.load(fh)
    return data["selected"]


def _try_load_rfa_selected(tag: str) -> list[str] | None:
    """Load RFA-selected feature list from rfa_hyperparams_<tag>.json."""
    path = FS_DIR / f"rfa_hyperparams_{tag}.json"
    if not path.exists():
        print(f"[ablation] RFA selection not found at {path} -- condition will be skipped")
        return None
    with open(path) as fh:
        data = json.load(fh)
    return data["selected"]


def _add_flow_pair_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Append flow-pair (difference + ratio) features to each fold.

    Computed *within each fold* (leakage-clean variant, per the module's
    docstring). Returns the augmented frames plus the new column names.
    """
    new_cols: list[str] = []
    out = []
    for frame in (train, val, test):
        numeric = (
            frame.drop(columns=list(LABEL_COLS)).select_dtypes(include="number").columns.tolist()
        )
        derived = extract_all_bigram_features(
            frame[numeric],
            lag=1,
            include={"difference", "ratio"},
            keep_original=False,
            fill=0.0,
        )
        derived.index = frame.index
        augmented = pd.concat(
            [frame.reset_index(drop=True), derived.reset_index(drop=True)], axis=1
        )
        out.append(augmented)
        new_cols = list(derived.columns)
    return out[0], out[1], out[2], new_cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["multiclass", "binary"],
        default="multiclass",
        help="Which target to use (attack_cat or label).",
    )
    parser.add_argument(
        "--filter-top-k",
        type=int,
        default=30,
        help="Top-k file to use for the filter condition (must already exist).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds for the multi-seed runs.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="RF n_estimators (held fixed across conditions).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="RF max_depth (held fixed across conditions). Default: None (no cap).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-seed progress lines.",
    )
    args = parser.parse_args()

    target = "attack_cat" if args.task == "multiclass" else "label"
    print(f"[ablation] task={args.task}  target={target}  seeds={args.seeds}")

    # ---- load folds --------------------------------------------------------
    print(f"[ablation] loading data/processed/{{train, val, test}}.parquet")
    train = _load_fold("train")
    val = _load_fold("val")
    test = _load_fold("test")
    x_train = train.drop(columns=list(LABEL_COLS))
    x_val = val.drop(columns=list(LABEL_COLS))
    x_test = test.drop(columns=list(LABEL_COLS))
    y_train = train[target].to_numpy()
    y_val = val[target].to_numpy()
    y_test = test[target].to_numpy()
    all_features = list(x_train.columns)
    print(f"[ablation] X_train={x_train.shape}  X_val={x_val.shape}  X_test={x_test.shape}")

    # ---- assemble the feature sets per condition --------------------------
    feature_sets: dict[str, list[str]] = {}

    # 1. baseline - all features
    feature_sets["baseline_all_features"] = all_features

    # 2. filter - top-k consensus
    filter_target = "attack_cat"  # filter ranking is multiclass
    filter_sel = _try_load_filter_selected(filter_target, top_k=args.filter_top_k)
    if filter_sel is not None:
        feature_sets[f"filter_top{args.filter_top_k}"] = filter_sel

    # 3. RFA original (SVM cost function) - always against the binary label
    rfa_svm_sel = _try_load_rfa_selected("svm_cost_function_label")
    if rfa_svm_sel is not None:
        feature_sets["rfa_svm_original"] = rfa_svm_sel

    # 4. RFA proposal (RF / val macro-F1)
    rfa_rf_sel = _try_load_rfa_selected("rf_macro_f1_attack_cat")
    if rfa_rf_sel is not None:
        feature_sets["rfa_rf_proposal"] = rfa_rf_sel

    # 5. RFA + flow-pair features (best of (3)/(4); fall back to whichever exists)
    best_rfa = None
    if rfa_rf_sel is not None and rfa_svm_sel is not None:
        # Will be resolved AFTER the ablation results are in; for now, use the
        # RF/F1 RFA as the default base (it's multiclass and matches the task).
        best_rfa = rfa_rf_sel
    elif rfa_rf_sel is not None:
        best_rfa = rfa_rf_sel
    elif rfa_svm_sel is not None:
        best_rfa = rfa_svm_sel

    if best_rfa is not None:
        print("[ablation] computing flow-pair features (within-fold; lag=1; diff+ratio)")
        train_a, val_a, test_a, new_cols = _add_flow_pair_features(train, val, test)
        feature_sets["rfa_plus_flow_pair"] = best_rfa + new_cols
        # rebind data with augmented columns so the new columns are available
        x_train, x_val, x_test = (
            train_a.drop(columns=list(LABEL_COLS)),
            val_a.drop(columns=list(LABEL_COLS)),
            test_a.drop(columns=list(LABEL_COLS)),
        )

    print(f"[ablation] {len(feature_sets)} conditions to run:")
    for name, feats in feature_sets.items():
        print(f"           - {name:<28s} ({len(feats):>4d} features)")

    # ---- run ---------------------------------------------------------------
    rf_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    result = run_ablation(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        feature_sets=feature_sets,
        seeds=args.seeds,
        task=args.task,
        rf_kwargs=rf_kwargs,
        verbose=not args.quiet,
    )

    # ---- save --------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = pd.concat(
        {name: cond.per_seed_metrics for name, cond in result.conditions.items()},
        names=["condition", "seed"],
    ).reset_index()
    per_seed.to_csv(OUT_DIR / f"ablation_per_seed_{args.task}.csv", index=False)

    summary = result.summary_table()
    summary.to_csv(OUT_DIR / f"ablation_summary_{args.task}.csv", index=False)

    result.pairwise_comparisons.to_csv(
        OUT_DIR / f"ablation_comparisons_{args.task}.csv", index=False
    )

    # Markdown report
    md_lines = [
        f"# Ablation results -- {args.task} task",
        "",
        f"Primary metric: **{result.primary_metric}**  |  " f"seeds: {result.seeds}",
        "",
        "## Per-condition summary",
        "",
        _df_to_markdown(summary),
        "",
        "## Pairwise comparisons (Holm--Bonferroni corrected, effect-size floor 0.005)",
        "",
        _df_to_markdown(result.pairwise_comparisons),
        "",
    ]
    (OUT_DIR / f"ablation_summary_{args.task}.md").write_text("\n".join(md_lines))

    print()
    print(f"[ablation] saved per-seed       -> {OUT_DIR}/ablation_per_seed_{args.task}.csv")
    print(f"[ablation] saved summary        -> {OUT_DIR}/ablation_summary_{args.task}.csv")
    print(f"[ablation] saved comparisons    -> {OUT_DIR}/ablation_comparisons_{args.task}.csv")
    print(f"[ablation] saved markdown table -> {OUT_DIR}/ablation_summary_{args.task}.md")
    print()
    print("Summary table:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
