"""RQ1 analysis: feature-engineering strategy + feature-group predictiveness (WBS Part 10.1).

Answers RQ1 in three reproducible pieces:

1. Strategy comparison (model-free)  -> reports/figures/rq1_ablation_strategy.png
   Which feature-engineering strategy wins overall, from the frozen
   ablation summary (baseline / filter / RFA variants / flow-pair).

2. RFA group composition (model-free) -> reports/figures/rq1_rfa_group_composition.png
   Which semantic feature groups the winning RFA selection draws from.

3. Per-class group importance (needs a trained model) ->
   reports/figures/rq1_feature_group_importance.png + reports/tables/rq1_group_importance.csv
   Permutation importance per attack class: shuffle each semantic group's
   columns in the test fold and measure the drop in that class's F1. This
   is the direct RQ1 answer - which feature group drives each attack type.

Run the model-free parts anywhere:

    python scripts/run_rq1_analysis.py --skip-permutation

Run the full analysis in the project env (needs the saved model + libs):

    python scripts/run_rq1_analysis.py --model xgboost
"""

from __future__ import annotations

import os

# Pin native thread pools to one thread BEFORE numpy/xgboost import. XGBoost's
# OpenMP runtime can segfault on Linux when it clashes with the BLAS thread
# pool; single-threaded prediction is plenty fast for this analysis and avoids
# the crash. Must be set before the numerical libraries load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import sys
from pathlib import Path

import matplotlib  # noqa: E402

# Load the native numerical / IO stack (numpy, pyarrow) BEFORE matplotlib.
# matplotlib pulls in PIL and kiwisolver native extensions whose bundled
# libstdc++ clashes with pyarrow's in some conda envs, segfaulting on the
# first parquet read if matplotlib initialises first. Importing pyarrow up
# front claims the correct runtime and avoids the crash.
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401  (force-load before matplotlib)
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.class_analysis import per_class_metrics  # noqa: E402

PROCESSED = Path("data/processed")
RUNS = Path("reports/runs")
TABLES = Path("reports/tables")
FIGURES = Path("reports/figures")
GROUPS_YAML = Path("configs/feature_groups.yaml")
CLASSES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "Reconnaissance",
    "DoS",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]


# --------------------------------------------------------------------------
# Map encoded columns -> semantic groups
# --------------------------------------------------------------------------
def build_group_map(columns: list[str], groups_yaml: Path) -> dict[str, str]:
    """Assign each (possibly one-hot) column to its semantic group.

    One-hot columns from proto/service/state inherit the parent's group
    (per the feature_groups.yaml note); everything else matches directly.
    """
    groups = yaml.safe_load(open(groups_yaml))
    raw_to_group = {feat: g for g, feats in groups.items() for feat in feats}
    out = {}
    for c in columns:
        if c in raw_to_group:
            out[c] = raw_to_group[c]
        elif c.startswith(("proto_", "service_", "state_")):
            parent = c.split("_", 1)[0]  # proto/service/state -> basic_flow
            out[c] = raw_to_group.get(parent, "other")
        else:
            out[c] = "other"
    return out


# --------------------------------------------------------------------------
# 1. Strategy comparison figure (model-free)
# --------------------------------------------------------------------------
def plot_strategy(task: str) -> None:
    f = TABLES.parent / "ablation" / f"ablation_summary_{task}.csv"
    df = pd.read_csv(f).sort_values("test_macro_f1_mean")
    # Display-only relabelling (underlying condition names are unchanged).
    display_names = {"rfa_rf_proposal": "rfa_rf_proposed"}
    labels = [display_names.get(c, c) for c in df["condition"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))
    ax.barh(y, df["test_macro_f1_mean"], xerr=df["test_macro_f1_std"], capsize=3, color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.annotate(
            f"{r['test_macro_f1_mean']:.3f}  ({int(r['n_features'])} feat)",
            (r["test_macro_f1_mean"] + 0.005, i),
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Test macro-F1 (mean ± std over seeds)")
    ax.set_xlim(0, max(df["test_macro_f1_mean"]) + 0.12)
    ax.set_title(
        f"RQ1: feature-engineering strategy comparison ({task}, fixed Random Forest)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "rq1_ablation_strategy.png", dpi=300, bbox_inches="tight")
    print(f"saved {FIGURES / 'rq1_ablation_strategy.png'}")


# --------------------------------------------------------------------------
# 2. RFA group composition (model-free)
# --------------------------------------------------------------------------
def plot_rfa_composition(group_map: dict[str, str]) -> None:
    rank = pd.read_csv(Path("reports/feature_selection/rfa_ranking_rf_macro_f1_attack_cat.csv"))
    selected = rank.dropna(subset=["rfa_rank"]).sort_values("rfa_rank")["feature"].tolist()
    comp = pd.Series([group_map.get(f, "other") for f in selected]).value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comp.sort_values().plot.barh(ax=ax, color="#55A868")
    for i, v in enumerate(comp.sort_values().values):
        ax.annotate(str(int(v)), (v + 0.05, i), va="center", fontsize=10)
    ax.set_xlabel("Number of RFA-selected features")
    ax.set_title(
        f"RQ1: semantic groups in the winning RFA selection ({len(selected)} features)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "rq1_rfa_group_composition.png", dpi=300, bbox_inches="tight")
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"feature": selected, "group": [group_map.get(f, "other") for f in selected]}
    ).to_csv(TABLES / "rq1_rfa_selection_groups.csv", index=False)
    print(f"saved {FIGURES / 'rq1_rfa_group_composition.png'}")


# --------------------------------------------------------------------------
# 3. Per-class group permutation importance (needs a trained model)
# --------------------------------------------------------------------------
def group_permutation_importance(model_name: str, task: str, n_repeats: int, seed: int) -> None:
    import joblib

    train_cols = [
        c
        for c in pd.read_parquet(PROCESSED / "train.parquet").columns
        if c not in ("label", "attack_cat")
    ]
    group_map = build_group_map(train_cols, GROUPS_YAML)
    groups = sorted(set(group_map.values()))

    test = pd.read_parquet(PROCESSED / "test.parquet")
    target = "attack_cat" if task == "multiclass" else "label"
    x_test = test.drop(columns=["label", "attack_cat"])
    y_test = test[target].to_numpy()

    seed_dirs = sorted(glob.glob(str(RUNS / model_name / task / "seed_*")))
    rng = np.random.default_rng(seed)
    per_seed_mats = []
    for sd in seed_dirs:
        try:
            model = joblib.load(Path(sd) / "model.joblib")
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {Path(sd).name}: could not load model ({exc})")
            continue
        base = per_class_metrics(y_test, model.predict(x_test), CLASSES)["f1"]
        mat = pd.DataFrame(0.0, index=CLASSES, columns=groups)
        for g in groups:
            cols = [c for c in x_test.columns if group_map[c] == g]
            if not cols:
                continue
            drops = []
            for _ in range(n_repeats):
                xp = x_test.copy()
                for c in cols:
                    xp[c] = rng.permutation(xp[c].to_numpy())
                f1p = per_class_metrics(y_test, model.predict(xp), CLASSES)["f1"]
                drops.append((base - f1p).to_numpy(dtype=float))
            mat[g] = np.mean(drops, axis=0)
        per_seed_mats.append(mat)
        print(f"  permuted {Path(sd).name}")
    if not per_seed_mats:
        raise SystemExit(f"no usable {model_name} models under {RUNS}/{model_name}/{task}/")
    imp = sum(per_seed_mats) / len(per_seed_mats)
    TABLES.mkdir(parents=True, exist_ok=True)
    imp.round(4).to_csv(TABLES / "rq1_group_importance.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(imp.values, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    for i in range(imp.shape[0]):
        for j in range(imp.shape[1]):
            ax.text(
                j,
                i,
                f"{imp.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#222" if imp.values[i, j] < imp.values.max() * 0.5 else "white",
            )
    ax.set_title(
        f"RQ1: per-class feature-group importance ({model_name}, {task})\n(F1 drop when group is permuted)",
        fontsize=12,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("F1 drop")
    fig.tight_layout()
    fig.savefig(FIGURES / "rq1_feature_group_importance.png", dpi=300, bbox_inches="tight")
    print(f"saved {FIGURES / 'rq1_feature_group_importance.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="multiclass")
    ap.add_argument("--model", default="xgboost", help="model family for permutation importance")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-permutation",
        action="store_true",
        help="only the model-free strategy + RFA-composition figures",
    )
    args = ap.parse_args()

    train_cols = [
        c
        for c in pd.read_parquet(PROCESSED / "train.parquet").columns
        if c not in ("label", "attack_cat")
    ]
    group_map = build_group_map(train_cols, GROUPS_YAML)

    plot_strategy(args.task)
    plot_rfa_composition(group_map)
    if not args.skip_permutation:
        group_permutation_importance(args.model, args.task, args.n_repeats, args.seed)
    print("RQ1 done.")


if __name__ == "__main__":
    main()
