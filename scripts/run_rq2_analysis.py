"""RQ2 analysis: per-class difficulty + FPR (WBS Part 10.2).

Reproduces every RQ2 artefact from the saved predictions_test.parquet
files (no model reload, no retraining):

Tables -> reports/tables/
  * rq2_per_class_f1.csv          (models x classes, mean over seeds)
  * rq2_per_class_fpr.csv         (models x classes one-vs-rest FPR)
  * rq2_difficulty_ranking.csv    (mean +/- std F1 across models, support)
  * rq2_most_confused_pairs.csv   (largest true->predicted confusions)

Figures -> reports/figures/
  * rq2_per_class_f1_heatmap.png
  * rq2_class_difficulty.png

The deep models read from their --dl-tag subfolder (default
tuned); the classical models read the untagged runs.

Usage:

    python scripts/run_rq2_analysis.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib  # noqa: E402

# Load numpy/pyarrow before matplotlib: matplotlib's PIL/kiwisolver native
# extensions can clash with pyarrow's bundled libstdc++ in some conda envs and
# segfault on the first parquet read. Importing pyarrow first avoids the crash.
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401  (force-load before matplotlib)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.class_analysis import (  # noqa: E402
    difficulty_ranking,
    most_confused_pairs,
    per_class_metrics,
)

RUNS = Path("reports/runs")
TABLES = Path("reports/tables")
FIGURES = Path("reports/figures")
# (label, folder, uses_dl_tag)
MODELS = [
    ("Rule-based", "rule_based", False),
    ("SVM", "svm", False),
    ("Random Forest", "random_forest", False),
    ("XGBoost", "xgboost", False),
    ("1D-CNN", "cnn_1d", True),
    ("LSTM", "lstm", True),
]
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


def _seed_files(runs: Path, folder: str, task: str, tag: str | None) -> list[str]:
    tagseg = f"{tag}/" if tag else ""
    return sorted(
        glob.glob(str(runs / folder / task / f"{tagseg}seed_*" / "predictions_test.parquet"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--task", default="multiclass")
    ap.add_argument("--dl-tag", default="tuned")
    args = ap.parse_args()
    runs = Path(args.runs_dir)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    f1_rows, fpr_rows = {}, {}
    all_true: list = []
    all_pred: list = []
    support = None
    for label, folder, uses_tag in MODELS:
        files = _seed_files(runs, folder, args.task, args.dl_tag if uses_tag else None)
        if not files:
            print(f"[warn] no predictions for {label}/{args.task}")
            continue
        f1s, fprs = [], []
        for fp in files:
            df = pd.read_parquet(fp)
            m = per_class_metrics(df["y_true"], df["y_pred"], CLASSES)
            f1s.append(m["f1"].to_numpy(dtype=float))
            fprs.append(m["fpr"].to_numpy(dtype=float))
            all_true.extend(df["y_true"].tolist())
            all_pred.extend(df["y_pred"].tolist())
            if support is None:
                support = df["y_true"].value_counts()
        f1_rows[label] = np.nanmean(f1s, axis=0)
        fpr_rows[label] = np.nanmean(fprs, axis=0)

    f1 = pd.DataFrame(f1_rows, index=CLASSES).T
    fpr = pd.DataFrame(fpr_rows, index=CLASSES).T
    rank = difficulty_ranking(f1)
    order = list(rank.index)
    rank["test_support"] = [int(support.get(c, 0)) for c in order]

    f1[order].round(4).to_csv(TABLES / "rq2_per_class_f1.csv")
    fpr[order].round(5).to_csv(TABLES / "rq2_per_class_fpr.csv")
    rank.round(4).to_csv(TABLES / "rq2_difficulty_ranking.csv")
    most_confused_pairs(all_true, all_pred, CLASSES).round(4).to_csv(
        TABLES / "rq2_most_confused_pairs.csv", index=False
    )

    # ---- Figure: per-class F1 heatmap (hardest -> easiest) ----
    mat = f1[order]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(mat.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )
    ax.set_title(
        "Per-class F1 by model — test (classes ordered hardest → easiest)",
        fontsize=12,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01)
    cb.set_label("F1")
    fig.tight_layout()
    fig.savefig(FIGURES / "rq2_per_class_f1_heatmap.png", dpi=300, bbox_inches="tight")

    # ---- Figure: class difficulty ranking with support ----
    means = rank["mean_f1"].to_numpy()
    stds = rank["std_f1"].to_numpy()
    sup = rank["test_support"].to_numpy()
    colors = [
        "#d73027" if m < 0.2 else "#fc8d59" if m < 0.5 else "#fee08b" if m < 0.75 else "#1a9850"
        for m in means
    ]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="#555", linewidth=0.4)
    for i in range(len(order)):
        ax.annotate(
            f"n={sup[i]}", (i, means[i] + stds[i] + 0.025), ha="center", fontsize=8, color="#333"
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Mean F1 across models")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Class difficulty ranking (mean ± std across models; n = test support)",
        fontsize=12,
        fontweight="bold",
    )
    legend = [
        Patch(color="#d73027", label="Very hard (<0.2)"),
        Patch(color="#fc8d59", label="Hard (0.2–0.5)"),
        Patch(color="#fee08b", label="Moderate (0.5–0.75)"),
        Patch(color="#1a9850", label="Easy (>0.75)"),
    ]
    ax.legend(handles=legend, frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq2_class_difficulty.png", dpi=300, bbox_inches="tight")

    print("RQ2 tables -> reports/tables/ ; figures -> reports/figures/")
    print("\nDifficulty ranking (mean F1 across models):")
    for c in order:
        print(f"  {c:16} F1={rank.loc[c, 'mean_f1']:.3f}  support={rank.loc[c, 'test_support']}")


if __name__ == "__main__":
    main()
