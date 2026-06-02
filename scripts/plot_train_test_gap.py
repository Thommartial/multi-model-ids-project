"""Plot the train-vs-test generalisation gap by model (WBS 10.3 / 17.2.3).

Reproduces reports/figures/train_test_gap.png from the per-seed
metrics.json files (which carry train_macro_f1 and
test_macro_f1 after the harness change / backfill_train_metrics).
A large train-test gap flags overfitting - the generalisation check for
the non-iterative classical models, which have no epoch curve.

The deep models read from their --dl-tag subfolder (default
tuned); the classical models read from the untagged runs.

Usage:

    python scripts/plot_train_test_gap.py
    python scripts/plot_train_test_gap.py --dl-tag tuned
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNS = Path("reports/runs")
# (label, folder, uses_dl_tag)
MODELS = [
    ("Rule-based", "rule_based", False),
    ("SVM", "svm", False),
    ("Random Forest", "random_forest", False),
    ("XGBoost", "xgboost", False),
    ("1D-CNN", "cnn_1d", True),
    ("LSTM", "lstm", True),
]
TRAIN_C = "#4C72B0"  # conventional muted blue
TEST_C = "#DD8452"  # conventional muted orange


def collect(runs: Path, folder: str, task: str, tag: str | None) -> tuple[list[float], list[float]]:
    tagseg = f"{tag}/" if tag else ""
    tr, te = [], []
    for f in glob.glob(str(runs / folder / task / f"{tagseg}seed_*" / "metrics.json")):
        d = json.load(open(f))
        if "train_macro_f1" in d:
            tr.append(d["train_macro_f1"])
        if "test_macro_f1" in d:
            te.append(d["test_macro_f1"])
    return tr, te


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--dl-tag", default="tuned", help="tag subfolder for the DL runs")
    ap.add_argument("--out", default="reports/figures/train_test_gap.png")
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    handles = None
    for ax, task in zip(axes, ["binary", "multiclass"]):
        labels = [m[0] for m in MODELS]
        tr_m, tr_s, te_m, te_s = [], [], [], []
        for _, folder, uses_tag in MODELS:
            tr, te = collect(runs, folder, task, args.dl_tag if uses_tag else None)
            tr_m.append(np.mean(tr) if tr else np.nan)
            tr_s.append(np.std(tr) if tr else 0.0)
            te_m.append(np.mean(te) if te else np.nan)
            te_s.append(np.std(te) if te else 0.0)
        x = np.arange(len(labels))
        w = 0.38
        b1 = ax.bar(x - w / 2, tr_m, w, yerr=tr_s, capsize=3, label="Train", color=TRAIN_C)
        b2 = ax.bar(x + w / 2, te_m, w, yerr=te_s, capsize=3, label="Test", color=TEST_C)
        handles = (b1, b2)
        for i in range(len(labels)):
            gap = tr_m[i] - te_m[i]
            ax.annotate(
                f"Δ{gap:.2f}",
                (x[i], max(tr_m[i], te_m[i]) + 0.025),
                ha="center",
                fontsize=8,
                color="#444",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Macro-F1")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{task.capitalize()} task", fontsize=12, fontweight="bold")

    fig.legend(handles, ["Train", "Test"], loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False, fontsize=11)
    fig.suptitle(
        "Train vs. test macro-F1 — generalisation gap by model (mean ± std over seeds)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
