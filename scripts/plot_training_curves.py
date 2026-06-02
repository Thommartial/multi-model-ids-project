"""Plot deep-model train-vs-validation curves (WBS 13.1.7).

Reproduces reports/figures/dl_training_curves.png from the
TensorBoard event files written during training. For each deep model x
task it draws epoch_loss and epoch_accuracy with the train and
validation means and a +/- std band across seeds - the overfitting
check for the iterative models.

Usage:

    python scripts/plot_training_curves.py                 # tuned runs
    python scripts/plot_training_curves.py --tag ""         # untagged runs
    python scripts/plot_training_curves.py --out reports/figures/dl_training_curves.png
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.tb_curves import read_series, seed_band  # noqa: E402

RUNS = Path("reports/runs")
SPECS = [("cnn_1d", "binary"), ("cnn_1d", "multiclass"), ("lstm", "binary"), ("lstm", "multiclass")]
COL = {"train": "#1b9e77", "val": "#d95f02"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--tag", default="tuned", help="run subfolder/tag; pass '' for untagged runs")
    ap.add_argument("--out", default="reports/figures/dl_training_curves.png")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    tagseg = f"{args.tag}/" if args.tag else ""
    fig, axes = plt.subplots(4, 2, figsize=(11, 15))

    for r, (model, task) in enumerate(SPECS):
        seed_dirs = sorted(glob.glob(str(runs / model / task / f"{tagseg}seed_*")))
        for c, (tag, ylab) in enumerate([("epoch_loss", "Loss"), ("epoch_accuracy", "Accuracy")]):
            ax = axes[r, c]
            for split, lab in [("train", "Train"), ("validation", "Validation")]:
                band = seed_band([read_series(sd, split, tag) for sd in seed_dirs])
                if band is None:
                    continue
                x, mean, std = band
                key = "train" if split == "train" else "val"
                ax.plot(x, mean, color=COL[key], label=lab, lw=2)
                ax.fill_between(x, mean - std, mean + std, color=COL[key], alpha=0.2)
            ax.set_title(f"{model} — {task}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, loc="best", fontsize=9)

    fig.suptitle(
        "Deep models: train vs. validation (mean ± std over seeds)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
