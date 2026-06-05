"""Run the filter-only feature-selection condition (Part 5.1).

Loads the train fold, runs the consensus filter (Extra-Trees importance +
mutual information), and writes the full ranking and top-k selection to
reports/feature_selection/.

Default target is multiclass attack_cat (Section 5 of the experimental
protocol); pass --target label for the binary task.

Usage
    python scripts/run_filter_selection.py [--top-k N] [--mode rank_mean|intersection]
                                           [--target label|attack_cat]
                                           [--mi-sample N]
                                           [--et-trees N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# allow running as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_selection import consensus_selection  # noqa: E402

TRAIN = Path("data/processed/train.parquet")
OUT_DIR = Path("reports/feature_selection")
LABEL_COLS = ("label", "attack_cat")


def main(
    top_k: int = 30,
    mode: str = "rank_mean",
    target: str = "attack_cat",
    mi_sample: int | None = 30_000,
    et_trees: int = 400,
) -> None:
    if target not in LABEL_COLS:
        raise SystemExit(f"--target must be one of {LABEL_COLS}, got {target!r}")

    print(f"[filter] loading {TRAIN}")
    df = pd.read_parquet(TRAIN)
    x = df.drop(columns=list(LABEL_COLS))
    y = df[target].to_numpy()
    print(
        f"[filter] X shape: {x.shape}  |  target: {target} " f"({pd.Series(y).nunique()} classes)"
    )

    print(
        f"[filter] running consensus_selection (mode={mode}, k={top_k}, "
        f"ET trees={et_trees}, MI sample={mi_sample}) ..."
    )
    t0 = time.time()
    result = consensus_selection(
        x,
        y,
        k=top_k,
        mode=mode,
        target_name=target,
        et_kwargs={"n_estimators": et_trees},
        mi_kwargs={"sample_size": mi_sample},
    )
    print(f"[filter] done in {time.time() - t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"filter_ranking_{target}.csv"
    json_path = OUT_DIR / f"filter_selected_top{top_k}_{target}_{mode}.json"
    result.ranking.to_csv(csv_path, index=False)
    with open(json_path, "w") as fh:
        json.dump(
            {
                "target": target,
                "top_k": top_k,
                "mode": mode,
                "n_features_in": x.shape[1],
                "n_selected": len(result.selected),
                "selected": result.selected,
            },
            fh,
            indent=2,
        )
    print(f"[filter] saved ranking  -> {csv_path}")
    print(f"[filter] saved selected -> {json_path}")
    print()
    print(f"Top 20 features by consensus rank ({mode}, target={target}):")
    top20 = result.ranking.head(20)[["feature", "et_rank", "mi_rank", "consensus_rank"]]
    print(top20.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--mode", choices=["rank_mean", "intersection"], default="rank_mean")
    parser.add_argument(
        "--target",
        default="attack_cat",
        choices=list(LABEL_COLS),
        help="label (binary) or attack_cat (multiclass).",
    )
    parser.add_argument(
        "--mi-sample",
        type=int,
        default=30_000,
        help="MI stratified subsample size; 0 = use full training pool.",
    )
    parser.add_argument(
        "--et-trees",
        type=int,
        default=400,
        help="Number of Extra-Trees estimators.",
    )
    args = parser.parse_args()
    main(
        top_k=args.top_k,
        mode=args.mode,
        target=args.target,
        mi_sample=None if args.mi_sample == 0 else args.mi_sample,
        et_trees=args.et_trees,
    )
