#!/usr/bin/env python
"""Run the preprocessing pipeline on the UNSW-NB15 dataset.

Usage:
    python scripts/run_preprocessing.py

Produces, under data/processed/:
    train.parquet, val.parquet, test.parquet  -- encoded, unscaled splits,
        each carrying the `label` and `attack_cat` target columns
    preprocessor.joblib                       -- the fitted Preprocessor
        (one-hot encoder, z-score scaler, imputation medians)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402

from src.data.loader import load_unsw_nb15, verify_checksums  # noqa: E402
from src.data.preprocessor import Preprocessor  # noqa: E402
from src.data.validation import validate_dataset  # noqa: E402

OUT_DIR = Path("data/processed")


def _summarise(name: str, x, y_bin, y_multi) -> None:
    attack_pct = 100 * int((y_bin == 1).sum()) / len(x)
    worms = int((y_multi == "Worms").sum())
    print(
        f"  {name:5s}: {len(x):>7,} rows | {x.shape[1]} features | "
        f"attack {attack_pct:5.2f}% | Worms={worms}"
    )


def main() -> None:
    print("=" * 60)
    print("UNSW-NB15 -- Preprocessing pipeline")
    print("=" * 60)

    verify_checksums()
    df = load_unsw_nb15(combine=True)
    validate_dataset(df, "pooled UNSW-NB15")

    pre = Preprocessor()
    df = pre.clean(df)
    train_df, val_df, test_df = pre.stratified_split(df)
    pre.fit(train_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nProcessed splits (encoded, unscaled):")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        x, y_bin, y_multi = pre.transform(part, scale=False)
        _summarise(name, x, y_bin, y_multi)
        out = x.copy()
        out["label"] = y_bin
        out["attack_cat"] = y_multi
        out.to_parquet(OUT_DIR / f"{name}.parquet", index=False)

    joblib.dump(pre, OUT_DIR / "preprocessor.joblib")
    print(f"\nFeature count after one-hot encoding: {len(pre.feature_names_)}")
    print(f"Saved train/val/test .parquet + preprocessor.joblib -> {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
