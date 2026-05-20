"""Preprocessing pipeline for the Multi-Model IDS project.

Turns the raw UNSW-NB15 records into model-ready data through a single,
leakage-safe sequence:

    clean  ->  stratified split  ->  fit encoders on TRAIN only  ->  transform

Why the order matters
---------------------
Every transformation that *learns* something from the data -- the one-hot
category vocabulary, the z-score mean and standard deviation, the median
used for imputation -- is fitted on the **training split alone** and then
applied unchanged to the validation and test splits. Fitting on the whole
dataset would leak information from validation/test back into training
and silently inflate every score the project later reports. Splitting
*before* fitting is what keeps the evaluation honest.

Scaling policy (proposal Section 5.1)
-------------------------------------
:meth:`Preprocessor.transform` takes a ``scale`` flag. Neural-network
inputs are z-score standardised (``scale=True``); tree-based models
(Random Forest, XGBoost) are invariant to monotonic feature scaling and
use the unscaled features (``scale=False``). One-hot columns are never
scaled. The processed files saved to disk are unscaled; the fitted
scaler travels with the saved :class:`Preprocessor` so a network can
apply it on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import load_config
from src.utils.reproducibility import DEFAULT_SEED

# Identifier/target columns -- never model features.
ARTEFACT_COLS = ["id"]  # UNSW-NB15 partitioned set; CIC-IDS2017 adds IP/timestamp columns
CATEGORICAL_COLS = ["proto", "service", "state"]
TARGET_BINARY = "label"
TARGET_MULTICLASS = "attack_cat"


class Preprocessor:
    """Leakage-safe clean / split / encode / scale pipeline."""

    def __init__(self, config=None, min_category_freq: int = 20):
        # min_category_freq: categorical values seen fewer than this many
        # times in TRAIN are merged into one "infrequent" bucket. This
        # tames high-cardinality columns such as `proto` (~130 values,
        # most of them rare) without discarding information outright.
        self.config = config or load_config()
        self.min_category_freq = min_category_freq
        self.encoder: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None
        self.medians_: pd.Series | None = None
        self.numeric_cols_: list[str] | None = None
        self.feature_names_: list[str] | None = None

    # ----- 1. cleaning ----------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop identifier columns, remove duplicates, neutralise infinities.

        These steps are split-independent (they learn nothing from the
        data), so it is safe to run them before the train/test split.
        """
        df = df.drop(columns=[c for c in ARTEFACT_COLS if c in df.columns])
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"[preprocess] removed {removed:,} duplicate rows")
        numeric = [
            c
            for c in df.columns
            if c not in CATEGORICAL_COLS + [TARGET_BINARY, TARGET_MULTICLASS]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        # Replace +/-inf with NaN so they are handled by median imputation.
        df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
        return df

    # ----- 2. stratified split --------------------------------------
    def stratified_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into train/val/test, preserving class proportions.

        Stratifying on the 10-way ``attack_cat`` keeps every attack
        category -- including Worms, the rarest -- present in all three
        splits, and automatically balances the binary label too.
        """
        c = self.config.data
        holdout = c.val_ratio + c.test_ratio
        train, temp = train_test_split(
            df,
            test_size=holdout,
            stratify=df[TARGET_MULTICLASS],
            random_state=DEFAULT_SEED,
        )
        rel_test = c.test_ratio / holdout
        val, test = train_test_split(
            temp,
            test_size=rel_test,
            stratify=temp[TARGET_MULTICLASS],
            random_state=DEFAULT_SEED,
        )
        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    # ----- 3. fit (TRAIN ONLY) --------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        """Learn the encoder, scaler and imputation values from TRAIN."""
        feats = train_df.drop(columns=[TARGET_BINARY, TARGET_MULTICLASS])
        self.numeric_cols_ = [c for c in feats.columns if c not in CATEGORICAL_COLS]

        # Median imputation values (no-op on UNSW-NB15, which has no NaN,
        # but needed for the messier CIC-IDS2017 in Phase 2).
        self.medians_ = feats[self.numeric_cols_].median()

        # One-hot encoder for the categorical columns.
        self.encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=self.min_category_freq,
            sparse_output=False,
        )
        self.encoder.fit(feats[CATEGORICAL_COLS].astype(str))

        # z-score scaler, fitted on imputed TRAIN numeric features.
        train_numeric = feats[self.numeric_cols_].fillna(self.medians_)
        self.scaler = StandardScaler().fit(train_numeric)

        ohe_names = list(self.encoder.get_feature_names_out(CATEGORICAL_COLS))
        self.feature_names_ = self.numeric_cols_ + ohe_names
        return self

    # ----- 4. transform ---------------------------------------------
    def transform(
        self, df: pd.DataFrame, scale: bool = False
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Apply the fitted pipeline. Returns (X, y_binary, y_multiclass).

        ``scale=True`` z-score standardises the numeric features (for
        neural networks); ``scale=False`` leaves them raw (for tree
        models). One-hot columns are returned 0/1 either way.
        """
        if self.encoder is None or self.scaler is None:
            raise RuntimeError("Preprocessor.fit must be called before transform")

        feats = df.drop(columns=[TARGET_BINARY, TARGET_MULTICLASS])
        numeric = feats[self.numeric_cols_].fillna(self.medians_)
        if scale:
            numeric = pd.DataFrame(self.scaler.transform(numeric), columns=self.numeric_cols_)
        else:
            numeric = numeric.reset_index(drop=True)

        encoded = self.encoder.transform(feats[CATEGORICAL_COLS].astype(str))
        encoded = pd.DataFrame(
            encoded, columns=self.encoder.get_feature_names_out(CATEGORICAL_COLS)
        )

        x = pd.concat([numeric, encoded], axis=1)
        y_binary = df[TARGET_BINARY].reset_index(drop=True)
        y_multiclass = df[TARGET_MULTICLASS].reset_index(drop=True)
        return x, y_binary, y_multiclass


def make_sliding_windows(
    x: np.ndarray, y: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Turn 2-D records into overlapping 3-D sequences for LSTM / CNN-LSTM.

    Given ``x`` of shape (n_records, n_features), returns windows of shape
    (n_windows, window_size, n_features); each window's target is the
    label of its final record. The records are used in the order given --
    the ordering policy for sequential models is finalised in Part 6.3.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_windows = len(x) - window_size + 1
    if n_windows <= 0:
        raise ValueError(f"need at least {window_size} records, got {len(x)}")
    windows = np.stack([x[i : i + window_size] for i in range(n_windows)])
    targets = y[window_size - 1 :]
    return windows, targets
