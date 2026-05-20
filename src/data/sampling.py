"""Stratified subsampling for the Multi-Model IDS project.

The proposal (Section 4.3) planned subsampling to 200k-400k (binary),
100k-200k (multiclass) and 50k-100k (robustness) records to keep training
times feasible. Those targets were sized for the dataset *with* its ~37%
duplicate records. After deduplication the whole dataset is only 162,745
records (113,921 in the training split), so in practice:

* the binary and multiclass experiments use the **full training split**
  -- it is already below the proposal's lower bound, so no subsampling is
  needed;
* the robustness experiments, which repeat over many noise levels, use a
  ~75k stratified subsample to keep total run time reasonable;
* the RBF-SVM caps its training set at 50k records (proposal risk
  register), since its kernel cost scales steeply with sample size.

:func:`stratified_subsample` is the single primitive behind all three.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.reproducibility import DEFAULT_SEED


def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    stratify_col: str = "attack_cat",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return a class-proportional subsample of ``n`` rows.

    Stratifying on ``stratify_col`` keeps every class's share identical to
    the full set. If ``n`` is greater than or equal to ``len(df)`` the
    data is returned unchanged, since subsampling would be a no-op.
    """
    if n >= len(df):
        return df.reset_index(drop=True)
    subsample, _ = train_test_split(
        df,
        train_size=n,
        stratify=df[stratify_col],
        random_state=seed,
    )
    return subsample.reset_index(drop=True)
