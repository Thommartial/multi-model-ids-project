"""Filter-based feature selection for the Multi-Model IDS project.

Implements the proposal's §5.1 "filter" condition: a **consensus** of two
complementary feature-ranking methods.

* :func:`extra_trees_selection` -- impurity-based importance from an
  Extra-Trees ensemble. Captures non-linear, interaction-driven signal.
* :func:`mutual_information_selection` -- mutual information between each
  feature and the label. Captures distributional dependence; non-parametric.
* :func:`consensus_selection` -- fuses the two rankings (average-rank
  aggregation by default) so features both methods agree on rise to the top.
  Also supports an "intersection" mode (features in the top-k of both).

Leakage rule (experimental_protocol.md §3, §10)
-----------------------------------------------
The filter is fitted on the **training fold only**. The output is a ranking
of feature names; the same ranking is then used to subset validation and
test. Neither method ever sees val or test rows.

Notes
-----
- Extra-Trees uses ``class_weight="balanced"`` for fairness on the heavily
  imbalanced multiclass target.
- ``mutual_info_classif`` is non-parametric (k-NN estimator) and runtime
  scales poorly with row count; on the ~114k-row training pool the default
  is to stratify-subsample to 30,000 rows for MI. Set ``sample_size=None``
  to use the full pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

from src.utils.reproducibility import DEFAULT_SEED


@dataclass
class FilterResult:
    """Output of :func:`consensus_selection`.

    Attributes
    ----------
    ranking
        One row per input feature. Columns: ``feature``, ``et_importance``,
        ``et_rank``, ``mi_score``, ``mi_rank``, ``consensus_rank``. Sorted by
        ``consensus_rank`` ascending (best feature first).
    selected
        Feature names selected -- top-k by consensus, or the intersection
        of the two top-k lists, depending on ``mode``.
    mode
        Fusion mode used ("rank_mean" or "intersection").
    target
        Name of the target column used for the ranking.
    """

    ranking: pd.DataFrame
    selected: list[str]
    mode: str
    target: str

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "target": self.target,
            "selected": self.selected,
            "ranking": self.ranking.to_dict(orient="records"),
        }


def extra_trees_selection(
    x: pd.DataFrame,
    y,
    n_estimators: int = 400,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
) -> pd.Series:
    """Per-feature Extra-Trees impurity-based importance.

    Higher is more important. Fitted on the training set only.

    Returns a ``pd.Series`` indexed by feature name, *not* sorted -- sort the
    caller's way if a ranking is needed.
    """
    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    et.fit(x, y)
    return pd.Series(et.feature_importances_, index=x.columns, name="et_importance")


def mutual_information_selection(
    x: pd.DataFrame,
    y,
    n_neighbors: int = 3,
    seed: int = DEFAULT_SEED,
    sample_size: int | None = 30_000,
) -> pd.Series:
    """Per-feature mutual information with the label.

    Higher is more informative. Fitted on the training set only.

    ``sample_size`` -- the MI estimator is O(n) per feature but with a
    large k-NN constant, so on the ~114k-row training pool the default
    sub-samples to 30,000 rows (stratified on ``y``) to keep runtime sensible.
    Set ``sample_size=None`` to use the full pool.

    Returns a ``pd.Series`` indexed by feature name, *not* sorted.
    """
    x_eff, y_eff = x, y
    if sample_size is not None and len(x) > sample_size:
        _, x_eff, _, y_eff = train_test_split(
            x,
            y,
            test_size=sample_size,
            stratify=y,
            random_state=seed,
        )

    mi = mutual_info_classif(
        x_eff,
        y_eff,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
    return pd.Series(mi, index=x_eff.columns, name="mi_score")


def consensus_selection(
    x: pd.DataFrame,
    y,
    k: int | None = None,
    mode: str = "rank_mean",
    target_name: str = "y",
    seed: int = DEFAULT_SEED,
    et_kwargs: dict | None = None,
    mi_kwargs: dict | None = None,
) -> FilterResult:
    """Rank features by consensus of Extra-Trees importance and mutual information.

    Parameters
    ----------
    x, y
        Training-set features and labels (typically multiclass ``attack_cat``).
    k
        How many top features to select. ``None`` keeps every feature; only a
        ranking is produced.
    mode
        ``"rank_mean"`` (default) -- average of the two per-method ranks; the
        top-k features by the resulting consensus rank are selected.
        ``"intersection"`` -- a feature is selected only if it sits in the
        top-k of *both* methods.
    target_name
        A label for the target, recorded on the result so saved artefacts can
        record which target the ranking was built against (e.g. "label" or
        "attack_cat").
    seed
        Random seed shared by both methods (so the result is reproducible).
    et_kwargs, mi_kwargs
        Optional extra kwargs forwarded to the underlying methods.
    """
    if mode not in {"rank_mean", "intersection"}:
        raise ValueError(
            f"unknown mode: {mode!r} (expected 'rank_mean' or 'intersection')"
        )
    if mode == "intersection" and k is None:
        raise ValueError("mode='intersection' requires k to be set")

    et_kwargs = dict(et_kwargs or {})
    mi_kwargs = dict(mi_kwargs or {})

    et = extra_trees_selection(x, y, seed=seed, **et_kwargs)
    mi = mutual_information_selection(x, y, seed=seed, **mi_kwargs)

    # Higher score = better; rank 1 = best.
    et_rank = et.rank(ascending=False, method="average")
    mi_rank = mi.rank(ascending=False, method="average")

    ranking = pd.DataFrame(
        {
            "feature": list(x.columns),
            "et_importance": et.reindex(x.columns).to_numpy(),
            "et_rank": et_rank.reindex(x.columns).to_numpy(),
            "mi_score": mi.reindex(x.columns).to_numpy(),
            "mi_rank": mi_rank.reindex(x.columns).to_numpy(),
        }
    )
    ranking["consensus_rank"] = (ranking["et_rank"] + ranking["mi_rank"]) / 2.0
    ranking = ranking.sort_values("consensus_rank", ascending=True).reset_index(drop=True)

    if mode == "rank_mean":
        selected = (
            ranking["feature"].head(k).tolist() if k is not None else ranking["feature"].tolist()
        )
    else:  # intersection
        et_top = set(ranking.loc[ranking["et_rank"] <= k, "feature"])
        mi_top = set(ranking.loc[ranking["mi_rank"] <= k, "feature"])
        selected_set = et_top & mi_top
        # Order the intersection by consensus rank (best first).
        selected = [f for f in ranking["feature"] if f in selected_set]

    return FilterResult(ranking=ranking, selected=selected, mode=mode, target=target_name)
