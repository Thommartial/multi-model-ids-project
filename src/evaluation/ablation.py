"""Ablation-study framework for the feature-engineering comparison.

Runs the **five-condition** comparison from
`docs/experimental_protocol.md` §10.1 under the project's statistical
rigour (§8):

* 5 seeds per condition (the protocol's fixed list ``[42, 43, 44, 45, 46]``).
* Paired Wilcoxon signed-rank test on per-seed macro-F1 across conditions.
* Holm–Bonferroni correction across the family of pairwise comparisons.
* 95% bootstrap CIs on each condition's per-seed metric distribution.
* Effect-size floor (a delta below 0.005 macro-F1 is not claimed).

The conditions (all sharing a Random Forest downstream classifier so the
comparison is clean):

1. **Baseline** — all 190 features.
2. **Filter only** — top-k from the consensus filter
   (`src/data/feature_selection.py`).
3. **RFA original** — top-k from the SVM cost-function RFA
   (`src/data/rfa.py`).
4. **RFA proposal** — top-k from the RF / val macro-F1 RFA
   (`src/data/rfa.py`).
5. **RFA + flow-pair** — best of (3) vs (4) plus flow-pair features
   (`src/data/bigram_features.py`).

This is the headline feature-engineering result of the project. The
framework here runs the conditions; the caller assembles the feature
sets (typically by loading the ranking files written by the upstream
scripts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.utils.reproducibility import DEFAULT_SEED

DEFAULT_SEEDS: list[int] = [42, 43, 44, 45, 46]


@dataclass
class ConditionResult:
    """Per-condition results across the protocol's seed list.

    Attributes
    ----------
    name
        Human-readable condition name (e.g. ``"filter_top30"``).
    feature_set
        The exact list of feature names used for this condition.
    n_features
        Convenience -- ``len(feature_set)``.
    per_seed_metrics
        Indexed by seed; columns include ``val_macro_f1``, ``test_macro_f1``,
        ``val_balanced_accuracy``, ``test_balanced_accuracy``, and for the
        binary task additionally ``val_auroc``, ``val_pr_auc``,
        ``test_auroc``, ``test_pr_auc``.
    runtime_seconds
        Wall-clock time for the full 5-seed run of this condition.
    """

    name: str
    feature_set: list[str]
    n_features: int
    per_seed_metrics: pd.DataFrame
    runtime_seconds: float

    def summary(self, metric: str) -> dict[str, float]:
        """Mean, std, and 95% bootstrap CI for ``metric`` across seeds."""
        values = self.per_seed_metrics[metric].to_numpy()
        ci_lo, ci_hi = bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=DEFAULT_SEED)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
        }


@dataclass
class AblationResult:
    """Result of running multiple conditions side by side."""

    conditions: dict[str, ConditionResult]
    pairwise_comparisons: pd.DataFrame
    seeds: list[int]
    primary_metric: str
    task: str

    def summary_table(self) -> pd.DataFrame:
        """One row per condition with mean ± std + 95% CI on the primary metric."""
        rows = []
        for name, cond in self.conditions.items():
            s = cond.summary(self.primary_metric)
            rows.append(
                {
                    "condition": name,
                    "n_features": cond.n_features,
                    f"{self.primary_metric}_mean": s["mean"],
                    f"{self.primary_metric}_std": s["std"],
                    f"{self.primary_metric}_ci95_lo": s["ci95_lo"],
                    f"{self.primary_metric}_ci95_hi": s["ci95_hi"],
                    "runtime_s": cond.runtime_seconds,
                }
            )
        df = pd.DataFrame(rows)
        return df.sort_values(f"{self.primary_metric}_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true,
    y_pred,
    y_proba: np.ndarray | None = None,
    task: str = "multiclass",
) -> dict[str, float]:
    """Per-row metric block for one (y_true, y_pred[, y_proba]) prediction.

    Always returns macro-F1 and balanced accuracy. For binary tasks with
    a predicted probability vector, additionally returns AUROC and PR-AUC
    (PR-AUC is preferred over AUROC under heavy class imbalance, per
    protocol §6.1).
    """
    metrics: dict[str, float] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if task == "binary" and y_proba is not None:
        metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return metrics


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Two-sided percentile bootstrap CI on the mean.

    Returns ``(lo, hi)`` for the ``(1 - alpha)`` interval.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        means[i] = values[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def holm_bonferroni(p_values: Iterable[float]) -> np.ndarray:
    """Holm–Bonferroni-adjusted p-values, in the original order.

    Implementation: sort p-values ascending, multiply the *i*-th sorted
    value by ``n - i`` (i 0-indexed), enforce a running max for
    monotonicity, then cap at 1.
    """
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    if n == 0:
        return p
    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]
    multipliers = (n - np.arange(n)).astype(float)
    adjusted = sorted_p * multipliers
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(n, dtype=float)
    out[sort_idx] = adjusted
    return out


# ---------------------------------------------------------------------------
# Running a single condition
# ---------------------------------------------------------------------------


def run_condition(
    name: str,
    feature_set: list[str],
    x_train: pd.DataFrame,
    y_train,
    x_val: pd.DataFrame,
    y_val,
    x_test: pd.DataFrame,
    y_test,
    *,
    seeds: list[int] = DEFAULT_SEEDS,
    task: str = "multiclass",
    rf_kwargs: dict | None = None,
    verbose: bool = False,
) -> ConditionResult:
    """Train Random Forest on ``feature_set`` with each seed, collect metrics.

    Default RF hyperparameters are the protocol's safe defaults for the
    *ablation*; the headline model run will use the grid-searched
    hyperparameters from `configs/hparam_spaces/random_forest.yaml`. The
    point of the ablation is to *compare* feature sets, so any defensible
    RF that is held fixed across conditions is acceptable here.
    """
    default_rf = {
        "n_estimators": 100,
        "max_depth": None,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    default_rf.update(rf_kwargs or {})

    t0 = time.time()
    rows: list[dict] = []
    for seed in seeds:
        rf = RandomForestClassifier(random_state=seed, **default_rf)
        rf.fit(x_train[feature_set], y_train)

        row: dict = {"seed": seed}
        for split_name, x_split, y_split in (
            ("val", x_val, y_val),
            ("test", x_test, y_test),
        ):
            y_pred = rf.predict(x_split[feature_set])
            y_proba = None
            if task == "binary":
                y_proba = rf.predict_proba(x_split[feature_set])[:, 1]
            m = compute_metrics(y_split, y_pred, y_proba=y_proba, task=task)
            for k, v in m.items():
                row[f"{split_name}_{k}"] = v
        rows.append(row)
        if verbose:
            print(
                f"[ablation] {name} seed={seed:>3d}  "
                f"val_macro_f1={row['val_macro_f1']:.4f}  "
                f"test_macro_f1={row['test_macro_f1']:.4f}"
            )

    per_seed = pd.DataFrame(rows).set_index("seed").sort_index()
    return ConditionResult(
        name=name,
        feature_set=list(feature_set),
        n_features=len(feature_set),
        per_seed_metrics=per_seed,
        runtime_seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Pairwise comparison across conditions
# ---------------------------------------------------------------------------


def pairwise_comparisons(
    conditions: dict[str, ConditionResult],
    metric: str = "test_macro_f1",
    effect_size_floor: float = 0.005,
) -> pd.DataFrame:
    """Paired Wilcoxon + Holm–Bonferroni between every pair of conditions.

    The effect-size floor follows the protocol's §8 rule: a delta below
    ``effect_size_floor`` (default 0.005) is not claimed as a meaningful
    win even if its corrected p-value is significant. The output's
    ``claim`` column flags pairs that pass both gates.
    """
    rows: list[dict] = []
    names = list(conditions.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            va = conditions[a].per_seed_metrics[metric].to_numpy()
            vb = conditions[b].per_seed_metrics[metric].to_numpy()
            try:
                _, p = wilcoxon(va, vb)
            except ValueError:
                # All differences zero -- wilcoxon raises.
                p = np.nan
            rows.append(
                {
                    "cond_a": a,
                    "cond_b": b,
                    "mean_a": float(va.mean()),
                    "mean_b": float(vb.mean()),
                    "delta_a_minus_b": float(va.mean() - vb.mean()),
                    "p_wilcoxon": p,
                }
            )
    df = pd.DataFrame(rows)
    if len(df) > 0:
        ok = df["p_wilcoxon"].notna()
        if ok.any():
            df.loc[ok, "p_holm"] = holm_bonferroni(df.loc[ok, "p_wilcoxon"].to_numpy())
        else:
            df["p_holm"] = np.nan
        df["claim"] = (
            (df["p_holm"] < 0.05)
            & (df["delta_a_minus_b"].abs() >= effect_size_floor)
        )
    return df


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_ablation(
    x_train: pd.DataFrame,
    y_train,
    x_val: pd.DataFrame,
    y_val,
    x_test: pd.DataFrame,
    y_test,
    *,
    feature_sets: dict[str, list[str]],
    seeds: list[int] = DEFAULT_SEEDS,
    task: str = "multiclass",
    primary_metric: str = "test_macro_f1",
    rf_kwargs: dict | None = None,
    verbose: bool = False,
) -> AblationResult:
    """Run the multi-condition ablation under the project's protocol.

    Parameters
    ----------
    x_*, y_*
        Train / val / test folds.
    feature_sets
        ``{condition_name: [feature_name, ...]}``. Each condition is run
        independently with the same Random Forest hyperparameters.
    seeds
        Default ``[42, 43, 44, 45, 46]`` per protocol §8.
    task
        ``"multiclass"`` (default) or ``"binary"`` — controls which
        metrics are computed.
    primary_metric
        Column to use for pairwise comparisons and the summary ordering.
    """
    conditions: dict[str, ConditionResult] = {}
    for name, feats in feature_sets.items():
        if verbose:
            print(f"[ablation] >>> condition: {name} ({len(feats)} features)")
        result = run_condition(
            name=name,
            feature_set=feats,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            seeds=seeds,
            task=task,
            rf_kwargs=rf_kwargs,
            verbose=verbose,
        )
        conditions[name] = result
        if verbose:
            s = result.summary(primary_metric)
            print(
                f"[ablation] <<< {name}: {primary_metric}="
                f"{s['mean']:.4f} ± {s['std']:.4f} "
                f"(95% CI [{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}])  "
                f"runtime={result.runtime_seconds:.1f}s"
            )

    comparisons = pairwise_comparisons(conditions, metric=primary_metric)
    return AblationResult(
        conditions=conditions,
        pairwise_comparisons=comparisons,
        seeds=list(seeds),
        primary_metric=primary_metric,
        task=task,
    )
