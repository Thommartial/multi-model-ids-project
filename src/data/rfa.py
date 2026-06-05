"""Recursive Feature Addition (RFA) for the Multi-Model IDS project.

Two complementary variants. Both produce a forward-selection ranking of the
input features, starting from an empty set and adding one feature per
iteration; they are compared head-to-head in Part 5.4
(experimental_protocol.md section 10.1 conditions 3 and 4). The two-variant
comparison was the supervisor decision recorded on 2026-05-22
(docs/rfa_bigram_spec.md section 7).

* rfa_random_forest - the proposal's variant. At each step,
  adds the feature whose inclusion gives the largest gain in validation
  macro-F1 of a shallow Random Forest. Stops on patience-based early
  termination. Multiclass target (attack_cat). Cited as
  *"RFA-style forward selection, in the spirit of Hamed, Dara & Kremer
  (2018)"* - not as their exact algorithm.

* rfa_svm_cost_function - the original algorithm (Hamed, Dara
  & Kremer 2018; thesis Algorithm 2; Eq. 3.10). At each step, adds the
  feature whose inclusion produces the largest decrease in the SVM dual
  objective ½αᵀHα. The SVM is trained once per *added* feature, and
  candidate features are scored against the current α without
  retraining (Guyon-Weston-style cost-function approximation). Binary
  target (the original is binary).

Leakage rule (experimental_protocol.md section 3)
RFA is fitted on the training fold. The RF variant additionally consults
the validation fold for its F1 score; that is by design (the protocol's
hyperparameter-selection budget) and uses no test data.

Tractability
The SVM RFA is heavy: per protocol section 9 the SVM uses a stratified subsample.
For RFA specifically the subsample is set smaller (default 5,000) because
the kernel matrix on the support vectors is *O(N²)* and is recomputed
once per added feature.

Initialisation
Algorithm 2 in the thesis is written as a while |S| < N loop and does
not specify how to score features when S is empty (there is no SVM yet
to compute α against). For the empty-set case, the SVM RFA here picks
the first feature by mutual information with the label - a fast,
deterministic, defensible initialisation. The cost-function machinery
proper begins from k = 1. This deviation is documented in the
selection path (init_mi column).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.utils.reproducibility import DEFAULT_SEED


@dataclass
class RFAResult:
    """Output of an RFA run.

    Attributes
    method
        Name of the RFA variant: "rf_macro_f1" or
        "svm_cost_function".
    selected
        Feature names in the order they were added (best first).
    selection_path
        One row per iteration. Columns include k and feature_added;
        method-specific columns are val_macro_f1/improvement (RF) or
        dj/init_mi (SVM).
    ranking
        Two columns: feature and rfa_rank (1 = best, NaN for
        features not added if a stopping rule fired before they were).
    hyperparams
        The hyperparameters used by the run - saved alongside the result so
        the run is self-describing.
    runtime_seconds
        Wall-clock time of the call, in seconds.
    """

    method: str
    selected: list[str]
    selection_path: pd.DataFrame
    ranking: pd.DataFrame
    hyperparams: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "selected": self.selected,
            "selection_path": self.selection_path.to_dict(orient="records"),
            "ranking": self.ranking.to_dict(orient="records"),
            "hyperparams": self.hyperparams,
            "runtime_seconds": self.runtime_seconds,
        }


def _build_ranking(selected: list[str], all_features: list[str]) -> pd.DataFrame:
    """Build the per-feature ranking DataFrame from an ordered selected list."""
    rank_map = {f: i + 1 for i, f in enumerate(selected)}
    rows = [{"feature": f, "rfa_rank": rank_map.get(f, np.nan)} for f in all_features]
    out = pd.DataFrame(rows)
    return out.sort_values("rfa_rank", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Variant 1 - the proposal's RF / val-macro-F1 forward selection
# ---------------------------------------------------------------------------


def rfa_random_forest(
    x_train: pd.DataFrame,
    y_train,
    x_val: pd.DataFrame,
    y_val,
    *,
    max_features: int | None = None,
    patience: int = 3,
    improvement_threshold: float = 0.001,
    n_estimators: int = 100,
    max_depth: int | None = 10,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
    candidate_pool: list[str] | None = None,
    verbose: bool = False,
) -> RFAResult:
    """RFA with a shallow Random Forest evaluator and validation macro-F1.

    At each step, adds the feature whose inclusion gives the largest gain in
    val macro-F1 of a shallow Random Forest trained on the current feature
    subset. Stops when the gains over the last patience additions are all
    below improvement_threshold, or when max_features is reached.

    Parameters
    x_train, y_train
        Training fold (multiclass target attack_cat recommended).
    x_val, y_val
        Validation fold for scoring.
    max_features
        Hard cap on the number of features to add. None -> all features.
    patience
        Stop if every gain in the most recent patience iterations is
        below improvement_threshold.
    improvement_threshold
        The "no meaningful improvement" floor for the patience rule
        (default 0.001, per experimental_protocol.md section 10.2).
    n_estimators, max_depth
        Random-Forest hyperparameters; defaults are "shallow" per the
        proposal.
    candidate_pool
        Restrict the candidate features to this subset (preserves caller
        order). Useful for staged or warm-started runs. None = all
        columns of x_train.
    """
    t0 = time.time()
    all_features = list(x_train.columns)
    if candidate_pool is None:
        remaining = list(all_features)
    else:
        remaining = [f for f in candidate_pool if f in all_features]

    max_features_eff = max_features if max_features is not None else len(remaining)

    y_train_a = np.asarray(y_train)
    y_val_a = np.asarray(y_val)

    selected: list[str] = []
    selection_path: list[dict[str, Any]] = []
    recent_improvements: deque[float] = deque(maxlen=patience)
    best_score = -np.inf

    while remaining and len(selected) < max_features_eff:
        best_candidate: str | None = None
        best_candidate_score = -np.inf

        for f in remaining:
            features = selected + [f]
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            )
            rf.fit(x_train[features], y_train_a)
            preds = rf.predict(x_val[features])
            score = f1_score(y_val_a, preds, average="macro", zero_division=0)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = f

        # Add the winner
        assert best_candidate is not None
        improvement = best_candidate_score - best_score
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        selection_path.append(
            {
                "k": len(selected),
                "feature_added": best_candidate,
                "val_macro_f1": best_candidate_score,
                "improvement": improvement,
            }
        )
        recent_improvements.append(improvement)
        best_score = best_candidate_score

        if verbose:
            print(
                f"[rfa-rf] k={len(selected):3d}: added {best_candidate:<20s}  "
                f"macro-F1={best_candidate_score:.4f}  (Δ={improvement:+.4f})"
            )

        # Patience-based early stopping (only fires once the deque is full)
        if len(recent_improvements) == patience and all(
            d < improvement_threshold for d in recent_improvements
        ):
            if verbose:
                print(
                    f"[rfa-rf] stop: no gain >= {improvement_threshold} "
                    f"over last {patience} additions"
                )
            break

    runtime = time.time() - t0
    return RFAResult(
        method="rf_macro_f1",
        selected=selected,
        selection_path=pd.DataFrame(selection_path),
        ranking=_build_ranking(selected, all_features),
        hyperparams={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "patience": patience,
            "improvement_threshold": improvement_threshold,
            "seed": seed,
        },
        runtime_seconds=runtime,
    )


# ---------------------------------------------------------------------------
# Variant 2 - the original SVM cost-function RFA (Hamed et al. 2018)
# ---------------------------------------------------------------------------


def _compute_scale_gamma(x_arr: np.ndarray) -> float:
    """Reproduce sklearn's gamma='scale' formula explicitly.

    gamma = 1 / (n_features * X.var()). Used so we can compute the
    kernel ourselves with the same gamma sklearn used internally.
    """
    var = x_arr.var()
    if var <= 0:
        return 1.0
    return 1.0 / (x_arr.shape[1] * var)


def rfa_svm_cost_function(
    x_train: pd.DataFrame,
    y_train,
    *,
    max_features: int | None = None,
    subsample: int = 5_000,
    C: float = 1.0,
    seed: int = DEFAULT_SEED,
    candidate_pool: list[str] | None = None,
    verbose: bool = False,
) -> RFAResult:
    """RFA with SVM-RBF cost-function approximation (Hamed, Dara & Kremer 2018).

    The SVM is trained once per *added* feature. Candidate features are
    scored against the *current* α by computing the change in the dual
    objective ½αᵀHα element-wise on the kernel: adding feature f
    multiplies the kernel matrix entrywise by exp(-γ Δ²_f) where
    Δ²_f is the pairwise squared-distance matrix on the new feature.

    The first feature is selected by mutual information with the label
    (Algorithm 2 leaves the empty-set base case unspecified; this is a
    fast, deterministic initialisation - the init_mi column of the
    selection path records it).

    The SVM is binary by construction; y_train must have exactly two
    unique values.

    Parameters
    x_train, y_train
        Training fold; y_train must be binary.
    max_features
        Hard cap on the ranking length. None = rank every feature.
    subsample
        Stratified subsample size for SVM tractability. The kernel matrix
        on support vectors is O(N²); the default of 5,000 keeps a typical
        run under an hour on a laptop. Set None to use the full pool
        (not recommended for >10k rows).
    C
        SVM regularisation parameter.
    seed
        Random seed for the subsample and the SVM.
    candidate_pool
        Optional subset of feature names to rank. None = all columns.
    """
    t0 = time.time()
    all_features = list(x_train.columns)
    candidates = (
        list(all_features)
        if candidate_pool is None
        else [f for f in candidate_pool if f in all_features]
    )

    y_arr = np.asarray(y_train)
    classes = np.unique(y_arr)
    if len(classes) != 2:
        raise ValueError(
            "rfa_svm_cost_function expects a binary target; "
            f"got {len(classes)} unique label values: {classes.tolist()}"
        )

    # Stratified subsample for tractability.
    if subsample is not None and len(x_train) > subsample:
        _, x_eff, _, y_eff = train_test_split(
            x_train,
            y_arr,
            test_size=subsample,
            stratify=y_arr,
            random_state=seed,
        )
        x_eff = x_eff.reset_index(drop=True)
        y_eff = np.asarray(y_eff)
    else:
        x_eff = x_train.reset_index(drop=True)
        y_eff = y_arr

    if verbose:
        print(
            f"[rfa-svm] subsample={len(x_eff)} (stratified)  "
            f"candidates={len(candidates)}  C={C}  seed={seed}"
        )

    selected: list[str] = []
    selection_path: list[dict[str, Any]] = []

    # --- initialisation -----------------------------------------------------
    if verbose:
        print("[rfa-svm] init: picking first feature by mutual information")
    mi = mutual_info_classif(x_eff[candidates], y_eff, random_state=seed, n_neighbors=3)
    first_idx = int(np.argmax(mi))
    first_feature = candidates[first_idx]
    selected.append(first_feature)
    candidates.remove(first_feature)
    selection_path.append(
        {
            "k": 1,
            "feature_added": first_feature,
            "dj": np.nan,
            "init_mi": float(mi[first_idx]),
        }
    )
    if verbose:
        print(f"[rfa-svm] k=  1: added {first_feature:<20s}  MI={mi[first_idx]:.4f}  (init)")

    max_features_eff = max_features if max_features is not None else len(all_features)

    # --- main loop ----------------------------------------------------------
    while candidates and len(selected) < max_features_eff:
        x_sel = x_eff[selected].to_numpy(dtype=np.float64)
        gamma = _compute_scale_gamma(x_sel)

        clf = SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed)
        clf.fit(x_sel, y_eff)

        sv = x_eff.iloc[clf.support_].reset_index(drop=True)
        a_y = clf.dual_coef_.ravel().astype(np.float64)  # shape (n_sv,), already alpha * y
        n_sv = len(sv)

        # Current kernel quadratic form J = 0.5 * a_y^T K a_y
        sv_sel = sv[selected].to_numpy(dtype=np.float64)
        k_current = rbf_kernel(sv_sel, sv_sel, gamma=gamma)
        j_current = 0.5 * float(a_y @ k_current @ a_y)

        # Score each candidate by the entrywise kernel update
        best_dj = -np.inf
        best_feature: str | None = None
        for f in candidates:
            f_vals = sv[f].to_numpy(dtype=np.float64)
            d_f = (f_vals[:, None] - f_vals[None, :]) ** 2
            e_f = np.exp(-gamma * d_f)
            k_new = k_current * e_f
            j_new = 0.5 * float(a_y @ k_new @ a_y)
            dj = j_current - j_new
            if dj > best_dj:
                best_dj = dj
                best_feature = f

        assert best_feature is not None
        selected.append(best_feature)
        candidates.remove(best_feature)
        selection_path.append(
            {
                "k": len(selected),
                "feature_added": best_feature,
                "dj": float(best_dj),
                "init_mi": np.nan,
            }
        )

        if verbose:
            print(
                f"[rfa-svm] k={len(selected):3d}: added {best_feature:<20s}  "
                f"DJ={best_dj:.6f}  |SV|={n_sv}"
            )

    runtime = time.time() - t0
    return RFAResult(
        method="svm_cost_function",
        selected=selected,
        selection_path=pd.DataFrame(selection_path),
        ranking=_build_ranking(selected, all_features),
        hyperparams={
            "subsample": min(subsample, len(x_train)) if subsample else len(x_train),
            "C": C,
            "kernel": "rbf",
            "gamma": "scale (computed)",
            "seed": seed,
        },
        runtime_seconds=runtime,
    )


# ---------------------------------------------------------------------------
# Selection-path utilities (WBS 5.2.2)
# ---------------------------------------------------------------------------


def get_selection_path_dataframe(result: RFAResult) -> pd.DataFrame:
    """Return the selection-path DataFrame for a result (a defensive copy)."""
    return result.selection_path.copy()


def plot_selection_path(
    result: RFAResult,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot RFA's per-iteration score against the number of features added.

    * RF variant: validation macro-F1 vs. k.
    * SVM variant: per-step ΔJ vs. k (init row is omitted since it has no
      ΔJ; the score is the change in dual objective, not a cumulative one).

    Saves to save_path if provided. Uses the project's academic style.
    """
    import matplotlib.pyplot as plt  # local import to keep matplotlib optional

    from src.visualization.style import set_academic_style

    set_academic_style()

    path = result.selection_path.copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    if result.method == "rf_macro_f1":
        ax.plot(path["k"], path["val_macro_f1"], marker="o", color="#4c72b0")
        ax.set_xlabel("Features added (k)")
        ax.set_ylabel("Validation macro-F1")
        ax.set_title("RFA (RF / val-macro-F1) — selection path")
    elif result.method == "svm_cost_function":
        valid = path.dropna(subset=["dj"])
        ax.plot(valid["k"], valid["dj"], marker="o", color="#c44e52")
        ax.set_xlabel("Features added (k)")
        ax.set_ylabel(r"$\Delta J = \frac{1}{2} \alpha^T (H - H_{+i}) \alpha$")
        ax.set_title("RFA (SVM cost function) — selection path")
    else:
        raise ValueError(f"unknown method: {result.method!r}")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[rfa] selection-path plot -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)
