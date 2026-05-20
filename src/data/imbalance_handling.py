"""Class-imbalance handling for the Multi-Model IDS project.

UNSW-NB15 is severely imbalanced -- after deduplication the training
split holds roughly 60,000 Normal records but only ~120 Worms. Three
complementary tools are provided; an experiment may use one, or compare
several in the ablation study:

* :func:`compute_class_weights` -- inverse-frequency weights for
  cost-sensitive loss. Random Forest, XGBoost and the neural networks all
  accept class weights, and no synthetic data is created.
* :func:`apply_smote` -- SMOTE oversampling of the minority classes.
* :func:`augment_rare_classes` -- stronger oversampling (BorderlineSMOTE,
  SVMSMOTE, or SMOTE+ENN) aimed at the rarest classes.

Leakage rule (proposal Section 5.1; WBS 4.3.3)
----------------------------------------------
Resampling is applied to the **training fold only**. Validation and test
sets must keep their natural class distribution -- oversampling them
would score the model on synthetic points and make the metrics
meaningless. Every function here is intended to be called on training
data only.

Note on SMOTE and one-hot features
----------------------------------
SMOTE interpolates between samples, so on the project's one-hot-encoded
columns it can produce fractional values. This is harmless for the tree
and neural models used here; it is noted for transparency.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.utils.reproducibility import DEFAULT_SEED


def compute_class_weights(y) -> dict:
    """Return a ``{class: weight}`` dict of balanced inverse-frequency weights.

    Rare classes receive proportionally larger weights, so a
    cost-sensitive learner pays more attention to them -- without any
    synthetic data being generated.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def apply_smote(x, y, sampling_strategy="auto", k_neighbors: int = 5, seed: int = DEFAULT_SEED):
    """SMOTE-oversample the TRAINING fold. Returns ``(x_resampled, y_resampled)``.

    ``sampling_strategy="auto"`` lifts every minority class up to the
    majority-class size. On a 500x imbalance this creates a large amount
    of synthetic data, so an experiment may instead pass a float or dict
    to oversample only partially.
    """
    from imblearn.over_sampling import SMOTE

    sampler = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=seed,
    )
    return sampler.fit_resample(x, y)


def augment_rare_classes(x, y, method: str = "borderline", seed: int = DEFAULT_SEED):
    """Oversample with a SMOTE variant tuned for the rarest classes.

    ``method`` is one of:

    * ``"borderline"`` -- BorderlineSMOTE: synthesises points near the
      decision boundary, where the rare classes are hardest to separate.
    * ``"svm"`` -- SVMSMOTE: uses an SVM to decide where to place
      synthetic points.
    * ``"smoteenn"`` -- SMOTE followed by Edited-Nearest-Neighbours
      cleaning, which also drops ambiguous majority-class points.

    Caveat: BorderlineSMOTE and SVMSMOTE synthesise only from minority
    points that lie near the decision boundary. A class as rare as Worms
    (~120 training records) can have *no* points that qualify, in which
    case these methods leave it untouched -- a smoke test confirmed Worms
    was not oversampled by BorderlineSMOTE. Plain :func:`apply_smote`, or
    class weighting, is more reliable for the very rarest classes.

    Returns ``(x_resampled, y_resampled)``.
    """
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE

    samplers = {
        "borderline": BorderlineSMOTE(random_state=seed),
        "svm": SVMSMOTE(random_state=seed),
        "smoteenn": SMOTEENN(random_state=seed),
    }
    if method not in samplers:
        raise ValueError(f"method must be one of {sorted(samplers)}")
    return samplers[method].fit_resample(x, y)
