"""Global reproducibility utilities.

A single call to :func:`set_global_seed` fixes every source of randomness
the project uses -- Python's ``random`` module, NumPy, and TensorFlow --
so that a given seed always produces the same data splits, the same
network weight initialisation, and the same stratified subsamples.

Why this matters
----------------
The project's research questions are answered by *comparing* models. If
two runs differ only because of random noise, the comparison is not
trustworthy. Fixing the seed makes every result exactly reproducible,
and repeating each experiment across the five :data:`PROJECT_SEEDS`
lets us report ``mean +/- standard deviation`` with bootstrap confidence
intervals rather than a single fragile number (proposal Section 6.3).
"""

from __future__ import annotations

import os
import random

import numpy as np

# The five seeds used for every repeated experiment (proposal Section 6.3).
PROJECT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 101112)

# The default single seed for one-off, non-repeated operations.
DEFAULT_SEED: int = 42


def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Fix every random source the project uses.

    Parameters
    ----------
    seed:
        The integer seed to apply.
    deterministic:
        If ``True``, also ask TensorFlow to use deterministic operations.
        This guarantees identical results across runs at a small speed
        cost; set ``False`` for faster but slightly non-deterministic
        GPU training.

    Returns
    -------
    int
        The seed that was applied (handy for logging).
    """
    # Python hash randomisation (affects set/dict iteration order)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python's own random module
    random.seed(seed)

    # NumPy -- underlies pandas, scikit-learn, imbalanced-learn, ...
    np.random.seed(seed)

    # TensorFlow -- imported lazily so this module is usable without it
    try:
        import tensorflow as tf
    except ImportError:
        return seed

    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - API guard across TF versions
        pass
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - API guard across TF versions
            pass

    return seed


def seeded_runs(seeds: tuple[int, ...] = PROJECT_SEEDS):
    """Yield each project seed in turn, applying it before yielding.

    Use this to repeat an experiment across all five seeds::

        for seed in seeded_runs():
            model = train_model(...)
            record_metrics(seed, evaluate(model))
    """
    for seed in seeds:
        set_global_seed(seed)
        yield seed
