"""Read training curves back from TensorBoard event files.

Keras 3 logs epoch_loss/epoch_accuracy as tensor summaries rather than
plain scalars, so we decode the tensor protos here. This lets the figure
scripts rebuild the curves from saved event files without retraining.
"""

from __future__ import annotations

import os

import numpy as np


def read_series(run_dir: str, split: str, tag: str) -> np.ndarray | None:
    """Per-epoch values for tag under run_dir/tensorboard/<split>.

    split is "train" or "validation". Returns None if the directory or
    the tag is missing.
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    from tensorboard.util import tensor_util

    d = os.path.join(run_dir, "tensorboard", split)
    if not os.path.isdir(d):
        return None
    ea = EventAccumulator(d, size_guidance={"tensors": 0, "scalars": 0})
    ea.Reload()
    tags = ea.Tags()
    if tag in tags.get("tensors", []):
        pts = [(e.step, float(tensor_util.make_ndarray(e.tensor_proto))) for e in ea.Tensors(tag)]
    elif tag in tags.get("scalars", []):
        pts = [(e.step, float(e.value)) for e in ea.Scalars(tag)]
    else:
        return None
    pts.sort()
    return np.array([v for _, v in pts], dtype=float)


def seed_band(
    series_list: list[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Mean and std across seeds, aligned by epoch index.

    Seeds early-stop at different epochs, so the curves have different
    lengths; shorter ones are right-padded with NaN and aggregated with
    nan-aware stats. Returns (epoch_index, mean, std), or None if empty.
    """
    series = [s for s in series_list if s is not None and len(s)]
    if not series:
        return None
    length = max(len(s) for s in series)
    mat = np.full((len(series), length), np.nan)
    for i, s in enumerate(series):
        mat[i, : len(s)] = s
    return np.arange(length), np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
