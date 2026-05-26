"""Deep-learning Phase-1 models (WBS Part 6.3).

Two Keras model wrappers, both implementing the :class:`BaseModel`
protocol from :mod:`src.evaluation.model_runner` so they slot into the
shared harness like every other Phase-1 model:

* :class:`OneDCNN`  -- a 1-D convolutional network on flat-feature
  records reshaped to ``(n_features, 1)``.
* :class:`LSTMClassifier` -- an LSTM (optionally bi-directional) on
  sliding windows of consecutive flow records (window size is a
  hyperparameter; default 8). Predictions are returned **per row** by
  padding the first ``window_size - 1`` rows with the first available
  window's prediction.

Both wrappers:

* lazily import TensorFlow so the module is fast to import in tests;
* handle string class labels via an internal encoder;
* train with class-weighted loss
  (``imbalance_handling.compute_class_weights``);
* enable early stopping on val macro accuracy / loss + best-on-val
  checkpoint (per protocol §7 / WBS 6.3.4-5);
* support ``save`` / ``load`` round-trips via the native ``.keras``
  format plus a sidecar JSON for the label encoder.

GPU note: :func:`src.utils.gpu_management.configure_gpu` is called once
by the runner to enable memory growth on the GTX 1050 (4 GB). The
wrappers do not call it themselves so tests on CPU stay clean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.reproducibility import DEFAULT_SEED


# ---------------------------------------------------------------------------
# Shared label-encoding helpers
# ---------------------------------------------------------------------------


class _LabelEncoder:
    """Tiny class-label encoder (string/object -> 0..K-1)."""

    def __init__(self) -> None:
        self._to_int: dict | None = None
        self._to_label: list | None = None

    def fit(self, y) -> "_LabelEncoder":
        classes = sorted(set(np.asarray(y).tolist()))
        self._to_int = {c: i for i, c in enumerate(classes)}
        self._to_label = list(classes)
        return self

    @property
    def n_classes(self) -> int:
        if self._to_int is None:
            raise RuntimeError("encoder not fit")
        return len(self._to_int)

    def transform(self, y) -> np.ndarray:
        if self._to_int is None:
            raise RuntimeError("encoder not fit")
        return np.array([self._to_int[c] for c in np.asarray(y)], dtype=np.int32)

    def inverse_transform(self, y_int) -> np.ndarray:
        if self._to_label is None:
            raise RuntimeError("encoder not fit")
        return np.array([self._to_label[i] for i in np.asarray(y_int)])


def _seed_tensorflow(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow random generators."""
    import os
    import random

    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _class_weight_dict(class_weights, encoder: _LabelEncoder) -> dict | None:
    """Translate user-facing ``{label: weight}`` to Keras' ``{int_id: weight}``."""
    if class_weights is None:
        return None
    return {encoder._to_int[c]: float(w) for c, w in class_weights.items() if c in encoder._to_int}  # type: ignore[union-attr]


# ===========================================================================
# 1-D CNN
# ===========================================================================


@dataclass
class OneDCNNConfig:
    conv_blocks: int = 2
    filters_per_block: int = 64
    kernel_size: int = 5
    dropout: float = 0.2
    dense_units: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    early_stopping_patience: int = 5
    optimizer: str = "adam"


class OneDCNN:
    """1-D CNN over flat-feature records.

    Each input row of shape ``(n_features,)`` is reshaped to
    ``(n_features, 1)`` and passed through ``conv_blocks`` blocks of
    Conv1D + BatchNorm + ReLU + MaxPool, then GlobalAveragePool ->
    Dense -> softmax. Adapts the output layer to the number of classes
    seen in ``y_train``.
    """

    def __init__(self, *, seed: int = DEFAULT_SEED, **hparams) -> None:
        self.seed = seed
        self.cfg = OneDCNNConfig(**hparams)
        self._model: Any = None
        self._encoder = _LabelEncoder()

    def _build(self, n_features: int, n_classes: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        inp = layers.Input(shape=(n_features, 1))
        x = inp
        for _ in range(self.cfg.conv_blocks):
            x = layers.Conv1D(self.cfg.filters_per_block, self.cfg.kernel_size, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPool1D(pool_size=2, padding="same")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.cfg.dropout)(x)
        x = layers.Dense(self.cfg.dense_units, activation="relu")(x)
        x = layers.Dropout(self.cfg.dropout)(x)
        out = layers.Dense(n_classes, activation="softmax")(x)

        model = models.Model(inp, out)
        opt = (
            tf.keras.optimizers.Adam(self.cfg.learning_rate)
            if self.cfg.optimizer == "adam"
            else tf.keras.optimizers.AdamW(self.cfg.learning_rate)
        )
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _reshape(self, x) -> np.ndarray:
        arr = x.to_numpy(dtype=np.float32) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]  # (n, n_features, 1)
        return arr

    def fit(self, x_train, y_train, x_val=None, y_val=None, class_weights=None) -> "OneDCNN":
        import tensorflow as tf

        _seed_tensorflow(self.seed)
        self._encoder.fit(y_train)
        y_int = self._encoder.transform(y_train)
        x_arr = self._reshape(x_train)
        n_features = x_arr.shape[1]
        self._model = self._build(n_features, self._encoder.n_classes)

        validation_data = None
        callbacks: list = []
        if x_val is not None and y_val is not None:
            y_val_int = self._encoder.transform(y_val)
            validation_data = (self._reshape(x_val), y_val_int)
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        self._model.fit(
            x_arr,
            y_int,
            validation_data=validation_data,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.max_epochs,
            class_weight=_class_weight_dict(class_weights, self._encoder),
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict_proba(self, x) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict_proba before fit")
        return self._model.predict(self._reshape(x), verbose=0)

    def predict(self, x) -> np.ndarray:
        proba = self.predict_proba(x)
        return self._encoder.inverse_transform(np.argmax(proba, axis=1))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._model.save(p / "model.keras")
        with open(p / "encoder.json", "w") as fh:
            json.dump(
                {
                    "to_int": self._encoder._to_int,
                    "to_label": list(self._encoder._to_label),  # type: ignore[arg-type]
                    "cfg": self.cfg.__dict__,
                    "seed": self.seed,
                },
                fh,
                default=str,
            )

    @classmethod
    def load(cls, path: str | Path) -> "OneDCNN":
        import tensorflow as tf

        p = Path(path)
        with open(p / "encoder.json") as fh:
            meta = json.load(fh)
        m = cls(seed=meta["seed"], **meta["cfg"])
        m._encoder._to_int = {k: int(v) for k, v in meta["to_int"].items()}
        m._encoder._to_label = list(meta["to_label"])
        m._model = tf.keras.models.load_model(p / "model.keras")
        return m


def get_onedcnn(seed: int = DEFAULT_SEED, **hparams) -> OneDCNN:
    return OneDCNN(seed=seed, **hparams)


# ===========================================================================
# LSTM (with internal sliding-window builder)
# ===========================================================================


@dataclass
class LSTMConfig:
    window_size: int = 8
    lstm_layers: int = 1
    hidden_size: int = 128
    bidirectional: bool = False
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    early_stopping_patience: int = 5


def _build_windows(x: np.ndarray, y: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Sliding-window builder; returns (windows, targets, n_padded_at_front).

    Windows are stride-1, contiguous; the target of a window is the label
    of its **last** record (many-to-one).
    """
    n = len(x)
    if n < window_size:
        raise ValueError(f"need at least {window_size} rows, got {n}")
    windows = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=0)
    # shape: (n - W + 1, W, n_features)  -- but stride_tricks returns
    # (n - W + 1, n_features, W); we need to transpose:
    windows = np.transpose(windows, (0, 2, 1))
    targets = y[window_size - 1 :]
    return windows.astype(np.float32), targets, window_size - 1


class LSTMClassifier:
    """LSTM with internal sliding-window builder.

    ``fit`` and ``predict`` accept 2-D ``(n_records, n_features)``
    input; the wrapper builds (n - W + 1) overlapping length-W windows
    inside. Predictions are returned **per input row** by padding the
    first ``W - 1`` rows with the prediction of the first available
    window.
    """

    def __init__(self, *, seed: int = DEFAULT_SEED, **hparams) -> None:
        self.seed = seed
        self.cfg = LSTMConfig(**hparams)
        self._model: Any = None
        self._encoder = _LabelEncoder()

    def _build(self, n_features: int, n_classes: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        inp = layers.Input(shape=(self.cfg.window_size, n_features))
        x = inp
        for i in range(self.cfg.lstm_layers):
            return_seq = i < self.cfg.lstm_layers - 1
            cell = layers.LSTM(self.cfg.hidden_size, return_sequences=return_seq, dropout=self.cfg.dropout)
            x = layers.Bidirectional(cell)(x) if self.cfg.bidirectional else cell(x)
        x = layers.Dropout(self.cfg.dropout)(x)
        out = layers.Dense(n_classes, activation="softmax")(x)

        model = models.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.cfg.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _windows(self, x, y=None):
        arr = x.to_numpy(dtype=np.float32) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=np.float32)
        if y is None:
            # Need windows for predict -- still build them with a dummy y.
            n = len(arr)
            if n < self.cfg.window_size:
                raise ValueError(f"need at least {self.cfg.window_size} rows, got {n}")
            w = np.transpose(
                np.lib.stride_tricks.sliding_window_view(arr, self.cfg.window_size, axis=0),
                (0, 2, 1),
            ).astype(np.float32)
            return w, None, self.cfg.window_size - 1
        return _build_windows(arr, np.asarray(y), self.cfg.window_size)

    def fit(self, x_train, y_train, x_val=None, y_val=None, class_weights=None) -> "LSTMClassifier":
        import tensorflow as tf

        _seed_tensorflow(self.seed)
        self._encoder.fit(y_train)
        y_int = self._encoder.transform(y_train)

        x_tr_w, y_tr_w, _ = self._windows(x_train, y_int)
        n_features = x_tr_w.shape[2]
        self._model = self._build(n_features, self._encoder.n_classes)

        validation_data = None
        callbacks: list = []
        if x_val is not None and y_val is not None and len(x_val) >= self.cfg.window_size:
            y_val_int = self._encoder.transform(y_val)
            x_val_w, y_val_w, _ = self._windows(x_val, y_val_int)
            validation_data = (x_val_w, y_val_w)
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        self._model.fit(
            x_tr_w,
            y_tr_w,
            validation_data=validation_data,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.max_epochs,
            class_weight=_class_weight_dict(class_weights, self._encoder),
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict_proba(self, x) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict_proba before fit")
        x_w, _, pad = self._windows(x, None)
        proba = self._model.predict(x_w, verbose=0)
        # pad first `pad` rows with the first prediction so output length == input length
        if pad > 0:
            first = np.broadcast_to(proba[0:1], (pad, proba.shape[1]))
            proba = np.concatenate([first, proba], axis=0)
        return proba

    def predict(self, x) -> np.ndarray:
        proba = self.predict_proba(x)
        return self._encoder.inverse_transform(np.argmax(proba, axis=1))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._model.save(p / "model.keras")
        with open(p / "encoder.json", "w") as fh:
            json.dump(
                {
                    "to_int": self._encoder._to_int,
                    "to_label": list(self._encoder._to_label),  # type: ignore[arg-type]
                    "cfg": self.cfg.__dict__,
                    "seed": self.seed,
                },
                fh,
                default=str,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LSTMClassifier":
        import tensorflow as tf

        p = Path(path)
        with open(p / "encoder.json") as fh:
            meta = json.load(fh)
        m = cls(seed=meta["seed"], **meta["cfg"])
        m._encoder._to_int = {k: int(v) for k, v in meta["to_int"].items()}
        m._encoder._to_label = list(meta["to_label"])
        m._model = tf.keras.models.load_model(p / "model.keras")
        return m


def get_lstm(seed: int = DEFAULT_SEED, **hparams) -> LSTMClassifier:
    return LSTMClassifier(seed=seed, **hparams)
