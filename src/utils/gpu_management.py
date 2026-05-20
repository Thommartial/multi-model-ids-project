"""GPU configuration utilities for the NVIDIA GTX 1050 (4 GB).

The training machine has a single 4 GB Pascal-generation GPU. That is a
tight memory budget, so these helpers:

* stop TensorFlow from grabbing all GPU memory at once (memory growth);
* optionally enforce a hard memory ceiling;
* offer mixed-precision training -- with the caveat below.

Gradient checkpointing (trading extra compute for lower memory) is a
per-model technique; it is applied when the deep networks are built in
Parts 6-7, not here.

Mixed precision on Pascal
-------------------------
Mixed-precision (float16) training gives large speed-ups only on GPUs
with Tensor Cores (Volta generation and newer). The GTX 1050 is Pascal
and has **no** Tensor Cores, so float16 may give little or no speed-up
and can even be slower because of the extra cast operations. Always
measure with :func:`benchmark_mixed_precision` before enabling it for
real runs (Task 1.4.2.1).
"""

from __future__ import annotations


def configure_gpu(memory_growth: bool = True, memory_limit_mb: int | None = None) -> bool:
    """Configure the visible GPU(s) for safe use on a small card.

    Returns True if at least one GPU was configured, False otherwise.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("[gpu] TensorFlow not installed -- nothing to configure.")
        return False

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[gpu] No GPU visible -- running on CPU.")
        return False

    for gpu in gpus:
        try:
            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
            if memory_limit_mb is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
                )
        except RuntimeError as exc:
            # Memory settings must be applied before the GPU is initialised.
            print(f"[gpu] could not configure {gpu.name}: {exc}")
    print(f"[gpu] configured {len(gpus)} GPU(s); memory_growth={memory_growth}")
    return True


def enable_mixed_precision() -> None:
    """Enable float16 mixed-precision training.

    See the module docstring: this is unlikely to help on the Pascal
    GTX 1050. Benchmark first.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("[gpu] TensorFlow not installed -- cannot enable mixed precision.")
        return
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("[gpu] mixed-precision policy set to mixed_float16.")


def gpu_summary() -> dict:
    """Return a short summary of the visible compute devices."""
    info: dict = {"tensorflow": None, "gpus": []}
    try:
        import tensorflow as tf
    except ImportError:
        return info
    info["tensorflow"] = tf.__version__
    info["gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
    return info


def benchmark_mixed_precision(epochs: int = 3, samples: int = 20_000) -> dict:
    """Time a small training run with and without mixed precision.

    Task 1.4.2.1: on a Pascal GPU mixed precision may not help. This runs
    a tiny dense model both ways and returns the wall-clock times, so the
    decision is evidence-based rather than assumed. Requires TensorFlow
    and a visible GPU.
    """
    import time

    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        return {"error": "TensorFlow not installed"}

    if not tf.config.list_physical_devices("GPU"):
        return {"error": "no GPU visible"}

    x = np.random.rand(samples, 64).astype("float32")
    y = np.random.randint(0, 2, size=samples)

    def _run() -> float:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64,)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        start = time.time()
        model.fit(x, y, epochs=epochs, batch_size=256, verbose=0)
        return time.time() - start

    tf.keras.mixed_precision.set_global_policy("float32")
    fp32 = _run()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    fp16 = _run()
    tf.keras.mixed_precision.set_global_policy("float32")  # reset

    return {
        "fp32_seconds": round(fp32, 2),
        "mixed_float16_seconds": round(fp16, 2),
        "speedup": round(fp32 / fp16, 2) if fp16 else None,
    }
