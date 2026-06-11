"""Accelerator (GPU / MPS / CPU) selection.

Single helper used by every plug-in that *can* benefit from a non-CPU
device. The plug-in reads ``config.device``, calls :func:`resolve` to
canonicalise the value (and probe ``"auto"``), then sets up its own
backend (TensorFlow, PyTorch, etc.) accordingly.

``resolve`` never raises: if a requested accelerator is unavailable
it logs a single warning and returns ``"cpu"`` so plug-ins always have
a working backend.
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

Device = Literal["cpu", "mps", "cuda", "auto"]


def _tf_has_gpu() -> bool:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return False
    try:
        return bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        return False


def _torch_mps_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def probe() -> dict[str, bool]:
    """Snapshot of what's available right now."""
    return {
        "cuda_torch": _torch_cuda_available(),
        "mps_torch": _torch_mps_available(),
        "gpu_tf": _tf_has_gpu(),
    }


def resolve(device: Device | str | None = "auto") -> str:
    """Canonicalise a device string.

    Inputs: ``"cpu"``, ``"mps"``, ``"cuda"``, ``"auto"`` (or None).
    Returns one of ``"cpu" | "mps" | "cuda"``. Logs a warning and
    falls back to ``"cpu"`` if the requested device is unavailable.
    """
    raw = (device or "auto").strip().lower()
    avail = probe()

    if raw == "auto":
        if avail["cuda_torch"]:
            return "cuda"
        if avail["mps_torch"] or avail["gpu_tf"]:
            return "mps"
        return "cpu"

    if raw == "cuda":
        if avail["cuda_torch"]:
            return "cuda"
        warnings.warn(
            "Requested device='cuda' but torch.cuda is not available; "
            "install torch with CUDA or use --device cpu. Falling back to cpu.",
            RuntimeWarning,
        )
        return "cpu"

    if raw == "mps":
        # MPS counts as available if EITHER torch or TF can see a GPU.
        if avail["mps_torch"] or avail["gpu_tf"]:
            return "mps"
        warnings.warn(
            "Requested device='mps' but no MPS-capable backend was found. "
            "Install 'tensorflow-metal' (for the CNN AIF) and/or PyTorch built "
            "with MPS support. Falling back to cpu.",
            RuntimeWarning,
        )
        return "cpu"

    if raw == "cpu":
        return "cpu"

    warnings.warn(f"Unknown device {device!r}; falling back to cpu.",
                  RuntimeWarning)
    return "cpu"


def configure_tf_device(device: str) -> str:
    """Set up TensorFlow's visible devices for a given resolved device.

    Returns the device string TF will actually use. Idempotent across
    repeated calls within a single process (TF rejects re-configuration
    after the runtime is initialised — we just skip in that case).
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return "cpu"

    desired = "cpu" if device == "cpu" else "gpu"
    try:
        if desired == "gpu":
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return "cpu"
            try:
                # Memory growth makes MPS more cooperative with other workloads.
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
            try:
                tf.config.set_visible_devices(gpus, "GPU")
            except Exception:
                pass
            return device  # "mps" or "cuda"
        # cpu request: hide GPUs
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "cpu"
    except Exception:
        return "cpu"
