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
import platform
import subprocess
import sys
import warnings
from typing import Literal

Device = Literal["cpu", "mps", "cuda", "auto"]

# tensorflow-metal 1.2.0 (its latest release) is built against TensorFlow ~2.16 and
# its plugin dlopen()s `_pywrap_tensorflow_internal.so`, which TF removed after 2.16
# — so installing it next to a newer TF *breaks `import tensorflow` outright*, not
# just GPU. Never auto-install it above this cap.
_METAL_MAX_TF = (2, 16)


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


def _tf_version() -> tuple[int, int] | None:
    try:
        import tensorflow as tf  # type: ignore
        major, minor = tf.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return None


def _pip_install(pkg: str, log=None) -> bool:
    if log:
        log(f"installing {pkg} …")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg],
                       check=True, capture_output=True, text=True)
        return True
    except Exception as exc:
        if log:
            log(f"install of {pkg} failed: {str(exc)[:120]}")
        return False


def provision_mps(auto_install: bool = False, log=None) -> tuple[bool, str]:
    """Try to make a Metal (MPS) backend available on Apple Silicon, installing only
    what will *actually work*. Returns ``(available, message)``.

    The GPU-capable stages (CNN AIF, SynthSeg) are TensorFlow, and TF on macOS needs
    Apple's ``tensorflow-metal`` plugin — which only loads on TF ≤ 2.16. This never
    installs a package that would break the running environment: on a too-new TF it
    explains and stays on CPU rather than installing a plugin that stops TF importing.
    """
    p = probe()
    if p["mps_torch"] or p["gpu_tf"]:
        return True, ""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, "no Metal GPU on this platform; using cpu."
    tfv = _tf_version()
    cap = f"{_METAL_MAX_TF[0]}.{_METAL_MAX_TF[1]}"
    if tfv is None:
        return False, ("no TensorFlow here — the GPU stages need it; install a "
                       f"Metal-compatible TensorFlow (≤ {cap}) plus tensorflow-metal.")
    if tfv > _METAL_MAX_TF:
        return False, (f"'mps' unavailable here: tensorflow-metal supports TF ≤ {cap}, but "
                       f"this env has TF {tfv[0]}.{tfv[1]}. Using cpu — identical results, "
                       "only slower. Drop --device mps (cpu is the default) to silence, or "
                       f"make a TF {cap} env for the Apple GPU.")
    # TF is Metal-compatible — the plugin is simply missing.
    if not auto_install:
        return False, ("Apple GPU available but tensorflow-metal is not installed; run "
                       "`pbrain setup` or `pip install tensorflow-metal`.")
    if _pip_install("tensorflow-metal", log):
        return False, ("installed tensorflow-metal — re-run to use the Apple GPU (TF loads "
                       "the plugin at import, so this run stays on CPU).")
    return False, "tensorflow-metal install failed; using cpu."


def resolve(device: Device | str | None = "auto", *,
            auto_install: bool = False, log=None) -> str:
    """Canonicalise a device string.

    Inputs: ``"cpu"``, ``"mps"``, ``"cuda"``, ``"auto"`` (or None).
    Returns one of ``"cpu" | "mps" | "cuda"``. Falls back to ``"cpu"`` if the
    requested device is unavailable. With ``auto_install`` it will provision a
    Metal backend when — and only when — that can be done without breaking the
    environment (see :func:`provision_mps`); otherwise it explains why and uses cpu.
    """
    raw = (device or "auto").strip().lower()
    avail = probe()

    def _notify(msg: str) -> None:
        # A clean one-line heads-up through the caller's logger; only fall back to
        # warnings.warn (with its file:line banner) when no logger was supplied —
        # i.e. library use outside the CLI, where there's nothing else to surface it.
        if log is not None:
            log(msg)
        else:
            warnings.warn(msg, RuntimeWarning)

    if raw == "auto":
        if avail["cuda_torch"]:
            return "cuda"
        if avail["mps_torch"] or avail["gpu_tf"]:
            return "mps"
        return "cpu"

    if raw == "cuda":
        if avail["cuda_torch"]:
            return "cuda"
        _notify("'cuda' unavailable: torch.cuda is not present. Using cpu — install a "
                "CUDA build of torch, or drop --device cuda (cpu is the default) to silence.")
        return "cpu"

    if raw == "mps":
        # MPS counts as available if EITHER torch or TF can see a GPU.
        if avail["mps_torch"] or avail["gpu_tf"]:
            return "mps"
        ok, note = provision_mps(auto_install=auto_install, log=log)
        after = probe()
        if ok or after["mps_torch"] or after["gpu_tf"]:
            return "mps"
        if note:
            _notify(note)
        return "cpu"

    if raw == "cpu":
        return "cpu"

    _notify(f"unknown device {device!r}; using cpu.")
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
