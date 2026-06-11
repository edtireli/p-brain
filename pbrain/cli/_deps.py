"""Dependency checker for ``pbrain``.

Invoked via ``python -m pbrain check-deps``. Walks the required third-party
imports, reports which are missing, and offers to pip-install them in the
current Python environment.

The required-vs-optional split:

* **Required** for any pipeline run: numpy, scipy, matplotlib, nibabel.
* **Optional**:
    - ``tensorflow``  → CNN AIF plug-in (``aif.cnn`` / ``aif.cnn_sss_shifted``).
                        On Apple Silicon, install ``tensorflow-metal`` as well
                        to enable MPS acceleration.
    - ``torch``       → enables ``--device mps`` / ``--device cuda`` device
                        probing.
    - ``pydicom``     → only when the DICOM loader is selected.

The checker treats optional deps as soft warnings — missing them does NOT
fail the run unless the user actually selects a plug-in that needs them.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class _Dep:
    module: str       # module name to ``importlib.import_module``
    pip_name: str     # name to pass to ``pip install``
    required: bool
    purpose: str


_DEPS: tuple[_Dep, ...] = (
    _Dep("numpy",      "numpy",      True,  "core arrays / math"),
    _Dep("scipy",      "scipy",      True,  "linear algebra, signal processing, interpolation"),
    _Dep("matplotlib", "matplotlib", True,  "diagnostic plots, montages"),
    _Dep("nibabel",    "nibabel",    True,  "NIfTI / PAR-REC I/O"),
    _Dep("dipy",       "dipy",       False, "diffusion models (DTI, DKI, CSD, FW-DTI, NODDI)"),
    _Dep("amico",      "dmri-amico", False, "fast NODDI via AMICO (microstructure)"),
    _Dep("tensorflow", "tensorflow", False, "CNN AIF plug-in (aif.cnn, aif.cnn_sss_shifted)"),
    _Dep("torch",      "torch",      False, "MPS / CUDA device probing for --device flag"),
    _Dep("pydicom",    "pydicom",    False, "DICOM input loader"),
    _Dep("yaml",       "pyyaml",     False, "YAML config files (--config *.yaml; TOML built in)"),
)


def _try_import(module: str) -> tuple[bool, str | None]:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "?")
        return True, str(ver)
    except Exception:
        return False, None


def check_and_install() -> int:
    """Report missing deps and (interactively) offer to pip-install them."""
    print("== pbrain dependency check ==\n")

    missing_required: list[_Dep] = []
    missing_optional: list[_Dep] = []
    for dep in _DEPS:
        ok, ver = _try_import(dep.module)
        tag = "REQ " if dep.required else "opt "
        if ok:
            print(f"  [✓] {tag}{dep.module:<14} ({ver})   — {dep.purpose}")
        else:
            print(f"  [ ] {tag}{dep.module:<14}            — {dep.purpose}")
            (missing_required if dep.required else missing_optional).append(dep)

    if not missing_required and not missing_optional:
        print("\nall dependencies present — you're set.")
        return 0

    if missing_required:
        print(f"\n{len(missing_required)} required dep(s) missing: "
              f"{', '.join(d.pip_name for d in missing_required)}")
    if missing_optional:
        print(f"{len(missing_optional)} optional dep(s) missing: "
              f"{', '.join(d.pip_name for d in missing_optional)}")

    to_install = missing_required + missing_optional
    if not sys.stdin.isatty():
        print("\nnon-interactive shell — re-run `python -m pbrain check-deps` "
              "in a terminal to install, or do `pip install " +
              " ".join(d.pip_name for d in to_install) + "` manually.")
        return 1 if missing_required else 0

    answer = input(f"\npip install {' '.join(d.pip_name for d in to_install)}? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("aborted — nothing installed.")
        return 1 if missing_required else 0

    cmd = [sys.executable, "-m", "pip", "install", *[d.pip_name for d in to_install]]
    print(f"\n$ {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"\npip install failed (exit code {proc.returncode}).")
        return proc.returncode
    print("\ndone — re-running check…\n")
    return check_and_install()
