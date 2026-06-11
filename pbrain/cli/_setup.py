"""``python -m pbrain setup`` — get a machine ready to run, end to end.

A friendly, do-it-for-me wizard for people who just want the pipeline to work.
It checks **everything** a run needs and offers to install what's missing:

1. Python packages (required + the optional ones per plug-in) — via pip.
2. ``dcm2niix`` — needed to read PAR/REC and DICOM (offers a pip install).
3. FreeSurfer / ``mri_synthseg`` — needed for the default segmentation
   backend; large, so it explains and offers only with consent.
4. GPU acceleration availability (informational).

It never installs anything without asking, and prints a clear per-plug-in
readiness summary at the end so you know exactly what you can run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def _ok(msg: str) -> None: print(f"  \033[32m✓\033[0m {msg}")
def _warn(msg: str) -> None: print(f"  \033[33m!\033[0m {msg}")
def _hdr(msg: str) -> None: print(f"\n\033[1m{msg}\033[0m")


def _ask(prompt: str) -> bool:
    try:
        return input(f"  {prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _pip_install(pkgs: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def setup() -> int:
    from pbrain.cli._deps import _DEPS

    print("p-Brain setup — checking everything needed to run.\n"
          "Nothing is installed without your confirmation.")

    # ── 1. Python packages ────────────────────────────────────────────
    _hdr("Python packages")
    missing_req, missing_opt = [], []
    for d in _DEPS:
        try:
            __import__(d.module)
            _ok(f"{d.module} — {d.purpose}")
        except Exception:
            (missing_req if d.required else missing_opt).append(d)
            (_warn if not d.required else _warn)(
                f"{d.module} MISSING ({'required' if d.required else 'optional'}) — {d.purpose}")

    if missing_req and _ask(f"Install {len(missing_req)} required package(s)?"):
        _pip_install([d.pip_name for d in missing_req])
    if missing_opt and _ask(f"Install {len(missing_opt)} optional package(s) "
                            f"(diffusion / CNN-AIF / DICOM / YAML …)?"):
        _pip_install([d.pip_name for d in missing_opt])

    # ── 2. dcm2niix (PAR/REC + DICOM input) ───────────────────────────
    _hdr("Image conversion (PAR/REC & DICOM input)")
    if shutil.which("dcm2niix"):
        _ok("dcm2niix found — PAR/REC and DICOM input supported.")
    else:
        _warn("dcm2niix not found — needed only for PAR/REC or DICOM input "
              "(NIfTI works without it).")
        if _ask("Install dcm2niix via pip?"):
            _pip_install(["dcm2niix"])

    # ── 3. FreeSurfer / SynthSeg (default segmentation) ───────────────
    _hdr("Segmentation backend (tissue parcellation)")
    if shutil.which("mri_synthseg") or shutil.which("recon-all"):
        _ok("FreeSurfer / mri_synthseg found — '--tissue-roi synthseg' ready.")
    else:
        _warn("FreeSurfer not found. The default 'synthseg' and 'fastsurfer' "
              "backends need it. Alternatives that need NO install:")
        print("      • --tissue-roi preloaded   (use a parcellation you already have)")
        print("      • --tissue-roi command     (run any segmentation tool you give)")
        print("      • --tissue-roi voxelwise   (whole-brain mask, no parcellation)")
        print("    FreeSurfer install: https://surfer.nmr.mgh.harvard.edu/fswiki/"
              "DownloadAndInstall  (large; manual — follow the page, then "
              "`source $FREESURFER_HOME/SetUpFreeSurfer.sh`).")

    # ── 4. GPU (optional acceleration) ────────────────────────────────
    _hdr("Acceleration (optional)")
    try:
        from pbrain.core import probe_devices
        probe = probe_devices()
        if any(probe.values()):
            _ok(f"Accelerator available: {probe} — use --device auto.")
        else:
            _warn("No GPU/MPS detected — runs on CPU (fine, just slower).")
    except Exception:
        _warn("Could not probe devices — CPU will be used.")

    # ── 5. Readiness summary ──────────────────────────────────────────
    _hdr("You can run")
    def have(m):
        try: __import__(m); return True
        except Exception: return False
    print(f"  {'✓' if all(have(d.module) for d in _DEPS if d.required) else '✗'} "
          f"core pipeline (Patlak, Tikhonov, T1/M0, AIF)")
    print(f"  {'✓' if have('dipy') else '○'} diffusion (DTI/DKI/CSD/…)  "
          f"{'✓' if shutil.which('dcm2niix') else '○'} PAR-REC/DICOM input  "
          f"{'✓' if have('tensorflow') else '○'} CNN-AIF  "
          f"{'✓' if (shutil.which('mri_synthseg') or shutil.which('recon-all')) else '○'} SynthSeg")
    print("\nRun the phantom demo to confirm: python -m pbrain.demo --clean")
    return 0
