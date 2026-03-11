"""
SynthSeg segmentation wrapper for Windows.

Provides a pure-Python brain segmentation path using SynthSeg
(https://github.com/BBillot/SynthSeg).  No FreeSurfer or FastSurfer
binaries are required.

SynthSeg is used via its Python API: we import ``SynthSeg.predict_synthseg``
directly and call ``predict()``.  This requires that the SynthSeg repo
has been cloned and its models downloaded.

Setup (one-time):
    git clone https://github.com/BBillot/SynthSeg.git
    # Download models per SynthSeg README
    pip install tensorflow nibabel numpy matplotlib

Usage from the Windows CLI:
    python -m pbrain_windows --id <subject> --synthseg-home /path/to/SynthSeg

If ``--synthseg-home`` is not given, the env var ``SYNTHSEG_HOME`` is used,
falling back to ``<p-brain>/vendor/SynthSeg``.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def _resolve_synthseg_home(hint: Optional[str] = None) -> str:
    """Return the root of the SynthSeg repository.

    If SynthSeg is not found, attempts automatic installation via
    ``git clone`` + ``pip install``.
    """
    candidates = [
        hint,
        os.environ.get("SYNTHSEG_HOME"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor", "SynthSeg"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)

    # ── Auto-install SynthSeg ──
    installed = _auto_install_synthseg()
    if installed:
        return installed

    raise FileNotFoundError(
        "SynthSeg repository not found.  Clone it with:\n"
        "  git clone https://github.com/BBillot/SynthSeg.git\n"
        "Then set --synthseg-home or SYNTHSEG_HOME."
    )


def _auto_install_synthseg() -> Optional[str]:
    """Attempt to automatically install SynthSeg (Python package).

    Clones the SynthSeg repo into ``<p-brain>/vendor/SynthSeg`` and
    installs its Python dependencies.

    Returns the SynthSeg home directory on success, ``None`` on failure
    or if the user declines.
    """
    import shutil
    import subprocess

    print()
    print("=" * 60)
    print("  SynthSeg not found on this system.")
    print("  p-brain can install it automatically for you.")
    print("=" * 60)
    try:
        answer = input("[?] Install SynthSeg now? (y/n, default y): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if answer not in ("", "y", "yes"):
        return None

    vendor_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
    synthseg_dir = os.path.join(vendor_dir, "SynthSeg")

    git = shutil.which("git")
    if not git:
        print("[install] git not found — cannot clone SynthSeg.")
        return None

    os.makedirs(vendor_dir, exist_ok=True)

    # Clone
    if not os.path.isdir(synthseg_dir):
        print("[install] Cloning SynthSeg repository ...")
        try:
            res = subprocess.run(
                [git, "clone", "--depth", "1",
                 "https://github.com/BBillot/SynthSeg.git", synthseg_dir],
                capture_output=True, text=True, timeout=300,
            )
            if res.returncode != 0:
                print(f"[install] git clone failed: {res.stderr}")
                return None
            print("[install] SynthSeg cloned successfully.")
        except Exception as exc:
            print(f"[install] git clone failed: {exc}")
            return None

    # Install Python dependencies
    print("[install] Installing SynthSeg Python dependencies ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "tensorflow", "nibabel", "numpy", "matplotlib"],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as exc:
        print(f"[install] pip install warning: {exc}")

    # Verify import
    try:
        if synthseg_dir not in sys.path:
            sys.path.insert(0, synthseg_dir)
        import SynthSeg  # noqa: F401
        print("[install] SynthSeg installation verified.")
        os.environ["SYNTHSEG_HOME"] = synthseg_dir
        return synthseg_dir
    except ImportError:
        print("[install] SynthSeg installed but import failed.")
        print("[install] You may need to download the model weights manually.")
        print("[install] See: https://github.com/BBillot/SynthSeg#readme")
        return synthseg_dir  # return path anyway — models can be added later


def _ensure_synthseg_importable(synthseg_home: str) -> None:
    """Add SynthSeg's root to ``sys.path`` so we can import it."""
    if synthseg_home not in sys.path:
        sys.path.insert(0, synthseg_home)
    # Quick smoke test
    try:
        import SynthSeg  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"Cannot import SynthSeg from {synthseg_home}.  "
            "Ensure the repo is cloned and dependencies installed."
        ) from exc


def run_synthseg(
    input_path: str,
    output_path: str,
    *,
    synthseg_home: Optional[str] = None,
    robust: bool = True,
    fast: bool = False,
    parc: bool = False,
    vol_csv: Optional[str] = None,
    qc_csv: Optional[str] = None,
    cpu: bool = False,
) -> str:
    """Run SynthSeg brain segmentation on a single T1w image.

    Parameters
    ----------
    input_path : str
        Path to the input T1-weighted NIfTI (or MGZ) image.
    output_path : str
        Path where the segmentation label map will be written.
    synthseg_home : str, optional
        Root of the cloned SynthSeg repository.
    robust : bool
        Use SynthSeg-robust (more accurate, slower).  Default True.
    fast : bool
        Skip some post-processing for speed.  Default False.
    parc : bool
        Also output cortical parcellation.  Default False.
    vol_csv : str, optional
        Write region volumes to this CSV file.
    qc_csv : str, optional
        Write QC scores to this CSV file.
    cpu : bool
        Force CPU inference (no GPU).  Default False.

    Returns
    -------
    str
        Path to the generated segmentation file.
    """
    synthseg_home = _resolve_synthseg_home(synthseg_home)
    _ensure_synthseg_importable(synthseg_home)

    # Force CPU if requested (must be set before TF import)
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model_dir = os.path.join(synthseg_home, "models")
    labels_dir = os.path.join(synthseg_home, "data", "labels_classes_priors")

    # Verify models exist
    v1 = False
    if robust:
        fast = True  # SynthSeg-robust always uses fast mode
        model_file = "synthseg_2.0_robust.h5"
    else:
        model_file = "synthseg_2.0.h5"

    model_path = os.path.join(model_dir, model_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"SynthSeg model not found: {model_path}\n"
            "Download the models per the SynthSeg README."
        )

    # Build argument dict matching SynthSeg_predict.py
    from SynthSeg.predict_synthseg import predict

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Label lists
    labels_seg = os.path.join(labels_dir, "synthseg_segmentation_labels_2.0.npy")
    labels_denoiser = os.path.join(labels_dir, "synthseg_denoiser_labels_2.0.npy")
    labels_parcellation = os.path.join(labels_dir, "synthseg_parcellation_labels.npy")
    labels_qc = os.path.join(labels_dir, "synthseg_qc_labels_2.0.npy")
    names_seg = os.path.join(labels_dir, "synthseg_segmentation_names_2.0.npy")
    names_parc = os.path.join(labels_dir, "synthseg_parcellation_names.npy")
    names_qc = os.path.join(labels_dir, "synthseg_qc_names_2.0.npy")
    topology_classes = os.path.join(labels_dir, "synthseg_topological_classes_2.0.npy")
    n_neutral_labels = 19

    # Fall back to v1 labels if v2 not present
    if not os.path.isfile(labels_seg):
        v1 = True
        labels_seg = os.path.join(labels_dir, "synthseg_segmentation_labels.npy")
        labels_qc = os.path.join(labels_dir, "synthseg_qc_labels.npy")
        names_seg = os.path.join(labels_dir, "synthseg_segmentation_names.npy")
        names_qc = os.path.join(labels_dir, "synthseg_qc_names.npy")
        topology_classes = os.path.join(labels_dir, "synthseg_topological_classes.npy")
        n_neutral_labels = 18

    model_parcellation = os.path.join(model_dir, "synthseg_parc_2.0.h5")
    model_qc = os.path.join(model_dir, "synthseg_qc_2.0.h5")

    print(f"[segmentation] Running SynthSeg {'robust' if robust else '2.0'} ...")
    print(f"[segmentation]   input:  {input_path}")
    print(f"[segmentation]   output: {output_path}")

    predict(
        path_images=input_path,
        path_segmentations=output_path,
        path_model_segmentation=model_path,
        labels_segmentation=labels_seg,
        robust=robust,
        fast=fast,
        v1=v1,
        n_neutral_labels=n_neutral_labels,
        labels_denoiser=labels_denoiser,
        path_posteriors=None,
        path_resampled=None,
        path_volumes=vol_csv,
        do_parcellation=parc,
        path_model_parcellation=model_parcellation if parc else None,
        labels_parcellation=labels_parcellation if parc else None,
        path_qc_scores=qc_csv,
        path_model_qc=model_qc if qc_csv else None,
        labels_qc=labels_qc,
        cropping=None,
        names_segmentation=names_seg,
        names_parcellation=names_parc if parc else None,
        names_qc=names_qc if qc_csv else None,
        topology_classes=topology_classes,
    )

    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"SynthSeg completed but no output file was created: {output_path}"
        )

    print(f"[segmentation] SynthSeg completed → {output_path}")
    return output_path
