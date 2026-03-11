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


def _patch_synthseg_numpy_compat(synthseg_dir: str) -> None:
    """Patch SynthSeg source for NumPy >= 2.0 compatibility.

    Replaces removed ``np.int``, ``np.float``, ``np.bool``, ``np.complex``,
    ``np.int_``, ``np.float_``, ``np.bool_``, ``np.str_``, ``np.object_``
    aliases with their surviving equivalents (``np.int64``, ``np.float64``,
    ``np.bool_`` → ``bool``, etc.).
    """
    import re

    # Order matters — longer patterns first so np.float_ is matched
    # before np.float, and np.int_ before np.int.
    _REPLACEMENTS = [
        # numpy 2.0 removals  (np.*_ forms)
        (r'\bnp\.int_\b',     'np.int64'),
        (r'\bnp\.float_\b',   'np.float64'),
        (r'\bnp\.bool_\b',    'bool'),
        (r'\bnp\.complex_\b', 'np.complex128'),
        (r'\bnp\.str_\b',     'str'),
        (r'\bnp\.object_\b',  'object'),
        # numpy 1.24 removals (bare np.* forms, no trailing _)
        (r'\bnp\.int\b',      'np.int64'),
        (r'\bnp\.float\b',    'np.float64'),
        (r'\bnp\.bool\b',     'bool'),
        (r'\bnp\.complex\b',  'np.complex128'),
        (r'\bnp\.str\b',      'str'),
        (r'\bnp\.object\b',   'object'),
    ]

    patched_files = 0
    for root, _dirs, files in os.walk(synthseg_dir):
        for fname in files:
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(root, fname)
            try:
                text = open(fpath, 'r', encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            new_text = text
            for pattern, repl in _REPLACEMENTS:
                new_text = re.sub(pattern, repl, new_text)
            if new_text != text:
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(new_text)
                patched_files += 1
    if patched_files:
        print(f"[install] Patched {patched_files} SynthSeg file(s) for NumPy compat.")


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

    # Patch for NumPy >= 1.24 (np.int / np.float removed)
    _patch_synthseg_numpy_compat(synthseg_dir)

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

    # Provision SynthSeg 2.0 model weights from FreeSurfer
    _models_ok = _provision_synthseg_models(
        os.path.join(synthseg_dir, "models")
    )
    if not _models_ok:
        print("[install] Continuing without 2.0 models — SynthSeg will fall back to 1.0.")

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
        print("[install] You may need to install tensorflow manually.")
        return synthseg_dir  # return path anyway — can retry later


_SYNTHSEG_MODEL_FILES = [
    "synthseg_robust_2.0.h5",   # 205 MB — robust segmentation
    "synthseg_2.0.h5",          #  51 MB — fast segmentation
    "synthseg_parc_2.0.h5",     #  51 MB — cortical parcellation
    "synthseg_qc_2.0.h5",       #  48 MB — QC scores
]

_SYNTHSEG_RELEASE_BASE = (
    "https://github.com/edtireli/p-brain/releases/download/synthseg-models-v2.0"
)


def _download_synthseg_models(model_dir: str) -> bool:
    """Download SynthSeg 2.0 model weights from the p-brain GitHub release.

    Returns ``True`` if at least the robust model was downloaded.
    """
    import urllib.request
    import urllib.error

    os.makedirs(model_dir, exist_ok=True)
    ok = False
    for fname in _SYNTHSEG_MODEL_FILES:
        dst = os.path.join(model_dir, fname)
        if os.path.isfile(dst):
            continue
        url = f"{_SYNTHSEG_RELEASE_BASE}/{fname}"
        print(f"[install] Downloading {fname} …")
        try:
            urllib.request.urlretrieve(url, dst, reporthook=_dl_progress)
            print()  # newline after progress
            if os.path.isfile(dst) and os.path.getsize(dst) > 1_000_000:
                print(f"[install]   ✓ {fname}  ({os.path.getsize(dst) // 1048576} MB)")
                if fname == "synthseg_robust_2.0.h5":
                    ok = True
            else:
                print(f"[install]   ✗ {fname} — file too small, removing")
                os.remove(dst)
        except (urllib.error.URLError, OSError) as exc:
            print(f"\n[install]   ✗ {fname} — download failed: {exc}")
            if os.path.isfile(dst):
                os.remove(dst)
    return ok


def _dl_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Progress callback for ``urllib.request.urlretrieve``."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded // 1048576
        mb_total = total_size // 1048576
        print(f"\r[install]   {pct:3d}%  ({mb_done}/{mb_total} MB)", end="", flush=True)
    else:
        print(f"\r[install]   {downloaded // 1048576} MB downloaded", end="", flush=True)


def _provision_synthseg_models(model_dir: str) -> bool:
    """Copy SynthSeg 2.0 model weights into *model_dir*.

    Search order:
      1. Already present in *model_dir*
      2. ``SYNTHSEG_MODELS`` environment variable
      3. FreeSurfer installations (macOS / Linux / Windows)
      4. Sibling ``synthseg_models`` directory next to p-brain
      5. **Download from GitHub Release** (no auth required, ~355 MB)

    Returns ``True`` if at least the robust model was provisioned.
    """
    import shutil
    import pathlib

    os.makedirs(model_dir, exist_ok=True)

    # Already have the robust model?
    if os.path.isfile(os.path.join(model_dir, "synthseg_robust_2.0.h5")):
        return True

    # ── Locate candidate source directories ──
    source_dirs: list[str] = []

    # 0. SYNTHSEG_MODELS env — user can point directly at a directory
    #    containing the .h5 files (e.g. on a USB drive)
    _sm_env = os.environ.get("SYNTHSEG_MODELS", "")
    if _sm_env and os.path.isdir(_sm_env):
        source_dirs.append(_sm_env)

    # 1. FREESURFER_HOME env
    fs_home = os.environ.get("FREESURFER_HOME", "")
    if fs_home:
        d = os.path.join(fs_home, "models")
        if os.path.isdir(d):
            source_dirs.append(d)

    # 2. Common macOS install paths
    _fs_base = pathlib.Path("/Applications/freesurfer")
    if _fs_base.is_dir():
        for ver_dir in sorted(_fs_base.iterdir(), reverse=True):
            d = ver_dir / "models"
            if d.is_dir():
                source_dirs.append(str(d))

    # 3. Common Linux paths
    for prefix in ["/usr/local/freesurfer", "/opt/freesurfer",
                   os.path.expanduser("~/freesurfer")]:
        d = os.path.join(prefix, "models")
        if os.path.isdir(d):
            source_dirs.append(d)

    # 4. Windows — common FreeSurfer / portable model locations
    if sys.platform == "win32":
        for drive in ["C", "D", "E", "F"]:
            for candidate in [
                f"{drive}:\\freesurfer\\models",
                f"{drive}:\\FreeSurfer\\models",
                f"{drive}:\\synthseg_models",
            ]:
                if os.path.isdir(candidate):
                    source_dirs.append(candidate)

    # 5. Alongside the p-brain repo  (../synthseg_models)
    _pbrain_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _sibling = os.path.join(_pbrain_parent, "synthseg_models")
    if os.path.isdir(_sibling):
        source_dirs.append(_sibling)

    # ── Copy from first valid source ──
    for fs_dir in source_dirs:
        robust = os.path.join(fs_dir, "synthseg_robust_2.0.h5")
        if os.path.isfile(robust):
            print(f"[install] Found FreeSurfer models at {fs_dir}")
            for fname in _SYNTHSEG_MODEL_FILES:
                src = os.path.join(fs_dir, fname)
                dst = os.path.join(model_dir, fname)
                if os.path.isfile(src) and not os.path.isfile(dst):
                    print(f"[install]   Copying {fname} ({os.path.getsize(src) // 1048576} MB) ...")
                    shutil.copy2(src, dst)
            print("[install] SynthSeg 2.0 model weights provisioned.")
            return True

    # ── No local source found — download from GitHub Release ──
    print("[install] No local SynthSeg 2.0 models found — downloading from GitHub …")
    print("[install] (Total ~355 MB, this is a one-time download)")
    if _download_synthseg_models(model_dir):
        print("[install] SynthSeg 2.0 model weights downloaded successfully.")
        return True

    print("[install] ⚠ Automatic download failed.")
    print("[install]   You can download the models manually from:")
    print(f"[install]   {_SYNTHSEG_RELEASE_BASE}")
    print(f"[install]   and place them in: {model_dir}")
    print("[install]")
    print("[install]   Or set SYNTHSEG_MODELS=/path/to/folder/with/h5/files")
    return False


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
    parc: bool = True,
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
        Also output cortical parcellation.  Default True.
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
        model_file = "synthseg_robust_2.0.h5"
    else:
        model_file = "synthseg_2.0.h5"

    model_path = os.path.join(model_dir, model_file)
    if not os.path.isfile(model_path):
        # Attempt to provision 2.0 models from FreeSurfer
        _provision_synthseg_models(model_dir)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"SynthSeg model not found: {model_path}\n"
            "The SynthSeg 2.0 model weights are not included in the git repo.\n"
            "They ship with FreeSurfer 7.4+. Install FreeSurfer or manually\n"
            "copy the .h5 files into: {model_dir}"
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
