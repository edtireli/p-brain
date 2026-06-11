"""DWI loader — accepts NIfTI, PAR/REC, or DICOM, plus the gradient table.

DWI loading needs a 4-D volume **plus** the gradient table (bvals/bvecs), so
:func:`load_dwi` returns a :class:`DWISeries`. It handles **any input format**:

  * NIfTI  — ``<stem>.nii(.gz)`` with FSL ``<stem>.bval`` / ``<stem>.bvec``
             sidecars (the conventional layout);
  * PAR/REC — a Philips ``.PAR`` (diffusion gradients live in the header);
  * DICOM  — a directory of ``.dcm`` slices.

For PAR/REC and DICOM the volume **and** the FSL gradient table are produced on
the fly with ``dcm2niix`` (so DTI runs straight off raw scanner exports — no
manual conversion). The :class:`DiffusionStage` calls this directly.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class DWISeries:
    """A loaded DWI scan + gradient table."""

    data: np.ndarray            # (X, Y, Z, N)
    affine: np.ndarray          # (4, 4)
    bvals: np.ndarray           # (N,) — s/mm²
    bvecs: np.ndarray           # (N, 3) — unit vectors
    voxel_size: tuple[float, float, float]
    n_b0: int                   # count of b≤50 volumes (b=0 in FSL convention)
    shells: tuple[int, ...]     # sorted unique b-value labels (rounded to 10)


def _sidecar(path: Path, suffix: str) -> Path:
    """Find a sidecar by trying both ``.nii`` and ``.nii.gz`` stems."""
    p = Path(path)
    for stem_drop in (".nii.gz", ".nii"):
        if str(p).endswith(stem_drop):
            base = Path(str(p)[: -len(stem_drop)])
            cand = base.with_suffix(suffix)
            if cand.exists():
                return cand
    cand = p.with_suffix(suffix)
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"DWI sidecar not found for {p} (expected {p.stem}{suffix})"
    )


def _convert_dwi(src: Path):
    """Convert a PAR/REC or DICOM DWI to NIfTI + FSL bval/bvec with dcm2niix.

    Returns ``(nii, bval, bvec, tmpdir)`` — the temp dir is returned so the
    caller keeps it alive until the arrays are read into memory. dcm2niix may
    emit several series (e.g. when handed a directory); the diffusion one is
    the output with a ``.bval`` sidecar carrying ≥ 6 directions + a b0.
    """
    if not shutil.which("dcm2niix"):
        raise RuntimeError(
            "dcm2niix not on PATH — required to convert PAR/REC or DICOM DWI. "
            "Install it (`conda install -c conda-forge dcm2niix` or "
            "`brew install dcm2niix`), or pass an already-converted NIfTI."
        )
    tmp = tempfile.TemporaryDirectory(prefix="pbrain_dwi_")
    out = Path(tmp.name)
    cmd = ["dcm2niix", "-z", "y", "-o", str(out), "-f", "%s_%p", str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        tmp.cleanup()
        raise RuntimeError(
            f"dcm2niix failed for {src}: {r.stderr.strip() or r.stdout.strip()}"
        )
    best = None
    for nii in sorted(out.glob("*.nii.gz")) + sorted(out.glob("*.nii")):
        stem = nii.name[:-7] if nii.name.endswith(".nii.gz") else nii.stem
        bval, bvec = out / f"{stem}.bval", out / f"{stem}.bvec"
        if not (bval.exists() and bvec.exists()):
            continue
        b = np.loadtxt(str(bval)).ravel()
        n_dir = int((b > 50).sum())
        if n_dir >= 6 and (b <= 50).any() and (best is None or n_dir > best[3]):
            best = (nii, bval, bvec, n_dir)
    if best is None:
        tmp.cleanup()
        raise RuntimeError(
            f"No diffusion series (NIfTI + bval/bvec with ≥6 directions) found "
            f"after converting {src}."
        )
    return best[0], best[1], best[2], tmp


def load_dwi(
    dwi_path: Path | str,
    *,
    bval_path: Path | str | None = None,
    bvec_path: Path | str | None = None,
    b0_threshold: float = 50.0,
) -> DWISeries:
    """Load a DWI series with its FSL gradient sidecars.

    Parameters
    ----------
    dwi_path
        Path to the 4-D DWI NIfTI.
    bval_path, bvec_path
        Optional explicit paths. If absent, derived from ``dwi_path`` by
        replacing the NIfTI extension with ``.bval`` / ``.bvec``.
    b0_threshold
        b-values at or below this are considered b=0 (s/mm²).

    Returns
    -------
    DWISeries
    """
    import nibabel as nib

    dwi_path = Path(dwi_path)
    if not dwi_path.exists():
        raise FileNotFoundError(f"DWI file not found: {dwi_path}")

    # PAR/REC or DICOM → convert (volume + bval/bvec) with dcm2niix. The temp
    # dir is held open until the data + gradients are read into memory below.
    _tmp = None
    suffix = dwi_path.suffix.lower()
    if dwi_path.is_dir() or suffix in (".par", ".rec", ".dcm", ".ima"):
        nii, bval_path, bvec_path, _tmp = _convert_dwi(dwi_path)
        dwi_path = nii
    else:
        bval_path = Path(bval_path) if bval_path else _sidecar(dwi_path, ".bval")
        bvec_path = Path(bvec_path) if bvec_path else _sidecar(dwi_path, ".bvec")

    img = nib.load(str(dwi_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(
            f"DWI volume must be 4-D, got shape {data.shape}. "
            f"Did you point at the wrong file?"
        )

    bvals = np.loadtxt(str(bval_path)).astype(float).ravel()
    bvecs_raw = np.loadtxt(str(bvec_path)).astype(float)
    if bvecs_raw.shape[0] == 3:                # FSL row-major (3, N)
        bvecs = bvecs_raw.T
    else:
        bvecs = bvecs_raw                       # (N, 3)
    bvecs = np.atleast_2d(bvecs)

    if bvals.size != data.shape[-1] or bvecs.shape[0] != data.shape[-1]:
        raise ValueError(
            f"Gradient table mismatch: data has {data.shape[-1]} directions "
            f"but bvals={bvals.size}, bvecs={bvecs.shape[0]}."
        )

    # Normalise bvecs (FSL convention requires unit length; tolerate scaling).
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    bvecs = bvecs / norms

    n_b0 = int(np.sum(bvals <= b0_threshold))
    shells = tuple(sorted({int(np.round(b, -1)) for b in bvals}))

    voxel_size = tuple(float(z) for z in img.header.get_zooms()[:3])

    series = DWISeries(
        data=data,
        affine=np.asarray(img.affine, dtype=float),
        bvals=bvals,
        bvecs=bvecs,
        voxel_size=voxel_size,
        n_b0=n_b0,
        shells=shells,
    )
    if _tmp is not None:                # arrays are now in memory — drop the temp
        _tmp.cleanup()
    return series
