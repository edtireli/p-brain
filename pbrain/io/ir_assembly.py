"""Assemble an inversion-/saturation-recovery series from separate per-TI files.

Some scanners export the recovery experiment as one volume per inversion time
(``…TI_00120…``, ``…TI_00300…``, …) instead of a single 4-D stack — and in
whatever format the scanner produced: **NIfTI, PAR/REC, or DICOM**.
:func:`assemble_ir` finds those per-TI volumes (by filename for NIfTI, by the
PAR "Protocol name" header otherwise), converts non-NIfTI inputs to NIfTI with
``dcm2niix``, orders them by TI, and writes a single 4-D NIfTI the T1/M0 fitter
consumes. The TI values come from the file/protocol names; when they match the
pipeline's ``inversion_times_ms`` the downstream stage uses those directly.

Usage::

    from pbrain.io.ir_assembly import assemble_ir
    result = assemble_ir(subject_raw_dir, out_path)   # searches NIfTI + PAR/REC
    if result: ir_path, tis_ms = result
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# TI (ms) in a NIfTI filename or a PAR protocol name; skip phase/imaginary.
_TI_RE = re.compile(r"TI[_\- ]?(\d{3,5})", re.I)
_EXCLUDE = re.compile(r"imag|phase|real|_ph\b", re.I)


def _ti_from_par(par: Path) -> int | None:
    """Read a PAR header's Protocol name and pull the TI (ms) out of it."""
    try:
        head = par.read_text(errors="ignore")[:4000]
    except Exception:
        return None
    m = re.search(r"Protocol name\s*:\s*(.+)", head)
    name = m.group(1) if m else ""
    if _EXCLUDE.search(name):
        return None
    t = _TI_RE.search(name)
    return int(t.group(1)) if t else None


def find_ir_files(search_dir: Path | str) -> list[tuple[int, Path]]:
    """Return ``[(ti_ms, path), …]`` for the per-TI volumes, sorted by TI.

    Prefers already-converted NIfTI (filename-based); falls back to PAR/REC
    (protocol-name-based). ``search_dir`` may be the subject root or its NIfTI
    sub-folder — both are searched.
    """
    roots = [Path(search_dir)]
    nd = Path(search_dir) / "NIfTI"
    if nd.is_dir():
        roots.append(nd)

    found: dict[int, Path] = {}
    # NIfTI first (no conversion needed).
    for root in roots:
        for f in sorted(root.glob("*.nii")) + sorted(root.glob("*.nii.gz")):
            if f.name.startswith("._") or _EXCLUDE.search(f.name):
                continue
            m = _TI_RE.search(f.stem)
            if m:
                found.setdefault(int(m.group(1)), f)
    if len(found) >= 3:
        return sorted(found.items())
    # PAR/REC fallback (protocol name carries the TI).
    for root in roots:
        for par in sorted(root.glob("*.PAR")) + sorted(root.glob("*.par")):
            if par.name.startswith("._"):
                continue
            ti = _ti_from_par(par)
            if ti is not None:
                found.setdefault(ti, par)
    return sorted(found.items())


def _load_vol(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a per-TI volume as (3-D magnitude array, affine), converting
    PAR/REC or DICOM with dcm2niix when needed."""
    import nibabel as nib

    if path.suffix.lower() in (".nii",) or path.name.endswith(".nii.gz"):
        img = nib.load(str(path))
    else:
        if not shutil.which("dcm2niix"):
            raise RuntimeError("dcm2niix not on PATH — needed to convert PAR/REC "
                               "or DICOM IR volumes.")
        tmp = tempfile.mkdtemp(prefix="pbrain_ir_")
        subprocess.run(["dcm2niix", "-z", "y", "-o", tmp, "-f", "out", str(path)],
                       capture_output=True, text=True, timeout=300)
        niis = sorted(Path(tmp).glob("out*.nii.gz")) or sorted(Path(tmp).glob("out*.nii"))
        if not niis:
            raise RuntimeError(f"dcm2niix produced no NIfTI for {path}")
        img = nib.load(str(niis[0]))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.ndim == 4:                       # magnitude is the first volume
        arr = arr[..., 0]
    return arr, np.asarray(img.affine, dtype=float)


def assemble_ir(search_dir: Path | str, out_path: Path | str
                ) -> tuple[Path, list[int]] | None:
    """Stack the per-TI recovery volumes (any format) into one 4-D NIfTI.
    Returns ``(path, tis_ms)`` or ``None`` if fewer than 3 TIs are found."""
    import nibabel as nib

    files = find_ir_files(search_dir)
    if len(files) < 3:
        return None
    tis = [ti for ti, _ in files]
    vols, affine = [], None
    for _, f in files:
        arr, aff = _load_vol(f)
        if affine is None:
            affine = aff
        vols.append(arr)
    stack = np.stack(vols, axis=-1)             # (X, Y, Z, nTI)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(stack, affine), str(out_path))
    return out_path, tis
