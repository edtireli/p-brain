"""Assemble an inversion-recovery series from separate per-TI NIfTI files.

Some scanners export the IR experiment as one magnitude volume per inversion
time (e.g. ``WIPTI_00120.nii``, ``WIPTI_00300.nii`` … where the number is the
TI in ms) rather than a single 4-D stack. :func:`assemble_ir` finds those
files, orders them by TI, and writes a single 4-D NIfTI the T1/M0 fitter can
consume. The TI values are recovered from the filenames; when they match the
pipeline's ``inversion_times_ms`` the downstream stage uses those directly.

Usage::

    from pbrain.io.ir_assembly import assemble_ir
    ir_path, tis_ms = assemble_ir(subject_nifti_dir, out_path)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Match a TI (ms) anywhere in the stem of an IR magnitude file, excluding the
# imaginary/phase companions some exports include.
_TI_RE = re.compile(r"(?:TI[_-]?)(\d{3,5})", re.I)
_EXCLUDE = re.compile(r"imag|phase|real|_ph\b", re.I)


def find_ir_files(nifti_dir: Path | str) -> list[tuple[int, Path]]:
    """Return ``[(ti_ms, path), …]`` for the per-TI magnitude NIfTIs, sorted."""
    nifti_dir = Path(nifti_dir)
    found: dict[int, Path] = {}
    for f in sorted(nifti_dir.glob("*.nii")) + sorted(nifti_dir.glob("*.nii.gz")):
        if f.name.startswith("._") or _EXCLUDE.search(f.name):
            continue
        m = _TI_RE.search(f.stem)
        if not m:
            continue
        ti = int(m.group(1))
        # First (magnitude) file per TI wins; companions already excluded.
        found.setdefault(ti, f)
    return sorted(found.items())


def assemble_ir(nifti_dir: Path | str, out_path: Path | str
                ) -> tuple[Path, list[int]] | None:
    """Stack the per-TI IR files into one 4-D NIfTI. Returns ``(path, tis_ms)``
    or ``None`` if fewer than 3 TIs are found (can't fit T1)."""
    import nibabel as nib

    files = find_ir_files(nifti_dir)
    if len(files) < 3:
        return None
    tis = [ti for ti, _ in files]
    ref = nib.load(str(files[0][1]))
    vols = []
    for _, f in files:
        arr = np.asarray(nib.load(str(f)).dataobj, dtype=np.float32)
        if arr.ndim == 4:                       # collapse a singleton 4th dim
            arr = arr[..., 0]
        vols.append(arr)
    stack = np.stack(vols, axis=-1)             # (X, Y, Z, nTI)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(stack, ref.affine), str(out_path))
    return out_path, tis
