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

# Every dataset in this study ships this 7-point inversion-recovery series.
# Used to (a) name precisely which TIs are missing and (b) set the default
# expected count the conversion verifies against.
STANDARD_IR_TIS_MS = (120, 300, 600, 1000, 2000, 4000, 10000)


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
    # NIfTI first (no conversion needed) — but skip 0-byte / truncated files a
    # failed dcm2niix run can leave behind, so an empty NIfTI never shadows the
    # intact source PAR/REC for that TI (which the fallback below re-converts).
    for root in roots:
        for f in sorted(root.glob("*.nii")) + sorted(root.glob("*.nii.gz")):
            if f.name.startswith("._") or _EXCLUDE.search(f.name):
                continue
            try:
                if f.stat().st_size < 2048:          # empty/truncated → not usable
                    continue
            except OSError:
                continue
            m = _TI_RE.search(f.stem)
            if m:
                found.setdefault(int(m.group(1)), f)
    # PAR/REC pass (protocol name carries the TI). Always run it: valid NIfTIs
    # already claimed their TIs via setdefault above, so this only *fills gaps*
    # — TIs whose NIfTI was missing or empty (skipped). Reading PAR headers is
    # cheap; conversion happens later only for the gap TIs actually returned.
    for root in roots:
        for par in sorted(root.glob("*.PAR")) + sorted(root.glob("*.par")):
            if par.name.startswith("._"):
                continue
            ti = _ti_from_par(par)
            if ti is not None:
                found.setdefault(ti, par)
    return sorted(found.items())


def _reorient_las(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorient ``(data, affine)`` to LAS+ so a nibabel-converted PAR matches
    dcm2niix output (which always reorients to LAS+). Without this a nibabel
    fallback volume would be flipped relative to dcm2niix volumes in the same
    stack. Ported from legacy ``start.py``."""
    import numpy as _np
    from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation

    aff = _np.asarray(affine, dtype=float).copy()
    ornt = io_orientation(_np.diag([-1, 1, 1, 1]).dot(aff))
    if _np.array_equal(ornt, _np.array([[0, 1], [1, 1], [2, 1]])):
        return data, aff
    t_aff = inv_ornt_aff(ornt, data.shape[:3])
    return apply_orientation(data, ornt), _np.dot(aff, t_aff)


def _pick_dynamic(arr: np.ndarray, dynamic_index: int) -> np.ndarray:
    """Collapse a 4-D per-TI volume to a single dynamic frame.

    Legacy pbrain fits the **2nd dynamic** (``build_voxel_matrix(volume_index=1)``,
    "matching MATLAB case-3"). Frame 0 yields a *different* T1/M0 — on the
    reference subject the vessel T1 is 1601 ms (frame 0) vs 1930 ms (frame 1,
    == legacy), which corrupts the single-voxel AIF magnitude. The per-slice
    T1 map matches legacy on 9/10 slices with frame 1. Default therefore
    follows legacy (``dynamic_index=1``); clamped to the available frames so
    single-dynamic volumes are unaffected.
    """
    if arr.ndim == 4:
        return arr[..., min(int(dynamic_index), arr.shape[-1] - 1)]
    return arr


def _nibabel_parrec_magnitude(par_path: Path, dynamic_index: int = 1) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert a Philips IR PAR/REC to a (3-D magnitude, LAS+ affine) with
    nibabel, tolerant of a **truncated REC** (``permit_truncated=True``) that
    makes dcm2niix emit a 0-byte file. Magnitude-only (image_type_mr==0); the
    fitted dynamic is ``dynamic_index`` (default 1, matching legacy). Returns
    ``None`` if nibabel also cannot read it. Ported from legacy
    ``_nibabel_convert_parrec``."""
    import nibabel as nib

    try:
        img = nib.parrec.load(str(par_path), permit_truncated=True,
                              strict_sort=True, scaling="fp")
    except Exception:                            # noqa: BLE001 — unreadable even truncated
        return None
    try:
        data = np.asanyarray(img.dataobj, dtype=np.float32)
        defs = img.header.image_defs
        if data.ndim == 4 and np.unique(defs["image_type_mr"]).size > 1:
            # strict_sort orders [mag…, real…, imag…]; keep the magnitude block.
            ndyn = max(int(np.unique(defs["dynamic scan number"]).size), 1)
            data = data[..., :ndyn]
        data = _pick_dynamic(data, dynamic_index)   # legacy 2nd dynamic (clamped)
        data, aff = _reorient_las(data, np.asarray(img.affine, dtype=float))
        return np.asarray(data, dtype=np.float32), aff
    except Exception:                            # noqa: BLE001
        return None


def recover_dce_magnitude_from_par(par_path: Path, out_path: Path) -> tuple[int, int] | None:
    """Recover a 4-D **magnitude** DCE time-series from a Philips ``hperf`` PAR/REC
    that dcm2niix refuses (aborted/truncated acquisition → slice count not
    divisible by slices-per-volume). Reads the complete dynamics with nibabel
    ``permit_truncated``, keeps only the magnitude channel (``image_type_mr==0``)
    in dynamic-scan order, reorients to LAS+ (matching dcm2niix), and writes
    ``out_path``. Returns ``(n_dynamics, n_slices)`` on success, else ``None``.

    Mirrors the IR truncated-PAR recovery (:func:`_nibabel_parrec_magnitude`) but
    preserves the time axis instead of collapsing to a single frame. The complete
    final dynamic is the natural cutoff — the aborted partial volume is dropped,
    so no interpolation or zero-padding is introduced."""
    import nibabel as nib

    try:
        img = nib.parrec.load(str(par_path), permit_truncated=True, scaling="fp")
    except Exception:                            # noqa: BLE001 — unreadable even truncated
        return None
    try:
        vl = img.header.get_volume_labels()      # per-4th-axis-volume properties
        itm = np.asarray(vl["image_type_mr"])    # 0=magnitude, 1=real, 2=imaginary
        dyn = np.asarray(vl["dynamic scan number"])
        mag = np.where(itm == 0)[0]
        if mag.size == 0:                        # no labelled magnitude channel
            return None
        order = mag[np.argsort(dyn[mag], kind="stable")]   # temporal order
        arr = np.asanyarray(img.dataobj, dtype=np.float32)
        if arr.ndim != 4:                        # not a 4-D dynamic series
            return None
        arr = arr[..., order]
        arr, aff = _reorient_las(arr, np.asarray(img.affine, dtype=float))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32), aff), str(out_path))
        return int(arr.shape[3]), int(arr.shape[2])
    except Exception:                            # noqa: BLE001
        return None


def _load_vol(path: Path, dynamic_index: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Load a per-TI volume as (3-D magnitude array, affine), converting
    PAR/REC or DICOM with dcm2niix — and, if dcm2niix fails or emits an empty
    file, falling back to nibabel's truncation-tolerant PAR reader.

    ``dynamic_index`` (default 1, matching legacy ``volume_index=1``) selects
    which dynamic frame of a 4-D per-TI volume is fitted; see ``_pick_dynamic``.
    """
    import nibabel as nib

    if path.suffix.lower() in (".nii",) or path.name.endswith(".nii.gz"):
        img = nib.load(str(path))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        arr = _pick_dynamic(arr, dynamic_index)
        return arr, np.asarray(img.affine, dtype=float)

    # PAR/REC (or DICOM): dcm2niix first, verify the output is real, else nibabel.
    img = None
    if shutil.which("dcm2niix"):
        tmp = tempfile.mkdtemp(prefix="pbrain_ir_")
        try:
            subprocess.run(["dcm2niix", "-z", "y", "-o", tmp, "-f", "out", str(path)],
                           capture_output=True, text=True, timeout=300)
        except Exception:                        # noqa: BLE001 — fall through to nibabel
            pass
        niis = sorted(Path(tmp).glob("out*.nii.gz")) or sorted(Path(tmp).glob("out*.nii"))
        mag = [n for n in niis if not _EXCLUDE.search(n.name)]
        for cand in (mag or niis):
            try:
                if cand.stat().st_size < 2048:   # dcm2niix emitted an empty file
                    continue
                img = nib.load(str(cand))
                break
            except Exception:                    # noqa: BLE001
                continue

    if img is not None:
        arr = np.asarray(img.dataobj, dtype=np.float32)
        arr = _pick_dynamic(arr, dynamic_index)
        return arr, np.asarray(img.affine, dtype=float)

    # dcm2niix absent / failed / produced only empty files → nibabel fallback.
    if path.suffix.lower() == ".par" or path.suffix.lower() == ".rec":
        res = _nibabel_parrec_magnitude(path, dynamic_index)
        if res is not None:
            return res
    raise RuntimeError(f"Could not convert IR volume (dcm2niix + nibabel both "
                       f"failed) for {path}")


def _sidecar_path(out_path: Path) -> Path:
    """BIDS-style ``.json`` sidecar path for an IR NIfTI (drops .nii/.nii.gz)."""
    stem = out_path
    while stem.suffix in (".gz", ".nii"):
        stem = stem.with_suffix("")
    return stem.with_suffix(".json")


def assemble_ir(search_dir: Path | str, out_path: Path | str,
                *, expected_tis: int = 7, dynamic_index: int = 1) -> tuple[Path, list[int]] | None:
    """Stack the per-TI recovery volumes (any format) into one 4-D NIfTI.
    Returns ``(path, tis_ms)`` or ``None`` if fewer than 3 *readable* TIs.

    ``expected_tis`` (default 7 — every dataset here ships 7 inversion times)
    is the count the assembly verifies against: if fewer real volumes survive
    conversion+recovery, a **loud warning** is emitted naming the missing TIs,
    so a short IR series never passes silently. It is never padded.

    The **actual** TI of every stacked volume is written to a BIDS-style
    ``InversionTimes`` sidecar (seconds) next to the NIfTI, so the loader sets
    ``axis4_kind="ti"`` and the T1 fit consumes the real per-file TIs directly
    — never the configured-count fallback. This mirrors the legacy pipeline,
    which fits with *whatever* TIs the present files carry.

    Robust to individual TI volumes that fail to load — e.g. a 0-byte NIfTI
    left by a failed ``dcm2niix`` conversion (``ImageFileError: Empty file``),
    or one whose grid disagrees with the rest. Such a volume is **dropped**
    (not fabricated by interpolation); the recovery is simply fit from the
    remaining real TIs, exactly as if the scan had one fewer inversion time.
    Previously a single bad volume aborted the whole subject.
    """
    import json

    import nibabel as nib

    files = find_ir_files(search_dir)
    if len(files) < 3:
        return None

    loaded: list[tuple[int, np.ndarray]] = []
    affine = None
    ref_shape: tuple[int, ...] | None = None
    for ti, f in files:
        try:
            arr, aff = _load_vol(f, dynamic_index=dynamic_index)
        except Exception as exc:                 # noqa: BLE001 — empty/corrupt file
            print(f"  [assemble_ir] dropping TI={ti}ms — unreadable "
                  f"({type(exc).__name__}): {f.name}")
            continue
        if affine is None:
            affine, ref_shape = aff, arr.shape
        if arr.shape != ref_shape:               # grid disagreement — drop it
            print(f"  [assemble_ir] dropping TI={ti}ms — grid {arr.shape} "
                  f"!= {ref_shape}: {f.name}")
            continue
        loaded.append((int(ti), arr))

    if len(loaded) < 3:                          # too few real volumes to trust
        return None

    loaded.sort(key=lambda p: p[0])
    tis = [ti for ti, _ in loaded]               # actual TIs of the stacked vols
    if len(tis) < int(expected_tis):
        all_tis = sorted({ti for ti, _ in files})
        missing = [ti for ti in all_tis if ti not in tis]
        extra = "" if len(all_tis) >= expected_tis else \
            f" (only {len(all_tis)} TI source files were even present)"
        print(f"  [assemble_ir] ⚠ WARNING: assembled {len(tis)}/{expected_tis} "
              f"inversion times{extra}; missing/unrecoverable: "
              f"{missing or 'see source-file count'} in {Path(search_dir).name}. "
              f"Proceeding with the {len(tis)} real TIs — no fabrication.")
    stack = np.stack([a for _, a in loaded], axis=-1)   # (X, Y, Z, nTI)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(stack, affine), str(out_path))

    # Sidecar so the loader/T1 fit use these real TIs (seconds), not the
    # configured count-matched fallback — correct even when a TI was dropped.
    try:
        _sidecar_path(out_path).write_text(
            json.dumps({"InversionTimes": [ti / 1000.0 for ti in tis]})
        )
    except Exception:                            # noqa: BLE001 — sidecar is best-effort
        pass
    return out_path, tis


def ensure_ir_niftis(search_dir: Path | str, *, expected: int = 7) -> dict:
    """Verify the per-TI IR NIfTIs and **actively recover** any that are missing
    or empty by re-converting their source PAR/REC (dcm2niix → nibabel-truncated
    fallback), writing the recovered volumes into the NIfTI dir so the set is
    complete and cached. The conversion never silently proceeds short.

    Returns a status dict::

        {expected, have, from_nifti, recovered, failed, missing, complete}

    Only **real** volumes that exist as PAR/REC are materialised — nothing is
    interpolated or padded. ``complete`` is ``have >= expected``. Idempotent:
    a recovered TI is a valid NIfTI on the next call, so it is not redone.
    """
    import nibabel as nib

    search = Path(search_dir)
    nd = search / "NIfTI"
    out_dir = nd if nd.is_dir() else search
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_ir_files(search)            # valid NIfTI (>2 KB) + PAR gap-fills
    from_nifti = sorted(ti for ti, p in files
                        if p.suffix.lower() not in (".par", ".rec"))
    par_gap = [(ti, p) for ti, p in files
               if p.suffix.lower() in (".par", ".rec")]

    recovered: list[int] = []
    failed: list[int] = []
    for ti, par in par_gap:
        try:
            arr, aff = _load_vol(par)        # dcm2niix → nibabel(permit_truncated)
            out = out_dir / f"WIPTI_{int(ti):05d}_recovered.nii.gz"
            nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32), aff), str(out))
            recovered.append(int(ti))
        except Exception as exc:             # noqa: BLE001 — genuinely unconvertible
            failed.append(int(ti))
            print(f"  [ensure_ir] could NOT recover TI={ti}ms from {par.name}: "
                  f"{type(exc).__name__}")

    have = sorted(set(from_nifti) | set(recovered))
    missing = [ti for ti in STANDARD_IR_TIS_MS if ti not in have]
    if len(have) < int(expected) and len(STANDARD_IR_TIS_MS) != int(expected):
        # non-standard expected count → fall back to a plain shortfall list
        missing = missing or [f"<{int(expected) - len(have)} unaccounted>"]
    status = {
        "expected": int(expected), "have": len(have),
        "from_nifti": from_nifti, "recovered": sorted(recovered),
        "failed": sorted(failed), "missing": missing,
        "complete": len(have) >= int(expected),
    }
    if recovered:
        print(f"  [ensure_ir] recovered {len(recovered)} TI(s) from PAR/REC "
              f"{sorted(recovered)} → {len(have)}/{expected} present in {search.name}")
    if not status["complete"]:
        print(f"  [ensure_ir] ⚠ IR INCOMPLETE: {len(have)}/{expected} present, "
              f"missing {missing} in {search.name}")
    return status
