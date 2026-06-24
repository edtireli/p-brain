"""Auto-discover pipeline inputs from a raw Philips scanner subject directory.

For ``pbrain run-cohort --cohort``: each subject is a raw PAR/REC export with
~20-30 scans whose *numbers vary between subjects*, so inputs can't be templated
by filename. This resolves them by Philips **protocol name**:

* **DCE** — the dynamic perfusion scan (``hperf*``, technique T1TFE, the most
  dynamics). ``perf`` also matches, excluding ASL/QFLOW/POLD look-alikes.
* **IR**  — the saturation-recovery T1-mapping series (``TI_00120``…``TI_10000``
  scans), assembled into one 4-D NIfTI cached under the subject's pbrain dir.
* **T1**  — a 3-D T1 anatomical (``…3D_TFE…`` / ``3DI…`` / ``T1W_3D``, non-Gd)
  for SynthSeg.

Returns ``None`` for any input that can't be found; the caller decides whether a
missing input is fatal (no DCE ⇒ skip subject) or tolerable (no anatomical ⇒
SynthSeg falls back to the DCE).
"""

from __future__ import annotations

import re
from pathlib import Path

_PROTO = re.compile(r"Protocol name\s*:\s*(.+)", re.IGNORECASE)
_ASL_LIKE = ("pcasl", "qflow", "pold", "casl", "asl")


def _pars(subject_dir: Path) -> list[Path]:
    return sorted(subject_dir.glob("*.PAR")) + sorted(subject_dir.glob("*.par"))


def protocol_name(par: Path) -> str:
    """Read the Philips ``Protocol name`` header field (case/format tolerant)."""
    try:
        with open(par, "r", errors="ignore") as f:
            for line in f:
                m = _PROTO.search(line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return ""


def find_dce(subject_dir: Path) -> Path | None:
    """The dynamic perfusion scan (``hperf*`` / ``perf``, not an ASL/flow scan)."""
    for par in _pars(subject_dir):
        p = protocol_name(par).lower()
        if not p:
            continue
        if "hperf" in p or ("perf" in p and not any(x in p for x in _ASL_LIKE)):
            return par
    return None


def _is_reformat(protocol_lc: str) -> bool:
    """True for a re-sliced *reconstruction* (e.g. Philips ``ax VWIP ...``),
    not the original isotropic 3-D acquisition. These are thick single-plane
    views unsuitable for SynthSeg, and their protocol string often *contains*
    the real acquisition's name, so substring matching alone can't exclude them.
    """
    p = protocol_lc.strip()
    return (p.startswith("ax ") or p.startswith("ax_") or p.startswith("sag ")
            or p.startswith("cor ") or "vwip" in p or "reformat" in p
            or "recon -" in p or p.startswith("reg -"))


def _n_slices(par: Path) -> int:
    try:
        import nibabel as nib
        shp = nib.load(str(par)).shape
        return int(shp[2]) if len(shp) >= 3 else 0
    except Exception:
        return 0


def find_t1_anatomical(subject_dir: Path) -> Path | None:
    """The original 3-D T1-weighted anatomical (non-Gd) for SynthSeg.

    Prefers the true isotropic acquisition (e.g. ``WIP cs_T1W_3D_TFE``) over any
    axial/sagittal *reformat* (``ax VWIP cs_T1W_3D_TFE``). Among matches the
    volume with the **most slices** is the real acquisition (the reformat is a
    thick single-plane view, e.g. 64 vs 257 slices) — slice count is the robust
    tiebreaker because the reformat's protocol name contains the acquisition's.
    """
    def _t1_3d(p: str) -> bool:
        return ("3di" in p or "3d_tfe" in p or "t1w_3d" in p or "3dt1" in p) and "gd" not in p

    def _t1_loose(p: str) -> bool:
        return "t1" in p and ("3d" in p or "tfe" in p) and "gd" not in p

    for match in (_t1_3d, _t1_loose):
        # original acquisitions only (exclude reformats)
        cands = [par for par in _pars(subject_dir)
                 if match(protocol_name(par).lower())
                 and not _is_reformat(protocol_name(par).lower())]
        if cands:
            return max(cands, key=_n_slices)
        # nothing but reformats? fall back to the most-slices reformat
        cands = [par for par in _pars(subject_dir)
                 if match(protocol_name(par).lower())]
        if cands:
            return max(cands, key=_n_slices)
    return None


def discover_subject_inputs(subject_dir: Path | str, *,
                            ir_out: Path | str | None = None,
                            assemble: bool = True) -> dict[str, Path | None]:
    """Resolve ``{dce, ir, t1}`` for one raw scanner subject directory.

    The IR series is assembled into ``ir_out`` (default
    ``<subject>/pbrain/ir_assembled.nii.gz``) and cached — re-running a cohort
    skips re-assembly. Set ``assemble=False`` to only locate inputs.
    """
    subject_dir = Path(subject_dir)
    dce = find_dce(subject_dir)
    t1 = find_t1_anatomical(subject_dir)

    ir: Path | None = None
    if assemble:
        from pbrain.io.ir_assembly import assemble_ir, find_ir_files
        if find_ir_files(subject_dir):
            ir_path = Path(ir_out) if ir_out else (subject_dir / "pbrain" / "ir_assembled.nii.gz")
            ir_path.parent.mkdir(parents=True, exist_ok=True)
            if not ir_path.exists():
                assemble_ir(subject_dir, ir_path)
            ir = ir_path

    return {"dce": dce, "ir": ir, "t1": t1}
