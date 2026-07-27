"""Index a Bruker ParaVision study and work out what each scan *is*.

A ParaVision study is a directory of numbered scans; nothing in the tree says
which one is the dynamic series, which are the flip-angle set, or which is the
anatomical. This reads each scan's headers and classifies it by role, so
``pbrain run <study>`` needs no hand-written config — the same auto-detection
human PAR/REC and BIDS layouts get.

Classification is by acquisition *facts*, not scan names: a series with >1
repetition is the dynamic; single-repetition spoiled-gradient scans that share a
geometry but differ in flip angle are the VFA set; a high-resolution spin-echo is
the anatomical. Names are used only to break ties and to label the result, so a
site that renames its protocols still resolves correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .loaders.bruker import _nums, read_jcamp

_GRE = ("FLASH", "GEFC", "FISP", "MGE")           # spoiled/gradient echo → T1-weighted
_SE = ("RARE", "MSME", "TURBORARE")               # spin echo → anatomical
_LOCALIZER = ("TRIPILOT", "LOCALIZER", "SCOUT")


@dataclass(frozen=True, slots=True)
class ScanInfo:
    """One reconstructed ParaVision scan, with what we need to place it."""
    number: int
    path: Path
    name: str = ""
    method: str = ""
    flip_deg: float = float("nan")
    tr_ms: float = float("nan")
    te_ms: float = float("nan")
    n_reps: int = 1
    n_slices: int = 1
    matrix: tuple[int, ...] = ()
    fov_mm: tuple[float, ...] = ()
    geom: tuple = ()                 # (matrix, fov, nslices, slice0 pos, normal)

    @property
    def is_dynamic(self) -> bool:
        return self.n_reps > 1

    @property
    def is_localizer(self) -> bool:
        return any(k in self.name.upper() for k in _LOCALIZER)

    @property
    def is_gre(self) -> bool:
        return any(k in self.method.upper() for k in _GRE)

    @property
    def is_se(self) -> bool:
        return any(k in self.method.upper() for k in _SE)

    @property
    def voxels(self) -> int:
        return int(np.prod(self.matrix)) if self.matrix else 0


def _scan_dirs(study: Path) -> list[Path]:
    return sorted((d for d in study.iterdir()
                   if d.is_dir() and d.name.isdigit() and (d / "pdata").is_dir()),
                  key=lambda d: int(d.name))


def read_scan(scan: Path) -> ScanInfo | None:
    """Header-only read of one scan (no image data)."""
    reco = next((r for r in sorted((scan / "pdata").iterdir())
                 if (r / "visu_pars").exists()), None) if (scan / "pdata").is_dir() else None
    if reco is None:
        return None
    v = read_jcamp(reco / "visu_pars")
    method = read_jcamp(scan / "method") if (scan / "method").exists() else {}
    acqp = read_jcamp(scan / "acqp") if (scan / "acqp").exists() else {}
    from .loaders.bruker import _frame_groups
    groups = _frame_groups(v)
    n_sl = next((n for n, nm in groups if nm == "FG_SLICE"), 1)
    n_rep = next((n for n, nm in groups
                  if nm in {"FG_CYCLE", "FG_MOVIE", "FG_DYNAMIC", "FG_REPETITION"}), 1)
    pos = np.asarray(_nums(v.get("CorePosition")) or [0, 0, 0], dtype=float).reshape(-1, 3)
    o = np.asarray(_nums(v.get("CoreOrientation")) or [1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    normal = tuple(np.round(o[:9].reshape(3, 3).T[:, 2], 3)) if o.size >= 9 else (0., 0., 1.)
    matrix = tuple(int(x) for x in _nums(v.get("CoreSize"))[:3])
    fov = tuple(round(float(x), 3) for x in _nums(v.get("CoreExtent"))[:3])
    return ScanInfo(
        number=int(scan.name), path=scan,
        name=(acqp.get("ACQ_scan_name") or "").strip("<> "),
        method=(method.get("Method") or "").replace("<Bruker:", "").rstrip(">"),
        flip_deg=float((_nums(v.get("AcqFlipAngle")) or [float("nan")])[0]),
        tr_ms=float((_nums(method.get("PVM_RepetitionTime")) or [float("nan")])[0]),
        te_ms=float((_nums(v.get("AcqEchoTime")) or [float("nan")])[0]),
        n_reps=int(n_rep), n_slices=int(n_sl), matrix=matrix, fov_mm=fov,
        geom=(matrix, fov, int(n_sl), tuple(np.round(pos[0], 3)), normal),
    )


def index_study(study: Path | str) -> list[ScanInfo]:
    """Every readable scan in a ParaVision study, in scan order."""
    study = Path(study)
    return [s for s in (read_scan(d) for d in _scan_dirs(study)) if s is not None]


@dataclass
class StudyRoles:
    """What each scan is used for. ``vfa`` is ordered by flip angle."""
    dce: ScanInfo | None = None
    vfa: list[ScanInfo] = field(default_factory=list)
    anat: ScanInfo | None = None
    angio: ScanInfo | None = None
    others: list[ScanInfo] = field(default_factory=list)

    @property
    def flip_angles(self) -> list[float]:
        return [s.flip_deg for s in self.vfa]


def classify(scans: list[ScanInfo]) -> StudyRoles:
    """Assign roles from acquisition facts (see module docstring)."""
    usable = [s for s in scans if not s.is_localizer]
    roles = StudyRoles()

    # DCE = the dynamic series; if several, the one with the most repetitions.
    dynamics = [s for s in usable if s.is_dynamic]
    roles.dce = max(dynamics, key=lambda s: (s.n_reps, s.voxels)) if dynamics else None

    # VFA = single-rep gradient-echo scans sharing ONE geometry with ≥2 distinct
    # flip angles. Prefer the geometry the DCE was acquired on, so the T1 map is
    # voxel-aligned to the dynamic and needs no registration.
    cand = [s for s in usable
            if s.is_gre and not s.is_dynamic and np.isfinite(s.flip_deg)]
    by_geom: dict[tuple, list[ScanInfo]] = {}
    for s in cand:
        by_geom.setdefault(s.geom, []).append(s)
    groups = [g for g in by_geom.values() if len({s.flip_deg for s in g}) >= 2]
    if groups:
        if roles.dce is not None and roles.dce.geom in by_geom:
            best = by_geom[roles.dce.geom]
            if len({s.flip_deg for s in best}) >= 2:
                groups = [best]
        roles.vfa = sorted(max(groups, key=len), key=lambda s: s.flip_deg)

    # Anatomical = highest-resolution spin-echo; angiography = a flow-compensated
    # or TOF gradient echo (bright vessels — a strong AIF prior).
    se = [s for s in usable if s.is_se]
    roles.anat = max(se, key=lambda s: s.voxels) if se else None
    roles.angio = next((s for s in usable
                        if "TOF" in s.name.upper() or "FC" in s.method.upper()), None)

    claimed = {id(x) for x in ([roles.dce, roles.anat, roles.angio] + roles.vfa) if x}
    roles.others = [s for s in scans if id(s) not in claimed]
    return roles


def assemble_vfa(roles: StudyRoles, out_path: Path | str) -> Path | None:
    """Stack the flip-angle scans into one 4-D NIfTI + a BIDS ``FlipAngles`` sidecar.

    The sidecar is the whole point: :mod:`pbrain.io.loaders.nifti` maps
    ``FlipAngles`` → ``axis4_kind="fa"``, so the load stage writes the *real*
    angles to ``ir_ti_s.npy`` and the VFA T1 fit receives them directly — mouse
    VFA travels the identical path human inversion-recovery does. Returns None if
    there are fewer than two angles (the fit needs ≥2).
    """
    import json

    import nibabel as nib

    from .loaders.bruker import BrukerLoader
    if len(roles.vfa) < 2:
        return None
    loader = BrukerLoader()
    vols: list[np.ndarray] = []
    angles: list[float] = []
    affine = None
    for s in roles.vfa:
        ser = loader.load(s.path)
        if affine is None:
            affine = ser.affine
        elif ser.data.shape[:3] != vols[0].shape:
            raise ValueError(
                f"VFA scan #{s.number} is {ser.data.shape[:3]}, expected {vols[0].shape}; "
                "the flip-angle series must share one geometry."
            )
        vols.append(ser.data[..., 0])
        angles.append(float(s.flip_deg))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.stack(vols, axis=-1).astype(np.float32), affine),
             str(out_path))
    stem = out_path
    while stem.suffix in (".gz", ".nii"):
        stem = stem.with_suffix("")
    stem.with_suffix(".json").write_text(json.dumps({"FlipAngles": angles}),
                                         encoding="utf-8")
    return out_path


def describe(roles: StudyRoles) -> str:
    """One-line-per-role summary for the run log / --dry-run."""
    def s(x):
        return f"#{x.number} {x.name or x.method}" if x else "—"
    out = [f"  dce    {s(roles.dce)}"
           + (f"  ({roles.dce.n_reps} reps, {roles.dce.n_slices} sl)" if roles.dce else ""),
           f"  vfa    {', '.join(f'#{v.number}@{v.flip_deg:g}°' for v in roles.vfa) or '—'}",
           f"  anat   {s(roles.anat)}",
           f"  angio  {s(roles.angio)}"]
    return "\n".join(out)
