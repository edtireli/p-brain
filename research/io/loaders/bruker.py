"""Bruker ParaVision loader — reads a reconstructed ``2dseq`` into a Series4D.

ParaVision stores each reconstruction as a flat binary ``2dseq`` beside JCAMP-DX
parameter files (``visu_pars``, ``reco``) with the acquisition parameters one level
up (``method``, ``acqp``). This reads that triple directly — no vendor tools, no
extra dependency — so preclinical (mouse/rat) studies enter the *same* pipeline as
human DICOM/NIfTI: every downstream stage sees a plain :class:`Series4D`.

Frame order comes from ``VisuFGOrderDesc``, which lists the frame groups
fastest-varying first, e.g.::

    (9, <FG_SLICE>, <>, 0, 2) (150, <FG_CYCLE>, <>, 2, 0)

= 9 slices cycling 150 times → 1350 stored frames → ``(96, 96, 9, 150)``. A group
named ``FG_CYCLE``/``FG_MOVIE`` becomes the 4th (time) axis; a study whose flip
angles live in *separate scans* (the usual VFA design) is stitched together by the
layout/assembly layer, not here — this loader returns one scan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import Series4D

#: ParaVision word types → numpy dtypes.
_WORD = {
    "_8BIT_UNSGN_INT": "u1", "_16BIT_SGN_INT": "i2", "_32BIT_SGN_INT": "i4",
    "_32BIT_FLOAT": "f4", "_64BIT_FLOAT": "f8",
}
_TIME_GROUPS = {"FG_CYCLE", "FG_MOVIE", "FG_DYNAMIC", "FG_REPETITION"}


def read_jcamp(path: Path) -> dict[str, str]:
    """Parse a JCAMP-DX parameter file (``visu_pars``/``method``/``acqp``).

    Values that declare a shape (``##$Key=( 9, 3 )``) continue on the following
    lines until the next ``##``/``$$`` record; those lines are joined so the caller
    can pull a flat numeric array out of them.
    """
    out: dict[str, str] = {}
    key: str | None = None
    buf: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        if raw.startswith(("##", "$$")):
            if key is not None:
                out[key] = " ".join(buf).strip()
            key, buf = None, []
            m = re.match(r"##\$?([A-Za-z0-9_]+)=(.*)$", raw)
            if m:
                key, rest = m.group(1), m.group(2).strip()
                if rest.startswith("("):          # shaped → values follow
                    buf = []
                else:
                    out[key], key = rest, None
        elif key is not None:
            buf.append(raw.strip())
    if key is not None:
        out[key] = " ".join(buf).strip()
    # visu_pars keys are uniformly ``Visu``-prefixed (``VisuCoreSize``); expose a
    # stripped alias so call sites read ``CoreSize`` without repeating the prefix.
    for k in list(out):
        if k.startswith("Visu") and k[4:] not in out:
            out[k[4:]] = out[k]
    return out


_RLE = re.compile(r"@(\d+)\*\(([^)]*)\)")


def _nums(txt: str | None) -> list[float]:
    """Numbers out of a JCAMP value, expanding run-length groups first.

    ParaVision compresses constant arrays as ``@1350*(2.937)`` — "1350 copies of
    2.937". Read naively that is two numbers, and tiling them across the frames
    scales alternate frames by ~460x, so the RLE must be expanded here."""
    if not txt:
        return []
    txt = _RLE.sub(lambda m: (" " + m.group(2).strip()) * int(m.group(1)), txt)
    return [float(t) for t in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", txt)]


def _frame_groups(visu: dict[str, str]) -> list[tuple[int, str]]:
    """``VisuFGOrderDesc`` → ``[(length, name), …]``, fastest-varying first."""
    return [(int(n), nm) for n, nm in
            re.findall(r"\(\s*(\d+)\s*,\s*<([^>]*)>", visu.get("FGOrderDesc", ""))]


# ── subject frame → NIfTI RAS+ ────────────────────────────────────────────────
# ParaVision reports geometry in the *magnet/subject* frame: X horizontal
# (left-right), Y vertical, Z along the bore. NIfTI requires RAS+ (x=Right,
# y=Anterior, z=Superior), which is defined from human anatomy. For a BIPED the
# two already agree, but for a QUADRUPED lying along the bore they do not: the
# animal's rostro-caudal axis runs along Z and its dorso-ventral axis is vertical
# (Y), so A-P and S-I come out transposed. dcm2niix sees the same thing and warns
# ("Anatomical Orientation Type is QUADRUPED: rotate coordinates accordingly")
# but leaves the rotation to the caller — so we apply it here, once, at the door.
#
# Head_Prone (belly down, back up) ⇒ +Y_magnet is dorsal = Superior, and the bore
# axis is rostro-caudal:  RAS_x = +X,  RAS_y = -Z,  RAS_z = +Y   (det = +1; a bare
# axis swap would be improper and would silently mirror left-right).
#
# Verified on real mouse data two independent ways: the measured inter-slice step
# is -Y_magnet (slices descend ventrally), and enclosed air voids — nasal sinuses
# and ear bullae, which are ventral — rise from 4.0 % to 14.3 % along +k.
_QUADRUPED_TO_RAS = {
    "head_prone": np.array([[1, 0, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=float),
}


def _subject_frame_to_ras(visu: dict[str, str]) -> np.ndarray:
    """World-space correction taking the Bruker subject frame to NIfTI RAS+.

    Identity for bipeds (already RAS-compatible) and for any subject
    type/position combination we have not validated — an unverified guess would
    silently mis-label anatomy, which is worse than leaving the header honest
    about being in the scanner frame."""
    stype = (visu.get("SubjectType") or "").strip().lower()
    spos = (visu.get("SubjectPosition") or "").strip().lower()
    if stype == "quadruped":
        return _QUADRUPED_TO_RAS.get(spos, np.eye(4))
    return np.eye(4)


@dataclass(frozen=True, slots=True)
class BrukerLoader:
    """Reads a ParaVision scan directory (or a ``2dseq`` path) into Series4D."""

    key: ClassVar[str] = "bruker"
    name: ClassVar[str] = "Bruker ParaVision (2dseq)"
    description: ClassVar[str] = (
        "Reads a reconstructed Bruker ParaVision 2dseq with its visu_pars/method "
        "headers into a Series4D — preclinical studies use the same pipeline as "
        "human data."
    )
    accepts: ClassVar[dict[str, type]] = {"path": Path}
    produces: ClassVar[dict[str, type]] = {"series": Series4D}
    extensions: ClassVar[tuple[str, ...]] = ("2dseq",)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _resolve(path: Path) -> tuple[Path, Path]:
        """``path`` → ``(2dseq, its pdata/N dir)``. Accepts the 2dseq itself, a
        ``pdata/N`` dir, or a scan dir (uses its lowest-numbered reconstruction)."""
        p = Path(path)
        if p.is_file() and p.name == "2dseq":
            return p, p.parent
        if p.is_dir():
            if (p / "2dseq").exists():
                return p / "2dseq", p
            pdata = p / "pdata"
            if pdata.is_dir():
                for reco in sorted(pdata.iterdir(), key=lambda d: (not d.name.isdigit(), d.name)):
                    if (reco / "2dseq").exists():
                        return reco / "2dseq", reco
        raise ValueError(f"no Bruker 2dseq under {path!s}")

    def detect(self, path: Path) -> bool:
        try:
            self._resolve(Path(path))
            return True
        except Exception:
            return False

    # -- load ----------------------------------------------------------------
    def load(self, path: Path, **opts: Any) -> Series4D:
        seq, reco_dir = self._resolve(Path(path))
        visu = read_jcamp(reco_dir / "visu_pars")
        scan_dir = reco_dir.parent.parent            # …/<scan>/pdata/<n> → <scan>
        method = read_jcamp(scan_dir / "method") if (scan_dir / "method").exists() else {}
        acqp = read_jcamp(scan_dir / "acqp") if (scan_dir / "acqp").exists() else {}

        size = [int(v) for v in _nums(visu.get("CoreSize"))]
        if len(size) < 2:
            raise ValueError(f"{seq}: unreadable VisuCoreSize")
        nx, ny = size[0], size[1]
        word = visu.get("CoreWordType", "_16BIT_SGN_INT").strip()
        dt = np.dtype(_WORD.get(word, "i2"))
        dt = dt.newbyteorder("<" if "little" in visu.get("CoreByteOrder", "little").lower() else ">")

        raw = np.frombuffer(seq.read_bytes(), dtype=dt)
        n_frames = int(_nums(visu.get("CoreFrameCount")) [0]) if visu.get("CoreFrameCount") \
            else raw.size // (nx * ny)
        in_plane = nx * ny
        raw = raw[: n_frames * in_plane].astype(np.float32)

        # per-frame slope/offset restore the real-valued signal
        slope = np.asarray(_nums(visu.get("CoreDataSlope")) or [1.0], dtype=np.float32)
        offs = np.asarray(_nums(visu.get("CoreDataOffs")) or [0.0], dtype=np.float32)
        slope = np.resize(slope, n_frames)
        offs = np.resize(offs, n_frames)
        vol = raw.reshape(n_frames, in_plane) * slope[:, None] + offs[:, None]
        vol = vol.reshape(n_frames, ny, nx).transpose(2, 1, 0)      # → (x, y, frame)

        # frame groups: split the frame axis into (slices, 4th axis)
        groups = _frame_groups(visu)
        n_slices = next((n for n, nm in groups if nm == "FG_SLICE"), None)
        n_t = next((n for n, nm in groups if nm in _TIME_GROUPS), 1)
        if n_slices is None:
            n_slices = max(1, n_frames // max(1, n_t))
        if n_slices * n_t != n_frames:               # unexpected grouping → all slices
            n_slices, n_t = n_frames, 1
        # FGOrderDesc lists the FASTEST-varying group first. With FG_SLICE first the
        # stored order is ``frame = slice + n_slices*cycle`` — slices vary fastest, so
        # they must be the trailing axis of the C-order reshape and are transposed
        # back afterwards. (Getting this backwards interleaves slices into the time
        # axis and shows up as a periodic ripple on the volume-mean curve.)
        if groups and groups[0][1] == "FG_SLICE":
            data = vol.reshape(nx, ny, n_t, n_slices).transpose(0, 1, 3, 2)
        else:
            data = vol.reshape(nx, ny, n_slices, n_t)

        extent = _nums(visu.get("CoreExtent"))
        dx = (extent[0] / nx) if len(extent) > 0 and nx else 1.0
        dy = (extent[1] / ny) if len(extent) > 1 and ny else 1.0
        pos = np.asarray(_nums(visu.get("CorePosition")), dtype=float)
        pos = pos.reshape(-1, 3) if pos.size >= 3 else np.zeros((1, 3))
        dz = float(np.linalg.norm(pos[1] - pos[0])) if pos.shape[0] > 1 else \
            float((_nums(visu.get("CoreFrameThickness")) or [1.0])[0])
        dz = dz or 1.0

        affine = self._affine(visu, pos, dx, dy, dz)

        kind = "time" if n_t > 1 else "static"
        tr_s = float((_nums(method.get("PVM_RepetitionTime")) or [0.0])[0]) / 1000.0
        dt_s = float(opts.get("dt_s") or self._frame_dt(visu, method, n_t))
        values = (np.arange(n_t, dtype=float) * dt_s) if kind == "time" \
            else np.zeros(1, dtype=float)

        meta = {
            "source": "bruker", "path": str(seq), "scan": scan_dir.name,
            "scan_name": (acqp.get("ACQ_scan_name") or "").strip("<> "),
            "method": (method.get("Method") or "").replace("<Bruker:", "").rstrip(">"),
            "flip_angle_deg": float((_nums(visu.get("AcqFlipAngle")) or [np.nan])[0]),
            "tr_s": tr_s,
            "te_ms": float((_nums(visu.get("AcqEchoTime")) or [np.nan])[0]),
            "n_reps": int(n_t), "n_slices": int(n_slices), "dt_s": dt_s,
            "frame_groups": groups, "word_type": word,
            "subject_position": (visu.get("SubjectPosition") or "").strip(),
        }
        return Series4D(data=np.ascontiguousarray(data, dtype=np.float32), affine=affine,
                        voxel_size=(float(dx), float(dy), float(dz)),
                        axis4_kind=kind, axis4_values=values, meta=meta)



    # -- geometry ------------------------------------------------------------
    @staticmethod
    def _affine(visu: dict[str, str], pos: np.ndarray, dx: float, dy: float, dz: float
                ) -> np.ndarray:
        """Voxel → world affine from ``VisuCoreOrientation`` + ``VisuCorePosition``.

        ParaVision stores, per slice, a row-major 3×3 whose **rows** are the image
        axes expressed in subject coordinates; the transpose maps image → subject.
        Slice direction is taken from the first two slice positions when available
        (that is the true spacing, gaps included) rather than assuming contiguity."""
        o = np.asarray(_nums(visu.get("CoreOrientation")), dtype=float)
        R = o[:9].reshape(3, 3).T if o.size >= 9 else np.eye(3)
        if pos.shape[0] > 1:                       # measured slice axis beats the normal
            step = pos[1] - pos[0]
            n = np.linalg.norm(step)
            if n > 1e-6:
                R[:, 2] = step / n
        aff = np.eye(4)
        aff[:3, 0] = R[:, 0] * dx
        aff[:3, 1] = R[:, 1] * dy
        aff[:3, 2] = R[:, 2] * dz
        aff[:3, 3] = pos[0]
        return _subject_frame_to_ras(visu) @ aff

    @staticmethod
    def _frame_dt(visu: dict[str, str], method: dict[str, str], n_t: int) -> float:
        """Seconds between dynamics: total scan time / repetitions when ParaVision
        reports it, else the per-frame repetition time. 0 for a static scan."""
        if n_t <= 1:
            return 0.0
        for src, key, scale in ((visu, "AcqScanTime", 1e-3), (method, "PVM_ScanTime", 1e-3)):
            v = _nums(src.get(key))
            if v and v[0] > 0:
                return float(v[0]) * scale / n_t
        return 0.0


PLUGIN = BrukerLoader()
