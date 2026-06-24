"""Philips PAR/REC loader.

Two strategies in order:

1. **dcm2niix** (preferred) — invokes ``dcm2niix -o <tmp> <par>``, then
   reads the resulting NIfTI via the NIfTI loader. Matches the paper's
   ingestion path and produces vendor-standard scaling.
2. **nibabel.parrec fallback** — pure-Python PAR/REC reader. Used when
   ``dcm2niix`` is not on ``PATH``.

Time-axis values come from the PAR sidecar's ``dyn_scan_begin_time``
field when available; otherwise they're frame-index * 0 (the orchestrator
can override via Config.dt_s).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import ImageLoader, Series4D
from .nifti import PLUGIN as _NIFTI


@dataclass(frozen=True, slots=True)
class _ParRecLoader:
    key: ClassVar[str] = "parrec"
    name: ClassVar[str] = "Philips PAR/REC loader"
    description: ClassVar[str] = (
        "Reads Philips .PAR/.REC pairs. Prefers dcm2niix subprocess; "
        "falls back to nibabel.parrec when dcm2niix is unavailable."
    )
    accepts: ClassVar[dict[str, type]] = {"path": Path}
    produces: ClassVar[dict[str, type]] = {"series": Series4D}
    extensions: ClassVar[tuple[str, ...]] = (".par", ".PAR")

    def detect(self, path: Path) -> bool:
        s = str(path)
        return s.lower().endswith(".par")

    def load(self, path: Path, **opts: Any) -> Series4D:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PAR file not found: {path}")
        rec = path.with_suffix(".REC")
        if not rec.exists():
            rec = path.with_suffix(".rec")
        if not rec.exists():
            raise FileNotFoundError(f"REC sidecar not found for {path}")

        if shutil.which("dcm2niix"):
            series = self._load_via_dcm2niix(path)
        else:
            series = self._load_via_nibabel(path)

        # The NIfTI zoom (and nibabel's pixdim) give the *mean* dynamic
        # spacing, which the prep/dummy gap before the first dynamic inflates
        # (2.439 s vs the true 2.430 s median on 20250217x4 — a 0.35 % Patlak-Ki
        # offset, since Ki ∝ 1/∫Ca·dt). The paper's pipeline uses the robust
        # *median* per-dynamic spacing read straight from the PAR's
        # ``dyn_scan_begin_time``; reproduce it here so the DCE time axis is
        # bit-exact with ``Analysis/Fitting/time_points_s.npy``.
        if series.axis4_kind == "time" and series.n_frames > 1:
            dt = self._dce_dt_s_from_par(path)
            if dt is not None and dt > 0:
                n = series.n_frames
                series = replace(series, axis4_values=np.arange(n, dtype=float) * dt)
        return series

    @staticmethod
    def _dce_dt_s_from_par(par: Path) -> float | None:
        """Median per-dynamic frame spacing (s) from the PAR ``dyn_scan_begin_time``.

        The pixdim/zoom spacing is the *mean* over dynamics, which the prep gap
        before the first dynamic inflates. The paper uses the *median*
        (``np.median(np.diff(begin_times))``), robust to that gap. Returns
        ``None`` when the field is absent so the caller keeps the zoom value.
        """
        try:
            import nibabel as nib

            idf = nib.load(str(par)).header.image_defs
            names = idf.dtype.names or ()
            if "dynamic scan number" not in names or "dyn_scan_begin_time" not in names:
                return None
            dyn = np.asarray(idf["dynamic scan number"])
            tb = np.asarray(idf["dyn_scan_begin_time"], dtype=float)
            ud = np.unique(dyn)
            if ud.size < 3:
                return None
            times = np.array([tb[dyn == d][0] for d in ud], dtype=float)
            dts = np.diff(times)
            dts = dts[np.isfinite(dts) & (dts > 0)]
            if dts.size == 0:
                return None
            return float(np.median(dts))
        except Exception:
            return None

    @staticmethod
    def _load_via_dcm2niix(par: Path) -> Series4D:
        with tempfile.TemporaryDirectory(prefix="pbrain_parrec_") as tmp:
            tmp_dir = Path(tmp)
            cmd = ["dcm2niix", "-z", "y", "-o", str(tmp_dir), "-f", "out", str(par)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(
                    f"dcm2niix failed for {par}: {result.stderr.strip() or result.stdout.strip()}"
                )
            niftis = sorted(tmp_dir.glob("out*.nii.gz")) or sorted(tmp_dir.glob("out*.nii"))
            if not niftis:
                raise RuntimeError(f"dcm2niix produced no NIfTI output in {tmp_dir}")
            return _NIFTI.load(niftis[0])

    @staticmethod
    def _load_via_nibabel(par: Path) -> Series4D:
        import nibabel as nib

        img = nib.load(str(par))
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = np.asarray(img.affine, dtype=float)
        zooms = img.header.get_zooms()
        voxel_size = tuple(float(z) for z in zooms[:3])

        if data.ndim == 3:
            data = data[..., None]

        meta: dict[str, Any] = {}
        general = getattr(img.header, "general_info", None)
        if isinstance(general, dict):
            meta.update(
                {
                    "TR": float(general.get("repetition_time", 0.0) or 0.0) / 1000.0,
                    "FlipAngle": float(general.get("flip_angle", 0.0) or 0.0),
                    "ScannerSerial": general.get("scn_serial_no", None),
                    "Protocol": general.get("protocol_name", None),
                }
            )

        n = data.shape[-1]
        if n > 1:
            kind = "time"
            values = np.arange(n, dtype=float) * (float(zooms[3]) if len(zooms) > 3 else 0.0)
        else:
            kind = "static"
            values = np.zeros(1, dtype=float)

        return Series4D(
            data=data,
            affine=affine,
            voxel_size=voxel_size,  # type: ignore[arg-type]
            axis4_kind=kind,  # type: ignore[arg-type]
            axis4_values=values,
            meta=meta,
        )


PLUGIN = _ParRecLoader()
