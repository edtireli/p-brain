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
from dataclasses import dataclass
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
            return self._load_via_dcm2niix(path)
        return self._load_via_nibabel(path)

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
