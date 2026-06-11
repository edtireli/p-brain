"""NIfTI loader.

Reads ``.nii``/``.nii.gz`` via nibabel, applying any proxy scaling
(slope/intercept) the file declares. Time-axis values come from a
sibling ``*.json`` (BIDS-style) if present, otherwise ``header.zooms[3]``
times a frame index, otherwise zeros.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import ImageLoader, Series4D


def _detect_kind_from_meta(meta: dict[str, Any], n: int) -> tuple[str, np.ndarray]:
    """Heuristic: pick axis4 kind from meta + frame count."""
    if n <= 1:
        return "static", np.zeros(1, dtype=float)
    if "InversionTimes" in meta:
        return "ti", np.asarray(meta["InversionTimes"], dtype=float)
    if "FlipAngles" in meta:
        return "fa", np.asarray(meta["FlipAngles"], dtype=float)
    if "RepetitionTimeExcitation" in meta or "RepetitionTime" in meta:
        return "time", np.asarray(meta.get("TimePoints", np.arange(n)), dtype=float)
    return "time", np.arange(n, dtype=float)


@dataclass(frozen=True, slots=True)
class _NiftiLoader:
    key: ClassVar[str] = "nifti"
    name: ClassVar[str] = "NIfTI loader"
    description: ClassVar[str] = (
        "Reads NIfTI-1 (.nii / .nii.gz) via nibabel, honouring slope/inter "
        "from the file proxy. BIDS-style sibling .json populates meta."
    )
    accepts: ClassVar[dict[str, type]] = {"path": Path}
    produces: ClassVar[dict[str, type]] = {"series": Series4D}
    extensions: ClassVar[tuple[str, ...]] = (".nii", ".nii.gz")

    def detect(self, path: Path) -> bool:
        s = str(path).lower()
        return s.endswith(".nii") or s.endswith(".nii.gz")

    def load(self, path: Path, **opts: Any) -> Series4D:
        import nibabel as nib

        path = Path(path)
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = np.asarray(img.affine, dtype=float)
        zooms = img.header.get_zooms()
        voxel_size = tuple(float(z) for z in zooms[:3])  # type: ignore[assignment]

        # Embed 3D as (X, Y, Z, 1)
        if data.ndim == 3:
            data = data[..., None]

        meta = self._load_sidecar(path)
        kind, values = _detect_kind_from_meta(meta, data.shape[-1])
        if kind == "time" and len(values) == data.shape[-1] and values[0] == 0 and len(zooms) >= 4 and zooms[3] > 0:
            # Use header dt if no explicit time points
            values = np.arange(data.shape[-1], dtype=float) * float(zooms[3])

        return Series4D(
            data=data,
            affine=affine,
            voxel_size=voxel_size,  # type: ignore[arg-type]
            axis4_kind=kind,  # type: ignore[arg-type]
            axis4_values=values,
            meta=meta,
        )

    @staticmethod
    def _load_sidecar(nifti_path: Path) -> dict[str, Any]:
        for stem in (nifti_path.with_suffix(""), nifti_path.with_suffix("").with_suffix("")):
            sidecar = Path(str(stem) + ".json")
            if sidecar.exists():
                try:
                    with sidecar.open() as f:
                        return json.load(f)
                except Exception:
                    return {}
        return {}


PLUGIN = _NiftiLoader()
