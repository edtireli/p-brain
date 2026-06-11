"""AIF from an external ROI mask file (NIfTI or MATLAB .mat).

Useful when you have a manually drawn ROI saved to disk (e.g., from a
previous session, or from MATLAB's legacy tooling). The plug-in averages
the DCE signal within the supplied mask and returns the resulting curve.

Pass the mask path via ``opts["mask_path"]``. The mask must be in the
same voxel grid as the DCE (no resampling here).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import AIFExtractor, InputFunction


def _load_mask(mask_path: Path) -> np.ndarray:
    mask_path = Path(mask_path)
    suffix = mask_path.suffix.lower()
    if suffix in (".nii", ".gz"):
        import nibabel as nib
        return np.asarray(nib.load(str(mask_path)).dataobj).astype(bool)
    if suffix == ".mat":
        from scipy.io import loadmat
        d = loadmat(str(mask_path))
        for key, val in d.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(val)
            if arr.ndim >= 3:
                return arr.astype(bool)
        raise ValueError(f"No 3-D array found in {mask_path}")
    if suffix == ".npy":
        return np.load(mask_path).astype(bool)
    raise ValueError(f"Unsupported mask format: {suffix}")


@dataclass(frozen=True, slots=True)
class _FromFileAIF:
    key: ClassVar[str] = "from_file"
    name: ClassVar[str] = "AIF from external mask file"
    description: ClassVar[str] = (
        "Load a 3-D ROI mask from .nii/.nii.gz, .mat, or .npy and "
        "average the DCE signal within it."
    )
    accepts: ClassVar[dict[str, type]] = {"dce_data": np.ndarray, "mask_path": Path}
    produces: ClassVar[dict[str, type]] = {"input_function": InputFunction}

    def extract(
        self,
        dce_data: np.ndarray,
        t_s: np.ndarray,
        dce_affine: np.ndarray,
        *,
        t1_map: np.ndarray | None = None,
        m0_map: np.ndarray | None = None,
        t1w_volume: np.ndarray | None = None,
        t1w_affine: np.ndarray | None = None,
        source: str = "user",
        mask_path: Path | str | None = None,
        **_: Any,
    ) -> InputFunction:
        if mask_path is None:
            raise ValueError("from_file AIF requires opts['mask_path']")
        d = np.asarray(dce_data, dtype=np.float32)
        roi = _load_mask(Path(mask_path))
        if roi.shape != d.shape[:3]:
            raise ValueError(
                f"Mask shape {roi.shape} != DCE spatial shape {d.shape[:3]}"
            )
        if not roi.any():
            raise ValueError(f"Mask {mask_path} is empty")
        ca = d[roi].mean(axis=0)
        return InputFunction(
            c_a=ca, t_s=np.asarray(t_s, dtype=float),
            mask=roi, source=source,
            meta={"algorithm": "from_file", "mask_path": str(mask_path),
                  "n_voxels": int(roi.sum())},
        )


PLUGIN = _FromFileAIF()
