"""Precomputed brain-mask tissue ROI — species-agnostic, no atlas.

Reads a binary brain-mask NIfTI (already on the DCE grid, e.g. from a simple
threshold of the mean DCE) and uses it as a single whole-brain ROI (label 1,
region ``"brain"``). Unlike :mod:`preloaded`, it does **not** filter by
FreeSurfer tissue-class label IDs, so it works for any species and any mask —
in particular rodent DCE, where the human CNN/SynthSeg and the VFA-dependent
voxelwise provider do not apply.

    --tissue-roi mask --opt tissue_roi.mask.mask_path=/path/to/brainmask.nii.gz

Extends, and does not modify, the existing tissue-ROI providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider


@dataclass(frozen=True, slots=True)
class _MaskROI:
    key: ClassVar[str] = "mask"
    name: ClassVar[str] = "Precomputed brain mask (single ROI)"
    description: ClassVar[str] = (
        "Read a binary brain-mask NIfTI and use it as a single whole-brain ROI "
        "(label 1). Species-agnostic: no FreeSurfer/atlas label filtering. For "
        "rodent or other non-human DCE where SynthSeg does not apply."
    )
    accepts: ClassVar[dict[str, type]] = {"mask_path": Path}
    produces: ClassVar[dict[str, type]] = {"tissue_roi": TissueROI}

    def extract(
        self,
        t1w_volume: np.ndarray,
        t1w_affine: np.ndarray,
        *,
        out_dir: Path,
        target_affine: np.ndarray | None = None,
        target_shape: tuple[int, int, int] | None = None,
        mask_path: Path | str | None = None,
        **_: Any,
    ) -> TissueROI:
        import nibabel as nib

        if mask_path is None:
            raise ValueError(
                "tissue_roi mask requires tissue_roi.mask.mask_path=<brainmask.nii.gz>"
            )
        path = Path(mask_path)
        if not path.exists():
            raise FileNotFoundError(f"mask_path not found: {path}")
        img = nib.load(str(path))
        parc = (np.asarray(img.dataobj) > 0).astype(np.int32)
        return TissueROI(
            parcellation=parc,
            affine=np.asarray(img.affine, dtype=float),
            labels={1: "brain"},
            region_map={"brain": [1]},
            meta={
                "algorithm": "precomputed_mask",
                "path": str(path),
                "voxels": int(parc.sum()),
            },
        )


PLUGIN = _MaskROI()
