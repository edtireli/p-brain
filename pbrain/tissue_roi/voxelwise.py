"""Voxelwise tissue ROI — no parcellation, just a brain mask.

Returns a single-label parcellation: 1 inside the brain, 0 outside.
Useful when downstream aggregators only need voxelwise or whole-brain
summaries (skips the expensive segmentation step). Brain mask via
Otsu threshold on the T1-weighted volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider


def _otsu(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=256)
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + 0.5 * np.diff(edges)))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b2 = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return float(edges[int(np.argmax(sigma_b2))])


@dataclass(frozen=True, slots=True)
class _VoxelwiseTissue:
    key: ClassVar[str] = "voxelwise"
    name: ClassVar[str] = "Voxelwise (brain mask only)"
    description: ClassVar[str] = (
        "Single-label parcellation: 1 = brain (Otsu threshold), "
        "0 = background. Use when no parcellation is required."
    )
    accepts: ClassVar[dict[str, type]] = {"t1w_volume": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"tissue_roi": TissueROI}

    def extract(
        self,
        t1w_volume: np.ndarray,
        t1w_affine: np.ndarray,
        *,
        out_dir: Path,
        target_affine: np.ndarray | None = None,
        target_shape: tuple[int, int, int] | None = None,
        **_: Any,
    ) -> TissueROI:
        t1 = np.asarray(t1w_volume, dtype=float)
        if t1.ndim != 3:
            raise ValueError(f"t1w_volume must be 3-D; got {t1.shape}")
        thr = _otsu(t1.ravel())
        parc = (t1 > thr).astype(np.int16)
        return TissueROI(
            parcellation=parc,
            affine=np.asarray(t1w_affine, dtype=float),
            labels={1: "brain"},
            region_map={"whole_brain": [1]},
            meta={"algorithm": "otsu_brain_mask", "threshold": thr},
        )


PLUGIN = _VoxelwiseTissue()
