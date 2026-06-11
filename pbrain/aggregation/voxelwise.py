"""Voxelwise aggregator — writes one NIfTI per output map.

This is the simplest aggregator: it passes the model's voxelwise maps
through unchanged, writes them as NIfTI volumes via nibabel, and emits
per-map histogram summaries (count, percentiles 5/50/95) for QC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import Aggregator, AggregationResult


@dataclass(frozen=True, slots=True)
class _VoxelwiseAggregator:
    key: ClassVar[str] = "voxelwise"
    name: ClassVar[str] = "Voxelwise NIfTI aggregator"
    description: ClassVar[str] = (
        "Writes one NIfTI volume per output map. Computes histogram "
        "summaries (count, p5/p50/p95) for QC."
    )
    accepts: ClassVar[dict[str, type]] = {"maps": dict, "reference_affine": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"nifti_files": dict}

    def aggregate(
        self,
        maps: dict[str, np.ndarray],
        units: dict[str, str],
        *,
        reference_affine: np.ndarray | None,
        parcellation: np.ndarray | None = None,
        labels: dict[int, str] | None = None,
        region_map: dict[str, list[int]] | None = None,
        slice_axis: int = 2,
        out_dir: Path = Path("."),
        **opts: Any,
    ) -> AggregationResult:
        import nibabel as nib

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        affine = np.asarray(reference_affine) if reference_affine is not None else np.eye(4)
        files: dict[str, Path] = {}
        summary: dict[str, Any] = {}

        for name, arr in maps.items():
            arr = np.asarray(arr)
            path = out_dir / f"{name}.nii.gz"
            if arr.ndim == 3:
                img = nib.Nifti1Image(arr.astype(np.float32), affine)
                nib.save(img, str(path))
            else:
                # Scalar or 1-D — save as a tiny 3-D for consistency
                vol = np.atleast_3d(np.asarray(arr, dtype=np.float32))
                img = nib.Nifti1Image(vol, affine)
                nib.save(img, str(path))
            files[name] = path

            finite = arr[np.isfinite(arr)]
            summary[name] = {
                "unit": units.get(name, ""),
                "n_finite": int(finite.size),
                "p5":  float(np.percentile(finite, 5))  if finite.size else float("nan"),
                "p50": float(np.percentile(finite, 50)) if finite.size else float("nan"),
                "p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
            }

        summary_path = out_dir / "histogram_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        files["histogram_summary"] = summary_path

        return AggregationResult(files=files, summary=summary)


PLUGIN = _VoxelwiseAggregator()
