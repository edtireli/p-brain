"""Slice-wise aggregator — paper Fig. 7 / Supporting Info distributions.

Reports per-slice mean / median / p25 / p75 for each map along the
superior-inferior axis (default ``slice_axis=2``). Output is a CSV per
map with one row per slice index. Useful for QC and for the cohort
boxplots in the paper's results gallery.

When a brain mask is supplied via ``opts["brain_mask"]`` (3-D bool),
stats are computed only over within-mask voxels per slice.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import Aggregator, AggregationResult


def _slice_stats(arr_slice: np.ndarray) -> dict[str, float]:
    v = arr_slice[np.isfinite(arr_slice)]
    if v.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p25": float("nan"), "p75": float("nan"), "n": 0}
    return {
        "mean":   float(np.mean(v)),
        "median": float(np.median(v)),
        "p25":    float(np.percentile(v, 25)),
        "p75":    float(np.percentile(v, 75)),
        "n":      int(v.size),
    }


@dataclass(frozen=True, slots=True)
class _SliceWiseAggregator:
    key: ClassVar[str] = "slice_wise"
    name: ClassVar[str] = "Slice-wise distribution aggregator"
    description: ClassVar[str] = (
        "Per-slice mean/median/p25/p75/N along the superior-inferior "
        "axis. Drives the paper's slice-wise QC boxplots."
    )
    accepts: ClassVar[dict[str, type]] = {"maps": dict}
    produces: ClassVar[dict[str, type]] = {"csv_files": dict}

    def aggregate(
        self,
        maps: dict[str, np.ndarray],
        units: dict[str, str],
        *,
        reference_affine: np.ndarray | None = None,
        parcellation: np.ndarray | None = None,
        labels: dict[int, str] | None = None,
        region_map: dict[str, list[int]] | None = None,
        slice_axis: int = 2,
        out_dir: Path = Path("."),
        **opts: Any,
    ) -> AggregationResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        brain_mask = opts.get("brain_mask")
        files: dict[str, Path] = {}
        summary: dict[str, Any] = {}

        for name, arr in maps.items():
            arr = np.asarray(arr)
            if arr.ndim != 3:
                # Skip non-3D maps (1-D from ROI fits aren't slice-wise meaningful)
                continue

            if brain_mask is not None:
                mask3 = np.asarray(brain_mask, dtype=bool)
                if mask3.shape != arr.shape:
                    raise ValueError("brain_mask shape mismatch")
                arr = np.where(mask3, arr, np.nan)

            n_slices = arr.shape[slice_axis]
            rows: list[dict[str, Any]] = []
            for k in range(n_slices):
                sl = np.take(arr, k, axis=slice_axis)
                rows.append({"slice_idx": k, **_slice_stats(sl)})

            path = out_dir / f"{name}_slice.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["slice_idx", "mean", "median", "p25", "p75", "n"],
                )
                writer.writeheader()
                writer.writerows(rows)
            files[name] = path
            summary[name] = {"unit": units.get(name, ""), "n_slices": n_slices}

        return AggregationResult(files=files, summary=summary)


PLUGIN = _SliceWiseAggregator()
