"""Parcel-mean aggregator — one CSV per map, one row per parcel label.

Each row holds the mean / median / SD / N of the map's values inside
that parcel. Requires a label volume (``parcellation``) and a
``labels`` dict mapping integer labels to human-readable names.

If no parcellation is supplied, writes a single-row CSV with whole-brain
stats (over finite voxels only).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import Aggregator, AggregationResult


def _stats(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "sd": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "n": int(v.size),
    }


@dataclass(frozen=True, slots=True)
class _ParcelAggregator:
    key: ClassVar[str] = "parcel"
    name: ClassVar[str] = "Parcel-mean aggregator"
    description: ClassVar[str] = (
        "Per-parcel mean/median/SD/N for each map; written as CSV with "
        "one row per parcel label (or one row for whole-brain when no "
        "parcellation is supplied)."
    )
    accepts: ClassVar[dict[str, type]] = {"maps": dict, "parcellation": np.ndarray, "labels": dict}
    produces: ClassVar[dict[str, type]] = {"csv_files": dict}

    def aggregate(
        self,
        maps: dict[str, np.ndarray],
        units: dict[str, str],
        *,
        reference_affine: np.ndarray | None = None,
        parcellation: np.ndarray | None,
        labels: dict[int, str] | None,
        region_map: dict[str, list[int]] | None = None,
        slice_axis: int = 2,
        out_dir: Path = Path("."),
        **opts: Any,
    ) -> AggregationResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files: dict[str, Path] = {}
        summary: dict[str, Any] = {}

        for name, arr in maps.items():
            arr = np.asarray(arr)
            rows: list[dict[str, Any]] = []

            if parcellation is None or arr.ndim != 3:
                rows.append({"label_id": 0, "label_name": "whole_brain",
                             **_stats(arr.ravel())})
            else:
                pc = np.asarray(parcellation)
                if pc.shape != arr.shape:
                    raise ValueError(
                        f"parcellation shape {pc.shape} != map shape {arr.shape}"
                    )
                names = labels or {}
                for lab in sorted(np.unique(pc).tolist()):
                    if int(lab) == 0:
                        continue   # skip background
                    rows.append({
                        "label_id": int(lab),
                        "label_name": names.get(int(lab), f"label_{int(lab)}"),
                        **_stats(arr[pc == lab]),
                    })

            path = out_dir / f"{name}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["label_id", "label_name", "mean", "median", "sd", "n"],
                )
                writer.writeheader()
                writer.writerows(rows)
            files[name] = path
            summary[name] = {"unit": units.get(name, ""), "n_rows": len(rows)}

            # JSON: same rows keyed by label (uniform 3-format output).
            import json as _json
            jpath = out_dir / f"{name}.json"
            with jpath.open("w", encoding="utf-8") as f:
                _json.dump({str(r["label_id"]): r for r in rows}
                           | {"__unit__": {"unit": units.get(name, "")}}, f, indent=2)
            files[f"{name}_json"] = jpath

            # Painted-back NIfTI: each parcel filled with its median, so every
            # model output also exists as a nii.gz at the parcel level.
            if parcellation is not None and arr.ndim == 3:
                import nibabel as nib
                aff = np.asarray(reference_affine) if reference_affine is not None else np.eye(4)
                painted = np.full(arr.shape, np.nan, dtype=np.float32)
                for row in rows:
                    painted[pc == int(row["label_id"])] = row["median"]
                npath = out_dir / f"{name}.nii.gz"
                nib.save(nib.Nifti1Image(painted, aff), str(npath))
                files[f"{name}_nii"] = npath

        return AggregationResult(files=files, summary=summary)


PLUGIN = _ParcelAggregator()
