"""Region-mean aggregator — collapses parcels into broader regions.

Takes a ``region_map: {region_name: [label_id, …]}`` (e.g.
``{"GM": [11,12,13,…], "WM": [2,41,…], "Cerebellum": [7,8,46,47]}``)
and reports per-region mean / median / SD over the union of voxels in
the listed parcels. The paper's whole-brain GM/WM/cerebellum/boundary
summaries are produced by this aggregator with the appropriate
``region_map``.

Output per map (every model output, every format): ``<map_name>.json``,
``<map_name>.csv`` (one row per region) and ``<map_name>.nii.gz`` (each
region painted with its median).
"""

from __future__ import annotations

import csv
import json
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
class _RegionAggregator:
    key: ClassVar[str] = "region"
    name: ClassVar[str] = "Region-mean aggregator"
    description: ClassVar[str] = (
        "Collapses parcels into broader regions (GM, WM, Cerebellum, …) "
        "via a {region_name: [label_ids]} mapping. Per-region "
        "mean/median/SD per map; written as JSON."
    )
    accepts: ClassVar[dict[str, type]] = {"maps": dict, "parcellation": np.ndarray,
                                          "region_map": dict}
    produces: ClassVar[dict[str, type]] = {"json_files": dict}

    def aggregate(
        self,
        maps: dict[str, np.ndarray],
        units: dict[str, str],
        *,
        reference_affine: np.ndarray | None = None,
        parcellation: np.ndarray | None,
        labels: dict[int, str] | None = None,
        region_map: dict[str, list[int]] | None,
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
            per_region: dict[str, dict[str, float]] = {}

            if parcellation is None or arr.ndim != 3 or not region_map:
                per_region["whole_brain"] = _stats(arr.ravel())
            else:
                pc = np.asarray(parcellation)
                if pc.shape != arr.shape:
                    raise ValueError(
                        f"parcellation shape {pc.shape} != map shape {arr.shape}"
                    )
                for region, label_ids in region_map.items():
                    mask = np.isin(pc, list(label_ids))
                    per_region[region] = _stats(arr[mask])

            per_region["__unit__"] = {"unit": units.get(name, "")}  # type: ignore
            path = out_dir / f"{name}.json"
            with path.open("w") as f:
                json.dump(per_region, f, indent=2)
            files[name] = path
            summary[name] = {"unit": units.get(name, ""), "n_regions": len(per_region) - 1}

            # CSV: one row per region (uniform 3-format output).
            cpath = out_dir / f"{name}.csv"
            with cpath.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["region", "mean", "median", "sd", "n"])
                w.writeheader()
                for region, st in per_region.items():
                    if region == "__unit__":
                        continue
                    w.writerow({"region": region, **{k: st.get(k) for k in ("mean", "median", "sd", "n")}})
            files[f"{name}_csv"] = cpath

            # Painted-back NIfTI: each region filled with its median, so every
            # model output also exists as a nii.gz at the region level.
            if parcellation is not None and arr.ndim == 3 and region_map:
                import nibabel as nib
                aff = np.asarray(reference_affine) if reference_affine is not None else np.eye(4)
                painted = np.full(arr.shape, np.nan, dtype=np.float32)
                for region, label_ids in region_map.items():
                    painted[np.isin(pc, list(label_ids))] = per_region[region]["median"]
                npath = out_dir / f"{name}.nii.gz"
                nib.save(nib.Nifti1Image(painted, aff), str(npath))
                files[f"{name}_nii"] = npath

        return AggregationResult(files=files, summary=summary)


PLUGIN = _RegionAggregator()
