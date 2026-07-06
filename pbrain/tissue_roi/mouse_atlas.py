"""Atlas-based mouse-brain tissue ROI — pre-warped labelmap loader.

Consumes a mouse-brain parcellation that has ALREADY been registered onto the DCE
grid upstream (the FSL helper ``animal_data/convert_and_register.py`` warps a
BrainGlobe mouse atlas — e.g. ``dorr_mouse_mri_32um`` — through the T2 RARE onto
the DCE), plus an optional companion LUT JSON giving structure names and a coarse
region grouping. The heavy registration (atlas fetch + flirt) is done upstream, so
this plug-in has NO ANTs/brainglobe dependency at runtime and just loads the label
volume — deterministic and portable.

    --tissue-roi mouse_atlas \\
        --opt tissue_roi.mouse_atlas.labelmap_path=.../atlas_labels_dce.nii.gz \\
        --opt tissue_roi.mouse_atlas.lut_path=.../atlas_lut.json

Modes:
  1. ``labelmap_path`` present  → parcellation from the warped atlas (primary).
  2. ``brainmask_path`` present → single whole-brain ROI (graceful mouse fallback).
  else                         → ``ValueError`` (the stage then applies its own
                                  whole-brain fallback), so a run never hard-crashes.

The LUT JSON (optional) is ``{"labels": {"<id>": "<name>"}, "region_map":
{"<coarse group>": [<id>, ...]}}``. Absent → integer labels and a single
whole-brain region. Extends, does not modify, the existing tissue-ROI providers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider


@dataclass(frozen=True, slots=True)
class _MouseAtlasROI:
    key: ClassVar[str] = "mouse_atlas"
    name: ClassVar[str] = "Mouse atlas parcellation (pre-warped labelmap)"
    description: ClassVar[str] = (
        "Load a mouse-brain atlas parcellation already registered to the DCE grid "
        "(BrainGlobe atlas -> T2 -> DCE via FSL, upstream), with an optional LUT for "
        "structure names and a coarse region grouping. Falls back to a whole-brain "
        "mask. No ANTs/brainglobe dependency at runtime."
    )
    accepts: ClassVar[dict[str, type]] = {
        "labelmap_path": Path,
        "lut_path": Path,
        "brainmask_path": Path,
    }
    produces: ClassVar[dict[str, type]] = {"tissue_roi": TissueROI}

    def extract(
        self,
        t1w_volume: np.ndarray,
        t1w_affine: np.ndarray,
        *,
        out_dir: Path,
        target_affine: np.ndarray | None = None,
        target_shape: tuple[int, int, int] | None = None,
        labelmap_path: Path | str | None = None,
        lut_path: Path | str | None = None,
        brainmask_path: Path | str | None = None,
        **_: Any,
    ) -> TissueROI:
        import nibabel as nib

        if labelmap_path is not None and Path(labelmap_path).exists():
            img = nib.load(str(labelmap_path))
            parc = np.asarray(img.dataobj).astype(np.int32)
            return self._package(parc, np.asarray(img.affine, dtype=float),
                                 lut_path, str(labelmap_path))

        if brainmask_path is not None and Path(brainmask_path).exists():
            img = nib.load(str(brainmask_path))
            parc = (np.asarray(img.dataobj) > 0).astype(np.int32)
            return TissueROI(
                parcellation=parc,
                affine=np.asarray(img.affine, dtype=float),
                labels={1: "brain"},
                region_map={"brain": [1]},
                meta={"algorithm": "mouse_atlas_brainmask_fallback",
                      "path": str(brainmask_path), "voxels": int(parc.sum())},
            )

        raise ValueError(
            "mouse_atlas requires tissue_roi.mouse_atlas.labelmap_path=<warped atlas "
            "labels on the DCE grid> (or brainmask_path=<brain mask> for a whole-brain ROI)."
        )

    @staticmethod
    def _package(parc: np.ndarray, aff: np.ndarray,
                 lut_path: Path | str | None, src: str) -> TissueROI:
        present = [int(v) for v in np.unique(parc) if v > 0]
        labels: dict[int, str] = {}
        region_map: dict[str, list[int]] = {}
        if lut_path is not None and Path(lut_path).exists():
            lut = json.load(open(lut_path))
            labels = {int(k): str(v) for k, v in lut.get("labels", {}).items()
                      if int(k) in present}
            region_map = {g: [i for i in ids if i in present]
                          for g, ids in lut.get("region_map", {}).items()}
            region_map = {g: ids for g, ids in region_map.items() if ids}
        if not labels:
            labels = {l: f"label_{l}" for l in present}
        if not region_map:
            region_map = {"brain": present} if present else {}
        return TissueROI(
            parcellation=parc.astype(np.int32),
            affine=aff.astype(float),
            labels=labels,
            region_map=region_map,
            meta={"algorithm": "mouse_atlas_labelmap", "path": src,
                  "n_labels": len(present), "n_regions": len(region_map)},
        )


PLUGIN = _MouseAtlasROI()
