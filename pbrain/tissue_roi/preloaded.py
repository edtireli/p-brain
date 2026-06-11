"""Preloaded tissue ROI provider — read precomputed parcellation/masks.

Useful when:

* A prior run (or external tool: FastSurfer, SynthSeg, manual editing)
  has already produced a parcellation NIfTI in DCE space.
* You want byte-equivalent voxel sets to compare against legacy
  outputs.

Two modes (auto-detected):

1. **Parcellation NIfTI** — pass ``parcellation_path`` pointing at a
   label volume (e.g. ``aparc.DKTatlas+aseg.deep_in_DCE.nii.gz``). Use
   ``include_labels`` to restrict to tissue classes (default: the
   legacy whole-brain GM+WM+brainstem+cerebellum union).
2. **Tissue-mask collection** — pass ``tissue_dir`` pointing at a
   directory containing ``cortical_gm.nii.gz``, ``subcortical_gm.nii.gz``,
   ``wm.nii.gz``, etc. The union of present masks becomes the brain mask
   (label 1).

Configuration::

    --tissue-roi preloaded \\
    --opt tissue_roi.preloaded.parcellation_path=/path/to/aparc_in_DCE.nii.gz

or::

    --opt tissue_roi.preloaded.tissue_dir=/path/to/segmentation/mri
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider


# Legacy p-Brain default tissue-class set (excludes CSF, vessels, choroid plexus).
# Matches the union of cortical_gm + subcortical_gm + wm + brainstem +
# cerebellum (GM+WM) + wm_cc in modules/AI_tissue_functions.py.
_LEGACY_TISSUE_LABELS_FREESURFER = (
    # Cortical GM (DKT cortical parcels, both hemispheres)
    *range(1000, 1036), *range(2000, 2036),
    # Subcortical GM
    10, 11, 12, 13, 17, 18, 26,        # left
    49, 50, 51, 52, 53, 54, 58,        # right
    # White matter (cerebral)
    2, 41, 77, 251, 252, 253, 254, 255,
    # Brainstem
    16,
    # Cerebellum
    7, 8, 46, 47,
)

_LEGACY_REGION_MAP_FS = {
    "cortical_GM": [*range(1000, 1036), *range(2000, 2036)],
    "subcortical_GM": [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58],
    "WM": [2, 41, 77, 251, 252, 253, 254, 255],
    "Cerebellum_GM": [8, 47],
    "Cerebellum_WM": [7, 46],
    "Brainstem": [16],
}


_TISSUE_FILENAMES = {
    "cortical_gm": ("cortical_gm.nii.gz", 1),
    "subcortical_gm": ("subcortical_gm.nii.gz", 2),
    "wm": ("wm.nii.gz", 3),
    "gm_brainstem": ("gm_brainstem.nii.gz", 4),
    "gm_cerebellum": ("gm_cerebellum.nii.gz", 5),
    "wm_cerebellum": ("wm_cerebellum.nii.gz", 6),
    "wm_cc": ("wm_cc.nii.gz", 7),
    "wmh": ("wmh.nii.gz", 8),
}


@dataclass(frozen=True, slots=True)
class _PreloadedTissue:
    key: ClassVar[str] = "preloaded"
    name: ClassVar[str] = "Preloaded parcellation / tissue masks"
    description: ClassVar[str] = (
        "Read an existing parcellation (FreeSurfer-style aparc+aseg) or "
        "a directory of tissue-class masks (cortical_gm/wm/etc.) from "
        "disk. The default include_labels reproduces the legacy "
        "p-Brain whole-brain tissue set (GM+WM+brainstem+cerebellum)."
    )
    accepts: ClassVar[dict[str, type]] = {
        "parcellation_path": Path, "tissue_dir": Path,
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
        parcellation_path: Path | str | None = None,
        tissue_dir: Path | str | None = None,
        include_labels: tuple[int, ...] | None = None,
        **_: Any,
    ) -> TissueROI:
        import nibabel as nib

        if parcellation_path is not None:
            path = Path(parcellation_path)
            if not path.exists():
                raise FileNotFoundError(f"parcellation_path not found: {path}")
            img = nib.load(str(path))
            parc_full = np.asarray(img.dataobj, dtype=np.int32)
            affine = np.asarray(img.affine, dtype=float)

            labels_to_keep = (set(include_labels) if include_labels is not None
                              else set(_LEGACY_TISSUE_LABELS_FREESURFER))
            keep_mask = np.isin(parc_full, list(labels_to_keep))
            parc = np.where(keep_mask, parc_full, 0).astype(np.int32)
            present = sorted(int(x) for x in np.unique(parc) if int(x) > 0)
            labels = {lab: f"label_{lab}" for lab in present}

            return TissueROI(
                parcellation=parc,
                affine=affine,
                labels=labels,
                region_map={k: v for k, v in _LEGACY_REGION_MAP_FS.items()},
                meta={
                    "algorithm": "preloaded_parcellation",
                    "path": str(path),
                    "labels_kept": len(present),
                    "voxels_kept": int(keep_mask.sum()),
                },
            )

        if tissue_dir is not None:
            tdir = Path(tissue_dir)
            if not tdir.is_dir():
                raise FileNotFoundError(f"tissue_dir not found: {tdir}")
            # Build a single label volume from the per-class NIfTIs we find.
            ref_img = None
            shape: tuple[int, int, int] | None = None
            affine: np.ndarray | None = None
            tissue_layers: dict[int, np.ndarray] = {}
            for name, (fname, label_id) in _TISSUE_FILENAMES.items():
                p = tdir / fname
                if not p.exists():
                    continue
                img = nib.load(str(p))
                if ref_img is None:
                    ref_img = img
                    shape = tuple(img.shape[:3])  # type: ignore
                    affine = np.asarray(img.affine, dtype=float)
                m = (np.asarray(img.dataobj) > 0).astype(np.uint8)
                tissue_layers[label_id] = m
            if not tissue_layers:
                raise FileNotFoundError(
                    f"No tissue-class NIfTIs found in {tdir} "
                    f"(expected one of: {list(_TISSUE_FILENAMES)})"
                )
            assert shape is not None and affine is not None
            parc = np.zeros(shape, dtype=np.int32)
            for label_id in sorted(tissue_layers):
                # First-write wins per voxel (consistent with single-class assignment).
                m = tissue_layers[label_id].astype(bool)
                parc[m & (parc == 0)] = label_id
            inv = {v[1]: k for k, v in _TISSUE_FILENAMES.items()}
            labels = {lid: inv[lid] for lid in sorted(tissue_layers)}
            region_map = {
                "cortical_GM": [1],
                "subcortical_GM": [2],
                "WM": [3, 7],
                "Brainstem": [4],
                "Cerebellum_GM": [5],
                "Cerebellum_WM": [6],
                "WMH": [8],
            }
            return TissueROI(
                parcellation=parc,
                affine=affine,
                labels=labels,
                region_map=region_map,
                meta={
                    "algorithm": "preloaded_tissue_masks",
                    "dir": str(tdir),
                    "labels_found": sorted(tissue_layers.keys()),
                    "voxels_kept": int((parc > 0).sum()),
                },
            )

        raise ValueError(
            "preloaded tissue_roi requires either tissue_roi.preloaded.parcellation_path "
            "or tissue_roi.preloaded.tissue_dir."
        )


PLUGIN = _PreloadedTissue()
