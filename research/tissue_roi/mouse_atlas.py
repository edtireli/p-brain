"""Atlas-based mouse-brain tissue ROI.

Turns a mouse-brain atlas into a parcellation on the analysis grid, plus an optional
LUT JSON giving structure names and a coarse region grouping. No ANTs/brainglobe/FSL
dependency at runtime — deterministic and portable.

    --tissue-roi mouse_atlas \\
        --opt tissue_roi.mouse_atlas.atlas_dir=.../atlas \\
        --opt tissue_roi.mouse_atlas.lut_path=.../atlas_lut.json

Modes, in precedence order:
  1. ``labelmap_path``          → a labelmap already on the target grid.
  2. ``brainmask_path``         → single whole-brain ROI (graceful fallback).
  3. ``atlas_dir`` (ref + lbl)  → resample the atlas onto the target grid.
  else                          → ``ValueError`` (the stage then applies its own
                                   whole-brain fallback), so a run never hard-crashes.

**Mode 3 does NOT register by default, and that is deliberate.** An atlas prepared
for a subject already carries an affine placing it in that subject's world space, so
resampling through the affines alone puts every label in the right physical spot —
including when the voxel-axis order differs (RAS vs RIP), which is indexing, not
geometry. Fitting a transform on top of an already-correct alignment *destroys* it:
measured on real data, identity resampling gave 27/28 labelled slices where a 12-dof
fit gave 10/28. Pass ``register=true`` only for a genuinely unaligned stock template.

The LUT JSON (optional) is ``{"labels": {"<id>": "<name>"}, "region_map":
{"<coarse group>": [<id>, ...]}}``. Absent → integer labels and a single
whole-brain region. Extends, does not modify, the existing tissue-ROI providers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider

_log = logging.getLogger(__name__)


def _reorient_like(vol: np.ndarray, aff: np.ndarray,
                   target_aff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Permute/flip ``vol`` so its voxel axes carry the same anatomical directions as
    ``target_aff``. Pure axis reordering — no interpolation, so labels are untouched.
    The affine is updated to match, leaving the volume in the same world position."""
    import nibabel as nib
    from nibabel.orientations import (apply_orientation, inv_ornt_aff,
                                      io_orientation, ornt_transform)
    xf = ornt_transform(io_orientation(np.asarray(aff, dtype=float)),
                        io_orientation(np.asarray(target_aff, dtype=float)))
    out = apply_orientation(vol, xf)
    return out, np.asarray(aff, dtype=float) @ inv_ornt_aff(xf, vol.shape)


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
        "atlas_dir": Path,
        "atlas_ref_path": Path,
        "atlas_lbl_path": Path,
        "labelmap_path": Path,
        "lut_path": Path,
        "brainmask_path": Path,
        "atlas_dir": Path,
        "register": bool,
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
        atlas_dir: Path | str | None = None,
        atlas_ref_path: Path | str | None = None,
        atlas_lbl_path: Path | str | None = None,
        register: bool = False,
        nbins: int = 32,
        level_iters: list[int] | None = None,
        pipeline: str | tuple[str, ...] | None = None,
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

        # 3. Register an atlas here, in-process (no FSL/ANTs): affine-align the atlas
        #    reference to this subject's structural, then carry the labels through the
        #    same transform straight onto the target grid.
        ref_p, lbl_p = self._atlas_files(atlas_dir, atlas_ref_path, atlas_lbl_path)
        if ref_p is not None and lbl_p is not None:
            if t1w_volume is None or not np.size(t1w_volume):
                raise ValueError(
                    "mouse_atlas registration needs a structural volume to register the "
                    "atlas to; none was supplied (is the anatomical missing?)."
                )
            stages = (tuple(s.strip() for s in pipeline.split(",") if s.strip())
                      if isinstance(pipeline, str) else
                      tuple(pipeline) if pipeline else self.DEFAULT_PIPELINE)
            parc, aff = self._resample(
                np.asarray(t1w_volume, dtype=float), np.asarray(t1w_affine, dtype=float),
                ref_p, lbl_p, target_affine, target_shape,
                out_dir=out_dir, register=bool(register),
                nbins=int(nbins), level_iters=level_iters, pipeline=stages,
            )
            if lut_path is None:
                cand = Path(ref_p).parent / "atlas_lut.json"
                lut_path = cand if cand.exists() else None
            return self._package(parc, aff, lut_path, f"registered:{Path(lbl_p).name}")

        raise ValueError(
            "mouse_atlas needs either tissue_roi.mouse_atlas.atlas_dir=<dir with "
            "atlas_ref.nii.gz + atlas_lbl.nii.gz> to register here, or a pre-warped "
            "labelmap_path=<atlas labels on the DCE grid> (or brainmask_path=<brain "
            "mask> for a whole-brain ROI)."
        )

    @staticmethod
    def _atlas_files(atlas_dir, ref_path, lbl_path) -> tuple[Path | None, Path | None]:
        """Locate the atlas reference + label volumes from an explicit pair or a dir."""
        if ref_path and lbl_path and Path(ref_path).exists() and Path(lbl_path).exists():
            return Path(ref_path), Path(lbl_path)
        if not atlas_dir:
            return None, None
        d = Path(atlas_dir)
        if not d.is_dir():
            return None, None
        ref = next((p for n in ("atlas_ref.nii.gz", "atlas_ref.nii", "*ref*.nii*")
                    for p in sorted(d.glob(n))), None)
        lbl = next((p for n in ("atlas_lbl.nii.gz", "atlas_lbl.nii", "*lbl*.nii*",
                                "*label*.nii*") for p in sorted(d.glob(n))), None)
        return (ref, lbl) if (ref and lbl) else (None, None)

    #: Registration stages, coarse → fine.
    #:
    #: Ends at the full 12-dof affine. A 7-dof similarity (``rigid_isoscaling``) was
    #: tried on the theory that a subject brain differs from a template by roughly one
    #: uniform scale, and that granting three independent axis scales lets the
    #: optimiser squash the template along the worst-sampled axis — the 0.5 mm slice
    #: direction, ~17 samples across the brain against 32 µm in the template.
    #:
    #: **It measured worse, and the reason is instructive.** Constraining to one shared
    #: scale did not lift the bad through-plane estimate; it dragged the good in-plane
    #: ones down to compromise with it. Subject 1, extents in mm (true mouse brain
    #: ~10-11 / 12-13 / 7-8)::
    #:
    #:     12-dof affine      L-R 11.2   A-P 13.2   I-S 5.6
    #:     7-dof similarity   L-R  7.8   A-P 11.6   I-S 5.1
    #:
    #: So the through-plane squash is not the optimiser overfitting a free parameter —
    #: it is that thick-slice data carries too little through-plane evidence for ANY
    #: intensity-driven fit to recover the scale. Constraining the model just spreads
    #: the error. Left configurable because the failure is worth being able to re-test.
    DEFAULT_PIPELINE: ClassVar[tuple[str, ...]] = (
        "center_of_mass", "translation", "rigid", "affine",
    )

    @staticmethod
    def _resample(t1w: np.ndarray, t1w_aff: np.ndarray, ref_p: Path, lbl_p: Path,
                  target_affine, target_shape, *, out_dir: Path, register: bool = False,
                  nbins: int = 32, level_iters=None,
                  pipeline: tuple[str, ...] = (
                      "center_of_mass", "translation", "rigid", "affine"),
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Put the atlas labels on the target grid, **nearest-neighbour**.

        By default this is a pure affine resample — NO registration. An atlas prepared
        for a subject already carries an affine that places it in that subject's world
        space, so the affines alone put every label in the right physical location;
        differing voxel-axis orders (RAS vs RIP) are just indexing and are handled by
        the affine, not by rotating data. Registering on top of an already-correct
        alignment *destroys* it — measured here: identity gave 27/28 labelled slices,
        a 12-dof fit gave 10/28.

        ``register=True`` opts into a 12-dof affine fit, for the genuinely unaligned
        case (a stock template that was never brought into subject space). Never
        deformable: a few thick slices give a warp no through-plane evidence, so it
        invents structure an affine cannot. Labels are resampled once, atlas → target,
        and never linearly — interpolating label *ids* blends them into meaningless
        intermediate values.
        """
        import nibabel as nib
        from dipy.align.imaffine import AffineMap

        ref_img, lbl_img = nib.load(str(ref_p)), nib.load(str(lbl_p))
        lbl = np.asarray(lbl_img.dataobj)
        lbl_aff = np.asarray(lbl_img.affine, dtype=float)

        static = np.nan_to_num(t1w[..., 0] if t1w.ndim == 4 else t1w)
        xform = np.eye(4)
        if register:
            from dipy.align import affine_registration
            ref = np.asarray(ref_img.dataobj, dtype=float)
            ref_aff = np.asarray(ref_img.affine, dtype=float)
            iters = list(level_iters) if level_iters else [10000, 1000, 100]
            _, xform = affine_registration(
                moving=ref, static=static, moving_affine=ref_aff, static_affine=t1w_aff,
                nbins=int(nbins), level_iters=iters, pipeline=list(pipeline),
                sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1],
            )

        # Warp the labels onto the requested output grid (the DCE grid when the stage
        # supplies one, else the structural grid) in a single nearest-neighbour pass.
        out_aff = (np.asarray(target_affine, dtype=float)
                   if target_affine is not None else t1w_aff)
        out_shape = tuple(int(s) for s in (target_shape if target_shape is not None
                                           else static.shape[:3]))
        parc = AffineMap(xform, out_shape, out_aff, lbl.shape, lbl_aff).transform(
            lbl.astype(np.int32), interpolation="nearest").astype(np.int32)

        try:                                   # keep the warped map for inspection/reuse
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(parc, out_aff),
                     str(Path(out_dir) / "atlas_labels_target.nii.gz"))
        except Exception:                      # noqa: BLE001 — provenance is best-effort
            pass
        return parc, out_aff

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

        # Did the labels land on the anatomy? Overlap scores cannot tell — a
        # transposed affine once scored 27/28 slices while painting into empty air.
        # Structure positions can. Warn rather than fail: a wrong-looking layout is
        # worth stopping for, but the caller may know something we do not.
        from ._anatomy_qc import check_orientation
        anat = check_orientation(parc, aff, region_map)
        if anat["status"] == "warn":
            _log.warning("tissue_roi: parcellation may be mis-oriented — %s. "
                         "Check the affine before trusting any regional result.",
                         anat["summary"])
        elif anat["status"] == "ok":
            _log.info("tissue_roi: anatomy check passed (%s)", anat["summary"])

        return TissueROI(
            parcellation=parc.astype(np.int32),
            affine=aff.astype(float),
            labels=labels,
            region_map=region_map,
            meta={"algorithm": "mouse_atlas_labelmap", "path": src,
                  "n_labels": len(present), "n_regions": len(region_map),
                  "anatomy_check": anat},
        )


PLUGIN = _MouseAtlasROI()
