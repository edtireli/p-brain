"""Tractography — whole-brain deterministic streamlines via dipy.

Reconstructs per-voxel fibre directions (CSD fODF peaks by default, or a
single-tensor DTI fit), then runs dipy deterministic :class:`LocalTracking`
seeded throughout the brain mask. Tracking propagates along the peak
directions and stops where anisotropy (GFA for CSD, FA for DTI) drops below
a threshold (:class:`ThresholdStoppingCriterion`).

Outputs:

* ``track_density`` — per-voxel count of streamlines passing through each
                      voxel (a "track density imaging"-style scalar map),
                      shape ``(X, Y, Z)``. This is the aggregatable volume.

The streamlines themselves are written as a ``.tck`` sidecar
(``track_density`` aside, see ``aux["sidecars"]["streamlines"]``) in the
DWI's own world (RASMM) space via nibabel's ``TckFile`` — the standard
MRtrix/dipy interchange format. The :class:`pbrain.stages.DiffusionStage`
persists it next to the native maps as ``streamlines.tck``.

This is deliberately OFF the default critical path: like all of the
``--diffusion`` track it only runs when explicitly requested, and the
weights-free demo never touches it.

Key options (``--opt diffusion.tractography.<name>=...``):

* ``recon``                    — ``"csd"`` (default) or ``"dti"``.
* ``stop_thr``                 — anisotropy stopping threshold (0.1).
* ``seed_density``             — seeds per voxel per axis (1).
* ``step_size``                — tracking step in mm (0.5).
* ``min_length``               — discard streamlines shorter than this mm (0 = keep all).
* ``relative_peak_threshold`` / ``min_separation_angle`` / ``sh_order`` /
  ``fit_bval`` / ``shell_tol`` — CSD fODF-peak controls (mirror ``csd.py``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import DWIInputs, DiffusionResult


@dataclass(frozen=True, slots=True)
class _Tractography:
    key:         ClassVar[str] = "tractography"
    name:        ClassVar[str] = "Deterministic tractography"
    description: ClassVar[str] = (
        "Whole-brain deterministic streamline tractography via "
        "dipy.tracking.local_tracking.LocalTracking. Reconstructs fibre "
        "directions (CSD fODF peaks or DTI) and tracks from a brain-mask "
        "seed set with an FA/GFA stopping criterion. Writes streamlines as "
        "a .tck (nibabel TckFile) plus a streamline-density map."
    )
    accepts:  ClassVar[dict[str, type]] = {
        "signal": np.ndarray, "bvals": np.ndarray, "bvecs": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "track_density": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("track_density",)
    units: ClassVar[dict[str, str]] = {
        "track_density": "(streamlines/voxel)",
    }
    primary_map: ClassVar[str] = "track_density"

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        from dipy.core.gradients import gradient_table
        from dipy.data import get_sphere
        from dipy.direction import peaks_from_model
        from dipy.tracking.local_tracking import LocalTracking
        from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
        from dipy.tracking.streamline import Streamlines
        from dipy.tracking.utils import seeds_from_mask
        from nibabel.streamlines import Tractogram, TckFile

        recon = str(opts.get("recon", "csd")).lower()
        stop_thr = float(opts.get("stop_thr", 0.1))
        seed_density = int(opts.get("seed_density", 1))
        step_size = float(opts.get("step_size", 0.5))
        min_length = float(opts.get("min_length", 0.0))
        max_streamlines = int(opts.get("max_streamlines", 100000))

        signal = np.asarray(inputs.signal, dtype=np.float32)
        bvals = np.asarray(inputs.bvals, dtype=float)
        bvecs = np.asarray(inputs.bvecs, dtype=float)
        affine = np.asarray(inputs.affine, dtype=float)
        shape = signal.shape[:3]

        if inputs.mask is not None:
            mask = np.asarray(inputs.mask, dtype=bool)
        else:
            b0 = signal[..., bvals <= 50].mean(axis=-1)
            thr = float(np.percentile(b0[b0 > 0], 25)) if (b0 > 0).any() else 0.0
            mask = b0 > thr

        sphere = get_sphere(name="symmetric724")
        rel_peak = float(opts.get("relative_peak_threshold", 0.5))
        min_sep = float(opts.get("min_separation_angle", 25.0))

        if recon == "dti":
            from dipy.reconst.dti import TensorModel

            gtab = gradient_table(bvals, bvecs=bvecs)
            model = TensorModel(gtab)
            peaks = peaks_from_model(
                model, signal, sphere,
                relative_peak_threshold=rel_peak,
                min_separation_angle=min_sep,
                mask=mask, return_sh=True, npeaks=2,
            )
            stop_metric = np.nan_to_num(
                np.asarray(model.fit(signal, mask=mask).fa, dtype=np.float32),
                nan=0.0,
            )
        else:  # CSD (default) — single-shell fODF peaks (mirrors csd.py).
            from dipy.reconst.csdeconv import (
                ConstrainedSphericalDeconvModel, auto_response_ssst,
            )

            sh_order = int(opts.get("sh_order", 6))
            fa_thr = float(opts.get("response_fa_thr", 0.7))
            fit_bval = float(opts.get("fit_bval", 1500.0))
            shell_tol = float(opts.get("shell_tol", 200.0))

            keep = (bvals <= 50) | (np.abs(bvals - fit_bval) <= shell_tol)
            n_dwi = int(np.sum(keep & (bvals > 50)))
            if n_dwi < 6:
                raise RuntimeError(
                    f"tractography(csd) needs ≥6 directions near "
                    f"b={fit_bval}±{shell_tol}; got {n_dwi}."
                )
            # Cap the spherical-harmonic order to what the direction count can
            # support: a CSD fit needs (L+1)(L+2)/2 ≤ n_directions, else the
            # fODF is under-determined and the peaks (hence the streamlines) are
            # unstable/one-sided. Reduce to the largest even order that fits.
            _supported = max([L for L in (2, 4, 6, 8)
                              if (L + 1) * (L + 2) // 2 <= n_dwi] or [2])
            if sh_order > _supported:
                import logging
                logging.getLogger("pbrain.diffusion.tractography").warning(
                    "tractography(csd): %d directions support SH order ≤%d; "
                    "reducing sh_order %d→%d (use recon='dti' for very few "
                    "directions).", n_dwi, _supported, sh_order, _supported)
                sh_order = _supported
            gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])
            sub_signal = signal[..., keep]
            response, _ratio = auto_response_ssst(
                gtab, sub_signal, roi_radii=10, fa_thr=fa_thr,
            )
            model = ConstrainedSphericalDeconvModel(
                gtab, response, sh_order_max=sh_order,
            )
            peaks = peaks_from_model(
                model, sub_signal, sphere,
                relative_peak_threshold=rel_peak,
                min_separation_angle=min_sep,
                mask=mask, return_sh=True, npeaks=2,
            )
            try:
                from dipy.reconst.odf import gfa as _gfa     # dipy ≥ 1.10
            except ImportError:                              # pragma: no cover
                from dipy.reconst.shm import gfa as _gfa     # older dipy
            stop_metric = np.nan_to_num(
                np.asarray(_gfa(peaks.shm_coeff), dtype=np.float32), nan=0.0,
            )

        stopping = ThresholdStoppingCriterion(stop_metric, stop_thr)
        seeds = seeds_from_mask(mask, affine=affine, density=seed_density)

        streamlines = Streamlines(
            LocalTracking(
                peaks, stopping, seeds, affine=affine, step_size=step_size,
            )
        )

        from dipy.tracking.utils import length as _length

        if min_length > 0 and len(streamlines):
            lengths = np.fromiter(_length(streamlines), dtype=float)
            streamlines = Streamlines(
                [s for s, ln in zip(streamlines, lengths) if ln >= min_length]
            )
        if max_streamlines and len(streamlines) > max_streamlines:
            streamlines = Streamlines(streamlines[:max_streamlines])

        # Track-density: count streamline visits per voxel.
        from dipy.tracking.utils import density_map

        if len(streamlines):
            density = density_map(streamlines, affine=affine, vol_dims=shape)
        else:
            density = np.zeros(shape, dtype=np.int64)
        density = density.astype(np.float32)

        # Serialise streamlines to a .tck byte payload in RASMM/world space.
        tractogram = Tractogram(list(streamlines), affine_to_rasmm=np.eye(4))
        buf = io.BytesIO()
        TckFile(tractogram).save(buf)

        return DiffusionResult(
            maps={"track_density": density},
            units=dict(self.units),
            aux={
                "recon": recon,
                "n_streamlines": int(len(streamlines)),
                "n_seeds": int(len(seeds)),
                "sidecars": {"streamlines": ("tck", buf.getvalue())},
            },
        )


PLUGIN = _Tractography()
