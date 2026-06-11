"""CSD — constrained spherical deconvolution via dipy.

Recovers per-voxel fibre orientation distribution functions (fODFs)
from single-shell DWI by deconvolving the signal with a single-fibre
response function (Tournier 2007). Robust in crossing-fibre regions
where DTI's single-tensor model fails.

Outputs:

* ``gfa``        — generalised FA (Tuch 2004) — DTI-like anisotropy from
                   the fODF SH coefficients.
* ``peak1_dir``  — primary fibre direction, shape ``(X,Y,Z,3)``.
* ``peak1_amp``  — primary fODF peak amplitude.
* ``nufo``       — "number of fibre orientations" — count of fODF peaks
                   above a threshold per voxel (1, 2 or 3+).

Uses the b=1500 shell by default; configurable via ``--opt
diffusion.csd.fit_bval=...``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import DWIInputs, DiffusionResult


@dataclass(frozen=True, slots=True)
class _CSD:
    key:         ClassVar[str] = "csd"
    name:        ClassVar[str] = "Constrained Spherical Deconvolution"
    description: ClassVar[str] = (
        "CSD (Tournier 2007) via dipy.reconst.csdeconv. Recovers fibre "
        "ODFs and reports generalised FA, primary fibre direction + "
        "amplitude, and the number of fibres per voxel (NuFO)."
    )
    accepts:  ClassVar[dict[str, type]] = {
        "signal": np.ndarray, "bvals": np.ndarray, "bvecs": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "gfa": np.ndarray, "peak1_dir": np.ndarray,
        "peak1_amp": np.ndarray, "nufo": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("gfa", "peak1_dir", "peak1_amp", "nufo")
    units: ClassVar[dict[str, str]] = {
        "gfa": "(unitless)", "peak1_dir": "(unit vector)",
        "peak1_amp": "(amplitude)", "nufo": "(count)",
    }
    primary_map: ClassVar[str] = "gfa"

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        from dipy.core.gradients import gradient_table
        from dipy.reconst.csdeconv import (
            ConstrainedSphericalDeconvModel, auto_response_ssst,
        )
        try:
            from dipy.reconst.odf import gfa as _gfa     # dipy ≥ 1.10
        except ImportError:                              # pragma: no cover
            from dipy.reconst.shm import gfa as _gfa     # older dipy
        from dipy.direction import peaks_from_model
        from dipy.data import get_sphere

        # sh_order default 6 (28 SH coeffs); set lower (4 → 15 coeffs) for
        # sparse single-shell data with ≲30 directions.
        sh_order = int(opts.get("sh_order", 6))
        fa_thr = float(opts.get("response_fa_thr", 0.7))
        relative_peak_threshold = float(opts.get("relative_peak_threshold", 0.5))
        min_separation_angle = float(opts.get("min_separation_angle", 25.0))
        nufo_amp_thr = float(opts.get("nufo_amp_thr", 0.10))
        fit_bval = float(opts.get("fit_bval", 1500.0))
        shell_tol = float(opts.get("shell_tol", 200.0))

        signal = np.asarray(inputs.signal, dtype=np.float32)
        bvals = np.asarray(inputs.bvals, dtype=float)
        bvecs = np.asarray(inputs.bvecs, dtype=float)

        # Single-shell pick: b=0 + the chosen mid-shell.
        keep = (bvals <= 50) | (np.abs(bvals - fit_bval) <= shell_tol)
        if int(np.sum(keep & (bvals > 50))) < 6:
            raise RuntimeError(
                f"CSD needs ≥6 directions near b={fit_bval}±{shell_tol}; "
                f"got {int(np.sum(keep & (bvals > 50)))}."
            )

        gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])
        sub_signal = signal[..., keep]

        response, _ratio = auto_response_ssst(
            gtab, sub_signal, roi_radii=10, fa_thr=fa_thr,
        )
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
        sphere = get_sphere("symmetric724")
        peaks = peaks_from_model(
            csd_model, sub_signal, sphere,
            relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle,
            mask=inputs.mask,
            return_sh=True, normalize_peaks=False,
        )

        gfa = _gfa(peaks.shm_coeff).astype(np.float32)
        peak_amps = np.asarray(peaks.peak_values, dtype=np.float32)   # (X,Y,Z,K)
        peak_dirs = np.asarray(peaks.peak_dirs, dtype=np.float32)     # (X,Y,Z,K,3)
        peak1_amp = peak_amps[..., 0] if peak_amps.size else np.zeros(signal.shape[:3], dtype=np.float32)
        peak1_dir = peak_dirs[..., 0, :] if peak_dirs.size else np.zeros(signal.shape[:3] + (3,), dtype=np.float32)
        nufo = (peak_amps > nufo_amp_thr).sum(axis=-1).astype(np.float32)

        if inputs.mask is not None:
            m = np.asarray(inputs.mask, dtype=bool)
            gfa[~m] = np.nan
            peak1_amp[~m] = np.nan
            nufo[~m] = np.nan
            peak1_dir[~m] = 0.0

        return DiffusionResult(
            maps={
                "gfa": gfa, "peak1_dir": peak1_dir,
                "peak1_amp": peak1_amp, "nufo": nufo,
            },
            units=dict(self.units),
            aux={"response": response, "sh_order": sh_order, "fit_bval": fit_bval},
        )


PLUGIN = _CSD()
