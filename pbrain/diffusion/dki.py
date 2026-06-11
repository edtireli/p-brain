"""DKI — diffusion kurtosis imaging via dipy.

Adds kurtosis tensors on top of the diffusion tensor (Jensen et al. 2005,
Tabesh et al. 2011). Multi-shell required (≥ 2 non-zero b-values).

Outputs:

* ``mk``  — mean kurtosis
* ``ak``  — axial kurtosis
* ``rk``  — radial kurtosis
* ``kfa`` — kurtosis fractional anisotropy (Hansen, Lund, Sangill 2013)
* ``fa``, ``md``, ``ad``, ``rd`` — DTI scalars from the DKI fit (often
  cleaner than the pure DTI fit because DKI explicitly accounts for the
  non-Gaussian signal at higher b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import DWIInputs, DiffusionResult


@dataclass(frozen=True, slots=True)
class _DKI:
    key:         ClassVar[str] = "dki"
    name:        ClassVar[str] = "Diffusion Kurtosis Imaging"
    description: ClassVar[str] = (
        "DKI (Jensen 2005) via dipy.reconst.dki.DiffusionKurtosisModel. "
        "Adds mean/axial/radial kurtosis + KFA to the standard DTI "
        "scalars. Requires multi-shell DWI (≥2 non-zero b-values)."
    )
    accepts:  ClassVar[dict[str, type]] = {
        "signal": np.ndarray, "bvals": np.ndarray, "bvecs": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "mk": np.ndarray, "ak": np.ndarray, "rk": np.ndarray, "kfa": np.ndarray,
        "fa": np.ndarray, "md": np.ndarray, "ad": np.ndarray, "rd": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = (
        "mk", "ak", "rk", "kfa", "fa", "md", "ad", "rd",
    )
    units: ClassVar[dict[str, str]] = {
        "mk": "(unitless)", "ak": "(unitless)", "rk": "(unitless)",
        "kfa": "(unitless)",
        "fa": "(unitless)", "md": "mm²/s", "ad": "mm²/s", "rd": "mm²/s",
    }
    primary_map: ClassVar[str] = "mk"

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        from dipy.core.gradients import gradient_table
        from dipy.reconst.dki import DiffusionKurtosisModel

        max_bval = float(opts.get("max_bval", 3000.0))   # DKI tolerates higher b than DTI
        fit_method = str(opts.get("fit_method", "WLS"))

        signal = np.asarray(inputs.signal, dtype=np.float32)
        bvals = np.asarray(inputs.bvals, dtype=float)
        bvecs = np.asarray(inputs.bvecs, dtype=float)

        keep = bvals <= max_bval
        bvals_k = bvals[keep]
        nonzero = bvals_k > 50
        shells = {int(round(b / 100.0) * 100) for b in bvals_k[nonzero]}
        if len(shells) < 2:
            raise RuntimeError(
                f"DKI requires ≥2 non-zero shells; got {sorted(shells)}. "
                f"Single-shell data → use the `dti` plug-in instead."
            )

        gtab = gradient_table(bvals_k, bvecs=bvecs[keep])
        model = DiffusionKurtosisModel(gtab, fit_method=fit_method)
        fit = model.fit(signal[..., keep], mask=inputs.mask)

        # Kurtosis scalars are clipped to a physiological range (0, 3]
        mk = fit.mk(0.0, 3.0).astype(np.float32)
        ak = fit.ak(0.0, 3.0).astype(np.float32)
        rk = fit.rk(0.0, 3.0).astype(np.float32)
        kfa = fit.kfa.astype(np.float32)
        fa = fit.fa.astype(np.float32)
        md = fit.md.astype(np.float32)
        ad = fit.ad.astype(np.float32)
        rd = fit.rd.astype(np.float32)

        if inputs.mask is not None:
            m = np.asarray(inputs.mask, dtype=bool)
            for arr in (mk, ak, rk, kfa, fa, md, ad, rd):
                arr[~m] = np.nan

        return DiffusionResult(
            maps={"mk": mk, "ak": ak, "rk": rk, "kfa": kfa,
                  "fa": fa, "md": md, "ad": ad, "rd": rd},
            units=dict(self.units),
            aux={"fit_method": fit_method, "kept_shells": sorted(shells)},
        )


PLUGIN = _DKI()
