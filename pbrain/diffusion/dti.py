"""DTI — diffusion tensor imaging via dipy.

Outputs (paper-standard scalar maps):

* ``fa``       — fractional anisotropy (unitless, 0–1)
* ``md``       — mean diffusivity (mm²/s)
* ``ad``       — axial diffusivity (mm²/s)
* ``rd``       — radial diffusivity (mm²/s)
* ``colorfa``  — direction-encoded RGB anatomical FA (channels packed
                 into the 4-th axis as ``(X, Y, Z, 3)``)

Default behaviour: drops shells above ``max_bval`` (default 1500) before
fitting, because the tensor model assumes Gaussian diffusion which breaks
down at b ≳ 2000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import DWIInputs, DiffusionResult


@dataclass(frozen=True, slots=True)
class _DTI:
    key:         ClassVar[str] = "dti"
    name:        ClassVar[str] = "Diffusion Tensor Imaging"
    description: ClassVar[str] = (
        "Per-voxel rank-2 tensor model (Basser & Pierpaoli 1996) via "
        "dipy.reconst.dti.TensorModel. FA, MD, AD, RD, plus a colour-"
        "coded direction-encoded FA volume."
    )
    accepts:  ClassVar[dict[str, type]] = {
        "signal": np.ndarray, "bvals": np.ndarray, "bvecs": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "fa": np.ndarray, "md": np.ndarray, "ad": np.ndarray,
        "rd": np.ndarray, "colorfa": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("fa", "md", "ad", "rd", "colorfa")
    units: ClassVar[dict[str, str]] = {
        "fa": "(unitless)", "md": "mm²/s", "ad": "mm²/s", "rd": "mm²/s",
        "colorfa": "RGB",
    }
    primary_map: ClassVar[str] = "fa"

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        from dipy.core.gradients import gradient_table
        from dipy.reconst.dti import TensorModel, color_fa

        max_bval = float(opts.get("max_bval", 1500.0))
        fit_method = str(opts.get("fit_method", "WLS"))   # WLS / LS / RT / NLLS / RESTORE

        signal = np.asarray(inputs.signal, dtype=np.float32)
        bvals = np.asarray(inputs.bvals, dtype=float)
        bvecs = np.asarray(inputs.bvecs, dtype=float)

        keep = bvals <= max_bval
        if int(np.sum(keep & (bvals > 50))) < 6:
            raise RuntimeError(
                f"DTI needs ≥6 directions with b≤{max_bval}; have "
                f"{int(np.sum(keep & (bvals > 50)))}."
            )

        gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])
        model = TensorModel(gtab, fit_method=fit_method)
        fit = model.fit(signal[..., keep], mask=inputs.mask)

        fa = fit.fa.astype(np.float32)
        md = fit.md.astype(np.float32)
        ad = fit.ad.astype(np.float32)
        rd = fit.rd.astype(np.float32)
        cfa = color_fa(np.clip(fa, 0.0, 1.0), fit.evecs).astype(np.float32)

        # Mask-aware NaN handling for downstream stats
        if inputs.mask is not None:
            m = np.asarray(inputs.mask, dtype=bool)
            for arr in (fa, md, ad, rd):
                arr[~m] = np.nan
            cfa[~m] = 0.0

        return DiffusionResult(
            maps={"fa": fa, "md": md, "ad": ad, "rd": rd, "colorfa": cfa},
            units=dict(self.units),
            aux={"fit_method": fit_method, "kept_directions": int(np.sum(keep))},
        )


PLUGIN = _DTI()
