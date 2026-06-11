"""Free-Water DTI — Pasternak two-compartment FW elimination via dipy.

Models the DWI signal as a mixture of tissue + free water (CSF / edema)
and reports the FW-corrected tensor (Pasternak 2009). Cleaner WM
metrics in voxels with partial-volume contamination, and the free-water
fraction itself is informative (oedema, atrophy, ageing).

Outputs:

* ``tfa``  — tissue FA (FW-corrected)
* ``tmd``  — tissue MD (FW-corrected)
* ``tad``  — tissue AD (FW-corrected)
* ``trd``  — tissue RD (FW-corrected)
* ``fw``   — free-water volume fraction (0–1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import DWIInputs, DiffusionResult


@dataclass(frozen=True, slots=True)
class _FWDTI:
    key:         ClassVar[str] = "fwdti"
    name:        ClassVar[str] = "Free-Water DTI (Pasternak)"
    description: ClassVar[str] = (
        "Two-compartment free-water elimination (Pasternak 2009) via "
        "dipy.reconst.fwdti. Returns tissue-only FA/MD/AD/RD plus the "
        "free-water fraction. Multi-shell strongly recommended."
    )
    accepts:  ClassVar[dict[str, type]] = {
        "signal": np.ndarray, "bvals": np.ndarray, "bvecs": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "tfa": np.ndarray, "tmd": np.ndarray, "tad": np.ndarray,
        "trd": np.ndarray, "fw": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("tfa", "tmd", "tad", "trd", "fw")
    units: ClassVar[dict[str, str]] = {
        "tfa": "(unitless)", "tmd": "mm²/s", "tad": "mm²/s",
        "trd": "mm²/s", "fw": "(fraction)",
    }
    primary_map: ClassVar[str] = "tfa"

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        from dipy.core.gradients import gradient_table
        from dipy.reconst import fwdti

        max_bval = float(opts.get("max_bval", 2000.0))
        signal = np.asarray(inputs.signal, dtype=np.float32)
        bvals = np.asarray(inputs.bvals, dtype=float)
        bvecs = np.asarray(inputs.bvecs, dtype=float)

        keep = bvals <= max_bval
        if int(np.sum(keep & (bvals > 50))) < 6:
            raise RuntimeError(
                f"FW-DTI needs ≥6 directions at b≤{max_bval}; have "
                f"{int(np.sum(keep & (bvals > 50)))}."
            )

        gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])
        model = fwdti.FreeWaterTensorModel(gtab)
        fit = model.fit(signal[..., keep], mask=inputs.mask)

        tfa = fit.fa.astype(np.float32)
        tmd = fit.md.astype(np.float32)
        tad = fit.ad.astype(np.float32)
        trd = fit.rd.astype(np.float32)
        fw  = np.asarray(fit.f, dtype=np.float32)

        if inputs.mask is not None:
            m = np.asarray(inputs.mask, dtype=bool)
            for arr in (tfa, tmd, tad, trd, fw):
                arr[~m] = np.nan

        return DiffusionResult(
            maps={"tfa": tfa, "tmd": tmd, "tad": tad, "trd": trd, "fw": fw},
            units=dict(self.units),
            aux={"kept_directions": int(np.sum(keep))},
        )


PLUGIN = _FWDTI()
