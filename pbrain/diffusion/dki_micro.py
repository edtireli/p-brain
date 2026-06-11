"""DKI-microstructure — white-matter microstructure from kurtosis.

Implements the Fieremans–Hansen white-matter standard-model (WMTI):
fits the DKI tensors and converts them to compartment fractions /
diffusivities under the assumption of a two-compartment WM
microstructure (intra-axonal + extra-axonal). Closest pragmatic
substitute for true B-tensor encoded μFA on LTE-only acquisitions.

Outputs:

* ``awf``       — axonal water fraction (≈ NODDI's intra-cellular vol)
* ``tortuosity``— extra-axonal tortuosity ``D_axial / D_radial``
* ``de_axial``  — extra-axonal axial diffusivity (mm²/s)
* ``de_radial`` — extra-axonal radial diffusivity (mm²/s)
* ``ufa``       — microscopic FA from the DKI tensors (Hansen 2017
                  approximation — closest scalar to true μFA without
                  B-tensor encoding)

Requires multi-shell. The plug-in is gated on ≥ 2 non-zero b-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import DWIInputs, DiffusionResult


@dataclass(frozen=True, slots=True)
class _DKIMicro:
    key:         ClassVar[str] = "dki_micro"
    name:        ClassVar[str] = "DKI-microstructure (Fieremans/Hansen)"
    description: ClassVar[str] = (
        "Two-compartment WM standard-model parameters derived from the "
        "DKI tensors (Fieremans 2011, Hansen 2017). Outputs axonal "
        "water fraction, extra-axonal diffusivities, tortuosity, and "
        "μFA — the closest substitute for B-tensor μFA on LTE-only data."
    )
    accepts:  ClassVar[dict[str, type]] = {
        "signal": np.ndarray, "bvals": np.ndarray, "bvecs": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "awf": np.ndarray, "tortuosity": np.ndarray,
        "de_axial": np.ndarray, "de_radial": np.ndarray,
        "ufa": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = (
        "awf", "tortuosity", "de_axial", "de_radial", "ufa",
    )
    units: ClassVar[dict[str, str]] = {
        "awf": "(fraction)", "tortuosity": "(unitless)",
        "de_axial": "mm²/s", "de_radial": "mm²/s",
        "ufa": "(unitless)",
    }
    primary_map: ClassVar[str] = "awf"

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        from dipy.core.gradients import gradient_table
        from dipy.reconst.dki_micro import KurtosisMicrostructureModel

        max_bval = float(opts.get("max_bval", 3000.0))
        signal = np.asarray(inputs.signal, dtype=np.float32)
        bvals = np.asarray(inputs.bvals, dtype=float)
        bvecs = np.asarray(inputs.bvecs, dtype=float)

        keep = bvals <= max_bval
        shells = {int(round(b / 100.0) * 100) for b in bvals[keep] if b > 50}
        if len(shells) < 2:
            raise RuntimeError(
                f"DKI-microstructure requires ≥2 non-zero shells; got {sorted(shells)}."
            )

        gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])
        model = KurtosisMicrostructureModel(gtab)
        fit = model.fit(signal[..., keep], mask=inputs.mask)

        awf = fit.awf.astype(np.float32)
        de_axial = fit.axonal_diffusivity.astype(np.float32) if hasattr(fit, "axonal_diffusivity") else fit.hindered_ad.astype(np.float32)
        de_radial = fit.hindered_rd.astype(np.float32)
        # Tortuosity = D_∥ / D_⊥ in extra-axonal space
        with np.errstate(divide="ignore", invalid="ignore"):
            tort = (de_axial / np.where(de_radial > 0, de_radial, np.nan)).astype(np.float32)

        # μFA via Hansen 2017: derived from KFA + FA blend. Use the fitted
        # KFA directly as a robust proxy where DKI tensors are well-posed.
        from dipy.reconst.dki import DiffusionKurtosisModel
        dki_fit = DiffusionKurtosisModel(gtab).fit(signal[..., keep], mask=inputs.mask)
        kfa = dki_fit.kfa.astype(np.float32)
        fa = dki_fit.fa.astype(np.float32)
        # Hansen-style blend: μFA² ≈ FA² + (KFA²)·(1 − FA²)
        ufa2 = fa ** 2 + (kfa ** 2) * (1.0 - fa ** 2)
        ufa = np.sqrt(np.clip(ufa2, 0.0, 1.0)).astype(np.float32)

        if inputs.mask is not None:
            m = np.asarray(inputs.mask, dtype=bool)
            for arr in (awf, tort, de_axial, de_radial, ufa):
                arr[~m] = np.nan

        return DiffusionResult(
            maps={
                "awf": awf, "tortuosity": tort,
                "de_axial": de_axial, "de_radial": de_radial,
                "ufa": ufa,
            },
            units=dict(self.units),
            aux={"kept_shells": sorted(shells)},
        )


PLUGIN = _DKIMicro()
