"""Tikhonov — legacy-parity variant (L-curve + linear λ-grid).

Thin wrapper over :class:`pbrain.models.tikhonov.TikhonovModel` that
pins ``lambda_selection="lcurve"`` and ``lambda_spacing="linear"`` and
restores the legacy ``lambda_min=0.05`` floor — bit-equal to the legacy
``models/tikhonov.build_tikhonov_solver``.

Use alongside the smart ``tikhonov`` default to get both readings in a
single pipeline run::

    --models tikhonov,tikhonov_legacy

The smart ``tikhonov`` writes to ``07_kinetic/tikhonov/voxelwise/{cbf,mtt,cth,lambda_opt}.nii.gz``;
``tikhonov_legacy`` writes to ``07_kinetic/tikhonov_legacy/voxelwise/...``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import CurveInputs, ModelResult
from .tikhonov import PLUGIN as _SMART


@dataclass(frozen=True, slots=True)
class _TikhonovLegacy:
    key:         ClassVar[str] = "tikhonov_legacy"
    name:        ClassVar[str] = "Tikhonov — L-curve + linear λ-grid (legacy parity)"
    description: ClassVar[str] = (
        "Bit-equal to legacy ``models/tikhonov.build_tikhonov_solver``. "
        "Restores the L-curve max-curvature picker on the original "
        "``linspace(0.05, σ_max, 121)`` grid. Run in parallel with the "
        "smart ``tikhonov`` GCV+log-spaced default to cross-validate "
        "against historical / published values."
    )
    accepts:  ClassVar[dict[str, type]] = dict(_SMART.accepts)
    produces: ClassVar[dict[str, type]] = dict(_SMART.produces)
    outputs:  ClassVar[tuple[str, ...]] = tuple(_SMART.outputs)
    units:    ClassVar[dict[str, str]] = dict(_SMART.units)
    primary_map: ClassVar[str] = "cbf"

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        opts = dict(opts)
        opts.setdefault("lambda_selection", "lcurve")
        opts.setdefault("lambda_spacing",   "linear")
        opts.setdefault("lambda_min",       0.05)       # legacy floor
        opts.setdefault("lambda_max",       40.0)       # legacy fixed grid top
        opts.setdefault("n_lambdas",        121)        # legacy grid resolution
        opts.setdefault("mtt_cth_method",   "residue_integral")
        return _SMART.fit(inputs, **opts)

    def predict(self, maps: dict[str, Any], c_input: np.ndarray,
                t_s: np.ndarray) -> np.ndarray:
        return _SMART.predict(maps, c_input, t_s)


PLUGIN = _TikhonovLegacy()
