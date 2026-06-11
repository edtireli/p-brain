"""Patlak — legacy-parity variant (OLS + fixed upper-2/3 tail).

Thin wrapper over :class:`pbrain.models.patlak.PatlakModel` that pins
``tail_mode="legacy"`` so the regression matches the legacy
``models/patlak.fit_patlak`` byte-for-byte.

Use alongside the smart ``patlak`` default to get both readings in a
single pipeline run::

    --models patlak,patlak_legacy

The ``patlak`` outputs land at ``07_kinetic/patlak/voxelwise/{ki,vb}.nii.gz``;
the ``patlak_legacy`` outputs at ``07_kinetic/patlak_legacy/voxelwise/{ki,vb}.nii.gz``
so cohort scripts can pick whichever defaults match the published
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import CurveInputs, ModelResult
from .patlak import PLUGIN as _SMART


@dataclass(frozen=True, slots=True)
class _PatlakLegacy:
    key:         ClassVar[str] = "patlak_legacy"
    name:        ClassVar[str] = "Patlak — OLS + upper-2/3 tail (legacy parity)"
    description: ClassVar[str] = (
        "Bit-equal to the legacy ``models/patlak.fit_patlak`` solver. "
        "Uses OLS regression over the upper-2/3 ``x_max`` window. Run "
        "in parallel with the smart ``patlak`` default to cross-validate "
        "against historical / published values."
    )
    accepts:  ClassVar[dict[str, type]] = dict(_SMART.accepts)
    produces: ClassVar[dict[str, type]] = dict(_SMART.produces)
    outputs:  ClassVar[tuple[str, ...]] = tuple(_SMART.outputs)
    units:    ClassVar[dict[str, str]] = dict(_SMART.units)
    primary_map: ClassVar[str] = "ki"

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        opts = dict(opts)
        opts["tail_mode"] = "legacy"        # OLS over upper-2/3 (legacy semantics)
        # Byte-parity with the legacy dual-bolus fit, including its AIF-floor
        # bug (no low-Cₐ leverage-point exclusion). The smart ``patlak``
        # default fixes this; use that for robust cohort Ki.
        opts.setdefault("aif_floor_double_bolus", False)
        return _SMART.fit(inputs, **opts)


PLUGIN = _PatlakLegacy()
