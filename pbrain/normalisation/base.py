"""Curve-normaliser contract — paper §4.5.1.

A *normaliser* takes concentration-time curves (1-D or batched 2-D) and
returns baseline-corrected, scaled curves ready for kinetic modelling.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


@runtime_checkable
class CurveNormaliser(Plugin, Protocol):
    """Plugin sub-Protocol for concentration-curve normalisation."""

    def normalise(
        self,
        curve: np.ndarray,           # (T,) or (T, V)
        t_s: np.ndarray,             # (T,)
        *,
        baseline_frames: int = 5,
        reference_curve: np.ndarray | None = None,   # for shared-scale normalisers
        **opts: Any,
    ) -> np.ndarray: ...
