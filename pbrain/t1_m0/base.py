"""T1/M0 fitting contract — paper §4.3.

A *fitter* takes a stack of signal volumes (one per inversion time or
flip angle) plus the corresponding 4th-axis values and returns voxelwise
T1 (ms) and M0 (signal units) maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


@dataclass(frozen=True, slots=True)
class T1M0Result:
    """Voxelwise T1/M0 maps returned by a fitter.

    Attributes
    ----------
    t1_map_ms
        ``(X, Y, Z)`` longitudinal relaxation time in milliseconds.
    m0_map
        ``(X, Y, Z)`` equilibrium magnetisation in signal units.
    meta
        Free-form provenance (sequence, fit method, masked fraction, …).
    """

    t1_map_ms: np.ndarray            # (X, Y, Z)
    m0_map: np.ndarray               # (X, Y, Z)
    meta: dict[str, Any]


@runtime_checkable
class T1M0Fitter(Plugin, Protocol):
    """Plugin sub-Protocol for T1/M0 estimation."""

    def fit(
        self,
        signals: np.ndarray | None,  # (X, Y, Z, N) signals; None for preloaded-style plug-ins
        axis_values: np.ndarray | None,  # (N,) inversion times in seconds OR flip angles in deg
        *,
        mask: np.ndarray | None = None,
        **opts: Any,
    ) -> T1M0Result:
        """Estimate voxelwise T1 (ms) and M0 from a signal stack.

        ``signals`` is ``(X, Y, Z, N)`` with ``axis_values`` giving the
        ``N`` inversion times (s) or flip angles (deg); both are
        ``None`` for preloaded-style plug-ins that read maps off disk.
        Returns a :class:`T1M0Result`.
        """
        ...
