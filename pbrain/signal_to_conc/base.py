"""Signal-to-concentration converter contract — paper §4.5.

A *converter* maps DCE signal-time curves S(t) into gadolinium
concentration C(t) using a fitted T1 map, an M0 map, the readout flip
angle, and any sequence-specific timing parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


@runtime_checkable
class SignalToConcConverter(Plugin, Protocol):
    """Plugin sub-Protocol for S(t) → C(t) conversion."""

    def convert(
        self,
        signal: np.ndarray,          # (T,) or (T, V) or (X, Y, Z, T)
        t1_ms: np.ndarray,           # () scalar, (V,) or (X, Y, Z)
        m0: np.ndarray,              # same shape as t1_ms
        *,
        flip_angle_deg: float,
        tr_s: float,
        r1_per_s_mM: float = 4.0,
        **opts: Any,
    ) -> np.ndarray:                 # concentration mM, same shape as signal
        """Convert DCE signal S(t) into gadolinium concentration C(t).

        Uses the fitted ``t1_ms`` and ``m0`` maps, the readout
        ``flip_angle_deg``/``tr_s``, and the contrast relaxivity
        ``r1_per_s_mM`` (paper Eq. 2). ``signal`` may be ``(T,)``,
        ``(T, V)`` or ``(X, Y, Z, T)``; the result is the concentration
        in mM with the same shape.
        """
        ...
