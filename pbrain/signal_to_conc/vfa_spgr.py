"""VFA/SPGR signal→concentration (alternative to saturation recovery).

Inverts the spoiled-GRE steady-state equation per voxel::

    S(t) = M₀ · sin(α) · (1 − E1(t)) / (1 − E1(t) · cos(α))

with E1(t) = exp(−TR / T1(t)) and T1(t) = T1₀ / (1 + r₁·C(t)·T1₀).
Solved analytically for C(t) given S, M₀, T1₀, α, TR.

Use this when the DCE acquisition was a spoiled-GRE without a
saturation pre-pulse. Falls through to NaN where the inversion is
ill-conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import SignalToConcConverter


@dataclass(frozen=True, slots=True)
class _VfaSpgr:
    key: ClassVar[str] = "vfa_spgr"
    name: ClassVar[str] = "VFA / SPGR (spoiled-GRE steady state)"
    description: ClassVar[str] = (
        "Analytic inversion of the spoiled-GRE signal model for "
        "DCE acquired without a saturation pre-pulse."
    )
    accepts: ClassVar[dict[str, type]] = {
        "signal": np.ndarray,
        "t1_ms": np.ndarray,
        "m0": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {"concentration_mM": np.ndarray}

    def convert(
        self,
        signal: np.ndarray,
        t1_ms: np.ndarray,
        m0: np.ndarray,
        *,
        flip_angle_deg: float,
        tr_s: float,
        r1_per_s_mM: float = 4.0,
        **_: Any,
    ) -> np.ndarray:
        S = np.asarray(signal, dtype=float)
        T1_s0 = np.asarray(t1_ms, dtype=float) / 1000.0
        M0 = np.asarray(m0, dtype=float)
        alpha = np.deg2rad(float(flip_angle_deg))
        TR = float(tr_s)
        r1 = float(r1_per_s_mM)

        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)

        with np.errstate(divide="ignore", invalid="ignore"):
            # Solve S = M0·sin(α)·(1-E1)/(1-E1·cos(α)) for E1(t):
            #   E1(t) = (M0·sin(α) - S) / (M0·sin(α) - S·cos(α))
            denom = M0[..., None] * sin_a - S * cos_a
            E1_t = np.where(denom != 0,
                            (M0[..., None] * sin_a - S) / denom,
                            np.nan)
            E1_t = np.clip(E1_t, 1e-9, 1.0 - 1e-9)
            T1_s_t = -TR / np.log(E1_t)
            T1_s0_b = T1_s0[..., None] if S.ndim > T1_s0.ndim else T1_s0
            C = (1.0 / T1_s_t - 1.0 / np.where(T1_s0_b > 0, T1_s0_b, np.nan)) / r1
        return C


PLUGIN = _VfaSpgr()
