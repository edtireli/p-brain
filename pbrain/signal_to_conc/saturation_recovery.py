"""Saturation-recovery signal→concentration — paper §4.5, Eq. 2.

    C(t) = − 1/r₁ · [ 1/T_D · ln(1 − S(t) / (M₀ · sin α))  +  1/T₁ ]

* r₁ — gadolinium relaxivity (default 4 s⁻¹·mM⁻¹ at 3 T)
* T_D — prepulse-to-readout delay (default 120 ms)
* α  — readout flip angle in radians
* T₁, M₀ — fitted baseline maps

Clamps the log argument to (0, 1) for numerical robustness, returning
NaN where the signal/M0 inversion is ill-conditioned. The default
parameters reproduce the paper's Philips 3 T protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import SignalToConcConverter


@dataclass(frozen=True, slots=True)
class _SaturationRecovery:
    key: ClassVar[str] = "saturation_recovery"
    name: ClassVar[str] = "Saturation-recovery (paper Eq. 2)"
    description: ClassVar[str] = (
        "Spoiled-GRE saturation-recovery model. The default p-Brain "
        "signal→concentration conversion. Vectorised; same C(t) as "
        "legacy AI_tissue_functions for identical inputs."
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
        prepulse_to_readout_s: float = 0.120,
        **_: Any,
    ) -> np.ndarray:
        S = np.asarray(signal, dtype=float)
        T1_s = np.asarray(t1_ms, dtype=float) / 1000.0
        M0 = np.asarray(m0, dtype=float)
        alpha = np.deg2rad(float(flip_angle_deg))
        TD = float(prepulse_to_readout_s)
        r1 = float(r1_per_s_mM)

        # Broadcast scalar/3-D denom up to S's rank when S has a trailing time axis.
        denom = M0 * np.sin(alpha)
        if S.ndim > T1_s.ndim:
            denom_b = denom[..., None]
            inv_T1 = (1.0 / np.where(T1_s > 0, T1_s, np.nan))[..., None]
        else:
            denom_b = denom
            inv_T1 = 1.0 / np.where(T1_s > 0, T1_s, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom_b > 0, S / denom_b, np.nan)
            arg = 1.0 - ratio
            arg = np.clip(arg, 1e-12, 1.0 - 1e-12)
            ln_term = np.log(arg) / TD
            C = -(ln_term + inv_T1) / r1
        return C


PLUGIN = _SaturationRecovery()
