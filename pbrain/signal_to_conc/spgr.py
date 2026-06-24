"""Steady-state spoiled gradient-echo (SPGR/FLASH) signal→concentration.

A selectable *alternative* to the default saturation-recovery TurboFLASH
converter, for data acquired with a steady-state spoiled-GRE DCE readout
rather than a saturation-prepared TurboFLASH. Port of the supervisor's
``dceSignalToConcentration.m`` (`cFcn`):

    A = S / (M0·sin α)
    C = R1₀/r − ln( (A−1) / (A·cos α − 1) ) / (r·TR)

derived by inverting the Ernst/SPGR steady state
``S = M0·sin α·(1−E1)/(1−cos α·E1)``, ``E1 = exp(−TR·R1)``, for R1 and taking
``C = (R1 − R1₀)/r``. Unlike the saturation-recovery model this uses **both**
sin α and cos α and the repetition time ``TR`` (not a prepulse delay), and it
relies on a calibrated ``M0`` map (no baseline self-scaling). The constant
R1₀/r offset is removed by the baseline correction, so its sign convention is
immaterial — matching the reference, which baseline-corrects afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import SignalToConcConverter
from .baseline import resolve_baseline_frames


@dataclass(frozen=True, slots=True)
class _SPGR:
    key: ClassVar[str] = "spgr"
    name: ClassVar[str] = "Spoiled gradient-echo steady state (SPGR/FLASH)"
    description: ClassVar[str] = (
        "Steady-state spoiled-GRE conversion (port of dceSignalToConcentration). "
        "Alternative to the default saturation-recovery TurboFLASH; for "
        "steady-state spoiled-GRE DCE readouts. Uses sin α, cos α and TR, and a "
        "calibrated M0 map."
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
        baseline_frames: int = 10,
        baseline_method: str = "auto",
        **_: Any,
    ) -> np.ndarray:
        S = np.asarray(signal, dtype=float)
        T1_s = np.asarray(t1_ms, dtype=float) / 1000.0
        M0 = np.asarray(m0, dtype=float)
        TR = float(tr_s)
        r1 = float(r1_per_s_mM)
        alpha = np.deg2rad(float(flip_angle_deg))
        sin_a, cos_a = np.sin(alpha), np.cos(alpha)

        nbf = resolve_baseline_frames(S, method=baseline_method,
                                      fallback=int(baseline_frames))

        inv_T1 = 1.0 / np.where(T1_s > 0, T1_s, np.nan)            # R1₀ per voxel
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denom = M0 * sin_a
            scale = np.where(np.isfinite(denom) & (denom > 0), denom, np.nan)
            if S.ndim > T1_s.ndim:
                A = S / scale[..., None]
                inv_T1b = inv_T1[..., None]
            else:
                A = S / scale
                inv_T1b = inv_T1
            # E1 = (A−1)/(A·cosα−1) ∈ (0,1]; clamp to keep the log finite.
            num = A - 1.0
            den = A * cos_a - 1.0
            E1 = np.where(np.abs(den) > 1e-12, num / den, np.nan)
            E1 = np.clip(E1, 1e-12, 1.0)
            C = inv_T1b / r1 - np.log(E1) / (r1 * TR)
            # Baseline: subtract the pre-bolus mean so C(baseline)=0.
            if C.ndim >= 1 and C.shape[-1] >= nbf:
                base = np.nanmean(C[..., 2:nbf], axis=-1, keepdims=True)
                C = C - np.where(np.isfinite(base), base, 0.0)
        return C


PLUGIN = _SPGR()
