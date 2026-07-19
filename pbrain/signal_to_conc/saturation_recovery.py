"""Saturation-recovery signal→concentration — paper §4.5, Eq. 2.

    C(t) = − 1/r₁ · [ 1/T_D · ln(1 − S(t) / (M₀ · sin α))  +  1/T₁ ]

* r₁ — gadolinium relaxivity (default 4 s⁻¹·mM⁻¹ at 3 T)
* T_D — prepulse-to-readout delay (default 120 ms)
* α  — readout flip angle in radians
* T₁, M₀ — fitted baseline maps

Matches the validated pre-modular p-brain ``utils/plotting.turboflash``:
the log argument is RAW (no abs, no clamp); a saturated voxel
(S > M₀·sinα ⇒ ratio > 1) yields NaN and is zeroed at the end. Pass
``log_abs=True`` for the MATLAB ``real(log)`` (|1−ratio|) variant, which
keeps such bright voxels as large finite concentrations (the ~6 mM
artefact). The default parameters reproduce the paper's Philips 3 T
protocol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import SignalToConcConverter
# detect_baseline_frames re-exported for backward compatibility; the baseline
# detector is now modular (selectable via the ``baseline_method`` option).
from .baseline import detect_baseline_frames, resolve_baseline_frames  # noqa: F401


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
        baseline_frames: int = 10,
        baseline_method: str = "auto",
        log_abs: bool = False,
        **_: Any,
    ) -> np.ndarray:
        S = np.asarray(signal, dtype=float)
        T1_s = np.asarray(t1_ms, dtype=float) / 1000.0
        M0 = np.asarray(m0, dtype=float)
        TD = float(prepulse_to_readout_s)
        r1 = float(r1_per_s_mM)
        sin_a = float(np.sin(np.radians(float(flip_angle_deg))))

        # Pre-contrast baseline window — auto-detected pre-bolus period (ported
        # from the legacy find_shifted_baseline) rather than a fixed window.
        nbf = resolve_baseline_frames(S, method=baseline_method,
                                      fallback=int(baseline_frames))

        # TurboFLASH saturation-recovery inversion — the exact MATLAB conversion
        # (``create_MR_concentration_deck.m:218`` / legacy ``utils/plotting.turboflash``):
        #     C(t) = −1/(r₁·T_D) · [ ln(1 − S(t)/(M₀·sinα)) + T_D·R1₀ ],
        # baseline-corrected. The equilibrium scale is **M₀·sinα** with the
        # *fitted* M₀ and the readout flip angle α — NOT a baseline-derived K.
        # Both DCE and the T1/M0-fit IR series are dcm2niix-converted with their
        # Philips proxy slopes applied, so M₀ and S share one intensity scale and
        # S/(M₀·sinα) stays < 1 (verified ~0.17–0.20, no clamp). ``real(log(·))``
        # in MATLAB ⇒ ln|1−ratio| for the rare bright voxel where ratio>1.
        if not np.isfinite(sin_a) or abs(sin_a) < 1e-8:
            return np.full_like(S, np.nan, dtype=float)
        inv_T1 = 1.0 / np.where(T1_s > 0, T1_s, np.nan)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if S.ndim > M0.ndim:
                denom = (M0 * sin_a)[..., None]
                inv_T1b = inv_T1[..., None]
            else:
                denom = M0 * sin_a
                inv_T1b = inv_T1
            ratio = np.where(np.isfinite(denom) & (denom > 0), S / denom, np.nan)
            # OLD p-brain (utils/plotting._turboflash_raw_concentration): the log
            # argument is RAW — NO abs(), NO clamp. A voxel that saturates
            # (S > M0·sinα ⇒ ratio > 1) gives log(negative) → NaN, which the final
            # nan_to_num zeroes. log_abs=True keeps those bright voxels as large
            # finite values (MATLAB real(log) ⇒ |1−ratio|) — that is the ~6 mM
            # artefact and is NOT what the validated pre-modular pipeline produced.
            if log_abs:
                inside = np.abs(1.0 - ratio)
                inside = np.where(inside > 1e-12, inside, 1e-12)
            else:
                inside = 1.0 - ratio
            C = -(np.log(inside) / TD + inv_T1b) / r1
            # Baseline-correct (skip first 2 frames; mean of frames 3..nbf) so
            # C(baseline)=0, then zero the baseline window — matches old turboflash().
            if C.ndim >= 1 and C.shape[-1] >= nbf:
                # Background/air voxels are all-NaN across the baseline window, so
                # np.nanmean warns "Mean of empty slice" — expected here; the
                # np.where below already substitutes 0 for those non-finite means.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
                    base = np.nanmean(C[..., 2:nbf], axis=-1, keepdims=True)
                C = C - np.where(np.isfinite(base), base, 0.0)
                C[..., :nbf] = 0.0
        # Old pipeline zeroes NaN/±inf at the very end.
        return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)


PLUGIN = _SaturationRecovery()
