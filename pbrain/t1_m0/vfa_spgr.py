"""Variable Flip Angle (VFA) SPGR T1/M0 fit.

Linearised SPGR equation::

    S(α) = M0 · sin(α) · (1 − E1) / (1 − E1 · cos(α)),   E1 = exp(−TR/T1)

Reorganised to a linear form in (M0, T1) by plotting S/sin(α) vs
S/tan(α), with slope = E1 and intercept = M0·(1−E1). The closed-form
fit is fast and stable when ≥ 2 flip angles are present.

Use the inversion-recovery fitter when IR data is available (more
accurate); use VFA when only multi-flip-angle SPGR is available
(matches the legacy ``modules/opt01_T1_fit.py`` VFA branch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import T1M0Fitter, T1M0Result


@dataclass(frozen=True, slots=True)
class _VfaFitter:
    key: ClassVar[str] = "vfa_spgr"
    name: ClassVar[str] = "Variable flip angle SPGR T1/M0 fit"
    description: ClassVar[str] = (
        "Linearised SPGR fit: plot S/sin(α) vs S/tan(α), slope = exp(-TR/T1), "
        "intercept = M0·(1 − exp(-TR/T1)). Closed-form OLS per voxel."
    )
    accepts: ClassVar[dict[str, type]] = {
        "signals": np.ndarray,
        "axis_values": np.ndarray,    # flip angles in degrees
    }
    produces: ClassVar[dict[str, type]] = {
        "t1_ms": np.ndarray,
        "m0": np.ndarray,
    }

    def fit(
        self,
        signals: np.ndarray,
        axis_values: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        tr_s: float = 0.01118,
        t1_lo_ms: float = 100.0,
        t1_hi_ms: float = 6000.0,
        **_: Any,
    ) -> T1M0Result:
        signals = np.asarray(signals, dtype=float)
        if signals.ndim != 4:
            raise ValueError(f"signals must be 4-D (X,Y,Z,N); got {signals.shape}")
        alpha_deg = np.asarray(axis_values, dtype=float).ravel()
        if alpha_deg.size != signals.shape[-1]:
            raise ValueError("axis_values length must match signals.shape[-1]")
        if alpha_deg.size < 2:
            raise ValueError("VFA fit needs at least 2 flip angles")

        alpha_rad = np.deg2rad(alpha_deg)
        sin_a = np.sin(alpha_rad)
        tan_a = np.tan(alpha_rad)

        X = signals / sin_a                  # broadcast over voxels
        Y = signals / tan_a
        # OLS: x_v · slope + intercept = y_v across N flip angles, per voxel.
        mean_x = np.mean(X, axis=-1)
        mean_y = np.mean(Y, axis=-1)
        cov = np.mean((X - mean_x[..., None]) * (Y - mean_y[..., None]), axis=-1)
        var = np.mean((X - mean_x[..., None]) ** 2, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(var > 0, cov / var, np.nan)
            intercept = mean_y - slope * mean_x

        E1 = np.clip(slope, 1e-6, 1 - 1e-6)
        T1_s = -float(tr_s) / np.log(E1)
        T1_ms = T1_s * 1000.0
        M0 = intercept / np.maximum(1.0 - E1, 1e-12)

        T1_ms = np.where((T1_ms >= t1_lo_ms) & (T1_ms <= t1_hi_ms), T1_ms, np.nan)

        if mask is not None:
            keep = np.asarray(mask, dtype=bool)
            T1_ms = np.where(keep, T1_ms, np.nan)
            M0 = np.where(keep, M0, np.nan)

        return T1M0Result(
            t1_map_ms=T1_ms,
            m0_map=M0,
            meta={"fitter": "vfa_spgr", "tr_s": float(tr_s)},
        )


PLUGIN = _VfaFitter()
