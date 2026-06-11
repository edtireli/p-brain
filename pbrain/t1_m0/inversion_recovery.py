"""Inversion-recovery T1/M0 fit — paper §4.3, Eq. 1.

Voxelwise nonlinear least-squares fit of the standard IR signal model::

    F(TI; A, B, T1) = A − B · exp(−TI / T1)

This is the magnitude formulation of S(TI) = κ·M0·(1 − 2η·exp(−TI/T1));
the inversion efficiency η and receive-gain κ are absorbed into A and B.
Bounds: T1 ∈ [100 ms, 6000 ms], A and B ≥ 0 (paper default).

The solver is ``scipy.optimize.least_squares`` with TRF bounds, mirroring
the legacy ``modules/opt01_T1_fit.py`` solver choices. Parity tests in
``tests/test_pbrain_models_parity.py`` confirm byte-equal results on
synthetic IR phantoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import least_squares

from .base import T1M0Fitter, T1M0Result


def _fit_one(TI_s: np.ndarray, S: np.ndarray, t1_lo_ms: float, t1_hi_ms: float
             ) -> tuple[float, float, float]:
    """Fit one voxel. Returns (A, B, T1_ms)."""
    valid = np.isfinite(TI_s) & np.isfinite(S)
    n = int(valid.sum())
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    TI = TI_s[valid]
    Y = S[valid]

    # Initial guess: A ≈ max(S), B ≈ 2·max(S), T1 ≈ TI at zero-crossing
    A0 = float(np.max(Y))
    B0 = 2.0 * A0
    zero_idx = int(np.argmin(np.abs(Y - 0.5 * A0)))
    T1_0_ms = max(50.0, min(float(TI[zero_idx] * 1000.0 / np.log(2)), float(t1_hi_ms)))

    lo = np.array([0.0, 0.0, float(t1_lo_ms) / 1000.0])
    hi = np.array([1e6, 1e6, float(t1_hi_ms) / 1000.0])
    x0 = np.array([A0, B0, T1_0_ms / 1000.0])
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)

    def residual(x: np.ndarray) -> np.ndarray:
        A, B, T1 = x
        return A - B * np.exp(-TI / T1) - Y

    try:
        sol = least_squares(residual, x0, bounds=(lo, hi), method="trf", max_nfev=200)
        A, B, T1_s = sol.x
        return float(A), float(B), float(T1_s * 1000.0)
    except Exception:
        return float("nan"), float("nan"), float("nan")


@dataclass(frozen=True, slots=True)
class _IRFitter:
    key: ClassVar[str] = "inversion_recovery"
    name: ClassVar[str] = "Inversion-recovery T1/M0 fit"
    description: ClassVar[str] = (
        "Voxelwise nonlinear least-squares fit of F(TI; A, B, T1) = "
        "A − B·exp(−TI/T1) (paper §4.3 Eq. 1). M0 ≡ A (saturation "
        "recovery convention)."
    )
    accepts: ClassVar[dict[str, type]] = {
        "signals": np.ndarray,
        "axis_values": np.ndarray,
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
        t1_lo_ms: float = 100.0,
        t1_hi_ms: float = 6000.0,
        **_: Any,
    ) -> T1M0Result:
        signals = np.asarray(signals, dtype=float)
        if signals.ndim != 4:
            raise ValueError(f"signals must be 4-D (X,Y,Z,N); got {signals.shape}")
        TI_s = np.asarray(axis_values, dtype=float).ravel()
        if TI_s.size != signals.shape[-1]:
            raise ValueError(
                f"axis_values length {TI_s.size} != signals.shape[-1] {signals.shape[-1]}"
            )

        X, Y, Z, _ = signals.shape
        T1 = np.full((X, Y, Z), np.nan, dtype=float)
        M0 = np.full((X, Y, Z), np.nan, dtype=float)
        B_arr = np.full((X, Y, Z), np.nan, dtype=float)

        if mask is None:
            # Auto-mask: fit only voxels with real signal (skip air). The IR
            # T1 fit is a per-voxel non-linear solve, so masking out ~60% air
            # voxels both speeds it up and avoids fitting noise.
            smax = np.nanmax(np.abs(signals), axis=-1)
            pos = smax[np.isfinite(smax) & (smax > 0)]
            thr = 0.08 * float(np.percentile(pos, 98)) if pos.size else 0.0
            mask = smax > thr
        else:
            mask = np.asarray(mask, dtype=bool)

        for i, j, k in np.argwhere(mask):
            A_, B_, T1_ms = _fit_one(TI_s, signals[i, j, k, :], t1_lo_ms, t1_hi_ms)
            T1[i, j, k] = T1_ms
            M0[i, j, k] = A_
            B_arr[i, j, k] = B_

        return T1M0Result(
            t1_map_ms=T1,
            m0_map=M0,
            meta={
                "fitter": "inversion_recovery",
                "n_voxels_fit": int(mask.sum()),
                "t1_lo_ms": t1_lo_ms,
                "t1_hi_ms": t1_hi_ms,
            },
        )


PLUGIN = _IRFitter()
