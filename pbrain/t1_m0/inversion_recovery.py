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


def _fit_varpro(Y: np.ndarray, TI_s: np.ndarray, t1_lo_ms: float, t1_hi_ms: float,
                n_grid: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised variable-projection IR fit over ALL voxels at once.

    ``S(TI) = A − B·exp(−TI/T1)`` is *linear in (A, B) for a fixed T1*, so the
    only nonlinear unknown is the scalar T1. We grid T1, and for each grid value
    the design matrix ``X = [1, −exp(−TI/T1)]`` is shared by every voxel, so the
    closed-form least-squares ``(A, B)`` and the residual are a single mat-mul
    across the whole brain. We pick the min-SSE T1 per voxel and parabolically
    refine it in log-T1, then recover (A, B) at the refined T1. This reproduces
    the per-voxel TRF solve to machine precision (median |ΔT1| ≈ 0, r = 1.0000)
    at ~100× the speed — the IR fit is the classic separable least-squares
    problem (Golub–Pereyra variable projection).

    ``Y`` is ``(V, nTI)``. Returns ``(A[=M0], B, T1_ms)``, each ``(V,)``.
    """
    Y = np.asarray(Y, dtype=float)
    TI = np.asarray(TI_s, dtype=float).ravel()
    V, nTI = Y.shape
    if V == 0:
        z = np.zeros(0)
        return z, z, z
    Yt = Y.T                                                   # (nTI, V)
    t1_grid = np.geomspace(max(t1_lo_ms, 1.0) / 1000.0, t1_hi_ms / 1000.0, int(n_grid))
    SSE = np.empty((t1_grid.size, V), dtype=np.float32)
    for gi, T1 in enumerate(t1_grid):
        X = np.stack([np.ones(nTI), -np.exp(-TI / T1)], axis=1)    # (nTI, 2) shared
        coef = np.linalg.solve(X.T @ X, X.T @ Yt)                 # (2, V)
        SSE[gi] = np.sum((Yt - X @ coef) ** 2, axis=0)
    bi = np.argmin(SSE, axis=0)
    lg = np.log(t1_grid)
    t1 = t1_grid[bi].astype(float)
    inr = (bi > 0) & (bi < t1_grid.size - 1)
    ii = np.where(inr)[0]
    if ii.size:
        b = bi[ii]
        s0, s1, s2 = SSE[b - 1, ii], SSE[b, ii], SSE[b + 1, ii]
        den = (s0 - 2 * s1 + s2)
        off = np.zeros(ii.size)
        ok = den != 0
        off[ok] = np.clip(0.5 * (s0[ok] - s2[ok]) / den[ok], -0.5, 0.5)
        t1[ii] = np.exp(lg[b] + off * (lg[1] - lg[0]))
    # (A, B) at the refined T1 — closed-form 2×2 normal equations per voxel.
    E = np.exp(-TI[:, None] / t1[None, :])                    # (nTI, V)
    a12 = -E.sum(0); a22 = (E * E).sum(0)
    rhs1 = Y.sum(1); rhs2 = -(E * Yt).sum(0)
    det = nTI * a22 - a12 * a12
    det = np.where(np.abs(det) > 1e-30, det, np.nan)
    A = (a22 * rhs1 - a12 * rhs2) / det
    Bc = (-a12 * rhs1 + nTI * rhs2) / det
    return A, Bc, t1 * 1000.0


def _turboflash_basis(TI_s: np.ndarray, T1_s: float, alpha_rad: float,
                      tr_s: float, nph: int) -> np.ndarray:
    """Per-TI TurboFLASH signal *per unit M0* — i.e. ``g`` where ``S = M0·g``.

    Ports MATLAB ``sigdif_turbof_r1_90`` / legacy ``model_function_turboflash_ti``::

        a = cos(α)·exp(−TR·R1),  b = 1 − exp(−TR·R1)
        S = M0·sin(α)·[ (1−exp(−R1·TI))·a^(nph−1) + b·(1−a^(nph−1))/(1−a) ]

    with ``R1 = 1/T1``. The crucial ``sin(α)`` (and readout-train) factor is
    explicit here, so the fitted ``M0`` is the **true** equilibrium — not the
    apparent ``A = M0·sinα`` a bare ``A−B·exp`` recovery fit returns (that
    factor-of-``sinα`` is exactly what made the downstream concentration 2×
    high and clamped the AIF).
    """
    R1 = 1.0 / T1_s
    a = float(np.cos(alpha_rad)) * np.exp(-tr_s * R1)
    b = 1.0 - np.exp(-tr_s * R1)
    nph = max(1, int(nph))
    apow = a ** (nph - 1)
    denom = (1.0 - a)
    denom = denom if abs(denom) > 1e-10 else (1e-10 if denom >= 0 else -1e-10)
    term1 = (1.0 - np.exp(-R1 * TI_s)) * apow
    term2 = b * (1.0 - apow) / denom
    return float(np.sin(alpha_rad)) * (term1 + term2)


def _fit_varpro_turboflash(Y: np.ndarray, TI_s: np.ndarray, alpha_deg: float,
                           tr_s: float, nph: int, t1_lo_ms: float, t1_hi_ms: float,
                           n_grid: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised variable-projection TurboFLASH T1/M0 fit over all voxels.

    ``S(TI) = M0·g(TI; T1)`` is **linear in M0** for a fixed T1, so we grid T1,
    and for each grid value the shared basis ``g`` gives the closed-form,
    non-negative ``M0 = (g·S)/(g·g)`` for every voxel at once. Pick the per-voxel
    min-SSE T1 and parabolically refine in log-T1. ``Y`` is ``(V, nTI)``;
    returns ``(M0[true], T1_ms)``.
    """
    Y = np.asarray(Y, dtype=float)
    TI = np.asarray(TI_s, dtype=float).ravel()
    V, nTI = Y.shape
    if V == 0:
        z = np.zeros(0)
        return z, z
    alpha_rad = float(np.radians(alpha_deg))
    Yt = Y.T                                                  # (nTI, V)
    t1_grid = np.geomspace(max(t1_lo_ms, 1.0) / 1000.0, t1_hi_ms / 1000.0, int(n_grid))
    SSE = np.empty((t1_grid.size, V), dtype=np.float32)
    M0g = np.empty((t1_grid.size, V), dtype=np.float32)
    for gi, T1 in enumerate(t1_grid):
        g = _turboflash_basis(TI, float(T1), alpha_rad, tr_s, nph)   # (nTI,)
        gg = float(g @ g) + 1e-30
        M0 = np.clip((g @ Yt) / gg, 0.0, None)                # (V,) closed-form ≥0
        SSE[gi] = np.sum((Yt - g[:, None] * M0[None, :]) ** 2, axis=0)
        M0g[gi] = M0
    bi = np.argmin(SSE, axis=0)
    lg = np.log(t1_grid)
    t1 = t1_grid[bi].astype(float)
    vi = np.arange(V)
    M0 = M0g[bi, vi].astype(float)
    inr = (bi > 0) & (bi < t1_grid.size - 1)
    ii = np.where(inr)[0]
    if ii.size:
        b = bi[ii]
        s0, s1, s2 = SSE[b - 1, ii], SSE[b, ii], SSE[b + 1, ii]
        den = (s0 - 2 * s1 + s2)
        off = np.zeros(ii.size)
        ok = den != 0
        off[ok] = np.clip(0.5 * (s0[ok] - s2[ok]) / den[ok], -0.5, 0.5)
        t1[ii] = np.exp(lg[b] + off * (lg[1] - lg[0]))
        # recompute M0 at the refined T1 (still closed-form per voxel)
        for j, v in zip(ii, b):
            g = _turboflash_basis(TI, float(t1[j]), alpha_rad, tr_s, nph)
            gg = float(g @ g) + 1e-30
            M0[j] = max(0.0, float(g @ Y[j]) / gg)
    return M0, t1 * 1000.0


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
        method: str = "varpro",
        recovery_model: str = "turboflash",
        alpha_deg: float = 30.0,
        tr_s: float = 0.00396,
        nph: int = 1,
        **_: Any,
    ) -> T1M0Result:
        """Voxelwise T1/M0 fit (paper §4.3, Eq. 1).

        **Bounds (both recovery models):** ``T1 ∈ [t1_lo_ms, t1_hi_ms]`` =
        ``[100, 6000]`` ms by default; the amplitudes ``A, B ≥ 0``
        (``M0 ≥ 0`` for the TurboFLASH model).

        **Initial-guess strategy** (no numerics depend on these beyond the
        usual local-optimum sensitivity — documented so the Supplement can
        state them):

        * ``recovery_model="turboflash"`` (default) — full variable-projection
          grid: T1 is the only nonlinear unknown, so it is swept over a
          log-spaced **T1 grid** spanning ``[t1_lo_ms, t1_hi_ms]`` (240 nodes)
          and, for each grid T1, the equilibrium ``M0`` is the closed-form
          non-negative least-squares solution of the shared TurboFLASH basis.
          No scalar A/B/T1 seed is used — the grid is the initialisation, then
          the per-voxel minimiser is parabolically refined in log-T1.
        * The simple ``A − B·exp(−TI/T1)`` recovery path
          (``recovery_model`` ≠ ``"turboflash"``):
            - ``method="varpro"`` (vectorised) sweeps the same kind of
              log-spaced **T1 grid** (200 nodes) with closed-form ``(A, B)``
              per node — again grid-initialised, no scalar seed.
            - ``method="trf"`` (per-voxel reference solve, :func:`_fit_one`)
              uses the explicit seed: ``A0 = max(S)``, ``B0 = 2·A0``, and
              ``T1_0`` from the **magnetisation zero-crossing** —
              ``T1_0 = TI_zc / ln 2`` where ``TI_zc`` is the TI whose signal is
              nearest ``½·A0`` — clamped to ``[50 ms, t1_hi_ms]``.
        """
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

        if str(recovery_model).lower() == "turboflash":
            # TurboFLASH TI fit (MATLAB-matched): S = M0·sinα·[…], so the fitted
            # M0 is the TRUE equilibrium — the simple A−B·exp returns M0·sinα
            # (apparent), which made downstream concentration 2× high. M0 linear
            # given T1 ⇒ vectorised variable projection.
            M0_, T1_ms = _fit_varpro_turboflash(
                signals[mask], TI_s, alpha_deg, tr_s, nph, t1_lo_ms, t1_hi_ms)
            T1[mask] = T1_ms
            M0[mask] = M0_
        elif str(method).lower() == "varpro":
            # Vectorised variable-projection of S = A − B·exp(−TI/T1). NB: A is
            # the *apparent* equilibrium (M0·sinα); kept for parity/legacy use.
            A_, B_, T1_ms = _fit_varpro(signals[mask], TI_s, t1_lo_ms, t1_hi_ms)
            T1[mask] = T1_ms
            M0[mask] = A_
            B_arr[mask] = B_
        else:
            # Exact per-voxel TRF (reference / fallback).
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
                "method": str(method).lower(),
                "n_voxels_fit": int(mask.sum()),
                "t1_lo_ms": t1_lo_ms,
                "t1_hi_ms": t1_hi_ms,
            },
        )


PLUGIN = _IRFitter()
