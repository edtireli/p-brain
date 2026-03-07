"""Extended Tofts pharmacokinetic model for DCE-MRI permeability analysis.

Implements the Extended Tofts model (ETM) for estimating BBB permeability
parameters from concentration-time curves:

    Ct(t) = vp·Cp(t) + Ktrans · ∫₀ᵗ Cp(τ)·exp(-kep·(t-τ)) dτ

where kep = Ktrans / ve.

Parameters estimated:
    Ktrans  — volume transfer constant (1/s, reported as ml/100g/min × 6000)
    ve      — extravascular extracellular volume fraction
    vp      — plasma volume fraction

The model is fit using constrained least-squares with optional Tikhonov
regularisation to stabilise the solution.

References
----------
    Tofts PS, et al. (1999) JMRI 10:223–232
    Tofts PS (1997) JMRI 7:91–101

This module follows the same structure as ``models/patlak.py`` and
``models/tikhonov.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

if hasattr(np, "trapezoid"):
    _trapezoid = np.trapezoid
else:  # pragma: no cover — legacy NumPy fallback
    _trapezoid = np.trapz


# ── Model registry entry (mirrors patlak.py / tikhonov.py) ───────────────
MODEL = {
    "key": "extended_tofts",
    "env": "extended_tofts",
    "description": "Extended Tofts model for Ktrans / ve / vp estimation.",
    "aliases": {
        "extended_tofts",
        "etofts",
        "etm",
        "extended-tofts",
        "tofts",
    },
}


# ── Result dataclasses ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class EToftsFit:
    """Result of a single-voxel Extended Tofts fit.

    Units follow the legacy p-brain conventions:
        ktrans_per_min   — Ktrans in 1/min  (raw 1/s × 60)
        ktrans_ml_100g   — Ktrans in ml/100g/min  (raw 1/s × 6000)
        ve               — extra-vascular extra-cellular volume fraction [0-1]
        vp               — plasma volume fraction [0-1]
        kep_per_min      — kep = Ktrans/ve in 1/min
        fitted_curve     — model prediction at the input time points
        residuals        — Ct_measured − Ct_predicted
        cost             — sum of squared residuals
        success          — whether the optimiser converged
    """

    ktrans_per_min: float
    ktrans_ml_100g: float
    ve: float
    vp: float
    kep_per_min: float
    fitted_curve: np.ndarray
    residuals: np.ndarray
    cost: float
    success: bool


@dataclass(frozen=True, slots=True)
class EToftsBatchResult:
    """Vectorised result of Extended Tofts fits for many voxels.

    All arrays have shape ``(n_voxels,)`` except ``fitted_curves`` and
    ``residuals`` which are ``(n_voxels, n_timepoints)``.
    """

    ktrans_per_min: np.ndarray
    ktrans_ml_100g: np.ndarray
    ve: np.ndarray
    vp: np.ndarray
    kep_per_min: np.ndarray
    fitted_curves: np.ndarray
    residuals: np.ndarray
    cost: np.ndarray
    success: np.ndarray


# ── Forward model ────────────────────────────────────────────────────────

def forward(
    t: np.ndarray,
    Ktrans: float,
    ve: float,
    vp: float,
    Cp: np.ndarray,
) -> np.ndarray:
    """Evaluate the Extended Tofts model at time points *t*.

    Uses an O(n) trapezoidal recurrence when the time grid is strictly
    increasing; falls back to O(n²) direct quadrature otherwise.

    Parameters
    ----------
    t : 1-D array
        Time points in **seconds**.
    Ktrans : float
        Volume transfer constant in **1/s**.
    ve : float
        Extra-vascular extra-cellular volume fraction.
    vp : float
        Plasma volume fraction.
    Cp : 1-D array
        Plasma (AIF) concentration curve, same length as *t*.

    Returns
    -------
    Ct : 1-D array
        Predicted tissue concentration curve.
    """

    t = np.asarray(t, dtype=float).ravel()
    Cp = np.asarray(Cp, dtype=float).ravel()
    n = min(t.size, Cp.size)
    if n == 0:
        return np.zeros(0, dtype=float)

    t = t[:n]
    Cp = Cp[:n]

    Ktrans = float(Ktrans)
    ve = float(ve)
    vp = float(vp)
    if not (np.isfinite(Ktrans) and np.isfinite(ve) and np.isfinite(vp)):
        return np.full(n, np.nan, dtype=float)
    ve = max(ve, 1e-12)

    kep = Ktrans / ve
    dt = np.diff(t)

    if dt.size > 0 and np.all(np.isfinite(dt)) and np.all(dt > 0):
        # O(n) trapezoidal recurrence
        I = np.zeros(n, dtype=float)
        for i in range(1, n):
            dti = float(dt[i - 1])
            e = float(np.exp(-kep * dti))
            I[i] = I[i - 1] * e + 0.5 * dti * (Cp[i] + Cp[i - 1] * e)
    else:
        # O(n²) fallback for irregular grids
        I = np.zeros(n, dtype=float)
        for i in range(n):
            I[i] = _trapezoid(
                Cp[: i + 1] * np.exp(-(t[i] - t[: i + 1]) * kep),
                x=t[: i + 1],
            )

    return Ktrans * I + vp * Cp


# ── Single-voxel fitting ────────────────────────────────────────────────

_DEFAULT_X0 = (0.001, 0.2, 0.05)  # (Ktrans, ve, vp) in 1/s, -, -

# Physiological bounds:
#   Ktrans: 0 .. 0.25 1/s  (~1500 ml/100g/min — very generous upper limit)
#   ve:     0 .. 1.0
#   vp:     0 .. 1.0
_DEFAULT_BOUNDS_LO = (0.0, 0.0, 0.0)
_DEFAULT_BOUNDS_HI = (0.25, 1.0, 1.0)


def fit_voxel(
    Cp: np.ndarray,
    Ct: np.ndarray,
    t: np.ndarray,
    *,
    x0: tuple[float, float, float] = _DEFAULT_X0,
    bounds_lo: tuple[float, float, float] = _DEFAULT_BOUNDS_LO,
    bounds_hi: tuple[float, float, float] = _DEFAULT_BOUNDS_HI,
    tikhonov_lambda: float = 0.0,
    init_ktrans: float | None = None,
    init_vp: float | None = None,
) -> EToftsFit:
    """Fit the Extended Tofts model to a single tissue voxel.

    Parameters
    ----------
    Cp : 1-D array
        Arterial input function (plasma concentration).
    Ct : 1-D array
        Tissue concentration curve.
    t  : 1-D array
        Time points in seconds.
    x0 : tuple
        Initial guess for (Ktrans, ve, vp).  Overridden by *init_ktrans*
        and *init_vp* when provided.
    bounds_lo, bounds_hi : tuple
        Lower/upper bounds for (Ktrans, ve, vp).
    tikhonov_lambda : float
        Regularisation weight.  0 = no regularisation (ordinary least
        squares).  A small value (e.g. 0.01–0.1) adds a Tikhonov penalty
        ``√λ · w · θ`` to stabilise the fit.
    init_ktrans : float, optional
        If given, override ``x0[0]`` with this value (in 1/s).
        Useful when initialising from a Patlak Ki estimate
        (``Ki_ml_per_100g_min / 6000``).
    init_vp : float, optional
        If given, override ``x0[2]`` with this value.

    Returns
    -------
    EToftsFit
    """

    Cp = np.asarray(Cp, dtype=float).ravel()
    Ct = np.asarray(Ct, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()

    n = min(Cp.size, Ct.size, t.size)
    if n < 3:
        return _nan_fit(n)

    Cp = Cp[:n]
    Ct = Ct[:n]
    t = t[:n]

    # Drop non-finite samples
    valid = np.isfinite(Cp) & np.isfinite(Ct) & np.isfinite(t)
    if valid.sum() < 3:
        return _nan_fit(n)

    if not np.all(valid):
        Cp = Cp[valid]
        Ct = Ct[valid]
        t = t[valid]

    # Build initial guess
    x0_arr = np.array(list(x0), dtype=float)
    if init_ktrans is not None and np.isfinite(init_ktrans):
        x0_arr[0] = max(float(init_ktrans), 1e-8)
    if init_vp is not None and np.isfinite(init_vp):
        x0_arr[2] = np.clip(float(init_vp), 0.0, 0.99)
    x0_arr = np.clip(x0_arr, 1e-12, None)

    lo = np.array(list(bounds_lo), dtype=float)
    hi = np.array(list(bounds_hi), dtype=float)

    lam = float(tikhonov_lambda)

    def _residual(theta: np.ndarray) -> np.ndarray:
        theta = np.clip(theta, 1e-12, None)
        Ct_pred = forward(t, theta[0], theta[1], theta[2], Cp)
        misfit = Ct_pred - Ct
        if np.any(~np.isfinite(misfit)):
            return np.full(misfit.size + (theta.size if lam > 0 else 0), 1e6)
        if lam > 0:
            w = np.linalg.norm(misfit) / max(np.linalg.norm(theta), 1e-8)
            penalty = np.sqrt(lam) * w * theta
            return np.concatenate([misfit, penalty])
        return misfit

    try:
        sol = least_squares(
            _residual,
            x0_arr,
            bounds=(lo, hi),
            method="trf",
            max_nfev=500,
        )
        Ktrans_s, ve, vp = sol.x
        success = bool(sol.success)
        cost_val = float(sol.cost)
    except Exception:
        return _nan_fit(n)

    # Compute fitted curve on original (possibly full) grid
    fitted = forward(t, Ktrans_s, ve, vp, Cp)
    resid = Ct - fitted

    ve_safe = max(ve, 1e-12)
    return EToftsFit(
        ktrans_per_min=float(Ktrans_s * 60.0),
        ktrans_ml_100g=float(Ktrans_s * 6000.0),
        ve=float(ve),
        vp=float(vp),
        kep_per_min=float(Ktrans_s / ve_safe * 60.0),
        fitted_curve=fitted,
        residuals=resid,
        cost=cost_val,
        success=success,
    )


def _nan_fit(n: int) -> EToftsFit:
    """Return an all-NaN fit result."""

    return EToftsFit(
        ktrans_per_min=float("nan"),
        ktrans_ml_100g=float("nan"),
        ve=float("nan"),
        vp=float("nan"),
        kep_per_min=float("nan"),
        fitted_curve=np.full(max(n, 0), np.nan),
        residuals=np.full(max(n, 0), np.nan),
        cost=float("nan"),
        success=False,
    )


# ── Batch (multi-voxel) fitting ─────────────────────────────────────────

def fit_batch(
    Cp: np.ndarray,
    Ct_matrix: np.ndarray,
    t: np.ndarray,
    *,
    x0: tuple[float, float, float] = _DEFAULT_X0,
    bounds_lo: tuple[float, float, float] = _DEFAULT_BOUNDS_LO,
    bounds_hi: tuple[float, float, float] = _DEFAULT_BOUNDS_HI,
    tikhonov_lambda: float = 0.0,
    init_ktrans_array: np.ndarray | None = None,
    init_vp_array: np.ndarray | None = None,
) -> EToftsBatchResult:
    """Fit the Extended Tofts model to many voxels.

    Parameters
    ----------
    Cp : 1-D array, shape (n_time,)
        Shared arterial input function.
    Ct_matrix : 2-D array, shape (n_time, n_voxels) or (n_voxels, n_time)
        Tissue concentration curves.  Detected orientation: the axis
        matching ``len(t)`` is treated as time.
    t : 1-D array, shape (n_time,)
        Time points in seconds.
    x0, bounds_lo, bounds_hi : tuple
        Defaults for all voxels.
    tikhonov_lambda : float
        Regularisation weight passed to each voxel fit.
    init_ktrans_array : 1-D array, shape (n_voxels,), optional
        Per-voxel initial Ktrans (in 1/s).
    init_vp_array : 1-D array, shape (n_voxels,), optional
        Per-voxel initial vp.

    Returns
    -------
    EToftsBatchResult
    """

    Cp = np.asarray(Cp, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    Ct_mat = np.asarray(Ct_matrix, dtype=float)

    # Auto-detect orientation
    n_time = t.size
    if Ct_mat.ndim == 1:
        Ct_mat = Ct_mat.reshape(1, -1)
    if Ct_mat.ndim != 2:
        raise ValueError("Ct_matrix must be 1-D or 2-D")
    if Ct_mat.shape[0] == n_time and Ct_mat.shape[1] != n_time:
        # (n_time, n_voxels) → transpose to (n_voxels, n_time)
        Ct_mat = Ct_mat.T
    elif Ct_mat.shape[1] != n_time:
        raise ValueError(
            f"Ct_matrix shape {Ct_mat.shape} does not match len(t)={n_time}"
        )

    n_vox = Ct_mat.shape[0]

    # Pre-allocate outputs
    ktrans_pm = np.full(n_vox, np.nan)
    ktrans_ml = np.full(n_vox, np.nan)
    ve_arr = np.full(n_vox, np.nan)
    vp_arr = np.full(n_vox, np.nan)
    kep_arr = np.full(n_vox, np.nan)
    fitted = np.full((n_vox, n_time), np.nan)
    resid = np.full((n_vox, n_time), np.nan)
    cost_arr = np.full(n_vox, np.nan)
    success_arr = np.zeros(n_vox, dtype=bool)

    for j in range(n_vox):
        ik = None if init_ktrans_array is None else float(init_ktrans_array[j])
        iv = None if init_vp_array is None else float(init_vp_array[j])
        res = fit_voxel(
            Cp,
            Ct_mat[j],
            t,
            x0=x0,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            tikhonov_lambda=tikhonov_lambda,
            init_ktrans=ik,
            init_vp=iv,
        )
        ktrans_pm[j] = res.ktrans_per_min
        ktrans_ml[j] = res.ktrans_ml_100g
        ve_arr[j] = res.ve
        vp_arr[j] = res.vp
        kep_arr[j] = res.kep_per_min
        n_fit = min(res.fitted_curve.size, n_time)
        fitted[j, :n_fit] = res.fitted_curve[:n_fit]
        resid[j, :n_fit] = res.residuals[:n_fit]
        cost_arr[j] = res.cost
        success_arr[j] = res.success

    return EToftsBatchResult(
        ktrans_per_min=ktrans_pm,
        ktrans_ml_100g=ktrans_ml,
        ve=ve_arr,
        vp=vp_arr,
        kep_per_min=kep_arr,
        fitted_curves=fitted,
        residuals=resid,
        cost=cost_arr,
        success=success_arr,
    )


# ── Convenience: legacy-compatible tuple interface ───────────────────────

def fit_voxel_tuple(
    Cp: np.ndarray,
    Ct: np.ndarray,
    t: np.ndarray,
    *,
    tikhonov_lambda: float = 0.0,
    x0: tuple[float, float, float] = _DEFAULT_X0,
    init_ktrans: float | None = None,
    init_vp: float | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Legacy-compatible wrapper returning ``(Ktrans_1/s, ve, vp, fitted, residuals)``.

    This matches the return signature of the old
    ``kinetic_models.extended_tofts_tikhonov`` + forward-model call.
    """

    res = fit_voxel(
        Cp, Ct, t,
        x0=x0,
        tikhonov_lambda=tikhonov_lambda,
        init_ktrans=init_ktrans,
        init_vp=init_vp,
    )
    # Return raw Ktrans in 1/s (not scaled) for compatibility
    Ktrans_s = res.ktrans_per_min / 60.0 if np.isfinite(res.ktrans_per_min) else float("nan")
    return Ktrans_s, res.ve, res.vp, res.fitted_curve, res.residuals
