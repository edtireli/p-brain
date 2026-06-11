"""Extended Tofts model — Ktrans, v_e, v_p estimation.

    C_t(t) = v_p · C_p(t)  +  K^trans · ∫₀ᵗ C_p(τ) · exp(−k_ep·(t−τ)) dτ

with k_ep = K^trans / v_e. Fit by constrained Levenberg–Marquardt
least-squares (``scipy.optimize.least_squares`` with TRF bounds), with
optional Tikhonov regularisation to stabilise high-noise fits.

Clean port of ``models/extended_tofts.py`` (validated against MATLAB).
Numerics are unchanged so parity tests give byte-equal results.

References: Tofts PS et al., JMRI 10:223 (1999); Tofts PS, JMRI 7:91 (1997).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import least_squares

from .base import CurveInputs, ModelResult


# ────────────────────────────────────────────────────────────────────────
# Forward model
# ────────────────────────────────────────────────────────────────────────


def forward(
    t: np.ndarray,
    Ktrans: float,
    ve: float,
    vp: float,
    Cp: np.ndarray,
) -> np.ndarray:
    """Extended Tofts forward model. O(n) trapezoidal recurrence."""
    t = np.asarray(t, dtype=float).ravel()
    Cp = np.asarray(Cp, dtype=float).ravel()
    n = int(min(t.size, Cp.size))
    if n == 0:
        return np.zeros(0, dtype=float)
    t, Cp = t[:n], Cp[:n]

    Ktrans = float(Ktrans)
    ve = float(ve)
    vp = float(vp)
    if not (np.isfinite(Ktrans) and np.isfinite(ve) and np.isfinite(vp)):
        return np.full(n, np.nan, dtype=float)
    ve = max(ve, 1e-12)

    kep = Ktrans / ve
    dt = np.diff(t)

    I = np.zeros(n, dtype=float)
    if dt.size > 0 and np.all(np.isfinite(dt)) and np.all(dt > 0):
        # Trapezoidal recurrence:
        #   I[i] = exp(-kep·dt)·I[i-1] + 0.5·dt·(Cp[i] + Cp[i-1]·exp(-kep·dt))
        for i in range(1, n):
            dti = float(dt[i - 1])
            e = float(np.exp(-kep * dti))
            I[i] = I[i - 1] * e + 0.5 * dti * (Cp[i] + Cp[i - 1] * e)
    else:
        # Irregular-grid fallback
        for i in range(n):
            I[i] = float(np.trapezoid(
                Cp[: i + 1] * np.exp(-(t[i] - t[: i + 1]) * kep),
                x=t[: i + 1],
            ))

    return Ktrans * I + vp * Cp


# ────────────────────────────────────────────────────────────────────────
# Single-voxel fit
# ────────────────────────────────────────────────────────────────────────


_DEFAULT_X0 = (0.001, 0.2, 0.05)         # (Ktrans 1/s, ve, vp)
_DEFAULT_BOUNDS_LO = (0.0, 0.0, 0.0)
_DEFAULT_BOUNDS_HI = (0.25, 1.0, 1.0)    # Ktrans ≤ 1500 mL/100g/min


@dataclass(frozen=True, slots=True)
class EToftsFit:
    ktrans_per_min: float
    ktrans_ml_100g: float
    ve: float
    vp: float
    kep_per_min: float
    fitted_curve: np.ndarray
    residuals: np.ndarray
    cost: float
    success: bool


def _nan_fit(n: int) -> EToftsFit:
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
    """Fit Extended Tofts on one tissue voxel."""
    Cp = np.asarray(Cp, dtype=float).ravel()
    Ct = np.asarray(Ct, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()

    n = int(min(Cp.size, Ct.size, t.size))
    if n < 3:
        return _nan_fit(n)
    Cp, Ct, t = Cp[:n], Ct[:n], t[:n]

    valid = np.isfinite(Cp) & np.isfinite(Ct) & np.isfinite(t)
    if int(valid.sum()) < 3:
        return _nan_fit(n)
    if not np.all(valid):
        Cp, Ct, t = Cp[valid], Ct[valid], t[valid]

    x0_arr = np.array(list(x0), dtype=float)
    if init_ktrans is not None and np.isfinite(init_ktrans):
        x0_arr[0] = max(float(init_ktrans), 1e-8)
    if init_vp is not None and np.isfinite(init_vp):
        x0_arr[2] = float(np.clip(init_vp, 0.0, 0.99))
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
            _residual, x0_arr, bounds=(lo, hi), method="trf", max_nfev=500,
        )
        Ktrans_s, ve, vp = sol.x
        success = bool(sol.success)
        cost = float(sol.cost)
    except Exception:
        return _nan_fit(n)

    fitted = forward(t, Ktrans_s, ve, vp, Cp)
    resid = Ct - fitted
    ve_safe = max(float(ve), 1e-12)

    return EToftsFit(
        ktrans_per_min=float(Ktrans_s * 60.0),
        ktrans_ml_100g=float(Ktrans_s * 6000.0),
        ve=float(ve),
        vp=float(vp),
        kep_per_min=float(Ktrans_s / ve_safe * 60.0),
        fitted_curve=fitted,
        residuals=resid,
        cost=cost,
        success=success,
    )


# ────────────────────────────────────────────────────────────────────────
# Plug-in adapter
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExtendedToftsModel:
    key: ClassVar[str] = "extended_tofts"
    name: ClassVar[str] = "Extended Tofts model"
    description: ClassVar[str] = (
        "K^trans, v_e, v_p from constrained Levenberg-Marquardt fit "
        "(scipy.optimize.least_squares, TRF) of the extended Tofts "
        "compartment model. Optional Tikhonov regularisation."
    )
    accepts: ClassVar[dict[str, type]] = {
        "c_tissue": np.ndarray,
        "c_input": np.ndarray,
        "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "ktrans": np.ndarray,
        "ve": np.ndarray,
        "vp": np.ndarray,
        "kep": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("ktrans", "ve", "vp", "kep")
    units: ClassVar[dict[str, str]] = {
        "ktrans": "mL/100g/min",
        "ve": "(fraction)",
        "vp": "(fraction)",
        "kep": "1/min",
    }

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        c_t = np.asarray(inputs.c_tissue, dtype=float)
        c_a = np.asarray(inputs.c_input, dtype=float)
        t_s = np.asarray(inputs.t_s, dtype=float)

        if c_t.ndim == 1:
            fit = fit_voxel(c_a, c_t, t_s, **opts)
            return ModelResult(
                maps={
                    "ktrans": np.asarray(fit.ktrans_ml_100g),
                    "ve": np.asarray(fit.ve),
                    "vp": np.asarray(fit.vp),
                    "kep": np.asarray(fit.kep_per_min),
                },
                units=dict(self.units),
                aux={
                    "fitted_curve": fit.fitted_curve,
                    "residuals": fit.residuals,
                    "cost": float(fit.cost),
                    "success": bool(fit.success),
                },
            )

        if c_t.ndim != 2:
            raise ValueError(f"c_tissue must be 1-D or 2-D; got shape {c_t.shape}")

        n_t, n_v = c_t.shape
        ktrans = np.full(n_v, np.nan)
        ve = np.full(n_v, np.nan)
        vp = np.full(n_v, np.nan)
        kep = np.full(n_v, np.nan)

        if inputs.mask is None:
            voxels = range(n_v)
        else:
            voxels = np.flatnonzero(np.asarray(inputs.mask, dtype=bool))

        for v in voxels:
            f = fit_voxel(c_a, c_t[:, v], t_s, **opts)
            ktrans[v] = f.ktrans_ml_100g
            ve[v] = f.ve
            vp[v] = f.vp
            kep[v] = f.kep_per_min

        return ModelResult(
            maps={"ktrans": ktrans, "ve": ve, "vp": vp, "kep": kep},
            units=dict(self.units),
            aux={},
        )


PLUGIN = ExtendedToftsModel()
