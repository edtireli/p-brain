from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODEL = {
    "key": "patlak",
    "env": "patlak",
    "description": "Patlak-only permeability (skip Tikhonov flow).",
    # Keep aliases intentionally minimal to avoid ambiguity.
    "aliases": {"patlak"},
}


@dataclass(frozen=True, slots=True)
class PatlakFit:
    """Result of a Patlak fit on normalised concentration curves.

    Inputs are expected to be *normalised* concentration-time curves.
    Units follow the legacy p-brain conventions:
    - `ki_ml_per_100g_min` in mL/100g/min
    - `vp_ml_per_100g` in mL/100g (Patlak intercept)
    """

    ki_ml_per_100g_min: float
    vp_ml_per_100g: float
    sd_ki_ml_per_100g_min: float
    x_patlak: np.ndarray
    y_patlak: np.ndarray
    good_mask: np.ndarray


def fit_patlak(
    c_tissue: np.ndarray,
    c_input: np.ndarray,
    t_s: np.ndarray,
    *,
    bad_mask: np.ndarray | None = None,
    window_start_fraction: float = 1 / 3,
    single_bolus: bool = False,
    aif_min_fraction: float = 0.05,
    single_bolus_start_s: float = 60.0,
) -> PatlakFit:
    """Fit the Patlak model on one tissue curve.

    Parameters
    ----------
    c_tissue, c_input, t_s
        1D arrays of equal length (or longer; shortest length is used).
        Both curves must be normalised (p-brain standard).
    bad_mask
        Optional boolean mask marking samples to exclude.
    window_start_fraction
        Fraction of ``x_max`` to start the linear regression window.
        Used in the **two-bolus** protocol (default).
    single_bolus
        When *True*, switch to a time-based windowing strategy suitable
        for single-injection protocols.  The fit starts at
        ``single_bolus_start_s`` seconds after the AIF peak and excludes
        time-points where the AIF falls below ``aif_min_fraction`` of
        its peak value.
    aif_min_fraction
        Minimum AIF amplitude (as a fraction of ``max(AIF)``) for a
        time-point to be included.  Only active when *single_bolus* is
        True.  Default 0.05 (5 %).
    single_bolus_start_s
        Seconds after AIF peak to begin the Patlak fit window when
        *single_bolus* is True.  Default 60 s (skip first-pass
        recirculation).

    Returns
    -------
    PatlakFit
        Contains Ki, intercept (vp), SD, and the Patlak x/y coordinates.
    """

    c_t = np.asarray(c_tissue, dtype=float).reshape(-1)
    c_a = np.asarray(c_input, dtype=float).reshape(-1)
    t = np.asarray(t_s, dtype=float).reshape(-1)

    n = int(min(c_t.size, c_a.size, t.size))
    c_t = c_t[:n]
    c_a = c_a[:n]
    t = t[:n]
    if bad_mask is None:
        bad = np.zeros(n, dtype=bool)
    else:
        bad = np.asarray(bad_mask, dtype=bool).reshape(-1)[:n]

    if n == 0:
        return PatlakFit(
            float("nan"),
            float("nan"),
            float("nan"),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=bool),
        )

    dt = np.diff(t)
    if dt.size == 0:
        return PatlakFit(
            float("nan"),
            float("nan"),
            float("nan"),
            np.zeros(n, dtype=float),
            np.zeros(n, dtype=float),
            np.zeros(n, dtype=bool),
        )

    # ── Single-bolus AIF threshold masking ──
    # When Ca(t) → 0 after the bolus washes out, the Patlak ratios
    # x = ∫Ca / Ca(t) and y = Ct / Ca(t) diverge, producing unreliable
    # fits.  Mask those time-points before computing the coordinates.
    if single_bolus:
        ca_peak = float(np.nanmax(c_a))
        if ca_peak > 0:
            bad = bad | (c_a < aif_min_fraction * ca_peak)
        # Also exclude pre-arrival and impose time-based start.
        peak_idx = int(np.argmax(c_a))
        t_start = t[peak_idx] + float(single_bolus_start_s)
        bad = bad | (t < t_start)

    # Patlak coordinates (legacy convention).
    # x[i] = integral_0^t Ca / Ca[i]
    # y[i] = Ct[i] / Ca[i]
    x = np.concatenate(([0.0], np.cumsum(c_a[:-1] * dt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = x / c_a
        y = c_t / c_a

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(c_a)
        & (c_a != 0)
        & (~bad)
    )

    if not good.any():
        return PatlakFit(float("nan"), float("nan"), float("nan"), x, y, good)

    x_max = float(np.nanmax(x[good]))

    if single_bolus:
        # Time-based windowing already applied via bad mask above;
        # no additional x-fraction clipping needed.
        window = np.ones(n, dtype=bool)
    else:
        w0 = float(window_start_fraction)
        if not (0.0 < w0 < 1.0):
            w0 = 1 / 3
        window = (x >= w0 * x_max) & (x <= x_max)

    good = good & window

    if int(good.sum()) < 2:
        return PatlakFit(float("nan"), float("nan"), float("nan"), x, y, good)

    xg = x[good]
    yg = y[good]
    xm = float(xg.mean())
    ym = float(yg.mean())
    denom = float(((xg - xm) ** 2).sum())
    if not np.isfinite(denom) or denom <= 0:
        return PatlakFit(float("nan"), float("nan"), float("nan"), x, y, good)

    ki_raw = float(((xg - xm) * (yg - ym)).sum() / denom)
    vp_raw = float(ym - ki_raw * xm)

    resid = yg - (vp_raw + ki_raw * xg)
    dof = int(good.sum()) - 2
    if dof <= 0:
        sd_raw = float("nan")
    else:
        sd_raw = float(np.sqrt(float((resid**2).sum()) / denom / dof))

    # Convert to legacy p-brain units.
    # ki_raw is 1/s -> mL/100g/min via *6000
    # vp_raw is fraction -> mL/100g via *100
    return PatlakFit(
        ki_raw * 6000.0,
        vp_raw * 100.0,
        sd_raw * 6000.0,
        x,
        y,
        good,
    )


def fit_patlak_tuple(
    c_tissue: np.ndarray,
    c_input: np.ndarray,
    t_s: np.ndarray,
    bad_mask: np.ndarray | None = None,
    *,
    window_start_fraction: float = 1 / 3,
    single_bolus: bool = False,
    aif_min_fraction: float = 0.05,
    single_bolus_start_s: float = 60.0,
):
    """Compatibility wrapper returning the historical tuple shape."""

    res = fit_patlak(
        c_tissue,
        c_input,
        t_s,
        bad_mask=bad_mask,
        window_start_fraction=window_start_fraction,
        single_bolus=single_bolus,
        aif_min_fraction=aif_min_fraction,
        single_bolus_start_s=single_bolus_start_s,
    )
    return (
        res.ki_ml_per_100g_min,
        res.vp_ml_per_100g,
        res.sd_ki_ml_per_100g_min,
        res.x_patlak,
        res.y_patlak,
        res.good_mask,
    )

