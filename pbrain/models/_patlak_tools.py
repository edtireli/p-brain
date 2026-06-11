"""Vectorised Patlak primitives: smart tail detection + OLS / Huber regression.

The whole Patlak fit factorises into:

1. **One AIF-level decision** — where does the tail window start?  The same
   indices apply to every voxel, so this is computed once.
2. **One broadcasted regression** — OLS or Huber across all voxels in a
   single vectorised step (no Python loop over voxels).

Tail-detection modes:

* ``x_max_fraction`` (default)  – The physically motivated choice. In Patlak
  coordinates, ``x = ∫Cₐ/Cₐ`` grows large only when Cₐ has decayed (small
  denominator) — i.e. AFTER the bolus has passed and Cₐ enters quasi-
  steady-state. Picking ``x ≥ fraction · x_max`` selects the late-tail
  regime where the Patlak linear assumption holds. Matches the legacy
  semantics in ``fit_patlak`` (default fraction = 1/3 = upper 2/3 of
  x_max), but computed vectorised across all voxels.

* ``peak_decay``  – Time-space heuristic. Smooth + ``find_peaks`` →
  last peak → walk forward until ``Cₐ(t) ≤ baseline + decay·(peak − baseline)``.
  Robust to varying bolus timing / count, but does NOT directly test
  Patlak's steady-state assumption. Useful when ``x_max_fraction`` fails
  (no clear washout, very short acquisition).

* ``changepoint`` – Bayesian-style changepoint on ``|d log Cₐ/dt|``.
  Latest frame where the derivative magnitude drops below a low-amplitude
  threshold. Good for irregular protocols.

* ``fixed``       – Fraction of total frames in *time* (not Patlak-x).
  For backward compatibility only.

Regression modes:

* ``ols``   – ordinary least squares (broadcasted closed form).
* ``huber`` – iteratively reweighted least squares with Huber loss
  (``δ = 1.345`` → 95 % efficiency under Gaussian noise). Downweights
  outliers (high-leakage voxels, partial volume, segmentation edges)
  without throwing them out.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

TailMode = Literal["curvature", "x_max_fraction", "peak_decay", "changepoint", "fixed", "smart"]
Regression = Literal["ols", "huber"]


# ────────────────────────────────────────────────────────────────────────
# Tail detection on the AIF
# ────────────────────────────────────────────────────────────────────────


def _smooth_butter(y: np.ndarray, fs: float = 15.0, cutoff: float = 4.0,
                   order: int = 3) -> np.ndarray:
    try:
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype="low", analog=False)
        return filtfilt(b, a, y.astype(float))
    except Exception:
        return y.astype(float)


def find_patlak_tail(
    c_input: np.ndarray,
    t_s: np.ndarray,
    *,
    mode: TailMode = "curvature",
    x_fraction: float = 1.0 / 3.0,
    baseline_frames: int = 5,
    decay_threshold: float = 0.5,
    min_tail_frac: float = 0.20,
    curvature_threshold_frac: float = 0.05,
    curvature_smooth_window: int = 5,
    curvature_buffer_frames: int = 3,
    peak_prominence_frac: float = 0.10,
    peak_separation: int = 10,
    fixed_fraction: float = 1.0 / 3.0,
    fs: float = 15.0,
    cutoff: float = 4.0,
    order: int = 3,
) -> tuple[int, int]:
    """Return ``(tail_start, tail_end)`` *frame indices* for the Patlak fit.

    Applies to the AIF only; the same window is used for every voxel
    so the regression downstream stays vectorised across voxels.

    Modes:
      - ``curvature``      – default. Finds the latest frame where the
                             smoothed |d²Cₐ/dt²| is non-negligible
                             (any bolus, hump, kink). Tail starts after
                             that + a small buffer. Catches *all* features
                             regardless of prominence — works with any
                             number of boluses and any recirculation
                             humps.
      - ``x_max_fraction`` – Patlak-x space window x ≥ fraction · x_max.
                             Legacy semantics.
      - ``peak_decay``     – find last peak via ``find_peaks`` (prominence-
                             thresholded), walk forward to N% decay.
                             Misses low-prominence humps.
      - ``changepoint``    – legacy log-derivative changepoint.
      - ``fixed``          – fraction of total frames (backward compat).
    """
    c_a = np.asarray(c_input, dtype=float).ravel()
    t_arr = np.asarray(t_s, dtype=float).ravel()
    n = int(c_a.size)
    if n < 10:
        return 0, n

    m = (mode or "curvature").strip().lower()

    if m == "curvature":
        return _tail_curvature(
            c_a,
            curvature_threshold_frac=curvature_threshold_frac,
            curvature_smooth_window=curvature_smooth_window,
            buffer_frames=curvature_buffer_frames,
            min_tail_frac=min_tail_frac,
            fs=fs, cutoff=cutoff, order=order,
        )

    if m == "x_max_fraction":
        return _tail_x_max_fraction(c_a, t_arr, fraction=x_fraction)

    if m == "fixed":
        return int(round(fixed_fraction * n)), n

    if m == "changepoint":
        return _tail_changepoint(c_a, fs=fs, cutoff=cutoff, order=order,
                                 min_tail_frac=min_tail_frac)

    if m in ("peak_decay", "smart"):
        return _tail_smart(
            c_a,
            baseline_frames=baseline_frames,
            decay_threshold=decay_threshold,
            min_tail_frac=min_tail_frac,
            peak_prominence_frac=peak_prominence_frac,
            peak_separation=peak_separation,
            fs=fs, cutoff=cutoff, order=order,
        )

    raise ValueError(f"Unknown tail mode: {mode!r}")


def _tail_curvature(
    c_a: np.ndarray,
    *,
    curvature_threshold_frac: float = 0.05,
    curvature_smooth_window: int = 5,
    buffer_frames: int = 3,
    min_tail_frac: float = 0.20,
    fs: float = 15.0,
    cutoff: float = 4.0,
    order: int = 3,
) -> tuple[int, int]:
    """Latest-significant-curvature tail detection.

    Approach:
      1. Smooth Cₐ with the standard Butterworth low-pass.
      2. Compute |d²Cₐ/dt²| (curvature magnitude).
      3. Apply a small rolling-mean smoother to the curvature itself
         (kills single-frame spikes).
      4. Find the LAST frame whose curvature exceeds
         ``curvature_threshold_frac · max_curvature``.
      5. Tail starts ``buffer_frames`` frames after that, clamped so
         at least ``min_tail_frac`` of the curve remains.

    Catches every bolus, recirculation hump, transit-time kink — anything
    that changes the AIF's shape. Once curvature flatlines, we're in the
    quasi-steady-state regime where Patlak's linear assumption holds.
    """
    n = int(c_a.size)
    y = _smooth_butter(c_a, fs=fs, cutoff=cutoff, order=order)
    d2 = np.abs(np.gradient(np.gradient(y)))
    if d2.size == 0:
        return 0, n
    w = max(1, int(curvature_smooth_window))
    if w >= 2 and d2.size > w:
        kernel = np.ones(w, dtype=float) / w
        d2 = np.convolve(d2, kernel, mode="same")
    d2_max = float(np.nanmax(d2))
    if not np.isfinite(d2_max) or d2_max <= 0:
        return n // 2, n
    threshold = float(curvature_threshold_frac) * d2_max
    active = d2 > threshold
    if not active.any():
        return n // 4, n
    last_active = int(np.flatnonzero(active)[-1])
    tail_start = last_active + max(0, int(buffer_frames))
    max_start = int(n - max(1, int(min_tail_frac * n)))
    return int(min(tail_start, max_start)), n


def _tail_x_max_fraction(c_a: np.ndarray, t_s: np.ndarray,
                         *, fraction: float = 1.0 / 3.0) -> tuple[int, int]:
    """Patlak's natural late-window: x ≥ fraction · x_max in Patlak-x space.

    Builds x = ∫Cₐ/Cₐ on the supplied time axis, finds the latest stretch
    of frames where x stays above ``fraction · x_max``, and returns the
    earliest such frame as the tail start. This is the legacy ``fit_patlak``
    window semantics. Matches the physics: Cₐ small ↔ x large ↔ steady-state
    where Patlak's linearity assumption holds.
    """
    n = int(c_a.size)
    if n < 4:
        return 0, n
    # Patlak x on the full series
    dt = np.diff(t_s)
    integ = np.concatenate(([0.0], np.cumsum(c_a[:-1] * dt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = integ / c_a
    finite = np.isfinite(x) & np.isfinite(c_a) & (c_a > 0)
    if not finite.any():
        return n // 2, n
    x_max = float(np.nanmax(x[finite]))
    if not np.isfinite(x_max) or x_max <= 0:
        return n // 2, n
    in_window = finite & (x >= float(fraction) * x_max) & (x <= x_max)
    if not in_window.any():
        return n // 2, n
    start = int(np.flatnonzero(in_window)[0])
    return start, n


def _tail_smart(c_a, *, baseline_frames, decay_threshold, min_tail_frac,
                peak_prominence_frac, peak_separation, fs, cutoff, order):
    n = c_a.size
    y = _smooth_butter(c_a, fs=fs, cutoff=cutoff, order=order)

    b = max(1, min(baseline_frames, n))
    baseline = float(np.mean(y[:b]))
    amplitude_max = float(np.max(y) - baseline)
    if not np.isfinite(amplitude_max) or amplitude_max <= 0:
        return int(round((1.0 - min_tail_frac) * 0)), n  # whole curve

    prom = float(peak_prominence_frac) * amplitude_max
    peaks, _ = find_peaks(y, prominence=prom, distance=int(peak_separation))
    if peaks.size == 0:
        # No detectable bolus — use the entire curve.
        return 0, n

    p_last = int(peaks[-1])
    peak_val = float(y[p_last])
    target = baseline + float(decay_threshold) * (peak_val - baseline)

    tail_start = p_last
    for i in range(p_last + 1, n):
        if y[i] <= target:
            tail_start = i
            break

    # Guard: keep at least min_tail_frac of the curve in the window.
    max_start = int(n - max(1, int(min_tail_frac * n)))
    return min(tail_start, max_start), n


def _tail_changepoint(c_a, *, fs, cutoff, order, min_tail_frac):
    """Latest frame where |d log Cₐ / dt| drops to a low-amplitude regime."""
    n = c_a.size
    y = _smooth_butter(c_a, fs=fs, cutoff=cutoff, order=order)
    y_safe = np.maximum(np.abs(y), 1e-6)
    log_y = np.log(y_safe)
    d = np.abs(np.gradient(log_y))
    # Smooth the derivative to suppress per-frame noise.
    d_s = _smooth_butter(d, fs=fs, cutoff=cutoff / 2.0, order=order)
    threshold = float(np.median(d_s) + 0.5 * np.std(d_s))
    # Walk backward to find the latest frame *above* threshold; tail starts after.
    above = np.where(d_s > threshold)[0]
    if above.size == 0:
        return int((1 - min_tail_frac) * n), n
    last_active = int(above[-1])
    tail_start = min(last_active + 1, int(n - max(1, int(min_tail_frac * n))))
    return tail_start, n


# ────────────────────────────────────────────────────────────────────────
# Vectorised regression
# ────────────────────────────────────────────────────────────────────────


def vectorised_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast OLS slope/intercept across columns of ``y``.

    ``x``: shape ``(T,)``  ``y``: shape ``(T,)`` or ``(T, V)``.
    Returns ``(slope, intercept)`` matching the column dimension.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float)
    one_d = y.ndim == 1
    if one_d:
        y = y.reshape(-1, 1)

    xm = float(x.mean())
    ym = y.mean(axis=0)
    dx = x - xm
    den = float((dx ** 2).sum())
    if den <= 0:
        nan = np.full(y.shape[1], np.nan)
        return (float(nan[0]), float(nan[0])) if one_d else (nan, nan)
    num = (dx[:, None] * (y - ym[None, :])).sum(axis=0)
    slope = num / den
    intercept = ym - slope * xm
    if one_d:
        return float(slope[0]), float(intercept[0])
    return slope, intercept


def vectorised_huber(
    x: np.ndarray,
    y: np.ndarray,
    *,
    delta: float = 1.345,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast Huber-M regression via IRLS.

    Initialised from OLS; iterates re-weighting with Huber weights
    ``w = min(1, δ / |u|)`` on MAD-standardised residuals. Stops when
    the max slope change falls below ``tol`` or after ``max_iter``.

    Default ``δ = 1.345`` matches Holland-Welsch (95 % efficiency under
    Gaussian noise) — the standard choice. Fully vectorised across
    voxels: one matrix–vector op per iteration, no Python loop over V.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float)
    one_d = y.ndim == 1
    if one_d:
        y = y.reshape(-1, 1)

    # Warm start with OLS
    slope, intercept = vectorised_ols(x, y)
    slope = np.atleast_1d(np.asarray(slope, dtype=float))
    intercept = np.atleast_1d(np.asarray(intercept, dtype=float))

    for _ in range(int(max_iter)):
        resid = y - (intercept[None, :] + slope[None, :] * x[:, None])

        # Robust scale via MAD
        med = np.median(resid, axis=0, keepdims=True)
        s = 1.4826 * np.median(np.abs(resid - med), axis=0)
        s = np.where(s > 1e-12, s, 1.0)
        u = resid / s[None, :]

        abs_u = np.abs(u)
        w = np.where(abs_u <= delta, 1.0, delta / np.maximum(abs_u, 1e-12))

        # Weighted least squares
        sw = w.sum(axis=0)
        sw_safe = np.where(sw > 1e-12, sw, 1.0)
        xm_w = (w * x[:, None]).sum(axis=0) / sw_safe
        ym_w = (w * y).sum(axis=0) / sw_safe

        dx_w = x[:, None] - xm_w[None, :]
        dy_w = y - ym_w[None, :]

        den = (w * dx_w * dx_w).sum(axis=0)
        den = np.where(den > 1e-12, den, 1.0)
        num = (w * dx_w * dy_w).sum(axis=0)

        slope_new = num / den
        intercept_new = ym_w - slope_new * xm_w

        if float(np.max(np.abs(slope_new - slope))) < tol:
            slope, intercept = slope_new, intercept_new
            break
        slope, intercept = slope_new, intercept_new

    if one_d:
        return float(slope[0]), float(intercept[0])
    return slope, intercept


# ────────────────────────────────────────────────────────────────────────
# Top-level voxelwise Patlak (single function for the kinetic stage)
# ────────────────────────────────────────────────────────────────────────


def fit_patlak_vectorised(
    c_tissue: np.ndarray,         # (T,) or (T, V)
    c_input: np.ndarray,          # (T,)
    t_s: np.ndarray,              # (T,)
    *,
    mask: np.ndarray | None = None,
    tail_mode: TailMode = "curvature",
    regression: Regression = "huber",
    huber_delta: float = 1.345,
    huber_max_iter: int = 20,
    x_fraction: float = 1.0 / 3.0,
    baseline_frames: int = 5,
    decay_threshold: float = 0.5,
    min_tail_frac: float = 0.20,
    curvature_threshold_frac: float = 0.05,
    curvature_smooth_window: int = 5,
    curvature_buffer_frames: int = 3,
    peak_prominence_frac: float = 0.10,
    peak_separation: int = 10,
) -> dict[str, np.ndarray]:
    """Vectorised Patlak across all voxels.

    Returns a dict with keys ``ki_ml_per_100g_min``, ``vb_ml_per_100g``,
    ``sd_ki_ml_per_100g_min``, ``tail_start``, ``tail_end``,
    ``x_patlak``, ``y_patlak``.

    All voxels share the same tail window (computed once from the AIF)
    and the same regression broadcast (one OLS / IRLS step total).
    """
    c_t = np.asarray(c_tissue, dtype=float)
    one_curve = c_t.ndim == 1
    if one_curve:
        c_t = c_t.reshape(-1, 1)

    c_a = np.asarray(c_input, dtype=float).ravel()
    t = np.asarray(t_s, dtype=float).ravel()

    n = int(min(c_t.shape[0], c_a.size, t.size))
    c_t = c_t[:n]
    c_a = c_a[:n]
    t = t[:n]
    V = int(c_t.shape[1])

    if n < 3:
        nan = np.full(V, np.nan)
        return dict(
            ki_ml_per_100g_min=nan, vb_ml_per_100g=nan,
            sd_ki_ml_per_100g_min=nan,
            tail_start=0, tail_end=n,
            x_patlak=np.zeros(n), y_patlak=np.zeros((n, V)),
        )

    # Patlak coordinates — same x for every voxel
    dt = np.diff(t)
    integ = np.concatenate(([0.0], np.cumsum(c_a[:-1] * dt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = integ / c_a
        y = c_t / c_a[:, None]

    finite_x = np.isfinite(x) & np.isfinite(c_a) & (c_a > 0)

    tail_start, tail_end = find_patlak_tail(
        c_a, t,
        mode=tail_mode,
        x_fraction=x_fraction,
        baseline_frames=baseline_frames,
        decay_threshold=decay_threshold,
        min_tail_frac=min_tail_frac,
        curvature_threshold_frac=curvature_threshold_frac,
        curvature_smooth_window=curvature_smooth_window,
        curvature_buffer_frames=curvature_buffer_frames,
        peak_prominence_frac=peak_prominence_frac,
        peak_separation=peak_separation,
    )

    in_window = np.zeros(n, dtype=bool)
    in_window[tail_start:tail_end] = True
    good_t = finite_x & in_window

    ki = np.full(V, np.nan, dtype=float)
    vb = np.full(V, np.nan, dtype=float)
    sd = np.full(V, np.nan, dtype=float)

    if int(good_t.sum()) < 2:
        return dict(
            ki_ml_per_100g_min=ki, vb_ml_per_100g=vb,
            sd_ki_ml_per_100g_min=sd,
            tail_start=int(tail_start), tail_end=int(tail_end),
            x_patlak=x, y_patlak=y,
        )

    xg = x[good_t]
    yg = y[good_t, :]

    # NaN voxels (any frame NaN in window) → skip via masked fit.
    finite_v = np.isfinite(yg).all(axis=0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size == V:
            finite_v = finite_v & mask

    fit_idx = np.flatnonzero(finite_v)
    if fit_idx.size == 0:
        return dict(
            ki_ml_per_100g_min=ki, vb_ml_per_100g=vb,
            sd_ki_ml_per_100g_min=sd,
            tail_start=int(tail_start), tail_end=int(tail_end),
            x_patlak=x, y_patlak=y,
        )

    yg_fit = yg[:, fit_idx]

    reg = (regression or "huber").strip().lower()
    if reg == "ols":
        slope, intercept = vectorised_ols(xg, yg_fit)
    elif reg in ("huber", "robust"):
        slope, intercept = vectorised_huber(
            xg, yg_fit, delta=huber_delta, max_iter=huber_max_iter,
        )
    else:
        raise ValueError(f"Unknown regression mode: {regression!r}")

    # SD of slope (OLS approximation, fine for QC even under Huber)
    y_pred = intercept[None, :] + slope[None, :] * xg[:, None]
    resid = yg_fit - y_pred
    dof = max(int(good_t.sum()) - 2, 1)
    sse = (resid * resid).sum(axis=0)
    den_x = float(((xg - xg.mean()) ** 2).sum())
    sd_slope = np.sqrt(np.maximum(sse / dof / max(den_x, 1e-12), 0.0))

    # Units: 1/s → mL/100g/min ×6000, fraction → mL/100g ×100
    ki[fit_idx] = slope * 6000.0
    vb[fit_idx] = intercept * 100.0
    sd[fit_idx] = sd_slope * 6000.0

    return dict(
        ki_ml_per_100g_min=ki,
        vb_ml_per_100g=vb,
        sd_ki_ml_per_100g_min=sd,
        tail_start=int(tail_start),
        tail_end=int(tail_end),
        x_patlak=x,
        y_patlak=y,
    )
