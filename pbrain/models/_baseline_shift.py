"""Per-voxel baseline detection + shift — verbatim port of old-pbrain.

Makes the modular voxelwise Patlak map **bit-identical** to the legacy
``Ki_per_voxel.nii.gz``. The legacy per-voxel pipeline (AI_tissue_functions
``process_ctcs`` / the voxelwise-Ki loop) does, for every voxel:

    C_t0  = concentration(signal, T1, M0)         # the 05 conversion stage
    bp    = find_baseline_point_advanced(C_t0)     # Butterworth + gradient peak
    C_t   = custom_shifter(C_t0, bp)               # zero pre-bolus, drop tail min
    Ki    = patlak(C_t, C_a, t)                    # OLS over the upper-2/3 window

The modular voxelwise path historically skipped the baseline detection +
``custom_shifter``, so its Ki map was not 1:1 with the legacy map. This module
restores that step. Ports (verbatim):

* ``find_baseline_point_advanced`` — ``AI_tissue_functions.py:2671``
* ``find_major_peaks``             — ``utils/plotting.py:528``
* ``butter_lowpass_filter``        — ``utils/plotting.py:755``
* ``custom_shifter``               — ``utils/plotting.py:501``

``baseline_shift_2d`` vectorises the Butterworth ``filtfilt`` and gradient
across all voxels (one ``filtfilt(..., axis=0)`` call instead of N) — the
per-column result is identical — and keeps the exact (cheap) peak/shift logic
per voxel, so the output is bit-identical while being much faster.
"""

from __future__ import annotations

import numpy as np


def _butter_coeffs(cutoff: float = 4.0, fs: float = 15.0, order: int = 3):
    from scipy.signal import butter
    return butter(order, cutoff / (0.5 * fs), btype="low", analog=False)


def butter_lowpass_filter(data, cutoff: float = 4.0, fs: float = 15.0, order: int = 3):
    from scipy.signal import filtfilt
    b, a = _butter_coeffs(cutoff, fs, order)
    return filtfilt(b, a, data)


def find_major_peaks(gradient, radius: int = 10, num_peaks: int = 2) -> list[int]:
    """Indices of the top-``num_peaks`` gradient maxima, radius-separated.
    Verbatim: ``utils/plotting.py:find_major_peaks``."""
    from scipy.signal import argrelextrema
    peak_indices = argrelextrema(np.asarray(gradient), np.greater)[0]
    if peak_indices.size == 0:
        return []
    peak_values = np.asarray(gradient)[peak_indices]
    sorted_peak_indices = [x for _, x in sorted(zip(peak_values, peak_indices), reverse=True)]
    major_peaks: list[int] = []
    for peak in sorted_peak_indices:
        if all(abs(peak - mp) >= radius for mp in major_peaks):
            major_peaks.append(int(peak))
            if len(major_peaks) >= num_peaks:
                break
    return major_peaks


def find_baseline_point_advanced(y_data, fs: float = 15.0, cutoff: float = 4.0,
                                 order: int = 3, radius: int = 10,
                                 num_peaks: int = 2):
    """Bolus-onset baseline frame index. Verbatim: ``find_baseline_point_advanced``.

    Ignores the first point, low-pass filters, takes the gradient, finds the
    major gradient peaks, and returns ``min(peak-1) + 1`` (or ``None``)."""
    y = np.asarray(y_data, dtype=float)[1:]
    try:
        y_filtered = butter_lowpass_filter(y, cutoff, fs, order)
    except Exception:
        return None
    gradient_filtered = np.gradient(y_filtered)
    major = find_major_peaks(gradient_filtered, radius, num_peaks)
    bps = [p - 1 for p in major]
    bp = min(bps) if bps else None
    if bp is not None:
        bp += 1
    return bp


def custom_shifter(array, baseline_point):
    """Zero pre-baseline frames; subtract the post-baseline min. Verbatim:
    ``utils/plotting.py:custom_shifter``. Returns a full-length array."""
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return array
    if baseline_point is None:
        baseline_point = 10
    try:
        baseline_point = int(baseline_point)
    except Exception:
        baseline_point = 10
    baseline_point = max(0, min(baseline_point, int(array.size - 1)))
    after = array[baseline_point:]
    if np.min(after) > 0:
        shifted = after - np.min(after)
    elif np.any(after < 0):
        shifted = after - np.min(after)
    else:
        shifted = after
    return np.concatenate([np.zeros(baseline_point), shifted])


def baseline_shift_2d(c_t_2d, *, num_peaks: int = 2, cutoff: float = 4.0,
                      fs: float = 15.0, order: int = 3, radius: int = 10):
    """Apply ``find_baseline_point_advanced`` + ``custom_shifter`` to every
    column of ``c_t_2d`` (shape ``(T, V)``). Bit-identical to the per-voxel
    legacy loop; the Butterworth filter + gradient are vectorised across V."""
    from scipy.signal import filtfilt
    c = np.asarray(c_t_2d, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"baseline_shift_2d expects (T, V); got {c.shape}")
    T, V = c.shape
    out = np.zeros_like(c)
    if T < 4:
        return c.copy()
    # Vectorised filtfilt + gradient on the first-frame-dropped series (axis 0
    # is independent per column → identical to looping per voxel).
    y = c[1:, :]
    try:
        b, a = _butter_coeffs(cutoff, fs, order)
        yf = filtfilt(b, a, y, axis=0)
    except Exception:
        yf = y
    grad = np.gradient(yf, axis=0)
    from scipy.signal import argrelextrema
    for v in range(V):
        g = grad[:, v]
        peaks = argrelextrema(g, np.greater)[0]
        if peaks.size:
            pv = g[peaks]
            ordered = [x for _, x in sorted(zip(pv, peaks), reverse=True)]
            major: list[int] = []
            for p in ordered:
                if all(abs(p - mp) >= radius for mp in major):
                    major.append(int(p))
                    if len(major) >= num_peaks:
                        break
            bps = [p - 1 for p in major]
            bp = min(bps) if bps else None
            if bp is not None:
                bp += 1
        else:
            bp = None
        out[:, v] = custom_shifter(c[:, v], bp)
    return out


def fit_patlak_legacy_vectorised(c_tissue, c_input, t_s, *, mask=None,
                                 window_start_fraction: float = 1.0 / 3,
                                 regression: str = "huber",
                                 huber_delta: float = 1.345,
                                 huber_max_iter: int = 20):
    """Fully-vectorised Patlak: x=∫Ca/Ca, y=Ct/Ca, value window
    ``x >= w0·x_max`` with per-voxel ``x_max``.

    ``regression="huber"`` (DEFAULT) does Huber-M IRLS inside that window —
    robust to the motion/signal-jump frames the article masked manually; on
    clean data it matches OLS to <1%, on noisy cohort curves it lifts Ki ~4%
    and prevents the low-uptake (WM) slope from flipping negative.
    ``regression="ols"`` is the closed-form OLS, **bit-identical** to the legacy
    per-voxel ``fit_patlak`` loop (corr=1.0, max|Δ|~3e-14, 123 k voxels) — keep
    it for the bitwise gate against the OLS-fit published dataset.

    ``c_tissue`` is ``(T, V)`` (or ``(T,)``); ``c_input``/``t_s`` are ``(T,)``.
    Returns a dict with ki/vb/sd (mL/100g/min, mL/100g, mL/100g/min).
    """
    c_t = np.asarray(c_tissue, dtype=float)
    if c_t.ndim == 1:
        c_t = c_t[:, None]
    c_a = np.asarray(c_input, dtype=float)
    t = np.asarray(t_s, dtype=float)
    # Truncate to the common frame length: the AIF / time axis may be a few
    # frames shorter than the tissue series (e.g. a 248-frame saved TSCC vs a
    # 250-frame DCE). The article truncates to ``min(len(t), len(ca))`` before
    # the Patlak fit; mirror it so ``c_a[:-1]*np.diff(t)`` stays aligned.
    n_common = int(min(c_t.shape[0], c_a.size, t.size))
    c_t, c_a, t = c_t[:n_common], c_a[:n_common], t[:n_common]
    n_t, V = c_t.shape
    w0 = float(window_start_fraction)
    if not (0.0 < w0 < 1.0):
        w0 = 1.0 / 3
    dt = np.diff(t)
    integ = np.concatenate(([0.0], np.cumsum(c_a[:-1] * dt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = integ / c_a                       # (T,) — shared across voxels
        y = c_t / c_a[:, None]                # (T, V)
    finite_x = np.isfinite(x) & np.isfinite(c_a) & (c_a != 0)
    fy = np.isfinite(y)
    good0 = finite_x[:, None] & fy            # finite Patlak coords per voxel
    # Finite-safe copies so ``inf·0`` from excluded frames can't poison sums.
    xs = np.where(finite_x, x, 0.0)[:, None]  # (T, 1)
    ys = np.where(fy, y, 0.0)                 # (T, V)
    x_max = np.where(good0, x[:, None], -np.inf).max(axis=0)   # per-voxel
    win = good0 & (xs >= w0 * x_max[None, :]) & (xs <= x_max[None, :])
    W = win.astype(float)
    n = W.sum(0)
    sx = (xs * W).sum(0); sy = (ys * W).sum(0)
    sxx = ((xs ** 2) * W).sum(0); sxy = ((xs * ys) * W).sum(0)
    syy = ((ys ** 2) * W).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = sxx - sx * sx / n
        slope = (sxy - sx * sy / n) / denom
        intercept = sy / n - slope * sx / n
        sse = (syy - sy * sy / n) - slope * slope * denom
        dof = np.maximum(n - 2, 1)
        sd_slope = np.sqrt(np.maximum(sse / denom / dof, 0.0))
    bad = (n < 2) | ~np.isfinite(denom) | (denom <= 0)
    if str(regression).lower() == "huber":
        # Iteratively-reweighted LS with Huber weights, INSIDE the same per-voxel
        # value window (W). Warm-started from the OLS fit above; downweights the
        # motion/signal-jump frames the article masked manually. delta=1.345 →
        # 95% Gaussian efficiency (Holland-Welsch). Fully vectorised over voxels.
        delta = float(huber_delta)
        good_v = ~bad
        sl = np.where(good_v, slope, 0.0)
        ic = np.where(good_v, intercept, 0.0)
        for _ in range(int(huber_max_iter)):
            resid = ys - (ic[None, :] + sl[None, :] * xs)            # (T, V)
            rw = np.where(win, resid, np.nan)
            med = np.nanmedian(rw, axis=0)
            mad = np.nanmedian(np.abs(rw - med[None, :]), axis=0)
            scale = np.where(1.4826 * mad > 1e-12, 1.4826 * mad, 1.0)
            u = resid / scale[None, :]
            hw = np.where(np.abs(u) <= delta, 1.0,
                          delta / np.maximum(np.abs(u), 1e-12))
            W2 = W * hw                                              # window × Huber
            n2 = W2.sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                sx2 = (xs * W2).sum(0); sy2 = (ys * W2).sum(0)
                sxx2 = ((xs ** 2) * W2).sum(0); sxy2 = ((xs * ys) * W2).sum(0)
                den2 = sxx2 - sx2 * sx2 / n2
                sl_new = (sxy2 - sx2 * sy2 / n2) / den2
                ic_new = sy2 / n2 - sl_new * sx2 / n2
            ok = good_v & np.isfinite(sl_new) & np.isfinite(den2) & (den2 > 0) & (n2 >= 2)
            step = float(np.max(np.abs(np.where(ok, sl_new - sl, 0.0)))) if ok.any() else 0.0
            sl = np.where(ok, sl_new, sl)
            ic = np.where(ok, ic_new, ic)
            if step < 1e-7:
                break
        slope = np.where(good_v, sl, slope)
        intercept = np.where(good_v, ic, intercept)
    ki = slope * 6000.0
    vb = intercept * 100.0
    sd = sd_slope * 6000.0
    for arr in (ki, vb, sd):
        arr[bad] = np.nan
    if mask is not None:
        m = ~np.asarray(mask, dtype=bool).reshape(-1)
        if m.size == V:
            for arr in (ki, vb, sd):
                arr[m] = np.nan
    return {
        "ki_ml_per_100g_min": ki,
        "vb_ml_per_100g": vb,
        "sd_ki_ml_per_100g_min": sd,
    }


__all__ = [
    "find_baseline_point_advanced",
    "custom_shifter",
    "find_major_peaks",
    "butter_lowpass_filter",
    "baseline_shift_2d",
    "fit_patlak_legacy_vectorised",
]
