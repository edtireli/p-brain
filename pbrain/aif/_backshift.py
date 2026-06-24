"""SSS→rICA back-shift: align venous bolus peaks to the arterial peaks, PCHIP shift.

The venous (sagittal-sinus) curve is shifted onto the arterial (rICA) bolus
timing. Peaks are found in each curve by iterative maximum-finding with radius
exclusion — take the global maximum, blank out ``±peak_radius`` samples around
it, repeat ``num_peaks`` times (so a two-bolus protocol yields two peaks;
``num_peaks`` is configurable). The venous peaks are paired to the arterial
peaks in time order, refined to sub-sample precision by parabolic interpolation,
and the venous curve is advanced by the median peak offset with PCHIP
interpolation onto the shifted time axis — a continuous, sub-frame shift.

``shift_curve_pchip`` reproduces the reference MATLAB ``inputshift3``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def shift_curve_pchip(time_s: np.ndarray, curve: np.ndarray, shift_seconds: float) -> np.ndarray:
    """Shift ``curve`` by ``shift_seconds`` via PCHIP interpolation.

    Evaluates the samples at ``time - shift_seconds``: a positive shift delays the
    curve (pad leading zeros), a negative shift advances it (extend the tail with
    the last value). Continuous in ``shift_seconds`` — no whole-frame rounding.
    """
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    y = np.asarray(curve, dtype=float).reshape(-1)
    if time_s.size != y.size:
        n = min(time_s.size, y.size)
        time_s, y = time_s[:n], y[:n]
    if time_s.size <= 1 or float(shift_seconds) == 0.0:
        return y.copy()

    if time_s.size < 3:
        time_unit = float(time_s[1] - time_s[0])
    else:
        time_unit = float(time_s[2] - time_s[1])
    if not np.isfinite(time_unit) or time_unit <= 0:
        deltas = np.diff(time_s)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
        time_unit = float(deltas[0]) if deltas.size else 1.0

    deltaT = float(shift_seconds)
    notimeunit = int(np.ceil(abs(deltaT) / time_unit))
    n = int(y.size)

    if deltaT > 0:                                   # delay: pad leading zeros
        timetemp = (time_s[0] + deltaT) - (notimeunit * time_unit) \
            + np.arange(n + notimeunit, dtype=float) * time_unit
        sCa = np.zeros(n + notimeunit, dtype=float)
        sCa[notimeunit:notimeunit + n] = y[:n]
    else:                                            # advance: extend tail
        timetemp = (time_s[0] + deltaT) + np.arange(n + notimeunit, dtype=float) * time_unit
        sCa = np.zeros(n + notimeunit, dtype=float)
        sCa[:n] = y[:n]
        if notimeunit:
            sCa[n:] = y[n - 1]

    out = PchipInterpolator(timetemp, sCa, extrapolate=False)(time_s)
    return np.where(np.isfinite(out), out, 0.0).astype(float)


def find_top_peaks(curve: np.ndarray, num_peaks: int = 2, peak_radius: int = 10) -> list[int]:
    """Top-``num_peaks`` peaks by height, found iteratively.

    Take the global maximum, blank out ``±peak_radius`` samples around it, repeat.
    Returns the peak indices in time order. ``num_peaks`` controls how many boluses
    to locate (default 2)."""
    c = np.asarray(curve, dtype=float).copy()
    n = c.size
    peaks: list[int] = []
    for _ in range(max(1, int(num_peaks))):
        if not np.isfinite(c).any():
            break
        i = int(np.nanargmax(c))
        if not np.isfinite(c[i]) or c[i] <= 0:
            break
        peaks.append(i)
        lo, hi = max(0, i - int(peak_radius)), min(n, i + int(peak_radius) + 1)
        c[lo:hi] = -np.inf
    return sorted(peaks)


def _peak_time(curve: np.ndarray, t: np.ndarray, i: int) -> float:
    """Sub-sample peak time by parabolic interpolation around index ``i``."""
    n = len(curve)
    if i <= 0 or i >= n - 1:
        return float(t[i])
    y0, y1, y2 = float(curve[i - 1]), float(curve[i]), float(curve[i + 1])
    denom = y0 - 2.0 * y1 + y2
    frac = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
    frac = float(np.clip(frac, -1.0, 1.0))
    step = float(t[i + 1] - t[i]) if frac >= 0 else float(t[i] - t[i - 1])
    return float(t[i]) + frac * step


def backshift_sss_to_rica(
    vein_curve: np.ndarray, artery_curve: np.ndarray, time_s: np.ndarray,
    *, num_peaks: int = 2, peak_radius: int = 10,
    rescale: bool = False, max_shift_s: float = 20.0,
) -> tuple[np.ndarray, float, float]:
    """Shift the SSS (venous) curve so its bolus peaks land on the rICA peaks.

    Finds the top ``num_peaks`` peaks in each curve (iterative max + ``peak_radius``
    exclusion), pairs venous→arterial peaks in time order, and advances the venous
    curve by the median sub-sample peak offset via PCHIP, clamped to ``±max_shift_s``.
    Optional ``rescale`` lifts the venous amplitude to the arterial peak (never
    down). Returns ``(shifted_vein, shift_seconds, scale)``.
    """
    vein = np.asarray(vein_curve, dtype=float).reshape(-1)
    artery = np.asarray(artery_curve, dtype=float).reshape(-1)
    t = np.asarray(time_s, dtype=float).reshape(-1)
    n = int(min(vein.size, artery.size, t.size))
    if n == 0:
        return vein.astype(np.float32), 0.0, 1.0
    vein, artery, t = vein[:n], artery[:n], t[:n]

    bf = min(8, n)
    a0 = artery - float(np.nanmedian(artery[:bf]))   # baseline-zero for peak finding
    v0 = vein - float(np.nanmedian(vein[:bf]))
    # Find bolus peaks on 3-pt-smoothed curves so a single-frame near-saturation
    # spike in a hot arterial voxel is not mistaken for the first-pass bolus peak
    # (matches the old-pbrain ranker's moving-average peak detection).
    def _sm3(c: np.ndarray) -> np.ndarray:
        return np.convolve(c, np.ones(3) / 3.0, mode="same") if c.size >= 3 else c.copy()
    a_sm, v_sm = _sm3(a0), _sm3(v0)
    rica_pk = find_top_peaks(a_sm, num_peaks, peak_radius)
    sss_pk = find_top_peaks(v_sm, num_peaks, peak_radius)
    k = min(len(rica_pk), len(sss_pk))
    if k == 0:
        return vein.astype(np.float32), 0.0, 1.0

    # vein peaks land later than the arterial targets; advance by the median offset
    offsets = [_peak_time(v_sm, t, sss_pk[i]) - _peak_time(a_sm, t, rica_pk[i]) for i in range(k)]
    shift_s = float(np.clip(-float(np.median(offsets)), -float(max_shift_s), float(max_shift_s)))
    shifted = shift_curve_pchip(t, vein, shift_s)

    scale = 1.0
    if rescale:
        # Old-pbrain ``align_first_peaks`` (time_shifting.py): rescale by the
        # MEDIAN of matched bolus-peak-height ratios (artery/vein), with a
        # down-scale veto — NOT the ratio of global maxima, which a single hot
        # voxel's spike inflates (the spurious ~12 mM AIF).
        ratios = []
        for i in range(k):
            v = float(v_sm[sss_pk[i]])
            a = float(a_sm[rica_pk[i]])
            if np.isfinite(v) and np.isfinite(a) and abs(v) > 1e-8:
                ratios.append(a / v)
        if ratios:
            f = float(np.median(ratios))
            if np.isfinite(f) and f >= 1.0:
                shifted, scale = shifted * f, f
    return shifted.astype(np.float32), float(shift_s), float(scale)
