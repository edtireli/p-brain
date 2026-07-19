"""Modular pre-contrast baseline detection for signal→concentration.

Kinetic models assume C(baseline)=0, so every converter needs the length
of the quiet pre-bolus window. The *method* is pluggable and selected with
the ``baseline_method`` converter option:

* ``"auto"``       — p-Brain's robust bolus-onset finder (**default**). Mean
                     curve over the higher-signal voxels, locate the bolus
                     onset (half-max of the rise), return the minimum before
                     it. Robust to variable bolus-arrival time.
* ``"derivative"`` — the supervisor's ``detect_baseline`` (U. Lindberg):
                     differentiate, find the steepest-rise frame, take the
                     last non-positive slope before it.
* ``"fixed"``      — a fixed ``fallback`` window, no detection.

``resolve_baseline_frames`` returns the integer frame count; converters then
derive the baseline scale and subtract it. The auto finder is kept as the
validated p-Brain default — it is more robust than the derivative method on
noisy / slowly-rising curves.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _mean_high_signal_curve(S: np.ndarray, *, skip: int) -> np.ndarray | None:
    """Mean time-course over the upper-quartile-signal voxels (vessel/tissue)."""
    if S.ndim == 1:
        return np.asarray(S, dtype=float)
    if S.ndim < 2 or S.shape[-1] < 6:
        return None
    flat = S.reshape(-1, S.shape[-1])
    b0 = np.nanmean(flat[:, : max(2, skip)], axis=1)
    ok = np.isfinite(b0)
    if not ok.any():
        return None
    sel = flat[ok & (b0 >= np.nanpercentile(b0[ok], 75))]
    return np.nanmean(sel if sel.size else flat, axis=0)


def detect_baseline_frames(signal: np.ndarray, *, fallback: int = 10,
                           skip: int = 3, height_frac: float = 0.5) -> int:
    """p-Brain's robust pre-bolus finder (``"auto"`` method).

    Ports the legacy ``find_shifted_baseline``: take the mean curve over the
    higher-signal voxels, find the bolus onset (first frame reaching
    ``height_frac`` of the rise to peak), then return the index of the
    **minimum before that onset** (skipping the first ``skip`` settling
    frames). Falls back to ``fallback`` if undetectable.
    """
    S = np.asarray(signal, dtype=float)
    curve = _mean_high_signal_curve(S, skip=skip)
    if curve is None:
        return max(3, int(fallback))
    curve = np.asarray(curve, dtype=float)
    if curve.size < 6 or not np.isfinite(curve).any():
        return max(3, int(fallback))
    mx = float(np.nanmax(curve))
    base0 = float(np.nanmean(curve[: max(2, skip)]))
    if not np.isfinite(mx) or mx <= base0:
        return max(3, int(fallback))
    above = np.flatnonzero(curve >= base0 + height_frac * (mx - base0))
    if above.size == 0:
        return max(3, int(fallback))
    first_peak = int(above[0])
    if first_peak <= skip:
        return max(3, first_peak)
    return max(3, int(np.nanargmin(curve[skip:first_peak]) + skip))


def detect_baseline_derivative(signal: np.ndarray, *, fallback: int = 10,
                               skip: int = 3, **_: object) -> int:
    """Supervisor's ``detect_baseline`` (U. Lindberg, 2016) — derivative method.

    ``yp = diff(curve); idx = argmax(yp); nbp = last frame ≤ idx with yp ≤ 0``.
    The steepest rise marks the bolus; the last flat/declining frame before it
    is the baseline end.
    """
    S = np.asarray(signal, dtype=float)
    curve = _mean_high_signal_curve(S, skip=skip)
    if curve is None or curve.size < 4 or not np.isfinite(curve).any():
        return max(3, int(fallback))
    yp = np.diff(np.nan_to_num(curve, nan=0.0))
    idx = int(np.argmax(yp))
    if idx <= 0:
        return max(3, int(fallback))
    nonpos = np.flatnonzero(yp[:idx] <= 0)
    if nonpos.size == 0:
        return max(3, int(fallback))
    return max(3, int(nonpos[-1]) + 1)


def detect_baseline_gradient(signal: np.ndarray, *, fallback: int = 10,
                             skip: int = 3, **_: object) -> int:
    """Butterworth low-pass + gradient walk-back — the legacy p-Brain
    ``find_baseline_point_advanced`` on the mean high-signal curve.

    Low-pass filters the curve (Butterworth), differentiates it, finds the major
    gradient peak (the bolus rise), and walks back to the frame **just before** that
    rise — i.e. the *last* pre-contrast baseline frame before Gd arrives (which the
    minimum-before-onset ``auto`` method can undershoot on noisy baselines). Falls
    back to the derivative method, then ``fallback``, if the filter can't run.
    """
    S = np.asarray(signal, dtype=float)
    curve = _mean_high_signal_curve(S, skip=skip)
    if curve is None or curve.size < 6 or not np.isfinite(curve).any():
        return max(3, int(fallback))
    try:
        from pbrain.models._baseline_shift import find_baseline_point_advanced
        bp = find_baseline_point_advanced(np.nan_to_num(curve, nan=0.0))
    except Exception:
        bp = None
    if bp is None or not np.isfinite(bp):
        return detect_baseline_derivative(S, fallback=fallback, skip=skip)
    return max(3, int(bp))


#: Pluggable baseline detectors. Add an entry to extend.
BASELINE_METHODS: dict[str, Callable[..., int]] = {
    "auto": detect_baseline_frames,
    "derivative": detect_baseline_derivative,
    "gradient": detect_baseline_gradient,
    "advanced": detect_baseline_gradient,          # alias — the interactive default
    "fixed": lambda signal, *, fallback=10, **_: max(3, int(fallback)),
}


def resolve_baseline_frames(signal: np.ndarray, *, method: str = "auto",
                            fallback: int = 10, **kw: object) -> int:
    """Detect the pre-contrast baseline length using the named method."""
    fn = BASELINE_METHODS.get(str(method).strip().lower())
    if fn is None:
        raise ValueError(
            f"unknown baseline_method {method!r}; "
            f"choose from {sorted(BASELINE_METHODS)}"
        )
    return fn(signal, fallback=fallback, **kw)
