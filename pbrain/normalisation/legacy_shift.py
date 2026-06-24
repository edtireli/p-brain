"""Legacy p-Brain baseline shifter — paper §4.5.1 default in practice.

Ports the legacy preprocessing chain applied to voxelwise tissue curves
and the AIF *before* Patlak fitting:

1. **Turboflash post-baseline subtract** (``utils/plotting.py:turboflash``
   lines 327-336): subtract the mean of ``c[2:baseline_frames]`` (default
   frames 3-9, 0-indexed 2-9). This removes the constant offset left over
   from the signal→concentration conversion.
2. **Find baseline point** via ``find_baseline_intersection``
   (``modules/AI_tissue_functions.py:2914``): low-pass filter the curve
   (Butterworth, fs=15 Hz, cutoff=4 Hz, order=3), find the largest
   gradient peak (bolus onset), then walk forward to find the last
   "flat" frame whose deviation from the pre-bolus mean stays below
   a tolerance derived from the baseline noise.
3. **Custom shift** (``utils/plotting.py:custom_shifter``): drop frames
   ``[0:bp]``, subtract ``min(curve[bp:])`` from the rest, prepend
   ``bp`` zeros.

Result: pre-bolus is exactly zero (the dropped/padded region), bolus
onset is anchored at zero, and the curve is in mM (no p95 rescale —
Patlak is scale-invariant in Ki when both Ct and Ca share a scale).

To use as the default normaliser::

    --normaliser legacy_shift

Override the per-curve filter / search via opts::

    --opt normalisation.legacy_shift.fs=15
    --opt normalisation.legacy_shift.cutoff=4.0
    --opt normalisation.legacy_shift.peak_radius=10
    --opt normalisation.legacy_shift.turboflash_baseline_frames=10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.signal import argrelextrema, butter, filtfilt

from .base import CurveNormaliser


def _butter_lowpass_filter(y: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, y)


def _find_major_peaks(gradient: np.ndarray, *, radius: int, num_peaks: int) -> list[int]:
    """Top-``num_peaks`` local maxima in ``gradient``, each ``radius``-apart."""
    if gradient.size < 3:
        return []
    peak_indices = argrelextrema(gradient, np.greater)[0]
    if peak_indices.size == 0:
        return []
    peak_values = gradient[peak_indices]
    order = np.argsort(peak_values)[::-1]
    sorted_peaks = [int(peak_indices[i]) for i in order]

    out: list[int] = []
    for p in sorted_peaks:
        if all(abs(p - q) >= int(radius) for q in out):
            out.append(p)
            if len(out) >= int(num_peaks):
                break
    return out


def _find_baseline_point(
    y_data: np.ndarray,
    *,
    fs: float = 15.0,
    cutoff: float = 4.0,
    order: int = 3,
    peak_radius: int = 10,
    num_peaks: int = 2,
    y_filtered: np.ndarray | None = None,
) -> int:
    """Forward-walk baseline-departure detection (legacy semantics).

    Returns the index of the last flat frame before bolus onset.
    Falls back to ``10`` on any numerical failure. ``y_filtered`` lets a caller
    pass the low-pass-filtered curve in (e.g. from a single batched ``filtfilt``
    over all voxels), skipping the per-voxel filter — bit-identical result.
    """
    if y_filtered is not None:
        y_f = np.asarray(y_filtered, dtype=float)
        if not np.all(np.isfinite(y_f)):     # matches the per-voxel filter's except→10
            return 10
    else:
        try:
            y_f = _butter_lowpass_filter(y_data.astype(float), cutoff, fs, order)
        except Exception:
            return 10

    n = int(y_f.size)
    if n < 12:
        return min(8, n - 1)

    bl_mean = float(np.mean(y_f[3:9]))
    bl_std = float(np.std(y_f[3:9]))
    gradient = np.gradient(y_f)

    major_peaks = _find_major_peaks(gradient, radius=peak_radius, num_peaks=num_peaks)
    grad_peak = min(major_peaks) if major_peaks else n - 1
    end = min(grad_peak + 10, n)
    bolus_peak = float(np.max(y_f[grad_peak:end])) if end > grad_peak else float(y_f[-1])

    signal_range = abs(bolus_peak - bl_mean)
    threshold = max(bl_std * 4.0, signal_range * 0.05, 0.0005)

    bp = 8
    for f in range(8, min(grad_peak + 2, n - 2)):
        d0 = abs(y_f[f] - bl_mean)
        d1 = abs(y_f[f + 1] - bl_mean)
        d2 = abs(y_f[f + 2] - bl_mean)
        if d0 > threshold and d1 > threshold and d2 > threshold:
            bp = max(f - 1, 3)
            break
    return int(bp)


def _custom_shift(curve: np.ndarray, baseline_point: int) -> np.ndarray:
    """Drop the first ``baseline_point`` frames, subtract min(remainder),
    pad zeros back to original length. Matches legacy custom_shifter."""
    arr = np.asarray(curve, dtype=float)
    n = int(arr.size)
    if n == 0:
        return arr
    bp = int(np.clip(baseline_point if baseline_point is not None else 10, 0, n - 1))
    tail = arr[bp:]
    if tail.size == 0:
        return np.zeros(n, dtype=float)
    if np.min(tail) > 0 or np.any(tail < 0):
        shifted_tail = tail - np.min(tail)
    else:
        shifted_tail = tail
    out = np.empty(n, dtype=float)
    out[:bp] = 0.0
    out[bp:] = shifted_tail
    return out


def _turboflash_baseline_subtract(curve: np.ndarray, baseline_frames: int) -> np.ndarray:
    """Legacy turboflash post-baseline correction (utils/plotting.py:327-350).

    Two steps:

    1. Subtract ``mean(curve[2:baseline_frames])`` (default frames 3-9,
       0-indexed). Frames 0 and 1 are skipped because they can be
       transients at the start of the inversion-recovery preparation.
    2. **Zero the first ``baseline_frames`` frames** (line 348). This
       removes pre-bolus noise and gives ``find_baseline_intersection``
       a clean baseline window to scan, so the detected ``bp`` is
       deterministic.
    """
    arr = np.asarray(curve, dtype=float).copy()
    n = int(arr.size)
    if n < 3:
        return arr
    bf = int(min(max(baseline_frames, 3), n))
    if bf <= 2:
        return arr
    baseline = float(np.mean(arr[2:bf]))
    if not np.isfinite(baseline):
        baseline = 0.0
    arr = arr - baseline
    arr[:bf] = 0.0
    return arr


@dataclass(frozen=True, slots=True)
class _LegacyShift:
    key: ClassVar[str] = "legacy_shift"
    name: ClassVar[str] = "Legacy custom_shifter (per-voxel baseline-point + min-subtract)"
    description: ClassVar[str] = (
        "Reproduces the legacy curve preprocessing: per-curve baseline-"
        "point detection via low-pass-filtered forward walk, then "
        "drop-baseline + min-subtract + zero-pad. This is the "
        "normalisation actually applied before Patlak in the original "
        "codebase, yielding identical default behaviour."
    )
    accepts: ClassVar[dict[str, type]] = {
        "curve": np.ndarray,
        "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {"curve_normalised": np.ndarray}

    def normalise(
        self,
        curve: np.ndarray,
        t_s: np.ndarray,
        *,
        baseline_frames: int = 5,           # unused (legacy detects per-voxel)
        turboflash_baseline_frames: int = 10,
        fs: float = 15.0,
        cutoff: float = 4.0,
        order: int = 3,
        peak_radius: int = 10,
        num_peaks: int = 2,
        reference_curve: np.ndarray | None = None,   # unused — kept for contract
        **_: Any,
    ) -> np.ndarray:
        c = np.asarray(curve, dtype=float)
        if c.ndim == 1:
            c1 = _turboflash_baseline_subtract(c, turboflash_baseline_frames)
            bp = _find_baseline_point(
                c1, fs=fs, cutoff=cutoff, order=order,
                peak_radius=peak_radius, num_peaks=num_peaks,
            )
            return _custom_shift(c1, bp)

        if c.ndim == 2:
            # (T, V) — same per-voxel algorithm, but the expensive Butterworth
            # filtfilt (the bottleneck) is batched over all columns in one call;
            # only the cheap peak-walk stays per-voxel. Bit-identical to the loop.
            T, V = c.shape
            cf = c.astype(float)
            out = np.zeros_like(cf)
            valid = np.any(np.isfinite(cf), axis=0) & ~np.all(cf == 0, axis=0)

            # vectorised turboflash baseline subtract (== _turboflash_baseline_subtract)
            c1 = cf.copy()
            bf = int(min(max(turboflash_baseline_frames, 3), T))
            if bf > 2:
                base = np.mean(c1[2:bf], axis=0)
                base = np.where(np.isfinite(base), base, 0.0)
                c1 = c1 - base[None, :]
                c1[:bf] = 0.0

            # single batched low-pass filter over all voxels (NaN columns → NaN,
            # which _find_baseline_point maps to the legacy bp=10 fallback).
            yf = None
            try:
                nyq = 0.5 * fs
                b, a = butter(order, cutoff / nyq, btype="low", analog=False)
                yf = filtfilt(b, a, c1, axis=0)
            except Exception:
                yf = None

            for v in np.flatnonzero(valid):
                try:
                    bp = _find_baseline_point(
                        c1[:, v], fs=fs, cutoff=cutoff, order=order,
                        peak_radius=peak_radius, num_peaks=num_peaks,
                        y_filtered=(yf[:, v] if yf is not None else None),
                    )
                except Exception:
                    bp = 10
                out[:, v] = _custom_shift(c1[:, v], int(bp))
            return out

        raise ValueError(f"curve must be 1-D or 2-D; got shape {c.shape}")


PLUGIN = _LegacyShift()
