"""Utilities for identifying and masking signal jumps in tissue CTCs."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import os


__all__ = [
    "should_apply_jumpfix",
    "identify_drop_points",
    "apply_jumpfix",
]


def should_apply_jumpfix(analysis_directory: str | None) -> bool:
    """Return ``True`` when jump-fix correction should be enabled.

    A dataset opts-in to the correction by placing an ``apply_jumpfix.json``
    file next to the per-subject analysis directory.  The automatic pipeline
    already performs this check; exposing it here allows the slice-by-slice
    implementation to follow the same convention.
    """
    if not analysis_directory:
        return False
    jumpfix_path = os.path.join(os.path.dirname(analysis_directory), "apply_jumpfix.json")
    return os.path.exists(jumpfix_path)


def identify_drop_points(
    signal: np.ndarray,
    tail_start: int = 100,
    threshold_factor: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray | None, float | None]:
    """Detect abrupt drops in a concentration-time curve.

    Parameters
    ----------
    signal:
        1-D concentration-time curve.
    tail_start:
        Index marking the beginning of the tail region to analyse.
    threshold_factor:
        Multiplier applied to the residual standard deviation that determines
        the drop threshold.
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size

    if n <= tail_start + 1:
        return np.array([], dtype=int), None, None
    if np.all(np.isnan(signal)):
        return np.array([], dtype=int), None, None

    x = np.arange(tail_start, n)
    y = signal[tail_start:]
    good_tail = ~np.isnan(y)
    if good_tail.sum() < 2:
        return np.array([], dtype=int), None, None

    m, b = np.polyfit(x[good_tail], y[good_tail], 1)
    trend = m * np.arange(n) + b

    resid = signal - trend
    mu = np.nanmean(resid)
    sigma = np.nanstd(resid)
    thresh = mu - threshold_factor * sigma

    drop_mask = (resid < thresh) & (np.arange(n) >= tail_start)
    drop_idxs = np.where(drop_mask)[0]
    return drop_idxs.astype(int), trend, thresh


def apply_jumpfix(
    ctc: np.ndarray,
    *,
    tail_start: int = 100,
    thresh_factor: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask post-tail drops in *ctc* using the global jump-fix heuristic.

    Returns the masked curve, the boolean mask indicating dropped samples and
    the indices of the dropped points.  The behaviour mirrors the automatic
    pipeline so both global and slice-by-slice analyses treat signal jumps
    identically.
    """
    ctc = np.asarray(ctc, dtype=float)

    if ctc.size <= tail_start + 1:
        return (
            ctc.astype(float),
            np.zeros_like(ctc, dtype=bool),
            np.array([], dtype=int),
        )

    drop_idxs, *_ = identify_drop_points(ctc, tail_start, thresh_factor)
    bad_mask = np.zeros_like(ctc, dtype=bool)
    if drop_idxs.size:
        bad_mask[drop_idxs] = True

    masked = ctc.copy()
    masked[bad_mask] = np.nan
    return masked, bad_mask, drop_idxs
