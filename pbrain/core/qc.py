"""Physiological-range QC for kinetic / diffusion parameter maps.

A pragmatic guard against bad-data subjects (failed AIF, mis-registration):
for each produced map, compare the brain-masked median against a plausible
normal range and emit a ``pass`` / ``warn`` flag with the offending value.
A failed AIF, for example, can drive Patlak Ki orders of magnitude above its
physiological ceiling; these checks catch that automatically and write the
verdict into the stage manifest's ``qc`` block, so downstream analysis can
exclude flagged subjects without manual inspection.

Ranges are deliberately wide (flag the clearly-implausible, not the merely
unusual) and overridable via ``--opt qc.ranges.<map>=lo,hi``.
"""

from __future__ import annotations

import numpy as np

# (lo, hi) plausibility bounds on the *brain-median* of each map.
# Sources: normal-volunteer DCE/DSC/diffusion literature; intentionally loose.
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    # kinetic
    "ki":   (-0.05, 0.50),     # mL/100g/min — BBB influx (small; >0.5 implausible)
    "vb":   (0.0, 12.0),       # mL/100g
    "vp":   (0.0, 0.20),       # fraction
    "cbf":  (3.0, 120.0),      # mL/100g/min
    "mtt":  (0.5, 25.0),       # s
    "cth":  (0.1, 25.0),       # s
    "ktrans": (-0.01, 0.30),
    # diffusion (brain-median)
    "fa":   (0.05, 0.60), "md":  (0.3e-3, 1.5e-3),
    "mk":   (0.3, 1.5), "kfa": (0.2, 1.0),
    "fw":   (0.0, 0.95), "tfa": (0.05, 0.7),
    "restricted_fraction": (0.0, 0.8),
}


def voxelwise_cnr(
    conc: np.ndarray,
    *,
    baseline_frames: int = 5,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-voxel contrast-to-noise ratio of a 4-D time series.

    For each voxel the CNR is the *peak above the pre-bolus baseline* divided by
    the *standard deviation of the pre-bolus baseline window* (R3.7):

        CNR = (max_t S(t) − mean(S[:baseline_frames]))
              / std(S[:baseline_frames])

    A voxel whose baseline noise is zero (perfectly flat) yields ``0.0`` rather
    than ``inf``, so the map stays finite and the low-CNR fraction treats it as
    bad. **Pass the raw DCE signal, not the baseline-subtracted concentration:**
    every signal→conc converter forces ``C(baseline)=0``, so the concentration's
    baseline SD is 0 at *every* voxel and the CNR would collapse to 0 everywhere.

    Parameters
    ----------
    conc
        4-D series ``(X, Y, Z, T)`` — the raw DCE signal (real baseline noise).
    baseline_frames
        Number of leading frames defining the pre-bolus baseline window.
        Defaults to 5 (the pipeline's ``baseline_frames``). Clamped to
        ``[1, T-1]``.
    mask
        Optional 3-D boolean brain mask. Voxels outside the mask are set to
        ``NaN`` in the returned map (so brain-fraction stats ignore them).

    Returns
    -------
    np.ndarray
        3-D CNR map ``(X, Y, Z)``; NaN outside the mask.
    """
    c = np.asarray(conc, dtype=float)
    if c.ndim != 4:
        raise ValueError(f"conc must be 4-D (X,Y,Z,T); got shape {c.shape}")
    T = int(c.shape[-1])
    nb = int(max(1, min(int(baseline_frames), T - 1)))

    base = c[..., :nb]
    base_mean = np.nanmean(base, axis=-1)
    base_std = np.nanstd(base, axis=-1)
    peak = np.nanmax(c, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cnr = (peak - base_mean) / base_std
    # Flat-baseline voxels (std == 0) and any non-finite result → 0 (bad CNR).
    cnr = np.where(np.isfinite(cnr), cnr, 0.0)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != cnr.shape:
            raise ValueError(
                f"mask shape {m.shape} ≠ CNR map shape {cnr.shape}"
            )
        cnr = np.where(m, cnr, np.nan)
    return cnr.astype(float)


def cnr_summary(
    cnr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold: float = 2.0,
) -> dict:
    """Summarise a voxel-wise CNR map for the QC manifest.

    Returns the brain-masked median CNR, the fraction of brain voxels below
    ``threshold`` (default CNR < 2), the voxel counts, and a ``pass``/``warn``
    flag. ``warn`` triggers when more than half the brain falls below the
    threshold (a coarse "this acquisition is too noisy to trust" gate).
    """
    c = np.asarray(cnr, dtype=float)
    vals = c[np.asarray(mask, dtype=bool)] if mask is not None else c.ravel()
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    thr = float(threshold)
    if n == 0:
        return {"status": "warn", "median": None, "threshold": thr,
                "fraction_below": None, "n_voxels": 0, "n_below": 0,
                "reason": "no finite voxels"}
    n_below = int(np.count_nonzero(vals < thr))
    frac = n_below / n
    return {
        "status": "warn" if frac > 0.5 else "pass",
        "median": float(np.median(vals)),
        "threshold": thr,
        "fraction_below": float(frac),
        "n_voxels": n,
        "n_below": n_below,
    }


def check_maps(maps: dict[str, np.ndarray], *,
               mask: np.ndarray | None = None,
               ranges: dict[str, tuple[float, float]] | None = None,
               ) -> dict:
    """Return a QC dict: overall status + per-map median/flag.

    ``maps`` is ``{name: 3-D array}``. Non-scalar (vector/RGB) maps are skipped.
    A map flags ``warn`` if its finite brain-median falls outside the range.
    """
    rng = {**DEFAULT_RANGES, **(ranges or {})}
    checks: dict[str, dict] = {}
    any_warn = False
    for name, arr in maps.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim != 3:
            continue
        vals = a[mask] if mask is not None else a
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            checks[name] = {"status": "warn", "median": None, "reason": "all-NaN"}
            any_warn = True
            continue
        med = float(np.median(vals))
        key = name.lower()
        if key in rng:
            lo, hi = rng[key]
            ok = lo <= med <= hi
            checks[name] = {"status": "pass" if ok else "warn",
                            "median": med, "range": [lo, hi]}
            any_warn = any_warn or not ok
        else:
            checks[name] = {"status": "pass", "median": med, "range": None}
    return {"status": "warn" if any_warn else "pass", "maps": checks}
