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
