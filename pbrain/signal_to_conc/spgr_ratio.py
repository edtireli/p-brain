"""Ratio-based spoiled gradient-echo (SPGR/FLASH) signal→concentration.

A selectable *alternative* to :mod:`spgr` for DCE data whose baseline ``T1``
map comes from a **separate** acquisition (e.g. a variable-flip-angle series)
recorded at a different receiver gain than the DCE. In that common preclinical
setup the fitted ``M0`` is not on the DCE signal scale, so the absolute-``M0``
inversion in :mod:`spgr` (``A = S / (M0·sinα)``) leaves the domain and returns
nonsense. This variant cancels ``M0`` by working with the intra-DCE signal
ratio ``S(t)/S_baseline`` and anchoring the absolute ``R1`` to a baseline
``T1`` (a per-voxel map where trustworthy, otherwise a configurable uniform
value):

    R1₀      = 1 / T1₀                                     (baseline relaxation)
    f(E1)    = (1 − E1) / (1 − cosα·E1)                    spoiled-GRE steady state, up to M0·sinα
    E1₀      = exp(−TR·R1₀)
    g(t)     = [S(t)/S₀] · f(E1₀)                          = f(E1(t))
    E1(t)    = (1 − g) / (1 − cosα·g)                      invert f(·)
    R1(t)    = −ln E1(t) / TR
    C(t)     = (R1(t) − R1₀) / r1

``M0`` is accepted for interface compatibility but ignored. Robust to the
VFA/DCE gain mismatch, which is the norm when the T1 map is acquired in its
own scan. Extends, and does not modify, the existing converters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import SignalToConcConverter
from .baseline import resolve_baseline_frames


@dataclass(frozen=True, slots=True)
class _SPGRRatio:
    key: ClassVar[str] = "spgr_ratio"
    name: ClassVar[str] = "Ratio SPGR/FLASH (M0-free, T1-anchored)"
    description: ClassVar[str] = (
        "Ratio-based steady-state spoiled-GRE conversion for DCE whose baseline "
        "T1 comes from a separate (e.g. VFA) acquisition. Cancels M0 via the "
        "S(t)/S_baseline ratio and anchors R1 to a baseline T1 (per-voxel map "
        "or a uniform t1_0_ms). Robust to VFA/DCE receiver-gain mismatch."
    )
    accepts: ClassVar[dict[str, type]] = {
        "signal": np.ndarray,
        "t1_ms": np.ndarray,
        "m0": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {"concentration_mM": np.ndarray}

    def convert(
        self,
        signal: np.ndarray,
        t1_ms: np.ndarray,
        m0: np.ndarray,
        *,
        flip_angle_deg: float,
        tr_s: float,
        r1_per_s_mM: float = 4.0,
        baseline_frames: int = 10,
        baseline_method: str = "auto",
        t1_0_ms: float = 0.0,
        baseline_skip: int = 2,
        max_conc_mM: float = 20.0,
        **_: Any,
    ) -> np.ndarray:
        S = np.asarray(signal, dtype=float)
        TR = float(tr_s)
        r1 = float(r1_per_s_mM)
        alpha = np.deg2rad(float(flip_angle_deg))
        cos_a = float(np.cos(alpha))

        # Baseline relaxation R1₀. A uniform t1_0_ms (>0) overrides the map,
        # which is the right choice when the separate T1 fit is unreliable
        # (e.g. VFA without B1 correction). Otherwise use the per-voxel map,
        # keeping only physiologically plausible values (0.05–8 s).
        T1_map_s = np.asarray(t1_ms, dtype=float) / 1000.0
        if float(t1_0_ms) > 0:
            R1_0 = np.full(T1_map_s.shape, 1000.0 / float(t1_0_ms))
        else:
            good = np.isfinite(T1_map_s) & (T1_map_s > 0.05) & (T1_map_s < 8.0)
            R1_0 = 1.0 / np.where(good, T1_map_s, np.nan)

        # Pre-bolus baseline window. Skip the first frame(s) (spoiled-GRE
        # approach-to-steady-state transient) which otherwise corrupts S₀ and
        # fools automatic bolus-arrival detection.
        if str(baseline_method) == "fixed":
            nbf = int(baseline_frames)
        else:
            nbf = resolve_baseline_frames(S, method=baseline_method,
                                          fallback=int(baseline_frames))
        skip = max(int(baseline_skip), 0)
        lo, hi = skip, max(int(nbf), skip + 2)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            E1_0 = np.exp(-TR * R1_0)
            f0 = (1.0 - E1_0) / (1.0 - cos_a * E1_0)          # baseline steady-state factor

            S0 = np.nanmedian(S[..., lo:hi], axis=-1)         # (...) baseline signal
            ratio = S / np.where(S0 > 0, S0, np.nan)[..., None]
            g = ratio * f0[..., None]                         # = f(E1(t))
            E1 = (1.0 - g) / (1.0 - cos_a * g)
            E1 = np.clip(E1, 1e-6, 1.0 - 1e-9)
            R1 = -np.log(E1) / TR
            C = (R1 - R1_0[..., None]) / r1
            # Residual baseline zero (C(baseline) ≈ 0 by construction).
            base = np.nanmean(C[..., lo:hi], axis=-1, keepdims=True)
            C = C - np.where(np.isfinite(base), base, 0.0)
            # Clamp the ratio-explosion in near-zero-baseline voxels (edges, air,
            # CSF partial volume) to a physiological range so a handful of unstable
            # voxels do not create hotspots downstream.
            if float(max_conc_mM) > 0:
                C = np.clip(C, -float(max_conc_mM), float(max_conc_mM))
        return C


PLUGIN = _SPGRRatio()
