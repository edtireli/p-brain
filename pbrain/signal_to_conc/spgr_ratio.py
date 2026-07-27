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

    @staticmethod
    def _baseline_r1(t1_ms: np.ndarray, t1_0_ms: float) -> np.ndarray:
        """Baseline relaxation R1₀ (s⁻¹). A uniform ``t1_0_ms`` (>0) overrides the
        map, which is the right choice when the separate T1 fit is unreliable
        (e.g. VFA without B1 correction). Otherwise use the per-voxel map, keeping
        only physiologically plausible values (0.05–8 s)."""
        T1_map_s = np.asarray(t1_ms, dtype=float) / 1000.0
        if float(t1_0_ms) > 0:
            return np.full(T1_map_s.shape, 1000.0 / float(t1_0_ms))
        good = np.isfinite(T1_map_s) & (T1_map_s > 0.05) & (T1_map_s < 8.0)
        return 1.0 / np.where(good, T1_map_s, np.nan)

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

        R1_0 = self._baseline_r1(t1_ms, t1_0_ms)

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

    def saturation(
        self,
        signal: np.ndarray,
        t1_ms: np.ndarray,
        m0: np.ndarray | None = None,
        *,
        flip_angle_deg: float = 15.0,
        tr_s: float = 0.005,
        t1_0_ms: float = 0.0,
        baseline_frames: int = 5,
        baseline_skip: int = 0,
        baseline_method: str = "auto",
        **_: Any,
    ) -> dict[str, float]:
        """How much of this series falls outside what ratio-SPGR can represent.

        The inversion ``E1 = (1 − g)/(1 − cosα·g)`` is only defined while ``g < 1``,
        and ``g = (S/S₀)·f(E1₀)``. So the sequence has a hard, analytic ceiling on
        measurable enhancement::

            max S/S₀ = 1 / f(E1₀)

        Beyond it the voxel has left the model's domain, and there are **two**
        regimes — the second is the dangerous one::

            ceiling <= S/S₀ < ceiling/cosα   E1 goes negative, clips low,
                                             C explodes -> hits max_conc_mM
            S/S₀ >= ceiling/cosα             numerator AND denominator flip sign,
                                             E1 wraps above 1, clips high,
                                             C collapses to ~0

        The window between them is narrow (1.96x to 2.03x at 15°), so a voxel whose
        bolus crosses it does not saturate cleanly: it oscillates between the clamp
        and zero from frame to frame, giving a physically impossible curve that
        still looks like data. Measured on 7T mouse DCE: 3.5 % of brain voxels
        clamped, 17.8 % wrapped, and the wrapped set is the *most* enhancing —
        the vasculature, reported as though it never enhanced at all.

        The clamp in :meth:`convert` is written for "a handful of unstable voxels";
        when a real fraction of the *brain* exceeds the ceiling it is converting a
        systematic violation into plausible-looking numbers, and every downstream
        Ki/CBF inherits it.

        Note a ratio above the ceiling implies ``f > 1``, which **no T1 can
        produce** — so it is not a large concentration being clipped, it is signal
        the steady-state model cannot generate at all (inflow / non-steady-state in
        2-D multi-slice being the usual cause). Raising ``t1_0_ms`` widens the
        ceiling and will quiet this, but only by asserting a baseline T1 the data
        does not support.

        The ceiling is tight when the flip angle is near the Ernst angle for ``T1₀``
        — exactly the choice that maximises baseline SNR. At TR 70 ms / 15° / 2000 ms
        it is only 1.96×, which a blood pool reaches easily.

        Returns the ceiling, the measured enhancement quantiles and the fraction
        over the ceiling, for foreground voxels only (air would otherwise dominate).
        Reported, never enforced: what to do about it is an acquisition decision.
        """
        S = np.asarray(signal, dtype=float)
        if S.ndim == 3:
            S = S[..., None]
        a = np.deg2rad(float(flip_angle_deg))
        cos_a = float(np.cos(a))
        R1_0 = self._baseline_r1(t1_ms, t1_0_ms)

        if str(baseline_method) == "fixed":
            nbf = int(baseline_frames)
        else:
            nbf = resolve_baseline_frames(S, method=baseline_method,
                                          fallback=int(baseline_frames))
        skip = max(int(baseline_skip), 0)
        lo, hi = skip, max(int(nbf), skip + 2)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            E1_0 = np.exp(-float(tr_s) * R1_0)
            f0 = (1.0 - E1_0) / (1.0 - cos_a * E1_0)
            S0 = np.nanmedian(S[..., lo:hi], axis=-1)
            ratio = S / np.where(S0 > 0, S0, np.nan)[..., None]

        # Foreground = voxels with a real baseline. Air has S₀ ≈ 0, which makes the
        # ratio explode for reasons that have nothing to do with the ceiling.
        pos = S0[np.isfinite(S0) & (S0 > 0)]
        fg = (S0 > 0.25 * float(np.median(pos))) if pos.size else np.ones_like(S0, bool)

        ceiling = float(np.nanmedian(1.0 / np.where(f0 > 0, f0, np.nan)))
        wrap = ceiling / cos_a if cos_a > 0 else float("inf")
        peak = np.nanmax(ratio, axis=-1)[fg]
        peak = peak[np.isfinite(peak)]
        if not peak.size:
            return {"ceiling_ratio": ceiling, "wrap_ratio": wrap,
                    "over_ceiling_fraction": 0.0, "wrapped_fraction": 0.0,
                    "peak_ratio_median": float("nan"), "peak_ratio_p95": float("nan"),
                    "foreground_voxels": 0}
        return {
            "ceiling_ratio": ceiling,
            "wrap_ratio": float(wrap),
            "over_ceiling_fraction": float(np.mean(peak > ceiling)),
            "wrapped_fraction": float(np.mean(peak >= wrap)),
            "peak_ratio_median": float(np.median(peak)),
            "peak_ratio_p95": float(np.percentile(peak, 95)),
            "foreground_voxels": int(peak.size),
        }


PLUGIN = _SPGRRatio()
