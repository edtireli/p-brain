"""Bounded-inversion spoiled gradient-echo signal→concentration.

The robust counterpart to :mod:`spgr_ratio`. Same physics, same M0-free baseline
anchoring, but it never inverts the signal equation *algebraically* — and that one
change removes an entire class of silent failure.

**Why the algebraic inversion is unsafe.** ``spgr_ratio`` solves

    E1 = (1 − g) / (1 − cosα·g),      g = (S/S₀)·f(E1₀)

which is only defined while ``g < 1``. The sequence therefore has a hard ceiling on
representable enhancement, ``max S/S₀ = 1/f(E1₀)``, and past it the expression does
not merely clip — it passes through a pole. For ``1 < g < 1/cosα`` the result clips
low and the concentration explodes; for ``g > 1/cosα`` the numerator *and*
denominator flip sign together, ``E1`` reappears above 1, and the concentration
collapses to ≈0. Those two regimes sit ~0.07x apart, so a bolus crossing the ceiling
produces a curve that oscillates between the clamp and zero from frame to frame:

    -0.05, 0.33, 50.00, 7.48, -0.21, -0.21, 50.00, -0.21, ...   (a real voxel)

The voxels that do this are the *most* enhancing ones — the vasculature — so the
worst-affected compartment is exactly the one the input function is drawn from, and
it reports as though it never enhanced. Measured on 7T mouse DCE (TR 70 ms, 15°,
T1₀ 2000 ms → ceiling 1.96x): 35 % of foreground voxels over the ceiling, 29 %
wrapped to zero.

**What this converter does instead.** It keeps the *forward* model and searches for
the R1 that reproduces the measured signal, over a bounded physiological interval —
the approach the reference MATLAB implementation uses for its input function
(``fminbnd`` over an R1 bracket rather than a closed-form log). Signal that exceeds
what any T1 can produce then pins **monotonically** at the bracket edge instead of
wrapping, so the failure is ordered and visible rather than silent and oscillating.

**Why it is not slow.** ``f`` is strictly increasing in R1 — ``df/dE1 = (cosα−1) /
(1−E1cosα)² < 0`` and ``dE1/dR1 < 0``, so ``df/dR1 > 0`` everywhere. A monotone
function needs no per-voxel optimiser: one precomputed ``f(R1)`` table inverts the
whole 4-D volume in a single vectorised ``interp``. Exact to grid resolution, and
monotonicity is a *structural* guarantee rather than something to test for.

Selectable with ``--signal-to-conc spgr_bounded``. Extends, and does not modify, the
existing converters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import SignalToConcConverter
from .baseline import resolve_baseline_frames

# Physiological R1 bracket, s⁻¹ — T1 from 30 s down to 33 ms. Matches the bounds the
# reference implementation puts on its Look-Locker R1 fit. Wide enough that clipping
# means "no T1 explains this signal", not "the prior was too tight".
R1_MIN_DEFAULT = 0.033
R1_MAX_DEFAULT = 30.0


def _f(E1: np.ndarray | float, cos_a: float) -> np.ndarray:
    """Spoiled-GRE steady state up to M0·sinα: f(E1) = (1−E1)/(1−cosα·E1)."""
    return (1.0 - E1) / (1.0 - cos_a * E1)


def _r1_table(cos_a: float, tr_s: float, r1_min: float, r1_max: float,
              n: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """A strictly increasing (f, R1) pair for monotone inversion.

    Log-spaced in R1 because the resolution that matters is at the low-R1 end, where
    f changes fastest per unit concentration.
    """
    r1 = np.geomspace(max(r1_min, 1e-6), r1_max, int(n))
    f = _f(np.exp(-float(tr_s) * r1), cos_a)
    return f, r1


@dataclass(frozen=True, slots=True)
class _SPGRBounded:
    key: ClassVar[str] = "spgr_bounded"
    name: ClassVar[str] = "Bounded-inversion SPGR/FLASH (M0-free, saturation-safe)"
    description: ClassVar[str] = (
        "Forward-models the spoiled-GRE steady state and solves for R1 inside a "
        "physiological bracket instead of inverting the signal equation in closed "
        "form. Baseline-anchored, so it is immune to the VFA/DCE gain mismatch like "
        "spgr_ratio, but signal beyond the model's ceiling pins monotonically at the "
        "bracket edge rather than wrapping to ~0 mM."
    )
    accepts: ClassVar[dict[str, type]] = {"signal": np.ndarray, "t1_ms": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"concentration": np.ndarray}

    @staticmethod
    def _baseline_r1(t1_ms: np.ndarray, t1_0_ms: float) -> np.ndarray:
        """Baseline R1₀ (s⁻¹): a uniform ``t1_0_ms`` (>0) overrides the per-voxel map,
        which is the right choice when the separate T1 fit is unreliable (VFA without
        B1 correction). Otherwise the map, keeping 0.05–8 s."""
        T1_s = np.asarray(t1_ms, dtype=float) / 1000.0
        if float(t1_0_ms) > 0:
            return np.full(T1_s.shape, 1000.0 / float(t1_0_ms))
        good = np.isfinite(T1_s) & (T1_s > 0.05) & (T1_s < 8.0)
        return 1.0 / np.where(good, T1_s, np.nan)

    def convert(
        self,
        signal: np.ndarray,
        t1_ms: np.ndarray,
        m0: np.ndarray | None = None,
        *,
        flip_angle_deg: float,
        tr_s: float,
        r1_per_s_mM: float = 4.0,
        baseline_frames: int = 10,
        baseline_method: str = "auto",
        t1_0_ms: float = 0.0,
        baseline_skip: int = 2,
        max_conc_mM: float = 0.0,
        r1_min_per_s: float = R1_MIN_DEFAULT,
        r1_max_per_s: float = R1_MAX_DEFAULT,
        **_: Any,
    ) -> np.ndarray:
        S = np.asarray(signal, dtype=float)
        squeeze = S.ndim == 3
        if squeeze:
            S = S[..., None]
        cos_a = float(np.cos(np.deg2rad(float(flip_angle_deg))))
        TR, r1 = float(tr_s), float(r1_per_s_mM)
        R1_0 = self._baseline_r1(t1_ms, t1_0_ms)

        if str(baseline_method) == "fixed":
            nbf = int(baseline_frames)
        else:
            nbf = resolve_baseline_frames(S, method=baseline_method,
                                          fallback=int(baseline_frames))
        skip = max(int(baseline_skip), 0)
        lo, hi = skip, max(int(nbf), skip + 2)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Anchor to the observed baseline exactly as the ratio form does: M0 is
            # cancelled by expressing the target as f(E1(t)), so a receiver-gain
            # mismatch between the T1 fit and the DCE cannot bias the result.
            E1_0 = np.exp(-TR * R1_0)
            f0 = _f(E1_0, cos_a)
            S0 = np.nanmedian(S[..., lo:hi], axis=-1)
            g = (S / np.where(S0 > 0, S0, np.nan)[..., None]) * f0[..., None]

            # Monotone inversion. Clipping g to the table's range IS the bounded
            # search: values the model cannot reach saturate at the bracket edge, in
            # the correct direction, instead of crossing the pole.
            f_tab, r1_tab = _r1_table(cos_a, TR, float(r1_min_per_s),
                                      float(r1_max_per_s))
            finite = np.isfinite(g)
            R1 = np.full(g.shape, np.nan)
            R1[finite] = np.interp(np.clip(g[finite], f_tab[0], f_tab[-1]),
                                   f_tab, r1_tab)

            C = (R1 - R1_0[..., None]) / r1
            base = np.nanmean(C[..., lo:hi], axis=-1, keepdims=True)
            C = C - np.where(np.isfinite(base), base, 0.0)
            # A clamp is available but off by default: the R1 bracket already bounds
            # the output, so clamping here would only hide the saturation the bracket
            # is reporting honestly.
            if float(max_conc_mM) > 0:
                C = np.clip(C, -float(max_conc_mM), float(max_conc_mM))
        return C[..., 0] if squeeze else C

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
        r1_min_per_s: float = R1_MIN_DEFAULT,
        r1_max_per_s: float = R1_MAX_DEFAULT,
        **_: Any,
    ) -> dict[str, float]:
        """Fraction of foreground voxels whose enhancement pins at the R1 bracket.

        Same contract as :meth:`spgr_ratio.saturation`, and the same meaning — signal
        the steady-state model cannot produce. The difference is what happens to it:
        here it saturates monotonically, so a pinned voxel is a lower bound on the
        concentration rather than a number pointing the wrong way.
        """
        S = np.asarray(signal, dtype=float)
        if S.ndim == 3:
            S = S[..., None]
        cos_a = float(np.cos(np.deg2rad(float(flip_angle_deg))))
        R1_0 = self._baseline_r1(t1_ms, t1_0_ms)

        if str(baseline_method) == "fixed":
            nbf = int(baseline_frames)
        else:
            nbf = resolve_baseline_frames(S, method=baseline_method,
                                          fallback=int(baseline_frames))
        skip = max(int(baseline_skip), 0)
        lo, hi = skip, max(int(nbf), skip + 2)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            f0 = _f(np.exp(-float(tr_s) * R1_0), cos_a)
            S0 = np.nanmedian(S[..., lo:hi], axis=-1)
            ratio = S / np.where(S0 > 0, S0, np.nan)[..., None]

        pos = S0[np.isfinite(S0) & (S0 > 0)]
        fg = (S0 > 0.25 * float(np.median(pos))) if pos.size else np.ones_like(S0, bool)

        f_tab, _ = _r1_table(cos_a, float(tr_s), float(r1_min_per_s),
                             float(r1_max_per_s))
        ceiling = float(np.nanmedian(f_tab[-1] / np.where(f0 > 0, f0, np.nan)))
        peak = np.nanmax(ratio, axis=-1)[fg]
        peak = peak[np.isfinite(peak)]
        if not peak.size:
            return {"ceiling_ratio": ceiling, "wrap_ratio": float("inf"),
                    "over_ceiling_fraction": 0.0, "wrapped_fraction": 0.0,
                    "peak_ratio_median": float("nan"), "peak_ratio_p95": float("nan"),
                    "foreground_voxels": 0}
        return {
            "ceiling_ratio": ceiling,
            # Structurally unreachable: a monotone inversion has no second branch.
            "wrap_ratio": float("inf"),
            "over_ceiling_fraction": float(np.mean(peak > ceiling)),
            "wrapped_fraction": 0.0,
            "peak_ratio_median": float(np.median(peak)),
            "peak_ratio_p95": float(np.percentile(peak, 95)),
            "foreground_voxels": int(peak.size),
        }


PLUGIN = _SPGRBounded()
