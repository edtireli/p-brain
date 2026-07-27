"""Gamma transit-time model (Larsson form) — local-only kinetic model.

    h(t)  = ((t-dt)/(t_max-dt))^α · exp(α·(1 - (t-dt)/(t_max-dt)))     ← capillary TTD
    R(t)  = (1-E)·(1-H(t)) + E·(1-H_ev(t))                              ← residue
    Cₜ(t) = F · (Cₐ * R)(t)                                             ← forward model

with the 9-parameter vector ``[F, t_max, dt, α, E, k₂, offset, B, k₄]``
matching MATLAB's ``con2perf_gamma3_6.m``. Fitted by multi-restart TRF.

The legacy implementation at ``models/gamma.py`` (validated to MATLAB to
1e-12) optionally uses JAX for batch acceleration; this clean port is
NumPy-only for portability, with the same forward math.

**Note on git:** this file is in ``.gitignore`` so it never ships on the
public ``main`` branch. Discovery treats it identically to ``patlak.py``
when present — locally, the registry picks it up; on a fresh clone, it
simply isn't there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from pbrain._numpy_compat import trapezoid
from .base import CurveInputs, ModelResult


H_FACTOR = 40  # Upsampling factor for the convolution grid


# ────────────────────────────────────────────────────────────────────────
# Parameter bounds (MATLAB menu_11 defaults with linear constraints
# absorbed; dt / offset / B / k4 pinned for the standard fit)
# ────────────────────────────────────────────────────────────────────────


BOUNDS_LO = np.array([1e-5,  0.01, 0.0, 0.001, 0.0, 0.0,    0.0, 0.0, 0.0])
BOUNDS_HI = np.array([400.0 / 6000.0, 15.0, 0.0, 30.0, 1.0, 0.002, 0.0, 0.0, 0.0])
PARAM_NAMES = ("F", "t_max", "dt", "alpha", "E", "k2", "offset", "B", "k4")


# ────────────────────────────────────────────────────────────────────────
# AIF shift (MATLAB inputshift3)
# ────────────────────────────────────────────────────────────────────────


def _shift_curve_pchip(time_s: np.ndarray, ca: np.ndarray, deltaT: float) -> np.ndarray:
    """PCHIP shift of input function ``ca`` by ``deltaT`` seconds."""
    if abs(deltaT) < 1e-12:
        return np.asarray(ca, dtype=float).copy()
    time_s = np.asarray(time_s, dtype=float).ravel()
    ca = np.asarray(ca, dtype=float).ravel()
    n = ca.size
    timeunit = float(time_s[2] - time_s[1]) if n >= 3 else float(time_s[1] - time_s[0])
    notimeunit = int(np.ceil(abs(deltaT) / timeunit))
    if deltaT > 0:
        timetemp = (time_s[0] + deltaT - notimeunit * timeunit) + np.arange(
            notimeunit + n, dtype=float
        ) * timeunit
        sCa = np.zeros(notimeunit + n)
        sCa[notimeunit:] = ca
    else:
        timetemp = (time_s[0] + deltaT) + np.arange(n + notimeunit, dtype=float) * timeunit
        sCa = np.empty(n + notimeunit)
        sCa[:n] = ca
        sCa[n:] = ca[-1]
    return PchipInterpolator(timetemp, sCa, extrapolate=True)(time_s)


# ────────────────────────────────────────────────────────────────────────
# Larsson forward model + fitter
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GammaFit:
    cth_s: float
    mtt_cap_s: float
    mtt_total_s: float
    extraction: float
    k2_per_s: float
    alpha: float
    beta_s: float
    dt_s: float
    offset_s: float
    f_ml_100g_min: float
    sse: float


class _LarssonFitter:
    """NumPy-only Larsson fitter (FFT-based convolution, multi-restart TRF)."""

    def __init__(self, time_s: np.ndarray, ca: np.ndarray, h_factor: int = H_FACTOR,
                 weight_flag: int = 1):
        time_s = np.asarray(time_s, dtype=float).ravel()
        ca = np.asarray(ca, dtype=float).ravel()
        n = int(min(time_s.size, ca.size))
        self.time_s = time_s[:n]
        self.ca_orig = ca[:n]
        self.n_orig = n
        self.dt0 = float(time_s[1] - time_s[0])
        self.h_factor = int(h_factor)
        self.weight_flag = int(weight_flag)

        nh = (n - 1) * self.h_factor + 1
        self.nh = nh
        self.dth = self.dt0 / self.h_factor
        self.time_h = np.linspace(0.0, float(time_s[n - 1]), nh)

        if self.weight_flag == 1:
            kw = -np.log(0.25) / 1800.0
            self.w_h = np.exp(-kw * self.time_h)
        else:
            self.w_h = np.ones(nh)

        self.nfft = next_fast_len(2 * nh)
        self._aif_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._ds_idx = np.clip(np.searchsorted(self.time_h, self.time_s[:n]), 0, nh - 1)

    def _aif_for_offset(self, offset: float):
        key = round(float(offset) * 1e6) / 1e6
        cached = self._aif_cache.get(key)
        if cached is not None:
            return cached
        ca = self.ca_orig if abs(offset) < 1e-9 else _shift_curve_pchip(self.time_s, self.ca_orig, offset)
        ca_h = np.interp(self.time_h, self.time_s, ca)
        ca_h[-1] = ca_h[-2]
        ca_h_fft = np.fft.rfft(ca_h, n=self.nfft)
        cached = (ca_h, ca_h_fft)
        if len(self._aif_cache) < 64:
            self._aif_cache[key] = cached
        return cached

    def _interp_tissue(self, ct: np.ndarray) -> np.ndarray:
        ct_h = np.interp(self.time_h, self.time_s, ct[: self.n_orig])
        ct_h[-1] = ct_h[-2]
        return ct_h

    def forward(self, fit: "GammaFit", time_out: np.ndarray | None = None) -> np.ndarray:
        """Reconstruct the full-RIF model Cₜ for a fitted result.

        Uses the *complete* residue (capillary + extravasation), so the
        diagnostic overlay reflects what the model actually fit — not a
        capillary-only approximation.
        """
        f = float(fit.f_ml_100g_min) / 6000.0
        x = np.zeros(9)
        x[0] = f
        x[1] = float(fit.mtt_cap_s)
        x[3] = float(fit.alpha)
        x[4] = float(fit.extraction)
        x[5] = float(fit.k2_per_s)
        # _residuals(x, zeros) = w_h · ct_calc  → divide out the weights
        ct_calc_h = self._residuals(x, np.zeros(self.nh)) / self.w_h
        out_t = self.time_s if time_out is None else np.asarray(time_out, dtype=float)
        return np.interp(out_t, self.time_h, ct_calc_h)

    def _residuals(self, x: np.ndarray, ct_h: np.ndarray) -> np.ndarray:
        f, t_max, dt_val, alpha, E, k2, offset, B, k4 = x[:9]
        denom = t_max - dt_val
        if denom <= 1e-12 or alpha <= 1e-9 or f <= 0:
            return np.full(self.nh, 1e8)

        th, nh, dth = self.time_h, self.nh, self.dth
        tau = (th - dt_val) / denom
        h = np.where(tau > 0,
                     np.power(np.maximum(tau, 0.0), alpha) * np.exp(alpha * (1.0 - tau)),
                     0.0)
        i_mask = int(np.searchsorted(th, dt_val, side="right"))
        if i_mask > 0:
            h[:i_mask] = 0.0
        h_area = float(trapezoid(h, dx=dth))
        if h_area <= 1e-15:
            return np.full(nh, 1e8)
        h /= h_area
        h_fft = rfft(h, n=self.nfft)

        H = np.empty(nh)
        H[0] = 0.0
        H[1:] = np.cumsum(0.5 * (h[1:] + h[:-1])) * dth

        if E > 1e-9 and k2 > 1e-12:
            ev = k2 * np.exp(-k2 * th)
            ev_fft = rfft(ev, n=self.nfft)
            h_ev = irfft(h_fft * ev_fft, n=self.nfft)[:nh] * dth
            H_ev = np.empty(nh)
            H_ev[0] = 0.0
            H_ev[1:] = np.cumsum(0.5 * (h_ev[1:] + h_ev[:-1])) * dth
        else:
            H_ev = np.zeros(nh)

        if E > 1e-9 and B > 1e-9 and k4 > 1e-12:
            ev4 = k4 * np.exp(-k4 * th)
            ev4_fft = rfft(ev4, n=self.nfft)
            h_ev4 = irfft(h_fft * ev4_fft, n=self.nfft)[:nh] * dth
            H_ev4 = np.empty(nh)
            H_ev4[0] = 0.0
            H_ev4[1:] = np.cumsum(0.5 * (h_ev4[1:] + h_ev4[:-1])) * dth
        else:
            H_ev4 = np.zeros(nh)

        RIF = ((1.0 - E) * (1.0 - H)
               + E * (1.0 - B) * (1.0 - H_ev)
               + E * B * (1.0 - H_ev4))

        _, ca_h_fft = self._aif_for_offset(offset)
        rif_fft = rfft(RIF, n=self.nfft)
        ct_calc = irfft(ca_h_fft * rif_fft, n=self.nfft)[:nh] * dth * f
        return self.w_h * (ct_calc - ct_h)

    # Physiological shape-parameter starts (t_max, alpha, E, k2). Deterministic
    # — chosen to bracket realistic capillary TTDs, so the fit converges from a
    # sensible basin instead of stumbling on the degenerate alpha→0 spike that
    # random restarts over the full bounds discover (and which over-fits with
    # lower SSE while being physically meaningless).
    _SHAPE_STARTS = (
        (3.0, 4.0, 0.05, 1e-4),
        (4.0, 10.0, 0.10, 1e-3),
        (2.0, 15.0, 0.02, 5e-4),
        (6.0, 6.0, 0.20, 2e-3),
        (5.0, 20.0, 0.10, 1e-3),
        (3.5, 8.0, 0.30, 3e-3),
    )

    def fit(self, ct: np.ndarray, f_cbf: float, n_restarts: int = 16,
            pin_f: bool = True, early_stop_patience: int = 2) -> GammaFit:
        """Fit the Larsson gamma TTD.

        ``f_cbf`` is the flow (1/s). When ``pin_f`` is True (default,
        matching the legacy two-stage pipeline) F is held fixed at ``f_cbf``
        — which should be a robust prior estimate, e.g. the Tikhonov CBF /
        6000 — and only the TTD *shape* (t_max, α, E, k₂) is fitted. Pinning
        F removes the F-vs-shape degeneracy that otherwise lets F balloon and
        α collapse to a spike. When ``pin_f`` is False, F floats (legacy free
        fit, kept for completeness).
        """
        ct_h = self._interp_tissue(np.asarray(ct, dtype=float).ravel())
        rng = np.random.default_rng(0)

        if pin_f:
            free_idx = np.array([1, 3, 4, 5], dtype=int)     # t_max, alpha, E, k2
            pinned = np.zeros(9, dtype=float)
            pinned[0] = float(max(f_cbf, 1e-6))
        else:
            free_idx = np.array([0, 1, 3, 4, 5], dtype=int)  # F + shape
            pinned = np.zeros(9, dtype=float)

        lo = BOUNDS_LO[free_idx].copy()
        hi = BOUNDS_HI[free_idx].copy()
        if not pin_f:
            lo[0] = max(lo[0], 1e-5)

        # Deterministic physiological starts first; pad with random restarts.
        starts: list[np.ndarray] = []
        for s in self._SHAPE_STARTS:
            starts.append(np.array(s if pin_f else (float(np.clip(f_cbf, lo[0], hi[0])), *s)))
        while len(starts) < int(n_restarts):
            starts.append(lo + rng.random(len(free_idx)) * (hi - lo))

        best_x: np.ndarray | None = None
        best_sse = np.inf

        # Early stopping: the physiological starts almost always converge to
        # the same basin, so most restarts are redundant. Stop once
        # ``early_stop_patience`` consecutive starts fail to improve the SSE
        # by more than 0.5 % — this is lossless for the (common) converging
        # case and still exhausts all starts on genuinely multi-modal curves.
        patience = int(early_stop_patience)
        no_improve = 0

        for x_free0 in starts[:max(int(n_restarts), len(self._SHAPE_STARTS))]:
            def residuals(x_free: np.ndarray) -> np.ndarray:
                x = pinned.copy()
                x[free_idx] = x_free
                return self._residuals(x, ct_h)

            try:
                sol = least_squares(
                    residuals, x_free0, bounds=(lo, hi),
                    method="trf", max_nfev=200, ftol=1e-7, xtol=1e-7,
                )
                sse = float(np.dot(sol.fun, sol.fun))
                if sse < best_sse * (1.0 - 5e-3):       # meaningful improvement
                    best_sse = sse; best_x = sol.x.copy(); no_improve = 0
                else:
                    if sse < best_sse:                  # keep the marginally better
                        best_sse = sse; best_x = sol.x.copy()
                    no_improve += 1
                if best_x is not None and patience > 0 and no_improve >= patience:
                    break
            except Exception:
                continue

        if best_x is None:
            return GammaFit(*([float("nan")] * 11))

        x_full = pinned.copy()
        x_full[free_idx] = best_x
        f, t_max, _dt, alpha, E, k2, _off, B, k4 = x_full

        # Derived quantities
        beta_s = (t_max / alpha) if alpha > 0 else float("nan")
        # MTT_cap = alpha · beta = t_max
        mtt_cap = float(t_max)
        # MTT_total = MTT_cap + E · k2⁻¹ (one-compartment extension)
        mtt_total = mtt_cap + (E / k2 if k2 > 1e-12 else 0.0)
        # CTH = β · √α  (SD of gamma(α, β))
        cth = float(beta_s * math.sqrt(alpha)) if (alpha > 0 and beta_s > 0) else float("nan")

        return GammaFit(
            cth_s=cth,
            mtt_cap_s=mtt_cap,
            mtt_total_s=float(mtt_total),
            extraction=float(E),
            k2_per_s=float(k2),
            alpha=float(alpha),
            beta_s=float(beta_s),
            dt_s=0.0,
            offset_s=0.0,
            f_ml_100g_min=float(f * 6000.0),
            sse=float(best_sse),
        )


# ────────────────────────────────────────────────────────────────────────
# Variable-projection shape dictionary — voxelwise, vectorised, global
# ────────────────────────────────────────────────────────────────────────


class _GammaVarpro:
    """Voxelwise gamma fit by variable projection over a shape dictionary.

    The MATLAB ``con2perf_gamma3_6`` residue rearranges to

        R(t) = (1−H) + E·(H − H_ev),          H = ∫h,  H_ev = ∫(h*k₂e^{−k₂t})

    so the forward is **linear in the two amplitudes** ``A = F`` and ``C = F·E``
    once the *shape* ``(t_max, α, k₂)`` is fixed::

        Cₜ = A·(Cₐ⊗R_cap) + C·(Cₐ⊗(H−H_ev)) = A·b₁ + C·b₂.

    We grid the three shape parameters, precompute the two AIF-convolved bases
    ``b₁,b₂`` per grid point (shared by *every* voxel), and solve the
    non-negative 2-amplitude problem in closed form for all voxels at once,
    keeping each voxel's min-SSE grid point. This is **global** (no local-minima
    roulette, unlike the per-voxel multi-restart TRF), exact to the grid, and
    fully vectorised — fast enough to run voxelwise (seconds/brain vs ~10 h),
    which finally makes a true voxelwise CTH map (preferred over Tikhonov's).
    The gridded shape removes the F-vs-shape degeneracy that forced the TRF
    path to pin F, so here F is recovered directly.
    """

    def __init__(self, time_s: np.ndarray, ca: np.ndarray,
                 h_factor: int = H_FACTOR, weight_flag: int = 1):
        time_s = np.asarray(time_s, dtype=float).ravel()
        ca = np.asarray(ca, dtype=float).ravel()
        n = int(min(time_s.size, ca.size))
        self.time_s = time_s[:n]
        self.n = n
        self.dt0 = float(time_s[1] - time_s[0])
        self.h_factor = int(h_factor)
        nh = (n - 1) * self.h_factor + 1
        self.nh = nh
        self.dth = self.dt0 / self.h_factor
        self.time_h = np.linspace(0.0, float(time_s[n - 1]), nh)
        self.nfft = next_fast_len(2 * nh)
        ca_h = np.interp(self.time_h, self.time_s, ca[:n])
        ca_h[-1] = ca_h[-2]
        self.ca_h_fft = np.fft.rfft(ca_h, n=self.nfft)
        if weight_flag == 1:
            kw = -np.log(0.25) / 1800.0
            self.w = np.exp(-kw * self.time_s)
        else:
            self.w = np.ones(n)

    def _h_and_H(self, t_max: float, alpha: float):
        th, dth = self.time_h, self.dth
        tau = th / t_max
        h = np.where(tau > 0.0,
                     np.power(np.maximum(tau, 0.0), alpha) * np.exp(alpha * (1.0 - tau)),
                     0.0)
        area = float(trapezoid(h, dx=dth))
        if area <= 1e-15:
            return None, None
        h = h / area
        H = np.empty(self.nh)
        H[0] = 0.0
        H[1:] = np.cumsum(0.5 * (h[1:] + h[:-1])) * dth
        return h, H

    def _conv_down(self, kernel_h: np.ndarray) -> np.ndarray:
        """(Cₐ ⊗ kernel) on the upsampled grid, then sampled to the data grid."""
        full = np.fft.irfft(self.ca_h_fft * np.fft.rfft(kernel_h, n=self.nfft),
                            n=self.nfft)[:self.nh] * self.dth
        return np.interp(self.time_s, self.time_h, full)

    def fit(self, Ct: np.ndarray, *, grid_tmax: np.ndarray, grid_alpha: np.ndarray,
            grid_k2: np.ndarray, f_pin: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """``Ct`` is ``(n_t, V)`` tissue on the data grid. Returns param arrays.

        ``f_pin`` (per-voxel flow F in 1/s, e.g. Tikhonov CBF/6000): when given,
        A=F is **fixed** and only the leak amplitude C=F·E is solved per grid
        point — the legacy two-stage design (Tikhonov flow → gamma shape). This
        removes the F-vs-shape degeneracy that otherwise lets a spiky short-MTT
        residue trade against an inflated F. When ``None``, both amplitudes are
        recovered (the gridded shape still bounds the degeneracy for typical
        tissue, but pinning is preferred where a robust CBF prior exists).
        """
        w = self.w[:, None]
        Y = np.asarray(Ct, dtype=float)[:self.n] * w          # weighted tissue (n, V)
        yy = np.sum(Y * Y, axis=0)                            # ‖wCₜ‖² per voxel
        V = Y.shape[1]
        pin = f_pin is not None
        A_fix = (np.asarray(f_pin, dtype=float).reshape(-1) if pin
                 else np.zeros(V))
        best = np.full(V, np.inf)
        A_b = np.zeros(V); C_b = np.zeros(V)
        tm_b = np.zeros(V); al_b = np.zeros(V); k2_b = np.zeros(V)

        for tm in grid_tmax:
            for al in grid_alpha:
                h, H = self._h_and_H(float(tm), float(al))
                if h is None:
                    continue
                wb1 = self._conv_down(1.0 - H) * self.w        # weighted b₁
                g11 = float(wb1 @ wb1) + 1e-30
                r1 = wb1 @ Y                                   # (V,)
                # capillary-only (E=0): C=0; A pinned, or A = r1/g11 (≥0)
                A0 = A_fix if pin else np.clip(r1 / g11, 0.0, None)
                sse0 = yy - 2.0 * A0 * r1 + A0 * A0 * g11
                upd = sse0 < best
                if upd.any():
                    best = np.where(upd, sse0, best)
                    A_b = np.where(upd, A0, A_b); C_b = np.where(upd, 0.0, C_b)
                    tm_b = np.where(upd, tm, tm_b); al_b = np.where(upd, al, al_b)
                    k2_b = np.where(upd, 0.0, k2_b)
                h_fft = np.fft.rfft(h, n=self.nfft)
                for k2 in grid_k2:
                    if k2 <= 0:
                        continue
                    ev = float(k2) * np.exp(-float(k2) * self.time_h)
                    h_ev = np.fft.irfft(h_fft * np.fft.rfft(ev, n=self.nfft),
                                        n=self.nfft)[:self.nh] * self.dth
                    H_ev = np.empty(self.nh)
                    H_ev[0] = 0.0
                    H_ev[1:] = np.cumsum(0.5 * (h_ev[1:] + h_ev[:-1])) * self.dth
                    wb2 = self._conv_down(H - H_ev) * self.w
                    g12 = float(wb1 @ wb2); g22 = float(wb2 @ wb2) + 1e-30
                    r2 = wb2 @ Y
                    if pin:
                        # A fixed → C = (r2 − A·g12)/g22, clipped ≥ 0 (1-var NNLS)
                        A = A_fix
                        C = np.clip((r2 - A * g12) / g22, 0.0, None)
                    else:
                        det = g11 * g22 - g12 * g12
                        if abs(det) < 1e-30:
                            continue
                        A = (g22 * r1 - g12 * r2) / det
                        C = (-g12 * r1 + g11 * r2) / det
                        # project to A,C ≥ 0 (exact 2-var NNLS active set)
                        cneg = C < 0.0
                        A = np.where(cneg, np.clip(r1 / g11, 0.0, None), A)
                        C = np.where(cneg, 0.0, C)
                        aneg = A < 0.0
                        C = np.where(aneg, np.clip(r2 / g22, 0.0, None), C)
                        A = np.where(aneg, 0.0, A)
                    sse = (yy - 2.0 * (A * r1 + C * r2)
                           + A * A * g11 + 2.0 * A * C * g12 + C * C * g22)
                    upd = sse < best
                    if upd.any():
                        best = np.where(upd, sse, best)
                        A_b = np.where(upd, A, A_b); C_b = np.where(upd, C, C_b)
                        tm_b = np.where(upd, tm, tm_b); al_b = np.where(upd, al, al_b)
                        k2_b = np.where(upd, k2, k2_b)

        F = A_b
        E = np.where(F > 1e-12, C_b / np.maximum(F, 1e-12), 0.0)
        E = np.clip(E, 0.0, 1.0)
        cth = np.where(al_b > 0.0, tm_b / np.sqrt(np.maximum(al_b, 1e-9)), np.nan)
        return {
            "cbf": F * 6000.0, "mtt": tm_b, "cth": cth,
            "extraction": E, "k2": k2_b, "alpha": al_b, "sse": best,
        }


def _default_gamma_grid():
    """Physiological shape grid (t_max s, α, k₂ 1/s) for the VARPRO dictionary."""
    grid_tmax = np.linspace(0.5, 12.0, 18)
    grid_alpha = np.geomspace(1.0, 25.0, 18)
    grid_k2 = np.array([1e-4, 5e-4, 1e-3, 2e-3])     # k₂=0 (no leak) handled inline
    return grid_tmax, grid_alpha, grid_k2


# ────────────────────────────────────────────────────────────────────────
# Plug-in adapter
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GammaModel:
    key: ClassVar[str] = "gamma"
    name: ClassVar[str] = "Gamma transit-time (Larsson)"
    description: ClassVar[str] = (
        "MATLAB-equivalent Larsson gamma TTD model. CTH, MTT_cap, "
        "MTT_total, E, k2, F via multi-restart TRF on the 5 free "
        "parameters [F, t_max, alpha, E, k2] (dt/offset/B/k4 pinned)."
    )
    accepts: ClassVar[dict[str, type]] = {
        "c_tissue": np.ndarray,
        "c_input": np.ndarray,
        "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "cbf": np.ndarray,
        "mtt": np.ndarray,
        "cth": np.ndarray,
        "extraction": np.ndarray,
        "k2": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("cbf", "mtt", "cth", "extraction", "k2")
    # Voxelwise is now feasible: the variable-projection shape dictionary
    # (_GammaVarpro) fits the whole brain in ~10 s (vs ~10 h for the per-voxel
    # multi-restart TRF), so the KineticStage runs gamma per voxel and gets a
    # true voxelwise CTH map — preferred over Tikhonov's noisy CTH. The TRF path
    # remains for single-curve (parcel) fits and as a `method="trf"` fallback.
    supports_voxelwise: ClassVar[bool] = True
    units: ClassVar[dict[str, str]] = {
        "cbf": "mL/100g/min",
        "mtt": "s",
        "cth": "s",
        "extraction": "(fraction)",
        "k2": "1/s",
    }

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        c_t = np.asarray(inputs.c_tissue, dtype=float)
        c_a = np.asarray(inputs.c_input, dtype=float)
        t_s = np.asarray(inputs.t_s, dtype=float)

        n_restarts = int(opts.pop("n_restarts", 6))
        # Pin the gamma flow F to the Tikhonov CBF (default). F *is* the flow:
        # CBF = F · 6000, so "pin to Tikhonov CBF" ≡ "fix F = CBF/6000". Pinning
        # resolves the F-vs-shape degeneracy that otherwise lets F balloon and
        # α collapse to a spike (the legacy two-stage design: Tikhonov flow →
        # gamma shape). ``pin_cbf`` is the clear name; ``pin_f`` is the alias.
        pin_f = bool(opts.pop("pin_cbf", opts.pop("pin_f", True)))
        early_stop_patience = int(opts.pop("early_stop_patience", 2))
        f_cbf0 = float(opts.pop("f_cbf_initial", 50.0 / 6000.0))
        # Optional pre-computed flow to pin F to (1/s, or a per-voxel array).
        # When absent and pin_f is on, seed F per voxel from a fast Tikhonov
        # CBF pass — the legacy two-stage design (Tikhonov flow → gamma shape).
        f_cbf_in = opts.pop("f_cbf", None)
        fitter = _LarssonFitter(t_s, c_a, h_factor=int(opts.pop("h_factor", H_FACTOR)))

        def _seed_f(curves_2d: np.ndarray, mask=None) -> np.ndarray:
            """Per-curve F (1/s) for pinning, from a Tikhonov CBF pass."""
            from pbrain.models.tikhonov import build_tikhonov_solver
            solver = build_tikhonov_solver(
                t_s, c_a, lambda_selection="evidence", lambda_spacing="log",
                lambda_min=1e-3, mtt_cth_method="residue_integral",
            )
            batch = solver(curves_2d, offsets_s=None)
            cbf = np.asarray(batch.cbf_ml_per_100g_min, dtype=float)
            f = cbf / 6000.0
            f[~np.isfinite(f) | (f <= 0)] = f_cbf0
            return f

        if c_t.ndim == 1:
            if pin_f:
                f_seed = (float(f_cbf_in) if f_cbf_in is not None
                          else float(_seed_f(c_t.reshape(-1, 1))[0]))
            else:
                f_seed = f_cbf0
            fit = fitter.fit(c_t, f_cbf=f_seed, n_restarts=n_restarts, pin_f=pin_f,
                             early_stop_patience=early_stop_patience)
            return ModelResult(
                maps={
                    "cbf": np.asarray(fit.f_ml_100g_min),
                    "mtt": np.asarray(fit.mtt_cap_s),   # capillary MTT (pairs with CTH)
                    "cth": np.asarray(fit.cth_s),
                    "extraction": np.asarray(fit.extraction),
                    "k2": np.asarray(fit.k2_per_s),
                },
                units=dict(self.units),
                aux={
                    "alpha": float(fit.alpha),
                    "beta_s": float(fit.beta_s),
                    "mtt_total_s": float(fit.mtt_total_s),
                    "sse": float(fit.sse),
                },
            )

        if c_t.ndim != 2:
            raise ValueError(f"c_tissue must be 1-D or 2-D; got shape {c_t.shape}")

        n_t, n_v = c_t.shape
        cbf = np.full(n_v, np.nan)
        mtt = np.full(n_v, np.nan)
        cth = np.full(n_v, np.nan)
        E = np.full(n_v, np.nan)
        k2 = np.full(n_v, np.nan)

        if inputs.mask is None:
            voxels = np.arange(n_v)
        else:
            voxels = np.flatnonzero(np.asarray(inputs.mask, dtype=bool))
        if voxels.size == 0:
            return ModelResult(
                maps={"cbf": cbf, "mtt": mtt, "cth": cth, "extraction": E, "k2": k2},
                units=dict(self.units), aux={"method": "none"})

        # Per-voxel F (1/s) to pin the gamma flow to — the pipeline's Tikhonov
        # CBF when provided, else a fast Tikhonov pass over the kept voxels.
        if pin_f:
            if f_cbf_in is not None:
                f_arr = np.asarray(f_cbf_in, dtype=float).reshape(-1)
                f_all = (f_arr if f_arr.size == n_v else np.full(n_v, float(f_arr.flat[0])))
            else:
                f_all = np.full(n_v, f_cbf0)
                f_all[voxels] = _seed_f(c_t[:, voxels])
            f_pin = np.asarray(f_all[voxels], dtype=float)
        else:
            f_pin = None

        method = str(opts.pop("method", "varpro")).lower()
        if method == "varpro":
            # Vectorised variable-projection over the shape dictionary — global,
            # exact-to-grid, ~3000× faster than the per-voxel TRF, so a true
            # voxelwise CTH map is finally feasible at cohort scale.
            vp = _GammaVarpro(t_s, c_a, h_factor=int(opts.get("h_factor", H_FACTOR)))
            gt, ga, gk = _default_gamma_grid()
            res = vp.fit(c_t[:, voxels], grid_tmax=gt, grid_alpha=ga, grid_k2=gk, f_pin=f_pin)
            cbf[voxels] = res["cbf"]; mtt[voxels] = res["mtt"]; cth[voxels] = res["cth"]
            E[voxels] = res["extraction"]; k2[voxels] = res["k2"]
        else:
            # Legacy per-voxel multi-restart TRF (reference / fallback).
            f_seeds = (np.full(n_v, f_cbf0) if f_pin is None
                       else np.full(n_v, f_cbf0))
            if f_pin is not None:
                f_seeds[voxels] = f_pin
            for vi, v in enumerate(voxels):
                v = int(v)
                f = fitter.fit(c_t[:, v], f_cbf=float(f_seeds[v]),
                               n_restarts=n_restarts, pin_f=pin_f,
                               early_stop_patience=early_stop_patience)
                cbf[v] = f.f_ml_100g_min; mtt[v] = f.mtt_cap_s; cth[v] = f.cth_s
                E[v] = f.extraction; k2[v] = f.k2_per_s

        return ModelResult(
            maps={"cbf": cbf, "mtt": mtt, "cth": cth, "extraction": E, "k2": k2},
            units=dict(self.units),
            aux={"method": method},
        )


    def review(self, inputs, result, **_kw):
        """--mode verify view — mean tissue Cₜ + the gamma-variate fit + medians."""
        from ._review import curve_fit_review
        return curve_fit_review(self, inputs, result,
                                title="Gamma-variate · residue", fit_label="gamma fit",
                                show=("cbf", "mtt", "cth"))

    def predict(self, maps: dict[str, Any], c_input: np.ndarray,
                t_s: np.ndarray) -> np.ndarray:
        """Reconstruct Cₜ from the gamma maps via the Larsson forward (full RIF).

        ``alpha`` is recovered from the published shape identities
        ``MTT_cap = t_max = α·β`` and ``CTH = β·√α`` ⇒ ``α = (MTT/CTH)²``,
        ``t_max = MTT`` — so the overlay matches what the model actually fit.
        """
        mtt = float(maps.get("mtt", float("nan")))
        cth = float(maps.get("cth", float("nan")))
        alpha = (mtt / cth) ** 2 if (np.isfinite(mtt) and np.isfinite(cth) and cth > 0) else 5.0
        fitter = _LarssonFitter(np.asarray(t_s, dtype=float), np.asarray(c_input, dtype=float))
        fit = GammaFit(
            cth_s=cth, mtt_cap_s=mtt, mtt_total_s=float("nan"),
            extraction=float(maps.get("extraction", 0.0) or 0.0),
            k2_per_s=float(maps.get("k2", 0.0) or 0.0),
            alpha=float(alpha), beta_s=float("nan"), dt_s=0.0, offset_s=0.0,
            f_ml_100g_min=float(maps.get("cbf", float("nan"))), sse=float("nan"),
        )
        return fitter.forward(fit, time_out=np.asarray(t_s, dtype=float))


PLUGIN = GammaModel()
