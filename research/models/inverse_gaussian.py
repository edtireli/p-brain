"""Inverse-Gaussian (Wald) first-passage residue + Patlak leak — local model.

A *derived* transit-time residue for the intravascular bolus, from the physics
of advection–diffusion rather than a phenomenological curve. Model a tracer
particle on a capillary path as 1-D Brownian motion with drift::

        dX = v·dt + √(2D)·dW,     X(0)=0,   absorbing venous end at X=L.

The transit time is the **first-passage time** of that process to L, which is
*exactly* the Inverse-Gaussian (Wald) law — a theorem, not an assumption:

        h(t) = √(λ / 2π t³) · exp( −λ (t−μ)² / (2 μ² t) ),
        μ = L/v   (mean transit time),     λ = L²/(2D)   (shape).

Its survival function R_IG(t)=P(T>t)=1−F_IG(t) is closed-form through the
standard-normal CDF Φ (evaluated stably with ``log Φ``).

**Vascular transit + leak.** A bare survival function decays to zero, so a
single IG cannot represent tracer that crosses the BBB and is *retained* over
the acquisition window — it would fake a slow tail by inflating dispersion.
We therefore use the physically correct residue

        R(t) = (1−E)·R_IG(t) + E ,        E ∈ [0,1),

with R(0)=1 preserved. The plateau ``E`` is the fraction trapped/leaked over
the window; in the forward model it contributes ``F·E·∫₀ᵗ C_a`` — exactly the
Patlak irreversible-uptake term. The IG part is then free to fit the first
pass, so the recovered moments are the *vascular* ones.

Outputs (all moments finite — contrast the Mittag-Leffler residue):

    cbf    = F·6000                         (mL/100 g/min; F = flow)
    mtt    = μ = L/v                         (vascular mean transit time, s)
    cth    = √(μ³/λ)                         (capillary transit-time heterogeneity, s)
    pe     = 2λ/μ = v·L/D                     (Péclet: advection/dispersion)
    e_leak = E                               (retained/leaked fraction over the window)

Relation to the field: the IG / convective-dispersion (Local Density Random
Walk) law is classical in indicator-dilution theory and is used for AIF
*dispersion*; using it as the **tissue residue** with a voxelwise Péclet readout
— the brain-DCE literature uses gamma-variate transit-time distributions — is
the under-explored part. ``.gitignore``-d (local only).

Fitting: ROI/parcel (1-D) by multi-restart TRF; voxelwise (default) by a
separable VARPRO dictionary over the (μ,λ) shape grid with F pinned to a
Tikhonov seed and E solved in closed form per voxel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.special import log_ndtr, ndtr

from .base import CurveInputs, ModelResult


H_FACTOR = 20   # convolution up-sampling factor. NB: A/B-tested 20 vs 40 and
                # linear vs PCHIP AIF — no effect on the fit or on CTH/Péclet:
                # the dispersive residue smooths away sub-resolution AIF detail,
                # so the CTH limit is informational (2.4 s acquisition), not
                # numerical. PCHIP kept as the (harmless, standard) default.


# ────────────────────────────────────────────────────────────────────────
# Inverse-Gaussian survival + density
# ────────────────────────────────────────────────────────────────────────


def ig_survival(t: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Survival R_IG(t)=1−F_IG(t) for IG(μ,λ): R(0)=1, monotone↓, →0.

    Stable closed form via Φ (``ndtr``) and ``log Φ`` (``log_ndtr``).
    """
    t = np.asarray(t, dtype=float)
    R = np.ones_like(t)
    pos = t > 0
    if not np.any(pos):
        return R
    tp = t[pos]
    st = np.sqrt(lam / tp)
    a = st * (tp / mu - 1.0)
    b = -st * (tp / mu + 1.0)
    term2 = np.exp(2.0 * lam / mu + log_ndtr(b))      # e^{2λ/μ}·Φ(b), overflow-safe
    R[pos] = 1.0 - ndtr(a) - term2
    return np.clip(R, 0.0, 1.0)


def ig_pdf(t: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Wald density h(t) (the transit-time distribution)."""
    t = np.asarray(t, dtype=float)
    h = np.zeros_like(t)
    pos = t > 0
    tp = t[pos]
    h[pos] = np.sqrt(lam / (2.0 * np.pi * tp ** 3)) * np.exp(
        -lam * (tp - mu) ** 2 / (2.0 * mu ** 2 * tp))
    return h


# ────────────────────────────────────────────────────────────────────────
# Parameter vector [F, mu, lam, E]   (μ=MTT[s], λ=shape[s], E=leak fraction)
# ────────────────────────────────────────────────────────────────────────

BOUNDS_LO = np.array([1e-5, 0.30, 0.20, 0.0])
BOUNDS_HI = np.array([400.0 / 6000.0, 30.0, 400.0, 0.95])
PARAM_NAMES = ("F", "mu", "lam", "E")


@dataclass(frozen=True, slots=True)
class InvGaussFit:
    """Single-curve inverse-Gaussian (Wald) residue fit.

    ``f`` is CBF (mL/100g/min); ``mtt`` is the Wald mean μ; ``cth`` is
    √(μ³/λ); ``pe`` is the Péclet number 2λ/μ; ``e_leak`` is the
    retained/leak fraction E; ``lam_s`` is the Wald shape λ; ``sse`` is
    the residual sum of squares.
    """

    f_ml_100g_min: float
    mtt_s: float          # μ
    cth_s: float          # √(μ³/λ)
    pe: float             # 2λ/μ
    e_leak: float         # E
    lam_s: float
    sse: float


# ────────────────────────────────────────────────────────────────────────
# Fitter
# ────────────────────────────────────────────────────────────────────────


class _InvGaussFitter:
    def __init__(self, time_s: np.ndarray, ca: np.ndarray, h_factor: int = H_FACTOR,
                 weight_flag: int = 1):
        time_s = np.asarray(time_s, dtype=float).ravel()
        ca = np.asarray(ca, dtype=float).ravel()
        n = int(min(time_s.size, ca.size))
        self.time_s = time_s[:n]
        self.ca_orig = ca[:n]
        self.n_orig = n
        self.h_factor = int(h_factor)

        nh = (n - 1) * self.h_factor + 1
        self.nh = nh
        self.dth = float(time_s[1] - time_s[0]) / self.h_factor
        self.time_h = np.linspace(0.0, float(time_s[n - 1]), nh)

        if weight_flag == 1:
            kw = -np.log(0.25) / 1800.0
            self.w_h = np.exp(-kw * self.time_h)
            self.w_ts = np.exp(-kw * self.time_s)
        else:
            self.w_h = np.ones(nh)
            self.w_ts = np.ones(self.n_orig)

        self.nfft = next_fast_len(2 * nh)
        # Shape-preserving (PCHIP) up-sampling of the AIF — monotone, no overshoot,
        # keeps the bolus peak sharper than linear interpolation onto the fine grid.
        ca_h = PchipInterpolator(self.time_s, self.ca_orig, extrapolate=True)(self.time_h)
        ca_h[-1] = ca_h[-2]
        self.ca_h = ca_h
        self.ca_h_fft = rfft(ca_h, n=self.nfft)
        # Patlak / leak basis: (C_a ⊗ 1)(t) = ∫₀ᵗ C_a  (cumulative trapezoid).
        cum = np.zeros(nh)
        cum[1:] = np.cumsum(0.5 * (ca_h[1:] + ca_h[:-1])) * self.dth
        self.ca_cum_h = cum

    def _conv_RIG(self, mu: float, lam: float) -> np.ndarray:
        R = ig_survival(self.time_h, mu, lam)
        return irfft(self.ca_h_fft * rfft(R, n=self.nfft), n=self.nfft)[: self.nh] * self.dth

    def _ct_calc(self, f: float, mu: float, lam: float, E: float) -> np.ndarray:
        """C_t = F[(1−E)(C_a⊗R_IG) + E·(C_a⊗1)]."""
        return f * ((1.0 - E) * self._conv_RIG(mu, lam) + E * self.ca_cum_h)

    def _interp_tissue(self, ct: np.ndarray) -> np.ndarray:
        ct_h = np.interp(self.time_h, self.time_s, ct[: self.n_orig])
        ct_h[-1] = ct_h[-2]
        return ct_h

    def _residuals(self, x: np.ndarray, ct_h: np.ndarray) -> np.ndarray:
        f, mu, lam, E = x[:4]
        if mu <= 1e-6 or lam <= 1e-6 or f <= 0.0:
            return np.full(self.nh, 1e8)
        return self.w_h * (self._ct_calc(f, mu, lam, E) - ct_h)

    def forward(self, fit: "InvGaussFit", time_out: np.ndarray | None = None) -> np.ndarray:
        f = float(fit.f_ml_100g_min) / 6000.0
        ct_h = self._ct_calc(f, float(fit.mtt_s), float(fit.lam_s), float(fit.e_leak))
        out_t = self.time_s if time_out is None else np.asarray(time_out, dtype=float)
        return np.interp(out_t, self.time_h, ct_h)

    # (mu, lam, E) starts bracketing plug-flow→dispersive × no-leak→leaky.
    _SHAPE_STARTS = (
        (3.0, 60.0, 0.0), (4.0, 20.0, 0.05), (5.0, 8.0, 0.10),
        (3.5, 40.0, 0.0), (6.0, 15.0, 0.20), (2.5, 100.0, 0.02),
    )

    def fit(self, ct: np.ndarray, f_cbf: float, n_restarts: int = 12,
            pin_f: bool = True, early_stop_patience: int = 2) -> InvGaussFit:
        ct_h = self._interp_tissue(np.asarray(ct, dtype=float).ravel())
        rng = np.random.default_rng(0)

        if pin_f:
            free_idx = np.array([1, 2, 3], dtype=int)     # mu, lam, E
            pinned = np.zeros(4); pinned[0] = float(max(f_cbf, 1e-6))
        else:
            free_idx = np.array([0, 1, 2, 3], dtype=int); pinned = np.zeros(4)
        lo, hi = BOUNDS_LO[free_idx].copy(), BOUNDS_HI[free_idx].copy()

        starts: list[np.ndarray] = []
        for s in self._SHAPE_STARTS:
            starts.append(np.array(s if pin_f else (float(np.clip(f_cbf, lo[0], hi[0])), *s)))
        while len(starts) < int(n_restarts):
            starts.append(lo + rng.random(len(free_idx)) * (hi - lo))

        best_x, best_sse, no_improve = None, np.inf, 0
        for x_free0 in starts[: max(int(n_restarts), len(self._SHAPE_STARTS))]:
            def residuals(x_free: np.ndarray) -> np.ndarray:
                x = pinned.copy(); x[free_idx] = x_free
                return self._residuals(x, ct_h)
            try:
                sol = least_squares(residuals, x_free0, bounds=(lo, hi),
                                    method="trf", max_nfev=250, ftol=1e-7, xtol=1e-7)
                sse = float(np.dot(sol.fun, sol.fun))
                if sse < best_sse * (1.0 - 5e-3):
                    best_sse, best_x, no_improve = sse, sol.x.copy(), 0
                else:
                    if sse < best_sse:
                        best_sse, best_x = sse, sol.x.copy()
                    no_improve += 1
                if best_x is not None and early_stop_patience > 0 and no_improve >= early_stop_patience:
                    break
            except Exception:                          # noqa: BLE001
                continue

        if best_x is None:
            return InvGaussFit(*([float("nan")] * 7))
        x_full = pinned.copy(); x_full[free_idx] = best_x
        f, mu, lam, E = x_full
        cth = float(math.sqrt(mu ** 3 / lam)) if (mu > 0 and lam > 0) else float("nan")
        pe = float(2.0 * lam / mu) if mu > 0 else float("nan")
        return InvGaussFit(f_ml_100g_min=float(f * 6000.0), mtt_s=float(mu), cth_s=cth,
                           pe=pe, e_leak=float(E), lam_s=float(lam), sse=float(best_sse))


# ────────────────────────────────────────────────────────────────────────
# Plug-in adapter
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InverseGaussianModel:
    """Inverse-Gaussian first-passage residue model (``inverse_gaussian``).

    Advection–diffusion residue ``R(t) = (1−E)·R_IG(t) + E``; reports
    CBF, vascular MTT (μ), CTH (√(μ³/λ)), Péclet (2λ/μ) and leak fraction
    E. Voxelwise via a VARPRO (μ,λ)-dictionary with F pinned to a
    Tikhonov seed and E in closed form. See :class:`InvGaussFit`.
    """

    key: ClassVar[str] = "inverse_gaussian"
    name: ClassVar[str] = "Inverse-Gaussian (Wald) first-passage residue + leak"
    description: ClassVar[str] = (
        "Advection–diffusion first-passage residue R(t)=(1−E)·R_IG(t)+E. "
        "Reports CBF, vascular MTT (μ), CTH (√(μ³/λ)), Péclet (2λ/μ) and the "
        "retained/leak fraction E. Voxelwise via a VARPRO (μ,λ)-dictionary "
        "with F pinned to a Tikhonov seed and E in closed form."
    )
    accepts: ClassVar[dict[str, type]] = {
        "c_tissue": np.ndarray, "c_input": np.ndarray, "t_s": np.ndarray, "f_cbf": float,
    }
    produces: ClassVar[dict[str, type]] = {
        "cbf": np.ndarray, "mtt": np.ndarray, "cth": np.ndarray,
        "pe": np.ndarray, "e_leak": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("cbf", "mtt", "cth", "pe", "e_leak")
    supports_voxelwise: ClassVar[bool] = True
    primary_map: ClassVar[str] = "pe"
    units: ClassVar[dict[str, str]] = {
        "cbf": "mL/100g/min", "mtt": "s", "cth": "s", "pe": "(Péclet)", "e_leak": "(fraction)",
    }

    def _seed_f(self, t_s, c_a, curves_2d, f_cbf0):
        from pbrain.models.tikhonov import build_tikhonov_solver
        solver = build_tikhonov_solver(
            t_s, c_a, lambda_selection="evidence", lambda_spacing="log",
            lambda_min=1e-3, mtt_cth_method="residue_integral")
        cbf = np.asarray(solver(curves_2d, offsets_s=None).cbf_ml_per_100g_min, dtype=float)
        f = cbf / 6000.0
        f[~np.isfinite(f) | (f <= 0)] = f_cbf0
        return f

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        """Fit the inverse-Gaussian residue → CBF/MTT/CTH/Pe/leak maps.

        Seeds CBF from a Tikhonov pass, then solves the (μ,λ) dictionary
        per voxel. ``**opts`` tune the dictionary grid and restarts.
        """
        c_t = np.asarray(inputs.c_tissue, dtype=float)
        c_a = np.asarray(inputs.c_input, dtype=float)
        t_s = np.asarray(inputs.t_s, dtype=float)

        n_restarts = int(opts.pop("n_restarts", 12))
        pin_f = bool(opts.pop("pin_cbf", opts.pop("pin_f", True)))
        early_stop = int(opts.pop("early_stop_patience", 2))
        f_cbf0 = float(opts.pop("f_cbf_initial", 50.0 / 6000.0))
        f_cbf_in = opts.pop("f_cbf", None)
        fitter = _InvGaussFitter(t_s, c_a, h_factor=int(opts.pop("h_factor", H_FACTOR)))

        def _maps(fit: InvGaussFit) -> dict[str, np.ndarray]:
            return {"cbf": np.asarray(fit.f_ml_100g_min), "mtt": np.asarray(fit.mtt_s),
                    "cth": np.asarray(fit.cth_s), "pe": np.asarray(fit.pe),
                    "e_leak": np.asarray(fit.e_leak)}

        if c_t.ndim == 1:
            f_seed = (float(f_cbf_in) if (pin_f and f_cbf_in is not None)
                      else float(self._seed_f(t_s, c_a, c_t.reshape(-1, 1), f_cbf0)[0]) if pin_f
                      else f_cbf0)
            fit = fitter.fit(c_t, f_cbf=f_seed, n_restarts=n_restarts, pin_f=pin_f,
                             early_stop_patience=early_stop)
            return ModelResult(maps=_maps(fit), units=dict(self.units),
                               aux={"sse": float(fit.sse), "lam_s": float(fit.lam_s)})

        if c_t.ndim != 2:
            raise ValueError(f"c_tissue must be 1-D or 2-D; got shape {c_t.shape}")
        n_t, n_v = c_t.shape
        voxels = (np.arange(n_v) if inputs.mask is None
                  else np.flatnonzero(np.asarray(inputs.mask, dtype=bool)))
        method = str(opts.pop("method", "dictionary")).lower()
        if method == "dictionary":
            return self._fit_dictionary(fitter, c_t, voxels, n_v, opts,
                                        pin_f=pin_f, f_cbf0=f_cbf0, f_cbf_in=f_cbf_in)

        out = {k: np.full(n_v, np.nan) for k in self.outputs}
        if pin_f and f_cbf_in is None:
            f_seeds = np.full(n_v, f_cbf0)
            if voxels.size:
                f_seeds[voxels] = self._seed_f(t_s, c_a, c_t[:, voxels], f_cbf0)
        elif pin_f:
            fa = np.asarray(f_cbf_in, dtype=float).reshape(-1)
            f_seeds = fa if fa.size == n_v else np.full(n_v, float(fa.flat[0]))
        else:
            f_seeds = np.full(n_v, f_cbf0)
        for v in voxels:
            v = int(v)
            fit = fitter.fit(c_t[:, v], f_cbf=float(f_seeds[v]), n_restarts=n_restarts,
                             pin_f=pin_f, early_stop_patience=early_stop)
            for k, val in _maps(fit).items():
                out[k][v] = float(val)
        return ModelResult(maps=out, units=dict(self.units), aux={})

    def _fit_dictionary(self, fitter: "_InvGaussFitter", c_t: np.ndarray,
                        voxels: np.ndarray, n_v: int, opts: dict, *,
                        pin_f: bool = True, f_cbf0: float = 50.0 / 6000.0,
                        f_cbf_in=None) -> ModelResult:
        """Voxelwise VARPRO over the (μ,λ) grid with F pinned and E closed-form.

        For a fixed (μ,λ) and pinned F, the curve is linear in the leak
        fraction E:  ``y ≈ F·b_IG + F·E·(b_cum − b_IG)``. So per voxel
        ``E* = clip( ⟨y − F b_IG, d⟩ / (F‖d‖²), 0, 0.95 )`` with d=b_cum−b_IG,
        and the residual SSE is evaluated in closed form from pre-computed dot
        products — all vectorised across voxels. Best (μ,λ) = argmin_g SSE.
        """
        nmu = int(opts.pop("mu_grid_n", 32)); nlam = int(opts.pop("lam_grid_n", 36))
        mu_g = np.geomspace(float(opts.pop("mu_min", 0.5)), float(opts.pop("mu_max", 25.0)), nmu)
        lam_g = np.geomspace(float(opts.pop("lam_min", 0.3)), float(opts.pop("lam_max", 400.0)), nlam)

        w = fitter.w_ts
        nfft, nh, dth = fitter.nfft, fitter.nh, fitter.dth
        bcum = w * np.interp(fitter.time_s, fitter.time_h, fitter.ca_cum_h)   # weighted leak basis
        Bw, muf, lamf, cthf, pef = [], [], [], [], []
        for mu in mu_g:
            for lam in lam_g:
                R = ig_survival(fitter.time_h, mu, lam)
                ct_h = irfft(fitter.ca_h_fft * rfft(R, n=nfft), n=nfft)[:nh] * dth
                Bw.append(w * np.interp(fitter.time_s, fitter.time_h, ct_h))
                muf.append(mu); lamf.append(lam)
                cthf.append(math.sqrt(mu ** 3 / lam)); pef.append(2.0 * lam / mu)
        Bw = np.asarray(Bw)                       # (G, T)  vascular bases (weighted, unit F)
        muf = np.asarray(muf); lamf = np.asarray(lamf)
        cthf = np.asarray(cthf); pef = np.asarray(pef)
        D = bcum[None, :] - Bw                    # (G, T)  leak-minus-vascular directions
        BB = np.einsum("gt,gt->g", Bw, Bw)        # ‖b_IG‖²
        DD = np.maximum(np.einsum("gt,gt->g", D, D), 1e-30)   # ‖d‖²
        BD = np.einsum("gt,gt->g", Bw, D)         # ⟨b_IG, d⟩

        f_seeds = None
        if pin_f:
            if f_cbf_in is not None:
                fa = np.asarray(f_cbf_in, dtype=float).reshape(-1)
                f_seeds = fa if fa.size == n_v else np.full(n_v, float(fa.flat[0]))
            else:
                f_seeds = np.full(n_v, f_cbf0)
                if voxels.size:
                    f_seeds[voxels] = self._seed_f(fitter.time_s, fitter.ca_orig,
                                                   c_t[:, voxels], f_cbf0)

        out = {k: np.full(n_v, np.nan) for k in self.outputs}
        if voxels.size:
            Yw = c_t[:, voxels] * w[:, None]      # (T, nk)
            CHUNK = 2048
            for s0 in range(0, voxels.size, CHUNK):
                sl = slice(s0, min(s0 + CHUNK, voxels.size)); vidx = voxels[sl]
                F = f_seeds[vidx] if pin_f else np.full(vidx.size, f_cbf0)
                F = np.maximum(F, 1e-9)
                Y2 = np.einsum("tk,tk->k", Yw[:, sl], Yw[:, sl])            # ‖y‖² (nk,)
                YB = Bw @ Yw[:, sl]                                          # ⟨y,b_IG⟩ (G,nk)
                YD = D @ Yw[:, sl]                                           # ⟨y,d⟩    (G,nk)
                # closed-form E per (g, voxel):  E = (YD − F·BD)/(F·DD), clipped
                E = (YD - F[None, :] * BD[:, None]) / (F[None, :] * DD[:, None])
                E = np.clip(E, 0.0, 0.95)
                # SSE(g,v) = ‖y − F b_IG − F E d‖²  (expanded from dot products)
                SSE = (Y2[None, :]
                       + (F ** 2)[None, :] * BB[:, None]
                       + (F ** 2)[None, :] * (E ** 2) * DD[:, None]
                       - 2.0 * F[None, :] * YB
                       - 2.0 * F[None, :] * E * YD
                       + 2.0 * (F ** 2)[None, :] * E * BD[:, None])
                best = np.argmin(SSE, axis=0)
                cols = np.arange(vidx.size)
                ok = np.isfinite(F) & (F > 0)
                vi = vidx[ok]; bs = best[ok]
                out["cbf"][vi] = F[ok] * 6000.0
                out["mtt"][vi] = muf[bs]; out["cth"][vi] = cthf[bs]; out["pe"][vi] = pef[bs]
                out["e_leak"][vi] = E[bs, cols][ok]
        return ModelResult(maps=out, units=dict(self.units),
                           aux={"method": "dictionary", "pin_f": bool(pin_f),
                                "grid_mu": nmu, "grid_lam": nlam})

    def review(self, inputs, result, **_kw):
        """--mode verify view — mean tissue Cₜ + the inverse-Gaussian fit + medians."""
        from ._review import curve_fit_review
        return curve_fit_review(self, inputs, result,
                                title="Inverse-Gaussian · transit", fit_label="IG fit",
                                show=("cbf", "mtt", "cth"))

    def predict(self, maps: dict[str, float], c_input: np.ndarray, t_s: np.ndarray) -> np.ndarray:
        """Reconstruct Cₜ from one curve's IG residue parameters."""
        fitter = _InvGaussFitter(t_s, c_input)
        mu = float(maps.get("mtt", float("nan"))); cth = float(maps.get("cth", float("nan")))
        lam = (mu ** 3 / cth ** 2) if (np.isfinite(cth) and cth > 0) else float("nan")
        fit = InvGaussFit(float(maps.get("cbf", float("nan"))), mu, cth,
                          float(maps.get("pe", float("nan"))), float(maps.get("e_leak", 0.0)),
                          lam, float("nan"))
        return fitter.forward(fit, time_out=t_s)


PLUGIN = InverseGaussianModel()
