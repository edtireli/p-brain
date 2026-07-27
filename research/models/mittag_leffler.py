"""Fractional (Mittag-Leffler) transit/clearance model — local-only kinetic model.

Anomalous-transport residue for a geometrically disordered microvascular bed.
The integer-order one-compartment mass balance ``q' = -k q`` (which gives the
mono-exponential Tofts residue ``R = e^{-kt}``) is replaced by a *fractional*
Caputo balance of order ``α ∈ (0, 1]``::

        D_t^α q(t) = -(1/τ)^α q(t)      ⟹      R(t) = E_α(-(t/τ)^α),

where ``E_α`` is the one-parameter Mittag-Leffler function. This is the
continuum limit of a continuous-time random walk with heavy-tailed waiting
times (Montroll–Weiss / subordinated semigroup), i.e. tracer that is
transiently *trapped* in a disordered bed rather than cleared at a single rate.

Behaviour::

    α = 1            R(t) = e^{-t/τ}                 (recovers one-compartment Tofts)
    α < 1, t ≪ τ     R(t) ≈ exp(-(t/τ)^α/Γ(1+α))     (stretched exponential, fast early decay)
    α < 1, t ≫ τ     R(t) ≈ (t/τ)^{-α}/Γ(1-α)         (algebraic tail — slow retention/leak)

The order ``α`` is the headline biomarker: a *disorder / anomaly exponent*.
``α → 1`` is a healthy, well-mixed bed cleared at one rate; ``α < 1`` is a
trapping, heterogeneous bed whose slow power-law tail is a parsimonious model
of the late, low-amplitude retention the chapter attributes to leakage — one
extra parameter instead of a whole second compartment.

**Numerics.** ``E_α(-x)`` for ``x ≥ 0, 0 < α < 1`` is *completely monotone*,
and by Bernstein's theorem is the Laplace transform of a positive spectral
density (Pollard 1948)::

    E_α(-x) = ∫₀^∞ e^{-x r} K_α(r) dr,
    K_α(r) = (1/π) · sin(απ) · r^{α-1} / (r^{2α} + 2 r^α cos(απ) + 1).

We evaluate the integral by fixed log-spaced quadrature in ``r`` — so the ML
residue is literally computed as a positive *mixture of exponential
clearances*, the same object Bernstein's theorem puts at the centre of the
"rate-spectrum" view. Validated against the two closed forms
``E_1(-x) = e^{-x}`` and ``E_{1/2}(-x) = e^{x²}·erfc(x)``.

Derived outputs::

    cbf       F · 6000                       (mL/100 g/min; initial impulse height)
    alpha     α                              (anomaly exponent, dimensionless)
    tau       τ                              (fractional clearance time, s)
    tmed      τ · s½,  E_α(-s½^α)=½           (median transit time — finite even when
                                              the mean diverges for α<1; robust)
    mtt_win   ∫₀^{T_acq} R(t) dt              (windowed mean transit time; the true
                                              MTT = ∫₀^∞ R diverges for α<1, so we
                                              report the honest finite-window integral)

Two fitting modes:

* **ROI / parcel (1-D curve)** — multi-restart TRF, F pinned to a Tikhonov CBF
  seed to break the F-vs-shape degeneracy (mirrors ``gamma.py``). Accurate.
* **Voxelwise (2-D, the default for whole-brain)** — a separable / variable-
  projection (VARPRO) *dictionary* fit: the two nonlinear shape parameters
  (α, τ) are gridded, the linear amplitude F is the closed-form least-squares
  scale at each grid point, and every voxel is matched by one vectorised
  mat-mul (MR-fingerprinting-style). ~10⁵–10⁶ voxels in seconds. F is solved
  per voxel (no Tikhonov pin), so voxelwise CBF is the model's own estimate.

**Note on git:** this file is ``.gitignore``-d so it never ships on the public
``main`` branch — discovery treats it exactly like ``gamma.py`` / ``patlak.py``:
present on disk → the registry picks it up; absent on a fresh clone → it simply
isn't there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.optimize import least_squares

from pbrain._numpy_compat import trapezoid
from .base import CurveInputs, ModelResult


H_FACTOR = 20  # Up-sampling factor for the convolution grid (residue is smooth;
               # 20× resolves the sharp AIF peak without gamma's 40× cost).


# ────────────────────────────────────────────────────────────────────────
# Mittag-Leffler E_α(-x), x ≥ 0, via the Bernstein/Pollard spectral integral.
# Precompute the quadrature once at import; m_α(r) is rebuilt per α (cheap),
# and the exp kernel e^{-x r} is α-independent so it is precomputed too.
# ────────────────────────────────────────────────────────────────────────

_NQ = 1000                                           # spectral quadrature nodes
_U = np.linspace(-24.0, 24.0, _NQ)                   # u = ln r
_R_NODES = np.exp(_U)                                # r ∈ [e^-24, e^24]
_DU = float(_U[1] - _U[0])

# x-grid on which E_α(-x) is tabulated, then interpolated to the time grid.
# Dense near 0 (where R bends sharply), log-stretched into the tail.
_X_GRID = np.concatenate([
    np.linspace(0.0, 5.0, 220),
    np.geomspace(5.0 + 1e-6, 5.0e3, 220),
])
# α-independent exp kernel:  _EXP0[i, k] = exp(-x_i · r_k).   (n_x, n_q)
_EXP0 = np.exp(-np.outer(_X_GRID, _R_NODES))


def _ml_spectral_weights(alpha: float) -> np.ndarray:
    """Quadrature weights m_α(r_k)·Δu for the spectral integral of E_α(-x).

    ``E_α(-x) = Σ_k m_α(r_k) · e^{-x r_k}`` with the integration measure
    ``dr = r du`` folded in (so the bare r^{α-1} becomes r^{α}).
    """
    a = float(alpha)
    ra = _R_NODES ** a
    denom = _R_NODES ** (2.0 * a) + 2.0 * ra * math.cos(a * math.pi) + 1.0
    # numerator: (1/π) sin(απ) r^{α-1} · r (=r^α from dr=r·du)
    num = (math.sin(a * math.pi) / math.pi) * (_R_NODES ** a)
    w = (num / denom) * _DU
    # Enforce the exact zeroth moment  E_α(0) = ∫ K_α dr = 1  (⇒ R(0)=1),
    # which also removes the dominant quadrature bias at the same time.
    s = float(w.sum())
    return w / s if s > 0 else w


def _ml_relax_grid(alpha: float) -> np.ndarray:
    """Tabulate the relaxation function G_α(s) = E_α(-s^α) on ``_X_GRID``.

    The Gorenflo–Mainardi spectral integral ``∫ e^{-s r} K_α(r) dr`` evaluates
    to ``E_α(-s^α)`` (NOT ``E_α(-s)``) — i.e. the kernel already carries the
    ``^α``. The residue is then ``R(t) = E_α(-(t/τ)^α) = G_α(t/τ)``: just
    sample this grid at ``s = t/τ``.
    """
    if alpha >= 0.9995:                              # G_1(s) = E_1(-s) = e^{-s}
        return np.exp(-_X_GRID)
    return _EXP0 @ _ml_spectral_weights(alpha)


def ml_relaxation(s: np.ndarray, alpha: float) -> np.ndarray:
    """G_α(s) = E_α(-s^α), the fractional relaxation function, for s ≥ 0.

    This is the residue shape with s = t/τ. Closed-form checks:
    ``α=1 → e^{-s}``; ``α=½ → e^{s}·erfc(√s)``.
    """
    s = np.asarray(s, dtype=float)
    if alpha >= 0.9995:
        return np.exp(-np.maximum(s, 0.0))
    w = _ml_spectral_weights(alpha)
    return (np.exp(-np.outer(np.maximum(s, 0.0).ravel(), _R_NODES)) @ w).reshape(s.shape)


def _s_half(alpha: float) -> float:
    """s½ such that G_α(s½) = E_α(-s½^α) = ½  →  median transit time = τ·s½."""
    g = _ml_relax_grid(alpha)
    # g is monotone decreasing in s → invert by interpolation on reversed arrays
    return float(np.interp(0.5, g[::-1], _X_GRID[::-1]))


# ────────────────────────────────────────────────────────────────────────
# Parameter vector [F, tau, alpha]
# ────────────────────────────────────────────────────────────────────────

BOUNDS_LO = np.array([1e-5, 0.10, 0.30])
BOUNDS_HI = np.array([400.0 / 6000.0, 60.0, 1.00])
PARAM_NAMES = ("F", "tau", "alpha")


@dataclass(frozen=True, slots=True)
class MittagLefflerFit:
    """Single-curve fractional Mittag-Leffler residue fit.

    ``f`` is CBF (mL/100g/min); ``alpha`` is the anomaly exponent;
    ``tau_s`` the fractional time constant; ``tmed_s`` the median transit
    time; ``mtt_win_s`` the windowed mean transit time; ``sse`` the
    residual sum of squares.
    """

    f_ml_100g_min: float
    alpha: float
    tau_s: float
    tmed_s: float
    mtt_win_s: float
    sse: float


# ────────────────────────────────────────────────────────────────────────
# Fitter (FFT convolution, multi-restart TRF) — mirrors gamma._LarssonFitter
# ────────────────────────────────────────────────────────────────────────


class _MittagLefflerFitter:
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
        self.t_acq = float(time_s[n - 1])

        if weight_flag == 1:                         # gamma's late-time down-weight
            kw = -np.log(0.25) / 1800.0
            self.w_h = np.exp(-kw * self.time_h)
            self.w_ts = np.exp(-kw * self.time_s)
        else:
            self.w_h = np.ones(nh)
            self.w_ts = np.ones(self.n_orig)

        self.nfft = next_fast_len(2 * nh)
        ca_h = np.interp(self.time_h, self.time_s, self.ca_orig)
        ca_h[-1] = ca_h[-2]
        self.ca_h_fft = rfft(ca_h, n=self.nfft)

    def _residue(self, tau: float, alpha: float) -> np.ndarray:
        """R(t) = E_α(-(t/τ)^α) = G_α(t/τ) on the up-sampled time grid."""
        g = _ml_relax_grid(alpha)
        s = self.time_h / tau
        R = np.interp(s, _X_GRID, g, left=1.0, right=g[-1])
        R[0] = 1.0
        return np.clip(R, 0.0, 1.0)

    def _interp_tissue(self, ct: np.ndarray) -> np.ndarray:
        ct_h = np.interp(self.time_h, self.time_s, ct[: self.n_orig])
        ct_h[-1] = ct_h[-2]
        return ct_h

    def _ct_calc(self, f: float, tau: float, alpha: float) -> np.ndarray:
        R = self._residue(tau, alpha)
        rif_fft = rfft(R, n=self.nfft)
        return irfft(self.ca_h_fft * rif_fft, n=self.nfft)[: self.nh] * self.dth * f

    def _residuals(self, x: np.ndarray, ct_h: np.ndarray) -> np.ndarray:
        f, tau, alpha = x[:3]
        if tau <= 1e-9 or alpha <= 0.0 or f <= 0.0:
            return np.full(self.nh, 1e8)
        return self.w_h * (self._ct_calc(f, tau, alpha) - ct_h)

    def forward(self, fit: "MittagLefflerFit",
                time_out: np.ndarray | None = None) -> np.ndarray:
        f = float(fit.f_ml_100g_min) / 6000.0
        ct_h = self._ct_calc(f, float(fit.tau_s), float(fit.alpha))
        out_t = self.time_s if time_out is None else np.asarray(time_out, dtype=float)
        return np.interp(out_t, self.time_h, ct_h)

    # (tau, alpha) starts bracketing realistic capillary clearance + disorder.
    _SHAPE_STARTS = (
        (5.0, 0.90), (5.0, 0.70), (3.0, 0.80),
        (10.0, 0.60), (8.0, 0.95), (2.0, 0.50),
    )

    def fit(self, ct: np.ndarray, f_cbf: float, n_restarts: int = 12,
            pin_f: bool = True, early_stop_patience: int = 2) -> MittagLefflerFit:
        ct_h = self._interp_tissue(np.asarray(ct, dtype=float).ravel())
        rng = np.random.default_rng(0)

        if pin_f:
            free_idx = np.array([1, 2], dtype=int)   # tau, alpha
            pinned = np.zeros(3)
            pinned[0] = float(max(f_cbf, 1e-6))
        else:
            free_idx = np.array([0, 1, 2], dtype=int)
            pinned = np.zeros(3)

        lo, hi = BOUNDS_LO[free_idx].copy(), BOUNDS_HI[free_idx].copy()

        starts: list[np.ndarray] = []
        for s in self._SHAPE_STARTS:
            starts.append(np.array(s if pin_f
                                   else (float(np.clip(f_cbf, lo[0], hi[0])), *s)))
        while len(starts) < int(n_restarts):
            starts.append(lo + rng.random(len(free_idx)) * (hi - lo))

        best_x: np.ndarray | None = None
        best_sse = np.inf
        no_improve = 0

        for x_free0 in starts[: max(int(n_restarts), len(self._SHAPE_STARTS))]:
            def residuals(x_free: np.ndarray) -> np.ndarray:
                x = pinned.copy()
                x[free_idx] = x_free
                return self._residuals(x, ct_h)

            try:
                sol = least_squares(residuals, x_free0, bounds=(lo, hi),
                                    method="trf", max_nfev=200, ftol=1e-7, xtol=1e-7)
                sse = float(np.dot(sol.fun, sol.fun))
                if sse < best_sse * (1.0 - 5e-3):
                    best_sse, best_x, no_improve = sse, sol.x.copy(), 0
                else:
                    if sse < best_sse:
                        best_sse, best_x = sse, sol.x.copy()
                    no_improve += 1
                if best_x is not None and early_stop_patience > 0 \
                        and no_improve >= early_stop_patience:
                    break
            except Exception:                        # noqa: BLE001
                continue

        if best_x is None:
            return MittagLefflerFit(*([float("nan")] * 6))

        x_full = pinned.copy()
        x_full[free_idx] = best_x
        f, tau, alpha = x_full

        tmed = float(tau * _s_half(alpha)) if alpha > 0 else float("nan")
        R_acq = self._residue(tau, alpha)
        mtt_win = float(trapezoid(R_acq, self.time_h))

        return MittagLefflerFit(
            f_ml_100g_min=float(f * 6000.0),
            alpha=float(alpha),
            tau_s=float(tau),
            tmed_s=tmed,
            mtt_win_s=mtt_win,
            sse=float(best_sse),
        )


# ────────────────────────────────────────────────────────────────────────
# Plug-in adapter
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MittagLefflerModel:
    """Fractional Mittag-Leffler residue model (the ``mittag_leffler`` plug-in).

    Anomalous-transport residue ``R(t) = E_α(−(t/τ)^α)`` from a
    fractional (Caputo) one-compartment balance. Reports the anomaly
    exponent α, fractional time τ, CBF, median transit time and windowed
    MTT, with F pinned to a Tikhonov CBF seed. See :class:`MittagLefflerFit`.
    """

    key: ClassVar[str] = "mittag_leffler"
    name: ClassVar[str] = "Fractional Mittag-Leffler residue"
    description: ClassVar[str] = (
        "Anomalous-transport residue R(t)=E_α(-(t/τ)^α) from a fractional "
        "(Caputo) one-compartment balance. Reports the anomaly exponent α, "
        "fractional time τ, CBF, median transit time, and windowed MTT. "
        "F pinned to a Tikhonov CBF seed; parcel-level multi-restart TRF."
    )
    accepts: ClassVar[dict[str, type]] = {
        "c_tissue": np.ndarray,
        "c_input": np.ndarray,
        "t_s": np.ndarray,
        "f_cbf": float,          # signals the parcel path to pass a CBF pin
    }
    produces: ClassVar[dict[str, type]] = {
        "cbf": np.ndarray, "alpha": np.ndarray, "tau": np.ndarray,
        "tmed": np.ndarray, "mtt_win": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("cbf", "alpha", "tau", "tmed", "mtt_win")
    supports_voxelwise: ClassVar[bool] = True   # via the VARPRO dictionary (fit 2-D path)
    primary_map: ClassVar[str] = "alpha"
    units: ClassVar[dict[str, str]] = {
        "cbf": "mL/100g/min", "alpha": "(fraction)", "tau": "s",
        "tmed": "s", "mtt_win": "s",
    }

    def _seed_f(self, t_s, c_a, curves_2d, f_cbf0):
        """Per-curve F (1/s) for pinning, from a fast Tikhonov CBF pass."""
        from pbrain.models.tikhonov import build_tikhonov_solver
        solver = build_tikhonov_solver(
            t_s, c_a, lambda_selection="evidence", lambda_spacing="log",
            lambda_min=1e-3, mtt_cth_method="residue_integral",
        )
        cbf = np.asarray(solver(curves_2d, offsets_s=None).cbf_ml_per_100g_min, dtype=float)
        f = cbf / 6000.0
        f[~np.isfinite(f) | (f <= 0)] = f_cbf0
        return f

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        """Fit the Mittag-Leffler residue → α/τ/CBF/MTT maps.

        Seeds CBF from a Tikhonov pass, then runs a multi-restart TRF fit
        of (α, τ) per curve. ``**opts`` tune restarts and bounds.
        """
        c_t = np.asarray(inputs.c_tissue, dtype=float)
        c_a = np.asarray(inputs.c_input, dtype=float)
        t_s = np.asarray(inputs.t_s, dtype=float)

        n_restarts = int(opts.pop("n_restarts", 12))
        pin_f = bool(opts.pop("pin_cbf", opts.pop("pin_f", True)))
        early_stop = int(opts.pop("early_stop_patience", 2))
        f_cbf0 = float(opts.pop("f_cbf_initial", 50.0 / 6000.0))
        f_cbf_in = opts.pop("f_cbf", None)
        fitter = _MittagLefflerFitter(t_s, c_a, h_factor=int(opts.pop("h_factor", H_FACTOR)))

        def _maps(fit: MittagLefflerFit) -> dict[str, np.ndarray]:
            return {
                "cbf": np.asarray(fit.f_ml_100g_min),
                "alpha": np.asarray(fit.alpha),
                "tau": np.asarray(fit.tau_s),
                "tmed": np.asarray(fit.tmed_s),
                "mtt_win": np.asarray(fit.mtt_win_s),
            }

        if c_t.ndim == 1:
            if pin_f:
                f_seed = (float(f_cbf_in) if f_cbf_in is not None
                          else float(self._seed_f(t_s, c_a, c_t.reshape(-1, 1), f_cbf0)[0]))
            else:
                f_seed = f_cbf0
            fit = fitter.fit(c_t, f_cbf=f_seed, n_restarts=n_restarts,
                             pin_f=pin_f, early_stop_patience=early_stop)
            return ModelResult(maps=_maps(fit), units=dict(self.units),
                               aux={"sse": float(fit.sse)})

        if c_t.ndim != 2:
            raise ValueError(f"c_tissue must be 1-D or 2-D; got shape {c_t.shape}")

        n_t, n_v = c_t.shape
        voxels = (np.arange(n_v) if inputs.mask is None
                  else np.flatnonzero(np.asarray(inputs.mask, dtype=bool)))
        method = str(opts.pop("method", "dictionary")).lower()

        if method == "dictionary":
            return self._fit_dictionary(fitter, c_t, voxels, n_v, opts,
                                        pin_f=pin_f, f_cbf0=f_cbf0, f_cbf_in=f_cbf_in)

        # ---- exact per-voxel multi-restart TRF (method="trf"; slow, accurate) ----
        out = {k: np.full(n_v, np.nan) for k in self.outputs}
        if pin_f:
            if f_cbf_in is not None:
                fa = np.asarray(f_cbf_in, dtype=float).reshape(-1)
                f_seeds = fa if fa.size == n_v else np.full(n_v, float(fa.flat[0]))
            else:
                f_seeds = np.full(n_v, f_cbf0)
                if len(voxels) > 0:
                    f_seeds[voxels] = self._seed_f(t_s, c_a, c_t[:, voxels], f_cbf0)
        else:
            f_seeds = np.full(n_v, f_cbf0)

        for v in voxels:
            v = int(v)
            fit = fitter.fit(c_t[:, v], f_cbf=float(f_seeds[v]), n_restarts=n_restarts,
                             pin_f=pin_f, early_stop_patience=early_stop)
            for k, val in _maps(fit).items():
                out[k][v] = float(val)

        return ModelResult(maps=out, units=dict(self.units), aux={})

    def _fit_dictionary(self, fitter: "_MittagLefflerFitter", c_t: np.ndarray,
                        voxels: np.ndarray, n_v: int, opts: dict, *,
                        pin_f: bool = True, f_cbf0: float = 50.0 / 6000.0,
                        f_cbf_in=None) -> ModelResult:
        """Voxelwise separable dictionary fit.

        Grid (α, τ); each grid point's unit-flow forward curve is
        ``b_g = C_a ⊗ R(·; α_g, τ_g)``, weighted like the TRF path.

        * ``pin_f`` (default): F is pinned per voxel to a fast batched Tikhonov
          CBF seed (F = CBF/6000) — the same two-stage trick the parcel path
          uses to break the F-vs-shape degeneracy that otherwise balloons F and
          collapses τ to the grid floor. Best shape = ``argmin_g ‖F·b_g − y‖²``.
        * else: full variable projection — F is the closed-form LS scale per
          grid point, ``(α,τ,F) = argmax_g ⟨b_g,y⟩²/‖b_g‖²``.

        One vectorised mat-mul per voxel-chunk → ~10⁵–10⁶ voxels in seconds.
        """
        na = int(opts.pop("alpha_grid_n", 40))
        nt = int(opts.pop("tau_grid_n", 44))
        a_lo = float(opts.pop("alpha_min", 0.35)); a_hi = float(opts.pop("alpha_max", 1.0))
        t_lo = float(opts.pop("tau_min", 0.5));    t_hi = float(opts.pop("tau_max", 30.0))
        alpha_g = np.linspace(a_lo, a_hi, na)
        tau_g = np.geomspace(t_lo, t_hi, nt)

        w = fitter.w_ts
        nfft, nh, dth = fitter.nfft, fitter.nh, fitter.dth
        rows, af, tf, mttwin, tmed = [], [], [], [], []
        for a in alpha_g:
            g_relax = _ml_relax_grid(float(a))       # E_α grid: once per α
            s_half = _s_half(float(a))
            for tau in tau_g:
                R = np.interp(fitter.time_h / tau, _X_GRID, g_relax, left=1.0, right=g_relax[-1])
                R[0] = 1.0; R = np.clip(R, 0.0, 1.0)
                ct_h = irfft(fitter.ca_h_fft * rfft(R, n=nfft), n=nfft)[:nh] * dth
                rows.append(w * np.interp(fitter.time_s, fitter.time_h, ct_h))
                mttwin.append(float(trapezoid(R, fitter.time_h)))
                tmed.append(float(tau * s_half))
                af.append(float(a)); tf.append(float(tau))
        Bw = np.asarray(rows)                          # (G, T)
        af = np.asarray(af); tf = np.asarray(tf)
        mttwin = np.asarray(mttwin); tmed = np.asarray(tmed)
        Bw2 = np.maximum(np.einsum("gt,gt->g", Bw, Bw), 1e-30)

        # Per-voxel pinned flow F (1/s) from a batched Tikhonov CBF pass.
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
            Yw = c_t[:, voxels] * w[:, None]           # (T, n_keep)
            CHUNK = 4096
            for s0 in range(0, voxels.size, CHUNK):
                sl = slice(s0, min(s0 + CHUNK, voxels.size))
                vidx = voxels[sl]
                Gm = Bw @ Yw[:, sl]                    # (G, chunk) = ⟨b_g, y⟩ weighted
                if pin_f:
                    Fv = f_seeds[vidx]
                    best = np.argmin((Fv[None, :] ** 2) * Bw2[:, None]
                                     - 2.0 * Fv[None, :] * Gm, axis=0)
                else:
                    best = np.argmax((Gm * Gm) / Bw2[:, None], axis=0)
                    Fv = Gm[best, np.arange(Gm.shape[1])] / Bw2[best]
                ok = np.isfinite(Fv) & (Fv > 0)
                vi, bs = vidx[ok], best[ok]
                out["cbf"][vi] = Fv[ok] * 6000.0
                out["alpha"][vi] = af[bs]; out["tau"][vi] = tf[bs]
                out["tmed"][vi] = tmed[bs]; out["mtt_win"][vi] = mttwin[bs]
        return ModelResult(maps=out, units=dict(self.units),
                           aux={"method": "dictionary", "pin_f": bool(pin_f),
                                "grid_alpha": na, "grid_tau": nt})

    def review(self, inputs, result, **_kw):
        """--mode verify view — mean tissue Cₜ + the Mittag-Leffler fit + medians."""
        from ._review import curve_fit_review
        return curve_fit_review(self, inputs, result,
                                title="Mittag-Leffler · anomalous transit", fit_label="ML fit")

    def predict(self, maps: dict[str, float], c_input: np.ndarray,
                t_s: np.ndarray) -> np.ndarray:
        """Reconstruct C_t for the generic diagnostic overlay."""
        fitter = _MittagLefflerFitter(t_s, c_input)
        fit = MittagLefflerFit(
            f_ml_100g_min=float(maps.get("cbf", float("nan"))),
            alpha=float(maps.get("alpha", float("nan"))),
            tau_s=float(maps.get("tau", float("nan"))),
            tmed_s=float(maps.get("tmed", float("nan"))),
            mtt_win_s=float(maps.get("mtt_win", float("nan"))),
            sse=float("nan"),
        )
        return fitter.forward(fit, time_out=t_s)


PLUGIN = MittagLefflerModel()
