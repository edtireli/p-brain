"""Patlak graphical analysis — paper §4.6.1, Eq. 3.

Fits the linear Patlak model to a tissue concentration-time curve under
the assumption of negligible backflux (k₂ ≈ 0):

    C_t(t) = v_b · C_a(t) + K_i · ∫₀ᵗ C_a(τ) dτ

Dividing by C_a(t) yields a linear regression in:

    y = C_t / C_a   vs   x = ∫C_a / C_a

The slope is K_i (BBB influx constant); the intercept is v_b (fractional
blood volume). Both are estimated by OLS over the late phase of the
Patlak plot (default: upper two thirds of the time axis).

Units (legacy p-Brain convention):
    K_i  →  mL/100g/min     (raw 1/s × 6000)
    v_b  →  mL/100g         (raw fraction × 100)

Optional late-tail detrending compensates for scanner drift that
otherwise biases the Patlak slope. Five modes — see ``fit_patlak``.

This is a clean port of ``models/patlak.py`` (the validated reference);
the math is identical, but all options are explicit (no module-global
settings lookup).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from scipy.optimize import curve_fit

from .base import CurveInputs, ModelResult


# ────────────────────────────────────────────────────────────────────────
# Detrending helpers
# ────────────────────────────────────────────────────────────────────────


def _detrend_late_linear(curve: np.ndarray, t: np.ndarray, tail_frac: float
                         ) -> tuple[np.ndarray, float]:
    """Linear-tail detrend: fit a line to the last ``tail_frac`` of frames
    and subtract its slope. Returns ``(detrended, slope)``."""
    curve = np.asarray(curve, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = curve.size
    i0 = int((1.0 - float(tail_frac)) * n)
    if i0 >= n - 2:
        return curve.copy(), 0.0
    A = np.vstack([t[i0:], np.ones(n - i0)]).T
    sol, *_ = np.linalg.lstsq(A, curve[i0:], rcond=None)
    slope = float(sol[0])
    return (curve - slope * t), slope


def _rigid_drift_slope(t: np.ndarray, ca: np.ndarray, *,
                       tail_frac: float = 0.40
                       ) -> tuple[float, dict]:
    """Non-linear "rigid tail" detrend.

    Fits ``C_a_tail(t) = a · exp(-k·(t−t₀)) + b + c·t`` to the last
    ``tail_frac`` of the AIF. The exponential captures expected
    gadolinium washout; the linear coefficient ``c`` is treated as
    drift. Gentler than a pure linear-tail fit because some apparent
    slope is absorbed by the exponential.

    Returns ``(drift_slope, info_dict)``. Falls back to plain linear if
    the non-linear fit can't converge or the decay rate is unphysical.
    """
    t = np.asarray(t, dtype=float).ravel()
    ca = np.asarray(ca, dtype=float).ravel()
    n = int(min(t.size, ca.size))
    t = t[:n]
    ca = ca[:n]
    i0 = int((1.0 - float(tail_frac)) * n)
    if i0 >= n - 5:
        return 0.0, {}
    t_tail = t[i0:]
    c_tail = ca[i0:]
    t0 = float(t_tail[0])

    def model(t_, a, k, b, c):
        return a * np.exp(-k * (t_ - t0)) + b + c * t_

    a0 = max(float(c_tail[0] - c_tail[-1]), float(np.std(c_tail)))
    k0 = 2.0 / max(t_tail[-1] - t_tail[0], 1.0)
    b0 = float(c_tail[-1])
    try:
        popt, _ = curve_fit(
            model, t_tail, c_tail,
            p0=[a0, k0, b0, 0.0],
            bounds=([0, 1e-6, -10, -1.0], [50, 2.0, 10, 1.0]),
            maxfev=4000,
        )
        a, k, b, c = (float(x) for x in popt)
        # Physiological gad-washout band; outside it, exp is essentially
        # flat and the fit is degenerate.
        if not (3e-3 <= k <= 5e-2):
            sol, *_ = np.linalg.lstsq(
                np.vstack([t_tail, np.ones_like(t_tail)]).T, c_tail, rcond=None
            )
            slope = float(sol[0])
            return slope, {"fallback": "linear", "k_unphysical": k}
        return c, {"a": a, "k": k, "b": b, "c": c, "tail_t0": t0}
    except Exception:
        sol, *_ = np.linalg.lstsq(
            np.vstack([t_tail, np.ones_like(t_tail)]).T, c_tail, rcond=None
        )
        slope = float(sol[0])
        return slope, {"fallback": "linear"}


# ────────────────────────────────────────────────────────────────────────
# Single-curve Patlak fit
# ────────────────────────────────────────────────────────────────────────

DetrendMode = Literal["none", "aif", "self", "auto", "rigid", "rigid-auto"]


@dataclass(frozen=True, slots=True)
class PatlakFit:
    """Single-curve Patlak result (units: mL/100g/min, mL/100g)."""

    ki_ml_per_100g_min: float
    vb_ml_per_100g: float
    sd_ki_ml_per_100g_min: float
    x_patlak: np.ndarray
    y_patlak: np.ndarray
    good_mask: np.ndarray


def fit_patlak(
    c_tissue: np.ndarray,
    c_input: np.ndarray,
    t_s: np.ndarray,
    *,
    bad_mask: np.ndarray | None = None,
    window_start_fraction: float = 1.0 / 3.0,
    single_bolus: bool = False,
    single_bolus_start_s: float = 60.0,
    aif_min_fraction: float = 0.05,
    aif_floor_double_bolus: bool = False,
    detrend: DetrendMode = "none",
    detrend_tail_frac: float = 0.30,
    detrend_threshold: float = 5e-4,
) -> PatlakFit:
    """Fit Patlak on one tissue curve. All options explicit (no globals).

    Parameters mirror ``models/patlak.py:fit_patlak`` exactly so that
    parity tests in ``tests/test_pbrain_models_parity.py`` produce
    byte-identical numbers.
    """
    c_t = np.asarray(c_tissue, dtype=float).reshape(-1)
    c_a = np.asarray(c_input, dtype=float).reshape(-1)
    t = np.asarray(t_s, dtype=float).reshape(-1)

    n = int(min(c_t.size, c_a.size, t.size))
    c_t, c_a, t = c_t[:n], c_a[:n], t[:n]

    if bad_mask is None:
        bad = np.zeros(n, dtype=bool)
    else:
        bad = np.asarray(bad_mask, dtype=bool).reshape(-1)[:n]

    # ── Optional late-tail detrend ────────────────────────────────────
    mode = (detrend or "none").strip().lower()
    if mode in {"aif", "self", "auto", "rigid", "rigid-auto"} \
            and n >= 10 and 0.05 < detrend_tail_frac < 0.80:
        i0 = int((1.0 - float(detrend_tail_frac)) * n)
        if i0 < n - 2:
            A = np.vstack([t[i0:], np.ones(n - i0)]).T
            sol_aif, *_ = np.linalg.lstsq(A, c_a[i0:], rcond=None)
            aif_slope = float(sol_aif[0])
            mean_late_aif = float(np.mean(c_a[i0:]))
            aif_norm_slope = (aif_slope / mean_late_aif
                              if abs(mean_late_aif) > 1e-12 else 0.0)

            if mode == "auto":
                if abs(aif_norm_slope) >= float(detrend_threshold):
                    c_a = c_a - aif_slope * t
                    c_t = c_t - aif_slope * t
            elif mode == "aif":
                c_a = c_a - aif_slope * t
                c_t = c_t - aif_slope * t
            elif mode == "self":
                sol_t, *_ = np.linalg.lstsq(A, c_t[i0:], rcond=None)
                t_slope = float(sol_t[0])
                c_a = c_a - aif_slope * t
                c_t = c_t - t_slope * t
            elif mode in {"rigid", "rigid-auto"}:
                rigid_c, _ = _rigid_drift_slope(t, c_a, tail_frac=detrend_tail_frac)
                if mode == "rigid-auto":
                    mean_late = float(np.mean(c_a[i0:])) if i0 < n - 1 else 1.0
                    rigid_norm = (rigid_c / mean_late
                                  if abs(mean_late) > 1e-12 else 0.0)
                    if abs(rigid_norm) < float(detrend_threshold):
                        rigid_c = 0.0
                c_a = c_a - rigid_c * t
                c_t = c_t - rigid_c * t

    if n == 0 or np.diff(t).size == 0:
        return PatlakFit(
            float("nan"), float("nan"), float("nan"),
            np.zeros(n, dtype=float), np.zeros(n, dtype=float),
            np.zeros(n, dtype=bool),
        )

    # AIF-floor masking. Near-zero AIF samples are the Patlak x-axis
    # denominator (x = ∫Cₐ/Cₐ); a sample at e.g. 0.08 % of the peak inflates
    # x by 1000× and creates a high-leverage outlier that OLS fits, blowing
    # Ki up to non-physical values (observed cohort-wide). Excluding
    # Cₐ < aif_min_fraction·peak removes the leverage without touching the
    # well-conditioned bulk. The floor is ON by default (robust); single-bolus
    # always floors. The published method applies NO floor — for byte-parity
    # with patlak_xy use the ``patlak_legacy`` model, which pins
    # ``aif_floor_double_bolus=False``.
    if single_bolus or aif_floor_double_bolus:
        ca_peak = float(np.nanmax(c_a))
        if ca_peak > 0:
            bad = bad | (c_a < aif_min_fraction * ca_peak)

    # Single-bolus also restricts to the post-equilibration window.
    if single_bolus:
        peak_idx = int(np.argmax(c_a))
        t_start = t[peak_idx] + float(single_bolus_start_s)
        bad = bad | (t < t_start)

    # Patlak coordinates  x[i] = ∫₀ᵗ Cₐ / Cₐ[i] ;  y[i] = Cₜ[i] / Cₐ[i]
    dt = np.diff(t)
    x = np.concatenate(([0.0], np.cumsum(c_a[:-1] * dt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = x / c_a
        y = c_t / c_a

    good = (np.isfinite(x) & np.isfinite(y) & np.isfinite(c_a)
            & (c_a != 0) & (~bad))

    if not good.any():
        return PatlakFit(float("nan"), float("nan"), float("nan"), x, y, good)

    x_max = float(np.nanmax(x[good]))
    if single_bolus:
        window = np.ones(n, dtype=bool)
    else:
        w0 = float(window_start_fraction)
        if not (0.0 < w0 < 1.0):
            w0 = 1.0 / 3.0
        window = (x >= w0 * x_max) & (x <= x_max)

    good = good & window
    if int(good.sum()) < 2:
        return PatlakFit(float("nan"), float("nan"), float("nan"), x, y, good)

    xg = x[good]
    yg = y[good]
    xm = float(xg.mean())
    ym = float(yg.mean())
    denom = float(((xg - xm) ** 2).sum())
    if not np.isfinite(denom) or denom <= 0:
        return PatlakFit(float("nan"), float("nan"), float("nan"), x, y, good)

    ki_raw = float(((xg - xm) * (yg - ym)).sum() / denom)
    vp_raw = float(ym - ki_raw * xm)

    resid = yg - (vp_raw + ki_raw * xg)
    dof = int(good.sum()) - 2
    sd_raw = float(np.sqrt(float((resid ** 2).sum()) / denom / dof)) if dof > 0 else float("nan")

    return PatlakFit(
        ki_raw * 6000.0,
        vp_raw * 100.0,
        sd_raw * 6000.0,
        x, y, good,
    )


# ────────────────────────────────────────────────────────────────────────
# Plug-in adapter
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PatlakModel:
    key: ClassVar[str] = "patlak"
    name: ClassVar[str] = "Patlak graphical analysis"
    description: ClassVar[str] = (
        "BBB influx K_i and fractional blood volume v_b via Patlak "
        "graphical analysis (Patlak, Blasberg & Fenstermacher 1983). "
        "Assumes negligible backflux. Linear OLS over the late phase "
        "of the Patlak plot."
    )
    accepts: ClassVar[dict[str, type]] = {
        "c_tissue": np.ndarray,
        "c_input": np.ndarray,
        "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "ki": np.ndarray,
        "vb": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("ki", "vb")
    units: ClassVar[dict[str, str]] = {
        "ki": "mL/100g/min",
        "vb": "mL/100g",
    }

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        c_t = np.asarray(inputs.c_tissue, dtype=float)
        c_a = np.asarray(inputs.c_input, dtype=float)
        t_s = np.asarray(inputs.t_s, dtype=float)

        if c_t.ndim == 1:
            return self._fit_one(c_t, c_a, t_s, inputs.mask, opts)

        if c_t.ndim != 2:
            raise ValueError(f"c_tissue must be 1-D or 2-D; got shape {c_t.shape}")

        # 2-D voxelwise path.
        #
        # ``baseline_shift`` selects the per-voxel baseline convention:
        #   * False (DEFAULT) — MATLAB-exact: the signal→conc conversion already
        #     subtracts the pre-bolus baseline MEAN and zeros the fixed baseline
        #     window (``create_MR_concentration_deck.m`` / the 05 stage), and the
        #     AIF stays RAW. No extra shift. Matches the supervisor MATLAB Patlak.
        #   * True — additionally apply the legacy p-Brain per-voxel
        #     ``find_baseline_point_advanced`` + ``custom_shifter`` (detect bolus
        #     onset, zero pre-onset frames, subtract the post-onset MINIMUM). This
        #     reproduces the published ``Ki_per_voxel.nii.gz`` exactly, but the
        #     min-subtract lifts noisy tails and biases Ki vs the MATLAB value.
        # Tunable per run: ``--opt models.patlak.baseline_shift=true``.
        # ``tail_mode``: 'legacy' = exact per-voxel ``fit_patlak`` loop (slow
        # reference); 'x_max_fraction'/'smart' = the older index-window path.
        baseline_shift = bool(opts.get("baseline_shift", False))
        ct2d = c_t
        if baseline_shift:
            from ._baseline_shift import baseline_shift_2d
            ct2d = baseline_shift_2d(c_t)

        tail_mode = str(opts.get("tail_mode", "legacy_vectorised")).lower()
        w_start = float(opts.get("window_start_fraction", 1.0 / 3))

        if tail_mode == "legacy_vectorised":
            from ._baseline_shift import fit_patlak_legacy_vectorised
            regression = str(opts.get("regression", "huber")).lower()
            res = fit_patlak_legacy_vectorised(
                ct2d, c_a, t_s, mask=inputs.mask, window_start_fraction=w_start,
                regression=regression,
                huber_delta=float(opts.get("huber_delta", 1.345)),
                huber_max_iter=int(opts.get("huber_max_iter", 20)),
            )
            return ModelResult(
                maps={"ki": res["ki_ml_per_100g_min"], "vb": res["vb_ml_per_100g"]},
                units=dict(self.units),
                aux={"sd_ki": res["sd_ki_ml_per_100g_min"],
                     "baseline_shift": baseline_shift,
                     "tail_mode": "legacy_vectorised", "regression": regression},
            )

        from ._patlak_tools import fit_patlak_vectorised
        vector_keys = {
            "tail_mode", "regression", "huber_delta", "huber_max_iter",
            "baseline_frames", "decay_threshold", "min_tail_frac",
            "peak_prominence_frac", "peak_separation",
        }
        vector_opts = {k: opts[k] for k in list(opts.keys()) if k in vector_keys}

        if tail_mode == "legacy":
            # Exact per-voxel ``fit_patlak`` loop (slow reference path).
            legacy_opts = {k: v for k, v in opts.items()
                           if k not in vector_keys and k != "baseline_shift"}
            n_v = ct2d.shape[1]
            ki = np.full(n_v, np.nan)
            vb = np.full(n_v, np.nan)
            sd = np.full(n_v, np.nan)
            voxels = (range(n_v) if inputs.mask is None
                      else np.flatnonzero(np.asarray(inputs.mask, dtype=bool)))
            for v in voxels:
                fit = fit_patlak(ct2d[:, v], c_a, t_s, **legacy_opts)
                ki[v] = fit.ki_ml_per_100g_min
                vb[v] = fit.vb_ml_per_100g
                sd[v] = fit.sd_ki_ml_per_100g_min
            return ModelResult(
                maps={"ki": ki, "vb": vb},
                units=dict(self.units),
                aux={"sd_ki": sd, "baseline_shift": baseline_shift,
                     "tail_mode": "legacy", "regression": "ols"},
            )

        res = fit_patlak_vectorised(ct2d, c_a, t_s, mask=inputs.mask, **vector_opts)
        return ModelResult(
            maps={"ki": res["ki_ml_per_100g_min"], "vb": res["vb_ml_per_100g"]},
            units=dict(self.units),
            aux={
                "sd_ki": res["sd_ki_ml_per_100g_min"],
                "tail_start": res["tail_start"],
                "tail_end": res["tail_end"],
                "baseline_shift": baseline_shift,
                "tail_mode": vector_opts.get("tail_mode", "x_max_fraction"),
                "regression": vector_opts.get("regression", "ols"),
            },
        )

    def _fit_one(self, c_t, c_a, t_s, mask, opts) -> ModelResult:
        # Single-curve path uses the legacy OLS fit_patlak regardless of
        # tail_mode (tail_mode='legacy' and the vectorised-only knobs only
        # apply to the 2-D Huber path). Strip keys fit_patlak doesn't accept.
        _vector_only = {
            "tail_mode", "regression", "huber_delta", "huber_max_iter",
            "baseline_frames", "decay_threshold", "min_tail_frac",
            "peak_prominence_frac", "peak_separation",
        }
        opts = {k: v for k, v in opts.items() if k not in _vector_only}
        fit = fit_patlak(c_t, c_a, t_s, bad_mask=mask, **opts)
        return ModelResult(
            maps={
                "ki": np.asarray(fit.ki_ml_per_100g_min),
                "vb": np.asarray(fit.vb_ml_per_100g),
            },
            units=dict(self.units),
            aux={
                "sd_ki": float(fit.sd_ki_ml_per_100g_min),
                "x_patlak": fit.x_patlak,
                "y_patlak": fit.y_patlak,
                "good_mask": fit.good_mask,
            },
        )


    def predict(self, maps: dict[str, Any], c_input: np.ndarray,
                t_s: np.ndarray) -> np.ndarray:
        """Reconstruct Cₜ = (Ki/6000)·∫₀ᵗCₐ + (vb/100)·Cₐ from the fitted maps."""
        ca = np.asarray(c_input, dtype=float).ravel()
        t = np.asarray(t_s, dtype=float).ravel()
        n = int(min(ca.size, t.size))
        ca, t = ca[:n], t[:n]
        ki = float(maps.get("ki", float("nan"))) / 6000.0     # mL/100g/min → 1/s
        vb = float(maps.get("vb", float("nan"))) / 100.0      # mL/100g → fraction
        dt = np.diff(t)
        cumint = np.concatenate(([0.0], np.cumsum(ca[:-1] * dt)))
        return ki * cumint + vb * ca


PLUGIN = PatlakModel()
