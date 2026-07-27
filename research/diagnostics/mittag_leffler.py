"""Fractional Mittag-Leffler diagnostic — a deep dive into one fitted curve.

For a single tissue / parcel / voxel curve it runs the (accurate, 1-D) TRF fit
and renders a 6-panel figure plus a parameter banner:

  (a) Forward fit       Cₜ data + Mittag-Leffler fit + AIF (twin axis); R², SSE.
  (b) Residuals         (data − fit) over time with the RMSE band — fit quality.
  (c) Residue R(t)      E_α(-(t/τ)^α) with the median transit time (R=½) marked
                        and the windowed MTT (∫₀^Tacq R) shaded, vs an equal-area
                        mono-exponential (Tofts) — the heavy tail made visible.
  (d) Log–log tail      R(t) on log–log axes; the late slope → −α (the algebraic
                        retention exponent), with the fitted asymptote drawn.
  (e) Transit density   h(t) = −dR/dt vs the mono-exponential, median marked.
  (f) Rate spectrum     the Bernstein/Pollard spectral density K_α — i.e. the
                        residue written as a positive *mixture of exponential
                        clearances* over their time-constants; α=1 collapses to a
                        single rate (Tofts). This is the completely-monotone view.

Auto-discovered like any diagnostic; ``.gitignore``-d alongside the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from pbrain._numpy_compat import trapezoid
from .base import Diagnostic, DiagnosticContext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402


def _bernstein_density(r: np.ndarray, alpha: float) -> np.ndarray:
    """Pollard spectral density K_α(r): R(t)=E_α(-(t/τ)^α)=∫e^{-(t/τ)r}K_α(r)dr."""
    a = float(alpha)
    ra = r ** a
    return (math.sin(a * math.pi) / math.pi) * r ** (a - 1.0) / (
        r ** (2.0 * a) + 2.0 * ra * math.cos(a * math.pi) + 1.0)


@dataclass(frozen=True, slots=True)
class _MittagLefflerDiagnostic:
    key: ClassVar[str] = "mittag_leffler"
    name: ClassVar[str] = "Fractional Mittag-Leffler diagnostic"
    description: ClassVar[str] = (
        "Deep-dive panel: forward fit, residuals, fractional residue with "
        "median transit time and windowed MTT, log-log power-law tail, "
        "transit-time density, and the Bernstein rate-spectrum (mixture of "
        "exponential clearances)."
    )
    accepts: ClassVar[dict[str, type]] = {"context": DiagnosticContext}
    produces: ClassVar[dict[str, type]] = {"png": str}
    model_key: ClassVar[str] = "mittag_leffler"

    def plot(self, ctx: DiagnosticContext) -> None:
        from pbrain.models.mittag_leffler import (
            _MittagLefflerFitter, MittagLefflerFit, ml_relaxation,
        )
        from pbrain.models import REGISTRY, CurveInputs

        c_t = np.asarray(ctx.c_tissue, dtype=float).ravel()
        c_a = np.asarray(ctx.c_input, dtype=float).ravel()
        t = np.asarray(ctx.t_s, dtype=float).ravel()
        n = int(min(c_t.size, c_a.size, t.size))
        c_t, c_a, t = c_t[:n], c_a[:n], t[:n]
        t_acq = float(t[-1])

        # ── accurate TRF fit (1-D path; F pinned to a Tikhonov seed) ──
        res = REGISTRY["mittag_leffler"].fit(
            CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t))
        F = float(res.maps["cbf"]); alpha = float(res.maps["alpha"])
        tau = float(res.maps["tau"]); tmed = float(res.maps["tmed"])
        mtt_win = float(res.maps["mtt_win"]); sse = float(res.aux.get("sse", float("nan")))

        fitter = _MittagLefflerFitter(t, c_a)
        if np.isfinite(alpha) and np.isfinite(tau) and np.isfinite(F):
            ct_fit = fitter.forward(MittagLefflerFit(F, alpha, tau, tmed, mtt_win, sse))
        else:
            ct_fit = np.full_like(c_t, np.nan)
        resid = c_t - ct_fit
        ss_res = float(np.nansum(resid ** 2))
        ss_tot = float(np.nansum((c_t - np.nanmean(c_t)) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        rmse = math.sqrt(ss_res / max(np.isfinite(resid).sum(), 1))

        # ── residue / density on a fine grid ──
        tt = np.linspace(0.0, t_acq, 800)
        if np.isfinite(alpha) and np.isfinite(tau) and tau > 0:
            R = ml_relaxation(tt / tau, alpha)
            R_exp = np.exp(-tt / max(mtt_win, 1e-6))      # equal-(window)-area mono-exp
            h = -np.gradient(R, tt); h = np.clip(h, 0, None)
            h_exp = -np.gradient(R_exp, tt); h_exp = np.clip(h_exp, 0, None)
        else:
            R = R_exp = h = h_exp = np.full_like(tt, np.nan)

        # ─────────────────────────── figure ───────────────────────────
        fig = plt.figure(figsize=(16, 9.5))
        gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27,
                      top=0.83, bottom=0.07, left=0.06, right=0.985)
        C_FRAC, C_EXP, C_DAT = "#c0392b", "#2980b9", "#333333"

        # (a) forward fit + AIF
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(t, c_t, "o", ms=3, color=C_DAT, alpha=0.7, label="tissue $C_t$")
        ax.plot(t, ct_fit, "-", lw=2, color=C_FRAC, label="ML fit")
        ax.set_xlabel("time (s)"); ax.set_ylabel("$C_t$ (mM)")
        axb = ax.twinx(); axb.plot(t, c_a, "-", lw=1, color=C_EXP, alpha=0.5)
        axb.set_ylabel("AIF $C_a$ (mM)", color=C_EXP); axb.tick_params(labelcolor=C_EXP)
        ax.set_title(f"(a) forward fit   $R^2$={r2:.4f}", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)

        # (b) residuals
        ax = fig.add_subplot(gs[1, 0])
        ax.axhline(0, color="k", lw=0.6)
        ax.axhspan(-rmse, rmse, color="0.8", alpha=0.5, label=f"±RMSE ({rmse:.2g})")
        ax.plot(t, resid, ".", ms=3, color=C_FRAC)
        ax.set_xlabel("time (s)"); ax.set_ylabel("data − fit (mM)")
        ax.set_title("(b) residuals", fontsize=11); ax.legend(loc="upper right", fontsize=8)

        # (c) residue R(t) with tmed + windowed-MTT area
        ax = fig.add_subplot(gs[0, 1])
        ax.fill_between(tt, 0, R, color=C_FRAC, alpha=0.12)
        ax.plot(tt, R, "-", lw=2, color=C_FRAC, label=fr"$E_\alpha(-(t/\tau)^\alpha)$")
        ax.plot(tt, R_exp, "--", lw=1.6, color=C_EXP, label="mono-exp (Tofts), equal area")
        if np.isfinite(tmed):
            ax.plot([tmed, tmed], [0, 0.5], ":", color="k", lw=1)
            ax.plot([0, tmed], [0.5, 0.5], ":", color="k", lw=1)
            ax.annotate(f"$t_{{med}}$={tmed:.1f}s", (tmed, 0.5), fontsize=8,
                        xytext=(tmed + 0.04 * t_acq, 0.56))
        ax.set_xlabel("time (s)"); ax.set_ylabel("R(t)")
        ax.set_title(fr"(c) residue   shaded $\int_0^{{T}}R$ = MTT$_{{win}}$={mtt_win:.1f}s",
                     fontsize=11)
        ax.set_ylim(0, 1.02); ax.legend(loc="upper right", fontsize=8)

        # (d) log-log tail → slope -alpha
        ax = fig.add_subplot(gs[1, 1])
        good = (tt > 0) & (R > 1e-4)
        ax.loglog(tt[good], R[good], "-", lw=2, color=C_FRAC, label="R(t)")
        if np.isfinite(alpha) and np.isfinite(tau):
            tl = tt[(tt > max(3 * tau, t_acq * 0.2))]
            if tl.size:
                asy = (tl / tau) ** (-alpha) / math.gamma(1 - alpha) if alpha < 0.999 else np.exp(-tl / tau)
                ax.loglog(tl, asy, ":", color="k", lw=1.4,
                          label=(fr"$\propto t^{{-\alpha}}$, α={alpha:.3f}" if alpha < 0.999
                                 else "exp tail (α=1)"))
        ax.set_xlabel("time (s)"); ax.set_ylabel("R(t)")
        ax.set_title("(d) tail on log–log  (slope → −α)", fontsize=11)
        ax.legend(loc="lower left", fontsize=8)

        # (e) transit-time density h(t) = -dR/dt
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(tt, h, "-", lw=2, color=C_FRAC, label="h(t) = −dR/dt")
        ax.plot(tt, h_exp, "--", lw=1.4, color=C_EXP, label="mono-exp")
        if np.isfinite(tmed):
            ax.axvline(tmed, ls=":", color="k", lw=1, label=f"$t_{{med}}$={tmed:.1f}s")
        ax.set_xlabel("time (s)"); ax.set_ylabel("h(t)  (1/s)")
        ax.set_xlim(0, min(t_acq, max(6 * tau, 30) if np.isfinite(tau) else 60))
        ax.set_title("(e) transit-time density", fontsize=11); ax.legend(loc="upper right", fontsize=8)

        # (f) Bernstein rate spectrum: residue as a mixture of exponential clearances
        ax = fig.add_subplot(gs[1, 2])
        if np.isfinite(alpha) and np.isfinite(tau) and tau > 0:
            if alpha < 0.999:
                r = np.geomspace(1e-3, 1e3, 600)
                Tc = tau / r                                  # clearance time-constant (s)
                weight = r * _bernstein_density(r, alpha)     # mass per log-interval
                ax.semilogx(Tc, weight / trapezoid(weight[::-1], np.log(Tc[::-1])),
                            "-", lw=2, color=C_FRAC)
                ax.fill_between(Tc, 0, weight / trapezoid(weight[::-1], np.log(Tc[::-1])),
                                color=C_FRAC, alpha=0.12)
                ax.axvline(tau, ls=":", color="k", lw=1, label=fr"$\tau$={tau:.1f}s")
                ax.set_title("(f) Bernstein rate-spectrum: mixture of exp. clearances",
                             fontsize=10.5)
            else:
                ax.axvline(tau, color=C_FRAC, lw=3, label=fr"single rate $\tau$={tau:.1f}s")
                ax.set_title("(f) rate-spectrum — α=1: one rate (Tofts)", fontsize=10.5)
            ax.set_xlabel("clearance time-constant  τ/r  (s)")
            ax.set_ylabel("spectral weight"); ax.legend(loc="upper right", fontsize=8)
        else:
            ax.axis("off")

        # ── parameter banner ──
        finite = np.isfinite(alpha)
        anom = ("mono-exponential (α≈1, Tofts limit)" if finite and alpha > 0.97
                else "anomalous / sub-exponential — trapping, heavy retention tail"
                if finite else "fit failed")
        banner = (
            r"$\bf{Fractional\ Mittag\text{-}Leffler\ residue}$    "
            fr"$R(t)=E_\alpha(-(t/\tau)^\alpha)$        "
            fr"$\alpha$ = {alpha:.3f}   |   $\tau$ = {tau:.2f} s   |   "
            fr"CBF = {F:.1f} mL/100g/min   |   $t_{{med}}$ = {tmed:.2f} s   |   "
            fr"MTT$_{{win}}$ = {mtt_win:.1f} s   |   $R^2$ = {r2:.4f}"
            f"\nregime: {anom}")
        fig.text(0.06, 0.90, banner, fontsize=11, va="top", ha="left")
        fig.suptitle(f"Mittag-Leffler diagnostic — {ctx.label}", fontsize=13, x=0.06,
                     ha="left", y=0.975)

        ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ctx.out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


PLUGIN = _MittagLefflerDiagnostic()
