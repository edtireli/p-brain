"""Inverse-Gaussian (Wald) first-passage diagnostic — deep dive on one curve.

Panels:
  (a) Forward fit       Cₜ data + fit + AIF, with the vascular and Patlak-leak
                        components of the fit drawn separately.
  (b) Residuals         data − fit with the RMSE band.
  (c) Residue R(t)      (1−E)R_IG(t)+E, the retained plateau E shaded, the
                        vascular MTT (μ) and a ±CTH band marked.
  (d) Wald density      h(t)=transit-time distribution, with μ (mean), the mode,
                        and ±CTH; this is the actual first-passage law.
  (e) Péclet portrait   R_IG for a ladder of Péclet numbers with the fitted Pe
                        highlighted — what advection/dispersion ratio the curve
                        implies (and how flat the likelihood is in that axis).
  (f) Parameter banner.

Auto-discovered; ``.gitignore``-d alongside the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .base import Diagnostic, DiagnosticContext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402


@dataclass(frozen=True, slots=True)
class _InvGaussDiagnostic:
    key: ClassVar[str] = "inverse_gaussian"
    name: ClassVar[str] = "Inverse-Gaussian first-passage diagnostic"
    description: ClassVar[str] = (
        "Forward fit with vascular/leak decomposition, residuals, residue with "
        "retained plateau, the Wald transit-time density, and a Péclet portrait."
    )
    accepts: ClassVar[dict[str, type]] = {"context": DiagnosticContext}
    produces: ClassVar[dict[str, type]] = {"png": str}
    model_key: ClassVar[str] = "inverse_gaussian"

    def plot(self, ctx: DiagnosticContext) -> None:
        from pbrain.models.inverse_gaussian import (
            _InvGaussFitter, InvGaussFit, ig_survival, ig_pdf,
        )
        from pbrain.models import REGISTRY, CurveInputs

        c_t = np.asarray(ctx.c_tissue, float).ravel()
        c_a = np.asarray(ctx.c_input, float).ravel()
        t = np.asarray(ctx.t_s, float).ravel()
        n = int(min(c_t.size, c_a.size, t.size)); c_t, c_a, t = c_t[:n], c_a[:n], t[:n]
        t_acq = float(t[-1])

        res = REGISTRY["inverse_gaussian"].fit(CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t))
        F = float(res.maps["cbf"]); mu = float(res.maps["mtt"]); cth = float(res.maps["cth"])
        pe = float(res.maps["pe"]); E = float(res.maps["e_leak"])
        lam = float(res.aux.get("lam_s", mu ** 3 / max(cth, 1e-6) ** 2)); sse = float(res.aux.get("sse", np.nan))

        fitter = _InvGaussFitter(t, c_a)
        f = F / 6000.0
        finite = np.isfinite([F, mu, lam, E]).all()
        if finite:
            vasc = f * (1.0 - E) * np.interp(t, fitter.time_h, fitter._conv_RIG(mu, lam))
            leak = f * E * np.interp(t, fitter.time_h, fitter.ca_cum_h)
            ct_fit = vasc + leak
        else:
            vasc = leak = ct_fit = np.full_like(c_t, np.nan)
        resid = c_t - ct_fit
        ss_res = float(np.nansum(resid ** 2)); ss_tot = float(np.nansum((c_t - np.nanmean(c_t)) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        rmse = math.sqrt(ss_res / max(np.isfinite(resid).sum(), 1))

        tt = np.linspace(0.0, t_acq, 800)
        R_ig = ig_survival(tt, mu, lam) if finite else np.full_like(tt, np.nan)
        R = (1.0 - E) * R_ig + E
        h = ig_pdf(tt, mu, lam) if finite else np.full_like(tt, np.nan)
        mode = mu * (math.sqrt(1.0 + (1.5 * mu / lam) ** 2) - 1.5 * mu / lam) if finite else np.nan

        fig = plt.figure(figsize=(16, 9.5))
        gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27,
                      top=0.83, bottom=0.07, left=0.06, right=0.985)
        C_F, C_V, C_L, C_D = "#c0392b", "#16a085", "#8e44ad", "#333333"

        # (a) fit + decomposition
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(t, c_t, "o", ms=3, color=C_D, alpha=0.7, label="tissue $C_t$")
        ax.plot(t, ct_fit, "-", lw=2, color=C_F, label=f"IG+leak fit  $R^2$={r2:.4f}")
        ax.plot(t, vasc, "-", lw=1.2, color=C_V, label="vascular  $F(1{-}E)\\,C_a\\!*\\!R_{IG}$")
        ax.plot(t, leak, "-", lw=1.2, color=C_L, label="leak  $FE\\!\\int C_a$")
        axb = ax.twinx(); axb.plot(t, c_a, lw=1, color="#2980b9", alpha=0.35)
        axb.set_ylabel("AIF", color="#2980b9"); axb.tick_params(labelcolor="#2980b9")
        ax.set_xlabel("time (s)"); ax.set_ylabel("$C_t$ (mM)")
        ax.set_title("(a) forward fit + vascular/leak split", fontsize=11); ax.legend(fontsize=7.5, loc="upper right")

        # (b) residuals
        ax = fig.add_subplot(gs[1, 0])
        ax.axhline(0, color="k", lw=0.6); ax.axhspan(-rmse, rmse, color="0.8", alpha=0.5, label=f"±RMSE ({rmse:.2g})")
        ax.plot(t, resid, ".", ms=3, color=C_F)
        ax.set_xlabel("time (s)"); ax.set_ylabel("data − fit"); ax.set_title("(b) residuals", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

        # (c) residue with plateau + MTT + CTH band
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(tt, R, "-", lw=2, color=C_F, label="$R(t)=(1{-}E)R_{IG}+E$")
        ax.axhline(E, ls=":", color=C_L, lw=1.2, label=f"retained plateau E={E:.3f}")
        if finite:
            ax.axvline(mu, ls="--", color=C_V, lw=1, label=f"MTT μ={mu:.2f}s")
            ax.axvspan(max(mu - cth, 0), mu + cth, color=C_V, alpha=0.10, label=f"±CTH ({cth:.2f}s)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("R(t)"); ax.set_ylim(0, 1.02)
        ax.set_title("(c) residue: vascular decay + leak plateau", fontsize=11); ax.legend(fontsize=8, loc="upper right")

        # (d) Wald transit-time density
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(tt, h, "-", lw=2, color=C_V)
        ax.fill_between(tt, 0, h, color=C_V, alpha=0.12)
        if finite:
            ax.axvline(mu, ls="--", color="k", lw=1, label=f"mean μ={mu:.2f}s")
            ax.axvline(mode, ls=":", color=C_V, lw=1.2, label=f"mode={mode:.2f}s")
        ax.set_xlabel("transit time t (s)"); ax.set_ylabel("h(t)  (1/s)")
        ax.set_xlim(0, min(t_acq, max(4 * mu, 12) if finite else 30))
        ax.set_title("(d) Wald transit-time density (first passage)", fontsize=11); ax.legend(fontsize=8, loc="upper right")

        # (e) Péclet portrait
        ax = fig.add_subplot(gs[0, 2])
        if finite:
            for pe_show in [1.0, 3.0, 10.0, 30.0, 100.0]:
                lam_s = pe_show * mu / 2.0
                ax.plot(tt, ig_survival(tt, mu, lam_s), color="0.75", lw=1)
            ax.plot(tt, R_ig, color=C_F, lw=2.4, label=f"fitted  Pe={pe:.1f}")
            ax.text(0.97, 0.5, "Pe: 1→100\n(grey ladder)", transform=ax.transAxes,
                    ha="right", fontsize=8, color="0.4")
        ax.set_xlabel("time (s)"); ax.set_ylabel("$R_{IG}(t)$"); ax.set_ylim(0, 1.02)
        ax.set_xlim(0, min(t_acq, max(5 * mu, 15) if finite else 30))
        ax.set_title("(e) Péclet portrait (advection/dispersion)", fontsize=11); ax.legend(fontsize=8, loc="upper right")

        # (f) banner panel
        ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
        regime = ("plug-flow (Pe high) — dispersion below temporal resolution; CTH weakly identified"
                  if finite and pe > 50 else
                  "dispersive bed (Pe low–moderate) — CTH well resolved" if finite else "fit failed")
        txt = (f"INVERSE-GAUSSIAN (WALD) FIRST-PASSAGE\n"
               f"advection–diffusion first passage; R=(1−E)·R_IG+E\n\n"
               f"CBF      = {F:6.1f}  mL/100g/min\n"
               f"MTT (μ)  = {mu:6.2f} s     = L/v\n"
               f"CTH      = {cth:6.2f} s     = √(μ³/λ)\n"
               f"Péclet   = {pe:6.1f}        = vL/D = 2λ/μ\n"
               f"E (leak) = {E:6.3f}        retained fraction\n"
               f"λ        = {lam:6.2f} s\n"
               f"R²       = {r2:6.4f}\n\n"
               f"regime: {regime}")
        ax.text(0.0, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=10, family="monospace")

        fig.suptitle(f"Inverse-Gaussian first-passage diagnostic — {ctx.label}",
                     fontsize=13, x=0.06, ha="left", y=0.965)
        ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ctx.out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


PLUGIN = _InvGaussDiagnostic()
