"""Gamma transit-time diagnostic plot.

Shows the parametric gamma TTD ``h(t)``, residue ``R(t)``, fitted Cₜ
against measured Cₜ, and all the recovered Larsson parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from pbrain._numpy_compat import trapezoid
from .base import Diagnostic, DiagnosticContext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True, slots=True)
class _GammaDiagnostic:
    key: ClassVar[str] = "gamma"
    name: ClassVar[str] = "Gamma TTD diagnostic plot"
    description: ClassVar[str] = (
        "Fitted Cₜ overlay, capillary gamma TTD h(t), residue R(t), "
        "parameter annotations."
    )
    accepts: ClassVar[dict[str, type]] = {"context": DiagnosticContext}
    produces: ClassVar[dict[str, type]] = {"png": str}
    model_key: ClassVar[str] = "gamma"

    def plot(self, ctx: DiagnosticContext) -> None:
        from pbrain.models import REGISTRY as MODELS, CurveInputs
        try:
            gamma = MODELS["gamma"]
        except KeyError:
            # Gamma plug-in not loaded on this install — render placeholder.
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "Gamma model not available", ha="center", va="center")
            ax.axis("off")
            ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ctx.out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            return

        c_t = np.asarray(ctx.c_tissue, dtype=float).ravel()
        c_a = np.asarray(ctx.c_input, dtype=float).ravel()
        t = np.asarray(ctx.t_s, dtype=float).ravel()
        n = int(min(c_t.size, c_a.size, t.size))
        c_t, c_a, t = c_t[:n], c_a[:n], t[:n]

        opts = dict(ctx.model_opts or {})
        # Fit via the low-level fitter so we can reconstruct the FULL-RIF
        # forward model (capillary + extravasation), not a capillary-only
        # approximation. F is pinned to a Tikhonov CBF seed (legacy design).
        from pbrain.models.gamma import _LarssonFitter, H_FACTOR
        from pbrain.models.tikhonov import build_tikhonov_solver

        fitter = _LarssonFitter(t, c_a, h_factor=int(opts.get("h_factor", H_FACTOR)))
        tik = build_tikhonov_solver(
            t, c_a, lambda_selection="evidence", lambda_spacing="log",
            lambda_min=1e-3, mtt_cth_method="residue_integral",
        )
        f_seed = float(tik(c_t.reshape(-1, 1)).cbf_ml_per_100g_min[0]) / 6000.0
        if not np.isfinite(f_seed) or f_seed <= 0:
            f_seed = 50.0 / 6000.0
        gfit = fitter.fit(c_t, f_cbf=f_seed,
                          n_restarts=int(opts.get("n_restarts", 16)), pin_f=True)

        cbf = float(gfit.f_ml_100g_min)
        mtt_cap = float(gfit.mtt_cap_s)
        mtt = float(gfit.mtt_total_s)
        cth = float(gfit.cth_s)
        E = float(gfit.extraction)
        k2 = float(gfit.k2_per_s)
        alpha = float(gfit.alpha)
        beta_s = float(gfit.beta_s)
        sse = float(gfit.sse)
        # Real full-RIF forward fit (the curve the model actually minimised).
        ct_fit_full = fitter.forward(gfit, time_out=t)

        # Build parametric h(t) and R(t) on the displayed time axis
        if np.isfinite(alpha) and alpha > 0 and np.isfinite(beta_s) and beta_s > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                h_param = np.where(
                    t > 0,
                    (np.power(t / beta_s, alpha) * np.exp(alpha * (1 - t / beta_s)))
                    / (beta_s + 1e-12),
                    0.0,
                )
            h_param = np.where(np.isfinite(h_param), np.maximum(h_param, 0.0), 0.0)
            area = float(trapezoid(h_param, t))
            if area > 0:
                h_param = h_param / area
            H_cum = np.concatenate(([0.0], np.cumsum(0.5 * (h_param[1:] + h_param[:-1]) * np.diff(t))))
            R_param = 1.0 - H_cum
        else:
            h_param = np.zeros_like(t)
            R_param = np.ones_like(t)

        # Real full-RIF forward fit (capillary + extravasation).
        ct_fit = np.where(np.isfinite(ct_fit_full), ct_fit_full, 0.0)
        ss_res = float(np.sum((c_t - ct_fit) ** 2))
        ss_tot = float(np.sum((c_t - c_t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # ─── figure ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # (0, 0) — fitted vs measured tissue, plus AIF for context
        ax = axes[0, 0]
        ax.plot(t, c_t, ".", color="tab:green", ms=3.5, alpha=0.6, label="Cₜ measured")
        ax.plot(t, ct_fit, "k-", lw=1.5,
                label=f"Cₜ fitted (full RIF)  R²={r2:.3f}")
        ax2 = ax.twinx()
        ax2.plot(t, c_a, "tab:blue", lw=1.0, alpha=0.4, label="Cₐ")
        ax2.set_ylabel("Cₐ (mM)", color="tab:blue")
        ax.set_xlabel("time (s)"); ax.set_ylabel("Cₜ (mM)", color="tab:green")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.set_title("Gamma forward fit overlay")

        # capillary transit is a first-pass phenomenon — show the early window
        early = float(min(t[-1], 60.0))

        # (0, 1) — capillary gamma h(t)
        ax = axes[0, 1]
        ax.plot(t, h_param, "tab:orange", lw=1.5)
        ax.set_xlim(0, early)
        ax.set_xlabel("time (s)"); ax.set_ylabel("h(t)  (1/s)")
        ax.set_title(f"Capillary TTD  —  α={alpha:.2f}, β={beta_s:.2f}s, CTH={cth:.2f}s")
        ax.grid(alpha=0.3)

        # (1, 0) — residue
        ax = axes[1, 0]
        ax.plot(t, R_param, "tab:purple", lw=1.5)
        ax.set_xlim(0, early); ax.set_ylim(0, 1.05)
        ax.set_xlabel("time (s)"); ax.set_ylabel("R(t)")
        ax.set_title(f"Residue  —  MTT_cap={mtt_cap:.2f}s, MTT_total={mtt:.2f}s")
        ax.grid(alpha=0.3)

        # (1, 1) — parameter table
        ax = axes[1, 1]; ax.axis("off")
        text = (
            f"CBF (F)        : {cbf:.2f}  mL/100g/min\n"
            f"MTT_cap        : {mtt_cap:.2f}  s\n"
            f"MTT_total      : {mtt:.2f}  s\n"
            f"CTH            : {cth:.2f}  s\n"
            f"α (alpha)      : {alpha:.3f}\n"
            f"β (beta)       : {beta_s:.3f}  s\n"
            f"E (extraction) : {E:.3f}\n"
            f"k₂             : {k2:.5f}  1/s\n"
            f"SSE            : {sse:.3g}\n"
        )
        ax.text(0.0, 1.0, text, family="monospace", fontsize=10,
                verticalalignment="top", transform=ax.transAxes)

        fig.suptitle(f"Gamma diagnostic — {ctx.label}", fontsize=12, y=0.995)
        ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ctx.out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)


PLUGIN = _GammaDiagnostic()
