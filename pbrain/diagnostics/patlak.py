"""Patlak diagnostic plot — what the fit actually saw.

For one curve (tissue / parcel / voxel) shows:

  · Top row    – Cₐ(t) and Cₜ(t) time series with the tail window shaded.
  · Mid row    – Patlak (x, y) scatter: excluded points (grey), included
                 points (red), Huber fit (blue), OLS fit on the same window
                 (green dashed). Annotated Kᵢ from each.
  · Bottom row – Residuals (Huber-fit) vs Patlak-x for the included window.
                 Annotations: tail mode, n_included, x_max.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import Diagnostic, DiagnosticContext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


@dataclass(frozen=True, slots=True)
class _PatlakDiagnostic:
    key: ClassVar[str] = "patlak"
    name: ClassVar[str] = "Patlak diagnostic plot"
    description: ClassVar[str] = (
        "Time-domain curves with tail window, Patlak (x, y) scatter "
        "with included / excluded points and Huber + OLS fit lines, "
        "and the Huber residuals."
    )
    accepts: ClassVar[dict[str, type]] = {"context": DiagnosticContext}
    produces: ClassVar[dict[str, type]] = {"png": str}
    model_key: ClassVar[str] = "patlak"

    def plot(self, ctx: DiagnosticContext) -> None:
        from pbrain.models._patlak_tools import (
            find_patlak_tail, vectorised_huber, vectorised_ols,
        )

        c_t = np.asarray(ctx.c_tissue, dtype=float).ravel()
        c_a = np.asarray(ctx.c_input, dtype=float).ravel()
        t = np.asarray(ctx.t_s, dtype=float).ravel()
        n = int(min(c_t.size, c_a.size, t.size))
        c_t, c_a, t = c_t[:n], c_a[:n], t[:n]

        # Patlak coords
        dt = np.diff(t)
        integ = np.concatenate(([0.0], np.cumsum(c_a[:-1] * dt)))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_pat = integ / c_a
            y_pat = c_t / c_a
        ok_mask = np.isfinite(x_pat) & np.isfinite(y_pat) & (c_a > 0)

        # Tail window (same algorithm the kinetic stage used)
        opts = ctx.model_opts or {}
        # ``tail_mode`` in the model opts selects the *fit path* (legacy /
        # legacy_vectorised / patlak3) — not a tail-window algorithm. Those paths all
        # regress over the x_max/3 window, so map anything find_patlak_tail doesn't
        # know onto x_max_fraction rather than raising and losing the diagnostic.
        _WINDOW_MODES = {"curvature", "x_max_fraction", "peak_decay",
                         "changepoint", "fixed", "smart"}
        tail_mode = opts.get("tail_mode", "x_max_fraction")
        if tail_mode not in _WINDOW_MODES:
            tail_mode = "x_max_fraction"
        tail_start, tail_end = find_patlak_tail(
            c_a, t,
            mode=tail_mode,
            x_fraction=opts.get("x_fraction", 1.0 / 3.0),
            baseline_frames=opts.get("baseline_frames", 5),
            decay_threshold=opts.get("decay_threshold", 0.5),
            peak_prominence_frac=opts.get("peak_prominence_frac", 0.10),
            peak_separation=opts.get("peak_separation", 10),
        )
        in_window = np.zeros(n, dtype=bool)
        in_window[tail_start:tail_end] = True
        included = ok_mask & in_window
        excluded = ok_mask & ~in_window

        # Fits on the included window
        ki_hub = vp_hub = ki_ols = vp_ols = float("nan")
        slope_hub = intercept_hub = slope_ols = intercept_ols = float("nan")
        if int(included.sum()) >= 2:
            xg, yg = x_pat[included], y_pat[included]
            s_h, i_h = vectorised_huber(xg, yg.reshape(-1, 1))
            s_o, i_o = vectorised_ols(xg, yg.reshape(-1, 1))
            slope_hub = float(s_h[0]); intercept_hub = float(i_h[0])
            slope_ols = float(s_o[0]); intercept_ols = float(i_o[0])
            ki_hub = slope_hub * 6000.0; vp_hub = intercept_hub * 100.0
            ki_ols = slope_ols * 6000.0; vp_ols = intercept_ols * 100.0

        # ─── figure ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30,
                              height_ratios=[1, 1.4, 0.9])

        # Top: time-domain curves
        ax = fig.add_subplot(gs[0, :])
        ax.plot(t, c_a, "tab:blue", lw=1.5, label=f"Cₐ (mM)  peak={c_a.max():.3f}")
        ax2 = ax.twinx()
        ax2.plot(t, c_t, "tab:green", lw=1.2,
                 label=f"Cₜ (mM)  peak={c_t.max():.4f}")
        if tail_end > tail_start:
            ax.axvspan(t[tail_start], t[min(tail_end - 1, n - 1)],
                       color="tab:red", alpha=0.10,
                       label=f"tail window  frames {tail_start}–{tail_end}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Cₐ (mM)", color="tab:blue")
        ax2.set_ylabel("Cₜ (mM)", color="tab:green")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{ctx.label}   —   AIF + tissue curves (tail window shaded)")

        # Middle: Patlak (x, y)
        ax = fig.add_subplot(gs[1, :])
        ax.scatter(x_pat[excluded], y_pat[excluded], s=10, c="tab:gray",
                   alpha=0.40, label=f"excluded (n={int(excluded.sum())})")
        ax.scatter(x_pat[included], y_pat[included], s=24, c="tab:red",
                   edgecolors="darkred", linewidths=0.4, alpha=0.9,
                   label=f"included (n={int(included.sum())})")
        if int(included.sum()) >= 2:
            xs = np.linspace(x_pat[included].min(), x_pat[included].max(), 50)
            ax.plot(xs, intercept_hub + slope_hub * xs, "tab:blue", lw=1.8,
                    label=f"Huber  Ki={ki_hub:.4f}, vb={vp_hub:.3f}")
            ax.plot(xs, intercept_ols + slope_ols * xs, "tab:green", lw=1.0,
                    ls="--", label=f"OLS    Ki={ki_ols:.4f}, vb={vp_ols:.3f}")
        # zoom y to included if any
        if included.any():
            y_inc = y_pat[included]
            ypad = 0.15 * (y_inc.max() - y_inc.min() + 1e-6)
            ax.set_ylim(y_inc.min() - ypad, y_inc.max() + ypad)
            x_inc = x_pat[included]
            xpad = 0.05 * (x_inc.max() - x_inc.min() + 1e-6)
            ax.set_xlim(x_inc.min() - xpad, x_inc.max() + xpad)
        ax.set_xlabel(r"$x = \int_0^t C_a\,d\tau\,/\,C_a(t)$  (s)")
        ax.set_ylabel(r"$y = C_t / C_a$")
        ax.set_title(
            f"Patlak coordinates  —  tail = {tail_mode}, "
            f"x_max = {float(np.nanmax(x_pat[ok_mask])):.2f}"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        # Bottom: residuals from the Huber fit
        ax = fig.add_subplot(gs[2, :])
        if int(included.sum()) >= 2:
            xg = x_pat[included]; yg = y_pat[included]
            resid = yg - (intercept_hub + slope_hub * xg)
            ax.axhline(0, color="k", lw=0.6)
            ax.scatter(xg, resid, s=22, c="tab:red", alpha=0.75)
            ax.set_ylabel("residual (Huber)")
            ax.set_xlabel("Patlak x")
            mad = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
            ax.set_title(
                f"residuals  —  MAD={mad:.4g},  Ki(Huber)={ki_hub:.4f},  "
                f"Ki(OLS)={ki_ols:.4f}  →  Δ={ki_hub - ki_ols:+.4f}"
            )
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "not enough included points", ha="center",
                    va="center", transform=ax.transAxes)
            ax.axis("off")

        fig.suptitle(f"Patlak diagnostic — {ctx.label}", fontsize=12, y=0.995)
        ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ctx.out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)


PLUGIN = _PatlakDiagnostic()
