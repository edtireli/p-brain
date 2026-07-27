"""Empirical-Bayes Tikhonov diagnostic plot.

Per-curve 2×2 figure that visualises the evidence-based deconvolution and
its uncertainty:

  (0,0) AIF + measured Cₜ + Bayes fit (R² annotated)
  (0,1) recovered residue R(t) over the early window (0–60 s)
  (1,0) log-evidence Z(λ) over the λ grid with λ* marked — confirms the
        interior maximum (the whole point of the method)
  (1,1) CBF / MTT / CTH bar chart with ± posterior SD error bars
        (λ-marginalised sampling on this single curve)

There is only one ``tikhonov`` plug-in now, so this is not reached by key
lookup: ``diagnostics/tikhonov.py`` delegates here when the run used
``lambda_selection="evidence"`` (i.e. ``preset="bayes"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import Diagnostic, DiagnosticContext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _design(t, ca, dt):
    """Rebuild the B-spline + Toeplitz design (mirrors build_tikhonov_solver)."""
    from pbrain.models.tikhonov import (
        _toeplitz_ca_matrix, _bspline_basis, _get_l_diff,
    )
    n = t.size
    M = 4
    K = np.arange(0.0, float(t[-1]) + 1e-12, 5.0 * dt, dtype=float)
    if K.size < 3:
        K = np.array([0.0, 5.0 * dt, 10.0 * dt], dtype=float)
    length_K = int(K.size - 2)
    n_basis = int(length_K + M)
    B = np.zeros((n_basis, n), dtype=float)
    for i in range(n):
        B[:, i] = _bspline_basis(M, K, length_K, float(t[i]))
    Ca_mat = _toeplitz_ca_matrix(ca)
    D = Ca_mat @ B.T
    DTD = D.T @ D
    LTL = _get_l_diff(n_basis, 1).T @ _get_l_diff(n_basis, 1)
    return B, Ca_mat, D, DTD, LTL, n_basis


@dataclass(frozen=True, slots=True)
class _TikhonovBayesDiagnostic:
    key: ClassVar[str] = "tikhonov_bayes"
    name: ClassVar[str] = "Empirical-Bayes Tikhonov diagnostic"
    description: ClassVar[str] = (
        "Bayes fit + residue + log-evidence(λ) interior optimum + "
        "CBF/MTT/CTH with posterior SD error bars."
    )
    accepts: ClassVar[dict[str, type]] = {"context": DiagnosticContext}
    produces: ClassVar[dict[str, type]] = {"png": str}
    model_key: ClassVar[str] = "tikhonov_bayes"

    def plot(self, ctx: DiagnosticContext) -> None:
        from scipy.linalg import cho_factor, cho_solve
        from pbrain.models.tikhonov import build_tikhonov_solver, _residue_metrics_batch

        c_t = np.asarray(ctx.c_tissue, dtype=float).ravel()
        c_a = np.asarray(ctx.c_input, dtype=float).ravel()
        t = np.asarray(ctx.t_s, dtype=float).ravel()
        n = int(min(c_t.size, c_a.size, t.size))
        c_t, c_a, t = c_t[:n], c_a[:n], t[:n]
        dt = float(t[1] - t[0])

        opts = dict(ctx.model_opts or {})
        n_lambdas = int(opts.get("n_lambdas", 121))
        lambda_min = float(opts.get("lambda_min", 1e-3))

        # Run the model on this curve with sampling on (single curve → cheap)
        solver = build_tikhonov_solver(
            t, c_a, lambda_selection="evidence", lambda_spacing="log",
            lambda_min=lambda_min, n_lambdas=n_lambdas,
            mtt_cth_method="residue_integral",
            uncertainty_samples=400, uncertainty_seed=0,
        )
        res = solver(c_t.reshape(-1, 1))
        cbf = float(res.cbf_ml_per_100g_min[0]); cbf_sd = float(res.cbf_sd[0])
        mtt = float(res.mtt_s[0]); mtt_sd = float(res.mtt_sd[0] if res.mtt_sd is not None else np.nan)
        cth = float(res.cth_s[0]); cth_sd = float(res.cth_sd[0] if res.cth_sd is not None else np.nan)
        lam_opt = float(res.lambda_opt[0])

        # Rebuild design to reconstruct the fit + residue + evidence curve
        B, Ca_mat, D, DTD, LTL, n_basis = _design(t, c_a, dt)
        rank_L = n_basis - 1
        # λ grid (log spacing, SVD-derived top — matches the solver)
        sing_max = float(np.max(np.linalg.svd(Ca_mat, compute_uv=False)))
        lambdas = np.geomspace(max(lambda_min, 1e-12), sing_max, n_lambdas)
        rhs = c_t @ D
        bTb = float(c_t @ c_t)
        rss = np.empty(n_lambdas); pen = np.empty(n_lambdas); logdetA = np.empty(n_lambdas)
        for li, lam in enumerate(lambdas):
            cf = cho_factor(DTD + lam * lam * LTL, lower=True, check_finite=False)
            x = cho_solve(cf, rhs, check_finite=False)
            rss[li] = max(bTb - 2 * (x @ rhs) + (x @ DTD @ x), 1e-30)
            pen[li] = max(x @ LTL @ x, 1e-30)
            logdetA[li] = 2.0 * float(np.sum(np.log(np.abs(np.diag(cf[0])))))
        log_ev = -0.5 * n * np.log(rss + lambdas ** 2 * pen) + rank_L * np.log(lambdas) - 0.5 * logdetA
        li_star = int(np.argmin(np.abs(lambdas - lam_opt)))

        # Fit + residue at λ*
        cf = cho_factor(DTD + lam_opt ** 2 * LTL, lower=True, check_finite=False)
        x = cho_solve(cf, rhs, check_finite=False)
        rf = (B.T @ x) / dt
        F = float(np.max(rf[:min(50, n)]))
        fit = dt * (Ca_mat @ rf)
        r2 = 1.0 - np.sum((c_t - fit) ** 2) / max(np.sum((c_t - c_t.mean()) ** 2), 1e-12)
        R = np.clip(rf / max(F, 1e-12), 0.0, None)
        R = np.minimum.accumulate(R)

        early = float(min(t[-1], 60.0))

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # (0,0) AIF + Cₜ + Bayes fit
        ax = axes[0, 0]
        ax.plot(t, c_a, "tab:blue", lw=1.4, label=f"Cₐ peak={c_a.max():.3f}")
        ax2 = ax.twinx()
        ax2.plot(t, c_t, ".", color="tab:green", ms=3.5, alpha=0.6, label="Cₜ measured")
        ax2.plot(t, fit, "k-", lw=1.5, label=f"Bayes fit (R²={r2:.3f})")
        ax.set_xlabel("time (s)"); ax.set_ylabel("Cₐ (mM)", color="tab:blue")
        ax2.set_ylabel("Cₜ (mM)", color="tab:green")
        ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
        ax.set_title("AIF + tissue + Bayes fit")

        # (0,1) residue
        ax = axes[0, 1]
        ax.plot(t, R, "tab:purple", lw=1.6)
        ax.set_xlim(0, early); ax.set_ylim(0, 1.05)
        ax.set_xlabel("time (s)"); ax.set_ylabel("R(t)")
        ax.set_title(f"Residue (monotonised)  F={F:.4g}→CBF={cbf:.1f}")
        ax.grid(alpha=0.3)

        # (1,0) log-evidence — the interior optimum
        ax = axes[1, 0]
        ax.plot(lambdas, log_ev, "tab:blue", lw=1.4)
        ax.axvline(lam_opt, color="r", ls=":", lw=1.2,
                   label=f"λ* = {lam_opt:.3g} (idx {li_star}/{n_lambdas-1})")
        ax.set_xscale("log"); ax.set_xlabel("λ"); ax.set_ylabel("log evidence Z(λ)")
        interior = 0 < li_star < n_lambdas - 1
        ax.set_title(f"Evidence functional  [{'interior ✓' if interior else 'BOUNDARY ✗'}]")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3, which="both")

        # (1,1) parameters with posterior SD
        ax = axes[1, 1]
        names = ["CBF\n(mL/100g/min)", "MTT (s)", "CTH (s)"]
        vals = [cbf, mtt, cth]; errs = [cbf_sd, mtt_sd, cth_sd]
        colors = ["tab:red", "tab:orange", "tab:green"]
        bars = ax.bar(names, vals, yerr=errs, capsize=8, color=colors, alpha=0.75,
                      error_kw=dict(lw=1.5, ecolor="0.2"))
        for b, v, e in zip(bars, vals, errs):
            etxt = f"±{e:.2f}" if np.isfinite(e) else ""
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}\n{etxt}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title("Parameters ± posterior SD (λ-marginalised, K=400)")
        ax.grid(alpha=0.3, axis="y")

        fig.suptitle(
            f"Empirical-Bayes Tikhonov — {ctx.label}   "
            f"CBF={cbf:.1f}±{cbf_sd:.2f}  MTT={mtt:.2f}±{mtt_sd:.2f}  "
            f"CTH={cth:.2f}±{cth_sd:.2f}  λ={lam_opt:.3g}",
            fontsize=12, y=0.995,
        )
        ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ctx.out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)


PLUGIN = _TikhonovBayesDiagnostic()
