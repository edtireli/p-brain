"""Tikhonov diagnostic plot.

Mirrors the legacy ``modules/tikhonov_plots._residue_and_fit`` reconstruction
exactly so the diagnostic shows the same fit that the production solver
``pbrain.models.tikhonov.build_tikhonov_solver`` actually produced:

  1. Run the production solver on the single curve → ``CBF, MTT, CTH, λ_opt``.
  2. Re-solve the linear system at the already-chosen λ to recover the
     B-spline coefficients (the production solver does not return them).
  3. ``rf = (B.T @ coeff) / dt``  (impulse response, 1/s)
  4. ``fit = Ca_mat @ rf * dt``   (Cₜ reconstruction — the ``*dt`` factor
     was missing in the previous diagnostic, which is why fits looked too
     small).
  5. ``R(t) = clip(rf / F, 0, ∞)`` → ``np.minimum.accumulate``
     (legacy semantics: physical residue is non-negative and non-increasing).

The figure is a 2×2 panel:

  · (0,0) AIF + measured Cₜ + Tikhonov fit
  · (0,1) Monotone residue R(t) (0–60 s)
  · (1,0) L-curve with the production-chosen λ marked
  · (1,1) Outflow distribution h(t) = -dR/dt (0–60 s) + MTT/CTH summary
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


def _residue_and_fit(
    time_s: np.ndarray,
    ca: np.ndarray,
    c_t: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Re-solve the validated Tikhonov system at a single λ and recover
    ``(fit, residue_monotone, rf, F, peak_idx)``.

    The math here is the same as the production solver's per-voxel path —
    we only need the B-spline coefficients which the production solver
    does not return, so we redo one Cholesky solve at the already-chosen λ.
    """
    from pbrain.models.tikhonov import (
        _bspline_basis, _toeplitz_ca_matrix, _get_l_diff,
    )
    from scipy.linalg import cho_factor, cho_solve

    t = np.asarray(time_s, dtype=float).reshape(-1)
    ca = np.asarray(ca, dtype=float).reshape(-1)
    c_t = np.asarray(c_t, dtype=float).reshape(-1)
    n = int(min(t.size, ca.size, c_t.size))
    t, ca, c_t = t[:n], ca[:n], c_t[:n]
    dt = float(t[1] - t[0])

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
    Lreg = _get_l_diff(n_basis, 1)
    LTL = Lreg.T @ Lreg

    A = DTD + (float(lam) ** 2) * LTL
    cho = cho_factor(A, lower=True, check_finite=False)
    rhs = D.T @ c_t
    coeff = cho_solve(cho, rhs, check_finite=False)

    rf = (B.T @ coeff) / dt                           # IRF (1/s)
    f_win = min(50, rf.size)
    F = float(np.max(rf[:f_win])) if f_win > 0 else 0.0
    if not np.isfinite(F) or F <= 0:
        F = 1.0
    peak_idx = int(np.argmax(rf[:f_win])) if f_win > 0 else 0

    fit = (Ca_mat @ rf) * dt                          # the *dt factor

    R = np.clip(rf / F, 0.0, None)
    R = np.minimum.accumulate(R)
    return fit.astype(float), R.astype(float), rf.astype(float), float(F), peak_idx


def _build_lambda_grid(time_s: np.ndarray, ca: np.ndarray,
                      n_lambdas: int, lambda_min: float,
                      lambda_spacing: str = "log") -> np.ndarray:
    """Rebuild the same λ grid the solver uses (for the L-curve panel)."""
    from pbrain.models.tikhonov import _toeplitz_ca_matrix
    Ca0 = _toeplitz_ca_matrix(np.asarray(ca, dtype=float).reshape(-1))
    sing = np.linalg.svd(Ca0, compute_uv=False)
    sing_max = float(np.max(sing)) if sing.size else lambda_min
    L_max = sing_max if sing_max >= lambda_min else lambda_min
    if (lambda_spacing or "log").strip().lower() == "log":
        return np.geomspace(max(lambda_min, 1e-12), L_max, int(n_lambdas), dtype=float)
    return np.linspace(lambda_min, L_max, int(n_lambdas), dtype=float)


def _lcurve_norms(time_s: np.ndarray, ca: np.ndarray, c_t: np.ndarray,
                  lambdas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (‖D x − Cₜ‖², ‖L x‖²) on the same λ grid the solver uses."""
    from pbrain.models.tikhonov import (
        _bspline_basis, _toeplitz_ca_matrix, _get_l_diff,
    )
    from scipy.linalg import cho_factor, cho_solve

    t = np.asarray(time_s, dtype=float).reshape(-1)
    ca = np.asarray(ca, dtype=float).reshape(-1)
    c_t = np.asarray(c_t, dtype=float).reshape(-1)
    n = int(min(t.size, ca.size, c_t.size))
    t, ca, c_t = t[:n], ca[:n], c_t[:n]
    dt = float(t[1] - t[0])

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
    Lreg = _get_l_diff(n_basis, 1)
    LTL = Lreg.T @ Lreg
    rhs = D.T @ c_t
    bTb = float(c_t @ c_t)

    res = np.zeros(lambdas.size)
    reg = np.zeros(lambdas.size)
    for li, lam in enumerate(lambdas):
        cho = cho_factor(DTD + (float(lam) ** 2) * LTL, lower=True, check_finite=False)
        X = cho_solve(cho, rhs, check_finite=False)
        res[li] = max(bTb - 2.0 * float(X @ rhs) + float(X @ DTD @ X), 1e-30)
        reg[li] = max(float(X @ LTL @ X), 1e-30)
    return res, reg


@dataclass(frozen=True, slots=True)
class _TikhonovDiagnostic:
    key: ClassVar[str] = "tikhonov"
    name: ClassVar[str] = "Tikhonov diagnostic plot"
    description: ClassVar[str] = (
        "AIF + tissue + fitted Cₜ, monotone residue R(t), L-curve with the "
        "production-chosen λ marked, outflow distribution h(t)."
    )
    accepts: ClassVar[dict[str, type]] = {"context": DiagnosticContext}
    produces: ClassVar[dict[str, type]] = {"png": str}
    model_key: ClassVar[str] = "tikhonov"

    def plot(self, ctx: DiagnosticContext) -> None:
        from pbrain.models.tikhonov import build_tikhonov_solver, PRESETS

        # Now that there is a single ``tikhonov`` plug-in, diagnostics resolve by
        # model key alone — so dispatch on the mode the run actually used. The
        # evidence path has its own figure (log-evidence Z(λ) with the interior
        # maximum, posterior SD bars) that this plot cannot show.
        _o = dict(ctx.model_opts or {})
        _sel = _o.get("lambda_selection")
        if _sel is None and _o.get("preset"):
            _sel = PRESETS.get(str(_o["preset"]).lower(), {}).get("lambda_selection")
        if _sel == "evidence":
            from pbrain.diagnostics.tikhonov_bayes import PLUGIN as _BAYES
            _BAYES.plot(ctx)
            return

        c_t = np.asarray(ctx.c_tissue, dtype=float).ravel()
        c_a = np.asarray(ctx.c_input, dtype=float).ravel()
        t = np.asarray(ctx.t_s, dtype=float).ravel()
        n = int(min(c_t.size, c_a.size, t.size))
        c_t, c_a, t = c_t[:n], c_a[:n], t[:n]

        opts = dict(ctx.model_opts or {})
        n_lambdas = int(opts.get("n_lambdas", 121))
        lambda_min = float(opts.get("lambda_min", 1e-3))
        lambda_spacing = str(opts.get("lambda_spacing", "log"))
        solver_opts = {
            k: opts[k] for k in (
                "n_lambdas", "lambda_min", "offset_grouping_s", "f_win",
                "tissue_density", "hematocrit", "plasma_derived_aif",
                "lambda_selection", "lambda_spacing",
            ) if k in opts
        }

        # ── 1) Run the production solver → CBF/MTT/CTH/λ
        solver = build_tikhonov_solver(t, c_a, **solver_opts)
        res = solver(c_t.reshape(-1, 1))
        cbf = float(res.cbf_ml_per_100g_min[0])
        mtt = float(res.mtt_s[0])
        cth = float(res.cth_s[0])
        lam_opt = float(res.lambda_opt[0])

        # ── 2) Reconstruct fit + monotone R(t) at the chosen λ
        fit, R_mono, rf, F, _peak = _residue_and_fit(t, c_a, c_t, lam_opt)
        sse = float(np.sum((c_t - fit) ** 2))

        # ── 3) L-curve on the same λ grid
        lambdas = _build_lambda_grid(t, c_a, n_lambdas, lambda_min, lambda_spacing)
        res_n, reg_n = _lcurve_norms(t, c_a, c_t, lambdas)
        lam_idx = int(np.argmin(np.abs(lambdas - lam_opt)))

        # ── 4) h(t) from monotone R
        t_mid = 0.5 * (t[:-1] + t[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.diff(R_mono) / np.diff(t)
        h = np.where(np.isfinite(h), np.maximum(h, 0.0), 0.0)
        area = float(trapezoid(h, t_mid))
        if area > 0:
            h = h / area

        # x-range for IRF/residue/outflow panels: Tikhonov is physically
        # meaningful only over the early window — clamp to 60s.
        early_xmax = float(min(t[-1], 60.0))

        # ─────────────────────── figure ───────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # (0, 0) AIF + Cₜ measured + Tikhonov fit
        ax = axes[0, 0]
        ax.plot(t, c_a, "tab:blue", lw=1.5, label=f"Cₐ peak={c_a.max():.3f}")
        ax2 = ax.twinx()
        ax2.plot(t, c_t, ".", color="tab:green", ms=3.5, alpha=0.55,
                 label="Cₜ measured")
        ax2.plot(t, fit, "k-", lw=1.5, label="Tikhonov fit")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Cₐ (mM)", color="tab:blue")
        ax2.set_ylabel("Cₜ (mM)", color="tab:green")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.set_title("AIF + tissue + fitted")

        # (0, 1) monotone residue R(t)
        ax = axes[0, 1]
        ax.plot(t, R_mono, "tab:purple", lw=1.5)
        ax.set_xlim(0, early_xmax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("R(t)")
        ax.set_title(f"Residue (monotonised)  —  F = {F:.4g} (1/s) → CBF = {cbf:.1f}")
        ax.grid(alpha=0.3)

        # (1, 0) L-curve
        ax = axes[1, 0]
        ax.plot(res_n, reg_n, "tab:blue", lw=1.0, marker=".", ms=3)
        ax.plot(res_n[lam_idx], reg_n[lam_idx], "ro", ms=10,
                label=f"λ_opt = {lam_opt:.3g} (idx {lam_idx}/{lambdas.size - 1})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("‖D x − Cₜ‖²"); ax.set_ylabel("‖L x‖²")
        ax.set_title("L-curve (Hansen)")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3, which="both")

        # (1, 1) h(t)
        ax = axes[1, 1]
        ax.plot(t_mid, h, "tab:orange", lw=1.4)
        ax.set_xlim(0, early_xmax)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("h(t)  (unit-area)")
        ax.set_title(f"Outflow h(t) = −dR/dt    MTT = {mtt:.2f}s   CTH = {cth:.2f}s")
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Tikhonov diagnostic — {ctx.label}   "
            f"CBF={cbf:.1f}  MTT={mtt:.2f}s  CTH={cth:.2f}s  "
            f"λ={lam_opt:.3g}  SSE={sse:.4g}",
            fontsize=12, y=0.995,
        )
        ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ctx.out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)


PLUGIN = _TikhonovDiagnostic()
