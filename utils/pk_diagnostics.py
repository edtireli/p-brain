"""PK diagnostic grid plot.

Generates a 4×3 panel figure for a representative voxel showing:

  Tikhonov deconvolution panels (rows 0-2):
  1. Parameter map + crosshair
  2. L-curve (log-res vs log-reg)
  3. L-curve curvature κ(λ)
  4. Tikhonov fit in time domain (Ct vs Ct_fit)
  5. Residue function r(t) and f = max(r[1:fwin])
  6. Norms vs λ  (residual and regularisation)
  7. R(t) and 1-R(t)
  8. h(t) normalised  + mean/median/CTH
  9. Cumulative hazard H(t) = −log(R(t))

  Patlak panels (row 3):
  10. Patlak regression plot (x_patlak vs y_patlak)
  11. Patlak fit residuals
  12. Summary text box with all PK metrics
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fmt(v: float, fmt: str = ".2f") -> str:
    """Format a float, returning 'NaN' if not finite."""
    if not np.isfinite(v):
        return "NaN"
    return f"{v:{fmt}}"


def write_pk_diagnostic_grid(
    diag: dict,
    *,
    patlak: dict | None = None,
    map_2d: np.ndarray | None = None,
    voxel_xy: tuple[int, int] | None = None,
    map_label: str = "CTH (tik) [s]",
    save_path: str,
) -> None:
    """Render and save the 4×3 PK diagnostic grid plot.

    Parameters
    ----------
    diag : dict
        Output of ``kinetic_models.solve_single_voxel_diagnostic()``.
    patlak : dict, optional
        Patlak diagnostic data with keys: ``ki``, ``vp``, ``sd_ki``,
        ``x_patlak``, ``y_patlak``, ``good_mask``.
    map_2d : ndarray, optional
        2-D parametric map slice to show in the top-left panel.
    voxel_xy : (x, y), optional
        Coordinates of the diagnostic voxel within *map_2d*.
    map_label : str
        Colourbar label for the map panel.
    save_path : str
        Output PNG path.
    """

    time_s = np.asarray(diag["time_s"], dtype=float)
    ct = np.asarray(diag["ct"], dtype=float)
    ct_fit = np.asarray(diag["ct_fit"], dtype=float)
    lambdas = np.asarray(diag["lambdas"], dtype=float)
    log_res = np.asarray(diag["log_res"], dtype=float)
    log_reg = np.asarray(diag["log_reg"], dtype=float)
    kappa = np.asarray(diag["kappa"], dtype=float)
    lambda_opt = float(diag["lambda_opt"])
    best_idx = int(diag["best_idx"])
    rf = np.asarray(diag["rf"], dtype=float)
    residue = np.asarray(diag["residue"], dtype=float)
    h = np.asarray(diag["h"], dtype=float)
    mtt = float(diag["mtt"])
    cth = float(diag["cth"])
    cbf = float(diag["cbf"])
    cbv = float(diag["cbv"])
    offset_s = float(diag["offset_s"])
    dt = float(diag["dt"])
    res_norms = np.asarray(diag["res_norms"], dtype=float)
    reg_norms = np.asarray(diag["reg_norms"], dtype=float)

    # Patlak data (may be None).
    has_patlak = patlak is not None
    if has_patlak:
        ki = float(patlak.get("ki", float("nan")))
        vp = float(patlak.get("vp", float("nan")))
        sd_ki = float(patlak.get("sd_ki", float("nan")))
        x_pat = np.asarray(patlak.get("x_patlak", []), dtype=float)
        y_pat = np.asarray(patlak.get("y_patlak", []), dtype=float)
        good = np.asarray(patlak.get("good_mask", []), dtype=bool)
    else:
        ki = vp = sd_ki = float("nan")
        x_pat = y_pat = np.zeros(0)
        good = np.zeros(0, dtype=bool)

    # Time axes.
    t_frames = time_s
    t_residue = np.arange(residue.size, dtype=float) * dt

    # Enforce non-neg / monotone residue for display.
    R = np.clip(residue, 0.0, None)
    R = np.minimum.accumulate(R)

    fig, axs = plt.subplots(4, 3, figsize=(16, 16), dpi=150)

    vx, vy = (voxel_xy if voxel_xy else (0, 0))
    title_parts = [
        f"PK diagnostics @ (x={vx}, y={vy}), "
        f"offset={offset_s:.1f}s, "
        f"λ²={lambda_opt**2:.2f}",
    ]
    tik_line = (
        f"CBF={_fmt(cbf, '.1f')} ml/100g/min   "
        f"MTT={_fmt(mtt, '.2f')} s   "
        f"CTH={_fmt(cth, '.2f')} s   "
        f"CBV={_fmt(cbv, '.2f')} ml/100g"
    )
    pat_line = (
        f"Ki={_fmt(ki, '.3f')} ml/100g/min   "
        f"vp={_fmt(vp, '.2f')} ml/100g   "
        f"SD(Ki)={_fmt(sd_ki, '.3f')}"
    )
    title_parts.append(tik_line)
    title_parts.append(pat_line)
    fig.suptitle("\n".join(title_parts), fontsize=10, fontweight="bold")

    # ── Row 0: Map, L-curve, Curvature ──────────────────────────

    # (0,0) Parameter map + crosshair
    ax = axs[0, 0]
    if map_2d is not None:
        display = np.rot90(map_2d)
        im = ax.imshow(display, cmap="viridis", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        h_map, w_map = map_2d.shape
        rx, ry = vy, h_map - 1 - vx  # rot90 coordinate transform
        ax.plot(rx, ry, "ko", markersize=6, markerfacecolor="none", markeredgewidth=1.5)
        ax.axhline(ry, color="k", linewidth=0.3, alpha=0.5)
        ax.axvline(rx, color="k", linewidth=0.3, alpha=0.5)
    ax.set_title(map_label, fontsize=9)
    ax.tick_params(labelsize=7)

    # (0,1) L-curve (log-res vs log-reg)
    ax = axs[0, 1]
    ax.plot(log_res, log_reg, "k.-", markersize=3, linewidth=0.7)
    if 0 <= best_idx < log_res.size:
        ax.plot(log_res[best_idx], log_reg[best_idx], "ro", markersize=8,
                markerfacecolor="none", markeredgewidth=2,
                label=f"λ²={lambda_opt**2:.2f}")
    ax.set_xlabel(r"log $\|Dx - b\|^2$", fontsize=8)
    ax.set_ylabel(r"log $\|Lx\|^2$", fontsize=8)
    ax.set_title("L-curve (log-res vs log-reg)", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # (0,2) L-curve curvature κ(λ)
    ax = axs[0, 2]
    kappa_lambdas = lambdas[1:-1] if kappa.size == lambdas.size - 2 else lambdas[:kappa.size]
    ax.plot(kappa_lambdas, kappa, "k-", linewidth=0.8)
    if 0 <= best_idx < kappa_lambdas.size:
        ax.axvline(kappa_lambdas[best_idx], color="r", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("λ", fontsize=8)
    ax.set_ylabel("κ", fontsize=8)
    ax.set_title("L-curve curvature κ(λ)", fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Row 1: Tikhonov fit, rf(t), Norms ────────────────────────

    # (1,0) Tikhonov fit in time domain
    ax = axs[1, 0]
    ax.plot(t_frames, ct, "k-", linewidth=0.8, label="Ct")
    ax.plot(t_frames, ct_fit, "r--", linewidth=0.8, label="Ct_fit")
    ax.set_xlabel("t [s]", fontsize=8)
    ax.set_ylabel("mM", fontsize=8)
    ax.set_title("Tikhonov fit in time domain", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # (1,1) Residue function r(t) and f = max(r[1:fwin])
    ax = axs[1, 1]
    fwin = min(10, rf.size)
    ax.plot(t_residue, rf, "k-", linewidth=0.8)
    ax.axhspan(0, 0, color="gray", alpha=0.1)
    if fwin > 0 and fwin < len(t_residue):
        ax.axvspan(0, t_residue[fwin - 1], color="lightgray", alpha=0.3)
    ax.set_xlabel("t [s]", fontsize=8)
    ax.set_ylabel("1/s", fontsize=8)
    ax.set_title("Residue function rf(t) and f=max(rf[1:fwin])", fontsize=9)
    ax.tick_params(labelsize=7)

    # (1,2) Norms vs λ
    ax = axs[1, 2]
    ax.semilogy(lambdas, res_norms, "k-", linewidth=0.8, label="res")
    ax.semilogy(lambdas, reg_norms, "b-", linewidth=0.8, label="reg")
    if 0 <= best_idx < lambdas.size:
        ax.axvline(lambdas[best_idx], color="r", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("λ", fontsize=8)
    ax.set_title("Norms vs λ", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # ── Row 2: R(t), h(t), Cumulative hazard ─────────────────────

    # (2,0) R(t) and 1-R(t)
    ax = axs[2, 0]
    ax.plot(t_residue, R, "k-o", markersize=2, linewidth=0.5, alpha=0.5, label="R(t) constrained")
    try:
        from scipy.ndimage import uniform_filter1d
        R_smooth = uniform_filter1d(R, size=max(3, R.size // 20))
    except Exception:
        R_smooth = R
    ax.plot(t_residue, R_smooth, "k-", linewidth=1.2, label="R(t) smooth")
    one_minus_R = 1.0 - R_smooth
    ax.plot(t_residue, one_minus_R, "b--", linewidth=0.8, label="1-R(t) smooth")
    ax.set_xlabel("t [s]", fontsize=8)
    ax.set_ylabel("unitless", fontsize=8)
    ax.set_title("Residue extraction: R(t) and 1-R(t)", fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)

    # (2,1) h(t) normalised + mean/median/CTH
    ax = axs[2, 1]
    if np.any(np.isfinite(h)):
        ax.plot(t_residue[:h.size], h[:t_residue.size], "k-", linewidth=0.8, label="h(t) (norm)")
        if np.isfinite(mtt):
            ax.axvline(mtt, color="b", linestyle="--", linewidth=0.8,
                       label=f"mean={mtt:.1f}s")
        if np.isfinite(cth):
            ax.axvline(mtt + cth, color="gray", linestyle=":", linewidth=0.8,
                       label=f"sCTH={cth:.1f}s")
        if np.isfinite(mtt) and np.isfinite(cth):
            ax.axvline(mtt - cth, color="gray", linestyle=":", linewidth=0.8,
                       label=f"mean-s={mtt - cth:.1f}s")
        ax.legend(fontsize=6)
    else:
        ax.text(0.5, 0.5, "h(t) unavailable", transform=ax.transAxes, ha="center")
    ax.set_xlabel("t [s]", fontsize=8)
    ax.set_ylabel("h(t) [1/s]", fontsize=8)
    ax.set_title("h(t) (normalized) + mean/median/CTH", fontsize=9)
    ax.tick_params(labelsize=7)

    # (2,2) Cumulative hazard H(t) = −log(R(t))
    ax = axs[2, 2]
    R_safe = np.clip(R, 1e-30, None)
    H_cumul = -np.log(R_safe)
    ax.plot(t_residue, H_cumul, "m-", linewidth=1.0)
    ax.set_xlabel("t [s]", fontsize=8)
    ax.set_ylabel("unitless", fontsize=8)
    ax.set_title("Cumulative hazard H(t) = −log(R(t))", fontsize=9)
    ax.tick_params(labelsize=7)

    # ── Row 3: Patlak regression, residuals, summary ─────────────

    # (3,0) Patlak regression plot
    ax = axs[3, 0]
    if x_pat.size > 0 and y_pat.size > 0:
        # Excluded points (gray).
        excluded = ~good if good.size == x_pat.size else np.zeros(x_pat.size, dtype=bool)
        if np.any(excluded):
            ax.plot(x_pat[excluded], y_pat[excluded], ".", color="lightgray",
                    markersize=3, alpha=0.5, label="excluded")
        # Included points (black).
        if np.any(good):
            ax.plot(x_pat[good], y_pat[good], "k.", markersize=4, label="used")
        # Regression line.
        if np.isfinite(ki) and np.isfinite(vp) and np.any(good):
            xg = x_pat[good]
            x_line = np.linspace(float(np.nanmin(xg)), float(np.nanmax(xg)), 100)
            # Ki is in ml/100g/min (*6000 from raw slope); vp in ml/100g (*100).
            y_line = (ki / 6000.0) * x_line + (vp / 100.0)
            ax.plot(x_line, y_line, "r-", linewidth=1.2,
                    label=f"Ki={_fmt(ki, '.3f')}, vp={_fmt(vp, '.1f')}")
        ax.legend(fontsize=6)
    else:
        ax.text(0.5, 0.5, "Patlak data unavailable", transform=ax.transAxes,
                ha="center", fontsize=9)
    ax.set_xlabel("∫Ca / Ca(t)", fontsize=8)
    ax.set_ylabel("Ct / Ca(t)", fontsize=8)
    ax.set_title("Patlak regression", fontsize=9)
    ax.tick_params(labelsize=7)

    # (3,1) Patlak fit residuals
    ax = axs[3, 1]
    if x_pat.size > 0 and y_pat.size > 0 and np.isfinite(ki) and np.isfinite(vp) and np.any(good):
        y_pred = (ki / 6000.0) * x_pat + (vp / 100.0)
        pat_resid = y_pat - y_pred
        ax.plot(x_pat[good], pat_resid[good], "k.", markersize=4)
        ax.axhline(0, color="r", linestyle="--", linewidth=0.6, alpha=0.7)
        # ±1 SD band.
        if np.any(good):
            resid_std = float(np.nanstd(pat_resid[good]))
            if np.isfinite(resid_std) and resid_std > 0:
                ax.axhline(resid_std, color="gray", linestyle=":", linewidth=0.5)
                ax.axhline(-resid_std, color="gray", linestyle=":", linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "Patlak residuals unavailable", transform=ax.transAxes,
                ha="center", fontsize=9)
    ax.set_xlabel("∫Ca / Ca(t)", fontsize=8)
    ax.set_ylabel("residual", fontsize=8)
    ax.set_title("Patlak fit residuals", fontsize=9)
    ax.tick_params(labelsize=7)

    # (3,2) Summary text box
    ax = axs[3, 2]
    ax.axis("off")
    lines = [
        "── Tikhonov deconvolution ──",
        f"  CBF  = {_fmt(cbf, '.2f')} ml/100g/min",
        f"  MTT  = {_fmt(mtt, '.3f')} s",
        f"  CTH  = {_fmt(cth, '.3f')} s",
        f"  CBV  = {_fmt(cbv, '.3f')} ml/100g",
        f"  λ_opt = {_fmt(lambda_opt, '.4f')}  (λ²={_fmt(lambda_opt**2, '.4f')})",
        f"  offset = {_fmt(offset_s, '.2f')} s",
        "",
        "── Patlak ──",
        f"  Ki   = {_fmt(ki, '.4f')} ml/100g/min",
        f"  vp   = {_fmt(vp, '.3f')} ml/100g",
        f"  SD(Ki) = {_fmt(sd_ki, '.4f')}",
    ]
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("PK summary", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pk_diag] Saved {save_path}")


def pick_diagnostic_voxel(
    cbf_map: np.ndarray,
    mtt_map: np.ndarray | None = None,
    *,
    slice_idx: int | None = None,
    cbf_percentile: float = 90.0,
) -> tuple[int, int, int]:
    """Choose a representative voxel for the diagnostic plot.

    Strategy: among the middle 50 % of slices, pick the voxel closest to
    the *cbf_percentile*-th percentile of CBF that also has a finite,
    positive MTT (if an MTT map is supplied).  This avoids outlier voxels
    with extremely high CBF and zero MTT.

    If *slice_idx* is provided, restrict to that slice.

    Returns (x, y, z).
    """

    cbf = np.asarray(cbf_map, dtype=float)
    if cbf.ndim != 3:
        raise ValueError("cbf_map must be 3D")

    nz = cbf.shape[2]
    if slice_idx is not None:
        z0, z1 = int(slice_idx), int(slice_idx) + 1
    else:
        z0 = max(0, nz // 4)
        z1 = min(nz, nz - nz // 4)
        if z1 <= z0:
            z0, z1 = 0, nz

    sub = cbf[:, :, z0:z1].copy()

    # Build a validity mask.
    valid = np.isfinite(sub) & (sub > 0)
    if mtt_map is not None:
        mtt_sub = np.asarray(mtt_map, dtype=float)[:, :, z0:z1]
        valid &= np.isfinite(mtt_sub) & (mtt_sub > 0)

    if not np.any(valid):
        # Fallback: just pick max finite CBF.
        sub[~np.isfinite(sub)] = -np.inf
        flat = int(np.argmax(sub))
        x, y, zrel = np.unravel_index(flat, sub.shape)
        return int(x), int(y), int(z0 + zrel)

    # Target the requested percentile of valid CBF values.
    target = float(np.nanpercentile(sub[valid], cbf_percentile))
    diff = np.full_like(sub, np.inf)
    diff[valid] = np.abs(sub[valid] - target)
    flat = int(np.argmin(diff))
    x, y, zrel = np.unravel_index(flat, sub.shape)
    z = z0 + zrel
    return int(x), int(y), int(z)
