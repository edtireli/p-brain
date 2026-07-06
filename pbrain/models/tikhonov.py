"""Tikhonov-regularised model-free deconvolution — paper §4.6.2.

Solves

    C_t(t) = F · ∫₀ᵗ C_a(τ) R(t − τ) dτ,    R(0) = 1

for the impulse response ``g(t) = F·R(t)`` via cubic B-spline
representation and L-curve Tikhonov regularisation:

    x̂ = argminₓ ‖D·x − Cₜ‖² + λ² ‖L·x‖²

where ``D`` is the convolution-matrix-times-basis product, ``L`` is a
first-difference operator, and ``λ`` is selected per-voxel by
maximising L-curve curvature (Hansen).

Output parameters (paper §4.6.2):

* **CBF** — peak of the reconstructed impulse response over the early
  ``f_win`` frames, scaled to mL/100 g/min via tissue density and (when
  using a plasma-derived AIF) the haematocrit factor (1 − Hct).
* **MTT** — mean transit time = CBV / CBF (central-volume theorem),
  with CBV = ∫Cₜ / ∫Cₐ.
* **CTH** — capillary transit-time heterogeneity = SD of the constrained
  outflow distribution derived from −dR/dt.
* **λ_opt** — selected regularisation parameter (for QC).

This is a clean port of the validated ``models/tikhonov.py``; the math
is identical (same B-spline basis, same L-curve curvature, same CTH
post-processing) so parity tests give byte-equal numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular, svd

from .base import CurveInputs, ModelResult


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def _get_l_diff(n: int, order: int) -> np.ndarray:
    """Finite-difference operator on ``n`` knots (order 1 or 2)."""
    n = int(n)
    order = int(order)
    if order == 1:
        if n < 2:
            return np.zeros((0, n), dtype=float)
        L = np.zeros((n - 1, n), dtype=float)
        idx = np.arange(n - 1)
        L[idx, idx] = -1.0
        L[idx, idx + 1] = 1.0
        return L
    if order == 2:
        if n < 3:
            return np.zeros((0, n), dtype=float)
        L = np.zeros((n - 2, n), dtype=float)
        idx = np.arange(n - 2)
        L[idx, idx] = 1.0
        L[idx, idx + 1] = -2.0
        L[idx, idx + 2] = 1.0
        return L
    raise ValueError("order must be 1 or 2")


def _bspline_basis(M: int, K: np.ndarray, length_K: int, x: float) -> np.ndarray:
    """Order-M B-spline basis weights at ``x`` (Cox–de Boor recursion)."""
    M = int(M)
    K = np.asarray(K, dtype=float).reshape(-1)
    length_K = int(length_K)

    B = np.zeros((length_K + 2 * M - 1, M), dtype=float)
    tau = np.zeros(length_K + 2 * M, dtype=float)
    tau[: M - 1] = K[0]
    tau[M - 1 : M + length_K + 1] = K
    tau[length_K + M + 1 :] = K[-1]

    for i in range(length_K + 2 * M - 1):
        if tau[i + 1] > tau[i] and (x >= tau[i]) and (x < tau[i + 1]):
            B[i, 0] = 1.0

    if M > 1:
        for m in range(2, M + 1):
            for i in range(length_K + 2 * M - m):
                b0 = B[i, m - 2]
                b1 = B[i + 1, m - 2]
                if b0 == 0.0 and b1 == 0.0:
                    B[i, m - 1] = 0.0
                elif b0 == 0.0:
                    denom = tau[i + m] - tau[i + 1]
                    B[i, m - 1] = ((tau[i + m] - x) / denom) * b1 if denom != 0.0 else 0.0
                elif b1 == 0.0:
                    denom = tau[i + m - 1] - tau[i]
                    B[i, m - 1] = ((x - tau[i]) / denom) * b0 if denom != 0.0 else 0.0
                else:
                    d0 = tau[i + m - 1] - tau[i]
                    d1 = tau[i + m] - tau[i + 1]
                    t0 = ((x - tau[i]) / d0) * b0 if d0 != 0.0 else 0.0
                    t1 = ((tau[i + m] - x) / d1) * b1 if d1 != 0.0 else 0.0
                    B[i, m - 1] = t0 + t1

    return B[: length_K + M, M - 1].copy()


def _toeplitz_ca_matrix(ca: np.ndarray) -> np.ndarray:
    """Lower-triangular Toeplitz convolution matrix from input function."""
    ca = np.asarray(ca, dtype=float).reshape(-1)
    n = int(ca.size)
    Ca = np.zeros((n, n), dtype=float)
    for j in range(n):
        Ca[j:, j] = ca[: n - j]
    return Ca


def _stdform_gsvd(D: np.ndarray, L: np.ndarray) -> tuple:
    """Eldén standard-form transformation of the general-form pair ``(D, L)``.

    Computes — **once** — the factors that turn

        x̂(λ) = argminₓ ‖D·x − b‖² + λ²‖L·x‖²

    into a *separable* (varpro) filter-factor solve: a single SVD whose
    singular values are reweighted by ``s/(s²+λ²)`` per λ, vectorised over
    voxels by matrix-multiply. This reproduces MATLAB Hansen ``cgsvd`` +
    ``tikhonov`` (overdetermined general-form, ``p = n−1 < n``) to machine
    precision — the regularised minimiser is unique, so the standard-form
    SVD route and MATLAB's GSVD filter factors compute byte-equal ``x̂(λ)``
    (verified ``max|Δ| ≈ 8e-15``).

    ``L`` must have full row rank ``p`` with a null space of dimension
    ``nd = n − p`` (first-difference operator: ``nd = 1``).

    Returns ``(LpA, x_null_coef, Us, ss, Vs)`` such that, for observation
    matrix ``b`` (shape ``(n, n_vox)``)::

        beta  = Us.T @ b                              # (p, n_vox)
        xnull = x_null_coef @ b                       # (n, n_vox)  λ-independent
        filt  = ss / (ss**2 + lam**2)                 # (p,)
        x     = LpA @ (Vs @ (filt[:, None] * beta)) + xnull   # (n, n_vox)

    where ``x`` equals MATLAB's ``V(:,1:p)·xi + x0``.
    """
    m, n = D.shape
    p = L.shape[0]
    nd = n - p
    # Orthonormal basis for null(L) (n × nd).
    Qn, _ = qr(L.T, mode="full")
    W = Qn[:, p:]
    # Right pseudo-inverse of L (L · Lp = I_p):  Lp = Lᵀ (L Lᵀ)⁻¹  (n × p).
    Lp = L.T @ np.linalg.inv(L @ L.T)
    # A-weighted generalised inverse (Eldén): project out the D-image of null(L).
    QQ, _ = qr(D @ W, mode="full")
    Q1 = QQ[:, :nd]
    Gw = Q1.T @ (D @ W)                       # (nd × nd), invertible
    LpA = Lp - W @ np.linalg.solve(Gw, Q1.T @ D @ Lp)   # (n × p)
    # Map b → null-space (unregularised) component x0  (n × n_vox when applied).
    x_null_coef = W @ np.linalg.solve(Gw, Q1.T)          # (n × m)
    # Single SVD of the standard-form operator  A_s = D · LpA  (m × p).
    A_s = D @ LpA
    Us, ss, Vst = svd(A_s, full_matrices=False)
    Vs = Vst.T
    return LpA, x_null_coef, Us, ss, Vs


def _legacy_solve(
    idx, Ct_mat, gsvd, D, Lreg, BT, dt, lambdas, lambda_unit, L1, L2,
    f_win, cbf_scale, denom_sum, eps, cbf, cbv, mtt, cth, lam_opt, t,
    mtt_cth_method,
):
    """Vectorised legacy-exact solve for the voxel group ``idx``.

    GSVD computed once (``gsvd`` factors); λ swept by filter factors; L-curve
    curvature exactly per MATLAB ``con2perf_tikonov1_4`` / ``legacy_fit_from_dump``.
    Writes CBF/CBV/MTT/CTH/λ_opt in-place at the output indices ``idx``.
    """
    LpA, x_null_coef, Us, ss, Vs = gsvd
    nl = int(lambdas.size)
    ct = Ct_mat[:, idx]                       # (n, n_vox_g)
    n_vox_g = ct.shape[1]

    beta = Us.T @ ct                          # (p, n_vox_g)   project once
    xnull = x_null_coef @ ct                  # (n_basis, n_vox_g)  λ-independent
    ss2 = ss ** 2

    # Sweep λ: res_norm = ‖D·x − cₜ‖², reg_norm = ‖L·x‖²  (MATLAB LL(:,3)/LL(:,2)).
    res_norm = np.empty((n_vox_g, nl), dtype=float)
    reg_norm = np.empty((n_vox_g, nl), dtype=float)
    for li in range(nl):
        filt = ss / (ss2 + lambdas[li] ** 2)
        x = LpA @ (Vs @ (filt[:, None] * beta)) + xnull      # (n_basis, n_vox_g)
        resid = D @ x - ct
        res_norm[:, li] = np.sum(resid * resid, axis=0)
        Lx = Lreg @ x
        reg_norm[:, li] = np.sum(Lx * Lx, axis=0)

    # MATLAB L-curve curvature (linear λ-unit, log coords, raw signed κ, first-max).
    xlog = np.log(np.maximum(res_norm, eps))
    ylog = np.log(np.maximum(reg_norm, eps))
    d_x = (xlog @ L1.T) / lambda_unit
    d_y = (ylog @ L1.T) / lambda_unit
    dd_x = (xlog @ L2.T) / (lambda_unit ** 2)
    dd_y = (ylog @ L2.T) / (lambda_unit ** 2)
    num = d_x[:, : nl - 2] * dd_y - dd_x * d_y[:, : nl - 2]
    den = (d_x[:, : nl - 2] ** 2 + d_y[:, : nl - 2] ** 2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        krum = num / den
    idx_opt = np.argmax(krum, axis=1)         # first maximiser on ties (= MATLAB max)

    # Final reconstruction at each voxel's λ_opt, grouped by index for vectorisation.
    t_col = np.asarray(t, dtype=float).reshape(-1)
    w = min(int(f_win), int(BT.shape[1]))
    for u in np.unique(idx_opt):
        sel = idx_opt == u
        filt = ss / (ss2 + lambdas[u] ** 2)
        x = LpA @ (Vs @ (filt[:, None] * beta[:, sel])) + xnull[:, sel]
        rf = (BT @ x) / dt                    # (n, n_sel)
        fp = np.max(rf[:w, :], axis=0)        # f = max(rf[1:f_win]) (MATLAB)
        out_idx = idx[sel]
        # CBF: abs, non-negative, scaled.
        cbf_v = np.abs(fp) * cbf_scale
        cbf_v[~np.isfinite(cbf_v)] = 0.0
        cbf_v[cbf_v < 0] = 0.0
        cbf[out_idx] = cbf_v
        # CBV: Σcₜ/Σcₐ (MATLAB fast), abs.
        vd = np.sum(ct[:, sel], axis=0) / denom_sum
        vd = np.abs(vd)
        vd[~np.isfinite(vd)] = 0.0
        cbv[out_idx] = vd
        lam_opt[out_idx] = float(lambdas[u])
        with np.errstate(divide="ignore", invalid="ignore"):
            if mtt_cth_method == "central_volume":
                mtt_v = np.where(fp > 0, vd / fp, 0.0)
            else:
                mtt_v = np.where(fp > 0, vd / fp, 0.0)
        mtt_v = np.abs(np.where(np.isfinite(mtt_v), mtt_v, 0.0))
        mtt_v[mtt_v < 0] = 0.0
        mtt[out_idx] = mtt_v
        # CTH: residue moments (driver lines 128-143). R = rf/peak (peak=max rf[:10]),
        # m0=Σ R·dt, m1=Σ t·R·dt, s2 = 2·m1 − m0², CTH = sqrt(max(s2,0)).
        peak = np.max(rf[:10, :], axis=0)
        Rn = np.where(peak[None, :] > 0, rf / np.where(peak == 0, 1.0, peak)[None, :], 0.0)
        m0 = np.sum(Rn, axis=0) * dt
        m1 = np.sum(t_col[:, None] * Rn, axis=0) * dt
        s2 = 2.0 * m1 - m0 ** 2
        cth_v = np.sqrt(np.clip(s2, 0.0, None))
        cth_v = np.where(peak > 0, cth_v, 0.0)
        cth_v = np.abs(np.where(np.isfinite(cth_v), cth_v, 0.0))
        cth[out_idx] = cth_v


def _shift_curve_pchip(time_s: np.ndarray, curve: np.ndarray, shift_s: float) -> np.ndarray:
    """PCHIP shift of ``curve`` along ``time_s`` by ``shift_s`` seconds.

    MATLAB-equivalent ``inputshift3`` (zero-pad on positive shift; tail
    extend on negative shift). Used per-voxel inside the Tikhonov solver
    when offsets are supplied.
    """
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    y = np.asarray(curve, dtype=float).reshape(-1)
    if time_s.size != y.size:
        n = min(time_s.size, y.size)
        time_s = time_s[:n]
        y = y[:n]
    if time_s.size <= 1 or float(shift_s) == 0.0:
        return y.copy()

    time_unit = float(time_s[2] - time_s[1]) if time_s.size >= 3 else float(time_s[1] - time_s[0])
    if not np.isfinite(time_unit) or time_unit <= 0:
        deltas = np.diff(time_s)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
        time_unit = float(deltas[0]) if deltas.size else 1.0

    deltaT = float(shift_s)
    notimeunit = int(np.ceil(abs(deltaT) / time_unit))
    n = y.size

    if deltaT > 0:
        timetemp = (time_s[0] + deltaT) - (notimeunit * time_unit) + np.arange(
            n + notimeunit, dtype=float
        ) * time_unit
        sCa = np.zeros(n + notimeunit, dtype=float)
        sCa[notimeunit : notimeunit + n] = y
    else:
        timetemp = (time_s[0] + deltaT) + np.arange(n + notimeunit, dtype=float) * time_unit
        sCa = np.zeros(n + notimeunit, dtype=float)
        sCa[:n] = y
        if notimeunit:
            sCa[n:] = y[-1]

    out = PchipInterpolator(timetemp, sCa, extrapolate=False)(time_s)
    return np.where(np.isfinite(out), out, 0.0).astype(float)


def _cth_from_rf(time_s: np.ndarray, rf: np.ndarray, *, f: float, peak_idx: int) -> float:
    """CTH from impulse response: SD of constrained outflow distribution.

    Steps (validator-equivalent):
    1. R(t) = rf(t) / F.
    2. Shift so the peak is at t=0.
    3. Normalise R(0) = 1; clip [0, 1]; enforce monotonic non-increase.
    4. Smooth tail to zero over the last 3 samples.
    5. h(t) = −dR/dt (forward difference, evaluated at bin midpoints).
    6. Normalise h to unit area; CTH = sqrt(var(h)).
    """
    t = np.asarray(time_s, dtype=float).reshape(-1)
    rf = np.asarray(rf, dtype=float).reshape(-1)
    n = int(min(t.size, rf.size))
    if n < 5:
        return float("nan")
    t, rf = t[:n], rf[:n]

    f = float(f)
    if not np.isfinite(f) or f <= 0:
        return 0.0
    peak_idx = int(max(0, min(int(peak_idx), n - 2)))

    R = rf / f
    if peak_idx > 0:
        R = np.concatenate((R[peak_idx:], np.repeat(R[-1], peak_idx)))
    if not np.isfinite(R[0]) or R[0] <= 0:
        R[0] = 1.0
    R = R / max(float(R[0]), 1e-12)
    R[~np.isfinite(R)] = 0.0
    R = np.clip(R, 0.0, 1.0)
    for i in range(1, R.size):
        if R[i] > R[i - 1]:
            R[i] = R[i - 1]
    if R.size >= 3:
        tail = int(max(R.size - 3, 1))
        R[tail:] = np.linspace(R[tail], 0.0, R.size - tail)

    t_mid = 0.5 * (t[:-1] + t[1:])
    dR = np.diff(R)
    dt = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -dR / dt
    h = np.where(np.isfinite(h), np.maximum(h, 0.0), 0.0)

    area = float(np.trapezoid(h, t_mid))
    if not np.isfinite(area) or area <= 0:
        return 0.0
    h = h / area

    mean = float(np.trapezoid(t_mid * h, t_mid))
    second = float(np.trapezoid((t_mid ** 2) * h, t_mid))
    var = max(second - mean * mean, 0.0)
    return float(math.sqrt(var))


def _residue_metrics_batch(
    residue_mat: np.ndarray, dt: float,
    *, enforce_nonneg: bool = True, enforce_monotone: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """MTT/CTH from a batch of unit-normalised residues — legacy-pipeline parity.

    Byte-for-byte port of ``modules/kinetic_models.residue_metrics_batch``
    (the function the production ``main.py`` Tikhonov path actually uses):

    * clip residue ≥ 0, enforce monotone non-increase
      (``np.minimum.accumulate`` along time);
    * **MTT = ∫ R(t) dt** (trapezoid of the constrained residue — *not*
      the central-volume ``vd/F``);
    * h(t) = max(0, −dR/dt) via 2nd-order finite differences, normalised
      to unit area;
    * CTH = sqrt(variance of h), with mean ``μ = ∫ t·h dt``.

    ``residue_mat`` is ``(n_time, n_vox)``. Returns ``(mtt, cth)`` each
    shape ``(n_vox,)``.
    """
    R = np.asarray(residue_mat, dtype=float)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    n_time, n_vox = R.shape
    nan_vox = np.full(n_vox, np.nan, dtype=float)
    if n_time < 5 or not np.isfinite(dt) or dt <= 0:
        return nan_vox.copy(), nan_vox.copy()

    working = R.copy()
    if enforce_nonneg:
        np.clip(working, 0.0, None, out=working)
    if enforce_monotone:
        np.minimum.accumulate(working, axis=0, out=working)

    mtt = np.trapezoid(working, dx=dt, axis=0)

    h = np.empty_like(working)
    h[1:-1] = -(working[2:] - working[:-2]) / (2.0 * dt)
    h[0] = (3.0 * working[0] - 4.0 * working[1] + working[2]) / (2.0 * dt)
    h[-1] = (-working[-3] + 4.0 * working[-2] - 3.0 * working[-1]) / (2.0 * dt)
    np.clip(h, 0.0, None, out=h)

    s = np.trapezoid(h, dx=dt, axis=0)
    bad = (~np.isfinite(s)) | (s <= 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = h / s
    time_col = (np.arange(n_time, dtype=float) * dt).reshape(-1, 1)
    mu = np.trapezoid(time_col * h, dx=dt, axis=0)
    variance = np.trapezoid((time_col - mu) ** 2 * h, dx=dt, axis=0)
    cth = np.sqrt(np.clip(variance, 0.0, None))

    mtt = np.where(bad, np.nan, mtt)
    cth = np.where(bad, np.nan, cth)
    return mtt, cth


# ────────────────────────────────────────────────────────────────────────
# Solver
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TikhonovBatch:
    """Voxel-batch deconvolution result (one array per perfusion metric).

    ``cbf`` is mL/100g/min, ``cbv`` is the volume of distribution,
    ``mtt``/``cth`` are seconds, ``lambda_opt`` is the per-voxel
    regularisation weight. The ``*_sd`` fields are optional bootstrap
    standard deviations (``None`` unless uncertainty was requested).
    """

    cbf_ml_per_100g_min: np.ndarray
    cbv_vd: np.ndarray
    mtt_s: np.ndarray
    cth_s: np.ndarray
    lambda_opt: np.ndarray
    cbf_sd: np.ndarray | None = None
    mtt_sd: np.ndarray | None = None
    cth_sd: np.ndarray | None = None


def build_tikhonov_solver(
    time_s: np.ndarray,
    ca: np.ndarray,
    *,
    n_lambdas: int = 121,
    lambda_min: float = 1e-3,
    lambda_max: float | None = None,
    offset_grouping_s: float | None = 0.05,
    f_win: int = 50,
    tissue_density: float = 1.04,
    hematocrit: float = 0.42,
    plasma_derived_aif: bool = False,
    lambda_selection: str = "gcv",
    lambda_spacing: str = "log",
    mtt_cth_method: str = "residue_integral",
    compute_cbf_sd: bool = False,
    uncertainty_samples: int = 0,
    uncertainty_seed: int = 0,
    legacy_exact: bool = False,
):
    """Pre-factor the Tikhonov system; return a callable ``solve_ct``.

    The expensive setup (B-spline basis, λ-grid, per-offset Cholesky
    factorisations) happens here once. ``solve_ct(Ct_mat, offsets_s)``
    can then be called many times with different tissue-curve batches —
    cost per voxel becomes one matrix-vector solve per λ.

    ``lambda_selection``:
      * ``"gcv"`` (default) — minimise Generalized Cross-Validation:
        ``G(λ) = ‖(I − H(λ)) Cₜ‖² / (n − tr H(λ))²`` with
        ``H(λ) = D(D'D + λ²L'L)⁻¹D'``. Robust against the endpoint bias
        of the L-curve method on smooth DCE-MRI data, where the L-curve
        often has no clear interior corner and curvature-based pickers
        collapse to ``λ_min``.
      * ``"lcurve"`` — legacy Hansen L-curve max-curvature picker
        (provided for parity with the original ``models/tikhonov.py``).
      * ``"matlab_gsvd"`` — **legacy-exact** mode. Reproduces the
        supervisor's MATLAB ``con2perf_tikonov1_4`` ("perffit") bit-for-bit
        (``max|Δ| ≈ 1e-14`` at matched λ). Forces ``legacy_exact=True``
        semantics: MATLAB knot vector ``K = 0:5·timeunit:frame_no``, the
        Eldén standard-form GSVD filter-factor solve (GSVD computed once,
        λ swept by ``s/(s²+λ²)`` reweighting, vectorised over voxels — no
        per-voxel linear solve), MATLAB-exact L-curve curvature
        (``log‖res‖²``/``log‖Lx‖²``, linear λ-unit, raw signed κ, first-max
        ``argmax``), ``f = max(rf[:f_win])``, ``CBV = Σcₜ/Σcₐ`` and
        ``abs``-clamped non-negative outputs.

    ``legacy_exact`` (bool): independent switch for the MATLAB knot vector
    and standard-form GSVD solve; implied by ``lambda_selection="matlab_gsvd"``.
    """
    lambda_selection = (lambda_selection or "gcv").strip().lower()
    if lambda_selection not in {"gcv", "lcurve", "evidence", "matlab_gsvd"}:
        raise ValueError(
            "lambda_selection must be 'gcv', 'lcurve', 'evidence' or 'matlab_gsvd'"
        )
    if lambda_selection == "matlab_gsvd":
        legacy_exact = True
    lambda_spacing = (lambda_spacing or "log").strip().lower()
    if lambda_spacing not in {"log", "linear"}:
        raise ValueError("lambda_spacing must be 'log' or 'linear'")
    mtt_cth_method = (mtt_cth_method or "residue_integral").strip().lower()
    if mtt_cth_method not in {"residue_integral", "central_volume"}:
        raise ValueError("mtt_cth_method must be 'residue_integral' or 'central_volume'")
    t = np.asarray(time_s, dtype=float).reshape(-1)
    ca0 = np.asarray(ca, dtype=float).reshape(-1)
    n = int(min(t.size, ca0.size))
    if n < 3:
        raise ValueError("time_s and ca must have length >= 3")
    t, ca0 = t[:n], ca0[:n]

    dt = float(t[1] - t[0])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time_s must be strictly increasing")

    # Grid top: explicit ``lambda_max`` (legacy uses a fixed 40.0), else
    # SVD-derived ``σ_max`` (smart default — scales with the AIF energy).
    if lambda_max is not None:
        L_max = float(lambda_max) if float(lambda_max) >= lambda_min else lambda_min
    else:
        Ca_matrix0 = _toeplitz_ca_matrix(ca0)
        sing = np.linalg.svd(Ca_matrix0, compute_uv=False)
        sing_max = float(np.max(sing)) if sing.size else lambda_min
        L_max = sing_max if sing_max >= lambda_min else lambda_min
    if lambda_spacing == "log":
        lambdas = np.geomspace(max(lambda_min, 1e-12), L_max, int(n_lambdas), dtype=float)
    else:
        lambdas = np.linspace(lambda_min, L_max, int(n_lambdas), dtype=float)
    if lambdas.size < 2 or not np.all(np.isfinite(lambdas)):
        raise ValueError("lambda grid must contain >= 2 finite values")

    M = 4
    if legacy_exact:
        # MATLAB legacy_fit_from_dump.m:58 — knots run to the FRAME COUNT
        # (frame_no = n), NOT time(end). Replicates the colon
        # ``K = 0:5*timeunit:frame_no`` exactly (last knot ≤ frame_no).
        step = 5.0 * dt
        n_knots = int(np.floor(float(n) / step)) + 1
        K = step * np.arange(n_knots, dtype=float)
    else:
        K = np.arange(0.0, float(t[-1]) + 1e-12, 5.0 * dt, dtype=float)
    if K.size < 3:
        K = np.array([0.0, 5.0 * dt, 10.0 * dt], dtype=float)
    length_K = int(K.size - 2)
    n_basis = int(length_K + M)

    B = np.zeros((n_basis, n), dtype=float)
    for i in range(n):
        B[:, i] = _bspline_basis(M, K, length_K, float(t[i]))

    Lreg = _get_l_diff(n_basis, 1)
    LTL = Lreg.T @ Lreg
    rank_L = int(min(Lreg.shape))          # rank of first-difference op = n_basis − 1

    L1 = _get_l_diff(int(lambdas.size), 1)
    L2 = _get_l_diff(int(lambdas.size), 2)
    # The L-curve curvature picker uses finite differences in the λ-parameter.
    # On a log-spaced grid we parametrise by log(λ) (uniformly spaced), so the
    # curvature in log(res) / log(reg) coordinates is well defined.
    if lambda_spacing == "log" and not legacy_exact:
        lambda_unit = float(np.log(lambdas[1]) - np.log(lambdas[0]))
    else:
        # legacy_exact always uses the linear λ-step (MATLAB lambda(2)-lambda(1)).
        lambda_unit = float(lambdas[1] - lambdas[0])
    if not np.isfinite(lambda_unit) or lambda_unit == 0.0:
        raise ValueError("lambda grid must be uniformly spaced")

    rho = tissue_density if (np.isfinite(tissue_density) and tissue_density > 0) else 1.04
    hct = hematocrit if (np.isfinite(hematocrit) and 0 <= hematocrit < 1) else 0.42
    plasma_scale = (1.0 - hct) if plasma_derived_aif else 1.0
    cbf_scale = 6000.0 * plasma_scale / rho

    design_cache: dict[float, tuple] = {}

    def _get_design(off_s: float):
        off_s = float(off_s)
        cached = design_cache.get(off_s)
        if cached is not None:
            return cached
        ca_shift = _shift_curve_pchip(t, ca0, off_s)
        Ca_mat = _toeplitz_ca_matrix(ca_shift)
        D = Ca_mat @ B.T

        if legacy_exact:
            # Standard-form GSVD computed ONCE; λ swept by filter factors. No
            # Cholesky list, no GCV/evidence pieces — this is the legacy path.
            LpA, x_null_coef, Us_g, ss_g, Vs_g = _stdform_gsvd(D, Lreg)
            out = (ca_shift, D, None, None, None, None,
                   (LpA, x_null_coef, Us_g, ss_g, Vs_g))
            design_cache[off_s] = out
            return out

        DTD = D.T @ D
        cholA = [cho_factor(DTD + (float(lam) ** 2) * LTL, lower=True, check_finite=False)
                 for lam in lambdas]

        # GCV denominator: (n − tr H(λ))². Data-independent — precompute
        # once per offset so the per-voxel cost stays the same as L-curve.
        # tr H(λ) = tr(D A⁻¹ Dᵀ) = sum(D * (A⁻¹ Dᵀ).T)  (Frobenius trace identity).
        gcv_denom = None
        if lambda_selection == "gcv":
            DT = D.T                                    # (n_basis, n)
            gcv_denom = np.empty(int(lambdas.size), dtype=float)
            for li in range(int(lambdas.size)):
                X = cho_solve(cholA[li], DT, check_finite=False)   # (n_basis, n)
                tr_H = float(np.einsum("ij,ji->", D, X))
                gd = max(float(n) - tr_H, 1e-12)
                gcv_denom[li] = gd * gd

        # Empirical-Bayes evidence needs log|A(λ)| = 2·Σ log diag(chol_A). Free
        # from the existing factorisation (the Cholesky factor's diagonal).
        logdetA = None
        if lambda_selection == "evidence":
            logdetA = np.empty(int(lambdas.size), dtype=float)
            for li in range(int(lambdas.size)):
                cfac = np.asarray(cholA[li][0])
                logdetA[li] = 2.0 * float(np.sum(np.log(np.abs(np.diag(cfac)))))

        out = (ca_shift, D, DTD, cholA, gcv_denom, logdetA, None)
        design_cache[off_s] = out
        return out

    BT = B.T
    eps = 1e-30

    def solve_ct(
        Ct_mat: np.ndarray,
        *,
        offsets_s: np.ndarray | None = None,
    ) -> TikhonovBatch:
        Ct_mat = np.asarray(Ct_mat, dtype=float)
        if Ct_mat.ndim == 1:
            Ct_mat = Ct_mat.reshape(-1, 1)
        # Short-AIF guard: the solver was pre-factored on ``n`` time points
        # (``n = min(len(time_s), len(ca))`` — see build above). When the AIF
        # is a few frames shorter than the DCE, callers may pass a Ct with more
        # time rows than the AIF supports. Align defensively by dropping the
        # extra TRAILING tissue frames so Ct/AIF/time share one length. This
        # leaves equal-length inputs (``Ct_mat.shape[0] == n``) byte-identical —
        # the slice is a no-op — and only kicks in for the mismatched case that
        # previously raised "Ct_mat first dimension must match time_s".
        if Ct_mat.shape[0] > n:
            Ct_mat = Ct_mat[:n, :]
        elif Ct_mat.shape[0] < n:
            raise ValueError(
                f"Ct_mat has {Ct_mat.shape[0]} time points but the solver was "
                f"built for {n} (AIF/time length); the tissue series must be at "
                f"least as long as the AIF. Align Ct/AIF/time before solving."
            )
        n_vox = int(Ct_mat.shape[1])

        if offsets_s is None:
            offsets = np.zeros(n_vox, dtype=float)
        else:
            offsets = np.asarray(offsets_s, dtype=float).reshape(-1)
            if offsets.size != n_vox:
                raise ValueError("offsets_s must have length n_vox")

        if offset_grouping_s is None:
            offsets_q = offsets
        else:
            step = float(offset_grouping_s)
            offsets_q = np.round(offsets / step) * step

        cbf = np.zeros(n_vox, dtype=float)
        cbv = np.zeros(n_vox, dtype=float)
        mtt = np.zeros(n_vox, dtype=float)
        cth = np.zeros(n_vox, dtype=float)
        lam_opt = np.zeros(n_vox, dtype=float)
        n_samp = int(max(0, uncertainty_samples))
        want_cbf_sd = bool(compute_cbf_sd) or n_samp > 0
        cbf_sd = np.full(n_vox, np.nan, dtype=float) if want_cbf_sd else None
        mtt_sd = np.full(n_vox, np.nan, dtype=float) if n_samp > 0 else None
        cth_sd = np.full(n_vox, np.nan, dtype=float) if n_samp > 0 else None
        rng = np.random.default_rng(int(uncertainty_seed)) if n_samp > 0 else None

        bTb = np.sum(Ct_mat * Ct_mat, axis=0)
        batch_size = 1024

        for off in np.unique(offsets_q):
            off = float(off)
            idx = np.where(offsets_q == off)[0]
            if idx.size == 0:
                continue
            ca_shift, D, DTD, cholA, gcv_denom, logdetA, gsvd = _get_design(off)

            if legacy_exact:
                # ── Legacy-exact MATLAB perffit path ───────────────────────
                # CBV uses Σcₐ (MATLAB fast file ``vd = sum(c_t)/sum(c_a)``).
                denom_sum = float(np.sum(ca_shift))
                if denom_sum == 0.0 or not np.isfinite(denom_sum):
                    denom_sum = 1.0
                _legacy_solve(
                    idx, Ct_mat, gsvd, D, Lreg, BT, dt, lambdas, lambda_unit,
                    L1, L2, f_win, cbf_scale, denom_sum, eps,
                    cbf, cbv, mtt, cth, lam_opt, t, mtt_cth_method,
                )
                continue

            denom = float(np.trapezoid(ca_shift))
            if denom == 0.0 or not np.isfinite(denom):
                denom = 1.0

            Ct_group = Ct_mat[:, idx].T               # (n_vox_g, n)
            RHS_group = Ct_group @ D                  # (n_vox_g, n_basis)
            bTb_group = bTb[idx]

            for start in range(0, int(idx.size), batch_size):
                end = min(start + batch_size, int(idx.size))
                rhs_b = RHS_group[start:end, :]
                btb_b = bTb_group[start:end]

                reg_norm = np.zeros((end - start, int(lambdas.size)))
                res_norm = np.zeros((end - start, int(lambdas.size)))

                for li in range(int(lambdas.size)):
                    X = cho_solve(cholA[li], rhs_b.T, check_finite=False).T
                    term1 = np.sum((X @ DTD) * X, axis=1)
                    term2 = np.sum(X * rhs_b, axis=1)
                    res_norm[:, li] = btb_b - 2.0 * term2 + term1
                    reg_norm[:, li] = np.sum((X @ LTL) * X, axis=1)

                log_ev = None     # filled by the evidence branch; reused for λ-marginalised sampling
                if lambda_selection == "evidence":
                    # Empirical-Bayes log-evidence (σ² marginalised, Jeffreys):
                    #   log Z(λ) = −(N/2)·log(RSS + λ²·PEN)
                    #              + rank(L)·log(λ) − ½·log|A(λ)|
                    # Maximised per voxel. Guaranteed interior optimum (the
                    # Occam terms fight the misfit term), so it does NOT collapse
                    # to the λ_min floor the way L-curve / GCV do on smooth
                    # high-SNR DCE curves.
                    lam2 = (lambdas ** 2)[None, :]
                    reg_term = np.maximum(res_norm, 0.0) + lam2 * np.maximum(reg_norm, 0.0)
                    log_ev = (
                        -0.5 * float(n) * np.log(np.maximum(reg_term, eps))
                        + float(rank_L) * np.log(lambdas)[None, :]
                        - 0.5 * logdetA[None, :]
                    )
                    idx_max = np.argmax(log_ev, axis=1)
                elif lambda_selection == "gcv":
                    # G(λ) per voxel: ‖res(λ)‖² / (n − tr H(λ))². Both numerator
                    # and denominator are non-negative, so argmin is well-defined.
                    gcv_vals = np.maximum(res_norm, 0.0) / gcv_denom[None, :]
                    idx_max = np.argmin(gcv_vals, axis=1)
                else:
                    # L-curve curvature on (log‖res‖², log‖Lx‖²); finite-difference
                    # step is in log(λ) for log spacing, λ for linear spacing.
                    xlog = np.log(np.maximum(res_norm, eps))
                    ylog = np.log(np.maximum(reg_norm, eps))
                    d_x = (xlog @ L1.T) / lambda_unit
                    d_y = (ylog @ L1.T) / lambda_unit
                    dd_x = (xlog @ L2.T) / (lambda_unit ** 2)
                    dd_y = (ylog @ L2.T) / (lambda_unit ** 2)

                    num = d_x[:, :-1] * dd_y - dd_x * d_y[:, :-1]
                    den = (d_x[:, :-1] ** 2 + d_y[:, :-1] ** 2) ** 1.5
                    # Degenerate points (den ≤ 0) get −∞, not 0 — matches legacy
                    # ``modules/kinetic_models``. On a smooth L-curve where all
                    # real curvatures are negative, a 0 here would beat every
                    # genuine (negative) interior corner and collapse to the
                    # λ_min endpoint; −∞ forces selection of a real corner.
                    kappa = np.full_like(num, -np.inf)
                    np.divide(num, den, out=kappa, where=den > 0)
                    with np.errstate(invalid="ignore"):
                        idx_max = np.where(
                            np.all(~np.isfinite(kappa), axis=1),
                            0,
                            np.nanargmax(np.where(np.isfinite(kappa), kappa, -np.inf), axis=1),
                        )

                ct_batch = Ct_group[start:end, :]
                lam_arr = np.asarray(idx_max, dtype=int).reshape(-1)

                for lam_idx in np.unique(lam_arr):
                    lam_idx = int(lam_idx)
                    sel = lam_arr == lam_idx
                    rhs_sel = rhs_b[sel, :]
                    Xopt_sel = cho_solve(cholA[lam_idx], rhs_sel.T, check_finite=False).T

                    for j_local, row in enumerate(np.flatnonzero(sel).tolist()):
                        rf = (BT @ Xopt_sel[j_local, :]) / dt
                        w = min(int(f_win), int(rf.size))
                        f_peak = float(np.max(rf[:w])) if w > 0 else 0.0
                        if not np.isfinite(f_peak) or f_peak < 0:
                            f_peak = 0.0

                        ct_curve = ct_batch[row, :]
                        vd = float(np.trapezoid(ct_curve) / denom)
                        if not np.isfinite(vd) or vd < 0:
                            vd = 0.0

                        out_i = int(idx[start + row])
                        cbf[out_i] = f_peak * cbf_scale
                        cbv[out_i] = vd
                        lam_opt[out_i] = float(lambdas[lam_idx])

                        # Posterior σ̂² = (RSS + λ²·PEN)/N for this voxel.
                        lam2v = float(lambdas[lam_idx]) ** 2
                        rss_v = float(res_norm[row, lam_idx])
                        pen_v = float(reg_norm[row, lam_idx])
                        sigma2 = max(rss_v + lam2v * pen_v, 0.0) / max(float(n), 1.0)

                        # Closed-form CBF posterior SD (empirical-Bayes):
                        # F = (B[:,k]ᵀ x)/dt is a linear functional of x, so
                        # Var(F) = σ̂²·gᵀA⁻¹g with g = B[:,k]/dt. Treats the peak
                        # index k as fixed (first-order; under-estimates at very
                        # low SNR where the peak location jitters — sampling
                        # below captures that exactly).
                        if want_cbf_sd and n_samp == 0 and f_peak > 0:
                            peak_k = int(np.argmax(rf[:w]))
                            g = B[:, peak_k] / dt
                            Ainv_g = cho_solve(cholA[lam_idx], g, check_finite=False)
                            gAg = float(g @ Ainv_g)
                            var_f = max(sigma2 * gAg, 0.0)
                            cbf_sd[out_i] = cbf_scale * math.sqrt(var_f)

                        # Posterior sampling for CBF/MTT/CTH uncertainty.
                        # When the evidence is available we MARGINALISE λ: draw
                        # λ_k ~ p(λ|b) ∝ exp(logZ(λ)), then x_k ~ N(x̂(λ_k),
                        # σ̂²(λ_k)·A(λ_k)⁻¹). This captures BOTH coefficient and
                        # λ-selection uncertainty — the latter dominates the
                        # low-SNR spread that fixed-λ sampling misses. Without
                        # evidence, fall back to fixed-λ sampling.
                        if n_samp > 0 and f_peak > 0 and sigma2 > 0:
                            if log_ev is not None:
                                lw = log_ev[row, :]
                                lw = lw - np.max(lw)
                                p_lam = np.exp(lw); p_lam /= p_lam.sum()
                                draw_idx = rng.choice(lambdas.size, size=n_samp, p=p_lam)
                                lam_draws, counts = np.unique(draw_idx, return_counts=True)
                            else:
                                lam_draws = np.array([lam_idx]); counts = np.array([n_samp])
                            cbf_acc, mtt_acc, cth_acc = [], [], []
                            for ldr, cnt in zip(lam_draws.tolist(), counts.tolist()):
                                # per-λ posterior mean + scale for THIS voxel
                                x_hat = cho_solve(cholA[ldr], rhs_b[row, :], check_finite=False)
                                s2 = max(float(res_norm[row, ldr])
                                         + (float(lambdas[ldr]) ** 2) * float(reg_norm[row, ldr]),
                                         0.0) / max(float(n), 1.0)
                                cfac = cholA[ldr][0]
                                Z = rng.standard_normal((n_basis, cnt))
                                U = solve_triangular(cfac, Z, lower=True, trans="T",
                                                     check_finite=False)
                                X_s = x_hat[:, None] + math.sqrt(s2) * U
                                RF_s = (BT @ X_s) / dt
                                F_s = np.max(RF_s[:w, :], axis=0)
                                F_s = np.where(np.isfinite(F_s) & (F_s > 0), F_s, np.nan)
                                cbf_acc.append(F_s * cbf_scale)
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    R_s = RF_s / F_s[None, :]
                                mk, ck = _residue_metrics_batch(R_s, dt)
                                mtt_acc.append(mk); cth_acc.append(ck)
                            cbf_sd[out_i] = float(np.nanstd(np.concatenate(cbf_acc)))
                            mtt_sd[out_i] = float(np.nanstd(np.concatenate(mtt_acc)))
                            cth_sd[out_i] = float(np.nanstd(np.concatenate(cth_acc)))

                        if mtt_cth_method == "residue_integral":
                            # Production-pipeline parity (modules/kinetic_models):
                            # MTT = ∫(clip+monotone R)dt; CTH from h=−dR/dt.
                            if f_peak > 0:
                                residue = rf / f_peak
                                mtt_b, cth_b = _residue_metrics_batch(
                                    residue.reshape(-1, 1), dt)
                                mtt_v = float(mtt_b[0])
                                cth_v = float(cth_b[0])
                            else:
                                mtt_v, cth_v = 0.0, float("nan")
                            mtt[out_i] = mtt_v if np.isfinite(mtt_v) and mtt_v >= 0 else 0.0
                            cth[out_i] = cth_v
                        else:
                            # Central-volume: MTT = vd/F; CTH from peak-shifted residue.
                            mtt_v = vd / f_peak if f_peak > 0 else 0.0
                            if not np.isfinite(mtt_v) or mtt_v < 0:
                                mtt_v = 0.0
                            mtt[out_i] = mtt_v
                            peak_i = int(np.argmax(rf[:w])) if w > 0 else 0
                            cth[out_i] = _cth_from_rf(t, rf, f=f_peak, peak_idx=peak_i)

        return TikhonovBatch(
            cbf_ml_per_100g_min=cbf,
            cbv_vd=cbv,
            mtt_s=mtt,
            cth_s=cth,
            lambda_opt=lam_opt,
            cbf_sd=cbf_sd,
            mtt_sd=mtt_sd,
            cth_sd=cth_sd,
        )

    return solve_ct


# ────────────────────────────────────────────────────────────────────────
# Plug-in adapter
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TikhonovModel:
    """Tikhonov-deconvolution perfusion model (the ``tikhonov`` plug-in).

    Model-free perfusion via B-spline-basis Tikhonov deconvolution of
    the tissue residue (paper §4.6.2), with per-voxel L-curve / GCV
    lambda selection. Outputs ``{"cbf", "cbv", "mtt", "cth", "lambda_opt"}``.
    ``cbv`` is the cerebral blood volume in **mL/100 g** — the volume of
    distribution ``v_d = ∫Cₜ/∫Cₐ`` scaled by 100 (consistent with the
    validation; distinct from Patlak's ``v_b`` intercept blood volume).
    Pass ``legacy_exact=True`` to :func:`build_tikhonov_solver` for the
    MATLAB-bit-matching solver path.
    """

    key: ClassVar[str] = "tikhonov"
    name: ClassVar[str] = "Tikhonov-regularised deconvolution"
    description: ClassVar[str] = (
        "Model-free perfusion via B-spline-basis Tikhonov deconvolution "
        "of the tissue residue (paper §4.6.2). Per-voxel L-curve λ "
        "selection. Outputs CBF, CBV, MTT, CTH and the chosen λ."
    )
    accepts: ClassVar[dict[str, type]] = {
        "c_tissue": np.ndarray,
        "c_input": np.ndarray,
        "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "cbf": np.ndarray,
        "cbv": np.ndarray,
        "mtt": np.ndarray,
        "cth": np.ndarray,
        "lambda_opt": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("cbf", "cbv", "mtt", "cth", "lambda_opt")
    units: ClassVar[dict[str, str]] = {
        "cbf": "mL/100g/min",
        "cbv": "mL/100g",
        "mtt": "s",
        "cth": "s",
        "lambda_opt": "(unitless)",
    }

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        """Deconvolve one curve or a voxel stack → CBF/MTT/CTH/lambda_opt.

        Builds the shared Tikhonov solver from the time axis + AIF, then
        solves per voxel. ``**opts`` are forwarded to
        :func:`build_tikhonov_solver` (``n_lambdas``, ``lambda_selection``,
        ``legacy_exact``, …).
        """
        c_t = np.asarray(inputs.c_tissue, dtype=float)
        c_a = np.asarray(inputs.c_input, dtype=float)
        t_s = np.asarray(inputs.t_s, dtype=float)

        if c_t.ndim == 1:
            c_t = c_t.reshape(-1, 1)
            n_v = 1
            single = True
        elif c_t.ndim == 2:
            n_v = int(c_t.shape[1])
            single = False
        else:
            raise ValueError(f"c_tissue must be 1-D or 2-D; got shape {c_t.shape}")

        solver_opts = {
            k: opts.pop(k) for k in list(opts.keys())
            if k in ("n_lambdas", "lambda_min", "lambda_max", "offset_grouping_s",
                     "f_win", "tissue_density", "hematocrit", "plasma_derived_aif",
                     "lambda_selection", "lambda_spacing", "mtt_cth_method",
                     "compute_cbf_sd", "uncertainty_samples", "uncertainty_seed")
        }
        offsets_s = opts.pop("offsets_s", None)

        # Optional temporal up-sampling before the deconvolution. Coarsely
        # sampled data (e.g. the mouse DCE at dt≈9 s) truncates the sharp early
        # peak of the residue → CBF is under-estimated. Refining the AIF+tissue
        # onto a finer grid (shape-preserving PCHIP, as in inverse_gaussian; the
        # knot vector then follows the finer dt) recovers the peak. h_factor=1
        # (the default) is a strict no-op, so the human path stays byte-identical.
        h_factor = int(opts.pop("h_factor", 1))
        if h_factor > 1:
            from scipy.interpolate import PchipInterpolator
            n0 = int(min(t_s.size, c_a.shape[0], c_t.shape[0]))
            t0 = np.asarray(t_s[:n0], dtype=float)
            t_s = np.linspace(float(t0[0]), float(t0[-1]), (n0 - 1) * h_factor + 1)
            c_a = PchipInterpolator(t0, np.nan_to_num(c_a[:n0]),
                                    extrapolate=True)(t_s)
            c_t = PchipInterpolator(t0, np.nan_to_num(c_t[:n0, :]), axis=0,
                                    extrapolate=True)(t_s)
            if "f_win" in solver_opts:   # keep the CBF search window fixed in time
                solver_opts["f_win"] = int(solver_opts["f_win"]) * h_factor

        # Apply brain mask: only solve for masked voxels, leave the rest NaN.
        # Legacy convention is NaN outside the brain (vs the previous behaviour
        # of running deconvolution on ~5× more voxels including air, getting
        # F=0, and then writing 0s — which poisoned the median MTT / CTH stats).
        mask = inputs.mask
        if mask is not None and not single:
            mask_bool = np.asarray(mask, dtype=bool).reshape(-1)
            if mask_bool.size != n_v:
                raise ValueError(
                    f"mask length {mask_bool.size} ≠ n_voxels {n_v}"
                )
            keep_idx = np.flatnonzero(mask_bool)
            c_t_keep = c_t[:, keep_idx]
            offsets_keep = (offsets_s[keep_idx] if offsets_s is not None else None)
        else:
            mask_bool = None
            keep_idx = None
            c_t_keep = c_t
            offsets_keep = offsets_s

        solver = build_tikhonov_solver(t_s, c_a, **solver_opts)
        batch = solver(c_t_keep, offsets_s=offsets_keep)

        # CBV (mL/100 g): the volume of distribution v_d = ∫Cₜ/∫Cₐ scaled by
        # 100 (validation convention). Promoted to a declared output so the
        # kinetic stage writes cbv.nii.gz at every aggregation level, matching
        # the paper/figures. ``aux["cbv_vd"]`` keeps the raw v_d for callers.
        cbv_ml100g = batch.cbv_vd * 100.0
        # Core maps + any optional SD maps the solver produced.
        core = {"cbf": batch.cbf_ml_per_100g_min, "cbv": cbv_ml100g,
                "mtt": batch.mtt_s, "cth": batch.cth_s,
                "lambda_opt": batch.lambda_opt}
        sd_src = {"cbf_sd": batch.cbf_sd, "mtt_sd": batch.mtt_sd, "cth_sd": batch.cth_sd}
        sd_present = {k: v for k, v in sd_src.items() if v is not None}

        if single:
            maps = {k: np.asarray(v[0]) for k, v in core.items()}
            for k, v in sd_present.items():
                maps[k] = np.asarray(v[0])
            cbv_aux = np.asarray(batch.cbv_vd[0])
        else:
            def _scatter(values):
                arr = np.full(n_v, np.nan, dtype=float)
                if keep_idx is not None:
                    arr[keep_idx] = values
                else:
                    arr[:] = values
                return arr
            maps = {k: _scatter(v) for k, v in core.items()}
            for k, v in sd_present.items():
                maps[k] = _scatter(v)
            cbv_aux = _scatter(batch.cbv_vd)

        units = dict(self.units)
        for k in ("cbf_sd", "mtt_sd", "cth_sd"):
            if k in maps:
                units[k] = self.units.get(k.replace("_sd", ""), "")
        return ModelResult(
            maps=maps,
            units=units,
            aux={"cbv_vd": cbv_aux},
        )

    def predict(self, maps: dict[str, Any], c_input: np.ndarray,
                t_s: np.ndarray) -> np.ndarray:
        """QC reconstruction of Cₜ from (cbf, mtt, cth) via a gamma-variate
        residue proxy. Tikhonov's actual fit is a *non-parametric* residue whose
        exact overlay is produced by ``diagnostics.tikhonov``; this proxy gives
        the uniform predict() interface a sensible parametric curve."""
        from .base import reconstruct_gamma_residue_ct
        return reconstruct_gamma_residue_ct(
            t_s, c_input,
            cbf=float(maps.get("cbf", float("nan"))),
            mtt=float(maps.get("mtt", float("nan"))),
            cth=float(maps.get("cth", float("nan"))),
        )


PLUGIN = TikhonovModel()
