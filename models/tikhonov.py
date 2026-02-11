from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np

from scipy.linalg import cho_factor, cho_solve

from .matlab_offset_correction import shift_curve_pchip


MODEL = {
    "key": "tikhonov",
    "env": "tikhonov",
    "description": "Validated slow Tikhonov deconvolution (CBF/MTT/CTH) with L-curve lambda.",
    "aliases": {
        "tikhonov",
        "two_compartment",
        "two-compartment",
        # Back-compat aliases (implementation is still the validated slow solver).
        "tikhonov_fast",
        "tik_fast",
        "tikhonov_only",
        "tikhonov-only-fast",
    },
}


@dataclass(frozen=True, slots=True)
class TikhonovResult:
    cbf_ml_per_100g_min: np.ndarray
    cbv_vd: np.ndarray
    mtt_s: np.ndarray
    cth_s: np.ndarray
    lambda_opt: np.ndarray


def _get_l_diff(n: int, order: int) -> np.ndarray:
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
    """Order-M B-spline basis weights evaluated at `x` (MATLAB-compatible)."""

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
        else:
            B[i, 0] = 0.0

    if M > 1:
        for m in range(2, M + 1):
            for i in range(length_K + 2 * M - m):
                b0 = B[i, m - 2]
                b1 = B[i + 1, m - 2]
                if b0 == 0.0 and b1 == 0.0:
                    B[i, m - 1] = 0.0
                elif b0 == 0.0 and b1 != 0.0:
                    denom = (tau[i + m] - tau[i + 1])
                    B[i, m - 1] = ((tau[i + m] - x) / denom) * b1 if denom != 0.0 else 0.0
                elif b0 != 0.0 and b1 == 0.0:
                    denom = (tau[i + m - 1] - tau[i])
                    B[i, m - 1] = ((x - tau[i]) / denom) * b0 if denom != 0.0 else 0.0
                else:
                    denom0 = (tau[i + m - 1] - tau[i])
                    denom1 = (tau[i + m] - tau[i + 1])
                    term0 = ((x - tau[i]) / denom0) * b0 if denom0 != 0.0 else 0.0
                    term1 = ((tau[i + m] - x) / denom1) * b1 if denom1 != 0.0 else 0.0
                    B[i, m - 1] = term0 + term1

    return B[: length_K + M, M - 1].copy()


def _toeplitz_ca_matrix(ca: np.ndarray) -> np.ndarray:
    ca = np.asarray(ca, dtype=float).reshape(-1)
    n = int(ca.size)
    Ca = np.zeros((n, n), dtype=float)
    for j in range(n):
        Ca[j:, j] = ca[: n - j]
    return Ca


def _cth_from_rf(time_s: np.ndarray, rf: np.ndarray, *, f: float, peak_idx: int) -> float:
    """CTH computation matching validator semantics."""

    t = np.asarray(time_s, dtype=float).reshape(-1)
    rf = np.asarray(rf, dtype=float).reshape(-1)
    n = int(min(t.size, rf.size))
    if n < 5:
        return float("nan")
    t = t[:n]
    rf = rf[:n]

    f = float(f)
    if not np.isfinite(f) or f <= 0:
        return 0.0

    peak_idx = int(max(0, min(int(peak_idx), n - 2)))

    R_raw = rf / f

    R = np.asarray(R_raw, dtype=float).copy()
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
        tail_start = int(max(R.size - 3, 1))
        R[tail_start:] = np.linspace(R[tail_start], 0.0, R.size - tail_start)

    t_mid = 0.5 * (t[:-1] + t[1:])
    dR = np.diff(R)
    dt_full = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -dR / dt_full
    h = np.where(np.isfinite(h), np.maximum(h, 0.0), 0.0)

    area = float(np.trapz(h, t_mid))
    if not np.isfinite(area) or area <= 0:
        return 0.0
    h = h / area

    mean = float(np.trapz(t_mid * h, t_mid))
    second = float(np.trapz((t_mid**2) * h, t_mid))
    var = second - mean * mean
    if not np.isfinite(var) or var < 0:
        var = 0.0
    return float(math.sqrt(var))


def build_tikhonov_solver(
    time_s: np.ndarray,
    ca: np.ndarray,
    *,
    lambda_candidates: np.ndarray | None = None,
    offset_grouping_s: float | None = 0.05,
    f_win: int = 50,
    tissue_density: float = 1.04,
    hematocrit: float = 0.42,
    plasma_derived_aif: bool = False,
):
    """Build a solver matching the Hemisure validator slow Tikhonov.

    - B-spline basis on the true seconds time axis.
    - Lambda grid defaults to linspace(0.05, max(svd(Ca_matrix0)), 121).
    - Optional per-voxel offsets shift the AIF (pchip) inside the operator.
    - f_win=50 peak window for flow.
    """

    t = np.asarray(time_s, dtype=float).reshape(-1)
    ca0 = np.asarray(ca, dtype=float).reshape(-1)
    n = int(min(t.size, ca0.size))
    if n < 3:
        raise ValueError("time_s and ca must have length >= 3")
    t = t[:n]
    ca0 = ca0[:n]

    dt = float(t[1] - t[0])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time_s must be strictly increasing")

    if lambda_candidates is None:
        Ca_matrix0 = _toeplitz_ca_matrix(ca0)
        sing = np.linalg.svd(Ca_matrix0, compute_uv=False)
        sing_max = float(np.max(sing)) if sing.size else 0.05
        L_min = 0.05
        L_max = sing_max if sing_max >= L_min else L_min
        lambdas = np.linspace(L_min, L_max, 121, dtype=float)
    else:
        lambdas = np.asarray(lambda_candidates, dtype=float).reshape(-1)
    if lambdas.size < 2 or not np.all(np.isfinite(lambdas)):
        raise ValueError("lambda_candidates must contain >=2 finite values (for curvature)")

    M = 4
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

    L1 = _get_l_diff(int(lambdas.size), 1)
    L2 = _get_l_diff(int(lambdas.size), 2)
    lambda_unit = float(lambdas[1] - lambdas[0])
    if not np.isfinite(lambda_unit) or lambda_unit == 0.0:
        raise ValueError("lambda_candidates must be uniformly spaced")

    rho = float(tissue_density)
    rho = rho if np.isfinite(rho) and rho > 0 else 1.04
    hct = float(hematocrit)
    hct = hct if np.isfinite(hct) and 0 <= hct < 1 else 0.42
    plasma_scale = (1.0 - hct) if plasma_derived_aif else 1.0
    cbf_scale = 6000.0 * plasma_scale / rho

    design_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, bool]]]] = {}

    def _get_design(off_s: float):
        off_s = float(off_s)
        cached = design_cache.get(off_s)
        if cached is not None:
            return cached

        ca_shift = shift_curve_pchip(t, ca0, off_s)
        Ca_mat = _toeplitz_ca_matrix(ca_shift)
        D = Ca_mat @ B.T
        DTD = D.T @ D

        cholA = [None] * int(lambdas.size)
        for li, lam in enumerate(lambdas):
            A = DTD + (float(lam) ** 2) * LTL
            cholA[li] = cho_factor(A, lower=True, check_finite=False)

        out = (ca_shift, D, DTD, cholA)
        design_cache[off_s] = out
        return out

    def solve_ct(
        Ct_mat: np.ndarray,
        *,
        offsets_s: np.ndarray | None = None,
        implementation: str = "grouped",
    ) -> TikhonovResult:
        Ct_mat = np.asarray(Ct_mat, dtype=float)
        if Ct_mat.ndim == 1:
            Ct_mat = Ct_mat.reshape(-1, 1)
        if Ct_mat.ndim != 2:
            raise ValueError("Ct_mat must be 1D or 2D")
        if Ct_mat.shape[0] != n:
            raise ValueError("Ct_mat first dimension must match time_s")

        implementation = (implementation or "grouped").strip().lower()
        if implementation not in {"grouped", "per_voxel"}:
            raise ValueError("implementation must be 'grouped' or 'per_voxel'")

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
            if step <= 0:
                raise ValueError("offset_grouping_s must be >0 or None")
            offsets_q = np.round(offsets / step) * step

        unique_offsets = np.unique(offsets_q)

        cbf = np.zeros(n_vox, dtype=float)
        cbv = np.zeros(n_vox, dtype=float)
        mtt = np.zeros(n_vox, dtype=float)
        cth = np.zeros(n_vox, dtype=float)
        lambda_opt = np.zeros(n_vox, dtype=float)

        bTb = np.sum(Ct_mat * Ct_mat, axis=0)

        batch_size = 1024
        eps = 1e-30

        BT = B.T

        for off in unique_offsets:
            off = float(off)
            idx = np.where(offsets_q == off)[0]
            if idx.size == 0:
                continue

            ca_shift, D, DTD, cholA = _get_design(off)

            denom = float(np.trapz(ca_shift))
            if denom == 0.0 or not np.isfinite(denom):
                denom = 1.0

            Ct_group = Ct_mat[:, idx].T
            RHS_group = Ct_group @ D
            bTb_group = bTb[idx]

            for start in range(0, int(idx.size), batch_size):
                end = min(start + batch_size, int(idx.size))
                rhs_b = RHS_group[start:end, :]
                btb_b = bTb_group[start:end]

                reg_norm = np.zeros((end - start, int(lambdas.size)), dtype=float)
                res_norm = np.zeros((end - start, int(lambdas.size)), dtype=float)

                for li in range(int(lambdas.size)):
                    X = cho_solve(cholA[li], rhs_b.T, check_finite=False).T
                    term1 = np.sum((X @ DTD) * X, axis=1)
                    term2 = np.sum(X * rhs_b, axis=1)
                    res_norm[:, li] = btb_b - 2.0 * term2 + term1
                    reg_norm[:, li] = np.sum((X @ LTL) * X, axis=1)

                xlog = np.log(np.maximum(res_norm, eps))
                ylog = np.log(np.maximum(reg_norm, eps))

                d_x = (xlog @ L1.T) / lambda_unit
                d_y = (ylog @ L1.T) / lambda_unit
                dd_x = (xlog @ L2.T) / (lambda_unit**2)
                dd_y = (ylog @ L2.T) / (lambda_unit**2)

                num = d_x[:, :-1] * dd_y - dd_x * d_y[:, :-1]
                den = (d_x[:, :-1] ** 2 + d_y[:, :-1] ** 2) ** 1.5
                kappa = np.zeros_like(num)
                np.divide(num, den, out=kappa, where=den > 0)
                idx_max = np.argmax(kappa, axis=1)

                # Ct curves for this batch are needed for vd.
                ct_batch = Ct_group[start:end, :]

                if implementation == "per_voxel":
                    for row_in_batch, lam_idx in enumerate(idx_max.tolist()):
                        lam_idx = int(lam_idx)
                        Xopt = cho_solve(cholA[lam_idx], rhs_b[row_in_batch, :], check_finite=False)
                        rf = (BT @ Xopt) / dt

                        w = min(int(f_win), int(rf.size))
                        f = float(np.max(rf[:w])) if w > 0 else 0.0
                        if not np.isfinite(f) or f < 0:
                            f = 0.0

                        ct_curve = ct_batch[row_in_batch, :]
                        vd = float(np.trapz(ct_curve) / denom)
                        if not np.isfinite(vd) or vd < 0:
                            vd = 0.0

                        mtt_i = vd / f if f > 0 else 0.0
                        if not np.isfinite(mtt_i) or mtt_i < 0:
                            mtt_i = 0.0

                        out_i = int(idx[start + row_in_batch])
                        cbf[out_i] = f * cbf_scale
                        cbv[out_i] = vd
                        mtt[out_i] = mtt_i
                        lambda_opt[out_i] = float(lambdas[lam_idx])

                        peak_i = int(np.argmax(rf[:w])) if w > 0 else 0
                        cth[out_i] = _cth_from_rf(t, rf, f=f, peak_idx=peak_i)
                else:
                    # Group-by-lambda optimisation:
                    # Many voxels select the same lambda; solve those RHS together.
                    # This keeps the exact same lambda selection and downstream math,
                    # but avoids one Cholesky solve per voxel.
                    lam_idx_arr = np.asarray(idx_max, dtype=int).reshape(-1)

                    for lam_idx in np.unique(lam_idx_arr):
                        lam_idx = int(lam_idx)
                        sel = lam_idx_arr == lam_idx
                        if not np.any(sel):
                            continue

                        rhs_sel = rhs_b[sel, :]
                        # Solve all selected RHS at once: shape (n_sel, n_basis)
                        Xopt_sel = cho_solve(cholA[lam_idx], rhs_sel.T, check_finite=False).T

                        # Process each selected voxel with the same computations as before.
                        sel_rows = np.flatnonzero(sel)
                        for j_local, row_in_batch in enumerate(sel_rows.tolist()):
                            Xopt = Xopt_sel[j_local, :]
                            rf = (BT @ Xopt) / dt

                            w = min(int(f_win), int(rf.size))
                            f = float(np.max(rf[:w])) if w > 0 else 0.0
                            if not np.isfinite(f) or f < 0:
                                f = 0.0

                            ct_curve = ct_batch[row_in_batch, :]
                            vd = float(np.trapz(ct_curve) / denom)
                            if not np.isfinite(vd) or vd < 0:
                                vd = 0.0

                            mtt_i = vd / f if f > 0 else 0.0
                            if not np.isfinite(mtt_i) or mtt_i < 0:
                                mtt_i = 0.0

                            out_i = int(idx[start + row_in_batch])
                            cbf[out_i] = f * cbf_scale
                            cbv[out_i] = vd
                            mtt[out_i] = mtt_i
                            lambda_opt[out_i] = float(lambdas[lam_idx])

                            peak_i = int(np.argmax(rf[:w])) if w > 0 else 0
                            cth[out_i] = _cth_from_rf(t, rf, f=f, peak_idx=peak_i)

        return TikhonovResult(
            cbf_ml_per_100g_min=cbf,
            cbv_vd=cbv,
            mtt_s=mtt,
            cth_s=cth,
            lambda_opt=lambda_opt,
        )

    return solve_ct


def fit_tikhonov_tuple(
    c_input: np.ndarray,
    c_tissue: np.ndarray,
    t_s: np.ndarray,
    *,
    offsets_s: float | None = None,
    return_residue: bool = False,
):
    """Compatibility wrapper for legacy callers.

    Returns (Ki_1_per_s, vp_fraction, SD_Ki, fit_curve) or
            (Ki_1_per_s, vp_fraction, SD_Ki, fit_curve, impulse_response, cbf_ml_per_100g_min)
    """

    from .patlak import fit_patlak

    ca = np.asarray(c_input, dtype=float).reshape(-1)
    ct = np.asarray(c_tissue, dtype=float).reshape(-1)
    t = np.asarray(t_s, dtype=float).reshape(-1)
    n = int(min(ca.size, ct.size, t.size))
    ca = ca[:n]
    ct = ct[:n]
    t = t[:n]
    if n < 3:
        fit_curve = np.full(n, np.nan, dtype=float)
        if return_residue:
            return float("nan"), float("nan"), float("nan"), fit_curve, None, float("nan")
        return float("nan"), float("nan"), float("nan"), fit_curve

    # Permeability terms via Patlak, reported in base units for legacy callers.
    p = fit_patlak(ct, ca, t)
    ki_1_per_s = float(p.ki_ml_per_100g_min) / 6000.0 if np.isfinite(p.ki_ml_per_100g_min) else float("nan")
    vp_fraction = float(p.vp_ml_per_100g) / 100.0 if np.isfinite(p.vp_ml_per_100g) else float("nan")
    sd_ki = float("nan")

    # Reconstruct impulse response + fit curve for QA/compat.
    solver = build_tikhonov_solver(t, ca)
    res = solver(ct.reshape(-1, 1), offsets_s=(np.array([float(offsets_s)]) if offsets_s is not None else None))
    cbf = float(res.cbf_ml_per_100g_min.reshape(-1)[0])

    # Best-effort: compute the impulse response that produced the chosen lambda.
    # This mirrors the single-voxel path inside `build_tikhonov_solver`.
    # Note: we rebuild a small amount of state to avoid exposing internals.
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

    ca_shift = shift_curve_pchip(t, ca, float(offsets_s or 0.0))
    Ca_mat = _toeplitz_ca_matrix(ca_shift)
    D = Ca_mat @ B.T
    DTD = D.T @ D

    # Recreate lambda grid used by the solver when lambda_candidates is None.
    sing = np.linalg.svd(_toeplitz_ca_matrix(ca), compute_uv=False)
    sing_max = float(np.max(sing)) if sing.size else 0.05
    L_min = 0.05
    L_max = sing_max if sing_max >= L_min else L_min
    lambdas = np.linspace(L_min, L_max, 121, dtype=float)

    Lreg = _get_l_diff(n_basis, 1)
    LTL = Lreg.T @ Lreg

    cholA = [None] * int(lambdas.size)
    for li, lam in enumerate(lambdas):
        A = DTD + (float(lam) ** 2) * LTL
        cholA[li] = cho_factor(A, lower=True, check_finite=False)

    rhs = (ct.reshape(1, -1) @ D).reshape(-1)

    # Pick lambda index from the already selected lambda value.
    lam_sel = float(res.lambda_opt.reshape(-1)[0])
    lam_idx = int(np.argmin(np.abs(lambdas - lam_sel)))
    Xopt = cho_solve(cholA[lam_idx], rhs, check_finite=False)
    rf = (B.T @ Xopt) / dt
    fit_curve = (Ca_mat @ rf).astype(float)

    if return_residue:
        return ki_1_per_s, vp_fraction, sd_ki, fit_curve, rf, cbf
    return ki_1_per_s, vp_fraction, sd_ki, fit_curve
