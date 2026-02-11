import logging
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_function
from scipy.linalg import cho_factor, cho_solve
import utils.settings as settings


if hasattr(np, "trapezoid"):
    _trapezoid = np.trapezoid
else:  # pragma: no cover - legacy NumPy fallback
    _trapezoid = np.trapz


def extended_tofts_model(t, Ktrans, ve, vp, Cp):
    """Basic extended Tofts implementation used for the two-compartment model.

    This is performance-critical inside optimisation loops. The original
    implementation evaluated the convolution integral with an O(n^2) loop.
    Here we use an equivalent O(n) recurrence matching trapezoidal quadrature
    when the time grid is strictly increasing.
    """

    t = np.asarray(t, dtype=float).reshape(-1)
    Cp = np.asarray(Cp, dtype=float).reshape(-1)
    n = min(t.size, Cp.size)
    if n == 0:
        return np.zeros(0, dtype=float)

    t = t[:n]
    Cp = Cp[:n]

    Ktrans = float(Ktrans)
    ve = float(ve)
    vp = float(vp)
    if not (np.isfinite(Ktrans) and np.isfinite(ve) and np.isfinite(vp)):
        return np.full(n, np.nan, dtype=float)
    ve = max(ve, 1e-12)

    # Require a valid, strictly-increasing time axis for the fast recurrence.
    dt = np.diff(t)
    if dt.size == 0 or not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        # Fallback to the legacy behaviour for irregular time arrays.
        integrand = np.zeros(n, dtype=float)
        kep = Ktrans / ve
        for i in range(n):
            integral = _trapezoid(Cp[: i + 1] * np.exp(-(t[i] - t[: i + 1]) * kep), x=t[: i + 1])
            integrand[i] = integral
        return Ktrans * integrand + vp * Cp

    kep = Ktrans / ve

    # Trapezoidal recurrence:
    # I[i] = exp(-kep*dt)*I[i-1] + 0.5*dt*(Cp[i] + Cp[i-1]*exp(-kep*dt))
    I = np.zeros(n, dtype=float)
    for i in range(1, n):
        dti = float(dt[i - 1])
        e = float(np.exp(-kep * dti))
        I[i] = I[i - 1] * e + 0.5 * dti * (Cp[i] + Cp[i - 1] * e)

    return Ktrans * I + vp * Cp


def build_tikhonov_solver(A, lambd, *, penalty="identity"):
    """Pre-factor a Tikhonov system and return a fast solver.

    This is useful when the AIF (and therefore ``A``) and lambda are fixed
    across many tissue curves (e.g., voxelwise CBF/MTT/CTH maps). The returned
    callable supports one or many right-hand sides.
    """

    from scipy.linalg import cho_factor, cho_solve

    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    n_rows, n_cols = A.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("A must be non-empty")

    L = _prepare_penalty_matrix(n_cols, penalty)
    ata = A.T @ A
    ltl = L.T @ L if L.size else np.zeros((n_cols, n_cols), dtype=float)
    lam = float(max(lambd, 0.0))
    regularised = ata + lam**2 * ltl

    # SPD in typical use; Cholesky gives O(n^2) solves for multiple RHS.
    cho = cho_factor(regularised, overwrite_a=False, check_finite=False)
    At = A.T

    def solve(C_t):
        C_t_arr = np.asarray(C_t, dtype=float)
        if C_t_arr.ndim == 1:
            if C_t_arr.shape[0] != n_rows:
                raise ValueError("C_t length must match A.shape[0]")
            rhs = At @ C_t_arr
            return cho_solve(cho, rhs, check_finite=False)

        if C_t_arr.ndim == 2:
            # Expect shape (n_rows, n_rhs)
            if C_t_arr.shape[0] != n_rows:
                raise ValueError("C_t first dimension must match A.shape[0]")
            rhs = At @ C_t_arr
            return cho_solve(cho, rhs, check_finite=False)

        raise ValueError("C_t must be 1D or 2D")

    return solve


def _get_l_diff(n: int, order: int) -> np.ndarray:
    """Discrete finite-difference operator.

    order=1 -> first difference, shape (n-1, n)
    order=2 -> second difference, shape (n-2, n)
    """

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
    """Order-M B-spline basis weights evaluated at `x`."""

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
    """Lower-triangular Ca_matrix construction."""

    ca = np.asarray(ca, dtype=float).reshape(-1)
    n = int(ca.size)
    Ca = np.zeros((n, n), dtype=float)
    for j in range(n):
        # MATLAB:
        # for j=1:frame_no
        #   for i=j:frame_no
        #     Ca_matrix(i,j) = c_input(i+1-j);
        #   end
        # end
        Ca[j:, j] = ca[: n - j]
    return Ca


def _shift_curve_like_inputshift2(time: np.ndarray, ca: np.ndarray, delta_t: float) -> np.ndarray:
    """Replicate the core behaviour of `inputshift2`-style shifting.

    - For delta_t >= 0: build a zero-padded, shifted curve and interpolate it back
      onto the original time grid.
    - For delta_t < 0: use a continuous shift via interpolation.

    This supports sub-frame shifts (delta_t smaller than the sampling interval).
    """

    time = np.asarray(time, dtype=float).reshape(-1)
    ca = np.asarray(ca, dtype=float).reshape(-1)
    n = int(min(time.size, ca.size))
    time = time[:n]
    ca = ca[:n]
    if n < 3:
        return ca.copy()

    dt = float(time[1] - time[0])
    if not np.isfinite(dt) or dt <= 0:
        return ca.copy()

    delta_t = float(delta_t)
    if not np.isfinite(delta_t) or delta_t == 0.0:
        return ca.copy()

    if delta_t < 0:
        # interp1(time + deltaT, Ca, time)
        return np.interp(time, time + delta_t, ca, left=0.0, right=0.0)

    notimeunit = int(np.ceil(delta_t / dt))
    timetemp = np.arange(
        (time[0] + delta_t) - (notimeunit * dt),
        (time[n - 1] + delta_t) + 1e-12,
        dt,
        dtype=float,
    )
    s_ca = np.zeros(notimeunit + n, dtype=float)
    s_ca[notimeunit : notimeunit + n] = ca
    return np.interp(time, timetemp, s_ca, left=0.0, right=0.0)


def build_spline_lcurve_deconvolution_solver(
    time_s: np.ndarray,
    ca: np.ndarray,
    lambda_candidates: np.ndarray | None = None,
    *,
    bspline_order: int = 4,
    knot_step_frames: float | None = None,
    knot_end_mode: str = "n_time",
    aif_microshift: bool = False,
    aif_microshift_tol: float = 0.25,
):
    """Return a batched spline + Tikhonov (1st-derivative) deconvolution solver.

    - Lambda is chosen by L-curve curvature when multiple candidates are provided.
    - CBF is derived from the early maximum of the recovered impulse response.
    """

    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    ca = np.asarray(ca, dtype=float).reshape(-1)
    n_time = int(min(time_s.size, ca.size))
    if n_time < 2:
        raise ValueError("time_s and ca must have length >= 2")

    time_s = time_s[:n_time]
    ca = ca[:n_time]

    dt = float(time_s[1] - time_s[0])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time_s must be strictly increasing with finite dt")

    if lambda_candidates is None:
        # Default grid matches MATLAB menu_17: 0.05:0.3329:40 (~121 points)
        lambdas = np.linspace(settings.TIKHONOV_LAMBDA_MIN, settings.TIKHONOV_LAMBDA_MAX, settings.TIKHONOV_LAMBDA_STEPS, dtype=float)
    else:
        lambdas = np.asarray(lambda_candidates, dtype=float).reshape(-1)
    if lambdas.size == 0 or not np.all(np.isfinite(lambdas)):
        raise ValueError("lambda_candidates must be non-empty and finite")

    # MATLAB variants differ on the knot end-point:
    # - `menu_17.m`: K = 0:5*timeunit:frame_no_temp  (frame_no_temp = length(time))
    # - Several `con2perf_tikonov1_*.m`: K = 0:5*timeunit:time(end)
    if knot_step_frames is None:
        knot_step = 5.0 * dt
    else:
        knot_step = float(knot_step_frames)
    knot_end_mode_norm = str(knot_end_mode).lower()
    if knot_end_mode_norm in {"n", "n_time", "frames", "frame_no"}:
        knot_end = float(n_time)
    elif knot_end_mode_norm in {"n_time_dt", "n_dt", "frame_no_dt"}:
        knot_end = float(n_time) * float(dt)
    elif knot_end_mode_norm in {"time_end", "time", "seconds", "t_end"}:
        knot_end = float(time_s[-1])
    else:
        raise ValueError("knot_end_mode must be 'n_time', 'n_time_dt', or 'time_end'")
    if knot_step <= 0:
        raise ValueError("knot_step must be positive")

    K = np.arange(0.0, knot_end + 1e-12, knot_step, dtype=float)
    if K.size < 3:
        # Minimum for length_K = len(K)-2 to be >=1 when M=4
        K = np.array([0.0, knot_step, 2.0 * knot_step], dtype=float)

    M = int(bspline_order)
    length_K = int(K.size - 2)
    n_basis = int(length_K + M)

    B = np.zeros((n_basis, n_time), dtype=float)
    for i in range(n_time):
        B[:, i] = _bspline_basis(M, K, length_K, float(time_s[i]))

    Lreg = _get_l_diff(n_basis, 1)

    def _prep_for_ca(ca_use: np.ndarray):
        Ca_mat = _toeplitz_ca_matrix(ca_use)
        D = Ca_mat @ B.T  # (n_time, n_basis)
        Dt = D.T
        G = Dt @ D
        H = (Lreg.T @ Lreg) if Lreg.size else np.zeros_like(G)
        factors = []
        for lam in lambdas:
            lam = float(max(float(lam), 0.0))
            Mlam = G + (lam**2) * H
            factors.append(cho_factor(Mlam, lower=True, check_finite=False))
        return D, Dt, factors

    def _solve_ct_given(D: np.ndarray, Dt: np.ndarray, factors: list[tuple[np.ndarray, bool]], Ct_mat: np.ndarray):
        Ct_mat = np.asarray(Ct_mat, dtype=float)
        if Ct_mat.ndim == 1:
            Ct_mat = Ct_mat.reshape(-1, 1)
        if Ct_mat.ndim != 2:
            raise ValueError("Ct_mat must be 1D or 2D")
        if Ct_mat.shape[0] != n_time:
            raise ValueError("Ct_mat first dimension must match time_s length")

        n_vox = int(Ct_mat.shape[1])
        rhs = Dt @ Ct_mat  # (n_basis, n_vox)

        if lambdas.size == 1:
            coeff = cho_solve(factors[0], rhs, check_finite=False)
            rf = (B.T @ coeff) / dt
            f_internal = np.max(rf[: min(10, n_time), :], axis=0)
            f_internal = np.where(np.isfinite(f_internal), np.maximum(f_internal, 0.0), np.nan)
            cbf = f_internal * cbf_scale
            residue = rf / np.where(f_internal > 0.0, f_internal, np.nan)
            # Capillary stats from residue moments.
            mtt = np.full(n_vox, np.nan, dtype=float)
            cth = np.full(n_vox, np.nan, dtype=float)
            for k in range(n_vox):
                if np.isfinite(f_internal[k]) and f_internal[k] > 0.0:
                    mtt_k, cth_k, _, _ = residue_metrics(residue[:, k], dt)
                    mtt[k] = float(mtt_k)
                    cth[k] = float(cth_k)
            resid = (D @ coeff) - Ct_mat
            res_norm_best = np.sum(resid * resid, axis=0)
            return {
                "f_internal": f_internal,
                "cbf_ml_per_100g_min": cbf,
                "mtt_s": mtt,
                "cth_s": cth,
                "lambda_opt": np.full(n_vox, float(lambdas[0]), dtype=float),
                "res_norm": res_norm_best,
                "residue": residue,
            }

        # L-curve scan.
        eps = 1e-30
        log_reg = np.zeros((lambdas.size, n_vox), dtype=float)
        log_res = np.zeros((lambdas.size, n_vox), dtype=float)

        for li in range(lambdas.size):
            coeff = cho_solve(factors[li], rhs, check_finite=False)  # (n_basis, n_vox)
            reg = Lreg @ coeff
            reg_norm = np.sum(reg * reg, axis=0)
            resid = (D @ coeff) - Ct_mat
            res_norm = np.sum(resid * resid, axis=0)
            log_reg[li, :] = np.log(np.maximum(reg_norm, eps))
            log_res[li, :] = np.log(np.maximum(res_norm, eps))

        # Curvature of the L-curve (match MATLAB finite-difference approach).
        lambda_unit = float(lambdas[1] - lambdas[0])
        if not np.isfinite(lambda_unit) or lambda_unit == 0.0:
            raise ValueError("lambda_candidates must be uniformly spaced")

        d_x = np.diff(log_res, axis=0) / lambda_unit
        d_y = np.diff(log_reg, axis=0) / lambda_unit
        dd_x = np.diff(d_x, axis=0) / lambda_unit
        dd_y = np.diff(d_y, axis=0) / lambda_unit

        num = d_x[:-1, :] * dd_y - dd_x * d_y[:-1, :]
        den = (d_x[:-1, :] ** 2 + d_y[:-1, :] ** 2) ** 1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.where(den > 0.0, num / den, -np.inf)

        best_idx = np.nanargmax(kappa, axis=0)
        lambda_opt = lambdas[best_idx]

        # Solve again at the chosen lambda for each voxel (grouped by lambda index).
        coeff_best = np.zeros((n_basis, n_vox), dtype=float)
        for li in np.unique(best_idx):
            mask = best_idx == li
            if not np.any(mask):
                continue
            coeff_best[:, mask] = cho_solve(factors[int(li)], rhs[:, mask], check_finite=False)

        rf = (B.T @ coeff_best) / dt
        f_internal = np.max(rf[: min(10, n_time), :], axis=0)
        f_internal = np.where(np.isfinite(f_internal), np.maximum(f_internal, 0.0), np.nan)
        cbf = f_internal * cbf_scale

        residue = rf / np.where(f_internal > 0.0, f_internal, np.nan)
        mtt = np.full(n_vox, np.nan, dtype=float)
        cth = np.full(n_vox, np.nan, dtype=float)
        for k in range(n_vox):
            if np.isfinite(f_internal[k]) and f_internal[k] > 0.0:
                mtt_k, cth_k, _, _ = residue_metrics(residue[:, k], dt)
                mtt[k] = float(mtt_k)
                cth[k] = float(cth_k)

        resid = (D @ coeff_best) - Ct_mat
        res_norm_best = np.sum(resid * resid, axis=0)
        return {
            "f_internal": f_internal,
            "cbf_ml_per_100g_min": cbf,
            "mtt_s": mtt,
            "cth_s": cth,
            "lambda_opt": lambda_opt,
            "res_norm": res_norm_best,
            "residue": residue,
        }

    D0, Dt0, factors0 = _prep_for_ca(ca)

    # Scaling for CBF in mL/100 g/min, matching MATLAB fast Tikhonov.
    rho = float(settings.TISSUE_DENSITY) if np.isfinite(getattr(settings, "TISSUE_DENSITY", np.nan)) else 1.04
    hct = float(settings.HEMATOCRIT) if np.isfinite(getattr(settings, "HEMATOCRIT", np.nan)) else 0.42
    plasma_scale = (1.0 - hct) if getattr(settings, "PLASMA_DERIVED_AIF", False) else 1.0
    cbf_scale = 6000.0 * plasma_scale / max(rho, 1e-6)

    if not aif_microshift:
        def solve_ct(Ct_mat: np.ndarray):
            return _solve_ct_given(D0, Dt0, factors0, Ct_mat)

        return solve_ct

    # Sub-frame AIF shift search around 0, in the same units as `time_s`.
    tol = float(max(aif_microshift_tol, 0.0))
    dt_time = float(time_s[1] - time_s[0])
    shift_range = 0.25 * dt_time
    offsets = np.linspace(-shift_range, shift_range, 9, dtype=float)
    offsets_abs = np.abs(offsets)
    order = np.argsort(offsets_abs)

    prepped = []
    for off in offsets:
        ca_s = _shift_curve_like_inputshift2(time_s, ca, float(off))
        D, Dt, factors = _prep_for_ca(ca_s)
        prepped.append((float(off), D, Dt, factors))

    def solve_ct(Ct_mat: np.ndarray):
        # Compute per-offset solutions then pick the smallest |offset| among near-best residuals.
        sols = []
        for off, D, Dt, factors in prepped:
            sol = _solve_ct_given(D, Dt, factors, Ct_mat)
            sols.append((off, sol))

        # Stack residual norms: (n_off, n_vox)
        res_stack = np.stack([np.asarray(sol["res_norm"], dtype=float) for _off, sol in sols], axis=0)
        best_res = np.nanmin(res_stack, axis=0)
        thresh = best_res * (1.0 + tol)
        feasible = res_stack <= thresh

        n_vox = int(res_stack.shape[1])
        chosen = np.full(n_vox, -1, dtype=int)
        # Prefer smaller |offset| first.
        for idx in order:
            take = feasible[idx, :] & (chosen < 0)
            chosen[take] = int(idx)
        # Fallback: strict best residual.
        missing = chosen < 0
        if np.any(missing):
            chosen[missing] = np.nanargmin(res_stack[:, missing], axis=0)

        # Gather fields.
        out = {}
        for k in ("f_internal", "cbf_ml_per_100g_min", "mtt_s", "cth_s", "lambda_opt"):
            out[k] = np.full(n_vox, np.nan, dtype=float)
        out["aif_subframe_offset"] = np.full(n_vox, np.nan, dtype=float)

        for idx, (off, sol) in enumerate(sols):
            mask = chosen == idx
            if not np.any(mask):
                continue
            out["aif_subframe_offset"][mask] = float(off)
            for k in ("f_internal", "cbf_ml_per_100g_min", "mtt_s", "cth_s", "residue"):
                out[k][mask] = np.asarray(sol[k], dtype=float)[mask]
            out["lambda_opt"][mask] = np.asarray(sol["lambda_opt"], dtype=float)[mask]

        return out

    return solve_ct



def construct_convolution_matrix(C_a, delta_t):
    """Build a Toeplitz convolution matrix using trapezoidal quadrature."""

    C_a = np.asarray(C_a, dtype=float).reshape(-1)
    delta_t = float(delta_t)
    if C_a.size == 0 or not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("C_a must be non-empty and delta_t must be a positive finite number.")

    n = C_a.size
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, : i + 1] = C_a[i::-1] * delta_t
        if i == 0:
            A[i, 0] = 0.0
        else:
            A[i, 0] *= 0.5
            A[i, i] *= 0.5
    return A


def _prepare_penalty_matrix(size, penalty):
    """Return the Tikhonov penalty operator of appropriate shape."""

    if penalty is None or penalty == "identity":
        return np.eye(size, dtype=float)

    if penalty == "derivative":
        if size < 2:
            return np.zeros((0, size), dtype=float)
        L = np.zeros((size - 1, size), dtype=float)
        for i in range(size - 1):
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
        return L

    penalty_matrix = np.asarray(penalty, dtype=float)
    if penalty_matrix.ndim != 2 or penalty_matrix.shape[1] != size:
        raise ValueError("Penalty matrix must be two-dimensional with shape (m, n) where n matches A.shape[1].")
    return penalty_matrix


def tikhonov_regularization(A, C_t, lambd, *, penalty="identity"):
    r"""Solve ``A r = C_t`` with Tikhonov regularisation.

    Parameters
    ----------
    A : array_like
        Convolution matrix relating the residue to the tissue curve.
    C_t : array_like
        Tissue concentration samples.
    lambd : float
        Regularisation strength :math:`\lambda`.
    penalty : {"identity", "derivative"} or ndarray, optional
        Operator :math:`L` used in the quadratic penalty term. When an array is
        provided it must have shape ``(m, n)`` with ``n == A.shape[1]``.
    """

    A = np.asarray(A, dtype=float)
    C_t = np.asarray(C_t, dtype=float).reshape(A.shape[0])
    n_cols = A.shape[1]

    L = _prepare_penalty_matrix(n_cols, penalty)
    ata = A.T @ A
    ltl = L.T @ L if L.size else np.zeros((n_cols, n_cols), dtype=float)
    lam = float(max(lambd, 0.0))
    # Canonical Tikhonov: minimise ||A r - C_t||^2 + λ^2 ||L r||^2
    regularised = ata + lam**2 * ltl
    rhs = A.T @ C_t
    return np.linalg.solve(regularised, rhs)


def residue_to_cbf(impulse0, rho_tissue=None, hematocrit=None, aif_type=None):
    """CBF in mL/100 g/min from the impulse response at t=0.

    Defaults follow the global settings: tissue density, hematocrit and whether
    the AIF is plasma-derived (TSCC) or whole-blood. This keeps the fast
    Tikhonov path aligned with the MATLAB reference scaling.
    """

    r0 = float(impulse0)

    rho = float(rho_tissue) if rho_tissue is not None else float(getattr(settings, "TISSUE_DENSITY", 1.04))
    if not np.isfinite(rho) or rho <= 0:
        rho = 1.04

    hct = float(hematocrit) if hematocrit is not None else float(getattr(settings, "HEMATOCRIT", 0.42))
    if not np.isfinite(hct):
        hct = 0.42

    if aif_type is None:
        aif_type = "plasma" if getattr(settings, "PLASMA_DERIVED_AIF", False) else "whole_blood"

    scale = 1.0
    if aif_type == "plasma":
        scale *= (1.0 - hct)

    cbf = 6000.0 * r0 * scale / float(max(rho, 1e-6))
    return max(cbf, 0.0)


def residue_metrics(residue, dt, *, enforce_nonneg=True, enforce_monotone=True):
    r"""Return MTT/CTH statistics from a unit-normalised residue.

    MTT is computed as :math:`\int r(t)\,dt`, while the capillary transit time
    density :math:`h(t)` is formed via :math:`h(t) = -r'(t)` and re-normalised
    to unit area before its moments are evaluated. ``enforce_nonneg`` and
    ``enforce_monotone`` project the supplied residue onto the nearest
    non-negative, non-increasing curve; the stabilisation slightly perturbs the
    resulting moments and is therefore documented explicitly for downstream
    consumers.
    """

    residue = np.asarray(residue, dtype=float).reshape(-1)
    if residue.size < 5 or np.isnan(residue).any() or not np.isfinite(dt) or dt <= 0:
        return np.nan, np.nan, np.full(residue.shape, np.nan), np.nan

    working = residue.copy()
    if enforce_nonneg:
        working = np.clip(working, 0.0, None)
    if enforce_monotone:
        working = np.minimum.accumulate(working)

    mtt = float(_trapezoid(working, dx=dt))

    h = np.maximum(0.0, -np.gradient(working, dt, edge_order=2))
    s = float(_trapezoid(h, dx=dt))
    # If the density area is tiny, normalising will amplify numerical noise.
    if (not np.isfinite(s)) or s <= 1e-12:
        return mtt, np.nan, np.full_like(working, np.nan), np.nan

    h /= s
    time = np.arange(working.size, dtype=float) * dt
    mu = float(_trapezoid(time * h, dx=dt))
    variance = float(_trapezoid(((time - mu) ** 2) * h, dx=dt))
    cth = float(np.sqrt(max(variance, 0.0)))

    return mtt, cth, h, mu


def _gamma_density(t, a, b, t0):
    """Return the gamma-variate capillary transit-time density ``h(t)``."""

    shifted = np.asarray(t, dtype=float) - float(t0)
    h = np.zeros_like(shifted)
    mask = shifted > 0.0
    if not np.any(mask):
        return h

    positive = shifted[mask]
    norm = (float(b) ** (a + 1.0)) * gamma_function(a + 1.0)
    if norm <= 0.0:
        return np.zeros_like(shifted)

    h_val = (positive ** a) * np.exp(-positive / float(b)) / norm
    h[mask] = h_val
    return h


def _gamma_rif(t, a, b, t0, *, extraction=0.0):
    """Compute the residue impulse function from the gamma density."""

    h = _gamma_density(t, a, b, t0)
    if not np.any(h):
        return np.ones_like(h), h

    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    cumulative = np.cumsum((h + np.concatenate(([0.0], h[:-1]))) * 0.5 * dt)
    if extraction is None:
        extraction = 0.0
    extraction = float(np.clip(extraction, 0.0, 0.4))
    if extraction == 0.0:
        rif = 1.0 - cumulative
    else:
        rif = 1.0 - (1.0 - extraction) * cumulative
    return np.clip(rif, 0.0, None), h


def _trapezoid_convolution(a, b, dt):
    """Compute the discrete convolution using the trapezoidal rule."""

    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=float)

    dt = float(dt)
    n = a.size
    conv = np.convolve(a, b, mode="full")[:n]
    b_head = np.pad(b, (0, max(0, n - b.size)))[:n]
    a_head = a[:n]
    first = a_head[0] * b_head
    last = b_head[0] * a_head
    return dt * (conv - 0.5 * (first + last))


def gamma_fit_metrics(C_t, C_a, time_array, *,
                      cbf_seed=None, Ki=None, t0_seed=None,
                      flow_bounds=(5.0, 120.0), logger=None):
    """Fit a gamma-variate transit time density and derive MTT/CTH metrics."""

    if logger is None:
        logger = logging.getLogger(__name__)

    C_t = np.asarray(C_t, dtype=float).reshape(-1)
    C_a = np.asarray(C_a, dtype=float).reshape(-1)
    time_array = np.asarray(time_array, dtype=float).reshape(-1)

    n = min(C_t.size, C_a.size, time_array.size)
    if n < 2:
        return {
            "success": False,
            "message": "Insufficient samples",
        }

    C_t = C_t[:n]
    C_a = C_a[:n]
    time_array = time_array[:n]

    if (np.isnan(C_t).any() or np.isnan(C_a).any() or np.isnan(time_array).any() or
            np.isinf(C_t).any() or np.isinf(C_a).any() or np.isinf(time_array).any()):
        return {
            "success": False,
            "message": "Invalid inputs",
        }

    dt = np.diff(time_array)
    if not np.allclose(dt, dt[0]):
        dt = dt.mean()
    else:
        dt = float(dt[0])
    if not np.isfinite(dt) or dt <= 0.0:
        return {
            "success": False,
            "message": "Invalid time step",
        }

    # Initialisation matters a lot for this non-linear fit.
    # Seed b roughly at the sampling interval, and allow negative t0 to
    # capture sub-sample AIF/tissue offsets (MATLAB reports can be negative).
    a0 = 2.0
    b0 = float(max(dt, 1.0))
    t0_0 = 0.0 if t0_seed is None else float(t0_seed)

    flow_lo, flow_hi = flow_bounds
    flow_lo = float(flow_lo)
    flow_hi = float(flow_hi)

    if cbf_seed is not None and np.isfinite(cbf_seed):
        F0 = float(np.clip(cbf_seed, flow_lo, flow_hi))
    else:
        F0 = 40.0
    F0 = float(np.clip(F0, flow_lo, flow_hi))

    if Ki is not None and np.isfinite(Ki) and Ki > 0.0 and np.isfinite(F0):
        E0 = float(np.clip(F0 / Ki, 0.0, 0.4))
        optimise_E = True
    else:
        E0 = 0.0
        optimise_E = False

    flow_scale = settings.TISSUE_DENSITY / 6000.0

    t_max = float(time_array[-1]) if time_array.size else 1.0
    weights = 1.0 + 0.5 * (time_array / max(t_max, 1e-3))

    def pack_params(params):
        if optimise_E:
            a, b, t0, F, E = params
        else:
            a, b, t0, F = params
            E = E0
        return a, b, t0, F, np.clip(E, 0.0, 0.4)

    def residuals(params):
        a, b, t0, F_ml, E = pack_params(params)
        if not (0.5 <= a <= 50.0 and 0.2 <= b <= 12.0 and -6.0 <= t0 <= 6.0 and flow_lo <= F_ml <= flow_hi):
            return np.ones_like(C_t) * 1e6

        rif, _ = _gamma_rif(time_array, a, b, t0, extraction=E)
        F_internal = F_ml * flow_scale
        Ct_hat = F_internal * _trapezoid_convolution(C_a, rif, dt)
        misfit = Ct_hat - C_t
        return np.sqrt(weights) * misfit

    x0 = [a0, b0, float(np.clip(t0_0, -6.0, 6.0)), F0]
    bounds_lower = [0.5, 0.2, -6.0, flow_lo]
    bounds_upper = [50.0, 12.0, 6.0, flow_hi]
    if optimise_E:
        x0.append(E0)
        bounds_lower.append(0.0)
        bounds_upper.append(0.4)

    try:
        sol = least_squares(
            residuals,
            x0,
            bounds=(bounds_lower, bounds_upper),
            method="trf",
            x_scale="jac",
            max_nfev=5000,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Gamma fit failed: %s", exc)
        return {
            "success": False,
            "message": str(exc),
        }

    if not sol.success:
        logger.warning("Gamma fit did not converge: status=%s message=%s", sol.status, sol.message)
        return {
            "success": False,
            "message": sol.message,
        }

    a, b, t0, F_ml, E = pack_params(sol.x)

    rif, h = _gamma_rif(time_array, a, b, t0, extraction=E)
    F_internal = F_ml * flow_scale
    Ct_hat = F_internal * _trapezoid_convolution(C_a, rif, dt)
    residual_vec = Ct_hat - C_t
    residual_norm = float(np.linalg.norm(residual_vec))

    mtt_gamma = float((a + 1.0) * b)
    cth_gamma = float(np.sqrt(a + 1.0) * b)
    shape_ratio = float(1.0 / (a + 1.0))

    return {
        "success": True,
        "a": float(a),
        "b": float(b),
        "t0": float(t0),
        "F_ml_per_100g_min": float(F_ml),
        "E": float(np.clip(E, 0.0, 0.4)) if optimise_E else 0.0,
        "MTT_gamma": mtt_gamma,
        "CTH_gamma": cth_gamma,
        "shape_ratio": shape_ratio,
        "residual_norm": residual_norm,
        "iterations": int(getattr(sol, "nfev", 0)),
    }


def pick_lambda_via_l_curve(aif, tissue_curve, time_array, lambdas, *, penalty="identity"):
    """Return the lambda corresponding to the L-curve corner."""
    delta_t = np.diff(time_array).mean()
    A = construct_convolution_matrix(aif, delta_t)
    R, S = [], []
    for lam in lambdas:
        theta = tikhonov_regularization(A, tissue_curve, lam, penalty=penalty)
        R.append(np.linalg.norm(A.dot(theta) - tissue_curve))
        S.append(np.linalg.norm(theta))
    logR, logS = np.log(R), np.log(S)
    kappa = []
    for i in range(1, len(lambdas) - 1):
        x1, y1 = logR[i - 1], logS[i - 1]
        x2, y2 = logR[i], logS[i]
        x3, y3 = logR[i + 1], logS[i + 1]
        num = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        den = (np.hypot(x2 - x1, y2 - y1) *
               np.hypot(x3 - x2, y3 - y2) *
               np.hypot(x3 - x1, y3 - y1))
        kappa.append(num / den if den > 0 else 0)
    idx = np.argmax(kappa) + 1
    return lambdas[idx]


def plot_l_curve(aif, tissue_curve, time_array, lambdas, *, best=None, penalty="identity"):
    """Plot the L-curve and return the selected lambda."""
    import matplotlib.pyplot as plt

    delta_t = np.diff(time_array).mean()
    A = construct_convolution_matrix(aif, delta_t)
    R, S = [], []
    for lam in lambdas:
        theta = tikhonov_regularization(A, tissue_curve, lam, penalty=penalty)
        R.append(np.linalg.norm(A.dot(theta) - tissue_curve))
        S.append(np.linalg.norm(theta))

    if best is None:
        best = pick_lambda_via_l_curve(aif, tissue_curve, time_array, lambdas, penalty=penalty)
    idx = int(np.where(lambdas == best)[0][0])

    plt.figure()
    plt.loglog(R, S, marker='o')
    plt.scatter(R[idx], S[idx], color='red')
    plt.xlabel(r'$\|A\theta - C_t\|$')
    plt.ylabel(r'$\|\theta\|$')
    plt.title('L-curve')
    return best


def extended_tofts_tikhonov(Cp, Ct, t, lambd=settings.TIKHONOV_LAMBDA,
                            x0=(0.001, 0.2, 0.05)):
    Cp = np.asarray(Cp, dtype=float).reshape(-1)
    Ct = np.asarray(Ct, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)

    if Cp.shape[0] != Ct.shape[0] or Cp.shape[0] != t.shape[0]:
        raise ValueError("Cp, Ct and t must share the same length.")

    valid = np.isfinite(Cp) & np.isfinite(Ct) & np.isfinite(t)
    if not np.any(valid):
        raise ValueError("Cp, Ct and t must contain at least one finite sample.")

    if not np.all(valid):
        Cp = Cp[valid]
        Ct = Ct[valid]
        t = t[valid]

    if Cp.size < 3:
        raise ValueError("Need at least three finite samples to fit the extended Tofts model.")

    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (3,) or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must be a finite iterable of length three.")

    x0 = np.clip(x0, 1e-12, None)

    def residual(theta):
        theta = np.clip(theta, 1e-12, None)
        Ktrans, ve, vp = theta
        Ct_pred = extended_tofts_model(t, Ktrans, ve, vp, Cp)
        misfit = Ct_pred - Ct
        if np.any(~np.isfinite(misfit)):
            return np.full(misfit.size + theta.size, np.inf)
        w = np.linalg.norm(misfit) / max(np.linalg.norm(theta), 1e-8)
        penalty = np.sqrt(lambd) * w * theta
        return np.concatenate([misfit, penalty])

    sol = least_squares(residual, x0, bounds=(0, np.inf))
    Ktrans, ve, vp = sol.x
    return Ktrans, ve, vp


def two_compartment_tikhonov_fit(c_input, c_tissue, time_array,
                                 lambd=settings.TIKHONOV_LAMBDA,
                                 ktrans_initial=None,
                                 *, penalty="identity"):
    """Fit the extended Tofts model using Tikhonov regularisation.

    Parameters
    ----------
    c_input : array_like
        Arterial input function.
    c_tissue : array_like
        Measured tissue concentration curve.
    time_array : array_like
        Time points in seconds.
    lambd : float, optional
        Regularisation weight.
    ktrans_initial : float, optional
        Optional initial guess for Ktrans in 1/s. When ``None`` the default
        value 0.5/6000 is used.
    penalty : {"identity", "derivative"} or ndarray, optional
        Choice of Tikhonov penalty operator :math:`L`.

    Returns
    -------
    Ki : float
        Permeability in ml/100g/min.
    lam : float
        Dimensionless lambda (100*ve).
    vp : float
        Plasma volume fraction.
    CBF : float
        Cerebral blood flow in ml/100g/min estimated from deconvolution.
    fitted_curve : ndarray
        Model prediction at ``time_array``.
    residue : ndarray
        Estimated residue function :math:`r(t)`.
    """

    if c_input.shape[0] != c_tissue.shape[0]:
        raise ValueError("The number of time points in c_input and c_tissue must be the same.")

    if ktrans_initial is None or not np.isfinite(ktrans_initial):
        k0 = 0.5 / 6000
    else:
        k0 = max(float(ktrans_initial), 1e-8)
    initial_guess = [k0, 0.2, 0.05]

    def model(params):
        return extended_tofts_model(time_array, params[0], params[1], params[2], c_input)

    def objective(params):
        return np.concatenate([model(params) - c_tissue, np.sqrt(lambd) * params])

    res = least_squares(objective, initial_guess, bounds=(0, np.inf))
    Ktrans_fitted, ve_fitted, vp_fitted = res.x

    Ki = Ktrans_fitted * 6000
    lam = ve_fitted * 100

    fitted_curve = model(res.x)

    # Estimate CBF using model-free deconvolution
    delta_t = float(np.diff(time_array)[0])
    A = construct_convolution_matrix(c_input, delta_t)
    residue = tikhonov_regularization(A, c_tissue, lambd, penalty=penalty)
    CBF = residue_to_cbf(residue[0])

    return Ki, lam, vp_fitted, CBF, fitted_curve, residue
