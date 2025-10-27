import logging
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_function
import utils.settings as settings


def extended_tofts_model(t, Ktrans, ve, vp, Cp):
    """Basic extended Tofts implementation used for the two-compartment model."""
    integrand = np.zeros_like(t)
    for i in range(len(t)):
        min_len = min(len(Cp[:i+1]), len(t[:i+1]))
        integral = np.trapz(Cp[:min_len] * np.exp(-(t[i] - t[:min_len]) * Ktrans / ve), x=t[:min_len])
        integrand[i] = integral
    min_len = min(len(integrand), len(Cp))
    return Ktrans * integrand[:min_len] + vp * Cp[:min_len]



def construct_convolution_matrix(C_a, delta_t):
    """Build a Toeplitz convolution matrix discretising the Volterra integral."""

    C_a = np.asarray(C_a, dtype=float).reshape(-1)
    delta_t = float(delta_t)
    if C_a.size == 0 or not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("C_a must be non-empty and delta_t must be a positive finite number.")

    n = C_a.size
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, : i + 1] = C_a[i::-1] * delta_t
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
    """Solve ``A r = C_t`` with Tikhonov regularisation.

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


def residue_to_cbf(residue0, rho_tissue=1.04, hematocrit=None, aif_type="whole_blood"):
    """CBF in mL/100 g/min from r(0).

    Inputs are residue r(t) from deconvolution with:
      - AIF in plasma concentration [mM]
      - Tissue curve in [mM]
      - Time step in seconds
    If AIF is plasma, convert to whole-blood flow using (1 - Hct).
    """

    r0 = float(residue0)
    scale = 1.0
    if aif_type == "plasma":
        if hematocrit is None:
            hematocrit = 0.42
        scale *= (1.0 - float(hematocrit))
    cbf = 6000.0 * r0 * scale / float(rho_tissue)
    return max(cbf, 0.0)


def residue_metrics(residue, dt, *, enforce_nonneg=True, enforce_monotone=True):
    """CTH (capillary transit time heterogeneity) measures the spread of capillary transit times in seconds.
    It is defined as the standard deviation of the transit time distribution h(t) derived from the residue r(t).
    MTT = ∫ r(t) dt,  h(t) = -dr/dt normalized to unit area,  CTH = sqrt( ∫ (t - μ)^2 h(t) dt ),  μ = ∫ t h(t) dt.
    """

    residue = np.asarray(residue, dtype=float).reshape(-1)
    if residue.size < 5 or np.isnan(residue).any() or not np.isfinite(dt) or dt <= 0:
        return np.nan, np.nan, np.full(residue.shape, np.nan), np.nan

    working = residue.copy()
    if enforce_nonneg:
        working = np.clip(working, 0.0, None)
    if enforce_monotone:
        working = np.minimum.accumulate(working)

    # np.trapezoid was introduced in newer NumPy releases; np.trapz provides
    # the same functionality and is available in older versions as well.
    mtt = float(np.trapz(working, dx=dt))

    h = np.maximum(0.0, -np.gradient(working, dt, edge_order=2))
    s = float(np.trapz(h, dx=dt))
    if s <= 0.0:
        return mtt, np.nan, np.full_like(working, np.nan), np.nan

    h /= s
    time = np.arange(working.size, dtype=float) * dt
    mu = float(np.trapz(time * h, dx=dt))
    variance = float(np.trapz(((time - mu) ** 2) * h, dx=dt))
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

    a0 = 2.0
    b0 = 1.0
    t0_0 = 0.0 if t0_seed is None else float(max(t0_seed, 0.0))

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
        if not (0.5 <= a <= 10.0 and 0.2 <= b <= 12.0 and 0.0 <= t0 <= 6.0 and flow_lo <= F_ml <= flow_hi):
            return np.ones_like(C_t) * 1e6

        rif, _ = _gamma_rif(time_array, a, b, t0, extraction=E)
        F_internal = F_ml * flow_scale
        Ct_hat = F_internal * _trapezoid_convolution(C_a, rif, dt)
        misfit = Ct_hat - C_t
        return np.sqrt(weights) * misfit

    x0 = [a0, b0, t0_0, F0]
    bounds_lower = [0.5, 0.2, 0.0, flow_lo]
    bounds_upper = [10.0, 12.0, 6.0, flow_hi]
    if optimise_E:
        x0.append(E0)
        bounds_lower.append(0.0)
        bounds_upper.append(0.4)

    try:
        sol = least_squares(residuals, x0,
                            bounds=(bounds_lower, bounds_upper),
                            method="trf", x_scale="jac")
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


def pick_lambda_via_l_curve(aif, tissue_curve, time_array, lambdas):
    """Return the lambda corresponding to the L-curve corner."""
    delta_t = np.diff(time_array).mean()
    A = construct_convolution_matrix(aif, delta_t)
    R, S = [], []
    for lam in lambdas:
        theta = tikhonov_regularization(A, tissue_curve, lam)
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


def plot_l_curve(aif, tissue_curve, time_array, lambdas, *, best=None):
    """Plot the L-curve and return the selected lambda."""
    import matplotlib.pyplot as plt

    delta_t = np.diff(time_array).mean()
    A = construct_convolution_matrix(aif, delta_t)
    R, S = [], []
    for lam in lambdas:
        theta = tikhonov_regularization(A, tissue_curve, lam)
        R.append(np.linalg.norm(A.dot(theta) - tissue_curve))
        S.append(np.linalg.norm(theta))

    if best is None:
        best = pick_lambda_via_l_curve(aif, tissue_curve, time_array, lambdas)
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
