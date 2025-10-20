import numpy as np
from scipy.optimize import least_squares
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
    lam_sq = float(max(lambd, 0.0)) ** 2
    regularised = ata + lam_sq * ltl
    rhs = A.T @ C_t
    return np.linalg.solve(regularised, rhs)


def residue_to_cbf(residue0):
    """Convert the residue's initial value to CBF in ml/100g/min."""

    scale = 6000.0 / settings.TISSUE_DENSITY
    if settings.PLASMA_DERIVED_AIF:
        scale *= max(1.0 - settings.HEMATOCRIT, 0.0)
    cbf = float(residue0) * scale
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
    x0 = np.asarray(x0, dtype=float)

    def residual(theta):
        Ktrans, ve, vp = theta
        Ct_pred = extended_tofts_model(t, Ktrans, ve, vp, Cp)
        misfit = Ct_pred - Ct
        w = np.linalg.norm(misfit) / max(np.linalg.norm(theta), 1e-8)
        penalty = np.sqrt(lambd) * w * np.array(theta)
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
