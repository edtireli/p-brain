import numpy as np
from scipy.optimize import curve_fit, least_squares


def extended_tofts_model(t, Ktrans, ve, vp, Cp):
    """Basic extended Tofts implementation used for the two-compartment model."""
    integrand = np.zeros_like(t)
    for i in range(len(t)):
        min_len = min(len(Cp[:i+1]), len(t[:i+1]))
        integral = np.trapz(Cp[:min_len] * np.exp(-(t[i] - t[:min_len]) * Ktrans / ve), x=t[:min_len])
        integrand[i] = integral
    min_len = min(len(integrand), len(Cp))
    return Ktrans * integrand[:min_len] + vp * Cp[:min_len]


def two_compartment_fit(c_input, c_tissue, time_array):
    """Fit the extended Tofts two-compartment model and return Ki, lambda and
    the fitted tissue curve."""
    if c_input.shape[0] != c_tissue.shape[0]:
        raise ValueError("The number of time points in c_input and c_tissue must be the same.")

    initial_guess = [0.5/6000, 0.2, 0.05]
    popt, pcov = curve_fit(
        lambda t, Ktrans, ve, vp: extended_tofts_model(t, Ktrans, ve, vp, c_input),
        time_array,
        c_tissue,
        p0=initial_guess,
    )

    Ktrans_fitted, ve_fitted, vp_fitted = popt
    std_dev_Ktrans = np.sqrt(np.diag(pcov))[0]

    Ki = Ktrans_fitted * 6000
    Ki_std = std_dev_Ktrans * 6000
    lambda_val = ve_fitted * 100

    fitted_curve = extended_tofts_model(time_array, Ktrans_fitted, ve_fitted, vp_fitted, c_input)

    return Ki, lambda_val, Ki_std, fitted_curve


def construct_convolution_matrix(C_a, delta_t):
    """Build a Toeplitz matrix for convolution."""
    n = len(C_a)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, : i + 1] = C_a[i::-1] * delta_t
    return A


def tikhonov_regularization(A, C_t, lambd):
    """Solve A x = C_t with Tikhonov regularisation."""
    n = A.shape[1]
    L = np.eye(n)
    ATA = A.T @ A
    LTL = L.T @ L
    regularized = ATA + lambd * LTL
    return np.linalg.solve(regularized, A.T @ C_t)


def two_compartment_tikhonov_fit(c_input, c_tissue, time_array, lambd=0.1):
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
    """

    if c_input.shape[0] != c_tissue.shape[0]:
        raise ValueError("The number of time points in c_input and c_tissue must be the same.")

    initial_guess = [0.5 / 6000, 0.2, 0.05]

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
    delta_t = np.diff(time_array)[0]
    A = construct_convolution_matrix(c_input, delta_t)
    residue = tikhonov_regularization(A, c_tissue, lambd)
    CBF = residue[0] * 6000

    return Ki, lam, vp_fitted, CBF, fitted_curve
