import numpy as np
from scipy.optimize import curve_fit


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
    """Fit the extended Tofts two-compartment model and return Ki and lambda."""
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
    return Ki, lambda_val, Ki_std
