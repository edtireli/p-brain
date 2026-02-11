import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


def shift_curve_pchip(time_s: np.ndarray, curve: np.ndarray, shift_seconds: float) -> np.ndarray:
    """MATLAB-equivalent `inputshift3(time, Ca, deltaT)` using pchip.

    The curve is shifted by evaluating the original samples at `time - shift_seconds`
    with MATLAB's padding conventions:
    - shift_seconds > 0: pad with leading zeros
    - shift_seconds < 0: extend tail with last value
    """

    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    y = np.asarray(curve, dtype=float).reshape(-1)
    if time_s.size != y.size:
        n = min(time_s.size, y.size)
        time_s = time_s[:n]
        y = y[:n]

    if time_s.size == 0:
        return np.asarray([], dtype=float)
    if time_s.size == 1 or float(shift_seconds) == 0.0:
        return y.copy()

    if time_s.size < 3:
        time_unit = float(time_s[1] - time_s[0])
    else:
        time_unit = float(time_s[2] - time_s[1])
    if not np.isfinite(time_unit) or time_unit <= 0:
        # Fallback to median positive delta.
        deltas = np.diff(time_s)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
        time_unit = float(deltas[0]) if deltas.size else 1.0

    deltaT = float(shift_seconds)
    notimeunit = int(np.ceil(abs(deltaT) / time_unit))
    numberobs = int(y.size)

    if deltaT > 0:
        timetemp = np.arange(
            (time_s[0] + deltaT) - (notimeunit * time_unit),
            (time_s[numberobs - 1] + deltaT) + (0.5 * time_unit),
            time_unit,
            dtype=float,
        )
        # Ensure exact MATLAB length: numberobs + notimeunit
        if timetemp.size != (numberobs + notimeunit):
            timetemp = (time_s[0] + deltaT) - (notimeunit * time_unit) + np.arange(
                numberobs + notimeunit, dtype=float
            ) * time_unit

        sCa = np.zeros(numberobs + notimeunit, dtype=float)
        sCa[notimeunit : notimeunit + numberobs] += y[:numberobs]
    else:
        timetemp = np.arange(
            (time_s[0] + deltaT),
            (time_s[numberobs - 1] + deltaT + (notimeunit * time_unit)) + (0.5 * time_unit),
            time_unit,
            dtype=float,
        )
        if timetemp.size != (numberobs + notimeunit):
            timetemp = (time_s[0] + deltaT) + np.arange(numberobs + notimeunit, dtype=float) * time_unit

        sCa = np.zeros(numberobs + notimeunit, dtype=float)
        sCa[:numberobs] = y[:numberobs]
        if notimeunit:
            sCa[numberobs:] = y[numberobs - 1]

    # `pchip` interpolation (MATLAB interp1(...,'pchip')).
    interp = PchipInterpolator(timetemp, sCa, extrapolate=False)
    out = interp(time_s)

    # MATLAB never leaves NaNs here because timetemp spans time_s.
    # Still, be defensive.
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(float)


def _gamma_variate(time_s: np.ndarray, A: float, alpha: float, t_max: float) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    out = np.zeros_like(time_s, dtype=float)
    if A <= 0 or alpha <= 0 or t_max <= 0:
        return out

    t = time_s
    pos = t > 0
    if not np.any(pos):
        return out

    # Compute in log-domain for stability.
    tp = t[pos]
    logF = (
        np.log(A)
        + (-alpha * np.log(t_max))
        + alpha
        + (alpha * np.log(tp))
        + (-alpha * tp / t_max)
    )
    out[pos] = np.exp(logF)
    out[~np.isfinite(out)] = 0.0
    return out


def estimate_bolus_arrival_shift_seconds(
    curve: np.ndarray,
    time_s: np.ndarray,
    *,
    upsample_factor: int = 10,
) -> tuple[float, np.ndarray]:
    """MATLAB-equivalent `curve2InitEnhanc_2`.

    Returns:
      (shift_seconds, fitted_curve_on_upsampled_grid)

    Notes:
    - The returned shift is the fitted `shift1` parameter (MATLAB returns `x(4)`).
    - The upsampled fitted curve corresponds to the analytic gamma-variate evaluated
      on the upsampled time grid used for the fit.
    """

    c_obs = np.asarray(curve, dtype=float).reshape(-1)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    n = min(c_obs.size, time.size)
    c_obs = c_obs[:n]
    time = time[:n]

    if n < 4 or not np.all(np.isfinite(c_obs)) or not np.all(np.isfinite(time)):
        return 0.0, np.zeros(0, dtype=float)

    dt = float(time[1] - time[0])
    if not np.isfinite(dt) or dt <= 0:
        deltas = np.diff(time)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
        if deltas.size == 0:
            return 0.0, np.zeros(0, dtype=float)
        dt = float(deltas[0])

    h = int(max(2, upsample_factor))
    # MATLAB's `interp(time,h_factor)` then trims `-h_factor`, yielding (n-1)*h points.
    time_h = time[0] + np.arange((n - 1) * h, dtype=float) * (dt / h)
    c_obs_h = np.interp(time_h, time, c_obs).astype(float)
    if c_obs_h.size:
        c_obs_h[0] = 0.0

    dummy = int(np.round(60.0 / dt))
    dummy = max(1, min(dummy, n))
    A1 = float(np.max(c_obs[:dummy]))
    t_max_idx = int(np.argmax(c_obs[:dummy]))
    t_max = float(time[t_max_idx])

    NO = int(t_max_idx + 7)
    NO = max(1, min(NO, n))
    NO_h = int(NO * h)
    NO_h = max(1, min(NO_h, time_h.size))

    shift1 = 15.0
    t_max1 = t_max - shift1
    if t_max1 <= 0.1:
        t_max1 = 0.2
    alfa1 = 10.0

    x0 = np.array([A1 if np.isfinite(A1) else 0.0, alfa1, t_max1, shift1], dtype=float)

    lb = np.array([
        max((A1 - 0.2 * A1), 0.0),
        0.01,
        0.1,
        10.0,
    ], dtype=float)
    ub = np.array([
        np.inf,
        15.0,
        max(t_max1 + 0.2 * t_max1, 0.1),
        50.0,
    ], dtype=float)

    t_fit = time_h[:NO_h]
    y_fit = c_obs_h[:NO_h]

    def model(params: np.ndarray) -> np.ndarray:
        A, alpha, tmax, shift = (float(abs(params[0])), float(abs(params[1])), float(abs(params[2])), float(abs(params[3])))
        f = _gamma_variate(t_fit, A, alpha, tmax)
        f = shift_curve_pchip(t_fit, f, shift)
        return f

    def residuals(params: np.ndarray) -> np.ndarray:
        return model(params) - y_fit

    try:
        res = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            ftol=1e-6,
            xtol=1e-4,
            gtol=1e-6,
            max_nfev=500,
        )
        x = res.x
    except Exception:
        x = x0

    shift_seconds = float(abs(x[3])) if np.isfinite(x[3]) else 0.0
    c_analyt = model(x)
    c_analyt = np.where(c_analyt <= 1e-4, 0.0, c_analyt)
    return shift_seconds, c_analyt
