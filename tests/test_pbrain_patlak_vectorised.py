"""Parity + correctness tests for the vectorised Patlak path."""

from __future__ import annotations

import numpy as np
import pytest

from pbrain.models import REGISTRY, CurveInputs
from pbrain.models._patlak_tools import (
    fit_patlak_vectorised,
    find_patlak_tail,
    vectorised_huber,
    vectorised_ols,
)


def _synth(ki_true=0.5, vb_true=2.0, n=250, dt=2.463, noise_sd=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt

    def bolus(tt, t0, amp=1.0, alpha=3.0, tau=10.0):
        x = np.maximum((tt - t0) / tau, 0.0)
        return amp * (x ** alpha) * np.exp(alpha * (1 - x))

    c_a = bolus(t, 20.0, amp=1.5) + 0.7 * bolus(t, 180.0, amp=1.0)
    ki_per_s = ki_true / 6000.0
    vb_frac = vb_true / 100.0
    integ = np.concatenate(([0.0],
                            np.cumsum(0.5 * (c_a[1:] + c_a[:-1]) * np.diff(t))))
    c_t = vb_frac * c_a + ki_per_s * integ + rng.normal(0, noise_sd, c_a.shape)
    return c_t, c_a, t


def test_smart_tail_finds_after_last_peak():
    """The smart-tail start index must be after every bolus peak."""
    _, c_a, t = _synth()  # two boluses at t=20 and t=180
    start, end = find_patlak_tail(c_a, t, mode="smart")
    # Second bolus peaks around frame ~80 (180 s / 2.463 s/frame)
    assert start >= 75, f"tail starts too early: {start}"
    assert end == len(c_a)


def test_smart_tail_single_bolus():
    """Single-bolus AIF: tail starts after the one peak."""
    rng = np.random.default_rng(1)
    t = np.arange(250) * 2.463
    x = np.maximum((t - 20.0) / 10.0, 0.0)
    c_a = (x ** 3) * np.exp(3 * (1 - x))
    start, end = find_patlak_tail(c_a, t, mode="smart")
    assert start >= 10  # past the (only) peak
    assert end == 250


def test_vectorised_ols_matches_numpy_polyfit():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + 1.0 + rng.normal(0, 0.1, 50)
    slope, intercept = vectorised_ols(x, y)
    np_slope, np_intercept = np.polyfit(x, y, 1)
    assert slope == pytest.approx(np_slope, rel=1e-10)
    assert intercept == pytest.approx(np_intercept, rel=1e-10)


def test_vectorised_ols_broadcast():
    """Same slope on every column of a 2-D y array."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 50)
    Y = np.stack([2.5 * x + 1.0 + rng.normal(0, 0.01, 50) for _ in range(7)], axis=1)
    slope, intercept = vectorised_ols(x, Y)
    assert slope.shape == (7,)
    assert np.all(np.abs(slope - 2.5) < 0.01)


def test_huber_equals_ols_on_clean_data():
    rng = np.random.default_rng(4)
    x = np.linspace(0, 10, 50)
    Y = np.stack([2.5 * x + 1.0 + rng.normal(0, 0.05, 50) for _ in range(5)], axis=1)
    s_ols, _ = vectorised_ols(x, Y)
    s_huber, _ = vectorised_huber(x, Y)
    # Without outliers Huber ≈ OLS within a few % (slight efficiency loss)
    np.testing.assert_allclose(s_huber, s_ols, rtol=0.05)


def test_huber_resistant_to_outliers():
    """A single corrupting outlier should pull OLS far more than Huber.

    A mid-x outlier tilts the intercept far more than the slope; check both.
    """
    rng = np.random.default_rng(5)
    x = np.linspace(0, 10, 50)
    y_clean = 2.5 * x + 1.0
    y = y_clean + rng.normal(0, 0.05, 50)
    y[25] += 50.0   # massive outlier at mid-x

    s_ols, i_ols = vectorised_ols(x, y.reshape(-1, 1))
    s_huber, i_huber = vectorised_huber(x, y.reshape(-1, 1))

    err_ols_slope = abs(float(s_ols[0]) - 2.5)
    err_huber_slope = abs(float(s_huber[0]) - 2.5)
    err_ols_int = abs(float(i_ols[0]) - 1.0)
    err_huber_int = abs(float(i_huber[0]) - 1.0)

    # Huber should beat OLS on both, and dramatically on intercept.
    assert err_huber_slope < err_ols_slope, (
        f"Huber slope err {err_huber_slope:.4f} ≥ OLS {err_ols_slope:.4f}"
    )
    assert err_huber_int < err_ols_int / 10, (
        f"Huber intercept err {err_huber_int:.4f} ≥ OLS {err_ols_int:.4f}/10"
    )


def test_fit_patlak_vectorised_recovers_known_ki():
    ki_true = 0.50
    c_t, c_a, t = _synth(ki_true=ki_true, noise_sd=0.0)
    res = fit_patlak_vectorised(c_t.reshape(-1, 1), c_a, t,
                                 tail_mode="smart", regression="ols")
    ki = float(res["ki_ml_per_100g_min"][0])
    assert ki == pytest.approx(ki_true, abs=0.03), f"Ki={ki:.4f}"


def test_plugin_uses_vectorised_path_for_2d_input():
    """PatlakModel.fit should route 2-D input through fit_patlak_vectorised."""
    c_t1, c_a, t = _synth(ki_true=0.30, seed=10)
    c_t2, _, _ = _synth(ki_true=0.70, seed=11)
    c_t = np.stack([c_t1, c_t2], axis=1)
    res = REGISTRY["patlak"].fit(CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t))
    # aux should expose the vectorised metadata
    assert "tail_start" in res.aux
    assert res.aux["regression"] == "huber"


def test_legacy_tail_mode_falls_back_to_loop():
    c_t1, c_a, t = _synth(ki_true=0.30, seed=20)
    c_t = c_t1.reshape(-1, 1)
    res = REGISTRY["patlak"].fit(
        CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t),
        tail_mode="legacy",
    )
    assert res.aux["tail_mode"] == "legacy"
    assert res.aux["regression"] == "ols"
