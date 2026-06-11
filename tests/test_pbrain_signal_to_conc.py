"""Sanity tests for signal→concentration conversion (paper Eq. 2)."""

from __future__ import annotations

import numpy as np

from pbrain.signal_to_conc import REGISTRY


def test_saturation_recovery_zero_signal_returns_finite_concentration():
    """At S=0, ratio S/(M0·sin(α))=0; ln(1) = 0; C = -1/T1 / r1 (finite)."""
    conv = REGISTRY["saturation_recovery"]
    n = 10
    S = np.zeros(n)
    T1_ms = np.array(1000.0)    # scalar T1
    M0 = np.array(1000.0)
    C = conv.convert(S, T1_ms, M0, flip_angle_deg=8.0, tr_s=0.01118)
    assert np.all(np.isfinite(C))


def test_saturation_recovery_increases_with_signal():
    """C should increase monotonically with S for a fixed (T1, M0, α)."""
    conv = REGISTRY["saturation_recovery"]
    M0 = np.array(1000.0)
    T1_ms = np.array(1500.0)
    S = np.linspace(0.0, 0.5 * float(M0) * np.sin(np.deg2rad(8.0)), 20)
    C = conv.convert(S, T1_ms, M0, flip_angle_deg=8.0, tr_s=0.01118)
    diffs = np.diff(C)
    assert np.all(diffs >= -1e-9), f"C not monotonic non-decreasing: diffs={diffs}"


def test_vfa_spgr_baseline_signal_returns_zero_concentration():
    """Steady-state SPGR signal at the pre-bolus T1 must invert to C=0.

    S0 = M0·sin(α)·(1 − exp(−TR/T1₀)) / (1 − exp(−TR/T1₀)·cos(α))
    """
    conv = REGISTRY["vfa_spgr"]
    M0 = np.array(1000.0)
    T1_ms = np.array(1500.0)
    alpha = np.deg2rad(8.0)
    TR = 0.01118
    E1_0 = np.exp(-TR / (float(T1_ms) / 1000.0))
    S0 = float(M0) * np.sin(alpha) * (1.0 - E1_0) / (1.0 - E1_0 * np.cos(alpha))

    S = np.full(8, S0)
    C = conv.convert(S, T1_ms, M0, flip_angle_deg=8.0, tr_s=TR)
    np.testing.assert_array_almost_equal(C, 0.0, decimal=6)
