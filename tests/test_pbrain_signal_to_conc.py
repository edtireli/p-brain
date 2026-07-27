"""Sanity tests for signal→concentration conversion (paper Eq. 2)."""

from __future__ import annotations

import numpy as np
import pytest

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


def _spgr_ratio_ceiling(t1_0_ms: float, tr_s: float, flip_deg: float) -> float:
    """max S/S0 the ratio inversion can represent = 1 / f(E1_0)."""
    E1_0 = np.exp(-tr_s / (t1_0_ms / 1000.0))
    f0 = (1.0 - E1_0) / (1.0 - np.cos(np.deg2rad(flip_deg)) * E1_0)
    return float(1.0 / f0)


def test_spgr_ratio_reports_its_analytic_ceiling():
    """The reported ceiling must equal 1/f(E1_0), the point where the inversion
    E1 = (1-g)/(1-cos·g) leaves its domain."""
    conv = REGISTRY["spgr_ratio"]
    TR, FLIP, T1_0 = 0.07, 15.0, 2000.0
    S = np.full((4, 4, 2, 12), 100.0)
    sat = conv.saturation(S, np.full((4, 4, 2), T1_0), None,
                          flip_angle_deg=FLIP, tr_s=TR, t1_0_ms=T1_0,
                          baseline_frames=4, baseline_method="fixed")
    assert sat["ceiling_ratio"] == pytest.approx(
        _spgr_ratio_ceiling(T1_0, TR, FLIP), rel=1e-6)


def test_spgr_ratio_saturation_zero_when_enhancement_stays_in_range():
    conv = REGISTRY["spgr_ratio"]
    TR, FLIP, T1_0 = 0.07, 15.0, 2000.0
    ceiling = _spgr_ratio_ceiling(T1_0, TR, FLIP)
    S = np.full((4, 4, 2, 12), 100.0)
    S[..., 5:] = 100.0 * (0.5 * ceiling)            # well inside the domain
    sat = conv.saturation(S, np.full((4, 4, 2), T1_0), None,
                          flip_angle_deg=FLIP, tr_s=TR, t1_0_ms=T1_0,
                          baseline_frames=4, baseline_method="fixed")
    assert sat["over_ceiling_fraction"] == 0.0


def _convert(conv, S, TR, FLIP, T1_0):
    return conv.convert(S, np.full(S.shape[:3], T1_0), np.ones(S.shape[:3]),
                        flip_angle_deg=FLIP, tr_s=TR, t1_0_ms=T1_0,
                        baseline_frames=4, baseline_method="fixed",
                        baseline_skip=0, max_conc_mM=50.0)


def _saturation(conv, S, TR, FLIP, T1_0):
    return conv.saturation(S, np.full(S.shape[:3], T1_0), None,
                           flip_angle_deg=FLIP, tr_s=TR, t1_0_ms=T1_0,
                           baseline_frames=4, baseline_method="fixed")


def test_spgr_ratio_saturation_flags_the_clamped_regime():
    """Just past the ceiling, E1 clips low and C explodes to the clamp."""
    conv = REGISTRY["spgr_ratio"]
    TR, FLIP, T1_0 = 0.07, 15.0, 2000.0
    ceiling = _spgr_ratio_ceiling(T1_0, TR, FLIP)
    S = np.full((4, 4, 2, 12), 100.0)
    S[..., 5:] = 100.0 * (1.005 * ceiling)         # inside [ceiling, ceiling/cos)

    sat = _saturation(conv, S, TR, FLIP, T1_0)
    assert sat["over_ceiling_fraction"] == pytest.approx(1.0)
    assert sat["wrapped_fraction"] == 0.0
    # At the clamp bar the residual baseline subtraction, i.e. not a measurement.
    assert np.nanmax(_convert(conv, S, TR, FLIP, T1_0)) == pytest.approx(50.0, rel=0.05)


def test_spgr_ratio_saturation_flags_the_wrapped_regime():
    """Past ceiling/cosα both signs flip, E1 wraps above 1 and the concentration
    collapses to ~0 — the strongest-enhancing voxels report as unenhancing."""
    conv = REGISTRY["spgr_ratio"]
    TR, FLIP, T1_0 = 0.07, 15.0, 2000.0
    ceiling = _spgr_ratio_ceiling(T1_0, TR, FLIP)
    S = np.full((4, 4, 2, 12), 100.0)
    S[..., 5:] = 100.0 * (1.2 * ceiling)           # well past ceiling/cos

    sat = _saturation(conv, S, TR, FLIP, T1_0)
    assert sat["over_ceiling_fraction"] == pytest.approx(1.0)
    assert sat["wrapped_fraction"] == pytest.approx(1.0)
    assert sat["wrap_ratio"] == pytest.approx(ceiling / np.cos(np.deg2rad(FLIP)))

    C = _convert(conv, S, TR, FLIP, T1_0)
    assert np.nanmax(C) < 1.0, "a 2.3x enhancement must not read as ~0 mM silently"


def test_spgr_ratio_a_bolus_crossing_the_ceiling_oscillates():
    """The regimes are ~0.07x apart, so a bolus passing through gives a curve that
    jumps between the clamp and zero. Physically impossible, still looks like data."""
    conv = REGISTRY["spgr_ratio"]
    TR, FLIP, T1_0 = 0.07, 15.0, 2000.0
    ceiling = _spgr_ratio_ceiling(T1_0, TR, FLIP)
    S = np.full((2, 2, 1, 12), 100.0)
    ramp = np.array([1.0, 1.5, 1.99, 2.2, 2.3, 2.2, 1.99]) * ceiling / 2.0
    S[..., 5:] = 100.0 * ramp

    C = _convert(conv, S, TR, FLIP, T1_0)[0, 0, 0][5:]
    # The decisive property is not the magnitude but the ORDER: the brightest
    # frames report the lowest concentration. No monotone signal model can do that.
    brightest, dimmest = int(np.argmax(ramp)), int(np.argmin(ramp))
    assert C[brightest] < C[dimmest], (
        f"brightest frame ({ramp[brightest]:.2f}x) reported {C[brightest]:.2f} mM, "
        f"dimmest ({ramp[dimmest]:.2f}x) reported {C[dimmest]:.2f} mM")


def test_spgr_ratio_saturation_ignores_air():
    """Air has S0 ~ 0, so its ratio explodes for reasons unrelated to the ceiling."""
    conv = REGISTRY["spgr_ratio"]
    TR, FLIP, T1_0 = 0.07, 15.0, 2000.0
    S = np.full((6, 6, 2, 12), 100.0)
    S[0, :, :, :] = 1e-3                            # an air plane
    S[0, :, :, 5:] = 1.0                            # ratio 1000x, pure noise
    sat = conv.saturation(S, np.full((6, 6, 2), T1_0), None,
                          flip_angle_deg=FLIP, tr_s=TR, t1_0_ms=T1_0,
                          baseline_frames=4, baseline_method="fixed")
    assert sat["over_ceiling_fraction"] == 0.0
    assert sat["foreground_voxels"] == 60           # 6*6*2 minus the 6*2 air plane
