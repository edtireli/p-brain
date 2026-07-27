"""Tests for the bounded-inversion SPGR converter.

The property that matters is structural, not numerical: the mapping signal→[Gd] must
be monotone everywhere, including past the point where the model runs out of domain.
The closed-form converter is not, and that is the bug this exists to remove.
"""

from __future__ import annotations

import numpy as np
import pytest

from pbrain.signal_to_conc import REGISTRY
from pbrain.signal_to_conc.spgr_bounded import R1_MAX_DEFAULT

TR, FLIP, T1_0, R1_REL = 0.07, 15.0, 2000.0, 2.8
OPTS = dict(flip_angle_deg=FLIP, tr_s=TR, r1_per_s_mM=R1_REL, t1_0_ms=T1_0,
            baseline_method="fixed", baseline_frames=4, baseline_skip=0)


def _ceiling(t1_0_ms=T1_0, tr_s=TR, flip=FLIP) -> float:
    E1_0 = np.exp(-tr_s / (t1_0_ms / 1000.0))
    return float((1 - np.cos(np.deg2rad(flip)) * E1_0) / (1 - E1_0))


def _series(ratios: np.ndarray, shape=(2, 2, 1)) -> np.ndarray:
    """A series whose first 4 frames are baseline and whose tail follows `ratios`."""
    S = np.full((*shape, 4 + len(ratios)), 100.0)
    S[..., 4:] = 100.0 * np.asarray(ratios, dtype=float)
    return S


def _convert(key, S, **kw):
    return REGISTRY[key].convert(S, np.full(S.shape[:3], T1_0),
                                 np.ones(S.shape[:3]), **{**OPTS, **kw})


def test_monotone_in_signal_across_the_whole_range():
    """The decisive property. Sweep enhancement from flat to far past the ceiling;
    the reported concentration must never decrease."""
    ratios = np.linspace(1.0, 4.0, 200)
    C = _convert("spgr_bounded", _series(ratios))[0, 0, 0][4:]
    assert np.all(np.diff(C) >= -1e-9), "concentration decreased as signal increased"


def test_closed_form_is_not_monotone_which_is_why_this_exists():
    """Guards the premise: spgr_ratio inverts sign past ceiling/cosa."""
    ratios = np.linspace(1.0, 4.0, 200)
    C = _convert("spgr_ratio", _series(ratios), max_conc_mM=50.0)[0, 0, 0][4:]
    assert np.min(np.diff(C)) < -1.0, (
        "spgr_ratio no longer wraps — if this was fixed deliberately, "
        "spgr_bounded's rationale needs revisiting")


def test_agrees_with_the_closed_form_inside_the_valid_domain():
    """Same physics: where the analytic inversion is defined, both must match."""
    ratios = np.linspace(1.0, 0.95 * _ceiling(), 50)
    S = _series(ratios)
    a = _convert("spgr_bounded", S)[0, 0, 0][4:]
    b = _convert("spgr_ratio", S, max_conc_mM=0.0)[0, 0, 0][4:]
    np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)


def test_baseline_signal_gives_zero_concentration():
    C = _convert("spgr_bounded", _series(np.ones(8)))[0, 0, 0]
    np.testing.assert_allclose(C, 0.0, atol=1e-6)


def test_saturates_at_the_r1_bound_instead_of_wrapping():
    """Past the ceiling the curve must pin high — the failure mode being replaced is
    a collapse to ~0 mM at exactly these signal levels."""
    C = _convert("spgr_bounded", _series(np.array([1.0, 2.5, 3.0, 6.0])))[0, 0, 0][4:]
    bound = (R1_MAX_DEFAULT - 1000.0 / T1_0) / R1_REL
    assert C[-1] == pytest.approx(bound, rel=0.02)
    assert np.all(C[1:] > 1.0), "strongly enhancing frames must not report ~0 mM"


def test_output_is_bounded_without_any_clamp():
    """max_conc_mM defaults off because the R1 bracket already bounds the result."""
    C = _convert("spgr_bounded", _series(np.array([1.0, 50.0, 1000.0])))
    assert np.all(np.isfinite(C))
    assert np.nanmax(C) <= (R1_MAX_DEFAULT - 1000.0 / T1_0) / R1_REL + 1e-6


def test_reports_a_ceiling_but_no_wrap_population():
    S = _series(np.full(6, 3.0), shape=(4, 4, 2))
    sat = REGISTRY["spgr_bounded"].saturation(S, np.full(S.shape[:3], T1_0), None, **OPTS)
    assert sat["ceiling_ratio"] == pytest.approx(_ceiling(), rel=1e-2)
    assert sat["over_ceiling_fraction"] == pytest.approx(1.0)
    # A monotone inversion has no second branch, so this is structurally unreachable.
    assert sat["wrapped_fraction"] == 0.0
    assert not np.isfinite(sat["wrap_ratio"])


def test_handles_a_3d_volume():
    S = np.full((3, 3, 2), 100.0)
    C = REGISTRY["spgr_bounded"].convert(S, np.full((3, 3, 2), T1_0),
                                         np.ones((3, 3, 2)), **OPTS)
    assert C.shape == (3, 3, 2)


def test_handles_a_single_1d_curve():
    """The other converters accept a bare (T,) curve with a scalar T1; so must this."""
    S = np.concatenate([np.full(4, 100.0), np.full(6, 150.0)])
    C = REGISTRY["spgr_bounded"].convert(S, np.array(T1_0), np.array(1.0), **OPTS)
    assert C.shape == S.shape
    assert np.all(np.isfinite(C)) and C[-1] > C[0]


def test_zero_baseline_voxels_stay_finite():
    """Air has S0 = 0; it must not poison the volume with inf/nan-propagating values."""
    S = _series(np.full(6, 2.0), shape=(3, 3, 1))
    S[0, 0, 0, :] = 0.0
    C = _convert("spgr_bounded", S)
    assert np.all(np.isfinite(C[1:, 1:, :])), "a dead voxel corrupted its neighbours"
