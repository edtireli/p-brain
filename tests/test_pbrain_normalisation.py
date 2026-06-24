"""Baseline+p95 normaliser tests."""

from __future__ import annotations

import numpy as np
import pytest

from pbrain.core.config import Config
from pbrain.normalisation import REGISTRY


def test_baseline_p95_subtracts_baseline_first_b_frames():
    norm = REGISTRY["baseline_p95"]
    t = np.arange(20) * 0.1
    curve = np.ones(20)
    curve[5:] = 2.0
    out = norm.normalise(curve, t, baseline_frames=5, percentile=95.0)
    # Baseline removed → first 5 should be ~0 after subtract; then divided by p95.
    assert out[:5].max() <= 1e-9
    assert out[5:].max() > 0.5


def test_baseline_p95_handles_batch():
    norm = REGISTRY["baseline_p95"]
    t = np.arange(10) * 0.1
    curves = np.tile(np.linspace(0, 1, 10), (3, 1)).T   # (T=10, V=3)
    out = norm.normalise(curves, t, baseline_frames=3)
    assert out.shape == curves.shape


def test_baseline_p95_zero_scale_doesnt_divide_by_zero():
    norm = REGISTRY["baseline_p95"]
    t = np.arange(10) * 0.1
    curve = np.ones(10) * 5.0    # all equal → baseline = 5; corrected = zeros; p95 = 0
    out = norm.normalise(curve, t, baseline_frames=3)
    np.testing.assert_array_almost_equal(out, 0.0)


# ──────────────────────────────────────────────────────────────────────────
# Regression guard: the default normaliser MUST stay ``identity``.
#
# A silent revert to ``legacy_shift`` re-introduces the ~2× voxelwise Patlak Ki
# inflation seen on 20250217x4 (cortical-GM median 0.073 → 0.149): legacy_shift
# (a) re-baselines the tissue curve a SECOND time on top of the Patlak model's
# own ``baseline_shift`` (find_baseline_point_advanced + custom_shifter), and
# (b) shifts the AIF too — but the validated/article Patlak path keeps the AIF
# raw. ``identity`` feeds raw conc + raw AIF so the model does exactly one shift.
# ──────────────────────────────────────────────────────────────────────────


def test_default_normaliser_is_identity():
    """Guards the legacy_shift→identity fix (2× Ki inflation regression)."""
    assert Config().normaliser == "identity"


def test_identity_is_pure_passthrough():
    norm = REGISTRY["identity"]
    t = np.arange(20) * 0.1
    curve = np.r_[np.zeros(5), np.linspace(0, 1, 15)] - 0.03   # non-trivial, dips < 0
    np.testing.assert_array_equal(norm.normalise(curve, t), curve)
    curves = np.tile(curve, (4, 1)).T                          # (T, V)
    np.testing.assert_array_equal(norm.normalise(curves, t), curves)


def test_legacy_shift_lifts_dipping_tail_but_identity_does_not():
    """The documented inflation mechanism: legacy_shift's per-voxel
    ``min(curve[bp:])`` subtraction lifts a tail that dips below zero, raising
    the late-window ``y = Ct/Ca`` and thus the Patlak slope. identity must not."""
    t = np.arange(40) * 2.43
    # baseline ~0 (frames 0..9), bolus onset + slow uptake, then a late tail that
    # drifts BELOW zero (as ~88% of real GM voxels do on 20250217x4). legacy_shift
    # subtracts min(curve[bp:]) < 0 → lifts the whole post-bolus curve.
    curve = np.zeros(40)
    curve[10:35] = np.linspace(0.0, 0.15, 25)    # Patlak-like uptake
    curve[35:] = -0.05                           # negative late drift
    legacy = REGISTRY["legacy_shift"].normalise(curve, t)
    ident = REGISTRY["identity"].normalise(curve, t)
    late = slice(35, 40)
    np.testing.assert_array_equal(ident, curve)                  # identity = raw, dip intact
    assert np.mean(legacy[late]) > np.mean(curve[late]) + 1e-3   # legacy lifts the tail
