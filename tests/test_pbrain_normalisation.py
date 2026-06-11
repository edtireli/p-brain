"""Baseline+p95 normaliser tests."""

from __future__ import annotations

import numpy as np
import pytest

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
