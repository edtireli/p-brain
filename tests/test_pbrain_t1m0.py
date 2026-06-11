"""T1/M0 fitter tests on synthetic IR signal."""

from __future__ import annotations

import numpy as np

from pbrain.t1_m0 import REGISTRY


def test_inversion_recovery_recovers_synthetic_t1():
    """Generate a phantom with known A, B, T1; fitter must recover them."""
    fitter = REGISTRY["inversion_recovery"]

    A_true, B_true, T1_true_ms = 1000.0, 2000.0, 1500.0
    TI_s = np.array([0.12, 0.30, 0.60, 1.0, 2.0, 4.0, 10.0])
    signal_1d = A_true - B_true * np.exp(-TI_s / (T1_true_ms / 1000.0))

    # Pack into a (1, 1, 1, N) volume so the fitter sees a single voxel.
    signals = signal_1d.reshape(1, 1, 1, -1).astype(np.float32)
    res = fitter.fit(signals, TI_s)

    assert np.isfinite(res.t1_map_ms[0, 0, 0])
    assert abs(float(res.t1_map_ms[0, 0, 0]) - T1_true_ms) < 5.0   # 5 ms
    assert abs(float(res.m0_map[0, 0, 0]) - A_true) < 10.0
