"""T1/M0 fitter tests on synthetic IR signal."""

from __future__ import annotations

import numpy as np

from pbrain.t1_m0 import REGISTRY
from pbrain.t1_m0.inversion_recovery import _turboflash_basis


def test_inversion_recovery_recovers_synthetic_t1():
    """A−B·exp phantom: the plain inversion-recovery model recovers A, B, T1.

    ``recovery_model='ir'`` is required because the fitter now DEFAULTS to the
    TurboFLASH model (true-M0); its Look-Locker T1 differs from a bare A−B·exp
    recovery, so feeding an A−B·exp phantom to the default would mis-fit. The
    TurboFLASH default path is covered by the test below.
    """
    fitter = REGISTRY["inversion_recovery"]

    A_true, B_true, T1_true_ms = 1000.0, 2000.0, 1500.0
    TI_s = np.array([0.12, 0.30, 0.60, 1.0, 2.0, 4.0, 10.0])
    signal_1d = A_true - B_true * np.exp(-TI_s / (T1_true_ms / 1000.0))

    # Pack into a (1, 1, 1, N) volume so the fitter sees a single voxel.
    signals = signal_1d.reshape(1, 1, 1, -1).astype(np.float32)
    res = fitter.fit(signals, TI_s, recovery_model="ir")

    assert np.isfinite(res.t1_map_ms[0, 0, 0])
    assert abs(float(res.t1_map_ms[0, 0, 0]) - T1_true_ms) < 5.0   # 5 ms
    assert abs(float(res.m0_map[0, 0, 0]) - A_true) < 10.0


def test_turboflash_recovers_synthetic_t1_and_true_m0():
    """TurboFLASH phantom (the DEFAULT model): recover T1 and the TRUE M0.

    The TurboFLASH forward signal is S = M0·sinα·[…], so the fitter must return
    the *true* equilibrium M0, not the apparent M0·sinα a bare A−B·exp fit
    yields (that missing sinα factor is what doubled downstream concentration).
    """
    fitter = REGISTRY["inversion_recovery"]

    M0_true, T1_true_ms = 1000.0, 1500.0
    alpha_deg, tr_s, nph = 30.0, 0.00396, 1
    TI_s = np.array([0.12, 0.30, 0.60, 1.0, 2.0, 4.0, 10.0])

    basis = _turboflash_basis(TI_s, T1_true_ms / 1000.0, np.radians(alpha_deg), tr_s, nph)
    signal_1d = M0_true * np.asarray(basis)

    signals = signal_1d.reshape(1, 1, 1, -1).astype(np.float32)
    res = fitter.fit(signals, TI_s, recovery_model="turboflash",
                     alpha_deg=alpha_deg, tr_s=tr_s, nph=nph)

    assert np.isfinite(res.t1_map_ms[0, 0, 0])
    assert abs(float(res.t1_map_ms[0, 0, 0]) - T1_true_ms) < 5.0
    # recovered M0 is the TRUE equilibrium (~2× the apparent M0·sin30°)
    assert abs(float(res.m0_map[0, 0, 0]) - M0_true) < 10.0
