"""Voxel-wise CNR QC (R3.7) + short-AIF Tikhonov robustness (regression).

CNR is computed per voxel as

    CNR = (peak post-bolus conc − pre-bolus baseline mean) / pre-bolus SD

so a synthetic curve with a known baseline (mean / SD) and a known peak has a
CNR we can assert in closed form. The Tikhonov regression covers the
"AIF a few frames shorter than the DCE" crash that previously raised
``Ct_mat first dimension must match time_s``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pbrain.core.qc import cnr_summary, voxelwise_cnr


# ── Voxel-wise CNR ────────────────────────────────────────────────────────

def test_voxelwise_cnr_known_value():
    """Single voxel with a flat noisy baseline and a clean post-bolus peak.

    Baseline = 5 frames drawn so that mean ≈ 0 and we set them explicitly to a
    fixed pattern with known std; peak is set to a known height. CNR must equal
    (peak − base_mean) / base_std.
    """
    baseline_frames = 5
    base = np.array([1.0, -1.0, 1.0, -1.0, 0.0], dtype=float)  # mean 0, std=0.8
    peak = 10.0
    tail = np.array([peak, 4.0, 2.0], dtype=float)
    curve = np.concatenate([base, tail])

    conc = curve.reshape(1, 1, 1, -1)
    cnr = voxelwise_cnr(conc, baseline_frames=baseline_frames)

    base_mean = base.mean()
    base_std = base.std()
    expected = (peak - base_mean) / base_std
    assert cnr.shape == (1, 1, 1)
    assert cnr[0, 0, 0] == pytest.approx(expected, rel=1e-12)


def test_voxelwise_cnr_flat_baseline_is_zero_not_inf():
    """A perfectly flat (zero-SD) baseline must yield CNR=0, never inf/NaN."""
    curve = np.array([0.0, 0.0, 0.0, 5.0, 2.0], dtype=float)
    conc = curve.reshape(1, 1, 1, -1)
    cnr = voxelwise_cnr(conc, baseline_frames=3)
    assert np.isfinite(cnr).all()
    assert cnr[0, 0, 0] == 0.0


def test_voxelwise_cnr_mask_sets_outside_nan():
    rng = np.random.default_rng(0)
    conc = rng.standard_normal((4, 4, 2, 8))
    mask = np.zeros((4, 4, 2), dtype=bool)
    mask[0, 0, 0] = True
    cnr = voxelwise_cnr(conc, baseline_frames=3, mask=mask)
    assert np.isfinite(cnr[0, 0, 0])
    assert np.isnan(cnr[~mask]).all()


def test_cnr_summary_fraction_below_threshold():
    """Half the brain at CNR=1 (below 2), half at CNR=10 (above)."""
    cnr = np.array([1.0, 1.0, 10.0, 10.0]).reshape(2, 2, 1)
    s = cnr_summary(cnr, threshold=2.0)
    assert s["n_voxels"] == 4
    assert s["n_below"] == 2
    assert s["fraction_below"] == pytest.approx(0.5)
    assert s["median"] == pytest.approx(5.5)
    # 50% below is not > 0.5, so this is a pass; push it over to warn.
    cnr_bad = np.array([1.0, 1.0, 1.0, 10.0]).reshape(2, 2, 1)
    assert cnr_summary(cnr_bad, threshold=2.0)["status"] == "warn"


# ── Short-AIF Tikhonov regression ─────────────────────────────────────────

def _synthetic_dce(n_time: int):
    t = np.arange(n_time, dtype=float) * 1.5
    # Gamma-variate-ish AIF and a scaled tissue response.
    aif = np.zeros(n_time)
    peak = 5
    aif[peak:] = (np.arange(n_time - peak) ** 2) * np.exp(-(np.arange(n_time - peak)) / 4.0)
    aif = aif / (aif.max() or 1.0) * 4.0
    ct = np.convolve(aif, np.exp(-t / 8.0))[:n_time] * 0.05
    return t, aif, ct


def test_tikhonov_short_aif_does_not_crash():
    """AIF/time 3 frames shorter than the tissue series must align defensively
    rather than raising 'Ct_mat first dimension must match time_s'."""
    from pbrain.models.tikhonov import build_tikhonov_solver

    n_dce = 40
    t, aif, ct = _synthetic_dce(n_dce)
    # AIF/time clipped to fewer frames than the tissue curve.
    n_aif = n_dce - 3
    solver = build_tikhonov_solver(t[:n_aif], aif[:n_aif], n_lambdas=20)

    ct_mat = np.column_stack([ct, ct * 0.8])  # (40, 2): longer than the AIF
    batch = solver(ct_mat)  # must not raise
    assert batch.cbf_ml_per_100g_min.shape == (2,)
    assert np.all(np.isfinite(batch.cbf_ml_per_100g_min))


def test_tikhonov_equal_length_unchanged_by_guard():
    """The short-AIF guard must be a no-op for equal-length inputs: solving a
    Ct of exactly ``n`` rows gives the same numbers as before (the slice
    ``Ct_mat[:n]`` is identity)."""
    from pbrain.models.tikhonov import build_tikhonov_solver

    n_dce = 36
    t, aif, ct = _synthetic_dce(n_dce)
    solver = build_tikhonov_solver(t, aif, n_lambdas=25)

    ct_mat = ct.reshape(-1, 1)
    b_equal = solver(ct_mat)
    # Same solver, oversized Ct (one extra trailing frame) → truncated to n,
    # must match the equal-length result exactly.
    ct_long = np.vstack([ct_mat, ct_mat[-1:]])  # (37, 1)
    solver2 = build_tikhonov_solver(t, aif, n_lambdas=25)
    b_trunc = solver2(ct_long)
    np.testing.assert_allclose(
        b_equal.cbf_ml_per_100g_min, b_trunc.cbf_ml_per_100g_min, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        b_equal.mtt_s, b_trunc.mtt_s, rtol=0, atol=0
    )


def test_tikhonov_too_short_ct_raises_clear_error():
    """A tissue series SHORTER than the AIF can't be aligned by trailing-drop;
    the solver must raise a clear, actionable error."""
    from pbrain.models.tikhonov import build_tikhonov_solver

    n_dce = 30
    t, aif, ct = _synthetic_dce(n_dce)
    solver = build_tikhonov_solver(t, aif, n_lambdas=15)
    with pytest.raises(ValueError, match="at least as long as the AIF"):
        solver(ct[:-4].reshape(-1, 1))
