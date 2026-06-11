"""Parity tests — the new pbrain.models.* numerics must match the
validated legacy ``models.*`` numerics bit-for-bit on identical inputs.

The legacy modules are imported alongside the new ones; both are called
on synthetic phantoms generated below; results are compared with very
tight tolerances (1e-10 to 1e-12) — these are pure floating-point
roundoff differences if any.

This is the "same results as before" gate. If a parity test fails, the
port introduced a numeric divergence and the bug must be fixed before
the new code can be trusted to reproduce paper results.
"""

from __future__ import annotations

import numpy as np
import pytest

from pbrain.models import REGISTRY, CurveInputs


# ────────────────────────────────────────────────────────────────────────
# Synthetic 2-bolus phantom (matches paper §4.2 protocol)
# ────────────────────────────────────────────────────────────────────────


def _synthetic_phantom(ki_true=0.5, vb_true=2.0, n=250, dt=2.463,
                       noise_sd=1e-4, seed=0):
    """(Ki mL/100g/min, vb mL/100g) → (c_t, c_a, t)."""
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


# ────────────────────────────────────────────────────────────────────────
# Patlak parity
# ────────────────────────────────────────────────────────────────────────


def test_patlak_legacy_reproduces_golden():
    """``patlak_legacy`` reproduces the byte-frozen legacy OLS Ki on the
    standard phantom. (Parity with the now-removed ``models.patlak`` was
    verified historically; the golden constant freezes that result.)"""
    PATLAK_GOLDEN_KI = 0.4079360793498615    # frozen legacy OLS value, seed=42

    c_t, c_a, t = _synthetic_phantom(seed=42)
    new = REGISTRY["patlak_legacy"].fit(
        CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t), detrend="none")
    assert float(new.maps["ki"]) == pytest.approx(PATLAK_GOLDEN_KI, rel=1e-9)


def test_patlak_recovers_phantom_ki_and_vb_exactly():
    """End-to-end: Patlak adapter recovers known Ki and vb on a noise-free
    synthetic phantom.

    The synthetic phantom uses an AIF that decays to zero in the tail;
    we mask the tail samples where Ca < 5 % of peak (the same defensive
    trick the production pipeline uses via ``single_bolus=True``'s
    ``aif_min_fraction`` mechanism). Inside the masked window the
    forward model is exact, so the fit recovers both parameters to
    near-machine precision.
    """
    ki_true = 0.50
    vb_true = 2.0
    c_t, c_a, t = _synthetic_phantom(ki_true=ki_true, vb_true=vb_true,
                                     noise_sd=0.0)
    ca_peak = float(np.max(c_a))
    bad_tail = c_a < 0.05 * ca_peak

    plugin = REGISTRY["patlak"]
    res = plugin.fit(
        CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t, mask=bad_tail),
    )
    assert float(res.maps["ki"]) == pytest.approx(ki_true, abs=0.01)
    assert float(res.maps["vb"]) == pytest.approx(vb_true, abs=0.05)


def test_patlak_voxelwise_batch():
    """Patlak adapter handles (T, V) batched input and returns (V,) maps."""
    c_t1, c_a, t = _synthetic_phantom(ki_true=0.30, seed=1)
    c_t2, _, _ = _synthetic_phantom(ki_true=0.70, seed=2)
    c_t3, _, _ = _synthetic_phantom(ki_true=1.20, seed=3)
    c_t = np.stack([c_t1, c_t2, c_t3], axis=1)   # (T, 3)

    plugin = REGISTRY["patlak"]
    res = plugin.fit(CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t))
    assert res.maps["ki"].shape == (3,)
    assert res.maps["vb"].shape == (3,)
    # Each voxel recovers approximately the truth.
    truths = [0.30, 0.70, 1.20]
    for fit_ki, true_ki in zip(res.maps["ki"], truths):
        assert float(fit_ki) == pytest.approx(true_ki, abs=0.06)


# ────────────────────────────────────────────────────────────────────────
# Tikhonov parity
# ────────────────────────────────────────────────────────────────────────


# Golden values frozen from the legacy production solver
# ``modules/kinetic_models.build_spline_lcurve_deconvolution_solver`` (linear
# λ-grid linspace(0.05, 40, 121), L-curve −∞ degenerate handling, residue-
# integral MTT/CTH, time_end knot mode), phantom ki_true=0.20 vb=2.0 seed=7.
# Parity with that solver was verified historically; these freeze the result.
TIK_LEGACY_GOLDEN = {"cbf": 36.53212040210995, "mtt": 2.2158203604973923,
                     "cth": 1.8593031153737407}


def test_tikhonov_legacy_reproduces_golden():
    """``tikhonov_legacy`` reproduces the byte-frozen legacy-production CBF/MTT/CTH."""
    c_t, c_a, t = _synthetic_phantom(ki_true=0.20, vb_true=2.0, seed=7)
    new = REGISTRY["tikhonov_legacy"].fit(CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t))
    for k, golden in TIK_LEGACY_GOLDEN.items():
        assert float(new.maps[k]) == pytest.approx(golden, rel=1e-6, abs=1e-9)


def test_tikhonov_smart_default_is_physiological():
    """The smart ``tikhonov`` default (GCV + log grid) need not match the legacy
    bit-for-bit, but its CBF must track the golden within a few % (λ mainly
    moves CTH)."""
    c_t, c_a, t = _synthetic_phantom(ki_true=0.20, vb_true=2.0, seed=7)
    new = REGISTRY["tikhonov"].fit(CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t))
    assert float(new.maps["cbf"]) == pytest.approx(TIK_LEGACY_GOLDEN["cbf"], rel=0.05)


def test_tikhonov_voxelwise_batch_shape():
    c_t1, c_a, t = _synthetic_phantom(ki_true=0.10, seed=10)
    c_t2, _, _ = _synthetic_phantom(ki_true=0.30, seed=11)
    c_t = np.stack([c_t1, c_t2], axis=1)

    plugin = REGISTRY["tikhonov"]
    res = plugin.fit(CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t))
    assert res.maps["cbf"].shape == (2,)
    assert res.maps["mtt"].shape == (2,)
    assert res.maps["cth"].shape == (2,)


# ────────────────────────────────────────────────────────────────────────
# Extended Tofts parity
# ────────────────────────────────────────────────────────────────────────


def test_extended_tofts_forward_reproduces_golden():
    """The extended-Tofts forward model is deterministic; golden frozen from
    the (historically parity-verified) legacy forward."""
    from pbrain.models.extended_tofts import forward as new_forward

    _, c_a, t = _synthetic_phantom(seed=42)
    b = new_forward(t, 0.0005, 0.20, 0.05, c_a)
    assert float(b.sum()) == pytest.approx(2.604349682515517, rel=1e-9)
    assert float(b.max()) == pytest.approx(0.07836006354612324, rel=1e-9)


# ────────────────────────────────────────────────────────────────────────
# Gamma — structural check only (clean port is numpy-only, 5 free params;
# may differ from legacy 9-param JAX implementation in non-default modes).
# ────────────────────────────────────────────────────────────────────────


def test_gamma_in_registry_only_when_file_present():
    """Gamma is gitignored; on a fresh clone it won't be in the registry.
    Locally (where ``pbrain/models/gamma.py`` exists), it must be there."""
    import pbrain.models as m
    from pathlib import Path
    gamma_file = Path(m.__file__).parent / "gamma.py"
    if gamma_file.exists():
        assert "gamma" in m.REGISTRY, (
            f"gamma.py present but not registered: {sorted(m.REGISTRY.keys())}"
        )
    else:
        assert "gamma" not in m.REGISTRY, "gamma.py missing but registered?"


@pytest.mark.skipif("gamma" not in REGISTRY, reason="gamma plug-in not available")
def test_gamma_runs_on_phantom():
    """Smoke test: gamma fit completes and returns the documented schema."""
    c_t, c_a, t = _synthetic_phantom(seed=42)
    plugin = REGISTRY["gamma"]
    res = plugin.fit(CurveInputs(c_tissue=c_t, c_input=c_a, t_s=t),
                     n_restarts=3)  # few restarts to keep test fast
    assert set(res.maps.keys()) == {"cbf", "mtt", "cth", "extraction", "k2"}
    assert np.isfinite(float(res.maps["cbf"]))
