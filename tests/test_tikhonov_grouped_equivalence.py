import numpy as np
import pytest

from models.tikhonov import build_tikhonov_solver


def _make_synthetic_ct(time_s: np.ndarray, n_vox: int, seed: int = 0) -> np.ndarray:
    """Generate smooth, positive Ct curves with per-voxel variation.

    Shape returned: (n_time, n_vox)
    """
    rng = np.random.default_rng(seed)
    t = time_s.reshape(-1, 1)

    # A mixture of decaying exponentials with small oscillation and noise.
    amps = rng.uniform(0.2, 3.0, size=(1, n_vox))
    tau1 = rng.uniform(4.0, 20.0, size=(1, n_vox))
    tau2 = rng.uniform(25.0, 80.0, size=(1, n_vox))
    w = rng.uniform(0.04, 0.12, size=(1, n_vox))

    ct = amps * (np.exp(-t / tau1) - 0.3 * np.exp(-t / tau2))
    ct += 0.05 * np.sin(w * t) * rng.uniform(0.5, 1.0, size=(1, n_vox))
    ct += rng.normal(0.0, 0.01, size=ct.shape)

    # Clamp to non-negative like real concentration curves after baseline.
    ct = np.maximum(ct, 0.0)
    return ct


@pytest.mark.parametrize("with_offsets", [False, True])
def test_grouped_vs_per_voxel_equivalence(with_offsets: bool) -> None:
    # Small but non-trivial sizes so we hit multiple batches and multiple lambdas.
    n_time = 180
    n_vox = 257

    time_s = np.linspace(0.0, 179.0, n_time, dtype=float)

    # AIF: positive, peaked, then decays.
    aif = np.exp(-0.03 * time_s) * (1.0 - np.exp(-0.4 * time_s))

    solve_ct = build_tikhonov_solver(
        time_s=time_s,
        ca=aif,
        lambda_candidates=np.logspace(-5, 2, 30),
    )

    Ct = _make_synthetic_ct(time_s, n_vox=n_vox, seed=123)

    offsets_s = None
    if with_offsets:
        # Some curves offset slightly, some not; integer seconds typical.
        rng = np.random.default_rng(321)
        offsets_s = rng.integers(-3, 4, size=(n_vox,)).astype(float)

    res_grouped = solve_ct(Ct, offsets_s=offsets_s, implementation="grouped")
    res_per_voxel = solve_ct(Ct, offsets_s=offsets_s, implementation="per_voxel")

    # If the math is identical and the solve order doesn't change numerics,
    # these should match to numerical precision. Use tight tolerances to be
    # robust across BLAS/LAPACK variations.
    for attr in ["cbf_ml_per_100g_min", "cbv_vd", "mtt_s", "cth_s", "lambda_opt"]:
        a = getattr(res_grouped, attr)
        b = getattr(res_per_voxel, attr)
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-9)

    # Also ensure the chosen lambda indices match exactly.
    np.testing.assert_array_equal(res_grouped.lambda_opt, res_per_voxel.lambda_opt)
