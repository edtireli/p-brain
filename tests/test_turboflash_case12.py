import math

import numpy as np


def _baseline_correct_case12(c: np.ndarray, baseline_frames: int) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    n = int(baseline_frames)
    if n <= 0:
        return c
    out = c.copy()
    if out.ndim >= 1 and out.shape[-1] >= n:
        # MATLAB/validator convention: subtract mean over frames 3..baseline (1-based)
        # i.e. Python indices [2:n].
        baseline = np.nanmean(out[..., 2:n], axis=-1, keepdims=True)
        out = out - baseline
        out[..., :n] = 0.0
    return out


def _expected_turboflash_validated(*, s, t1_ms, td_ms, r1, m0, flip_angle_deg, baseline_frames):
    s = np.asarray(s, dtype=float)
    td_s = float(td_ms) / 1000.0
    r1_s = float(r1) / 1000.0
    t1_s = float(t1_ms) / 1000.0
    r1_pre = 1.0 / t1_s

    sin_th = math.sin(math.radians(float(flip_angle_deg)))
    denom = float(m0) * sin_th
    c_raw = (-1.0 / r1_s) * ((1.0 / td_s) * np.log(1.0 - (s / denom)) + r1_pre)
    return _baseline_correct_case12(c_raw, baseline_frames)


def test_turboflash_validated_matches_formula(monkeypatch):
    from utils.plotting import turboflash
    from utils import settings

    # Keep ratio < 1 for all samples to avoid clamp-related differences.
    s = np.array([200, 201, 199, 200, 202, 250, 400, 500, 450, 300], dtype=float)
    t1_ms = 1500.0
    td_ms = 120.0
    r1 = 4000.0
    m0 = 5000.0
    fa = 10.0
    baseline_frames = 5

    monkeypatch.setattr(settings, "TURBOFLASH_BASELINE_FRAMES", baseline_frames)

    got = turboflash(s, t1_ms, TD=td_ms, r1=r1, m0=m0, flip_angle_deg=fa, ctc_model="turboflash")
    exp = _expected_turboflash_validated(
        s=s,
        t1_ms=t1_ms,
        td_ms=td_ms,
        r1=r1,
        m0=m0,
        flip_angle_deg=fa,
        baseline_frames=baseline_frames,
    )

    assert np.allclose(got, exp, rtol=0, atol=1e-12, equal_nan=True)


def test_advanced_alias_matches_turboflash(monkeypatch):
    from utils.plotting import turboflash
    from utils import settings

    s = np.array([200, 201, 199, 200, 202, 250, 400, 500, 450, 300], dtype=float)
    t1_ms = 1500.0
    td_ms = 120.0
    r1 = 4000.0
    m0 = 5000.0
    fa = 10.0
    baseline_frames = 5

    monkeypatch.setattr(settings, "TURBOFLASH_BASELINE_FRAMES", baseline_frames)

    got_advanced = turboflash(
        s, t1_ms, TD=td_ms, r1=r1, m0=m0, flip_angle_deg=fa, ctc_model="advanced"
    )
    got_turbo = turboflash(
        s, t1_ms, TD=td_ms, r1=r1, m0=m0, flip_angle_deg=fa, ctc_model="turboflash"
    )

    assert np.allclose(got_advanced, got_turbo, rtol=0, atol=1e-12, equal_nan=True)
