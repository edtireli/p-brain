import numpy as np


def test_compute_ctc_defaults_to_legacy_30deg_when_missing_flip_angle():
    from utils.plotting import compute_CTC

    s = np.array([100.0, 120.0, 140.0], dtype=float)
    t1 = 900.0  # ms

    c_missing = compute_CTC(s, t1, TD=120, r1=4000, m0=1000.0, prints=False, flip_angle_deg=None, ctc_model="saturation")
    c_30 = compute_CTC(s, t1, TD=120, r1=4000, m0=1000.0, prints=False, flip_angle_deg=30.0, ctc_model="saturation")

    assert np.allclose(c_missing, c_30)


def test_compute_ctc_uses_settings_override_when_flip_angle_missing(monkeypatch):
    from utils import settings
    from utils.plotting import compute_CTC

    monkeypatch.setattr(settings, "FLIP_ANGLE_DEG", 20.0)

    s = np.array([100.0, 120.0, 140.0], dtype=float)
    t1 = 900.0  # ms

    c_missing = compute_CTC(s, t1, TD=120, r1=4000, m0=1000.0, prints=False, flip_angle_deg=None, ctc_model="saturation")
    c_20 = compute_CTC(s, t1, TD=120, r1=4000, m0=1000.0, prints=False, flip_angle_deg=20.0, ctc_model="saturation")

    assert np.allclose(c_missing, c_20)
