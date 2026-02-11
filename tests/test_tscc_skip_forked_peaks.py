import numpy as np


def _forked_curve(n: int = 120, *, peak: float = 3.0, dip_frac: float = 0.10) -> np.ndarray:
    c = np.zeros(n, dtype=float)
    c[40:55] = np.linspace(0.0, 0.9 * peak, 15)
    # Two near-maximum local maxima separated by a dip.
    c[55] = peak
    c[56] = peak * (1.0 - dip_frac)
    c[57] = peak * 0.99
    c[58:80] = np.linspace(0.95 * peak, 0.25 * peak, 22)
    return c


def _plateau_curve(n: int = 120, *, peak: float = 3.0) -> np.ndarray:
    c = np.zeros(n, dtype=float)
    c[40:50] = np.linspace(0.0, 0.9 * peak, 10)
    c[50:54] = peak  # plateau top
    c[54:80] = np.linspace(0.95 * peak, 0.25 * peak, 26)
    return c


def _double_bolus_curve(n: int = 200, *, p1: float = 3.0, p2: float = 2.8) -> np.ndarray:
    c = np.zeros(n, dtype=float)
    c[35:55] = np.linspace(0.0, p1, 20)
    c[55:80] = np.linspace(p1, 0.4, 25)
    c[110:130] = np.linspace(0.0, p2, 20)
    c[130:160] = np.linspace(p2, 0.3, 30)
    return c


def test_is_tscc_forked_peak_flags_split_apex(monkeypatch):
    from modules import time_shifting as ts

    # Make the detector permissive for test clarity.
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MIN_PEAK_MM", 0.1, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_NEAR_FRAC", 0.95, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_WINDOW_FRAMES", 3, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MAX_SEPARATION_FRAMES", 3, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MIN_DIP_FRAC", 0.05, raising=False)

    c = _forked_curve(peak=3.2, dip_frac=0.10)
    assert ts._is_tscc_forked_peak(c) is True


def test_is_tscc_forked_peak_does_not_flag_plateau(monkeypatch):
    from modules import time_shifting as ts

    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MIN_PEAK_MM", 0.1, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_NEAR_FRAC", 0.95, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_WINDOW_FRAMES", 3, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MAX_SEPARATION_FRAMES", 3, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MIN_DIP_FRAC", 0.05, raising=False)

    c = _plateau_curve(peak=3.0)
    assert ts._is_tscc_forked_peak(c) is False


def test_is_tscc_forked_peak_does_not_flag_double_bolus(monkeypatch):
    from modules import time_shifting as ts

    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MIN_PEAK_MM", 0.1, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_NEAR_FRAC", 0.95, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_WINDOW_FRAMES", 3, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MAX_SEPARATION_FRAMES", 3, raising=False)
    monkeypatch.setattr(ts.settings, "TSCC_FORKED_PEAK_MIN_DIP_FRAC", 0.05, raising=False)

    c = _double_bolus_curve()
    assert ts._is_tscc_forked_peak(c) is False
