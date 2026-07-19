"""The shared review-clock: run/stage timers freeze while an interactive review
blocks the pipeline (so elapsed reflects compute time, not tab-open time)."""
import time

from pbrain import _clock
from pbrain._ui import _Cockpit, _Row


def _busy(seconds: float) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


def test_review_pause_accrues_and_flags():
    base = _clock.paused_total()
    assert not _clock.is_paused()
    with _clock.review_pause():
        assert _clock.is_paused()
        _busy(0.03)
    assert not _clock.is_paused()
    assert _clock.paused_total() - base >= 0.03


def test_reentrancy_does_not_double_count():
    with _clock.review_pause():
        before = _clock.paused_total()
        with _clock.review_pause():          # inner is a no-op
            pass
        assert abs(_clock.paused_total() - before) < 0.02


def test_live_elapsed_freezes_during_review():
    r = _Row(1, 1, "kinetic")
    r.state, r.start, r.pause0 = "running", time.perf_counter(), _clock.paused_total()
    _busy(0.02)
    e_before = _Cockpit._live_elapsed(r)
    with _clock.review_pause():
        _busy(0.05)
        e_during = _Cockpit._live_elapsed(r)
    assert abs(e_during - e_before) < 0.03     # timer did not advance while parked
    _busy(0.02)
    assert _Cockpit._live_elapsed(r) > e_during - 0.01   # resumes afterwards
