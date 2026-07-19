"""Shared review clock.

When ``--mode verify/manual`` opens a browser checkpoint, the pipeline blocks on
the researcher — nothing is computing, but wall-time keeps passing. The stage,
cockpit and run-total timers subtract the seconds spent parked in a review so the
elapsed they show reflects *compute* time, not how long a tab sat open.

One in-flight review at a time (checkpoints are sequential and block the main
thread), so a single module-level accumulator is enough; re-entrancy is ignored.
"""

from __future__ import annotations

import time

_accum: float = 0.0                 # completed review seconds this process
_active_since: float | None = None  # perf_counter at the current review's start


def paused_total() -> float:
    """Total wall-seconds spent parked in interactive reviews so far — including a
    review currently in progress, so a live timer reading this stays frozen while
    the browser tab is open."""
    base = _accum
    if _active_since is not None:
        base += time.perf_counter() - _active_since
    return base


def is_paused() -> bool:
    """True while a review is blocking the run."""
    return _active_since is not None


class review_pause:
    """Context manager wrapping a blocking review call; accrues its wall-time.

    Re-entrant calls are no-ops (the outer ``with`` owns the interval), so nesting
    can't double-count or lose the accumulator."""

    __slots__ = ("_owns",)

    def __enter__(self) -> "review_pause":
        global _active_since
        self._owns = _active_since is None
        if self._owns:
            _active_since = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> bool:
        global _accum, _active_since
        if self._owns and _active_since is not None:
            _accum += time.perf_counter() - _active_since
            _active_since = None
        return False
