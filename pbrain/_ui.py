"""p-Brain's terminal cockpit — the teal, spiral-style view of a pipeline run.

Built on ``rich``, but every live element is **TTY-gated**: piped, redirected,
``--quiet`` and cohort-subprocess runs get a no-op reporter and the plain logger,
so a run's stderr stays clean and parseable for downstream tools. The reporter is
the only thing that talks to the terminal during a run; the pipeline core stays
presentation-free and calls it through a small hook.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from pbrain._palette import palette


def _theme() -> Theme:
    """Build the rich theme from the active palette (clay by default). Style names
    (pb.accent / deep / lite) are palette-relative, so switching theme re-tints the
    whole cockpit at once."""
    base, deep, lite = palette()
    return Theme({
        "pb.accent": base,
        "pb.lite": lite,
        "pb.deep": deep,
        "pb.ink": "grey85",
        "pb.mut": "grey58",
        "pb.dim": "grey39",
        "pb.warn": "yellow",
        "pb.fail": "red",
    })


def _grey_theme() -> Theme:
    """The cockpit's warm-up tint: for the first beat the accent styles render grey,
    then the real palette 'warms in'. Same style names, so only the colour changes —
    the layout is identical, no reflow."""
    return Theme({
        "pb.accent": "grey50", "pb.lite": "grey62", "pb.deep": "grey35",
        "pb.ink": "grey70", "pb.mut": "grey46", "pb.dim": "grey30",
        "pb.warn": "grey58", "pb.fail": "grey58",
    })


THEME = _theme()

_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def console(stderr: bool = True) -> Console:
    """A p-Brain-themed console. Defaults to stderr, matching the logger, so a
    user redirecting stdout never captures the cockpit. Rebuilds the theme each
    call so a mid-session theme change is picked up."""
    return Console(stderr=stderr, theme=_theme(), highlight=False, emoji=False)


def _spin_frame() -> str:
    return _SPIN[int(time.perf_counter() * 12) % len(_SPIN)]


def _fmt_elapsed(s: float) -> str:
    if s < 60:
        return f"{s:0.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s"


def _clip(text: str, n: int) -> str:
    """Trim to ~n chars on a word boundary, adding an ellipsis when cut. Keeps the
    front (the informative part); the full line still shows in the activity feed."""
    s = " ".join(text.split())
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0] or s[:n]
    return cut.rstrip(" ,;·—-") + "…"


@dataclass
class _Row:
    idx: int
    total: int
    name: str
    plugin: str = ""
    state: str = "queued"          # queued | running | pass | warn | fail | skip
    start: float = 0.0
    pause0: float = 0.0            # review-clock reading when this stage started
    elapsed: float = 0.0
    detail: str = ""


_ACTIVITY_LINES = 5   # fixed-height feed → the Live region doesn't jump


class _Cockpit:
    """A live renderable: the stage table plus a rolling activity feed. Rendered on
    every auto-refresh tick, so the running stage's spinner/elapsed animate and the
    latest log lines scroll while the main thread computes."""

    def __init__(self, rows: list[_Row], title: str, activity: deque | None = None,
                 mode_get=None):
        self.rows = rows
        self.title = title
        self.activity = activity if activity is not None else deque(maxlen=_ACTIVITY_LINES)
        self.mode_get = mode_get or (lambda: "auto")
        self.reveal_n: int | None = None   # None = all rows; N = first N (intro cascade)

    @staticmethod
    def _live_elapsed(r: "_Row") -> float:
        """Seconds this stage has been *computing* — wall-time since it started,
        minus any spent parked in an interactive review (so the timer freezes
        while a checkpoint tab is open, then resumes)."""
        from . import _clock
        return max(0.0, time.perf_counter() - r.start - (_clock.paused_total() - r.pause0))

    def _table(self):
        t = Table.grid(padding=(0, 1))
        t.add_column(justify="right", no_wrap=True)   # mark
        t.add_column(justify="right", no_wrap=True)   # idx
        t.add_column(no_wrap=True)                     # name
        t.add_column(no_wrap=True)                     # detail / elapsed
        rows = self.rows if self.reveal_n is None else self.rows[:max(0, self.reveal_n)]
        for r in rows:
            idx = f"{r.idx}/{r.total}"
            if r.state == "running":
                el = _fmt_elapsed(self._live_elapsed(r))
                tail = f"{el} · {r.plugin}" if r.plugin else el
                if r.detail:
                    tail += f" · {_clip(r.detail, 34)}"
                t.add_row(Text(_spin_frame(), style="pb.accent"),
                          Text(idx, style="pb.mut"),
                          Text(r.name, style="pb.ink"),
                          Text(tail, style="pb.lite"))
            elif r.state == "queued":
                t.add_row(Text("·", style="pb.dim"), Text(idx, style="pb.dim"),
                          Text(r.name, style="pb.dim"), Text("queued", style="pb.dim"))
            elif r.state == "skip":
                t.add_row(Text("·", style="pb.dim"), Text(idx, style="pb.dim"),
                          Text(r.name, style="pb.dim"), Text("cached", style="pb.dim"))
            else:
                mark = {"pass": ("✓", "pb.accent"), "warn": ("⚠", "pb.warn"),
                        "fail": ("✗", "pb.fail")}[r.state]
                tail = f"{_fmt_elapsed(r.elapsed)}"
                if r.plugin:
                    tail += f"   {r.plugin}"
                if r.state in ("warn", "fail") and r.detail:
                    tail += f"   {_clip(r.detail, 40)}"
                style = "pb.warn" if r.state == "warn" else ("pb.fail" if r.state == "fail" else "pb.mut")
                t.add_row(Text(mark[0], style=mark[1]), Text(idx, style="pb.mut"),
                          Text(r.name, style="pb.ink"), Text(tail, style=style))
        return t

    def _feed(self):
        lines = list(self.activity)[-_ACTIVITY_LINES:]
        lines = [""] * (_ACTIVITY_LINES - len(lines)) + lines   # pad to fixed height
        body = Text("\n".join("› " + ln if ln else "" for ln in lines), style="pb.dim")
        return Panel(body, title="[pb.accent]activity[/]", title_align="left",
                     border_style="pb.deep", padding=(0, 1), height=_ACTIVITY_LINES + 2)

    def _status(self):
        """The single spiral-style status line at the very bottom: spinner ·
        p-Brain · current stage · elapsed · latest activity · [mode]."""
        run = next((r for r in self.rows if r.state == "running"), None)
        t = Text("  ")
        t.append(_spin_frame() + " ", style="pb.accent")
        t.append("p-Brain", style="pb.accent")
        if run is not None:
            from . import _clock
            t.append(f" · {run.name}", style="pb.ink")
            t.append(f" · {_fmt_elapsed(self._live_elapsed(run))}", style="pb.mut")
            if _clock.is_paused():
                t.append(" · verifying", style="pb.warn")   # timer frozen: awaiting review
            else:
                tail = run.detail or (list(self.activity)[-1] if self.activity else "")
                tail = _clip(tail.split(": ", 1)[-1], 24)
                if tail:
                    t.append(f" · {tail}", style="pb.lite")
        else:
            done = sum(1 for r in self.rows if r.state in ("pass", "warn", "fail", "skip"))
            t.append(f" · {done}/{len(self.rows)} stages", style="pb.mut")
        t.append(f"   {self.mode_get()}", style="pb.deep")
        t.append(" ⇥", style="pb.dim")
        return t

    def __rich__(self):
        stages = Panel(self._table(), title=f"[pb.accent]{self.title}[/]", title_align="left",
                       border_style="pb.deep", padding=(0, 1))
        # activity (top) → plan box (middle) → status line (bottom)
        return Group(self._feed(), stages, self._status())


class _KeyWatcher:
    """Non-blocking stdin reader so ⇥ can cycle the run mode mid-flight (like
    spiral / claude-code). cbreak lets keys arrive without Enter; the terminal is
    restored on stop. No-op unless stdin is a real TTY."""

    def __init__(self, on_key):
        self.on_key = on_key
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._saved = None

    def start(self) -> "_KeyWatcher":
        import sys
        try:
            import termios
            import tty
            if not sys.stdin.isatty():
                return self
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._saved = None
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        import select
        import sys
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.on_key(ch)
            except Exception:
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
        if self._saved is not None and self._fd is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass


class _LogTap(logging.Handler):
    """Feeds pbrain log records into the cockpit — appends to the rolling activity
    feed and sets the running stage's detail — so the run is transparent without
    the logger scribbling over the live region."""

    def __init__(self, reporter: "RunReporter"):
        super().__init__()
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage().strip()
        except Exception:
            return
        if not msg:
            return
        self.reporter.activity.append(msg[:80])
        # Drop the 'stage: ' prefix — the stage already has its own column — and
        # keep the front of the line (render sites clip to their width). The full
        # message stays in the activity feed above.
        detail = msg.split(": ", 1)[-1].strip()
        for r in self.reporter.rows:
            if r.state == "running":
                r.detail = detail
                break


class RunReporter:
    """Live cockpit for a single ``pbrain run``. Context manager: on enter it starts
    the ``Live`` region and taps the ``pbrain`` logger into the rolling activity feed;
    on exit it restores the logger and leaves the final cockpit in the scrollback."""

    def __init__(self, con: Console, header: list[tuple[str, str]], title: str = "analysis",
                 mode: str = "auto"):
        self.con = con
        self.header = header
        self.title = title
        self.rows: list[_Row] = []
        self.activity: deque = deque(maxlen=_ACTIVITY_LINES)
        self._mode = mode if mode in ("auto", "verify", "manual") else "auto"
        self._modes = ["auto", "verify", "manual"]
        self._cockpit: _Cockpit | None = None
        self._live = None
        self._saved_handlers: list[logging.Handler] = []
        self._tap: logging.Handler | None = None
        self._watcher = None
        self._animate = False     # set from the console's TTY-ness on enter
        self._greyed = False      # is the warm-up grey theme currently pushed?
        self._aborted = False     # user rejected an interactive checkpoint

    @property
    def mode(self) -> str:
        return self._mode

    def cycle_mode(self) -> None:
        self._mode = self._modes[(self._modes.index(self._mode) + 1) % len(self._modes)]

    def _new_cockpit(self) -> "_Cockpit":
        return _Cockpit(self.rows, self.title, self.activity, mode_get=lambda: self._mode)

    def _on_key(self, ch: str) -> None:
        """⇥ cycles the interactive mode (auto → verify → manual), shown live in
        the status line; the checkpoints in each stage read the current mode."""
        if ch == "\t":
            self.cycle_mode()

    def aborted(self) -> None:
        """Flag the run as stopped by the user at an interactive checkpoint; the
        closing summary then reads 'run stopped' instead of 'run complete'."""
        self._aborted = True

    # -- pipeline hook surface (called by Pipeline.run) ----------------------
    def plan(self, stages: list[tuple[str, str]]) -> None:
        n = len(stages)
        self.rows = [_Row(i + 1, n, name, plugin) for i, (name, plugin) in enumerate(stages)]
        self._cockpit = self._new_cockpit()
        if self._live is None:
            return
        if self._animate and self.rows:
            # cascade the stage rows in one at a time, still under the grey warm-up
            # tint, then 'warm' the whole cockpit to the palette colour in one beat.
            for k in range(1, len(self.rows) + 1):
                self._cockpit.reveal_n = k
                self._live.update(self._cockpit)
                time.sleep(0.05)
            time.sleep(0.08)
            self._cockpit.reveal_n = None
            self._warm()
        self._live.update(self._cockpit)

    def _find(self, name: str) -> _Row | None:
        return next((r for r in self.rows if r.name == name), None)

    def start(self, name: str) -> None:
        r = self._find(name)
        if r:
            from . import _clock
            r.state, r.start = "running", time.perf_counter()
            r.pause0 = _clock.paused_total()

    def skip(self, name: str) -> None:
        r = self._find(name)
        if r:
            r.state = "skip"

    def end(self, name: str, status: str, elapsed: float, detail: str = "") -> None:
        r = self._find(name)
        if r:
            r.state = {"pass": "pass", "warn": "warn", "fail": "fail"}.get(status, "pass")
            r.elapsed, r.detail = elapsed, detail

    def detail(self, name: str, text: str) -> None:
        r = self._find(name)
        if r:
            r.detail = text.split(": ", 1)[-1].strip()

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "RunReporter":
        from rich.live import Live
        self._animate = bool(getattr(self.con, "is_terminal", False))
        # header lines cascade in above the live region (staggered, not all at once)
        for label, value in self.header:
            self.con.print(f"  [pb.accent]▸ {label}[/]  [pb.mut]{value}[/]")
            if self._animate:
                time.sleep(0.055)
        self.con.print()
        self._cockpit = self._new_cockpit()
        if self._animate:
            # start the cockpit under the grey warm-up tint; plan() warms it in
            self.con.push_theme(_grey_theme())
            self._greyed = True
        self._live = Live(self._cockpit, console=self.con, refresh_per_second=12,
                          transient=False)
        self._live.start()
        self._route_logs()
        self._watcher = _KeyWatcher(self._on_key).start()
        return self

    def _warm(self) -> None:
        """Pop the grey warm-up tint so the cockpit settles into the palette colour."""
        if self._greyed:
            try:
                self.con.pop_theme()
            except Exception:
                pass
            self._greyed = False

    def __exit__(self, *exc) -> None:
        if self._watcher is not None:
            self._watcher.stop()
        self._warm()   # safety: never leave the grey warm-up tint pushed
        self._restore_logs()
        if self._live is not None:
            self._live.stop()

    def _route_logs(self) -> None:
        """Redirect the pbrain logger's console output into the cockpit: a tap
        appends each record to the rolling activity feed and sets the running
        stage's detail line, instead of scribbling over the live region. File
        handlers (from --log-file) stay — the full record still lands on disk."""
        logger = logging.getLogger("pbrain")
        self._saved_handlers = list(logger.handlers)
        level = min((h.level for h in logger.handlers if not isinstance(h, logging.FileHandler)),
                    default=logging.INFO)
        self._tap = _LogTap(self)
        self._tap.setLevel(level)
        for h in list(logger.handlers):
            if not isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
        logger.addHandler(self._tap)

    def _restore_logs(self) -> None:
        logger = logging.getLogger("pbrain")
        for h in list(logger.handlers):
            logger.removeHandler(h)
        for h in self._saved_handlers:
            logger.addHandler(h)

    # -- summary -------------------------------------------------------------
    def summary(self, seconds: float, extras: list[tuple[str, str]], out_dir: str,
                ok: bool = True) -> None:
        """Print the closing card. Ran/cached counts and the QC tally are read
        straight off the stage rows, so the summary can never disagree with what
        the cockpit showed."""
        ran = sum(1 for r in self.rows if r.state in ("pass", "warn", "fail"))
        cached = sum(1 for r in self.rows if r.state == "skip")
        qc: dict[str, int] = {}
        for r in self.rows:
            if r.state in ("pass", "warn", "fail"):
                qc[r.state] = qc.get(r.state, 0) + 1
        qc_bits = " · ".join(f"{v} {k}" for k, v in qc.items()) or "—"
        if self._aborted:
            head, border = "run stopped", "pb.warn"
        elif ok:
            head, border = "run complete", "pb.accent"
        else:
            head, border = "run failed", "pb.fail"
        lines = [Text(f"{ran} ran · {cached} cached · QC {qc_bits} · {_fmt_elapsed(seconds)}",
                      style="pb.ink")]
        for label, val in extras:
            lines.append(Text(f"{label:7}{val}", style="pb.mut"))
        lines.append(Text(f"{'→':7}{out_dir}", style="pb.mut"))
        self.con.print()
        self.con.print(Panel(Group(*lines), title=f"[pb.accent]{head}[/]", title_align="left",
                             border_style=border, padding=(0, 1)))


class NullReporter:
    """No-op reporter for non-TTY / quiet runs. The plain logger carries the run."""

    def plan(self, stages): ...
    def start(self, name): ...
    def skip(self, name): ...
    def end(self, name, status, elapsed, detail=""): ...
    def detail(self, name, text): ...
    def aborted(self): ...
    def summary(self, *a, **k): ...

    def __enter__(self): return self

    def __exit__(self, *exc): ...


def make_run_reporter(header: list[tuple[str, str]], *, enabled: bool | None = None,
                      title: str = "analysis", mode: str = "auto"):
    """Return a live ``RunReporter`` on an interactive terminal, else a
    ``NullReporter``. ``enabled`` forces the choice (used by tests / --quiet)."""
    con = console(stderr=True)
    if enabled is None:
        enabled = con.is_terminal
    return RunReporter(con, header, title, mode=mode) if enabled else NullReporter()


class CohortReporter:
    """Live progress for ``pbrain cohort``: a bar over subjects, a ✓/✗ line per
    subject as it finishes, and a closing tally. Per-subject runs execute in
    captured subprocesses (non-TTY), so their own cockpits self-suppress and only
    this cohort-level view is shown."""

    def __init__(self, con: Console, total: int, header: list[tuple[str, str]]):
        self.con = con
        self.total = total
        self.header = header
        self.ok = 0
        self.bad = 0
        self._progress = None
        self._task = None

    def __enter__(self) -> "CohortReporter":
        from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                                   MofNCompleteColumn, TimeElapsedColumn)
        for label, value in self.header:
            self.con.print(f"  [pb.accent]▸ {label}[/]  [pb.mut]{value}[/]")
        self.con.print()
        self._progress = Progress(
            SpinnerColumn(style="pb.accent"),
            TextColumn("[pb.mut]{task.description}"),
            BarColumn(complete_style="pb.accent", finished_style="pb.accent"),
            MofNCompleteColumn(), TimeElapsedColumn(),
            console=self.con,
        )
        self._task = self._progress.add_task("0 ok · 0 failed", total=self.total)
        self._progress.start()
        return self

    def done(self, name: str, status: str, msg: str = "") -> None:
        if status == "ok":
            self.ok += 1
            self._progress.console.print(f"  [pb.accent]✓[/] {name}")
        else:
            self.bad += 1
            self._progress.console.print(f"  [pb.fail]✗[/] {name}  [pb.dim]{msg}[/]")
        self._progress.update(self._task, advance=1,
                              description=f"{self.ok} ok · {self.bad} failed")

    def __exit__(self, *exc) -> None:
        if self._progress is not None:
            self._progress.stop()

    def summary(self, seconds: float) -> None:
        ok_txt = f"{self.ok} ok" if not self.bad else f"[pb.warn]{self.ok} ok · {self.bad} failed[/]"
        self.con.print()
        self.con.print(Panel(Text.from_markup(f"{self.total} subjects · {ok_txt} · {_fmt_elapsed(seconds)}",
                                              style="pb.ink"),
                             title="[pb.accent]cohort complete[/]", title_align="left",
                             border_style=("pb.accent" if not self.bad else "pb.warn"), padding=(0, 1)))


class NullCohortReporter:
    def __enter__(self): return self
    def __exit__(self, *exc): ...
    def done(self, *a, **k): ...
    def summary(self, *a, **k): ...


def make_cohort_reporter(total: int, header: list[tuple[str, str]], *, enabled: bool | None = None):
    con = console(stderr=True)
    if enabled is None:
        enabled = con.is_terminal
    return CohortReporter(con, total, header) if enabled else NullCohortReporter()


def show_plan(header: list[tuple[str, str]], rows: list[tuple[str, str]]) -> None:
    """Print the resolved pipeline without running it (``pbrain plan`` /
    ``run --dry-run``). Static output → stdout; rich drops colour when piped."""
    con = console(stderr=False)
    for label, value in header:
        con.print(f"  [pb.accent]▸ {label}[/]  [pb.mut]{value}[/]")
    con.print()
    t = Table.grid(padding=(0, 1))
    t.add_column(justify="right", no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_column(no_wrap=True)
    n = len(rows)
    for i, (name, plugin) in enumerate(rows, 1):
        t.add_row(Text(f"{i}/{n}", style="pb.dim"),
                  Text(name, style="pb.ink"),
                  Text(plugin or "—", style="pb.lite"))
    con.print(Panel(t, title="[pb.accent]pipeline plan[/]", title_align="left",
                    border_style="pb.deep", padding=(0, 1)))
    con.print("  [pb.dim]dry run — nothing computed. drop --dry-run to execute.[/]")
