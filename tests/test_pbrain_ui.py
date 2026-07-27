"""The teal cockpit is presentation, but two things about it are contractual and
must not regress: (1) it is silent on a non-interactive stream, so redirected /
piped / cohort-subprocess output stays clean; (2) its stage table and summaries
report what actually happened. These tests pin both without needing a real run.
"""
from __future__ import annotations

from rich.console import Console

import pbrain._ui as ui


def _rec() -> Console:
    return Console(record=True, theme=ui.THEME, width=80, force_terminal=True)


def test_reporters_are_null_when_not_a_terminal():
    # the clean-pipe contract: no live cockpit unless attached to a real TTY
    assert isinstance(ui.make_run_reporter([("subject", "x")], enabled=False), ui.NullReporter)
    assert isinstance(ui.make_cohort_reporter(3, [("cohort", "x")], enabled=False),
                      ui.NullCohortReporter)


def test_null_reporters_are_total_noops():
    r = ui.NullReporter()
    with r:
        r.plan([("load", "l")]); r.start("load"); r.detail("load", "x")
        r.skip("load"); r.end("load", "pass", 1.0)
    r.summary(1.0, [("models", "patlak")], "/out")   # must not raise
    c = ui.NullCohortReporter()
    with c:
        c.done("sub", "ok")
    c.summary(1.0)


def test_run_reporter_table_and_summary_reflect_state():
    rep = ui.RunReporter(ui.console(), header=[("subject", "sub-017")])
    rep.plan([("load", "l"), ("t1_m0", "ir"), ("aif", "cnn"), ("kinetic", "patlak")])
    rep.rows[0].state = "pass"; rep.rows[0].elapsed = 1.2
    rep.skip("t1_m0")
    rep.end("aif", "warn", 9.0, detail="3 voxels non-converged")
    cap = _rec(); rep.con = cap
    cap.print(rep._cockpit)
    rep.summary(159.0, [("models", "patlak")], "/out/derivatives")
    txt = cap.export_text()
    # every stage shows, marks + cached + warn detail present, counts are truthful
    for m in ("load", "t1_m0", "aif", "kinetic", "✓", "⚠", "cached", "queued",
              "3 voxels", "run complete", "2 ran", "1 cached", "patlak"):
        assert m in txt, f"missing from cockpit: {m!r}"


def test_cohort_reporter_counts_ok_and_failed():
    cap = _rec()
    rep = ui.CohortReporter(cap, total=3, header=[("cohort", "3 subjects")])
    with rep:
        rep.done("sub-1", "ok")
        rep.done("sub-2", "fail", "AIF failed")
        rep.done("sub-3", "ok")
    rep.summary(42.0)
    txt = cap.export_text()
    assert "sub-1" in txt and "AIF failed" in txt
    assert "2 ok · 1 failed" in txt and "cohort complete" in txt


def test_show_plan_lists_every_stage(capsys):
    ui.show_plan([("subject", "sub-017")],
                 [("load", "bids_like"), ("aif", "cnn_sss_shifted"), ("kinetic", "patlak")])
    out = capsys.readouterr().out
    for m in ("pipeline plan", "load", "aif", "cnn_sss_shifted", "kinetic", "dry run"):
        assert m in out, f"missing from plan: {m!r}"


def test_reporter_context_enters_and_cycles_mode():
    # regression: __enter__ starts the key-watcher + Live region; a missing
    # `import threading` once crashed real runs right here. Also covers ⇥ cycling.
    rep = ui.RunReporter(ui.console(), header=[("subject", "x")], mode="verify")
    with rep:
        rep.plan([("load", "l"), ("aif", "cnn")])
        rep.start("load")
        rep.end("load", "pass", 1.0)
    assert rep.mode == "verify"
    rep.cycle_mode(); assert rep.mode == "manual"
    rep.cycle_mode(); assert rep.mode == "auto"
    w = ui._KeyWatcher(lambda ch: None)   # watcher lifecycle must not raise
    w.start(); w.stop()


def test_pipeline_run_exposes_reporter_hook():
    # the hook is opt-in and keyword-only; default None keeps the core behaviour
    # (proven by the rest of the suite running with no reporter)
    import inspect
    from pbrain.core.pipeline import Pipeline
    params = inspect.signature(Pipeline.run).parameters
    assert "reporter" in params
    assert params["reporter"].default is None


def _run():
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for f in fns:
        try:
            f() if f.__code__.co_argcount == 0 else None
            print(f"  PASS {f.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"  FAIL {f.__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())


def test_banner_creature_switches_to_mouse():
    """`--animal/--profile mouse` swaps the banner art to the mouse. Dimensions are
    per-creature (the neuron is 10x5, the mouse 8x3), so the reveal machinery must
    size itself to whichever art is active rather than to a shared constant."""
    from pbrain import _banner as B
    assert B.creature_from_argv(["run", "x", "--animal", "mouse"]) == "mouse"
    assert B.creature_from_argv(["run", "x", "--animal=mouse"]) == "mouse"
    assert B.creature_from_argv(["run", "x", "--profile", "mouse"]) == "mouse"
    assert B.creature_from_argv(["run", "x", "--profile=mouse"]) == "mouse"
    assert B.creature_from_argv(["run", "x"]) == "human"
    assert B.creature_from_argv(["run", "x", "--animal"]) == "human"
    assert B.creature_from_argv(["run", "x", "--profile", "nope"]) == "human"
    try:
        # each creature's art is internally consistent (all rows the same width)
        for art in (B.NEURON, B.MOUSE):
            assert len({len(r) for r in art}) == 1
        B.set_creature("mouse")
        assert B.art() == B.MOUSE and B.active_creature() == "mouse"
        grid = B._decode()                       # sized to the mouse (8x3), not the neuron
        assert (len(grid), len(grid[0])) == (len(B.MOUSE) * 4, len(B.MOUSE[0]) * 2)
        for name in B._MIX:                      # every reveal ends on the exact art
            tmap = B._reveal_map(name, grid)
            assert B._encode(B._reveal(grid, tmap, 1.0)) == B.MOUSE
        assert B.set_creature("giraffe") == "human"      # unknown never breaks it
    finally:
        B.set_creature("human")
