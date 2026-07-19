"""Integration guard for the --mode verify/manual wiring.

The per-checkpoint tests exercise the payload/apply functions headlessly; these two
tests instead pin the *wiring* in ``stages/_builtin.py`` — the gap where someone
could delete or fire-and-forget a checkpoint call and the rest of the suite would
stay green:

* a **source contract** (AST): each checkpoint is actually called from the stage
  code, and the ones that RETURN a value to use (AIF / model / tissue) have that
  return captured, not dropped;
* a **behavioural guard**: every checkpoint entry point is a no-op in ``auto`` mode
  — none of them open a browser — so the ``active()`` gate can't silently regress.
"""
import ast
import inspect

import numpy as np
import pytest

from pbrain import _checkpoints as C
from pbrain.aif.base import InputFunction
from pbrain.core.pipeline import CheckpointAbort   # noqa: F401 (imported for parity/use)
from pbrain.stages import _builtin

_ENTRYPOINTS = {"aif_checkpoint", "baseline_checkpoint", "model_checkpoint", "tissue_checkpoint"}
# these hand back a value the stage must use downstream (vs baseline, which returns a frame)
_MUST_CAPTURE = {"aif_checkpoint", "model_checkpoint", "tissue_checkpoint"}


def _fn_name(call: ast.Call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def test_builtin_wires_every_checkpoint():
    tree = ast.parse(inspect.getsource(_builtin))
    called, captured = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _fn_name(node) in _ENTRYPOINTS:
            called.add(_fn_name(node))
        if isinstance(node, ast.Assign):                 # value (or anything inside it) assigned
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call) and _fn_name(sub) in _ENTRYPOINTS:
                    captured.add(_fn_name(sub))
    assert _ENTRYPOINTS <= called, f"stage code stopped calling {_ENTRYPOINTS - called}"
    assert _MUST_CAPTURE <= captured, f"checkpoint result dropped for {_MUST_CAPTURE - captured}"


def test_all_entrypoints_are_noops_in_auto_mode(monkeypatch):
    import pbrain._webreview as WR

    def _boom(*a, **k):
        raise AssertionError("a checkpoint opened the browser in auto mode")
    monkeypatch.setattr(WR, "review", _boom)

    class Auto:
        mode = "auto"
        subject_id = "s"
    cfg = Auto()

    X, Y, Z, T = 8, 8, 3, 12
    t = np.linspace(0, 40, T)
    ct = np.random.default_rng(0).random((X, Y, Z, T)).astype(np.float32)
    dce = ct * 100 + 50
    mask = np.zeros((X, Y, Z), bool)
    mask[3:5, 3:5, 1] = True
    ifn = InputFunction(c_a=ct[3, 4, 1, :], t_s=t, mask=mask, source="sss", meta={})

    # each entry point must return its input unchanged and NOT touch the browser
    assert C.aif_checkpoint(cfg, ifn, dce, ct, t) is ifn
    assert C.baseline_checkpoint(cfg, dce) is None
    parc = np.zeros((X, Y, Z), np.int16)
    parc[3:5, 3:5, 1] = 2
    assert C.tissue_checkpoint(cfg, parc, dce[..., 0], {2: "wm"}) is parc

    from pbrain.models import CurveInputs
    from pbrain.models.patlak import PatlakModel
    model = PatlakModel()
    inp = CurveInputs(c_tissue=ct.reshape(-1, T)[:10].T, c_input=ifn.c_a, t_s=t,
                      mask=np.ones(10, bool))
    res = model.fit(inp)
    assert C.model_checkpoint(cfg, "patlak", model, inp, res) is res
