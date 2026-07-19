"""Headless coverage of the --mode verify/manual checkpoints (the browser step
is mocked; everything else — payload, coordinate maths, result application,
abort — is exercised directly)."""
import numpy as np
import pytest

from pbrain import _checkpoints as C
from pbrain.aif.base import InputFunction
from pbrain.core.pipeline import CheckpointAbort


class _Cfg:
    mode = "verify"
    subject_id = "sub-01"


def _synth():
    X, Y, Z, T = 12, 10, 4, 20
    t = np.linspace(0, 60, T)
    ct = (np.random.default_rng(0).random((X, Y, Z, T)).astype(np.float32) * 0.1)
    ct[3, 4, 1, :] += np.exp(-((t - 20) ** 2) / 50) * 5      # the clear peak voxel
    dce = ct * 100 + 50
    mask = np.zeros((X, Y, Z), bool)
    mask[2:5, 3:6, 1] = True
    ifn = InputFunction(c_a=ct[3, 4, 1, :], t_s=t, mask=mask, source="sss", meta={})
    return _Cfg(), ifn, dce, ct, t


def test_active_only_in_review_modes():
    assert C.active("auto") is False           # never gates the default
    assert C.active("verify") is False         # pytest stdout is not a TTY


def test_build_payload_shape():
    cfg, ifn, dce, ct, t = _synth()
    p = C.build_aif_payload(cfg, ifn, dce, ct, t)
    assert p["checkpoint"] == "aif" and p["mode"] == "verify"
    assert "sss|max" in p["curves"] and len(p["curves"]["sss|max"]) == len(t)
    # a PNG per slice (so the slider works)
    pngs = p["slice"]["pngs"]
    assert len(pngs) == ifn.mask.shape[2] and pngs["1"].startswith("data:image/png;base64,")
    roi = p["rois"]["sss"]
    assert roi["n"] >= 1                                       # grown vessel region
    # peak voxel (3,4) → rotated + L-R-flipped display (u,v) = [(X-1-xi)/X, (Y-1-yi)/Y]
    assert roi["max"] == pytest.approx([8 / 12, 5 / 10])
    assert roi["max_slice"] == 1 and roi["slices"]["1"]["png"].startswith("data:image")


def test_coordinate_roundtrip():
    X, Y = 12, 10
    for xi in range(X):
        for yi in range(Y):
            u, v = C._vox_to_disp(xi, yi, X, Y)
            assert C._disp_to_vox(u, v, X, Y) == (xi, yi)


def test_reject_aborts():
    cfg, ifn, _, ct, _ = _synth()
    with pytest.raises(CheckpointAbort):
        C.apply_aif_result(cfg, ifn, ct, {"accepted": False})
    with pytest.raises(CheckpointAbort):
        C.apply_aif_result(cfg, ifn, ct, None)


def test_confirm_leaves_curve_unchanged():
    cfg, ifn, _, ct, _ = _synth()
    out = C.apply_aif_result(cfg, ifn, ct, {"accepted": True})
    assert np.array_equal(out.c_a, ifn.c_a)
    assert out.meta["review"]["mode"] == "verify"


def test_moved_max_reextracts_from_that_voxel():
    cfg, ifn, _, ct, _ = _synth()
    res = {"accepted": True, "slice": 1, "max": [5 / 12, 4 / 10],  # flipped-disp → voxel (6,5)
           "vessel": "sss", "stat": "max"}
    out = C.apply_aif_result(cfg, ifn, ct, res)
    assert np.allclose(out.c_a, ct[6, 5, 1, :])
    assert int(out.mask.sum()) == 1
    assert out.meta["review"]["max_voxel"] == [6, 5, 1]


def test_drawn_polygon_extracts_from_roi():
    cfg, ifn, _, ct, _ = _synth()
    # rectangle over xi∈[2,5], yi∈[3,6] on slice 1, in flipped-display [u,v] coords
    poly = [[9 / 12, 6 / 10], [6 / 12, 6 / 10], [6 / 12, 3 / 10], [9 / 12, 3 / 10]]
    out = C.apply_aif_result(cfg, ifn, ct, {"accepted": True, "slice": 1, "polygon": poly})
    assert out.source == "custom" and int(out.mask.sum()) >= 1      # a drawn ROI is "custom"
    assert out.meta["review"]["vessel"] == "custom"
    assert np.allclose(out.c_a, ct[3, 4, 1, :])     # max-peak voxel inside the ROI


# ── modular per-model verification (hybrid: spec or figure) ──────────────────

def test_patlak_review_returns_spec():
    from pbrain.models.patlak import PatlakModel
    from pbrain.models import CurveInputs
    T, V = 120, 40
    t = np.linspace(0, 250, T)
    def bolus(t0):
        x = np.maximum((t - t0) / 10, 0)
        return (x ** 3) * np.exp(3 * (1 - x))
    ca = bolus(20) * 1.5 + bolus(120) * 1.0
    integ = np.concatenate([[0], np.cumsum(0.5 * (ca[1:] + ca[:-1]) * np.diff(t))])
    c_t = np.stack([0.02 * ca + 0.0002 * (v + 1) * integ for v in range(V)], axis=1)
    inputs = CurveInputs(c_tissue=c_t, c_input=ca, t_s=t, mask=np.ones(V, bool))
    m = PatlakModel()
    spec = m.review(inputs, m.fit(inputs))
    assert spec["title"].startswith("Patlak")
    kinds = [p["kind"] for p in spec["panels"]]
    assert "scatter" in kinds and "values" in kinds
    sc = next(p for p in spec["panels"] if p["kind"] == "scatter")
    used = next(s for s in sc["series"] if s.get("role") != "muted")
    muted = next(s for s in sc["series"] if s.get("role") == "muted")
    assert len(used["x"]) >= 2 and sc["lines"]                # fitted data + fit line
    assert muted["x"]                                          # early points shown, separate
    assert "ylim" in sc and "xlim" in sc
    uy = used["y"]
    span = (max(uy) - min(uy)) or abs(max(uy)) or 1.0
    # y-window is tied to the FITTED points (hugs them) — so early high-leverage
    # outliers in the muted series can't have blown up the scale.
    assert sc["ylim"][0] <= min(uy) and sc["ylim"][1] >= max(uy)
    assert sc["ylim"][1] - max(uy) <= 0.5 * span
    assert min(uy) - sc["ylim"][0] <= 0.5 * span
    # x keeps the whole trajectory: the muted (early) points extend the range left.
    allx = muted["x"] + used["x"]
    assert sc["xlim"][0] <= min(allx) + 1e-9 and sc["xlim"][1] >= max(allx) - 1e-9


def test_spec_to_payload_dict_and_figure():
    from pbrain import _checkpoints as C
    class Cfg: mode = "verify"; subject_id = "s"
    spec = {"title": "T", "panels": [{"kind": "scatter", "series": [{"x": [1, 2], "y": [3.0, np.nan]}]},
                                      {"kind": "values", "items": {"a": "b"}}]}
    pay = C.spec_to_payload(spec, checkpoint="model", title="patlak", config=Cfg())
    assert pay["checkpoint"] == "model" and pay["title"] == "T"
    assert pay["panels"][0]["series"][0]["y"] == [3.0, None]      # NaN → JSON null
    import matplotlib; matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig = plt.figure(); plt.plot([1, 2, 3])
    pay2 = C.spec_to_payload(fig, checkpoint="model", title="custom", config=Cfg())
    assert pay2["figure"].startswith("data:image/png;base64,")


def test_review_checkpoint_proceeds_and_aborts(monkeypatch):
    from pbrain import _checkpoints as C
    import pbrain._webreview as WR
    from pbrain.core.pipeline import CheckpointAbort
    monkeypatch.setattr(C, "active", lambda m: True)
    class Cfg: mode = "verify"; subject_id = "s"
    class M:
        def review(self, a, b): return {"panels": [{"kind": "values", "items": {"k": "v"}}]}
    got = {}
    monkeypatch.setattr(WR, "review", lambda payload, **k: got.update(p=payload) or {"accepted": True})
    C.review_checkpoint(Cfg(), M(), checkpoint="model", title="patlak", args=(1, 2))
    assert got["p"]["checkpoint"] == "model" and got["p"]["title"] == "patlak"
    monkeypatch.setattr(WR, "review", lambda payload, **k: {"accepted": False})
    with pytest.raises(CheckpointAbort):
        C.review_checkpoint(Cfg(), M(), checkpoint="model", title="x", args=(1, 2))


def test_review_checkpoint_no_review_or_none_is_noop(monkeypatch):
    from pbrain import _checkpoints as C
    monkeypatch.setattr(C, "active", lambda m: True)
    class Cfg: mode = "verify"; subject_id = "s"
    C.review_checkpoint(Cfg(), object(), checkpoint="model", title="x")     # no review()
    class M:
        def review(self, *a): return None                                    # opts out
    C.review_checkpoint(Cfg(), M(), checkpoint="model", title="x", args=())


def test_grow_vessel_recovers_full_vessel_without_leaking():
    """Hysteresis segmentation captures the whole bright vessel (core + the dimmer
    margin the CNN blob under-covers) yet never bleeds into a separate bright blob."""
    rng = np.random.default_rng(0)
    X, Y, Z, T = 44, 44, 5, 10
    inten = rng.random((X, Y, Z)) * 0.3                      # background tissue
    tt = np.linspace(0, 9, T)
    yy, xx = np.mgrid[0:X, 0:Y]
    for z in range(1, 4):                                     # a tilted tube z=1..3
        r = np.sqrt((xx - (18 + z)) ** 2 + (yy - (20 - z)) ** 2)
        inten[:, :, z] = np.where(r <= 1.5, 5.0, np.where(r <= 3.0, 2.2, inten[:, :, z]))
    rb = np.sqrt((xx - 34) ** 2 + (yy - 34) ** 2)            # a SEPARATE bright blob
    inten[:, :, 2] = np.where(rb <= 2.5, 4.0, inten[:, :, 2])
    ct = (inten[..., None] * np.exp(-((tt - 5) ** 2) / 4)).astype(np.float32)

    seed = np.unravel_index(int(np.nanargmax(ct.max(-1))), (X, Y, Z))
    cnn_core = [(x, y, 2) for x in range(X) for y in range(Y)
                if np.sqrt((x - 20) ** 2 + (y - 18) ** 2) <= 1.0]      # under-coverage
    grown = C._grow_vessel(ct, seed, cnn_core)
    assert grown.shape[0] > len(cnn_core) * 3                # recovers far more than the core
    dim = sum(1 for x, y, z in grown
              if 1 <= z <= 3 and 1.5 < np.sqrt((x - (18 + z)) ** 2 + (y - (20 - z)) ** 2) <= 3.05)
    assert dim >= 8                                          # includes the dim margin
    assert not any(np.sqrt((x - 34) ** 2 + (y - 34) ** 2) <= 3 and z == 2
                   for x, y, z in grown)                     # no leak into the blob


def test_motion_toggle_switches_curve():
    """The motion-correction toggle (stat 'adaptive') applies the per-frame vessel
    curve instead of the fixed max voxel."""
    cfg, ifn, dce, ct, t = _synth()
    p = C.build_aif_payload(cfg, ifn, dce, ct, t)
    assert "sss|adaptive" in p["curves"] and p["stats"] == ["max", "adaptive"]
    out = C.apply_aif_result(cfg, ifn, ct, {"accepted": True, "vessel": "sss", "stat": "adaptive"},
                             curves=p["curves"])
    assert out.meta["review"]["curve"] == "sss|adaptive"
    assert np.allclose(np.nan_to_num(out.c_a), np.nan_to_num(p["curves"]["sss|adaptive"]))


def test_diffusion_summary_and_checkpoint(monkeypatch):
    """The generic diffusion review builds a map summary (primary-map slice + median
    scalars) for any model with no per-model code; auto is a no-op, reject aborts."""
    import pbrain._webreview as WR
    from pbrain.diffusion import REGISTRY as D
    X, Y, Z = 24, 24, 6
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:X, 0:Y]
    brain = np.stack([np.sqrt((xx - 12) ** 2 + (yy - 12) ** 2) < 9 for _ in range(Z)], -1)
    fa = np.where(brain, 0.2 + 0.5 * rng.random((X, Y, Z)), 0.0).astype(np.float32)
    md = np.where(brain, 8e-4 + 2e-4 * rng.random((X, Y, Z)), 0.0).astype(np.float32)

    class Res:
        maps = {"fa": fa, "md": md}
        units = {"fa": "", "md": "mm²/s"}
        aux = {}
    model = D["dti"]
    spec = C._diffusion_summary_spec(model, Res(), brain)
    kinds = [p["kind"] for p in spec["panels"]]
    assert "image" in kinds and "values" in kinds
    vals = next(p for p in spec["panels"] if p["kind"] == "values")
    assert "fa (median)" in vals["items"] and vals["items"]["fa (median)"] != "—"

    monkeypatch.setattr(C, "active", lambda m: m in ("verify", "manual"))

    class Auto:
        mode = "auto"; subject_id = "s"
    monkeypatch.setattr(WR, "review", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("browser opened in auto")))
    assert C.diffusion_checkpoint(Auto(), "dti", model, None, Res(), brain_mask=brain) is None

    class Ver:
        mode = "verify"; subject_id = "s"
    monkeypatch.setattr(WR, "review", lambda payload, **k: {"accepted": False})
    with pytest.raises(CheckpointAbort):
        C.diffusion_checkpoint(Ver(), "dti", model, None, Res(), brain_mask=brain)


def test_tissue_payload_and_manual_exclusion():
    """The tissue checkpoint bakes a per-slice mask overlay + region tally, and a
    manual exclusion polygon removes those voxels (label→0) on that slice only."""
    X, Y, Z = 32, 32, 6
    rng = np.random.default_rng(0)
    base = rng.random((X, Y, Z)) * 0.2
    parc = np.zeros((X, Y, Z), np.int16)
    yy, xx = np.mgrid[0:X, 0:Y]
    for z in range(1, 5):
        r = np.sqrt((xx - 16) ** 2 + (yy - 16) ** 2)
        parc[:, :, z][r < 10] = 2 if z % 2 else 41
        base[:, :, z][r < 10] = 1.0
    cfg = _Cfg()
    pay = C.build_tissue_payload(cfg, parc, base, {2: "WM-left", 41: "WM-right"})
    assert pay["checkpoint"] == "tissue" and len(pay["slices"]) == Z
    assert pay["n_regions"] == 2 and pay["n_voxels"] == int((parc > 0).sum())
    assert pay["slices"][str(pay["idx"])].startswith("data:image/png")
    # per-region colours + a hover grid + region names (issue: show segments in theme
    # shades, hover for details)
    assert set(pay["colors"]) == {"2", "41"} and all(v.startswith("#") for v in pay["colors"].values())
    assert pay["label_names"] == {2: "WM-left", 41: "WM-right"}
    g = pay["label_grid"][str(pay["idx"])]
    assert {v for row in g for v in row if v} <= set(pay["label_names"])   # every grid label named

    poly = [[0.03, 0.03], [0.97, 0.03], [0.97, 0.97], [0.03, 0.97]]     # whole slice 2
    before = int((parc[:, :, 2] > 0).sum())
    out = C.apply_tissue_result(cfg, parc, {"accepted": True,
                                            "exclusions": [{"slice": 2, "polygon": poly}]})
    assert before > 0 and int((out[:, :, 2] > 0).sum()) == 0            # excluded
    assert int((out[:, :, 3] > 0).sum()) > 0                           # other slices intact
    assert int((parc[:, :, 2] > 0).sum()) == before                    # original not mutated
    with pytest.raises(CheckpointAbort):
        C.apply_tissue_result(cfg, parc, {"accepted": False})


def test_model_checkpoint_manual_refit(monkeypatch):
    """model_checkpoint returns the original result on confirm, but a RE-FITTED one
    (recording the params) when the user edits controls in manual mode; reject aborts."""
    import pbrain._webreview as WR
    from pbrain.models import CurveInputs
    from pbrain.models.patlak import PatlakModel
    T, V = 90, 25
    t = np.linspace(0, 200, T)
    x = np.maximum((t - 20) / 10, 0)
    ca = (x ** 3) * np.exp(3 * (1 - x)) + 0.02
    integ = np.concatenate([[0], np.cumsum(0.5 * (ca[1:] + ca[:-1]) * np.diff(t))])
    c_t = np.stack([0.03 * ca + 0.0004 * (v + 1) * integ for v in range(V)], axis=1)
    inp = CurveInputs(c_tissue=c_t, c_input=ca, t_s=t, mask=np.ones(V, bool))
    model = PatlakModel()
    res = model.fit(inp)
    monkeypatch.setattr(C, "active", lambda m: m in ("verify", "manual"))

    class Man:
        mode = "manual"; subject_id = "s"
    monkeypatch.setattr(WR, "review", lambda payload, **k:
                        {"accepted": True, "params": {"window_start_fraction": "0.5", "regression": "ols"}})
    out = C.model_checkpoint(Man(), "patlak", model, inp, res)
    assert out is not res and out.aux.get("manual_params") == {
        "window_start_fraction": 0.5, "regression": "ols"}          # coerced str→float

    class Ver:
        mode = "verify"; subject_id = "s"
    monkeypatch.setattr(WR, "review", lambda payload, **k: {"accepted": True})
    assert C.model_checkpoint(Ver(), "patlak", model, inp, res) is res  # no edits → original
    monkeypatch.setattr(WR, "review", lambda payload, **k: {"accepted": False})
    with pytest.raises(CheckpointAbort):
        C.model_checkpoint(Man(), "patlak", model, inp, res)


def test_baseline_payload_zoomed_and_draggable(monkeypatch):
    import pbrain._webreview as WR
    T = 60
    t = np.linspace(0, 120, T)
    S = (np.ones((8, 8, 3, T)) * 100).astype(np.float32)
    bolus = np.where(t > 25, 200 * np.exp(-(t - 25) / 25), 0.0)
    for xy in [(4, 4), (4, 5), (3, 4)]:
        S[xy[0], xy[1], 1, :] = 100 + bolus
    cfg = _Cfg()
    p = C.build_baseline_payload(cfg, S, method="auto", t_s=t)
    assert p["checkpoint"] == "baseline"
    assert p["xlim"][1] < float(t[-1])                 # zoomed to the first-peak region
    assert 0 <= p["baseline_frame"] < T and len(p["curve"]) == T
    # the user drags the point to frame 7 → checkpoint returns it
    monkeypatch.setattr(C, "active", lambda m: True)
    monkeypatch.setattr(WR, "review", lambda payload, **k: {"accepted": True, "baseline_frame": 7})
    assert C.baseline_checkpoint(cfg, S, method="auto", t_s=t) == 7


def test_baseline_checkpoint_uses_gradient_walkback():
    """`method="auto"` upgrades to the gradient/walk-back finder, so the shown baseline
    point is the LAST pre-contrast frame before the bolus rise (not a noisy earlier
    minimum)."""
    from pbrain.signal_to_conc.baseline import detect_baseline_frames, detect_baseline_gradient
    rng = np.random.default_rng(3)
    T = 60
    t = np.linspace(0, 120, T)
    base = 100 + rng.normal(0, 1.2, T)                     # a noisy flat baseline
    bolus = np.where(t > 32, 220 * np.exp(-(t - 32) / 22), 0.0)
    S = np.tile((base + bolus), (6, 6, 2, 1)).astype(float)
    onset = int(np.argmax(t > 32))
    grad = detect_baseline_gradient(S)
    assert onset - 4 <= grad <= onset                       # sits just before the rise
    assert grad > detect_baseline_frames(S)                 # later than the min-before-onset
    pay = C.build_baseline_payload(_Cfg(), S, method="auto", t_s=t)
    assert pay["baseline_frame"] == grad and pay["method"] == "gradient"
