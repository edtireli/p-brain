"""Deterministic tests for the vision-assisted AIF localiser (`pbrain.aif_vision`).

These exercise the classic-CV scaffolding only — projections, orientation, region
boxes, peak refinement, the CNN cross-check, and file selection. The VLM call is
mocked, so nothing here downloads a model or needs a GPU.
"""
import numpy as np
import pytest

from pbrain import aif_vision as V


def test_max_projection_is_nan_hardened():
    vol = np.zeros((2, 2, 1, 3), np.float32)
    vol[0, 0, 0] = [1, 5, 2]                 # temporal peak = 5
    vol[1, 1, 0] = np.nan                    # an all-NaN voxel
    mx = V.max_projection(vol)
    assert mx.shape == (2, 2, 1)
    assert np.isfinite(mx).all()             # no NaN propagates
    assert mx[0, 0, 0] == 5.0
    assert mx[1, 1, 0] == 0.0                # all-NaN → 0, not NaN


def test_axial_anterior_up_puts_anterior_on_top():
    X, Y = 4, 6                              # RAS+ slice [i=L→R, j=P→A]
    sl = np.zeros((X, Y), np.float32)
    sl[:, Y - 1] = 1.0                       # brightest at the most-anterior j
    disp = V._axial_anterior_up(sl)          # display should put anterior at the top row
    assert disp.shape == (Y, X)
    assert disp[0].mean() == 1.0             # top row is anterior
    assert disp[-1].mean() == 0.0            # bottom row is posterior


def test_box_splits_left_right_at_true_midline():
    X = Y = 100
    lx, _ = V._box("central", "left", X, Y, mid_x=40)
    rx, _ = V._box("central", "right", X, Y, mid_x=40)
    assert lx == slice(0, 40)
    assert rx == slice(40, 100)
    # x=45 is left of the image centre (50) but right of the true midline (40):
    assert rx.start <= 45 < rx.stop         # → belongs to the RIGHT hemisphere


def test_refine_peak_respects_midline_for_laterality():
    mx = np.zeros((100, 100, 1), np.float32)
    mx[45, 60, 0] = 9.0                      # bright, x=45 (right of midline 40), anterior
    right = V.refine_peak(mx, 0, "anterior", "right", mid_x=40)
    assert right is not None and right["voxel"][0] == 45
    left = V.refine_peak(mx, 0, "anterior", "left", mid_x=40)
    assert left is None                      # the left box has no enhancement


def test_refine_peak_none_on_zero_signal_region():
    mx = np.zeros((50, 50, 1), np.float32)   # no enhancement anywhere
    assert V.refine_peak(mx, 0, "anterior", "right") is None


def test_refine_peak_out_of_range_slice():
    mx = np.ones((10, 10, 2), np.float32)
    assert V.refine_peak(mx, 5, "central", "midline") is None


def test_cnn_peak_voxel_centroid_then_peak(tmp_path):
    nib = pytest.importorskip("nibabel")
    m = np.zeros((10, 10, 3), np.float32)
    m[2:5, 2:5, 1] = 1                       # a 3×3 blob on z=1 (centroid at 3,3,1)
    p = tmp_path / "aif_mask.nii.gz"
    nib.save(nib.Nifti1Image(m, np.eye(4)), str(p))
    assert V.cnn_peak_voxel(p) == [3, 3, 1]  # no mx → centroid
    mx = np.zeros((10, 10, 3), np.float32)
    mx[4, 4, 1] = 7.0
    assert V.cnn_peak_voxel(p, mx) == [4, 4, 1]   # with mx → peak-in-mask


def test_cnn_peak_voxel_single_voxel(tmp_path):
    nib = pytest.importorskip("nibabel")
    m = np.zeros((8, 8, 2), np.float32)
    m[5, 1, 0] = 1
    p = tmp_path / "aif_mask.nii.gz"
    nib.save(nib.Nifti1Image(m, np.eye(4)), str(p))
    assert V.cnn_peak_voxel(p) == [5, 1, 0]  # degenerate case unchanged


def test_canonical_glob_skips_backup_tree(tmp_path):
    real = tmp_path / "derivatives" / "05_signal_to_conc" / "sr"
    bak = tmp_path / "derivatives.jun24_bak" / "05_signal_to_conc" / "sr"
    real.mkdir(parents=True)
    bak.mkdir(parents=True)
    (real / "concentration.nii.gz").write_bytes(b"")
    (bak / "concentration.nii.gz").write_bytes(b"")
    got = V.canonical_glob(tmp_path, "concentration.nii.gz")
    assert got is not None
    assert "jun24_bak" not in got
    assert got.replace("\\", "/").endswith("derivatives/05_signal_to_conc/sr/concentration.nii.gz")


def test_montage_png_is_size_capped(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    mx = np.random.default_rng(0).random((256, 256, 10)).astype(np.float32)
    out = V.montage_png(mx, tmp_path / "m.png")
    w, h = Image.open(out).size
    assert w <= 1100 and h <= 700            # small enough for the VLM patch budget


def test_temporal_gating_separates_vessel_from_mucosa():
    T = 250
    t = np.arange(T)
    # vessel bolus: fast wash-in to an early peak, then wash-out
    tp = 55
    bolus = np.empty(T)
    bolus[:tp] = np.linspace(0, 3, tp)
    bolus[tp:] = 3 * np.exp(-(t[tp:] - tp) / 40.0)
    fb = V.temporal_features(bolus)
    assert V.is_vessel(fb) and V.classify_curve(fb) == "vessel"
    assert fb["ttp_frac"] < 0.4 and fb["early_late"] > 2.0

    # mucosa: slow monotonic accumulation, peaks at the very end
    mucosa = np.linspace(0, 3, T)
    fm = V.temporal_features(mucosa)
    assert not V.is_vessel(fm) and V.classify_curve(fm) == "mucosa/tissue"
    assert fm["ttp_frac"] > 0.9 and fm["early_late"] < 1.0

    # region_curve picks the brightest-peak voxel; flat curve → "none"
    vol = np.zeros((3, 1, 1, T), np.float32)
    vol[0, 0, 0] = bolus
    vol[1, 0, 0] = 0.5 * bolus
    mask = np.zeros((3, 1, 1), bool)
    mask[:2, 0, 0] = True
    assert np.allclose(V.region_curve(vol, mask), bolus.astype(np.float32))
    assert V.classify_curve(V.temporal_features(np.zeros(T))) == "none"


def test_vesselness_map_favours_bolus_over_mucosa():
    T = 250
    t = np.arange(T)
    tp = 55
    bolus = np.empty(T)
    bolus[:tp] = np.linspace(0, 3, tp)
    bolus[tp:] = 3 * np.exp(-(t[tp:] - tp) / 40.0)
    mucosa = np.linspace(0, 3, T)
    vol = np.zeros((2, 1, 1, T), np.float32)
    vol[0, 0, 0] = bolus
    vol[1, 0, 0] = mucosa
    vk = V.vesselness_map(vol, enhance_pct=0)          # keep both voxels
    assert vk[0, 0, 0] > 2.0                            # bolus is early-dominated
    assert vk[1, 0, 0] < 1.0                            # mucosa is late-dominated
    assert vk[0, 0, 0] > vk[1, 0, 0]


def test_bbox_to_voxel_and_grow_in_bbox():
    X = Y = 256
    ir, jr = V._bbox_to_voxel([256, 128, 384, 256], X, Y, 512)
    assert ir == (128, 192)                 # x: 256..384 of 512 → 128..192 of 256
    assert jr == (127, 191)                 # y flips: rows 64..128 → j 127..191
    field = np.zeros((256, 256, 3), np.float32)
    field[150, 150, 1] = 5.0
    field[151, 150, 1] = 4.0
    m = V.grow_in_bbox(field, 1, (128, 192), (127, 191), frac=0.5)
    assert m is not None and m[150, 150, 1] and m.sum() >= 1
    assert V.grow_in_bbox(np.zeros((256, 256, 3), np.float32), 1, (128, 192), (127, 191)) is None


def test_region_metrics_overlap_and_iou():
    a = np.zeros((10, 10, 2), bool); a[2:5, 2:5, 0] = True
    b = np.zeros((10, 10, 2), bool); b[3:6, 3:6, 0] = True
    mm = V.region_metrics(a, b)
    assert mm["overlap"] and 0.0 < mm["iou"] < 1.0 and mm["centroid_dist"] > 0
    assert V.region_metrics(a, None) is None
    disjoint = np.zeros((10, 10, 2), bool); disjoint[8:10, 8:10, 1] = True
    assert V.region_metrics(a, disjoint)["overlap"] is False


def test_temporal_detector_labels_artery_and_vein():
    X = Y = 40
    Z = 3
    T = 250
    t = np.arange(T)

    def bolus(tp):
        c = np.empty(T)
        c[:tp] = np.linspace(0, 3, tp)
        c[tp:] = 3 * np.exp(-(t[tp:] - tp) / 40.0)
        return c

    vol = np.zeros((X, Y, Z, T), np.float32)
    vol[19:22, 24:27, 0, :] = bolus(30)      # artery: anterior (high j), skull-base z0, EARLY
    vol[18:22, 4:8, 2, :] = bolus(60)         # vein: posterior (low j), superior z2, LATER, bigger
    cands = V.detect_candidates(vol, min_voxels=5, vk_pct=0)
    assert len(cands) >= 2
    picks = V.label_vessels(cands, V.max_projection(vol))
    assert picks["artery"] and picks["vein"]
    # arteries fill before the sinus; the sinus sits posterior (lower j)
    assert picks["artery"]["features"]["arrival"] < picks["vein"]["features"]["arrival"]
    assert picks["vein"]["centroid"][1] < picks["artery"]["centroid"][1]


def test_find_regions_two_stage_mocked(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    X = Y = 40
    Z = 3
    T = 250
    t = np.arange(T)
    bolus = np.empty(T); bolus[:55] = np.linspace(0, 3, 55); bolus[55:] = 3 * np.exp(-(t[55:] - 55) / 40.0)
    vol = np.zeros((X, Y, Z, T), np.float32)
    vol[20, 20, 0, :] = bolus                # artery-like bolus, central, skull-base slice
    conc = tmp_path / "concentration.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(conc))

    monkeypatch.setattr(V, "vlm_available", lambda: True)

    def fake(png, repo, prompt=None):
        if "bbox_2d" in prompt:
            return {"bbox_2d": [230, 230, 290, 290]}   # ~voxel i,j 17..22 at size 512, X=40
        return {"artery": {"slice": 0}, "vein": {"slice": 2}}
    monkeypatch.setattr(V, "_vlm_locate", fake)

    res = V.find_regions(conc, "fake-repo", tmp_path)
    assert res["artery"]["klass"] == "vessel"           # temporal gate says bolus
    assert res["artery"]["voxels"] >= 1
    assert res["artery"]["slice"] == 0


def test_find_aif_crosschecks_cnn_and_calibrates_midline(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    X = Y = 40
    vol = np.zeros((X, Y, 1, 3), np.float32)
    vol[20, 5, 0, :] = [1, 3, 1]             # SSS: near-midline x=20, posterior y=5
    vol[30, 35, 0, :] = [1, 9, 1]            # right carotid: x=30 (>midline), anterior, bright
    conc = tmp_path / "concentration.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(conc))
    mask = np.zeros((X, Y, 1), np.float32)
    mask[20, 5, 0] = 1
    cnn = tmp_path / "aif_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(cnn))

    monkeypatch.setattr(V, "vlm_available", lambda: True)
    monkeypatch.setattr(V, "_vlm_locate", lambda png, repo: {
        "sss": {"slice": 0, "ap": "posterior", "lr": "midline", "found": True},
        "r_ica": {"slice": 0, "ap": "anterior", "lr": "right", "found": True},
        "l_ica": {"slice": 0, "ap": "anterior", "lr": "left", "found": True},
    })
    res = V.find_aif(conc, "fake-repo", tmp_path, cnn_mask=cnn)
    assert res["_cnn"] == [20, 5, 0]
    assert res["sss"]["voxel"] == [20, 5, 0]
    assert res["sss"]["dist_cnn"] == 0.0
    # midline calibrated from the SSS (x=20) → x=30 is correctly the RIGHT carotid
    assert res["r_ica"]["voxel"] == [30, 35, 0]
    assert res["r_ica"]["dist_cnn"] > 20     # anterior carotid sits far from the posterior CNN sinus
    # the left-anterior quadrant has no enhancement → honest None, not a bogus pick
    assert res["l_ica"] is None
