"""Layout auto-detection: single subject vs cohort, across adapters."""
from pathlib import Path

from pbrain.io import layout as L


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_flat_nifti_single_subject(tmp_path):
    _touch(tmp_path / "dce.nii.gz")
    _touch(tmp_path / "t1.nii.gz")
    _touch(tmp_path / "ir.nii.gz")
    r = L.resolve(tmp_path)
    assert r.kind == "subject" and r.adapter == "flat-nifti" and r.subjects == [tmp_path]
    inp = L.inputs_for(tmp_path, "flat-nifti")
    assert inp["dce"].name == "dce.nii.gz" and inp["t1"].name == "t1.nii.gz" and inp["ir"].name == "ir.nii.gz"


def test_flat_nifti_cohort(tmp_path):
    for s in ("sub01", "sub02", "sub03"):
        _touch(tmp_path / s / "dce.nii.gz")
    r = L.resolve(tmp_path)
    assert r.kind == "cohort" and r.adapter == "flat-nifti" and r.n == 3
    assert [p.name for p in r.subjects] == ["sub01", "sub02", "sub03"]


def test_subject_with_output_dir_is_not_a_cohort(tmp_path):
    # a real subject that has ALSO been run once (has a pbrain/ derivatives tree)
    _touch(tmp_path / "dce.nii.gz")
    _touch(tmp_path / "pbrain" / "derivatives" / "05_signal_to_conc" / "concentration.nii.gz")
    r = L.resolve(tmp_path)
    assert r.kind == "subject" and r.subjects == [tmp_path]     # never "cohort of one"


def test_unknown_layout(tmp_path):
    _touch(tmp_path / "notes.txt")
    assert L.resolve(tmp_path).kind == "unknown"
    assert L.resolve(tmp_path / "nope").kind == "unknown"


def test_parrec_single_and_cohort(tmp_path, monkeypatch):
    import pbrain.io.subject_discovery as SD
    monkeypatch.setattr(SD, "find_dce", lambda d: next(iter(Path(d).glob("*.PAR")), None))

    # single: a dir with a PAR that find_dce accepts
    subj = tmp_path / "one"
    _touch(subj / "s_27_1.PAR")
    r = L.resolve(subj)
    assert r.kind == "subject" and r.adapter == "parrec"

    # cohort: a root whose children are PAR subjects
    root = tmp_path / "study"
    _touch(root / "A" / "s_27_1.PAR")
    _touch(root / "B" / "s_27_1.PAR")
    r = L.resolve(root)
    assert r.kind == "cohort" and r.adapter == "parrec" and r.n == 2


def test_bids_dataset_and_single_subject(tmp_path):
    (tmp_path / "dataset_description.json").write_text('{"Name":"study","BIDSVersion":"1.8.0"}')
    for sub in ("sub-01", "sub-02"):
        _touch(tmp_path / sub / "anat" / f"{sub}_T1w.nii.gz")
        _touch(tmp_path / sub / "perf" / f"{sub}_dce.nii.gz")
    r = L.resolve(tmp_path)
    assert r.kind == "cohort" and r.adapter == "bids" and r.n == 2
    r1 = L.resolve(tmp_path / "sub-01")
    assert r1.kind == "subject" and r1.adapter == "bids"
    inp = L.inputs_for(tmp_path / "sub-01", "bids")
    assert inp["t1"].name == "sub-01_T1w.nii.gz" and inp["dce"].name == "sub-01_dce.nii.gz"


def test_bids_with_sessions(tmp_path):
    (tmp_path / "dataset_description.json").write_text("{}")
    _touch(tmp_path / "sub-01" / "ses-1" / "anat" / "sub-01_ses-1_T1w.nii.gz")
    _touch(tmp_path / "sub-01" / "ses-1" / "perf" / "sub-01_ses-1_dce.nii.gz")
    r = L.resolve(tmp_path)
    assert r.kind == "cohort" and r.adapter == "bids" and r.n == 1
    inp = L.inputs_for(tmp_path / "sub-01", "bids")
    assert inp["t1"] is not None and inp["dce"] is not None


def test_parrec_wins_over_flat_when_both_present(tmp_path, monkeypatch):
    # a subject that is unmistakably PAR/REC is reported as parrec, not flat-nifti
    import pbrain.io.subject_discovery as SD
    monkeypatch.setattr(SD, "find_dce", lambda d: next(iter(Path(d).glob("*.PAR")), None))
    _touch(tmp_path / "s_27_1.PAR")
    _touch(tmp_path / "dce.nii.gz")
    assert L.resolve(tmp_path).adapter == "parrec"


def test_frozen_layout_roundtrip_single(tmp_path):
    _touch(tmp_path / "perf" / "dyn.nii.gz")
    _touch(tmp_path / "anat" / "t1.nii.gz")
    prop = {"kind": "subject",
            "subjects": [{"dir": ".", "dce": "perf/dyn.nii.gz", "t1": "anat/t1.nii.gz", "ir": None}]}
    dest = L.write_frozen(tmp_path, prop)
    assert dest.name == "pbrain.layout.toml"
    r = L.resolve(tmp_path)
    assert r.kind == "subject" and r.adapter == "frozen"
    inp = L.inputs_for(tmp_path, "frozen")
    assert inp["dce"].name == "dyn.nii.gz" and inp["t1"].name == "t1.nii.gz" and inp["ir"] is None


def test_frozen_layout_roundtrip_cohort(tmp_path):
    for s in ("p1", "p2"):
        _touch(tmp_path / s / "perf" / "dyn.nii.gz")
    L.write_frozen(tmp_path, {"kind": "cohort", "subjects": [
        {"dir": "p1", "dce": "p1/perf/dyn.nii.gz"},
        {"dir": "p2", "dce": "p2/perf/dyn.nii.gz"}]})
    r = L.resolve(tmp_path)
    assert r.kind == "cohort" and r.adapter == "frozen" and r.n == 2
    assert L.inputs_for(tmp_path / "p1", "frozen")["dce"].name == "dyn.nii.gz"


def test_frozen_wins_over_heuristic(tmp_path):
    _touch(tmp_path / "dce.nii.gz")   # would be flat-nifti…
    L.write_frozen(tmp_path, {"kind": "subject", "subjects": [{"dir": ".", "dce": "dce.nii.gz"}]})
    assert L.resolve(tmp_path).adapter == "frozen"   # …but the frozen file wins


def test_gather_tree_is_names_only_and_skips_outputs(tmp_path):
    _touch(tmp_path / "sub-01" / "perf" / "dce.nii.gz")
    _touch(tmp_path / "sub-01" / "anat" / "t1.nii.gz")
    tree = L.gather_tree(tmp_path)
    assert "sub-01/" in tree and "perf/" in tree and "dce.nii.gz" in tree
    _touch(tmp_path / "derivatives" / "x.nii.gz")
    assert "derivatives" not in L.gather_tree(tmp_path)   # output folders are not offered to the model


def test_assist_layout_freezes_and_reresolves(tmp_path, monkeypatch):
    from pbrain import _assist
    from pbrain.cli import run as R
    _touch(tmp_path / "weird" / "scan_dynamic.nii.gz")
    monkeypatch.setattr(_assist, "available", lambda: True)
    monkeypatch.setattr(_assist, "model", lambda: "fake-model")
    monkeypatch.setattr(_assist, "propose_layout", lambda tree: {
        "kind": "subject",
        "subjects": [{"dir": ".", "dce": "weird/scan_dynamic.nii.gz", "t1": None, "ir": None}],
        "why": "single subject; the dynamic scan is the DCE"})
    res = R._assist_layout(tmp_path)             # non-TTY → auto-accepts
    assert res.kind == "subject" and res.adapter == "frozen"
    assert (tmp_path / "pbrain.layout.toml").is_file()
    assert L.inputs_for(tmp_path, "frozen")["dce"].name == "scan_dynamic.nii.gz"


def test_run_layout_fans_out_over_subjects(tmp_path, monkeypatch):
    import types
    import pbrain.cli.run as R
    from pbrain.cli import cohort as C
    for s in ("A", "B"):
        _touch(tmp_path / s / "dce.nii.gz")
        _touch(tmp_path / s / "t1.nii.gz")
    res = L.resolve(tmp_path)
    assert res.kind == "cohort" and res.n == 2

    calls: list[list[str]] = []
    monkeypatch.setattr(R, "main", lambda argv: calls.append(argv) or 0)
    args = types.SimpleNamespace(models="patlak", aggregations=None, device="cpu",
                                 derivatives_subdir="pbrain", force=True, verbose=False,
                                 quiet=True, log_file=None, workers=1)
    rc = C.run_layout(res, args)
    assert rc == 0 and len(calls) == 2
    by_subj = {a[a.index("--subject-dir") + 1]: a for a in calls}
    for sub in (str(tmp_path / "A"), str(tmp_path / "B")):
        a = by_subj[sub]
        assert a[a.index("--dce") + 1].endswith("dce.nii.gz")
        assert a[a.index("--t1") + 1].endswith("t1.nii.gz")
        assert "--force" in a and "--quiet" in a and a[a.index("--models") + 1] == "patlak"
    # the child-guard env is cleaned up afterwards
    import os
    assert "_PBRAIN_COHORT_CHILD" not in os.environ
