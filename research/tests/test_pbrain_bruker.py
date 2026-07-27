"""Bruker ParaVision loader — the two subtleties that silently corrupt data:
JCAMP run-length encoding of the per-frame scaling, and the frame-group ordering
(``FGOrderDesc`` lists the FASTEST-varying group first)."""
import numpy as np

from pbrain.io.loaders.bruker import BrukerLoader, _nums, read_jcamp

NX, NY, NSL, NT = 4, 4, 3, 2
SLOPE = 2.0


def _write_scan(root, *, slice_first=True, rle=True):
    """A minimal but faithful ParaVision scan: <scan>/pdata/1/{2dseq,visu_pars}."""
    scan = root / "7"
    reco = scan / "pdata" / "1"
    reco.mkdir(parents=True)
    n_frames = NSL * NT
    order = ("(3, <FG_SLICE>, <>, 0, 2) (2, <FG_CYCLE>, <>, 2, 0)" if slice_first
             else "(2, <FG_CYCLE>, <>, 0, 2) (3, <FG_SLICE>, <>, 2, 0)")
    slope = (f"@{n_frames}*({SLOPE})" if rle else " ".join([str(SLOPE)] * n_frames))
    (reco / "visu_pars").write_text("\n".join([
        f"##$VisuCoreFrameCount={n_frames}",
        "##$VisuCoreDim=2",
        "##$VisuCoreSize=( 2 )", f"{NX} {NY}",
        "##$VisuCoreExtent=( 2 )", "8 8",
        "##$VisuCoreWordType=_16BIT_SGN_INT",
        "##$VisuCoreByteOrder=littleEndian",
        f"##$VisuCoreDataSlope=( {n_frames} )", slope,
        f"##$VisuCoreDataOffs=( {n_frames} )", f"@{n_frames}*(0)",
        "##$VisuFGOrderDescDim=2", "##$VisuFGOrderDesc=( 2 )", order,
        "##$VisuCoreOrientation=( 3, 9 )",
        " ".join(["1 0 0 0 1 0 0 0 1"] * NSL),
        "##$VisuCorePosition=( 3, 3 )", "0 0 0  0 0 2  0 0 4",
        "##$VisuAcqFlipAngle=15",
        "##$END=",
    ]))
    # frame f encodes its (slice, cycle) so the reshape order is verifiable
    frames = np.zeros((n_frames, NY, NX), dtype="<i2")
    for s in range(NSL):
        for c in range(NT):
            f = (s + NSL * c) if slice_first else (c + NT * s)
            frames[f, :, :] = s * 10 + c
    (reco / "2dseq").write_bytes(frames.tobytes())
    (scan / "method").write_text("##$Method=<Bruker:FLASH>\n##$PVM_RepetitionTime=70\n##$END=\n")
    (scan / "acqp").write_text("##$ACQ_scan_name=( 64 )\n<T1_FLASH_DCE_15 (E7)>\n##$END=\n")
    return scan


def test_rle_expansion():
    """``@6*(2.0)`` is six copies of 2.0 — not the two numbers 6 and 2.0."""
    assert _nums("@6*(2.0)") == [2.0] * 6
    assert _nums("@3*(1.5) 9") == [1.5, 1.5, 1.5, 9.0]
    assert _nums("96 96") == [96.0, 96.0]


def test_jcamp_visu_alias(tmp_path):
    scan = _write_scan(tmp_path)
    v = read_jcamp(scan / "pdata" / "1" / "visu_pars")
    assert v["VisuCoreWordType"] == "_16BIT_SGN_INT"
    assert v["CoreWordType"] == "_16BIT_SGN_INT"      # stripped alias
    assert _nums(v["CoreSize"]) == [NX, NY]


def test_loads_dce_with_correct_frame_order_and_scaling(tmp_path):
    scan = _write_scan(tmp_path, slice_first=True)
    L = BrukerLoader()
    assert L.detect(scan)
    s = L.load(scan)
    assert s.data.shape == (NX, NY, NSL, NT)
    assert s.axis4_kind == "time"
    # every voxel of (slice s, cycle c) must equal (s*10+c) * SLOPE
    for sl in range(NSL):
        for c in range(NT):
            want = (sl * 10 + c) * SLOPE
            assert np.allclose(s.data[:, :, sl, c], want), (sl, c, s.data[0, 0, sl, c])
    assert s.voxel_size[0] == 8 / NX and s.voxel_size[2] == 2.0   # extent/size, slice step
    assert s.meta["n_slices"] == NSL and s.meta["n_reps"] == NT
    assert s.meta["flip_angle_deg"] == 15.0
    assert "DCE" in s.meta["scan_name"]


def test_cycle_first_ordering_also_unpacks(tmp_path):
    """A study whose cycles vary fastest must unpack just as correctly."""
    scan = _write_scan(tmp_path, slice_first=False)
    s = BrukerLoader().load(scan)
    assert s.data.shape == (NX, NY, NSL, NT)
    for sl in range(NSL):
        for c in range(NT):
            assert np.allclose(s.data[:, :, sl, c], (sl * 10 + c) * SLOPE)


def test_static_scan_has_no_time_axis(tmp_path):
    scan = _write_scan(tmp_path)
    vp = scan / "pdata" / "1" / "visu_pars"
    vp.write_text(vp.read_text()
                  .replace("(3, <FG_SLICE>, <>, 0, 2) (2, <FG_CYCLE>, <>, 2, 0)",
                           "(6, <FG_SLICE>, <>, 0, 2)"))
    s = BrukerLoader().load(scan)
    assert s.axis4_kind == "static" and s.data.shape[-1] == 1
