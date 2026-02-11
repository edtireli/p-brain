import json
from pathlib import Path


def _write_sidecar(tmp_path: Path, *, rep_time=None, rep_time_exc=None):
    nii = tmp_path / "dce.nii"
    nii.write_bytes(b"")
    sidecar = tmp_path / "dce.json"
    data = {}
    if rep_time is not None:
        data["RepetitionTime"] = rep_time
    if rep_time_exc is not None:
        data["RepetitionTimeExcitation"] = rep_time_exc
    sidecar.write_text(json.dumps(data))
    return str(nii)


def test_resolve_turboflash_tr_prefers_excitation(tmp_path):
    from utils.loading import resolve_turboflash_tr_s

    nifti_path = _write_sidecar(tmp_path, rep_time=2.4, rep_time_exc=0.004)
    assert resolve_turboflash_tr_s(nifti_path) == 0.004


def test_resolve_turboflash_tr_rejects_large_repetitiontime(tmp_path):
    from utils.loading import resolve_turboflash_tr_s

    nifti_path = _write_sidecar(tmp_path, rep_time=2.4)
    assert resolve_turboflash_tr_s(nifti_path, default=None) is None


def test_resolve_turboflash_tr_accepts_small_repetitiontime(tmp_path):
    from utils.loading import resolve_turboflash_tr_s

    nifti_path = _write_sidecar(tmp_path, rep_time=0.005)
    assert resolve_turboflash_tr_s(nifti_path) == 0.005


def test_resolve_turboflash_tr_uses_override(monkeypatch, tmp_path):
    from utils.loading import resolve_turboflash_tr_s
    from utils import settings

    monkeypatch.setattr(settings, "TURBOFLASH_TR_S", 0.006)
    nifti_path = _write_sidecar(tmp_path, rep_time=0.003)
    assert resolve_turboflash_tr_s(nifti_path) == 0.006


def test_resolve_turboflash_tr_slice_mode_multiplies_by_slice_timing_length(tmp_path):
    from utils.loading import resolve_turboflash_tr_s

    nifti_path = _write_sidecar(tmp_path, rep_time_exc=0.004)
    sidecar = tmp_path / "dce.json"
    # Add SliceTiming so slice count can be inferred without a valid NIfTI.
    sidecar.write_text('{"RepetitionTimeExcitation": 0.004, "SliceTiming": [0, 0.1, 0.2, 0.3]}')

    assert resolve_turboflash_tr_s(nifti_path, mode="slice") == 0.016
