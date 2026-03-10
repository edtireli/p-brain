"""Tests for resolve_dce_time_step_s and resolve_dce_time_points_s.

Verifies that FrameTimesStart is preferred over RepetitionTime when the
sidecar comes from nibabel PAR/REC conversion (where RepetitionTime is the
excitation TR, not the dynamic scan interval).
"""

import json
import os
import tempfile

import numpy as np
import pytest

from utils.loading import (
    _dt_from_frame_times,
    resolve_dce_time_step_s,
    resolve_dce_time_points_s,
    build_time_points_s,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti_with_sidecar(tmp_path, sidecar: dict, n_volumes: int = 10):
    """Create a minimal 4-D NIfTI + JSON sidecar pair and return the path."""
    import nibabel as nib

    nii_path = os.path.join(str(tmp_path), "test.nii.gz")
    json_path = os.path.join(str(tmp_path), "test.json")

    data = np.zeros((4, 4, 1, n_volumes), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, nii_path)

    with open(json_path, "w") as f:
        json.dump(sidecar, f)

    return nii_path


# ---------------------------------------------------------------------------
# _dt_from_frame_times
# ---------------------------------------------------------------------------

class TestDtFromFrameTimes:
    def test_uniform(self):
        assert _dt_from_frame_times([0, 1, 2, 3, 4]) == pytest.approx(1.0)

    def test_nonuniform_median(self):
        # median of [1.73, 1.22, 1.23, 1.22] = 1.225
        dt = _dt_from_frame_times([0.0, 1.73, 2.95, 4.18, 5.40])
        assert dt == pytest.approx(1.225, abs=0.01)

    def test_single_entry(self):
        assert _dt_from_frame_times([5.0]) is None

    def test_empty(self):
        assert _dt_from_frame_times([]) is None

    def test_non_finite(self):
        assert _dt_from_frame_times([0, float("nan"), float("nan")]) is None

    def test_negative_diff(self):
        # all diffs <= 0
        assert _dt_from_frame_times([5, 3, 1]) is None


# ---------------------------------------------------------------------------
# resolve_dce_time_step_s
# ---------------------------------------------------------------------------

class TestResolveDceTimeStepS:
    def test_dcm2niix_repetition_time_no_fts(self, tmp_path):
        """dcm2niix with RepetitionTime only -- should return it directly."""
        nii = _make_nifti_with_sidecar(tmp_path, {
            "RepetitionTime": 2.46,
        })
        assert resolve_dce_time_step_s(nii) == pytest.approx(2.46)

    def test_dcm2niix_repetition_time_with_matching_fts(self, tmp_path):
        """dcm2niix with both RepetitionTime and FrameTimesStart that agree.

        FrameTimesStart should be preferred (it's checked first).
        """
        fts = [i * 2.44 for i in range(10)]
        nii = _make_nifti_with_sidecar(tmp_path, {
            "RepetitionTime": 2.46,
            "FrameTimesStart": fts,
        }, n_volumes=10)
        dt = resolve_dce_time_step_s(nii)
        # Should get the FTS-derived dt (~2.44), which is close to RT (2.46)
        assert dt == pytest.approx(2.44, abs=0.05)

    def test_nibabel_excitation_tr_with_fts(self, tmp_path):
        """nibabel PAR/REC: RepetitionTime = excitation TR (wrong),
        FrameTimesStart = correct per-volume timing.

        THIS IS THE BUG THAT WAS FIXED.
        """
        fts = [i * 1.23 for i in range(749)]
        nii = _make_nifti_with_sidecar(tmp_path, {
            "RepetitionTime": 0.003947,
            "RepetitionTimeExcitation": 0.003947,
            "FrameTimesStart": fts,
        }, n_volumes=749)
        dt = resolve_dce_time_step_s(nii)
        assert dt == pytest.approx(1.23, abs=0.01)
        # Must NOT be the excitation TR
        assert dt > 0.1

    def test_fts_with_two_entries(self, tmp_path):
        """FrameTimesStart with only 2 entries (e.g. IR series)."""
        nii = _make_nifti_with_sidecar(tmp_path, {
            "RepetitionTime": 3.55,
            "FrameTimesStart": [0.0, 3.55],
        }, n_volumes=2)
        dt = resolve_dce_time_step_s(nii)
        assert dt == pytest.approx(3.55, abs=0.1)

    def test_fts_single_entry_falls_back_to_rt(self, tmp_path):
        """FTS with single entry -> dt_fts = None -> fall back to RT."""
        nii = _make_nifti_with_sidecar(tmp_path, {
            "RepetitionTime": 2.0,
            "FrameTimesStart": [0.0],
        })
        dt = resolve_dce_time_step_s(nii)
        assert dt == pytest.approx(2.0)

    def test_no_sidecar(self, tmp_path):
        """No JSON sidecar at all -- should fall back to NIfTI header."""
        import nibabel as nib
        nii_path = os.path.join(str(tmp_path), "test.nii.gz")
        data = np.zeros((4, 4, 1, 10), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header["pixdim"][4] = 1.5
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nib.save(img, nii_path)

        dt = resolve_dce_time_step_s(nii_path)
        assert dt == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# resolve_dce_time_points_s
# ---------------------------------------------------------------------------

class TestResolveDceTimePointsS:
    def test_returns_fts_array(self, tmp_path):
        """Should return the actual FrameTimesStart array when available."""
        fts = [0.0, 1.73, 2.95, 4.18, 5.40]
        nii = _make_nifti_with_sidecar(tmp_path, {
            "RepetitionTime": 0.004,
            "FrameTimesStart": fts,
        }, n_volumes=5)
        tp = resolve_dce_time_points_s(nii, n_volumes=5)
        np.testing.assert_allclose(tp, fts, atol=1e-4)

    def test_falls_back_to_uniform_grid(self, tmp_path):
        """No FTS -- should return uniform grid from RT."""
        nii = _make_nifti_with_sidecar(tmp_path, {
            "RepetitionTime": 2.0,
        }, n_volumes=5)
        tp = resolve_dce_time_points_s(nii, n_volumes=5)
        expected = np.arange(5, dtype=np.float32) * 2.0
        np.testing.assert_allclose(tp, expected, atol=1e-4)

    def test_fts_length_mismatch_falls_back(self, tmp_path):
        """FTS length != n_volumes -> should fall back to uniform grid."""
        nii = _make_nifti_with_sidecar(tmp_path, {
            "RepetitionTime": 2.0,
            "FrameTimesStart": [0.0, 2.0, 4.0],
        }, n_volumes=5)
        tp = resolve_dce_time_points_s(nii, n_volumes=5)
        assert len(tp) == 5
