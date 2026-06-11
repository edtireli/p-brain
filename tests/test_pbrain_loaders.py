"""Loader tests — NIfTI on a synthesised in-memory volume.

DICOM/PAR-REC need external converters (dcm2niix) so we only verify
detection here; integration tests live in ``validation/`` once real
data is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbrain.io.loaders import REGISTRY, load_4d


def test_nifti_loader_roundtrip(tmp_path: Path):
    import nibabel as nib

    rng = np.random.default_rng(0)
    arr4 = rng.standard_normal((4, 5, 6, 7)).astype(np.float32)
    affine = np.diag([0.9, 0.9, 1.0, 1.0])
    out = tmp_path / "test.nii.gz"
    nib.save(nib.Nifti1Image(arr4, affine), str(out))

    series = load_4d(out)
    assert series.data.shape == (4, 5, 6, 7)
    np.testing.assert_array_almost_equal(series.data, arr4, decimal=5)
    # NIfTI stores the affine as float32 internally; allow tiny round-trip error.
    np.testing.assert_allclose(series.affine, affine, atol=1e-6)
    assert series.voxel_size == pytest.approx((0.9, 0.9, 1.0), abs=1e-6)
    assert series.n_frames == 7
    assert series.shape3d == (4, 5, 6)


def test_nifti_loader_3d_becomes_n1(tmp_path: Path):
    import nibabel as nib

    arr3 = np.ones((3, 4, 5), dtype=np.float32)
    nib.save(nib.Nifti1Image(arr3, np.eye(4)), str(tmp_path / "t1.nii.gz"))
    series = load_4d(tmp_path / "t1.nii.gz")
    assert series.data.shape == (3, 4, 5, 1)
    assert series.axis4_kind == "static"


def test_nifti_loader_detect():
    nifti = REGISTRY["nifti"]
    assert nifti.detect(Path("foo.nii"))
    assert nifti.detect(Path("foo.nii.gz"))
    assert not nifti.detect(Path("foo.dcm"))
    assert not nifti.detect(Path("foo.par"))


def test_parrec_loader_detect():
    parrec = REGISTRY["parrec"]
    assert parrec.detect(Path("scan.par"))
    assert parrec.detect(Path("scan.PAR"))
    assert not parrec.detect(Path("scan.nii"))


def test_load_4d_unknown_format_raises(tmp_path: Path):
    bogus = tmp_path / "thing.xyz"
    bogus.write_text("not a real image")
    with pytest.raises(ValueError, match="No loader can read"):
        load_4d(bogus)
