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

    # Volume must exceed the loader's 2048-byte truncation guard, so use a
    # shape whose gzipped payload is comfortably larger (random data resists
    # compression). A 3-D NIfTI must be promoted to (X, Y, Z, 1).
    rng = np.random.default_rng(0)
    arr3 = rng.standard_normal((16, 16, 8)).astype(np.float32)
    out = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(arr3, np.eye(4)), str(out))
    assert out.stat().st_size > 2048
    series = load_4d(out)
    assert series.data.shape == (16, 16, 8, 1)
    assert series.axis4_kind == "static"


def test_nifti_loader_rejects_truncated_file(tmp_path: Path):
    """The <2048-byte guard must reject an empty/truncated NIfTI with a
    clear, file-naming error (failed conversion / disk-full / bad copy)."""
    from nibabel.filebasedimages import ImageFileError

    tiny = tmp_path / "truncated.nii.gz"
    tiny.write_bytes(b"\x1f\x8b" + b"\x00" * 50)   # 52 bytes — well under guard
    assert tiny.stat().st_size < 2048
    with pytest.raises(ImageFileError, match="empty or truncated"):
        load_4d(tiny)


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


# ── DICOM loader: detection (no external deps) ─────────────────────────────

def test_dicom_detect_extension():
    dicom = REGISTRY["dicom"]
    assert dicom.detect(Path("scan.dcm"))
    assert dicom.detect(Path("scan.IMA"))
    assert not dicom.detect(Path("scan.nii.gz"))


def test_dicom_detect_extensionless_via_magic(tmp_path: Path):
    """Enhanced/multi-frame DICOMs often have no extension; the 'DICM' magic
    at byte 128 must still classify them as DICOM."""
    dicom = REGISTRY["dicom"]
    f = tmp_path / "IM0001"   # no extension
    f.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 16)
    assert dicom.detect(f)
    # A directory containing such a file must also detect (series directory).
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    (series_dir / "IM0002").write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 16)
    assert dicom.detect(series_dir)


def test_dicom_missing_dcm2niix_raises_install_hint(tmp_path: Path, monkeypatch):
    """Without dcm2niix on PATH the loader must raise a clear install hint
    rather than a cryptic FileNotFoundError."""
    import pbrain.io.loaders.dicom as dmod

    f = tmp_path / "IM0001"
    f.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 16)
    monkeypatch.setattr(dmod.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="dcm2niix is required"):
        REGISTRY["dicom"].load(f)


@pytest.mark.skipif(
    __import__("shutil").which("dcm2niix") is None,
    reason="dcm2niix binary not on PATH",
)
def test_dicom_directory_series_roundtrip(tmp_path: Path):
    """End-to-end: synthesise a small multi-slice DICOM series, convert it via
    the loader (dcm2niix), and confirm a 3-D/4-D NIfTI comes back. Skips if
    pydicom is unavailable (it is an optional `[dicom]` extra)."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    series_dir = tmp_path / "series"
    series_dir.mkdir()
    rows, cols, n_slices = 8, 8, 4
    study_uid = generate_uid()
    series_uid = generate_uid()
    for i in range(n_slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "MR"
        ds.Rows, ds.Columns = rows, cols
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * 3.0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 3.0
        ds.PixelData = (np.full((rows, cols), i + 1, dtype=np.uint16)).tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        _dcm_out = str(series_dir / f"slice{i:03d}.dcm")
        try:
            ds.save_as(_dcm_out, enforce_file_format=True)      # pydicom >= 3
        except TypeError:
            ds.save_as(_dcm_out, write_like_original=False)     # pydicom 2.x

    series = load_4d(series_dir)
    # dcm2niix stacks the slices into a 3-D volume → promoted to (X, Y, Z, 1).
    assert series.data.ndim == 4
    assert series.data.shape[2] == n_slices or series.data.shape[3] == n_slices
