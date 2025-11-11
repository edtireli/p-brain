import json
from typing import Tuple

import numpy as np

from modules import opt08_fa as diffusion


class _FakeTensorFit:
    def __init__(self, data: np.ndarray, fa_values: np.ndarray, mode_values: np.ndarray):
        self._data = data
        self.fa = fa_values.astype(np.float32)
        self.md = np.full_like(self.fa, 0.5, dtype=np.float32)
        self.ad = np.full_like(self.fa, 0.6, dtype=np.float32)
        self.rd = np.full_like(self.fa, 0.4, dtype=np.float32)
        self.mode = mode_values.astype(np.float32)

    def predict(self, _gtab):
        return np.zeros_like(self._data, dtype=np.float32)


class _FakeTensorModel:
    def __init__(self, gtab):
        self._gtab = gtab

    def fit(self, data: np.ndarray) -> _FakeTensorFit:
        fa_values, mode_values = _tensor_payload()
        return _FakeTensorFit(data, fa_values, mode_values)


def _tensor_payload() -> Tuple[np.ndarray, np.ndarray]:
    fa_values = np.linspace(0.1, 0.8, 8, dtype=np.float32).reshape((2, 2, 2))
    mode_values = np.linspace(0.2, 0.9, 8, dtype=np.float32).reshape((2, 2, 2))
    return fa_values, mode_values


class _FakeLoadedImage:
    def __init__(self, data: np.ndarray):
        self._data = data.astype(np.float32)
        self.affine = np.eye(4, dtype=np.float32)
        self.header = {}

    def get_fdata(self, dtype=None):
        array = self._data
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def set_data_dtype(self, _dtype):
        return


class _FakeNiftiImage:
    def __init__(self, data: np.ndarray, affine, header):
        self._data = np.asarray(data)
        self.affine = affine
        self.header = header

    def set_data_dtype(self, dtype):
        self._data = self._data.astype(dtype)

    def get_fdata(self, dtype=None):
        array = self._data
        if dtype is not None:
            array = array.astype(dtype)
        return array


def test_compute_fa_includes_all_brain_tissues(monkeypatch, tmp_path):
    nifti_dir = tmp_path / "NIfTI"
    analysis_dir = tmp_path / "Analysis"
    nifti_dir.mkdir()
    analysis_dir.mkdir()

    dwi_path = nifti_dir / "fake_dwi.nii.gz"
    dwi_path.write_bytes(b"")
    bval_path = nifti_dir / "fake.bval"
    bvec_path = nifti_dir / "fake.bvec"
    np.savetxt(bval_path, np.array([0.0, 1000.0, 1000.0]))
    np.savetxt(bvec_path, np.eye(3))

    fa_values, mode_values = _tensor_payload()
    diffusion_data = np.ones((*fa_values.shape, 3), dtype=np.float32)

    saved_images = {}

    white_matter_mask = np.zeros_like(fa_values, dtype=bool)
    white_matter_mask[0] = True
    cortical_mask = np.zeros_like(fa_values, dtype=bool)
    cortical_mask[1, 0] = True
    subcortical_mask = np.zeros_like(fa_values, dtype=bool)
    subcortical_mask[1, 1, 0] = True
    brainstem_mask = np.zeros_like(fa_values, dtype=bool)
    brainstem_mask[1, 1, 1] = True

    tissue_masks = {
        "white_matter": (
            white_matter_mask,
            {"json_label": "white_matter", "plot_label": "White matter"},
        ),
        "cortical_gm": (
            cortical_mask,
            {"json_label": "cortical_gm", "plot_label": "Cortical GM"},
        ),
        "subcortical_gm": (
            subcortical_mask,
            {"json_label": "subcortical_gm", "plot_label": "Subcortical GM"},
        ),
        "brainstem": (
            brainstem_mask,
            {"json_label": "brainstem", "plot_label": "Brainstem"},
        ),
    }

    monkeypatch.setattr(
        diffusion,
        "find_dwi_files",
        lambda *_args, **_kwargs: (str(dwi_path), str(bval_path), str(bvec_path)),
    )
    monkeypatch.setattr(diffusion, "gradient_table", lambda **_kwargs: object())
    monkeypatch.setattr(diffusion, "TensorModel", _FakeTensorModel)
    monkeypatch.setattr(diffusion, "_collect_tissue_masks", lambda *_args: tissue_masks)
    monkeypatch.setattr(diffusion, "_ensure_image_directory", lambda *_args: None)
    monkeypatch.setattr(diffusion, "_plot_metric_histogram", lambda *_args: None)
    monkeypatch.setattr(diffusion, "_load_atlas_segmentation", lambda *_args: None)
    monkeypatch.setattr(diffusion, "_maybe_resample_to_dce", lambda img, *_args, **_kwargs: img)

    monkeypatch.setattr(diffusion.nib, "load", lambda _path: _FakeLoadedImage(diffusion_data))
    monkeypatch.setattr(diffusion.nib, "Nifti1Image", lambda data, affine, header: _FakeNiftiImage(data, affine, header))
    monkeypatch.setattr(diffusion.nib, "save", lambda img, path: saved_images.setdefault(path, img))

    diffusion.compute_fa(
        str(nifti_dir),
        str(analysis_dir),
    )

    stats_path = analysis_dir / "diffusion" / "diffusion_values_median_total.json"
    with open(stats_path) as fp:
        stats = json.load(fp)

    fa_stats = stats["fa"]
    assert "cortical_gm_median_total" in fa_stats
    assert fa_stats["brain_median_total"]["voxel_count"] == 8
    assert np.isclose(fa_stats["brain_median_total"]["mean"], float(np.mean(fa_values)))

    mo_stats = stats["mo"]
    assert "cortical_gm_median_total" in mo_stats
    assert mo_stats["brain_median_total"]["voxel_count"] == 8

    mean_fa_path = analysis_dir / "diffusion" / "fa_mean.txt"
    with open(mean_fa_path) as fp:
        mean_fa_value = float(fp.read().strip())
    assert np.isclose(mean_fa_value, float(np.mean(fa_values)))

    wm_mean_path = analysis_dir / "diffusion" / "fa_mean_wm.txt"
    with open(wm_mean_path) as fp:
        wm_mean_value = float(fp.read().strip())
    expected_wm_mean = float(np.mean(fa_values.ravel()[:4]))
    assert np.isclose(wm_mean_value, expected_wm_mean)

    fa_map_path = str(analysis_dir / "diffusion" / "fa_map.nii.gz")
    assert fa_map_path in saved_images
    fa_map = saved_images[fa_map_path].get_fdata()
    assert not np.isnan(fa_map).any()

