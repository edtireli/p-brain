import importlib.util
import pathlib

import nibabel as nib
import numpy as np


def _load_opt08_fa():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "modules" / "opt08_fa.py"
    spec = importlib.util.spec_from_file_location("opt08_fa_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_atlas_segmentation_accepts_4d_reference(tmp_path):
    atlas_dir = tmp_path / "segmentation" / "segmentation" / "mri"
    atlas_dir.mkdir(parents=True)

    atlas_data = np.zeros((2, 2, 2), dtype=np.int16)
    atlas_data[0, 0, 0] = 1
    atlas_img = nib.Nifti1Image(atlas_data.astype(np.float32), np.eye(4))
    atlas_path = atlas_dir / "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz"
    nib.save(atlas_img, str(atlas_path))

    reference = nib.Nifti1Image(np.zeros((2, 2, 2, 3), dtype=np.float32), np.eye(4))

    opt08_fa = _load_opt08_fa()

    result = opt08_fa._load_atlas_segmentation(str(tmp_path), reference)
    assert result is not None

    atlas_loaded, atlas_labels = result
    assert atlas_loaded.shape == (2, 2, 2)
    assert np.array_equal(np.sort(atlas_labels), np.array([1]))
