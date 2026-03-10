"""Tests for _load_map_in_dce_grid NaN-cache rejection.

When a previous T1-fitting run produced all-NaN maps (e.g. due to a bug),
the cached ``t1_map_in_dce.nii.gz`` will contain only NaN values.
``_load_map_in_dce_grid`` must detect this and regenerate the cache from
the source ``t1_map.nii.gz`` instead of silently returning all-NaN data.
"""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def dce_nifti_dir(tmp_path):
    """Create a tiny mock Nifti directory with a 4-D DCE file."""
    affine = np.eye(4)
    shape = (4, 4, 2, 10)  # small 3-D spatial + 10 time points
    data = np.random.default_rng(0).random(shape).astype(np.float32) * 1000
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(tmp_path / "dce.nii.gz"))
    return str(tmp_path)


@pytest.fixture
def fitting_dir_with_nan_cache(tmp_path, dce_nifti_dir):
    """Create Analysis/Fitting with a good source map + an all-NaN cache."""
    fit_dir = tmp_path / "Analysis" / "Fitting"
    fit_dir.mkdir(parents=True)

    affine = np.eye(4)
    spatial = (4, 4, 2)

    # Good source map (38% finite, rest NaN — similar to real data)
    good = np.full(spatial, np.nan)
    good[1:3, 1:3, :] = np.random.default_rng(1).random((2, 2, 2)) * 2000 + 500
    src_img = nib.Nifti1Image(good.astype(np.float32), affine)
    nib.save(src_img, str(fit_dir / "t1_map.nii.gz"))

    # Stale all-NaN cache (from a previous broken run)
    nan_cache = np.full(spatial, np.nan, dtype=np.float32)
    nib.save(nib.Nifti1Image(nan_cache, affine), str(fit_dir / "t1_map_in_dce.nii.gz"))

    return str(fit_dir.parent)  # analysis_directory


def test_nan_cache_rejected(dce_nifti_dir, fitting_dir_with_nan_cache):
    """An all-NaN _in_dce cache should be rejected and regenerated."""
    from utils.plotting import _load_map_in_dce_grid

    result = _load_map_in_dce_grid(
        fitting_dir_with_nan_cache,
        dce_nifti_dir,
        "dce.nii.gz",
        base_name="t1_map",
    )
    assert result is not None, "Should return data, not None"
    n_finite = int(np.isfinite(result).sum())
    assert n_finite > 0, f"Expected finite values but got {n_finite}"
    # The regenerated cache should match the source map
    src = nib.load(
        os.path.join(fitting_dir_with_nan_cache, "Fitting", "t1_map.nii.gz")
    ).get_fdata()
    np.testing.assert_array_equal(
        np.isfinite(result), np.isfinite(src),
        err_msg="Regenerated cache should have the same finite mask as source",
    )


def test_good_cache_accepted(dce_nifti_dir, tmp_path):
    """A cache with finite values should still be accepted normally."""
    from utils.plotting import _load_map_in_dce_grid

    fit_dir = tmp_path / "Analysis2" / "Fitting"
    fit_dir.mkdir(parents=True)

    affine = np.eye(4)
    spatial = (4, 4, 2)
    good = np.random.default_rng(2).random(spatial).astype(np.float32) * 1500 + 200

    nib.save(nib.Nifti1Image(good, affine), str(fit_dir / "t1_map.nii.gz"))
    nib.save(nib.Nifti1Image(good, affine), str(fit_dir / "t1_map_in_dce.nii.gz"))

    result = _load_map_in_dce_grid(
        str(fit_dir.parent),
        dce_nifti_dir,
        "dce.nii.gz",
        base_name="t1_map",
    )
    assert result is not None
    np.testing.assert_allclose(result, good, atol=1e-5)
