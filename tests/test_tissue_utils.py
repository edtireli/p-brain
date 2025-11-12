import logging
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import AI_tissue_functions as tissue


def test_clean_metric_inplace_fills_nan_stripe():
    data = np.linspace(0.0, 1.0, num=125, dtype=np.float32).reshape((5, 5, 5))
    original = data.copy()
    mask = np.ones_like(data, dtype=bool)
    data[:, 2, :] = np.nan

    tissue.clean_metric_inplace(data, mask, median_size=3, sigma_vox=0.0)

    assert not np.isnan(data[mask]).any()
    median = np.median(original[mask])
    mad = np.median(np.abs(original[mask] - median))
    diff = np.abs((data - original)[mask])
    assert np.max(diff) <= mad + 1e-6


def test_resample_like_identity():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(4, 4, 4)).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(data, affine)

    resampled = tissue.resample_like(img, img, order=1)

    np.testing.assert_allclose(resampled.get_fdata(), data)
    np.testing.assert_allclose(resampled.affine, affine)


def test_masked_percentiles_ignore_nan_and_inf():
    data = np.array(
        [
            [0.0, np.nan, 5.0, 10.0],
            [1.0, np.inf, 6.0, 12.0],
            [2.0, 3.0, 7.0, 14.0],
            [4.0, 8.0, 9.0, 16.0],
        ],
        dtype=np.float32,
    ).reshape((2, 2, 4))
    mask = np.ones_like(data, dtype=bool)

    lo, hi = tissue.masked_percentiles(data, mask, lo=25.0, hi=75.0)

    finite = data[np.isfinite(data)]
    expected_lo, expected_hi = np.nanpercentile(finite, (25.0, 75.0))
    assert lo == pytest.approx(expected_lo)
    assert hi == pytest.approx(expected_hi)


def test_robust_slice_indices_quantiles():
    mask = np.zeros((3, 3, 8), dtype=bool)
    for idx in range(8):
        mask[: (idx % 3) + 1, :, idx] = True

    result = tissue.robust_slice_indices(mask, axis=2, n=4)

    occupied = np.flatnonzero(np.any(mask, axis=(0, 1)))
    expected = np.rint(
        np.interp(
            np.linspace(0.0, 1.0, num=4),
            np.linspace(0.0, 1.0, num=occupied.size),
            occupied.astype(float),
        )
    ).astype(int)
    expected = np.clip(expected, occupied.min(), occupied.max())
    assert result == expected.tolist()


def test_get_tissue_masks_like_logs_once(tmp_path, caplog):
    data = np.ones((3, 3, 3), dtype=np.float32)
    affine = np.eye(4)
    metric_path = tmp_path / "metric.nii.gz"
    mask_path = tmp_path / "wm.nii.gz"
    nib.Nifti1Image(data, affine).to_filename(metric_path)
    nib.Nifti1Image(data, affine).to_filename(mask_path)

    ref_img = nib.load(str(metric_path))

    caplog.set_level(logging.INFO, logger=tissue.logger.name)
    first = tissue.get_tissue_masks_like(ref_img)
    second = tissue.get_tissue_masks_like(ref_img)

    assert "combined" in first
    assert "combined" in second

    resample_logs = [record.message for record in caplog.records if "Resampled mask" in record.message]
    assert len(resample_logs) == 1
