"""AIF plug-in tests — deterministic extractor on a synthetic 4-D volume."""

from __future__ import annotations

import numpy as np
import pytest

from pbrain.aif import REGISTRY


def _synthetic_dce(X=8, Y=8, Z=10, T=80, peak_z=8, bolus_frame=15, seed=0):
    """Tiny 4-D DCE phantom: brain mask + sharp arterial peak in top slices."""
    rng = np.random.default_rng(seed)
    bg = rng.standard_normal((X, Y, Z, T)).astype(np.float32) * 0.05 + 0.1
    # Brain mask (disk in XY)
    Xs, Ys = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    r2 = (Xs - X / 2) ** 2 + (Ys - Y / 2) ** 2
    brain = r2 < (X / 2) ** 2                                # (X, Y)
    brain3 = np.broadcast_to(brain[..., None], (X, Y, Z))
    bg[brain3, :] += 5.0
    # Add a sharp bolus in superior slices only
    bolus = np.zeros(T, dtype=np.float32)
    bolus[bolus_frame:bolus_frame + 5] = 10.0
    z_high = np.arange(Z) >= (peak_z - 1)                    # (Z,)
    target = brain3 & z_high[None, None, :]                  # (X, Y, Z)
    bg[target] = bg[target] + bolus[None, :]
    return bg


def test_deterministic_aif_picks_high_peak_voxels():
    dce = _synthetic_dce()
    t_s = np.arange(80) * 2.463
    aif = REGISTRY["deterministic"]
    ifn = aif.extract(dce, t_s, np.eye(4),
                      source="rica", baseline_frames=5, n_voxels=16,
                      z_range_fraction=(0.6, 1.0))
    assert ifn.c_a.shape == (80,)
    assert ifn.mask.shape == dce.shape[:3]
    assert int(ifn.mask.sum()) <= 16
    # The bolus peak should appear in the extracted AIF
    assert ifn.c_a.max() > ifn.c_a[:5].mean() + 1.0


def test_cnn_aif_raises_clearly_without_weights(tmp_path):
    """CNN plugin is no longer a stub — it raises only if weights missing."""
    aif = REGISTRY["cnn"]
    dce = _synthetic_dce()
    with pytest.raises(FileNotFoundError, match="CNN weights not found"):
        aif.extract(
            dce, np.arange(80) * 2.463, np.eye(4),
            slice_classifier_path=str(tmp_path / "nonexistent_sc.keras"),
            roi_model_path=str(tmp_path / "nonexistent_unet.keras"),
        )


def test_from_file_aif_loads_a_mask(tmp_path):
    import nibabel as nib

    dce = _synthetic_dce()
    mask = np.zeros(dce.shape[:3], dtype=np.uint8)
    mask[2:5, 2:5, -3:] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(tmp_path / "mask.nii.gz"))
    aif = REGISTRY["from_file"]
    ifn = aif.extract(dce, np.arange(80) * 2.463, np.eye(4),
                      mask_path=tmp_path / "mask.nii.gz")
    assert int(ifn.mask.sum()) == int(mask.sum())
