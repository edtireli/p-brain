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


def test_manual_aif_is_registered_and_not_a_stub():
    """`manual` must be importable, registered, and no longer raise NotImplementedError."""
    assert "manual" in REGISTRY
    plug = REGISTRY["manual"]
    assert plug.key == "manual"
    # The old stub's description called itself a stub; the real one does not.
    assert "stub" not in plug.description.lower()


def test_manual_aif_headless_core_extracts_known_roi():
    """Synthetic DCE + a known ROI mask over the bolus voxels → bolus curve.

    Exercises the headless core (no display). The default max_voxel path must
    pick a high-peak voxel inside the ROI, so the extracted AIF shows the bolus.
    """
    from pbrain.aif.manual import input_function_from_mask

    dce = _synthetic_dce(peak_z=8, bolus_frame=15)
    t_s = np.arange(dce.shape[-1]) * 2.463

    # ROI confined to superior slices where the bolus lives.
    mask = np.zeros(dce.shape[:3], dtype=bool)
    mask[2:6, 2:6, -2:] = True

    ifn = input_function_from_mask(dce, t_s, mask, source="rica")

    assert ifn.c_a.shape == (dce.shape[-1],)
    assert ifn.mask.shape == dce.shape[:3]
    assert ifn.source == "rica"
    # max_voxel → the curve came from a single voxel inside the ROI.
    assert int(ifn.mask.sum()) == 1
    assert bool(mask[ifn.mask]) is True
    # The bolus (frames 15:20) must dominate the baseline.
    assert ifn.c_a[15:20].max() > ifn.c_a[:5].mean() + 5.0
    assert ifn.meta["algorithm"] == "manual_roi"


def test_manual_aif_via_plugin_with_precomputed_mask():
    """The registered plug-in accepts a precomputed `mask=` headlessly."""
    dce = _synthetic_dce()
    t_s = np.arange(dce.shape[-1]) * 2.463
    mask = np.zeros(dce.shape[:3], dtype=bool)
    mask[2:6, 2:6, -2:] = True

    ifn = REGISTRY["manual"].extract(dce, t_s, np.eye(4), mask=mask, source="user")
    assert ifn.c_a.shape == (dce.shape[-1],)
    assert ifn.meta["roi_origin"] == "mask"
    assert ifn.c_a[15:20].max() > ifn.c_a[:5].mean() + 5.0


def test_manual_aif_polygon_rasterises_on_a_slice():
    """A polygon (+slice_z) is rasterised headlessly and yields the bolus curve."""
    dce = _synthetic_dce()
    t_s = np.arange(dce.shape[-1]) * 2.463
    z = dce.shape[2] - 1
    # Square polygon over the brain centre on the top slice.
    poly = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)]

    ifn = REGISTRY["manual"].extract(
        dce, t_s, np.eye(4), polygon=poly, slice_z=z, source="user",
    )
    assert ifn.meta["roi_origin"] == "polygon"
    assert ifn.meta["slice_z"] == z
    # The chosen voxel must lie on the drawn slice.
    assert np.argwhere(ifn.mask)[0, 2] == z
    assert ifn.c_a[15:20].max() > ifn.c_a[:5].mean() + 5.0


def test_manual_aif_headless_no_roi_raises_clearly():
    """Without a display and without a mask/polygon, it must raise a clear error."""
    dce = _synthetic_dce()
    t_s = np.arange(dce.shape[-1]) * 2.463
    with pytest.raises(RuntimeError, match="no ROI supplied"):
        REGISTRY["manual"].extract(dce, t_s, np.eye(4), interactive=False)


def test_manual_aif_empty_mask_raises():
    from pbrain.aif.manual import input_function_from_mask

    dce = _synthetic_dce()
    t_s = np.arange(dce.shape[-1]) * 2.463
    empty = np.zeros(dce.shape[:3], dtype=bool)
    with pytest.raises(ValueError, match="empty"):
        input_function_from_mask(dce, t_s, empty)
