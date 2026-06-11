"""Path scheme tests — verify deterministic output layout."""

from __future__ import annotations

from pathlib import Path

from pbrain.io.path_schemes import REGISTRY


def test_bids_like_kinetic_voxelwise():
    s = REGISTRY["bids_like"]
    p = s.output_path(Path("/sub-01"), "kinetic", "patlak", "voxelwise", "ki", "nii.gz")
    assert p == Path("/sub-01/derivatives/07_kinetic/patlak/voxelwise/ki.nii.gz")


def test_bids_like_t1m0_no_aggregation():
    s = REGISTRY["bids_like"]
    p = s.output_path(Path("/sub-01"), "t1_m0", "inversion_recovery", None,
                      "t1_map_ms", "nii.gz")
    assert p == Path("/sub-01/derivatives/02_t1m0/inversion_recovery/t1_map_ms.nii.gz")


def test_bids_like_manifest():
    s = REGISTRY["bids_like"]
    p = s.manifest_path(Path("/sub-01"), "aif", "deterministic")
    assert p == Path("/sub-01/derivatives/03_aif/deterministic/manifest.json")


def test_legacy_kinetic_uses_image_dirs():
    s = REGISTRY["legacy"]
    p = s.output_path(Path("/sub-01"), "kinetic", "patlak", "voxelwise", "ki", "nii.gz")
    assert p == Path("/sub-01/Images/AI_patlak/voxelwise/ki.nii.gz")


def test_legacy_t1m0_in_fitting():
    s = REGISTRY["legacy"]
    p = s.output_path(Path("/sub-01"), "t1_m0", "inversion_recovery", None,
                      "t1_map_ms", "nii.gz")
    assert p == Path("/sub-01/Analysis/Fitting/t1_map_ms.nii.gz")


def test_path_scheme_ext_normalisation():
    """Ext can be passed with or without a leading dot."""
    s = REGISTRY["bids_like"]
    a = s.output_path(Path("/x"), "load", None, None, "dce", "nii.gz")
    b = s.output_path(Path("/x"), "load", None, None, "dce", ".nii.gz")
    assert a == b
