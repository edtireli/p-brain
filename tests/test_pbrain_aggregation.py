"""Aggregator tests — voxelwise NIfTI, parcel CSV, region JSON, slice CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pbrain.aggregation import REGISTRY


def _synthetic_maps_and_parc():
    rng = np.random.default_rng(0)
    maps = {
        "ki":  rng.standard_normal((6, 6, 4)).astype(np.float32),
        "vb":  rng.standard_normal((6, 6, 4)).astype(np.float32),
    }
    parc = np.zeros((6, 6, 4), dtype=np.int32)
    parc[:3, :3, :] = 1   # label 1
    parc[3:, 3:, :] = 2   # label 2
    labels = {1: "left_thing", 2: "right_thing"}
    region_map = {"GM": [1], "WM": [2]}
    units = {"ki": "mL/100g/min", "vb": "mL/100g"}
    return maps, units, parc, labels, region_map


def test_voxelwise_writes_one_nifti_per_map(tmp_path: Path):
    import nibabel as nib

    maps, units, *_ = _synthetic_maps_and_parc()
    res = REGISTRY["voxelwise"].aggregate(
        maps, units,
        reference_affine=np.eye(4),
        parcellation=None, labels=None, region_map=None,
        out_dir=tmp_path,
    )
    for name in ("ki", "vb"):
        p = tmp_path / f"{name}.nii.gz"
        assert p.exists(), name
        arr = np.asarray(nib.load(str(p)).dataobj)
        np.testing.assert_array_almost_equal(arr, maps[name], decimal=5)

    summary = json.loads((tmp_path / "histogram_summary.json").read_text())
    assert "ki" in summary and summary["ki"]["n_finite"] == 6 * 6 * 4


def test_parcel_writes_csv_per_map_with_label_rows(tmp_path: Path):
    maps, units, parc, labels, _ = _synthetic_maps_and_parc()
    res = REGISTRY["parcel"].aggregate(
        maps, units,
        reference_affine=np.eye(4),
        parcellation=parc, labels=labels, region_map=None,
        out_dir=tmp_path,
    )
    for name in ("ki", "vb"):
        p = tmp_path / f"{name}.csv"
        assert p.exists(), name
        with p.open() as f:
            rows = list(csv.DictReader(f))
        # Two labels (1, 2), background excluded
        assert {r["label_name"] for r in rows} == {"left_thing", "right_thing"}
        for r in rows:
            assert int(r["n"]) > 0
            assert r["mean"]  # not empty


def test_region_writes_json(tmp_path: Path):
    maps, units, parc, labels, region_map = _synthetic_maps_and_parc()
    res = REGISTRY["region"].aggregate(
        maps, units,
        reference_affine=np.eye(4),
        parcellation=parc, labels=labels, region_map=region_map,
        out_dir=tmp_path,
    )
    for name in ("ki", "vb"):
        p = tmp_path / f"{name}.json"
        assert p.exists(), name
        d = json.loads(p.read_text())
        assert "GM" in d and "WM" in d
        assert d["GM"]["n"] > 0
        assert d["WM"]["n"] > 0


def test_slice_wise_writes_csv_per_map(tmp_path: Path):
    maps, units, *_ = _synthetic_maps_and_parc()
    res = REGISTRY["slice_wise"].aggregate(
        maps, units,
        reference_affine=np.eye(4),
        parcellation=None, labels=None, region_map=None,
        slice_axis=2,
        out_dir=tmp_path,
    )
    for name in ("ki", "vb"):
        p = tmp_path / f"{name}_slice.csv"
        assert p.exists(), name
        with p.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4   # 4 slices along axis 2
        assert all(int(r["n"]) > 0 for r in rows)
