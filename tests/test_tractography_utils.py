from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_modules_package() -> None:
    modules_pkg = sys.modules.get("modules")
    if modules_pkg is None or not hasattr(modules_pkg, "__path__"):
        modules_pkg = types.ModuleType("modules")
        modules_pkg.__path__ = [str(ROOT / "modules")]
        sys.modules["modules"] = modules_pkg
    else:
        modules_pkg.__path__ = [str(ROOT / "modules")]


def _load_tractography_module():
    _ensure_modules_package()

    if "modules.opt08_fa" not in sys.modules:
        stub = types.ModuleType("modules.opt08_fa")

        def _stub_find_dwi_files(*_args, **_kwargs):
            raise RuntimeError("stub – not expected in tests")

        stub.find_dwi_files = _stub_find_dwi_files  # type: ignore[attr-defined]
        sys.modules["modules.opt08_fa"] = stub

    spec = importlib.util.spec_from_file_location(
        "modules.tractography",
        ROOT / "modules" / "tractography.py",
        submodule_search_locations=[str(ROOT / "modules")],
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules["modules.tractography"] = module
    spec.loader.exec_module(module)
    return module


tractography = _load_tractography_module()


def test_seed_points_from_mask_passes_canonical_affine(monkeypatch):
    captured = {}

    def _fake_seeds(mask, affine, *, density):
        captured["mask"] = mask
        captured["affine"] = affine
        captured["density"] = density
        return np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    monkeypatch.setattr(tractography, "seeds_from_mask", _fake_seeds)

    raw_affine = np.eye(5)
    mask = np.ones((2, 2, 2), dtype=bool)

    seeds = tractography._seed_points_from_mask(mask, raw_affine, density=3)

    assert seeds.shape == (1, 3)
    assert np.array_equal(captured["mask"], mask)
    assert captured["density"] == 3
    assert captured["affine"].shape == (4, 4)
    assert np.allclose(captured["affine"], np.eye(4))


def test_load_background_volume_resamples_4d_reference(tmp_path):
    overlay_data = np.ones((4, 4, 4), dtype=np.float32)
    overlay_img = nib.Nifti1Image(overlay_data, np.eye(4))
    overlay_path = tmp_path / "overlay.nii.gz"
    nib.save(overlay_img, overlay_path)

    reference_data = np.zeros((6, 5, 4, 3), dtype=np.float32)
    reference_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference_img = nib.Nifti1Image(reference_data, reference_affine)

    resampled = tractography._load_background_volume(str(overlay_path), reference_img)

    assert resampled.shape[:3] == reference_img.shape[:3]
    assert np.allclose(resampled.affine, tractography._canonical_affine(reference_affine))
