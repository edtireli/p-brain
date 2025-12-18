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


def test_load_streamlines_world_invokes_to_world(monkeypatch):
    called = {"to_world": False}

    class _DummyTractogram:
        def __init__(self):
            self.affine_to_rasmm = np.array(
                [[2.0, 0.0, 0.0, 10.0], [0.0, 3.0, 0.0, -5.0], [0.0, 0.0, 4.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self.streamlines = [
                np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
            ]

        def to_world(self, *, lazy=False):
            called["to_world"] = True
            assert lazy is False
            lin = self.affine_to_rasmm[:3, :3]
            offset = self.affine_to_rasmm[:3, 3]
            transformed = []
            for streamline in self.streamlines:
                transformed.append(streamline @ lin.T + offset)
            self.streamlines = transformed
            return self

    class _DummyFile:
        def __init__(self):
            self.tractogram = _DummyTractogram()

    monkeypatch.setattr(
        tractography.nib_streamlines,
        "load",
        lambda _path: _DummyFile(),
    )

    streamlines = tractography._load_streamlines_world("dummy.trk")

    assert called["to_world"] is True
    assert streamlines is not None
    assert len(streamlines) == 1
    first = np.asarray(streamlines[0])
    assert first.shape == (2, 3)
    np.testing.assert_allclose(first[0], np.array([10.0, -5.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(first[1], np.array([12.0, -2.0, 5.0], dtype=np.float32))


def test_select_stopping_prefers_act(monkeypatch):
    fa = np.ones((2, 2, 2), dtype=np.float32)
    fake_act = object()
    strategy = {"use_act": True, "act": fake_act, "use_pft": False}
    criterion = tractography._select_stopping_criterion(0.2, fa, strategy)
    assert criterion is fake_act

    fallback = tractography._select_stopping_criterion(0.2, fa, None)
    assert isinstance(fallback, tractography.ThresholdStoppingCriterion)


def test_voxel_sizes_from_affine_returns_axis_lengths():
    affine = np.diag([2.0, 3.0, 4.5, 1.0])
    sizes = tractography._voxel_sizes_from_affine(affine)
    np.testing.assert_allclose(sizes, np.array([2.0, 3.0, 4.5], dtype=np.float64))


def test_tracking_config_defaults_include_act_and_pft(monkeypatch):
    filter_defaults = {"min_length": None, "max_length": None, "subsample_stride": 1, "subsample_min_count": 0}
    config = tractography._tracking_config_defaults(filter_defaults)
    assert "act_enabled" in config
    assert "pft_enabled" in config


def test_prepare_anatomical_strategy_passes_pft_geometry(monkeypatch, tmp_path):
    fake_maps = {
        "wm": np.ones((2, 2, 2), dtype=np.float32),
        "gm": np.ones((2, 2, 2), dtype=np.float32) * 0.5,
        "csf": np.ones((2, 2, 2), dtype=np.float32) * 0.2,
    }

    def _fake_gather(nifti_dir, analysis_dir, reference_img):
        meta = {"search_roots": [nifti_dir], "sources": {t: "stub" for t in fake_maps}}
        return fake_maps, meta

    monkeypatch.setattr(tractography, "_gather_tissue_probability_maps", _fake_gather)

    act_called = {}

    class _FakeAct:
        @staticmethod
        def from_pve(wm, gm, csf):
            act_called["wm"] = wm
            return "act"

    cmc_kwargs = {}

    class _FakeCmc:
        @staticmethod
        def from_pve(wm, gm, csf, **kwargs):
            cmc_kwargs.update(kwargs)
            return "cmc"

    monkeypatch.setattr(tractography, "ActStoppingCriterion", type("_", (), {"from_pve": _FakeAct.from_pve}))
    monkeypatch.setattr(tractography, "CmcStoppingCriterion", type("_", (), {"from_pve": _FakeCmc.from_pve}))

    reference_img = nib.Nifti1Image(
        np.zeros((2, 2, 2), dtype=np.float32),
        np.diag([2.0, 3.0, 4.0, 1.0]),
    )

    filter_defaults = {"min_length": None, "max_length": None, "subsample_stride": 1, "subsample_min_count": 0}
    tracking_config = tractography._tracking_config_defaults(filter_defaults)
    tracking_config["act_enabled"] = True
    tracking_config["pft_enabled"] = True

    strategy, debug = tractography._prepare_anatomical_strategy(
        str(tmp_path),
        str(tmp_path),
        reference_img,
        tracking_config,
    )

    assert strategy["use_pft"] is True
    assert strategy["pft_kwargs"]["voxel_size"] == (2.0, 3.0, 4.0)
    assert np.isclose(cmc_kwargs["step_size"], tracking_config["pft_step_size"])
    assert np.isclose(cmc_kwargs["average_voxel_size"], np.mean([2.0, 3.0, 4.0]))
