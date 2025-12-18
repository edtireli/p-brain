from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_opt08_module():
    modules_pkg = sys.modules.get("modules")
    if modules_pkg is None or not hasattr(modules_pkg, "__path__"):
        modules_pkg = types.ModuleType("modules")
        modules_pkg.__path__ = [str(ROOT / "modules")]
        sys.modules["modules"] = modules_pkg
    else:
        modules_pkg.__path__ = [str(ROOT / "modules")]

    spec = importlib.util.spec_from_file_location(
        "modules.opt08_fa",
        ROOT / "modules" / "opt08_fa.py",
        submodule_search_locations=[str(ROOT / "modules")],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["modules.opt08_fa"] = module
    spec.loader.exec_module(module)
    return module


opt08_fa = _load_opt08_module()


def test_classify_atlas_label_identifies_csf_variants():
    assert opt08_fa._classify_atlas_label("Left-Lateral-Ventricle") == "csf"
    assert opt08_fa._classify_atlas_label("4th-Ventricle") == "csf"
    assert opt08_fa._classify_atlas_label("Left-Inf-Lat-Vent") == "csf"
    assert opt08_fa._classify_atlas_label("Left-Choroid-Plexus") == "csf"


def test_derive_tissue_masks_from_atlas_adds_csf_mask():
    atlas_data = np.zeros((2, 2, 2), dtype=np.int32)
    atlas_data[0, 0, 0] = 1  # ventricle label
    atlas_data[1, 1, 1] = 2  # cortical gm label

    atlas_labels = np.array([1, 2], dtype=np.int32)
    label_lookup = {
        1: "Left-Lateral-Ventricle",
        2: "ctx-lh-insula",
    }

    masks = opt08_fa._derive_tissue_masks_from_atlas(atlas_data, atlas_labels, label_lookup)

    assert "csf" in masks
    assert masks["csf"].shape == atlas_data.shape
    assert masks["csf"][0, 0, 0]
    assert not masks["csf"][1, 1, 1]
    assert "cortical_gm" in masks