"""Plugin contract + auto-discovery tests.

Verifies that:
  * every plug-point exposes a non-empty REGISTRY keyed by plugin.key;
  * every discovered PLUGIN satisfies its plug-point's sub-Protocol;
  * every PLUGIN declares non-empty key/name/description;
  * accepts/produces are non-empty dicts;
  * key uniqueness is enforced.

Run: ``.venv/bin/python -m pytest tests/test_pbrain_contracts.py -v``
"""

from __future__ import annotations

import pytest


PLUG_POINTS = [
    ("pbrain.io.loaders",       "ImageLoader"),
    ("pbrain.io.path_schemes",  "PathScheme"),
    ("pbrain.t1_m0",            "T1M0Fitter"),
    ("pbrain.aif",              "AIFExtractor"),
    ("pbrain.tissue_roi",       "TissueROIProvider"),
    ("pbrain.signal_to_conc",   "SignalToConcConverter"),
    ("pbrain.normalisation",    "CurveNormaliser"),
    ("pbrain.models",           "KineticModel"),
    ("pbrain.aggregation",      "Aggregator"),
]


@pytest.mark.parametrize("module_name,protocol_name", PLUG_POINTS)
def test_registry_nonempty(module_name, protocol_name):
    mod = __import__(module_name, fromlist=["REGISTRY"])
    reg = getattr(mod, "REGISTRY")
    assert isinstance(reg, dict)
    assert len(reg) >= 1, f"{module_name}.REGISTRY is empty"


@pytest.mark.parametrize("module_name,protocol_name", PLUG_POINTS)
def test_registry_keys_are_canonical(module_name, protocol_name):
    mod = __import__(module_name, fromlist=["REGISTRY"])
    for key, plugin in mod.REGISTRY.items():
        assert key == plugin.key, (
            f"{module_name}.REGISTRY[{key!r}] != PLUGIN.key={plugin.key!r}"
        )


@pytest.mark.parametrize("module_name,protocol_name", PLUG_POINTS)
def test_plugin_metadata_populated(module_name, protocol_name):
    mod = __import__(module_name, fromlist=["REGISTRY"])
    for key, plugin in mod.REGISTRY.items():
        assert isinstance(plugin.key, str) and plugin.key, key
        assert isinstance(plugin.name, str) and plugin.name, key
        assert isinstance(plugin.description, str) and plugin.description, key
        assert isinstance(plugin.accepts, dict), key
        assert isinstance(plugin.produces, dict), key


@pytest.mark.parametrize("module_name,protocol_name", PLUG_POINTS)
def test_plugin_satisfies_subprotocol(module_name, protocol_name):
    mod = __import__(module_name, fromlist=["REGISTRY", protocol_name])
    proto = getattr(mod, protocol_name)
    for key, plugin in mod.REGISTRY.items():
        assert isinstance(plugin, proto), (
            f"{module_name}.{key} does not satisfy {protocol_name} Protocol"
        )


def test_models_registry_contains_patlak_and_tikhonov():
    from pbrain.models import REGISTRY
    assert "patlak" in REGISTRY
    assert "tikhonov" in REGISTRY
    # extended_tofts and gamma are bonus; gamma is gitignored so it
    # may or may not be present on a fresh clone.


def test_path_schemes_contains_both():
    from pbrain.io.path_schemes import REGISTRY
    assert "bids_like" in REGISTRY
    assert "legacy" in REGISTRY


def test_loaders_contains_all_three_formats():
    from pbrain.io.loaders import REGISTRY
    assert "nifti" in REGISTRY
    assert "parrec" in REGISTRY
    assert "dicom" in REGISTRY


def test_aggregators_contains_all_four():
    from pbrain.aggregation import REGISTRY
    for k in ("voxelwise", "parcel", "region", "slice_wise"):
        assert k in REGISTRY, f"missing aggregator: {k}"


def test_duplicate_key_raises(tmp_path, monkeypatch):
    """Two modules exporting PLUGIN.key='foo' must raise on discover()."""
    from pbrain.core import discover

    pkg = tmp_path / "dupkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    common = (
        "from dataclasses import dataclass\n"
        "from typing import ClassVar\n"
        "@dataclass(frozen=True, slots=True)\n"
        "class P:\n"
        "    key: ClassVar[str] = 'dup'\n"
        "    name: ClassVar[str] = 'x'\n"
        "    description: ClassVar[str] = 'y'\n"
        "    accepts: ClassVar[dict] = {}\n"
        "    produces: ClassVar[dict] = {}\n"
        "PLUGIN = P()\n"
    )
    (pkg / "a.py").write_text(common)
    (pkg / "b.py").write_text(common)

    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ValueError, match="Duplicate plugin key 'dup'"):
            discover("dupkg", str(pkg / "__init__.py"))
    finally:
        sys.path.remove(str(tmp_path))
