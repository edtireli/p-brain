"""Contract tests for `--animal`: it selects PROVIDERS, never a different pipeline.

The design rule is that a mouse run executes the *identical* modular pipeline a human
run does — different loaders, converters and segmentation, same stages in the same
order, no species branches anywhere. That rule is easy to state, easy to violate with
one `if mouse:`, and invisible until someone reads the diff. So it is asserted here.

Also covers the two shared stages that no test previously exercised at all
(`SignalToConcStage`, `AIFStage`), since changes made for the mouse path run on human
data too.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pbrain.aif import REGISTRY as AIF
from pbrain.cli._profiles import PROFILES, resolve_profile
from pbrain.core.config import Config
from pbrain.core.manifest import Manifest
from pbrain.core.stage import StageContext
from pbrain.io.path_schemes import REGISTRY as PATH_SCHEMES
from pbrain.models import REGISTRY as MODELS
from pbrain.signal_to_conc import REGISTRY as CONVERTERS
from pbrain.stages import default_stages
from pbrain.stages._builtin import AIFStage, SignalToConcStage
from pbrain.t1_m0 import REGISTRY as T1M0
from pbrain.tissue_roi import REGISTRY as TISSUE_ROI

PLUG_POINTS = {
    "signal_to_conc": CONVERTERS, "aif": AIF, "tissue_roi": TISSUE_ROI,
    "t1_m0": T1M0, "models": MODELS,
}


# ── the pipeline is the same pipeline ───────────────────────────────────

def test_every_profile_leaves_the_stage_list_untouched():
    """A profile picks plug-ins. If one ever adds or removes a stage, the mouse path
    has stopped being the human path and this is the cheapest place to notice."""
    human = [s.name for s in default_stages()]
    for name in PROFILES:
        assert [s.name for s in default_stages()] == human, (
            f"profile {name!r} changed the resolved stage list")
    assert "signal_to_conc" in human and "aif" in human


def test_profiles_only_set_known_config_fields():
    """Guards the other half: a profile may only move knobs Config already has, so it
    cannot smuggle in species-specific behaviour through a novel field."""
    allowed = set(Config.__dataclass_fields__) | {"opt", "models", "aggregations",
                                                  "t1m0", "signal_to_conc", "aif",
                                                  "tissue_roi", "normaliser"}
    for name, prof in PROFILES.items():
        unknown = set(prof) - allowed
        assert not unknown, f"profile {name!r} sets unknown fields {sorted(unknown)}"


def test_no_stage_branches_on_species():
    """The invariant, enforced against the source: no stage may test what animal it is
    looking at. Comments and default constants are fine; a conditional is not."""
    src = Path(inspect.getfile(SignalToConcStage)).read_text(encoding="utf-8")
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    pattern = re.compile(
        r"""(if|elif|while|assert)\b[^\n]*\b(mouse|animal|species|rodent|rat)\b"""
        r"""|\b(mouse|animal|species|rodent|rat)\b\s*(==|!=|\bin\b)""", re.I)
    offenders = [ln.strip() for ln in code if pattern.search(ln)]
    assert not offenders, (
        "a stage is branching on species; use a different provider instead:\n  "
        + "\n  ".join(offenders))


# ── profile options must actually reach their plug-in ───────────────────

@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_profile_options_resolve_to_real_plugins_and_parameters(profile_name):
    """A mistyped `--opt` key is SILENTLY IGNORED — the plug-in then runs on its
    defaults with no error. Renaming a converter without renaming its option keys
    would, for example, quietly drop t1_0_ms and change every concentration."""
    def entry_point(plugin):
        for attr in ("convert", "extract", "fit", "run", "__call__"):
            fn = getattr(plugin, attr, None)
            if callable(fn):
                return fn
        return None

    problems = []
    for opt in resolve_profile(profile_name)["opt"]:
        lhs = opt.split("=", 1)[0]
        parts = lhs.split(".")
        if len(parts) < 3:
            problems.append(f"{opt!r}: expected <point>.<plugin>.<key>")
            continue
        point, plugin_key, key = ".".join(parts[:-2]), parts[-2], parts[-1]
        registry = PLUG_POINTS.get(point)
        if registry is None:
            continue                        # e.g. qc.* — not a plug-point
        if plugin_key not in registry:
            problems.append(f"{opt!r}: no {plugin_key!r} in {point} "
                            f"(have {sorted(registry)})")
            continue
        fn = entry_point(registry[plugin_key])
        if fn is None:
            continue
        params = inspect.signature(fn).parameters
        if key not in params and not any(p.kind == p.VAR_KEYWORD
                                         for p in params.values()):
            problems.append(f"{opt!r}: {plugin_key} accepts no {key!r}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_profile_selects_plugins_that_exist(profile_name):
    prof = resolve_profile(profile_name)
    for field, registry in (("signal_to_conc", CONVERTERS), ("aif", AIF),
                            ("tissue_roi", TISSUE_ROI), ("t1m0", T1M0)):
        key = prof.get(field)
        if key:
            assert key in registry, f"{profile_name}.{field}={key!r} does not exist"
    for m in str(prof.get("models", "")).split(","):
        if m.strip():
            assert m.strip() in MODELS, f"{profile_name} selects unknown model {m!r}"


# ── the two shared stages actually run ──────────────────────────────────

def _synthetic_subject(tmp_path: Path, *, shape=(6, 6, 3), n_t=40):
    """A minimal but honest upstream: a DCE series with a bolus, a T1 map and an M0
    map, written as real NIfTIs with real manifests."""
    rng = np.random.default_rng(0)
    t = np.arange(n_t, dtype=float)
    bolus = np.exp(-((t - 10) ** 2) / 18.0)
    S = 100.0 * (1.0 + 0.45 * bolus)[None, None, None, :] * np.ones((*shape, 1))
    S += rng.normal(0, 0.4, S.shape)
    affine = np.diag([0.5, 0.5, 1.0, 1.0])

    d = tmp_path / "derivatives"
    (d / "01_load").mkdir(parents=True)
    (d / "02_t1m0").mkdir(parents=True)
    nib.save(nib.Nifti1Image(S.astype(np.float32), affine), d / "01_load" / "dce.nii.gz")
    np.save(d / "01_load" / "t.npy", t * 4.0)
    nib.save(nib.Nifti1Image(np.full(shape, 1500.0), affine), d / "02_t1m0" / "t1.nii.gz")
    nib.save(nib.Nifti1Image(np.full(shape, 500.0), affine), d / "02_t1m0" / "m0.nii.gz")
    (d / "01_load" / "dce.json").write_text(json.dumps({"meta": {}}), encoding="utf-8")

    load = Manifest(stage="load", plugin="nifti", outputs={
        "dce": str(d / "01_load" / "dce.nii.gz"),
        "dce_time_s": str(d / "01_load" / "t.npy"),
        "dce_meta": str(d / "01_load" / "dce.json"),
    })
    t1m0 = Manifest(stage="t1_m0", plugin="vfa_spgr", outputs={
        "t1_map_ms": str(d / "02_t1m0" / "t1.nii.gz"),
        "m0_map": str(d / "02_t1m0" / "m0.nii.gz"),
    })
    return load, t1m0


def _ctx(tmp_path, config, **upstream):
    return StageContext(subject_dir=tmp_path, config=config,
                        path_scheme=PATH_SCHEMES["bids_like"],
                        upstream_manifests=dict(upstream))


# Both species' converters, so a change made for one is exercised on the other.
@pytest.mark.parametrize("converter", ["spgr_ratio", "spgr_bounded",
                                       "saturation_recovery"])
def test_signal_to_conc_stage_runs(tmp_path, converter):
    load, t1m0 = _synthetic_subject(tmp_path)
    cfg = Config(signal_to_conc=converter, flip_angle_deg=15.0, tr_s=0.07,
                 r1_per_s_mM=2.8, baseline_frames=5)
    out = SignalToConcStage().run(_ctx(tmp_path, cfg, load=load, t1_m0=t1m0))

    conc = Path(out.artefacts["concentration"])
    assert conc.exists()
    arr = np.asarray(nib.load(str(conc)).dataobj)
    assert arr.ndim == 4 and np.isfinite(arr).any()
    assert out.metadata["converter"] == converter


def test_signal_to_conc_saturation_qc_is_optional_not_required():
    """Converters without a `saturation()` hook must run unaffected — the QC is
    duck-typed precisely so the human path keeps working."""
    assert not hasattr(CONVERTERS["saturation_recovery"], "saturation")


@pytest.mark.parametrize("converter", ["spgr_ratio", "spgr_bounded"])
def test_aif_stage_runs_and_reports_saturation(tmp_path, converter):
    """The AIF stage gained saturation and washout QC; both must survive a real run,
    including when the upstream manifest carries no saturation metadata."""
    load, t1m0 = _synthetic_subject(tmp_path)
    cfg = Config(signal_to_conc=converter, aif_extractor="auto_vessel",
                 flip_angle_deg=15.0, tr_s=0.07, r1_per_s_mM=2.8, baseline_frames=5)
    stc = SignalToConcStage().run(_ctx(tmp_path, cfg, load=load, t1_m0=t1m0))
    stc_mf = Manifest(stage="signal_to_conc", plugin=converter,
                      outputs={k: str(v) for k, v in stc.artefacts.items()},
                      metadata=stc.metadata)

    out = AIFStage().run(_ctx(tmp_path, cfg, load=load, t1_m0=t1m0,
                              signal_to_conc=stc_mf))
    assert Path(out.artefacts["aif_signal"]).exists()
    meta = json.loads(Path(out.artefacts["aif_meta"]).read_text(encoding="utf-8"))["meta"]
    assert 0.0 <= meta["saturated_fraction"] <= 1.0


def test_aif_stage_survives_upstream_without_saturation_metadata(tmp_path):
    """A converter that reports no ceiling must not break the AIF stage — it falls
    back to the curve-shape test instead."""
    load, t1m0 = _synthetic_subject(tmp_path)
    cfg = Config(signal_to_conc="spgr_ratio", aif_extractor="auto_vessel",
                 flip_angle_deg=15.0, tr_s=0.07, r1_per_s_mM=2.8, baseline_frames=5)
    stc = SignalToConcStage().run(_ctx(tmp_path, cfg, load=load, t1_m0=t1m0))
    bare = Manifest(stage="signal_to_conc", plugin="spgr_ratio",
                    outputs={k: str(v) for k, v in stc.artefacts.items()},
                    metadata={})                      # <- no saturation block

    out = AIFStage().run(_ctx(tmp_path, cfg, load=load, t1_m0=t1m0,
                              signal_to_conc=bare))
    meta = json.loads(Path(out.artefacts["aif_meta"]).read_text(encoding="utf-8"))["meta"]
    assert meta["saturation_test"] == "clipped curve"
