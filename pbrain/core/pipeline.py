"""Pipeline orchestrator — wires Stages together via manifests.

Usage::

    from pbrain.core import Pipeline, Config
    from pbrain.io.path_schemes import REGISTRY as PATH_SCHEMES

    pipe = Pipeline(stages=[...], path_scheme=PATH_SCHEMES["bids_like"])
    results = pipe.run(subject_dir, config)

The orchestrator runs each stage in declaration order, wiring upstream
manifests by name. Each stage's output is persisted as ``manifest.json``
via the PathScheme before the next stage runs.

The orchestrator deliberately makes no assumptions about *which*
stages exist or their order — pass the stages you want. The default
9-stage pipeline lives in ``pbrain.cli.run.default_pipeline()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .logs import get_logger
from .manifest import Manifest
from .path_scheme import PathScheme
from .stage import Stage, StageContext, StageOutput

_log = get_logger("pipeline")


def _provenance(config: Config) -> dict[str, Any]:
    """Compact, reproducible record of what produced a stage's output."""
    from pbrain import __version__
    return {
        "pbrain_version": __version__,
        "device": config.device,
        "plugins": {
            "t1_m0": config.t1_m0_fitter, "aif": config.aif_extractor,
            "tissue_roi": config.tissue_roi_provider,
            "signal_to_conc": config.signal_to_conc,
            "normaliser": config.normaliser,
            "models": list(config.kinetic_models),
            "diffusion": list(config.diffusion_models),
            "aggregations": list(config.analysis_levels),
            "path_scheme": config.path_scheme,
        },
        "options": config.plugin_options,
    }


@dataclass
class StageRunRecord:
    """One stage's result within a :class:`Pipeline` run.

    Attributes
    ----------
    name
        The stage name.
    output
        The :class:`StageOutput` the stage returned.
    manifest_path
        Path to the ``manifest.json`` written for the stage.
    skipped
        ``True`` if the stage was skipped (e.g. up-to-date / not
        applicable to this subject).
    """

    name: str
    output: StageOutput
    manifest_path: Path
    skipped: bool = False


@dataclass
class Pipeline:
    """Sequential pipeline runner. Stages share data only via manifests."""

    stages: list[Stage]
    path_scheme: PathScheme

    def run(self, subject_dir: Path, config: Config,
            *, force: bool = False) -> dict[str, StageRunRecord]:
        """Run the pipeline.

        Resume semantics: a stage is **skipped** when its manifest already
        exists with ``qc.status == "pass"`` AND no upstream stage re-ran in
        this invocation — so re-running the same command continues where it
        left off, in seconds. ``force=True`` re-runs everything. Changing a
        plug-in option does not auto-invalidate; pass ``force`` (or delete the
        stage's output) to recompute.

        Every produced manifest is stamped with provenance (pbrain version +
        resolved plug-in selection + options) in ``manifest.config``.
        """
        subject_dir = Path(subject_dir)
        records: dict[str, StageRunRecord] = {}
        prov = _provenance(config)
        dirty: set[str] = set()

        for stage in self.stages:
            plugin_key = getattr(stage, "plugin_key", "<stage>")
            manifest_path = self.path_scheme.manifest_path(
                subject_dir, stage.name, plugin_key
            )

            # ── resume: skip if already done and nothing upstream re-ran ──
            upstream_dirty = any(r in dirty for r in stage.requires)
            if (not force) and (not upstream_dirty) and manifest_path.exists():
                try:
                    existing = Manifest.read(manifest_path)
                    # "pass" or "warn" both mean the stage completed (warn = a
                    # QC flag, not an error); only "fail" forces a recompute.
                    if (existing.qc or {}).get("status") in ("pass", "warn"):
                        _log.info("  %-18s skip (cached)", stage.name)
                        records[stage.name] = StageRunRecord(
                            name=stage.name,
                            output=StageOutput(
                                artefacts={k: Path(v) for k, v in existing.outputs.items()},
                                metadata=existing.metadata,
                            ),
                            manifest_path=manifest_path,
                            skipped=True,
                        )
                        continue
                except Exception:
                    pass   # unreadable manifest → recompute

            upstream: dict[str, Manifest] = {}
            for upstream_name in stage.requires:
                if upstream_name not in records:
                    raise RuntimeError(
                        f"Stage {stage.name!r} requires {upstream_name!r} "
                        f"but no prior stage produced it. "
                        f"Available: {list(records.keys())}"
                    )
                upstream[upstream_name] = Manifest.read(records[upstream_name].manifest_path)

            ctx = StageContext(
                subject_dir=subject_dir,
                config=config,
                path_scheme=self.path_scheme,
                upstream_manifests=upstream,
            )

            manifest = Manifest.now(stage=stage.name, plugin=plugin_key)
            manifest.config = prov              # provenance stamp

            try:
                output = stage.run(ctx)
            except Exception as exc:
                manifest.qc = {"status": "fail", "messages": [str(exc)]}
                manifest.finish().write(manifest_path)
                raise

            manifest.outputs = {k: str(v) for k, v in output.artefacts.items()}
            manifest.metadata = dict(output.metadata)
            manifest.qc = dict(output.qc) if output.qc else {"status": "pass"}
            manifest.inputs = {
                name: str(rec.manifest_path) for name, rec in records.items()
                if name in stage.requires
            }
            manifest.finish()
            manifest.write(manifest_path)

            dirty.add(stage.name)               # this stage (re)ran → downstream invalid
            records[stage.name] = StageRunRecord(
                name=stage.name,
                output=output,
                manifest_path=manifest_path,
            )

        return records
