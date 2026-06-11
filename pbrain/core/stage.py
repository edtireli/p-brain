"""Stage Protocol — the contract a pipeline stage satisfies.

A *Stage* is a unit of work in the pipeline (loading, T1/M0 fit, AIF
extraction, ...). Each stage:

* Declares the upstream manifest names it reads (``requires``).
* Declares the manifest name it writes (``name``).
* Implements ``run(ctx) -> StageOutput`` taking a :class:`StageContext`.

The orchestrator (``pbrain.core.pipeline.Pipeline``) wires stages
together by looking up the requires/produces declarations on each
stage and reading/writing manifests via the configured PathScheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config import Config
from .manifest import Manifest
from .path_scheme import PathScheme


@dataclass
class StageContext:
    """Inputs handed to ``Stage.run``.

    The stage reads upstream artefacts via ``upstream_manifests[name]``
    (parsed Manifest objects) and resolves output paths via the
    PathScheme.
    """

    subject_dir: Path
    config: Config
    path_scheme: PathScheme
    upstream_manifests: dict[str, Manifest] = field(default_factory=dict)


@dataclass
class StageOutput:
    """What a stage returns when ``run`` completes.

    The orchestrator merges this with the started/finished timestamps,
    the plug-in key used, and persists as ``manifest.json``.
    """

    artefacts: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    """Pipeline stage contract."""

    name: str
    requires: tuple[str, ...]   # upstream stage names this stage reads

    def run(self, ctx: StageContext) -> StageOutput: ...
