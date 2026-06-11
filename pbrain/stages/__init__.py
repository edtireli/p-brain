"""Pipeline stages — a discoverable plug-point.

A *stage* is a unit of pipeline work (load, T1/M0, AIF, kinetic, …). Each
stage declares the upstream stage names it ``requires`` and the manifest
name it produces (its ``name``). The pipeline order is resolved by a
**topological sort** over those declarations — no hardcoded list.

Built-in stages live in ``_builtin.py`` / ``_diagnostics.py`` (underscore-
prefixed, registered explicitly below). A research group adds a new stage
by dropping ``pbrain/stages/<name>.py`` that exports a module-level
``PLUGIN`` with ``name``, ``requires`` and ``run(ctx)`` — it is
auto-discovered and slotted into the dependency order automatically.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from pbrain.core import Stage
from ._builtin import (
    LoadStage, T1M0Stage, SignalToConcStage, AIFStage, TissueROIStage,
    NormalisationStage, KineticStage, DiffusionStage, AggregationStage,
)
from ._diagnostics import DiagnosticsStage

# Built-in stages, registered explicitly (insertion order = stable tie-break).
_BUILTIN: list[Stage] = [
    LoadStage(), T1M0Stage(), SignalToConcStage(), AIFStage(),
    TissueROIStage(), NormalisationStage(), KineticStage(),
    DiffusionStage(), AggregationStage(), DiagnosticsStage(),
]


def _discover_extra() -> list[Stage]:
    """Auto-discover non-underscore ``pbrain/stages/*.py`` modules that
    export a ``PLUGIN`` stage (research-group additions)."""
    found: list[Stage] = []
    pkg_dir = Path(__file__).parent
    for path in sorted(pkg_dir.glob("*.py")):
        if path.stem.startswith("_") or path.stem == "base":
            continue
        module = importlib.import_module(f".{path.stem}", package=__name__)
        plugin = getattr(module, "PLUGIN", None)
        if plugin is not None and isinstance(plugin, Stage):
            found.append(plugin)
    return found


REGISTRY: dict[str, Stage] = {s.name: s for s in (_BUILTIN + _discover_extra())}


def resolve_pipeline(stages: list[Stage] | None = None) -> list[Stage]:
    """Topologically order ``stages`` by their ``requires`` declarations.

    Stable: among stages whose dependencies are all satisfied, the original
    registration order is preserved — so the built-in pipeline resolves to
    its canonical order while new stages slot in after their dependencies.

    Raises on a missing dependency or a cycle.
    """
    pool = list(stages) if stages is not None else list(REGISTRY.values())
    by_name = {s.name: s for s in pool}
    ordered: list[Stage] = []
    done: set[str] = set()

    # Kahn's algorithm with stable selection.
    remaining = list(pool)
    while remaining:
        ready = [
            s for s in remaining
            if all((r in done) or (r not in by_name) for r in s.requires)
        ]
        if not ready:
            unresolved = {s.name: s.requires for s in remaining}
            raise RuntimeError(
                f"Cannot order stages — missing dependency or cycle among "
                f"{unresolved}"
            )
        nxt = ready[0]                 # stable: first ready in registration order
        ordered.append(nxt)
        done.add(nxt.name)
        remaining.remove(nxt)
    return ordered


def default_stages() -> list[Stage]:
    """The built-in pipeline, dependency-ordered."""
    return resolve_pipeline(_BUILTIN)


__all__ = ["REGISTRY", "resolve_pipeline", "default_stages", "Stage"]
