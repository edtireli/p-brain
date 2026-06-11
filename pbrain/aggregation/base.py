"""Aggregator contract — the analysis-level plug-point.

The paper produces three (or four with slice-wise) reports per model:
voxelwise NIfTI maps, parcel-mean tables, region-mean summaries, and
slice-wise distributions. Each is a separate Aggregator plug-in. The
orchestrator runs ``{models} × {enabled aggregators}``, so adding a new
aggregator means dropping one file in ``pbrain/aggregation/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


@dataclass(frozen=True, slots=True)
class AggregationResult:
    files: dict[str, Path]                       # canonical artefact name → path
    summary: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Aggregator(Plugin, Protocol):
    """Plugin sub-Protocol for analysis-level reductions."""

    def aggregate(
        self,
        maps: dict[str, np.ndarray],             # model outputs by name
        units: dict[str, str],                   # unit string per map
        *,
        reference_affine: np.ndarray | None,     # for NIfTI writers
        parcellation: np.ndarray | None,         # label volume (X, Y, Z)
        labels: dict[int, str] | None,           # label → name
        region_map: dict[str, list[int]] | None, # region → list of label ids
        slice_axis: int = 2,                     # superior-inferior axis (0/1/2)
        out_dir: Path = Path("."),
        **opts: Any,
    ) -> AggregationResult: ...
