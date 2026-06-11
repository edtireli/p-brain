"""Diagnostic-plotter contract.

A *diagnostic* takes a single ``CurveInputs`` (one tissue curve + the AIF
+ time axis), runs the model's fit on that curve, and renders a model-
specific plot summarising what was fit and how.

Each kinetic model owns its diagnostic — Patlak shows the linear regression
in Patlak coords; Tikhonov shows the L-curve, recovered impulse response,
fit; Gamma shows the parametric h(t) + residue. Drop a new file in
``pbrain/diagnostics/`` that exports ``PLUGIN`` and it gets auto-discovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin
from pbrain.models import CurveInputs, ModelResult


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    """Single curve to diagnose (one tissue region, one parcel, or one voxel)."""

    label: str                              # e.g. "cortical_gm" or "parcel_1003_lateraloccipital"
    c_tissue: np.ndarray                    # (T,) — already averaged within the ROI
    c_input: np.ndarray                     # (T,) — same AIF used by the kinetic stage
    t_s: np.ndarray                         # (T,)
    out_path: Path                          # absolute path the plot must be written to
    model_opts: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Diagnostic(Plugin, Protocol):
    """Model-specific diagnostic-plot generator."""

    model_key: ClassVar[str]                # which kinetic model this diagnoses

    def plot(self, ctx: DiagnosticContext) -> None: ...
