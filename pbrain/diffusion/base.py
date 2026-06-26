"""Diffusion-model contract.

Mirrors ``pbrain/models/base.py`` (kinetic models) one-to-one but for
q-space (DWI) data. Every diffusion plug-in in ``pbrain/diffusion/``
implements :class:`DiffusionModel`.

* **Inputs** travel via :class:`DWIInputs` — a 4-D DWI signal volume,
  bvals (s/mm²), bvecs (unit vectors), affine, and an optional brain
  mask. Multi-shell data is fine — each plug-in declares the shell
  requirements it needs.
* **Outputs** travel via :class:`DiffusionResult.maps` — a dict whose
  keys are *the plug-in's choice* (DTI: ``{"fa","md","ad","rd","colorfa"}``;
  DKI: ``{"mk","ak","rk","kfa",…}``; CSD: ``{"gfa","nufo",…}``).
  Aggregators consume this dict generically — exactly the same code
  path used by kinetic-model outputs.
* **Units** travel alongside in :class:`DiffusionResult.units`.
* **Aux** is free-form (response function, fODF SH coefficients,
  peaks, …). Aggregators may ignore.

Optional attributes / methods (`getattr` lookup, sensible defaults):

* ``primary_map: ClassVar[str]`` — output map used by the diagnostic
  stage to pick a representative voxel. Defaults to ``outputs[0]``.
* ``diagnose(self, ctx) -> None`` — single-file diagnostic; same
  signature as :class:`pbrain.diagnostics.Diagnostic.plot`. Pipeline
  falls back to ``pbrain/diagnostics/<key>.py`` then to a generic
  histogram-plus-axial-slice fallback.
* ``min_shells: int`` / ``min_directions: int`` — declare data
  requirements. The DiffusionStage skips plug-ins whose requirements
  aren't met (e.g. DKI on single-shell data) with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


@dataclass(frozen=True, slots=True)
class DWIInputs:
    """Inputs a diffusion model receives.

    Parameters
    ----------
    signal
        4-D DWI signal volume ``(X, Y, Z, N)`` where ``N`` is the
        number of acquired directions (b=0 volumes counted).
    bvals
        b-values in s/mm². Shape ``(N,)``.
    bvecs
        Unit gradient directions. Shape ``(N, 3)``.
    affine
        ``(4, 4)`` voxel-to-world affine matrix of the DWI volume.
    mask
        Optional brain mask ``(X, Y, Z)`` bool. ``True`` selects voxels
        to fit; others receive NaN outputs. If absent, a percentile-
        based b=0 brain mask is auto-derived.
    """

    signal: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    affine: np.ndarray
    mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    """What every diffusion model returns.

    Same shape-contract as :class:`pbrain.models.ModelResult` so the
    aggregation stage can consume kinetic and diffusion outputs
    through one code path.
    """

    maps: dict[str, np.ndarray]
    units: dict[str, str]
    aux: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DiffusionModel(Plugin, Protocol):
    """Plug-in sub-Protocol every diffusion model implements."""

    outputs: ClassVar[tuple[str, ...]]
    units: ClassVar[dict[str, str]]

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        """Fit the diffusion model to a DWI volume.

        ``inputs`` carries the 4-D signal, bvals/bvecs, affine and an
        optional brain mask (see :class:`DWIInputs`). ``**opts`` are the
        per-plug-in options from
        ``config.plugin_options["diffusion.<key>"]``. Returns a
        :class:`DiffusionResult` whose ``maps`` match this model's
        ``outputs``.
        """
        ...
