"""Kinetic-model contract.

Every kinetic model in ``pbrain/models/`` (Patlak, Tikhonov, Extended
Tofts, gamma, future additions) implements :class:`KineticModel`.

* **Inputs** travel via :class:`CurveInputs` — concentration-time curves
  for tissue and the input function, plus the time axis. Tissue may be
  ``(T,)`` for an ROI-mean fit or ``(T, V)`` for voxel-wise fits.
* **Outputs** travel via :class:`ModelResult.maps` — a dict whose keys
  are *the model's choice* (Patlak: ``{"ki","vb"}``; Tikhonov:
  ``{"cbf","mtt","cth","lambda_opt"}``; gamma: its own set). Aggregators
  consume this dict generically — the framework never hard-codes which
  parameter names exist.
* **Units** travel alongside in :class:`ModelResult.units`.
* **Aux** is free-form per-model diagnostics (residuals, fit quality,
  optimal lambda, Patlak coordinates, …). Aggregators may ignore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


@dataclass(frozen=True, slots=True)
class CurveInputs:
    """Inputs a kinetic model receives.

    Parameters
    ----------
    c_tissue
        Tissue concentration. Shape ``(T,)`` for an ROI-mean fit, or
        ``(T, V)`` for voxel-wise (one column per voxel).
    c_input
        Input function concentration. Shape ``(T,)``.
    t_s
        Time axis in seconds. Shape ``(T,)``.
    mask
        Optional per-voxel boolean mask (``(V,)``). ``True`` selects
        voxels to fit; others receive NaN outputs.
    """

    c_tissue: np.ndarray
    c_input: np.ndarray
    t_s: np.ndarray
    mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class ModelResult:
    """What every kinetic model returns."""

    maps: dict[str, np.ndarray]
    units: dict[str, str]
    aux: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class KineticModel(Plugin, Protocol):
    """Plugin sub-Protocol every kinetic model implements.

    **Required attributes** (inherited from :class:`pbrain.core.Plugin`
    plus the two below):

    * ``key, name, description, accepts, produces`` — see :class:`Plugin`.
    * ``outputs`` — tuple of map names this model produces. Aggregation
      and diagnostic stages iterate these; ``fit()`` must return
      ``ModelResult.maps`` with exactly this key set (validated at
      runtime in ``KineticStage``).
    * ``units`` — ``{output_name: unit_string}``.

    **Required method**:

    * ``fit(inputs, **opts) -> ModelResult``.

    **Optional attributes / methods** — the framework looks for these
    via ``getattr``; absence triggers sensible defaults:

    * ``primary_map: ClassVar[str]`` — name of the output map the
      diagnostic stage should use when picking a "representative voxel"
      for the per-model voxel diagnostic. Defaults to ``outputs[0]``.
    * ``diagnose(self, ctx) -> None`` — single-file diagnostic. If
      defined, the diagnostic stage calls this in preference to looking
      up ``pbrain/diagnostics/<key>.py``. Same signature as
      ``Diagnostic.plot``.
    * ``predict(maps, c_input, t_s) -> np.ndarray`` — used by the
      generic fallback diagnostic to overlay a fitted-Cₜ curve. ``maps``
      is a dict of scalar parameter values for one curve.
    """

    outputs: ClassVar[tuple[str, ...]]
    units: ClassVar[dict[str, str]]

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult: ...
