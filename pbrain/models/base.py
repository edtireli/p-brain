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
    """What every kinetic model returns.

    Parameters
    ----------
    maps
        Output parameter maps keyed by the model's own names (e.g.
        ``{"ki", "vb"}`` for Patlak). Each value is ``(V,)`` for a
        voxel-wise fit or a scalar array for an ROI-mean fit.
    units
        ``{map_name: unit_string}`` for every key in ``maps``.
    aux
        Free-form per-model diagnostics (residuals, fit quality, optimal
        lambda, …). Aggregators may ignore it.
    """

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

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        """Fit the model to one curve or a voxel stack.

        Parameters
        ----------
        inputs
            Tissue/input concentration curves and time axis; see
            :class:`CurveInputs`. ``inputs.c_tissue`` may be ``(T,)``
            (ROI-mean) or ``(T, V)`` (voxel-wise).
        **opts
            Per-plug-in options from
            ``config.plugin_options["models.<key>"]``.

        Returns
        -------
        ModelResult
            ``maps`` keyed by this model's ``outputs``.
        """
        ...


def reconstruct_gamma_residue_ct(
    t_s: np.ndarray, c_input: np.ndarray, *,
    cbf: float, mtt: float, cth: float, e_leak: float = 0.0,
) -> np.ndarray:
    """Parametric QC reconstruction ``Cₜ = (cbf/6000)·(Cₐ ⊗ R)``.

    ``R`` is a gamma-variate residue with mean ``mtt`` and SD ``cth`` (plus an
    optional irreversible-leak plateau ``e_leak``). This is the shared backend
    for the **model-free** Tikhonov / Stieltjes ``predict()`` — they fit a
    non-parametric residue/spectrum whose *exact* overlay lives in their own
    diagnostics, so this gives the uniform ``predict()`` interface a sensible
    parametric curve rather than nothing. ``cbf`` is mL/100g/min (the gamma
    flow convention ``f = cbf/6000``); ``mtt``/``cth`` are seconds.
    """
    from scipy.stats import gamma as _gamma

    t = np.asarray(t_s, dtype=float).ravel()
    ca = np.asarray(c_input, dtype=float).ravel()
    n = int(min(t.size, ca.size))
    if n == 0 or not (np.isfinite(cbf) and np.isfinite(mtt) and mtt > 0):
        return np.full(n, np.nan)
    t, ca = t[:n], ca[:n]
    cth = cth if (np.isfinite(cth) and cth > 0) else 0.5 * mtt
    k = (mtt / cth) ** 2                       # gamma shape
    theta = cth ** 2 / mtt                     # gamma scale (mean k·θ = mtt)
    h = _gamma.pdf(np.maximum(t, 0.0), a=k, scale=theta)
    dt = np.diff(t)
    H = np.concatenate(([0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * dt)))
    R = np.clip(1.0 - H, 0.0, 1.0)
    e_leak = float(np.clip(e_leak, 0.0, 1.0))
    R = (1.0 - e_leak) * R + e_leak            # leak plateau
    dtf = np.concatenate(([dt[0] if dt.size else 1.0], dt))
    f = float(cbf) / 6000.0
    ct = np.empty(n, dtype=float)
    for i in range(n):                         # Cₐ ⊗ R (single QC curve → O(n²) is fine)
        ct[i] = f * float(np.sum(ca[: i + 1] * R[i::-1][: i + 1] * dtf[: i + 1]))
    return ct
