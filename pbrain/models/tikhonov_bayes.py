"""Tikhonov — empirical-Bayes λ-selection with per-voxel uncertainty.

A novel variant of the Tikhonov deconvolution that replaces the fragile
L-curve / GCV λ-pickers with **marginal-likelihood (evidence) maximisation**,
and additionally reports a closed-form posterior standard deviation on CBF.

Motivation
----------
On high-SNR DCE-MRI tissue curves the L-curve has no sharp corner and the
GCV functional is monotonic — both collapse to the λ_min grid floor.
Recasting Tikhonov as Bayesian linear
regression dissolves the problem: the MAP estimate
``x̂ = (DᵀD + λ²LᵀL)⁻¹Dᵀb`` is the posterior mean under the prior
``Lx ~ N(0, (σ²/λ²)I)``, and λ is chosen by maximising the evidence
``p(b | λ)`` with σ² marginalised under a Jeffreys prior:

    log Z(λ) = −(N/2)·log(RSS(λ) + λ²·PEN(λ))   ← data misfit
               + rank(L)·log(λ) − ½·log|A(λ)|     ← Occam complexity penalty

The Occam terms grow with λ while the misfit term shrinks, giving a
**guaranteed interior optimum** — no endpoint collapse, no hand-tuned floor.

Uncertainty
-----------
``F = (B[:,k]ᵀ x)/dt`` (peak of the impulse response → CBF) is a linear
functional of the spline coefficients, so its posterior variance is the
closed form ``σ̂²·gᵀA(λ)⁻¹g`` with ``g = B[:,k]/dt`` and
``σ̂² = (RSS + λ²·PEN)/N``. The extra ``cbf_sd`` map answers the question
DCE pipelines never address: *is this CBF (or, by extension, perfusion
deficit) resolvable above the fit noise?*

Outputs: ``cbf, mtt, cth, lambda_opt, cbf_sd``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import CurveInputs, ModelResult
from .tikhonov import PLUGIN as _SMART


@dataclass(frozen=True, slots=True)
class _TikhonovBayes:
    key:         ClassVar[str] = "tikhonov_bayes"
    name:        ClassVar[str] = "Tikhonov — empirical-Bayes λ + uncertainty"
    description: ClassVar[str] = (
        "Evidence-maximisation λ-selection (marginal likelihood, σ² "
        "integrated out) with a guaranteed interior optimum — fixes the "
        "L-curve / GCV endpoint collapse on smooth DCE curves. Adds a "
        "closed-form posterior SD on CBF (cbf_sd)."
    )
    accepts:  ClassVar[dict[str, type]] = dict(_SMART.accepts)
    produces: ClassVar[dict[str, type]] = {
        "cbf": np.ndarray, "mtt": np.ndarray, "cth": np.ndarray,
        "lambda_opt": np.ndarray, "cbf_sd": np.ndarray,
    }
    # Voxelwise default declares cbf_sd only (closed-form, fast). When
    # ``uncertainty_samples`` > 0 the solver also returns mtt_sd / cth_sd —
    # the diagnostic plot turns sampling on for the single ROI curve, where
    # the cost is negligible.
    outputs:  ClassVar[tuple[str, ...]] = ("cbf", "mtt", "cth", "lambda_opt", "cbf_sd")
    units:    ClassVar[dict[str, str]] = {
        "cbf": "mL/100g/min", "mtt": "s", "cth": "s",
        "lambda_opt": "(unitless)", "cbf_sd": "mL/100g/min",
        "mtt_sd": "s", "cth_sd": "s",
    }
    primary_map: ClassVar[str] = "cbf"

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        opts = dict(opts)
        opts.setdefault("lambda_selection", "evidence")
        opts.setdefault("lambda_spacing",   "log")     # dense low-λ sampling
        opts.setdefault("mtt_cth_method",   "residue_integral")
        # Closed-form CBF SD by default (fast). Posterior sampling — which
        # also yields mtt_sd / cth_sd and a jitter-aware cbf_sd — is opt-in via
        # ``--opt models.tikhonov_bayes.uncertainty_samples=N``.
        if int(opts.get("uncertainty_samples", 0)) <= 0:
            opts["compute_cbf_sd"] = True
        result = _SMART.fit(inputs, **opts)
        # If sampling produced mtt_sd / cth_sd, surface them in the contract so
        # the KineticStage validation accepts the extra maps.
        return result

    def review(self, inputs, result, **_kw):
        """--mode verify view — mean tissue Cₜ + the gamma-residue fit + medians."""
        from ._review import curve_fit_review
        return curve_fit_review(self, inputs, result,
                                title="Tikhonov (Bayesian) · residue", fit_label="gamma-residue fit",
                                show=("cbf", "mtt", "cth"))

    def predict(self, maps: dict[str, Any], c_input: np.ndarray,
                t_s: np.ndarray) -> np.ndarray:
        return _SMART.predict(maps, c_input, t_s)


PLUGIN = _TikhonovBayes()
