"""Baseline-subtract + 95th-percentile rescale normaliser — paper §4.5.1.

Two-step normalisation that the paper uses by default:

1. Subtract the mean of the first ``baseline_frames`` frames (zero out
   the pre-bolus baseline).
2. Divide by the 95th percentile of ``|values|`` (outlier-robust scale
   normalisation).

When a ``reference_curve`` is supplied, the p95 is computed from the
reference instead of the input. This is important for downstream Patlak
fitting, where tissue and AIF must share a common scale factor to keep
the Patlak slope physically meaningful (Ki = Δy/Δx is otherwise
multiplied by p95(Ca)/p95(Ct), which differ by ~10× in normal brain).
The NormalisationStage passes the AIF as the reference when normalising
tissue curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import CurveNormaliser


@dataclass(frozen=True, slots=True)
class _BaselineP95:
    key: ClassVar[str] = "baseline_p95"
    name: ClassVar[str] = "Baseline subtract + p95 rescale"
    description: ClassVar[str] = (
        "Subtract the mean of the first B frames (default 5) then divide "
        "by the 95th percentile of |values| (outlier-robust scale "
        "normalisation, paper §4.5.1)."
    )
    accepts: ClassVar[dict[str, type]] = {
        "curve": np.ndarray,
        "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {"curve_normalised": np.ndarray}

    def normalise(
        self,
        curve: np.ndarray,
        t_s: np.ndarray,
        *,
        baseline_frames: int = 5,
        percentile: float = 95.0,
        reference_curve: np.ndarray | None = None,
        **_: Any,
    ) -> np.ndarray:
        c = np.asarray(curve, dtype=float)
        b = max(1, int(min(baseline_frames, c.shape[0])))

        # Compute the rescale factor from the reference when given, else
        # from the curve itself (per-column for 2-D input).
        if reference_curve is not None:
            ref = np.asarray(reference_curve, dtype=float).ravel()
            ref_b = max(1, int(min(baseline_frames, ref.size)))
            ref_corrected = ref - float(np.mean(ref[:ref_b]))
            shared_scale = float(np.percentile(np.abs(ref_corrected), percentile))
            shared_scale = shared_scale if shared_scale > 0 else 1.0

            if c.ndim == 1:
                corrected = c - float(np.mean(c[:b]))
                return corrected / shared_scale
            if c.ndim == 2:
                baseline = np.mean(c[:b, :], axis=0)
                corrected = c - baseline[None, :]
                return corrected / shared_scale
            raise ValueError(f"curve must be 1-D or 2-D; got shape {c.shape}")

        # No reference: per-curve normalisation (use with care for Patlak).
        if c.ndim == 1:
            baseline = float(np.mean(c[:b]))
            corrected = c - baseline
            scale = float(np.percentile(np.abs(corrected), percentile))
            return corrected / scale if scale > 0 else corrected
        if c.ndim == 2:
            baseline = np.mean(c[:b, :], axis=0)
            corrected = c - baseline[None, :]
            scale = np.percentile(np.abs(corrected), percentile, axis=0)
            scale_safe = np.where(scale > 0, scale, 1.0)
            return corrected / scale_safe[None, :]
        raise ValueError(f"curve must be 1-D or 2-D; got shape {c.shape}")


PLUGIN = _BaselineP95()
