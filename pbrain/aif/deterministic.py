"""Deterministic AIF extractor — DCE-only, no CNN required.

Algorithm (paper §4.4, deterministic fallback):

1. Compute per-voxel **peak amplitude** (max − baseline) over the DCE
   time series. Baseline = mean of the first ``baseline_frames``.
2. Identify the brain mask via Otsu threshold on the time-averaged DCE
   volume.
3. Within the brain mask + the configured z-range (default: superior
   ~30 % of slices for arterial; inferior ~40 % for venous), select the
   top-``n_voxels`` voxels by peak amplitude.
4. Return the mean signal-time curve of those voxels as the AIF.

This is a deliberate simplification of the legacy
``modules/deterministic_input_functions.py`` that captures the same
spirit: peak detection + spatial constraint. For full PCA-based curve
separation between arterial and venous, use the legacy module directly
or contribute a richer plug-in (just drop another file in this folder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import AIFExtractor, InputFunction


def _otsu_threshold(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=256)
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + 0.5 * np.diff(edges)))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b2 = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return float(edges[int(np.argmax(sigma_b2))])


@dataclass(frozen=True, slots=True)
class _DeterministicAIF:
    key: ClassVar[str] = "deterministic"
    name: ClassVar[str] = "Deterministic peak-amplitude AIF"
    description: ClassVar[str] = (
        "DCE-only peak-amplitude AIF: brain mask via Otsu, spatial "
        "constraint via z-range, top-N voxels averaged. Drop-in fallback "
        "when CNN weights are unavailable."
    )
    accepts: ClassVar[dict[str, type]] = {"dce_data": np.ndarray, "t_s": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"input_function": InputFunction}

    def extract(
        self,
        dce_data: np.ndarray,
        t_s: np.ndarray,
        dce_affine: np.ndarray,
        *,
        t1_map: np.ndarray | None = None,
        m0_map: np.ndarray | None = None,
        t1w_volume: np.ndarray | None = None,
        t1w_affine: np.ndarray | None = None,
        source: str = "rica",
        baseline_frames: int = 5,
        n_voxels: int = 32,
        z_range_fraction: tuple[float, float] | None = None,
        **_: Any,
    ) -> InputFunction:
        d = np.asarray(dce_data, dtype=np.float32)
        if d.ndim != 4:
            raise ValueError(f"dce_data must be 4-D; got shape {d.shape}")
        X, Y, Z, T = d.shape

        b = max(1, min(int(baseline_frames), T - 1))
        baseline = d[..., :b].mean(axis=-1)
        peak = d[..., b:].max(axis=-1) - baseline

        mean_signal = d.mean(axis=-1)
        thr = _otsu_threshold(mean_signal.ravel())
        brain_mask = mean_signal > thr

        if z_range_fraction is None:
            # rICA/lICA enter at the skull base → inferior slices (low z);
            # the superior sagittal sinus runs through the vertex → superior
            # slices (high z). Defaults match the paper's anatomical priors.
            if source.lower() in {"sss", "vif", "venous"}:
                z_range_fraction = (0.6, 1.0)
            else:
                z_range_fraction = (0.0, 0.4)
        zlo = int(np.floor(z_range_fraction[0] * Z))
        zhi = int(np.ceil(z_range_fraction[1] * Z))
        z_mask = np.zeros(Z, dtype=bool)
        z_mask[zlo:zhi] = True

        valid = brain_mask & z_mask[None, None, :]

        # Fall back to the whole brain mask if the z-band has too few voxels
        # (small or single-slice acquisitions, off-center FOV).
        if int(valid.sum()) < int(n_voxels):
            valid = brain_mask

        masked_peak = np.where(valid, peak, -np.inf)
        n_take = min(int(n_voxels), int(valid.sum()))
        if n_take <= 0:
            raise RuntimeError(
                f"deterministic AIF: no candidate voxels (brain_mask sum "
                f"{int(brain_mask.sum())}, z-band {z_range_fraction})"
            )

        flat = masked_peak.ravel()
        top_idx = np.argpartition(flat, -n_take)[-n_take:]
        roi = np.zeros_like(valid, dtype=bool)
        # `.flat` writes through the underlying buffer; `.ravel()` may return
        # a *copy* when the source is not C-contiguous (which happens for the
        # result of `brain_mask & z_mask[None,None,:]`).
        roi.flat[top_idx] = True

        ca_signal = d[roi].mean(axis=0)  # (T,)

        return InputFunction(
            c_a=ca_signal,
            t_s=np.asarray(t_s, dtype=float),
            mask=roi,
            source=source,
            meta={
                "algorithm": "deterministic_peak",
                "n_voxels": int(roi.sum()),
                "z_range_fraction": list(z_range_fraction),
                "otsu_threshold": float(thr),
            },
        )


PLUGIN = _DeterministicAIF()
