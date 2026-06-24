"""CNN AIF: SSS back-shifted onto the rICA arterial input.

Composes the ``cnn`` extractor twice — rICA (artery) and SSS (sagittal sinus,
vein) — then shifts the venous SSS curve onto the arterial bolus timing:

1. rICA CNN → arterial curve.
2. SSS CNN → venous curve.
3. Back-shift the SSS onto the rICA bolus timing: find the top ``num_peaks``
   bolus peaks in each curve (iterative max + radius exclusion), align the SSS
   peaks to the rICA peaks (rICA = targets), and advance the SSS by the median
   offset with PCHIP sub-frame interpolation, bounded to ``max_shift_s``.
4. Optionally rescale the shifted SSS up to the rICA peak amplitude (never down).

The returned :class:`InputFunction` carries the shifted SSS curve as ``c_a`` and
the SSS mask; ``meta`` records the applied shift (seconds) and rescale factor.
``--aif cnn`` uses the rICA directly without the shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ._backshift import backshift_sss_to_rica
from .base import AIFExtractor, InputFunction
from .cnn import PLUGIN as _CNN


# ────────────────────────────────────────────────────────────────────────
# Plug-in
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _CnnSssShiftedAIF:
    key: ClassVar[str] = "cnn_sss_shifted"
    name: ClassVar[str] = "CNN SSS back-shifted to rICA"
    description: ClassVar[str] = (
        "Runs the CNN twice (rICA + SSS), back-shifts the SSS curve onto the "
        "rICA bolus timing by aligning the top num_peaks bolus peaks + PCHIP "
        "sub-frame interpolation, optionally rescaling SSS amplitude to the rICA peak. "
        "The shifted SSS is the AIF used by downstream Patlak. "
        "Use --aif cnn for rICA-direct."
    )
    accepts: ClassVar[dict[str, type]] = {
        "dce_data": np.ndarray, "t_s": np.ndarray,
    }
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
        concentration_data: np.ndarray | None = None,
        num_peaks: int = 2,
        peak_separation: int = 10,
        baseline_frames: int = 5,
        # Keep the SSS curve at its own venous amplitude by default; rescale=True
        # lifts it to the rICA arterial peak.
        rescale: bool = False,
        rescale_method: str = "peak",
        max_shift_s: float = 20.0,
        rot90_k: int = -1,
        curve_method: str = "max_voxel",
        rica_opts: dict | None = None,
        sss_opts: dict | None = None,
        **_: Any,
    ) -> InputFunction:
        rica_opts = dict(rica_opts or {})
        sss_opts = dict(sss_opts or {})
        rica_opts.setdefault("rot90_k", rot90_k)
        sss_opts.setdefault("rot90_k", rot90_k)
        # SSS voxel-aggregation: "max_voxel" (single brightest voxel, default,
        # stable) or "adaptive_max" (brightest voxel per frame — the legacy
        # "adaptive ROI" bolus curve; reproduces SSS-selected subjects but is
        # noisier). Selectable so the legacy analysis is reachable by a flag.
        sss_opts.setdefault("curve_method", curve_method)

        # Both passes see the signal (for CNN) AND the mM volume
        # (for curve extraction). The returned curves are therefore
        # in mM with per-voxel T1/M0 already applied.
        ica = _CNN.extract(dce_data, t_s, dce_affine,
                            t1_map=t1_map, m0_map=m0_map,
                            concentration_data=concentration_data,
                            source="rica", **rica_opts)
        sss = _CNN.extract(dce_data, t_s, dce_affine,
                            t1_map=t1_map, m0_map=m0_map,
                            concentration_data=concentration_data,
                            source="sss", **sss_opts)

        aligned, shift_s, factor = backshift_sss_to_rica(
            sss.c_a, ica.c_a, t_s, num_peaks=num_peaks, peak_radius=peak_separation,
            rescale=rescale, max_shift_s=max_shift_s,
        )

        return InputFunction(
            c_a=aligned,
            t_s=np.asarray(t_s, dtype=float),
            mask=sss.mask,                            # SSS anatomical ROI (mM curve)
            source="sss_shifted_to_rica",
            meta={
                "algorithm": "cnn_sss_shifted_to_rica_gamma_pchip",
                "shift_seconds": float(shift_s),
                "rescale_factor": float(factor),
                "rica_meta": ica.meta,
                "sss_meta": sss.meta,
                "rica_mask_voxels": int(ica.mask.sum()),
                "sss_mask_voxels": int(sss.mask.sum()),
            },
        )


PLUGIN = _CnnSssShiftedAIF()
