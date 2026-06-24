"""Identity (pass-through) normaliser — the correct default for Patlak.

Returns curves unchanged. The Patlak plugin performs its own per-voxel
baseline detection + shift (``baseline_shift=True``, the legacy
``find_baseline_point_advanced`` → ``custom_shifter`` chain that makes the
voxelwise Ki map 1:1 with ``Ki_per_voxel.nii.gz``). Normalising again here
would **double-shift** the tissue curve, and it would additionally shift the
AIF — which the legacy / article Patlak path keeps **raw** (the saved
TSCC-Max curve is used verbatim, never baseline-shifted).

With ``legacy_shift`` as the normaliser the voxelwise Ki came out ~4.7 % high
(conc scaled +2.4 %, AIF AUC −1.6 % → Ki ×1.04). With identity normalisation
— raw conc + raw AIF into the model's own single shift — the voxelwise Patlak
Ki map reproduces the legacy ``Ki_per_voxel.nii.gz`` at **medratio 1.0000,
corr 0.9994**, and the per-tissue median_curve matches ``dataset.xlsx``.

Use ``--normaliser legacy_shift`` only for a model that expects pre-shifted
curves and does **not** shift internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import CurveNormaliser


@dataclass(frozen=True, slots=True)
class _Identity:
    key: ClassVar[str] = "identity"
    name: ClassVar[str] = "Identity (pass-through) normaliser"
    description: ClassVar[str] = (
        "Returns curves unchanged. Correct default when the kinetic model "
        "does its own baseline shift (Patlak) and the AIF must stay raw "
        "(legacy/article Patlak uses the saved TSCC verbatim)."
    )
    accepts: ClassVar[dict[str, type]] = {"curve": np.ndarray, "t_s": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"curve_normalised": np.ndarray}

    def normalise(
        self,
        curve: np.ndarray,
        t_s: np.ndarray,
        *,
        baseline_frames: int = 5,
        reference_curve: np.ndarray | None = None,
        **opts: Any,
    ) -> np.ndarray:
        return np.asarray(curve, dtype=float)


PLUGIN = _Identity()
