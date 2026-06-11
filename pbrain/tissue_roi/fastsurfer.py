"""FastSurfer tissue ROI provider — STUB.

Scaffold for the FastSurfer deep-learning cortical parcellation backend
documented in paper §4.4. Requires a local FastSurfer install
(https://github.com/deep-mi/FastSurfer). Stub raises ``NotImplementedError``
until ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider


@dataclass(frozen=True, slots=True)
class _FastSurferTissue:
    key: ClassVar[str] = "fastsurfer"
    name: ClassVar[str] = "FastSurfer (deep-learning DKT)"
    description: ClassVar[str] = (
        "Subprocess wrapper for FastSurfer's cortical parcellation. "
        "Currently a stub; use 'synthseg' or 'voxelwise' until ported."
    )
    accepts: ClassVar[dict[str, type]] = {"t1w_volume": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"tissue_roi": TissueROI}

    def extract(self, *args: Any, **kwargs: Any) -> TissueROI:
        raise NotImplementedError(
            "FastSurfer provider is not yet ported. Use 'synthseg' "
            "(FreeSurfer 8+) or 'voxelwise' (no parcellation)."
        )


PLUGIN = _FastSurferTissue()
