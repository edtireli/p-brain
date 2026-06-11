"""Manual GUI tissue ROI drawing — STUB.

Stub for hand-drawn tissue ROIs via matplotlib widgets. Same status
as ``pbrain/aif/manual.py``: porting is straightforward but bulky and
benefits from a real display. Use ``voxelwise`` for now if no
parcellation backend is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider


@dataclass(frozen=True, slots=True)
class _ManualTissue:
    key: ClassVar[str] = "manual"
    name: ClassVar[str] = "Manual GUI tissue ROI drawing"
    description: ClassVar[str] = (
        "Interactive ROI drawing on T1-weighted slices. Currently a "
        "stub; use 'synthseg' or 'voxelwise' until ported."
    )
    accepts: ClassVar[dict[str, type]] = {"t1w_volume": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"tissue_roi": TissueROI}

    def extract(self, *args: Any, **kwargs: Any) -> TissueROI:
        raise NotImplementedError(
            "Manual tissue ROI drawing is not yet ported. Use 'voxelwise' "
            "(brain mask only) or 'synthseg' (FreeSurfer 8+ DKT parcellation)."
        )


PLUGIN = _ManualTissue()
