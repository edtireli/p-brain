"""AIF / VIF extractor contract — paper §4.4.

An *AIF extractor* takes a DCE 4-D series (+ optional T1/M0 maps, +
optional T1-weighted volume for anatomical guidance) and returns an
:class:`InputFunction`: the concentration-time curve to be used as
input to the kinetic models, plus a 3-D ROI mask declaring which voxels
were averaged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


@dataclass(frozen=True, slots=True)
class InputFunction:
    c_a: np.ndarray                    # (T,) concentration-time curve
    t_s: np.ndarray                    # (T,)
    mask: np.ndarray                   # (X, Y, Z) bool ROI mask
    source: str                        # "rica" | "lica" | "sss" | "user" | ...
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AIFExtractor(Plugin, Protocol):
    """Plugin sub-Protocol for input-function extraction."""

    def extract(
        self,
        dce_data: np.ndarray,          # (X, Y, Z, T) DCE *signal* (CNN expects training distribution)
        t_s: np.ndarray,               # (T,) time points
        dce_affine: np.ndarray,        # (4, 4)
        *,
        t1_map: np.ndarray | None = None,
        m0_map: np.ndarray | None = None,
        t1w_volume: np.ndarray | None = None,
        t1w_affine: np.ndarray | None = None,
        concentration_data: np.ndarray | None = None,  # (X, Y, Z, T) mM volume; if given, curves come from here
        **opts: Any,
    ) -> InputFunction: ...
