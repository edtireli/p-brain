"""Manual AIF — interactive GUI ROI drawing. STUB.

The legacy GUI ROI drawing widgets live across ``modules/`` (Tkinter +
matplotlib). Porting them into a plug-in is straightforward but bulky
(~400 lines) and benefits from a real display + user testing. Left as a
stub for now — use ``from_file`` after drawing ROIs in the legacy GUI
once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import AIFExtractor, InputFunction


@dataclass(frozen=True, slots=True)
class _ManualAIF:
    key: ClassVar[str] = "manual"
    name: ClassVar[str] = "Manual GUI ROI drawing"
    description: ClassVar[str] = (
        "Interactive ROI drawing on DCE slices via matplotlib widgets. "
        "Currently a stub; use 'from_file' after drawing once with the "
        "legacy GUI."
    )
    accepts: ClassVar[dict[str, type]] = {"dce_data": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"input_function": InputFunction}

    def extract(self, *args: Any, **kwargs: Any) -> InputFunction:
        raise NotImplementedError(
            "Manual GUI AIF drawing is not yet ported. After drawing "
            "ROIs in the legacy GUI, save the mask as NIfTI/.mat and "
            "use the 'from_file' plug-in to consume it."
        )


PLUGIN = _ManualAIF()
