"""Input loaders — read raw scanner output into a canonical :class:`Series4D`.

A *loader* takes a filesystem path and returns one Series4D regardless of
whether the source was NIfTI, DICOM, PAR/REC, or any other format
implemented as a plug-in. Downstream stages never inspect the source
format — only the Series4D fields.

The 4th axis is flexible: it may be ``"time"`` (DCE series),
``"ti"`` (inversion recovery), ``"fa"`` (variable flip angle), or
``"static"`` (3D structural volume, with axis4_values = [0]). The
``axis4_kind`` field tells consumers what the values mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


Axis4Kind = Literal["time", "ti", "fa", "static"]


@dataclass(frozen=True, slots=True)
class Series4D:
    """Canonical 4D (or 3D-embedded-as-4D-with-N=1) image container."""

    data: np.ndarray                          # (X, Y, Z, N)
    affine: np.ndarray                        # (4, 4) voxel → world
    voxel_size: tuple[float, float, float]    # mm
    axis4_kind: Axis4Kind                     # "time" | "ti" | "fa" | "static"
    axis4_values: np.ndarray                  # (N,) values along 4th axis
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[-1])

    @property
    def shape3d(self) -> tuple[int, int, int]:
        return tuple(self.data.shape[:3])  # type: ignore[return-value]


@runtime_checkable
class ImageLoader(Plugin, Protocol):
    """A plug-in that reads scanner output into Series4D."""

    extensions: ClassVar[tuple[str, ...]]    # e.g., (".nii", ".nii.gz")

    def detect(self, path: Path) -> bool:
        """True iff this loader can handle ``path``."""
        ...

    def load(self, path: Path, **opts: Any) -> Series4D:
        """Read ``path`` into a Series4D."""
        ...
