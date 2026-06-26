"""Tissue ROI provider contract — paper §4.4 (tissue ROIs).

A provider takes a T1-weighted structural volume and returns a parcellation:
an integer label volume plus the label → name dictionary. Optionally
ships a region_map collapsing parcels into broader regions (GM, WM,
Cerebellum, Brainstem, GM/WM-boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from pbrain.core import Plugin


@dataclass(frozen=True, slots=True)
class TissueROI:
    """A parcellation returned by a tissue-ROI provider.

    Attributes
    ----------
    parcellation
        ``(X, Y, Z)`` integer label volume.
    affine
        ``(4, 4)`` voxel-to-world affine of the parcellation grid.
    labels
        ``{label_id: name}``.
    region_map
        Optional ``{region_name: [label_id, …]}`` collapsing parcels
        into broader regions (GM, WM, Cerebellum, Brainstem, …).
    meta
        Free-form provenance (backend, version, resampling, …).
    """

    parcellation: np.ndarray                 # (X, Y, Z) int label volume
    affine: np.ndarray                       # (4, 4)
    labels: dict[int, str]                   # label id → name
    region_map: dict[str, list[int]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TissueROIProvider(Plugin, Protocol):
    """Plugin sub-Protocol for tissue parcellation."""

    def extract(
        self,
        t1w_volume: np.ndarray,
        t1w_affine: np.ndarray,
        *,
        out_dir: Path,
        target_affine: np.ndarray | None = None,   # DCE space, for resampling
        target_shape: tuple[int, int, int] | None = None,
        **opts: Any,
    ) -> TissueROI:
        """Parcellate a T1-weighted volume.

        Writes intermediate files under ``out_dir``. If
        ``target_affine``/``target_shape`` are given (DCE space), the
        label volume is resampled onto that grid. Returns a
        :class:`TissueROI`.
        """
        ...
