"""Preloaded T1/M0 fitter — read precomputed maps from disk.

Useful when validating downstream stages against a cohort that already
has T1/M0 maps cached (e.g., from a prior pipeline run, or from a
different fitter altogether). Skips the slow voxelwise NLS step.

Configuration::

    --t1m0 preloaded \\
    --opt t1_m0.preloaded.t1_ms_path=/path/to/t1_map.nii.gz \\
    --opt t1_m0.preloaded.m0_path=/path/to/m0_map.nii.gz \\
    --opt t1_m0.preloaded.t1_units=ms       # or "s"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import T1M0Fitter, T1M0Result


@dataclass(frozen=True, slots=True)
class _PreloadedT1M0:
    key: ClassVar[str] = "preloaded"
    name: ClassVar[str] = "Preloaded T1/M0 maps"
    description: ClassVar[str] = (
        "Reads precomputed T1 and M0 NIfTI volumes from disk. Skips "
        "any voxelwise fit. Use when validating downstream stages "
        "against cached maps from a different fitter."
    )
    accepts: ClassVar[dict[str, type]] = {
        "t1_ms_path": Path,
        "m0_path": Path,
    }
    produces: ClassVar[dict[str, type]] = {"t1_ms": np.ndarray, "m0": np.ndarray}

    def fit(
        self,
        signals: np.ndarray | None = None,
        axis_values: np.ndarray | None = None,
        *,
        mask: np.ndarray | None = None,
        t1_ms_path: Path | str | None = None,
        m0_path: Path | str | None = None,
        t1_units: str = "ms",
        **_: Any,
    ) -> T1M0Result:
        if t1_ms_path is None or m0_path is None:
            raise ValueError(
                "preloaded T1/M0 requires opts t1_m0.preloaded.t1_ms_path "
                "and t1_m0.preloaded.m0_path"
            )
        import nibabel as nib

        t1_img = nib.load(str(t1_ms_path))
        m0_img = nib.load(str(m0_path))
        t1 = np.asarray(t1_img.dataobj, dtype=float)
        m0 = np.asarray(m0_img.dataobj, dtype=float)
        if str(t1_units).lower() in ("s", "sec", "seconds"):
            t1 = t1 * 1000.0
        return T1M0Result(
            t1_map_ms=t1,
            m0_map=m0,
            meta={
                "fitter": "preloaded",
                "t1_path": str(t1_ms_path),
                "m0_path": str(m0_path),
                "t1_units_input": t1_units,
            },
        )


PLUGIN = _PreloadedT1M0()
