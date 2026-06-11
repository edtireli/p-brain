"""Legacy p-Brain output layout.

Preserves the directory structure ``main.py``/``enumerator.py`` and the
p-Brain Platform app currently expect::

    <subject>/Analysis/Fitting/        — T1/M0 maps
    <subject>/Analysis/CTC Data/       — concentration-time curves
    <subject>/Analysis/TSCC Data/      — time-shifted concentration curves
    <subject>/Analysis/ITC Data/       — input function curves
    <subject>/Analysis/ROI Data/       — ROI masks + metadata
    <subject>/Images/AI_patlak/        — Patlak figures + maps
    <subject>/Images/AI_tikhonov/      — Tikhonov figures + maps
    <subject>/Images/Fit/              — T1/M0 fit montages

Use this scheme when downstream tooling expects the old paths. For new
work, prefer the ``bids_like`` scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


_LEGACY_STAGE_DIR: dict[str, str] = {
    "load":           "NIfTI",
    "t1_m0":          "Analysis/Fitting",
    "aif":            "Analysis/ITC Data",
    "tissue_roi":     "Analysis/ROI Data",
    "signal_to_conc": "Analysis/CTC Data",
    "normalisation":  "Analysis/TSCC Data",
    "kinetic":        "Analysis/Kinetic",
    "summary":        "Analysis/Summary",
}

_LEGACY_MODEL_IMAGE_DIR: dict[str, str] = {
    "patlak":          "Images/AI_patlak",
    "tikhonov":        "Images/AI_tikhonov",
    "extended_tofts":  "Images/AI_etofts",
    "gamma":           "Images/AI_gamma",
}


@dataclass(frozen=True, slots=True)
class _LegacyScheme:
    key: ClassVar[str] = "legacy"
    name: ClassVar[str] = "Legacy p-Brain layout"
    description: ClassVar[str] = (
        "Original Analysis/* and Images/AI_* directory structure used by "
        "the legacy main.py and the p-Brain Platform app."
    )
    accepts: ClassVar[dict[str, type]] = {}
    produces: ClassVar[dict[str, type]] = {}

    def stage_dir(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str | None = None,
    ) -> Path:
        rel = _LEGACY_STAGE_DIR.get(stage_name, f"Analysis/{stage_name}")
        return subject_dir / rel

    def aggregation_dir(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str,
        aggregation_key: str,
    ) -> Path:
        if stage_name == "kinetic" and plugin_key in _LEGACY_MODEL_IMAGE_DIR:
            return subject_dir / _LEGACY_MODEL_IMAGE_DIR[plugin_key] / aggregation_key
        return self.stage_dir(subject_dir, stage_name, plugin_key) / aggregation_key

    def output_path(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str | None,
        aggregation_key: str | None,
        output_name: str,
        ext: str,
    ) -> Path:
        if aggregation_key:
            base = self.aggregation_dir(
                subject_dir, stage_name, plugin_key or "", aggregation_key
            )
        else:
            base = self.stage_dir(subject_dir, stage_name, plugin_key)
        ext = ext.lstrip(".")
        return base / f"{output_name}.{ext}"

    def manifest_path(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str | None = None,
    ) -> Path:
        return self.stage_dir(subject_dir, stage_name, plugin_key) / "manifest.json"


PLUGIN = _LegacyScheme()
