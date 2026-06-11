"""BIDS-flavoured derivatives layout (default scheme).

Layout::

    <subject_dir>/derivatives/
        01_load/
            manifest.json
            dce.nii.gz
        02_t1m0/
            inversion_recovery/
                manifest.json
                t1_map.nii.gz
                m0_map.nii.gz
        03_aif/
            deterministic/
                manifest.json
                aif.json
        04_tissue_roi/
            synthseg/
                manifest.json
                parcellation.nii.gz
                labels.json
        05_signal_to_conc/
            saturation_recovery/
                manifest.json
                ct.nii.gz
        06_normalisation/
            baseline_p95/
                manifest.json
        07_kinetic/
            patlak/
                voxelwise/   {ki.nii.gz, vb.nii.gz, manifest.json}
                parcel/      {ki.csv, vb.csv, manifest.json}
                region/      {ki.json, vb.json, manifest.json}
                slice_wise/  {ki_slice.csv, manifest.json}
            tikhonov/
                voxelwise/   {cbf.nii.gz, mtt.nii.gz, cth.nii.gz, lambda_opt.nii.gz, manifest.json}
                parcel/      {cbf.csv, mtt.csv, cth.csv, manifest.json}
                region/      {cbf.json, mtt.json, cth.json, manifest.json}
                slice_wise/  ...
            gamma/           # appears when models/gamma.py is on disk
                ...

The stage number prefix gives natural sort order in a file explorer.
``plugin_key`` is the directory layer under each stage; when omitted
(stages without alternative implementations, like ``01_load``) it sits
directly under the stage directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


_STAGE_PREFIX: dict[str, str] = {
    "load":            "01_load",
    "t1_m0":           "02_t1m0",
    "aif":             "03_aif",
    "tissue_roi":      "04_tissue_roi",
    "signal_to_conc":  "05_signal_to_conc",
    "normalisation":   "06_normalisation",
    "kinetic":         "07_kinetic",
    "summary":         "08_summary",
    "diffusion":       "09_diffusion",
    "diagnostics":     "00_diagnostics",
}


def _stage_token(stage_name: str) -> str:
    return _STAGE_PREFIX.get(stage_name, stage_name)


@dataclass(frozen=True, slots=True)
class _BidsLikeScheme:
    key: ClassVar[str] = "bids_like"
    name: ClassVar[str] = "BIDS-like derivatives layout"
    description: ClassVar[str] = (
        "Numbered stage subdirectories under <subject>/derivatives/ with "
        "per-plugin and per-aggregation subfolders. The default scheme."
    )
    accepts: ClassVar[dict[str, type]] = {}
    produces: ClassVar[dict[str, type]] = {}

    def stage_dir(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str | None = None,
    ) -> Path:
        d = subject_dir / "derivatives" / _stage_token(stage_name)
        if plugin_key:
            d = d / plugin_key
        return d

    def aggregation_dir(
        self,
        subject_dir: Path,
        stage_name: str,
        plugin_key: str,
        aggregation_key: str,
    ) -> Path:
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
            base = self.aggregation_dir(subject_dir, stage_name, plugin_key or "", aggregation_key)
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


PLUGIN = _BidsLikeScheme()
