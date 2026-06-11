"""Bring-your-own segmentation — run any external tool from ONE place.

This is the easy switch for using a different segmentation backend (FastSurfer,
SynthSeg, FSL FIRST, a custom model, …) without writing a plug-in. Select it
with ``--tissue-roi command`` and give the command in a single option:

    --opt tissue_roi.command.cmd="mri_synthseg --i {input} --o {output} --parc --cpu"

or, in a config file::

    [pipeline]
    tissue_roi = "command"
    [options]
    "tissue_roi.command.cmd" = "fastsurfer.sh --t1 {input} --seg {output} --seg_only"

The command must read the T1 NIfTI written to ``{input}`` and write a label
volume to ``{output}``. Labels are grouped into GM / WM / Cerebellum /
Brainstem with the standard FreeSurfer map; if your tool uses a different
labelling, override it with::

    --opt tissue_roi.command.region_map='{"WM":[2,41],"cortical_GM":[1000,2000]}'
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider
from .synthseg import _REGION_MAP_FS, _read_freesurfer_lut


@dataclass(frozen=True, slots=True)
class _CommandTissue:
    key:         ClassVar[str] = "command"
    name:        ClassVar[str] = "External segmentation command"
    description: ClassVar[str] = (
        "Run any user-supplied segmentation tool. Give the command via "
        "--opt tissue_roi.command.cmd=\"... {input} ... {output} ...\" "
        "(one place). Output must be a label volume; FreeSurfer-style labels "
        "are grouped automatically (override with .region_map)."
    )
    accepts: ClassVar[dict[str, type]] = {"t1w_volume": np.ndarray, "t1w_affine": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"tissue_roi": TissueROI}

    def extract(
        self,
        t1w_volume: np.ndarray,
        t1w_affine: np.ndarray,
        *,
        out_dir: Path,
        cmd: str | None = None,
        region_map: dict | str | None = None,
        timeout_s: int = 3600,
        **_: Any,
    ) -> TissueROI:
        import nibabel as nib

        if not cmd:
            raise RuntimeError(
                "tissue_roi.command.cmd is not set. Supply your segmentation "
                "command with {input} and {output} placeholders, e.g. "
                '--opt tissue_roi.command.cmd="mytool {input} {output}".'
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t1_path = out_dir / "_t1_in.nii.gz"
        parc_path = out_dir / "parcellation.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(t1w_volume, dtype=np.float32),
                                  np.asarray(t1w_affine, dtype=float)), str(t1_path))

        filled = cmd.format(input=str(t1_path), output=str(parc_path))
        result = subprocess.run(shlex.split(filled), capture_output=True,
                                text=True, timeout=timeout_s)
        if result.returncode != 0:
            raise RuntimeError(
                f"segmentation command failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}\n  cmd: {filled}"
            )
        if not parc_path.exists():
            raise RuntimeError(
                f"segmentation command did not write {parc_path}. Ensure the "
                f"command writes its label volume to the {{output}} path."
            )

        parc_img = nib.load(str(parc_path))
        parc = np.asarray(parc_img.dataobj, dtype=np.int32)
        present = sorted(int(x) for x in np.unique(parc) if int(x) > 0)
        lut = _read_freesurfer_lut()
        labels = {lab: lut.get(lab, f"label_{lab}") for lab in present}

        if isinstance(region_map, str):
            region_map = json.loads(region_map)
        rmap = region_map or {k: v for k, v in _REGION_MAP_FS.items()}

        return TissueROI(
            parcellation=parc,
            affine=np.asarray(parc_img.affine, dtype=float),
            labels=labels,
            region_map=rmap,
            meta={"algorithm": "external_command", "command": filled,
                  "n_labels": len(labels), "parc_path": str(parc_path)},
        )


PLUGIN = _CommandTissue()
