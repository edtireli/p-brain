"""SynthSeg tissue ROI provider — subprocess wrapper for FreeSurfer 8+.

Invokes ``mri_synthseg --parc --fast`` on the T1-weighted volume and
parses the DKT-cortical parcellation (~98 labels) it writes. Mirrors
the paper's default segmentation backend.

Requires ``mri_synthseg`` on ``PATH`` (ships with FreeSurfer ≥ 8.0).
If absent, raises ``RuntimeError`` with an install pointer.

Per-label region grouping (GM/WM/Cerebellum/Brainstem) is computed from
the standard FreeSurfer labelmap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider


# Standard FreeSurfer region groupings (paper §4.4)
_REGION_MAP_FS = {
    "cortical_GM": list(range(1000, 1036)) + list(range(2000, 2036)),
    "subcortical_GM": [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58],
    "WM": [2, 41, 77, 251, 252, 253, 254, 255],
    "Cerebellum": [7, 8, 46, 47],
    "Brainstem": [16],
}


def _read_freesurfer_lut() -> dict[int, str]:
    """Minimal FreeSurfer label LUT for region naming. Sourced from
    FreeSurferColorLUT.txt; here we ship the subset p-Brain needs."""
    return {
        2:  "Left-Cerebral-White-Matter",
        7:  "Left-Cerebellum-White-Matter",
        8:  "Left-Cerebellum-Cortex",
        10: "Left-Thalamus-Proper",
        11: "Left-Caudate",
        12: "Left-Putamen",
        13: "Left-Pallidum",
        16: "Brain-Stem",
        17: "Left-Hippocampus",
        18: "Left-Amygdala",
        26: "Left-Accumbens-area",
        41: "Right-Cerebral-White-Matter",
        46: "Right-Cerebellum-White-Matter",
        47: "Right-Cerebellum-Cortex",
        49: "Right-Thalamus-Proper",
        50: "Right-Caudate",
        51: "Right-Putamen",
        52: "Right-Pallidum",
        53: "Right-Hippocampus",
        54: "Right-Amygdala",
        58: "Right-Accumbens-area",
        77: "WM-hypointensities",
    }


@dataclass(frozen=True, slots=True)
class _SynthSegTissue:
    key: ClassVar[str] = "synthseg"
    name: ClassVar[str] = "SynthSeg (FreeSurfer 8+)"
    description: ClassVar[str] = (
        "Subprocess wrapper for `mri_synthseg --parc --fast`. ~98-label "
        "DKT cortical parcellation; ~2 min on CPU. The paper's default."
    )
    accepts: ClassVar[dict[str, type]] = {"t1w_volume": np.ndarray, "t1w_affine": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"tissue_roi": TissueROI}

    def extract(
        self,
        t1w_volume: np.ndarray,
        t1w_affine: np.ndarray,
        *,
        out_dir: Path,
        target_affine: np.ndarray | None = None,
        target_shape: tuple[int, int, int] | None = None,
        robust: bool = False,
        **_: Any,
    ) -> TissueROI:
        import nibabel as nib

        binary = shutil.which("mri_synthseg")
        if binary is None:
            raise RuntimeError(
                "mri_synthseg not on PATH. Install FreeSurfer 8.0+ and run "
                "`source $FREESURFER_HOME/SetUpFreeSurfer.sh`."
            )

        # mri_synthseg needs FREESURFER_HOME set. Auto-derive it from the
        # binary location (…/<FREESURFER_HOME>/bin/mri_synthseg) when the env
        # isn't already sourced, so a fresh shell just works.
        env = dict(os.environ)
        if not env.get("FREESURFER_HOME"):
            fs_home = Path(binary).resolve().parent.parent
            if (fs_home / "SetUpFreeSurfer.sh").exists():
                env["FREESURFER_HOME"] = str(fs_home)
                env.setdefault("SUBJECTS_DIR", str(fs_home / "subjects"))

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t1_path = out_dir / "_t1_in.nii.gz"
        parc_path = out_dir / "parcellation.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(t1w_volume, dtype=np.float32),
                                  np.asarray(t1w_affine, dtype=float)), str(t1_path))

        cmd = [binary, "--i", str(t1_path), "--o", str(parc_path),
               "--parc", "--cpu"]
        if not robust:
            cmd.append("--fast")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"mri_synthseg failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        parc_img = nib.load(str(parc_path))
        parc = np.asarray(parc_img.dataobj, dtype=np.int32)
        present_labels = sorted(int(x) for x in np.unique(parc) if int(x) > 0)
        lut = _read_freesurfer_lut()
        labels = {lab: lut.get(lab, f"label_{lab}") for lab in present_labels}

        return TissueROI(
            parcellation=parc,
            affine=np.asarray(parc_img.affine, dtype=float),
            labels=labels,
            region_map={k: v for k, v in _REGION_MAP_FS.items()},
            meta={
                "algorithm": "synthseg" + ("_robust" if robust else "_fast"),
                "n_labels": len(labels),
                "parc_path": str(parc_path),
            },
        )


PLUGIN = _SynthSegTissue()
