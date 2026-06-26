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

try:
    import fcntl
except ImportError:        # Windows has no fcntl; SynthSeg is POSIX-only (needs FreeSurfer) anyway
    fcntl = None
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import TissueROI, TissueROIProvider


# ── Cross-process SynthSeg concurrency cap ──────────────────────────────
# SynthSeg loads a UNet + the full-res T1 + intermediates (~3-5 GB) and, with
# --threads, saturates the CPU. When a cohort runs subjects in parallel, several
# SynthSeg processes at once will oversubscribe the cores and can OOM-crash the
# machine. This file-lock semaphore bounds the number of *concurrent* SynthSeg
# runs across all processes, independent of the worker count — so the light,
# fast stages (T1 VARPRO, kinetic, normalisation) can still run many-parallel
# while SynthSeg stays within a memory/CPU-safe limit.
_SYNTHSEG_SLOTDIR = Path(tempfile.gettempdir()) / "pbrain_synthseg_slots"


def _acquire_synthseg_slot(limit: int, poll: float = 3.0):
    """Block until one of ``limit`` slots is free; return the held lock handle."""
    _SYNTHSEG_SLOTDIR.mkdir(parents=True, exist_ok=True)
    if fcntl is None:      # no POSIX flock (e.g. Windows); SynthSeg does not run here anyway
        return open(_SYNTHSEG_SLOTDIR / "slot_0.lock", "w")
    limit = max(1, int(limit))
    while True:
        for i in range(limit):
            fh = open(_SYNTHSEG_SLOTDIR / f"slot_{i}.lock", "w")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fh
            except OSError:
                fh.close()
        time.sleep(poll)


def _release_synthseg_slot(fh) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    except Exception:        # noqa: BLE001
        pass


# Standard FreeSurfer region groupings (paper §4.4)
_REGION_MAP_FS = {
    # 3/42 = aseg cerebral cortex — fallback for SynthSeg runs WITHOUT --parc
    # (parcellated cortex uses the 1000/2000 DKT ranges; mutually exclusive, so
    # listing both is safe and stops cortex being silently dropped).
    "cortical_GM": [3, 42] + list(range(1000, 1036)) + list(range(2000, 2036)),
    # 28/60 = VentralDC (ventral diencephalon: hypothalamus, subthalamic nucleus,
    # substantia nigra, red nucleus, mammillary bodies) — deep GM, previously
    # omitted so it never entered the region/tissue maps or montages. NB: CSF /
    # ventricle labels (4,5,14,15,24,43,44) are intentionally NOT grouped.
    "subcortical_GM": [10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60],
    "WM": [2, 41, 77, 251, 252, 253, 254, 255],
    "Cerebellum": [7, 8, 46, 47],
    "Brainstem": [16],
}


def _read_freesurfer_lut() -> dict[int, str]:
    """Minimal FreeSurfer label LUT for region naming. Sourced from
    FreeSurferColorLUT.txt; here we ship the subset p-Brain needs."""
    return {
        2:  "Left-Cerebral-White-Matter",
        3:  "Left-Cerebral-Cortex",
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
        28: "Left-VentralDC",
        41: "Right-Cerebral-White-Matter",
        42: "Right-Cerebral-Cortex",
        46: "Right-Cerebellum-White-Matter",
        47: "Right-Cerebellum-Cortex",
        49: "Right-Thalamus-Proper",
        50: "Right-Caudate",
        51: "Right-Putamen",
        52: "Right-Pallidum",
        53: "Right-Hippocampus",
        54: "Right-Amygdala",
        58: "Right-Accumbens-area",
        60: "Right-VentralDC",
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
        threads: int = 0,
        max_concurrent: int = 1,
        reuse_existing: bool = True,
        **_: Any,
    ) -> TissueROI:
        import nibabel as nib

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        parc_path = out_dir / "parcellation.nii.gz"

        # Reuse an existing segmentation verbatim. Segmentation is
        # deterministic and expensive (~2 min/subject), so a ``--force`` re-run
        # of the kinetic chain must NOT re-segment. Disable with
        # ``--opt tissue_roi.synthseg.reuse_existing=false``.
        if reuse_existing and parc_path.exists() and parc_path.stat().st_size > 2048:
            try:
                _img = nib.load(str(parc_path))
                _parc = np.asarray(_img.dataobj, dtype=np.int32)
                if _parc.ndim == 3 and int(_parc.max()) > 0:
                    _present = sorted(int(x) for x in np.unique(_parc) if int(x) > 0)
                    _lut = _read_freesurfer_lut()
                    return TissueROI(
                        parcellation=_parc,
                        affine=np.asarray(_img.affine, dtype=float),
                        labels={l: _lut.get(l, f"label_{l}") for l in _present},
                        region_map={k: v for k, v in _REGION_MAP_FS.items()},
                        meta={"algorithm": "synthseg_reused",
                              "n_labels": len(_present),
                              "parc_path": str(parc_path), "reused": True},
                    )
            except Exception:
                pass  # corrupt/partial existing seg → fall through and re-segment

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

        t1_path = out_dir / "_t1_in.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(t1w_volume, dtype=np.float32),
                                  np.asarray(t1w_affine, dtype=float)), str(t1_path))

        # Thread budget + concurrency cap. With ``max_concurrent`` SynthSeg
        # processes allowed at once, give each ``cores / max_concurrent`` threads
        # so the total never oversubscribes the CPU. A single-subject run
        # (max_concurrent=1) gets all cores; a parallel cohort passes a higher
        # value and the semaphore below bounds memory/CPU regardless of workers.
        max_conc = max(1, int(max_concurrent or 1))
        nthreads = int(threads) if int(threads or 0) > 0 else max(1, (os.cpu_count() or 4) // max_conc)
        cmd = [binary, "--i", str(t1_path), "--o", str(parc_path),
               "--parc", "--cpu", "--threads", str(nthreads)]
        if not robust:
            cmd.append("--fast")

        # Route SynthSeg's TensorFlow/Python scratch to the **output volume** via
        # TMPDIR, not the system disk. mri_synthseg writes intermediates through
        # $TMPDIR; when the system /tmp is full it dies right after the banner
        # (degrading the subject to the whole-brain fallback). out_dir is on the
        # large derivatives volume, so scratch there is safe. Cleaned in finally.
        synth_tmp = out_dir / "_scratch"
        synth_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = env["TMP"] = env["TEMP"] = str(synth_tmp)

        slot = _acquire_synthseg_slot(max_conc)          # ≤ max_conc SynthSeg at once
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        finally:
            _release_synthseg_slot(slot)
            import shutil as _sh; _sh.rmtree(synth_tmp, ignore_errors=True)
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
