"""Utilities to compute and visualise diffusion tractography streamlines.

Heavy debug mode added. Set P_BRAIN_DEBUG_TRACKS=1 to enable extra prints.
Also writes a JSON snapshot alongside outputs for post-mortem inspection.
"""

from __future__ import annotations

import colorsys
import glob
import platform
import subprocess
from contextlib import contextmanager
from functools import wraps

import inspect
import multiprocessing as mp
import os
import queue
import traceback
import json
import datetime
import shutil
import sys
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np

try:  # ``fury`` is optional – fall back to matplotlib when unavailable.
    from dipy.viz import actor as dipy_actor
    from dipy.viz import colormap as dipy_colormap
    from dipy.viz import window as dipy_window

    _FURY_AVAILABLE = True
except Exception:  # pragma: no cover - fury is optional
    _FURY_AVAILABLE = False

import nibabel as nib
from nibabel import streamlines as nib_streamlines
try:  # ``StatefulTractogram`` helps preserve coordinate frames when saving.
    from nibabel.streamlines.stateful_tractogram import StatefulTractogram, Space
except ImportError:  # pragma: no cover - older nibabel without stateful helper
    StatefulTractogram = None  # type: ignore[assignment]
    Space = None  # type: ignore[assignment]

try:  # ``resample_from_to`` is only needed for anatomical overlays.
    from nibabel.processing import resample_from_to
except ImportError:  # pragma: no cover - helper ships with supported nibabel
    resample_from_to = None

from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel,
    auto_response_ssst,
)
from dipy.reconst.mcsd import (
    MultiShellDeconvModel,
    auto_response_msmt,
    multi_shell_fiber_response,
)
from dipy.reconst.shm import QballModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import (
    ActStoppingCriterion,
    CmcStoppingCriterion,
    ThresholdStoppingCriterion,
)
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import pft_tracking
from dipy.tracking.utils import seeds_from_mask

try:  # Optional import – tests may stub modules.opt08_fa partially.
    import modules.opt08_fa as _opt08_fa  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - executed only when module missing
    _opt08_fa = None

if TYPE_CHECKING:
    from modules.opt08_fa import DiffusionAcquisition as DiffusionAcquisitionType  # pragma: no cover
else:  # pragma: no cover - only needed for type checkers
    DiffusionAcquisitionType = object

DiffusionAcquisitionRuntime = (
    getattr(_opt08_fa, "DiffusionAcquisition", None) if _opt08_fa else None
)
_find_dwi_files = getattr(_opt08_fa, "find_dwi_files", None) if _opt08_fa else None

try:  # Optional dilation for seed masks.
    from scipy.ndimage import binary_dilation
except Exception:  # pragma: no cover - scipy optional
    binary_dilation = None

try:  # Optional animation writer.
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - imageio optional
    imageio = None

try:  # Optional ffmpeg backend for MP4 encoding.
    import imageio_ffmpeg  # type: ignore  # noqa: F401

    _IMAGEIO_HAS_FFMPEG = True
except Exception:  # pragma: no cover - optional dependency
    _IMAGEIO_HAS_FFMPEG = False

_DILATION_WARNING_EMITTED = False

_DBG = os.environ.get("P_BRAIN_DEBUG_TRACKS", "1").strip().lower() in {"1","true","yes","on"}

_SYSTEM_FFMPEG_BIN = shutil.which(os.environ.get("P_BRAIN_TRACK_FFMPEG_BIN", "ffmpeg"))



def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        _dbg(f"invalid float for {name}: {value}")
        return default
    return parsed


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        _dbg(f"invalid int for {name}: {value}")
        return default
    return max(1, parsed)


def _env_nonneg_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        _dbg(f"invalid int for {name}: {value}")
        return default
    return max(0, parsed)


def _env_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"auto", "none"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError:
        _dbg(f"invalid int for {name}: {value}")
        return default
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


_FS_MASKS_ENABLED = _env_bool("P_BRAIN_TRACK_ENABLE_FS_MASKS", True)
_FS_MASK_FORCE_REGEN = _env_bool("P_BRAIN_TRACK_FORCE_FS_MASKS", False) or _env_bool(
    "FORCE_RECREATE_MASKS",
    False,
)
_FS_MASK_KEEP_TEMP = _env_bool("P_BRAIN_TRACK_KEEP_FS_TEMP", False)
_FS_MASK_DIRNAME = os.environ.get("P_BRAIN_TRACK_FS_MASK_DIRNAME", "fs_tract_masks")
_FS_SEGMENTATION_BASENAMES = (
    "aparc.DKTatlas+aseg.deep.mgz",
    "aparc.DKTatlas+aseg.deep.nii.gz",
    "aparc.DKTatlas+aseg.mgz",
    "aparc.DKTatlas+aseg.nii.gz",
    "aparc+aseg.mgz",
    "aparc+aseg.nii.gz",
    "aseg.mgz",
    "aseg.nii.gz",
)


_DBG_STREAMLINE_SHAPES = _env_bool("P_BRAIN_DEBUG_STREAMLINE_SHAPES", False)
_DBG_STREAMLINE_SHAPES_LIMIT = (
    max(0, _env_int("P_BRAIN_DEBUG_STREAMLINE_SHAPES_LIMIT", 50))
    if _DBG_STREAMLINE_SHAPES
    else 0
)
_STREAMLINE_SHAPE_LOG_COUNT = 0

_CVX_VERBOSE = _env_bool("P_BRAIN_TRACK_VERBOSE_CVX", False)

_AUTO_ANATOMICAL_OVERLAY = _env_bool("P_BRAIN_TRACK_AUTO_ANATOMICAL", True)
_MONTAGE_MAX_POLYLINES = _env_int("P_BRAIN_TRACK_MONTAGE_MAX_POLYLINES", 1500)
_MONTAGE_SLICE_THICKNESS = float(
    max(0.25, _env_float("P_BRAIN_TRACK_MONTAGE_SLICE_THICKNESS", 0.75) or 0.75)
)
_RENDER_ONLY_DISABLE_SUBSAMPLE = _env_bool(
    "P_BRAIN_TRACK_RENDER_DISABLE_SUBSAMPLE",
    True,
)
_FORCE_MT_CSD = _env_bool("P_BRAIN_TRACK_FORCE_MT_CSD", False)
try:
    _CPU_COUNT = os.cpu_count() or 1
except Exception:  # pragma: no cover - extremely rare on exotic platforms
    _CPU_COUNT = 1
_TRACKING_WORKER_REQUEST = max(1, _env_int("P_BRAIN_TRACK_WORKERS", 1))
_TRACKING_PARALLEL_BACKEND = (
    os.environ.get("P_BRAIN_TRACK_PARALLEL_BACKEND", "thread").strip().lower() or "thread"
)


def _create_progress_bar(total: int, *, desc: str, unit: str):
    if total <= 0:
        return None
    if not _env_bool("P_BRAIN_TRACK_PROGRESS", True):
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore import
    except Exception:
        return None

    try:
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=False,
            dynamic_ncols=True,
            disable=False,
        )
    except Exception:
        return None


def _create_tracking_executor(max_workers: int):
    """Return an executor for tractography attempts when parallelism is enabled."""

    if max_workers <= 1:
        return None, "serial"

    backend = _TRACKING_PARALLEL_BACKEND
    if backend in {"process", "multiprocess", "mp"}:
        try:
            ctx = mp.get_context("spawn")
            executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
            return executor, "process"
        except Exception:
            if _DBG:
                _dbg("Process pool init failed – falling back to thread executor")
            backend = "thread"

    if backend in {"thread", "threads", "default"}:
        try:
            executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="tract",
            )
            return executor, "thread"
        except Exception:
            pass

    return None, "serial"


_TRACK_OPACITY = _env_float("P_BRAIN_TRACK_OPACITY", 0.9)
if _TRACK_OPACITY is not None:
    _TRACK_OPACITY = _clamp01(_TRACK_OPACITY)
    if _TRACK_OPACITY <= 0.0:
        _TRACK_OPACITY = None


def _set_actor_opacity(actor) -> None:
    """Apply uniform opacity to VTK actors when configured."""

    if actor is None or _TRACK_OPACITY is None:
        return

    getter = getattr(actor, "GetProperty", None)
    if getter is None:
        return

    try:
        prop = getter()
    except Exception:
        return

    if prop is None:
        return

    try:
        prop.SetOpacity(_TRACK_OPACITY)
    except Exception:
        pass


def _resolve_ffmpeg_binary() -> Optional[str]:
    override = os.environ.get("P_BRAIN_TRACK_FFMPEG_BIN")
    if override:
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return override
        maybe = shutil.which(override)
        if maybe:
            return maybe
    return _SYSTEM_FFMPEG_BIN


def _streamline_filter_defaults() -> dict[str, Optional[float]]:
    min_length = _env_float("P_BRAIN_TRACK_MIN_LENGTH", 30.0)
    if min_length is not None and min_length <= 0:
        min_length = None
    max_length = _env_float("P_BRAIN_TRACK_MAX_LENGTH", 200.0)
    if max_length is not None and max_length <= 0:
        max_length = None
    subsample = _env_int("P_BRAIN_TRACK_SUBSAMPLE", 5)
    subsample_min_count = _env_int("P_BRAIN_TRACK_SUBSAMPLE_MIN_COUNT", 250)
    return {
        "min_length": min_length,
        "max_length": max_length,
        "subsample_stride": subsample,
        "subsample_min_count": subsample_min_count,
    }


def _tracking_config_defaults(filter_defaults: dict[str, Optional[float]]) -> dict[str, object]:
    wm_threshold = _env_float("P_BRAIN_TRACK_WM_THRESHOLD", 0.2)
    if wm_threshold is None or wm_threshold <= 0:
        wm_threshold = 0.2

    stop_threshold = _env_float("P_BRAIN_TRACK_STOP_THRESHOLD", 0.15)
    if stop_threshold is None or stop_threshold <= 0:
        stop_threshold = 0.15

    fallback_stop = _env_float("P_BRAIN_TRACK_FALLBACK_STOP_THRESHOLD", None)
    if fallback_stop is None:
        fallback_stop = max(0.04, stop_threshold * 0.65)

    seed_density = _env_int("P_BRAIN_TRACK_SEED_DENSITY", 1)
    min_streamlines = _env_nonneg_int("P_BRAIN_TRACK_MIN_STREAMLINES", 400)

    fallback_stride = _env_int(
        "P_BRAIN_TRACK_FALLBACK_SUBSAMPLE",
        max(1, int(filter_defaults.get("subsample_stride", 1) or 1) // 2),
    )

    fallback_min_length = _env_float("P_BRAIN_TRACK_FALLBACK_MIN_LENGTH", None)
    if fallback_min_length is None:
        base = filter_defaults.get("min_length")
        fallback_min_length = 10.0 if base is None else max(5.0, 0.5 * float(base))

    target_default = max(1500, int(min_streamlines * 2) if min_streamlines else 1500)
    target_streamlines = _env_int("P_BRAIN_TRACK_TARGET_STREAMLINES", target_default)
    max_seed_points = _env_nonneg_int("P_BRAIN_TRACK_MAX_SEEDS", 900000)
    max_attempts = _env_int("P_BRAIN_TRACK_MAX_ATTEMPTS", 3)
    max_output_streamlines = _env_int(
        "P_BRAIN_TRACK_MAX_OUTPUT_STREAMLINES",
        max(target_streamlines, 6000),
    )

    csd_response_roi = _env_int("P_BRAIN_TRACK_CSD_RESPONSE_ROI", 8)
    csd_response_fa = _env_float("P_BRAIN_TRACK_CSD_RESPONSE_FA", 0.7)
    csd_sh_order = _env_optional_int("P_BRAIN_TRACK_CSD_SH_ORDER", None)
    if csd_sh_order is not None:
        csd_sh_order = max(2, int(csd_sh_order))
        if csd_sh_order % 2:
            csd_sh_order += 1

    qball_sh_order = _env_int("P_BRAIN_TRACK_QBALL_SH_ORDER", 6)
    if qball_sh_order % 2:
        qball_sh_order += 1
    gqi_sampling_length = _env_float("P_BRAIN_TRACK_GQI_SAMPLING_LENGTH", 1.2)
    if not gqi_sampling_length or gqi_sampling_length <= 0:
        gqi_sampling_length = 1.2
    gqi_normalize_peaks = _env_bool("P_BRAIN_TRACK_GQI_NORMALIZE_PEAKS", False)

    act_enabled = _env_bool("P_BRAIN_TRACK_ENABLE_ACT", False)
    pft_enabled = _env_bool("P_BRAIN_TRACK_ENABLE_PFT", False)
    pft_step_size = _env_float("P_BRAIN_TRACK_PFT_STEP_MM", 0.2)
    if not pft_step_size or pft_step_size <= 0:
        pft_step_size = 0.2
    pft_max_angle = _env_float("P_BRAIN_TRACK_PFT_MAX_ANGLE", 20.0) or 20.0
    pft_pmf_threshold = _env_float("P_BRAIN_TRACK_PFT_PMF_THRESHOLD", 0.1)
    if pft_pmf_threshold is None or pft_pmf_threshold <= 0:
        pft_pmf_threshold = 0.1
    pft_particle_count = _env_int("P_BRAIN_TRACK_PFT_PARTICLES", 15)
    pft_backtrack = _env_float("P_BRAIN_TRACK_PFT_BACKTRACK_MM", 2.0) or 2.0
    pft_front = _env_float("P_BRAIN_TRACK_PFT_FRONT_MM", 1.0) or 1.0
    pft_max_trials = _env_int("P_BRAIN_TRACK_PFT_MAX_TRIALS", 20)
    pft_seed_buffer = _env_float("P_BRAIN_TRACK_PFT_SEED_BUFFER", 1.0) or 1.0
    pft_min_wm_pve = _env_float("P_BRAIN_TRACK_PFT_MIN_WM_PVE", 0.0) or 0.0
    anatomical_min_coverage = _env_float("P_BRAIN_TRACK_MIN_ANATOMICAL_COVERAGE", 0.65)
    if anatomical_min_coverage is None:
        anatomical_min_coverage = 0.65
    anatomical_min_coverage = _clamp01(float(anatomical_min_coverage))

    return {
        "wm_threshold": wm_threshold,
        "stop_threshold": stop_threshold,
        "fallback_stop_threshold": fallback_stop,
        "seed_density": max(1, seed_density),
        "min_streamlines": min_streamlines,
        "target_streamlines": max(min_streamlines, target_streamlines),
        "fallback_subsample_stride": max(1, fallback_stride),
        "fallback_min_length": fallback_min_length,
        "max_seed_points": max_seed_points,
        "max_attempts": max(1, max_attempts),
        "max_output_streamlines": max_output_streamlines,
        "csd_response_roi": csd_response_roi,
        "csd_response_fa": csd_response_fa,
        "csd_sh_order": csd_sh_order,
        "qball_sh_order": qball_sh_order,
        "gqi_sampling_length": gqi_sampling_length,
        "gqi_normalize_peaks": gqi_normalize_peaks,
        "act_enabled": act_enabled,
        "pft_enabled": pft_enabled,
        "pft_step_size": pft_step_size,
        "pft_max_angle": pft_max_angle,
        "pft_pmf_threshold": pft_pmf_threshold,
        "pft_particle_count": pft_particle_count,
        "pft_backtrack_distance": pft_backtrack,
        "pft_front_distance": pft_front,
        "pft_max_trials": pft_max_trials,
        "pft_seed_buffer_fraction": pft_seed_buffer,
        "pft_min_wm_pve": pft_min_wm_pve,
        "anatomical_min_coverage": anatomical_min_coverage,
    }


def _find_anatomical_overlay_path(nifti_directory: str) -> Optional[str]:
    """Return a T1-weighted volume aligned to DCE space when present."""

    preferred = (
        os.path.join(
            nifti_directory,
            "segmentation",
            "segmentation",
            "mri",
            "T1w_conformed_in_DCE.nii.gz",
        ),
        os.path.join(
            nifti_directory,
            "segmentation",
            "segmentation",
            "mri",
            "T1w_in_DCE.nii.gz",
        ),
        os.path.join(
            nifti_directory,
            "segmentation",
            "segmentation",
            "mri",
            "T1_in_DCE.nii.gz",
        ),
    )

    for candidate in preferred:
        if os.path.isfile(candidate):
            return candidate

    search_roots = (
        os.path.join(nifti_directory, "segmentation", "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation"),
        nifti_directory,
    )

    patterns = (
        "*T1*DCE*.nii.gz",
        "*T1*DCE*.nii",
        "*T1*_in_DCE*.nii.gz",
        "*T1*_in_DCE*.nii",
    )

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pattern in patterns:
            search_pattern = os.path.join(root, "**", pattern)
            for path in sorted(glob.iglob(search_pattern, recursive=True)):
                if os.path.isfile(path):
                    return path

    return None


_TISSUE_VOLUME_PATTERNS = {
    "wm": (
        "pve_wm.nii.gz",
        "pve_wm.nii",
        "wm_prob.nii.gz",
        "wm_prob.nii",
        "wm_mask.nii.gz",
        "white_matter.nii.gz",
        "white_matter.nii",
        "wm.nii.gz",
        "wm.nii",
    ),
    "gm": (
        "pve_gm.nii.gz",
        "pve_gm.nii",
        "gm_prob.nii.gz",
        "gm_prob.nii",
        "cortical_gm.nii.gz",
        "cortical_gm.nii",
        "subcortical_gm.nii.gz",
        "subcortical_gm.nii",
        "gm_mask.nii.gz",
        "gm_mask.nii",
    ),
    "csf": (
        "pve_csf.nii.gz",
        "pve_csf.nii",
        "csf_prob.nii.gz",
        "csf_prob.nii",
        "csf_mask.nii.gz",
        "csf_mask.nii",
        "csf.nii.gz",
        "csf.nii",
        "ventricles.nii.gz",
        "ventricles.nii",
    ),
}

_ACT_WM_MASKS = ("white_matter", "wm_cerebellum", "wm_cc", "brainstem")
_ACT_GM_MASKS = ("cortical_gm", "subcortical_gm", "gm_cerebellum", "gm_brainstem")
_ACT_CSF_MASKS = ("csf",)


def _tissue_probability_search_roots(
    nifti_directory: str,
    analysis_directory: str,
) -> list[str]:
    roots = [
        os.path.join(nifti_directory, "segmentation", "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation"),
        nifti_directory,
        analysis_directory,
        os.path.dirname(os.path.abspath(analysis_directory)),
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root:
            continue
        normalized = os.path.abspath(root)
        if normalized in seen:
            continue
        if os.path.isdir(normalized):
            ordered.append(normalized)
            seen.add(normalized)
    return ordered


def _find_probability_volume(
    search_roots: Sequence[str], patterns: Sequence[str]
) -> Optional[str]:
    normalized_patterns = tuple(p.lower() for p in patterns)
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        try:
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    lower = filename.lower()
                    if any(lower.endswith(pattern) for pattern in normalized_patterns):
                        return os.path.join(dirpath, filename)
        except Exception:
            continue
    return None


def _load_probability_volume(
    volume_path: str,
    reference_img: nib.Nifti1Image,
) -> Optional[np.ndarray]:
    try:
        img = nib.load(volume_path)
    except Exception:
        return None

    data = np.asarray(img.get_fdata(dtype=np.float32))
    while data.ndim > 3 and data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)
    if data.ndim > 3:
        return None

    target_shape = reference_img.shape[:3]
    target_affine = _canonical_affine(reference_img.affine)
    need_resample = data.shape != tuple(target_shape) or not np.allclose(
        _canonical_affine(getattr(img, "affine", np.eye(4))),
        target_affine,
    )
    if need_resample:
        if resample_from_to is None:
            return None
        try:
            resampled = resample_from_to(img, (target_shape, target_affine), order=1)
            data = np.asarray(resampled.get_fdata(dtype=np.float32))
        except Exception:
            return None

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(data)) if data.size else 0.0
    if max_val > 0.0:
        if max_val > 1.0:
            data = data / max_val
    data = np.clip(data, 0.0, 1.0)
    return np.ascontiguousarray(data.astype(np.float32))


def _collect_opt_tissue_masks(
    nifti_directory: str,
    analysis_directory: str,
    reference_img: nib.Nifti1Image,
) -> dict[str, tuple[np.ndarray, dict[str, str]]]:
    collector = getattr(_opt08_fa, "_collect_tissue_masks", None)
    if callable(collector):
        try:
            return collector(nifti_directory, analysis_directory, reference_img)
        except Exception:
            if _DBG:
                _dbg("opt08_fa._collect_tissue_masks failed; continuing without cached masks")
    return {}


def _combine_mask_set(
    masks: dict[str, tuple[np.ndarray, dict[str, str]]],
    keys: Sequence[str],
) -> Optional[np.ndarray]:
    combined: Optional[np.ndarray] = None
    for key in keys:
        if key not in masks:
            continue
        mask = np.asarray(masks[key][0], dtype=np.float32)
        combined = mask if combined is None else np.maximum(combined, mask)
    if combined is None:
        return None
    return np.ascontiguousarray(np.clip(combined, 0.0, 1.0).astype(np.float32))


def _gather_tissue_probability_maps(
    nifti_directory: str,
    analysis_directory: str,
    reference_img: nib.Nifti1Image,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    # For tractography we rely exclusively on FreeSurfer-derived masks built
    # from the specified segmentation and ignore DCE/T2 probability maps.
    # This keeps ACT/PFT independent of perfusion geometry.

    maps: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    fs_metadata: dict[str, object] = {}

    if _FS_MASKS_ENABLED:
        try:
            fs_outputs, fs_metadata = _ensure_fs_mask_files(nifti_directory)
        except RuntimeError as exc:
            fs_metadata = {"status": "failed", "error": str(exc)}
            fs_outputs = {}

        # Load only the FS masks we created, on the diffusion reference grid.
        for label, path in sorted(fs_outputs.items()):
            if not os.path.isfile(path):
                continue
            volume = _load_probability_volume(path, reference_img)
            if volume is None:
                continue
            key = f"fs_{label}"
            maps[key] = volume
            sources[key] = path

    metadata = {
        "sources": sources,
        "available_maps": sorted(maps.keys()),
    }
    if fs_metadata:
        metadata["fs_masks"] = fs_metadata
    return maps, metadata


def _fs_segmentation_candidates(nifti_directory: str) -> list[str]:
    roots = [
        os.path.join(nifti_directory, "segmentation", "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation", "mri"),
        os.path.join(nifti_directory, "segmentation"),
        nifti_directory,
    ]
    candidates: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root:
            continue
        normalized = os.path.abspath(root)
        if normalized in seen or not os.path.isdir(normalized):
            continue
        seen.add(normalized)
        for basename in _FS_SEGMENTATION_BASENAMES:
            path = os.path.join(normalized, basename)
            if os.path.isfile(path):
                candidates.append(path)
    return candidates


def _fs_mask_command_plan(
    seg_path: str, mask_dir: str
) -> tuple[list[dict[str, object]], str, list[str]]:
    aseg_base, aseg_ext = os.path.splitext(seg_path)
    if aseg_ext == ".gz":
        aseg_base, inner_ext = os.path.splitext(aseg_base)
        aseg_ext = inner_ext + aseg_ext
    aseg_nii = os.path.join(mask_dir, os.path.basename(aseg_base)) + ".nii.gz"
    commands: list[dict[str, object]] = []
    commands.append(
        {
            "label": "aseg_convert",
            "outputs": [aseg_nii],
            "cmd": ["mri_convert", seg_path, aseg_nii],
        }
    )

    def bin_mask(label: str, args: list[str]) -> dict[str, object]:
        path = os.path.join(mask_dir, f"{label}.nii.gz")
        return {
            "label": label,
            "outputs": [path],
            "cmd": ["mri_binarize", "--i", aseg_nii, *args, "--o", path],
            "path": path,
        }

    commands.extend(
        [
            bin_mask("temp_wm", ["--all-wm"]),
            bin_mask("temp_subcortical_gm", ["--subcort-gm"]),
            bin_mask("gm", ["--gm"]),
            bin_mask("gm_brainstem", ["--match", "16"]),
            bin_mask("gm_cerebellum", ["--match", "8", "47"]),
            bin_mask("wm_cerebellum", ["--match", "7", "46"]),
            bin_mask("wm_cc", ["--match", "251", "252", "253", "254", "255"]),
            bin_mask("csf", ["--ventricles"]),
        ]
    )

    def fsl_math(label: str, args: list[str], sources: list[str]) -> dict[str, object]:
        path = os.path.join(mask_dir, f"{label}.nii.gz")
        return {
            "label": label,
            "outputs": [path],
            "cmd": ["fslmaths", *args, path],
            "path": path,
            "sources": sources,
        }

    tg = os.path.join(mask_dir, "gm.nii.gz")
    temp_subcort = os.path.join(mask_dir, "temp_subcortical_gm.nii.gz")
    gm_brainstem = os.path.join(mask_dir, "gm_brainstem.nii.gz")
    gm_cereb = os.path.join(mask_dir, "gm_cerebellum.nii.gz")
    wm_cereb = os.path.join(mask_dir, "wm_cerebellum.nii.gz")
    wm_cc = os.path.join(mask_dir, "wm_cc.nii.gz")
    temp_wm = os.path.join(mask_dir, "temp_wm.nii.gz")

    commands.extend(
        [
            fsl_math(
                "cortical_gm",
                [tg, "-sub", temp_subcort, "-sub", gm_brainstem, "-sub", gm_cereb, "-thr", "0.5", "-bin"],
                [tg, temp_subcort, gm_brainstem, gm_cereb],
            ),
            fsl_math(
                "subcortical_gm",
                [temp_subcort, "-sub", gm_brainstem, "-sub", gm_cereb, "-thr", "0.5", "-bin"],
                [temp_subcort, gm_brainstem, gm_cereb],
            ),
            fsl_math(
                "white_matter",
                [temp_wm, "-sub", wm_cereb, "-sub", wm_cc, "-thr", "0.5", "-bin"],
                [temp_wm, wm_cereb, wm_cc],
            ),
        ]
    )

    cleanups = [temp_wm, temp_subcort, tg]
    return commands, aseg_nii, cleanups


def _fs_command_env() -> dict[str, str]:
    env = os.environ.copy()
    path_entries: list[str] = []
    fs_home = env.get("FREESURFER_HOME")
    if fs_home:
        path_entries.append(os.path.join(fs_home, "bin"))
    fsl_dir = env.get("FSLDIR")
    if fsl_dir:
        path_entries.append(os.path.join(fsl_dir, "bin"))
        env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    path_entries.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)
    return env


def _run_mask_command(cmd: Sequence[str], *, label: str, env: dict[str, str], cwd: str) -> None:
    if _DBG:
        _dbg(f"[{label}] running: {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
        )
    except subprocess.CalledProcessError as exc:  # noqa: PERF203 - need stdout/stderr
        raise RuntimeError(
            f"{label} failed (exit {exc.returncode}). stdout: {exc.stdout}\nstderr: {exc.stderr}"
        ) from exc


def _ensure_fs_mask_files(nifti_directory: str) -> tuple[dict[str, str], dict[str, object]]:
    metadata: dict[str, object] = {"enabled": bool(_FS_MASKS_ENABLED)}
    outputs: dict[str, str] = {}
    if not _FS_MASKS_ENABLED:
        metadata["status"] = "disabled"
        return outputs, metadata

    seg_candidates = _fs_segmentation_candidates(nifti_directory)
    if not seg_candidates:
        metadata["status"] = "missing_segmentation"
        return outputs, metadata

    seg_path = seg_candidates[0]
    mask_dir = os.path.join(os.path.dirname(seg_path), _FS_MASK_DIRNAME)
    os.makedirs(mask_dir, exist_ok=True)
    metadata.update({
        "status": "pending",
        "segmentation": seg_path,
        "mask_dir": mask_dir,
    })

    final_targets = {
        "white_matter": os.path.join(mask_dir, "white_matter.nii.gz"),
        "cortical_gm": os.path.join(mask_dir, "cortical_gm.nii.gz"),
        "subcortical_gm": os.path.join(mask_dir, "subcortical_gm.nii.gz"),
        "gm_brainstem": os.path.join(mask_dir, "gm_brainstem.nii.gz"),
        "gm_cerebellum": os.path.join(mask_dir, "gm_cerebellum.nii.gz"),
        "wm_cerebellum": os.path.join(mask_dir, "wm_cerebellum.nii.gz"),
        "wm_cc": os.path.join(mask_dir, "wm_cc.nii.gz"),
        "csf": os.path.join(mask_dir, "csf.nii.gz"),
    }

    if not _FS_MASK_FORCE_REGEN and all(os.path.isfile(path) for path in final_targets.values()):
        metadata["status"] = "cached"
        metadata["available"] = sorted(final_targets.keys())
        return dict(final_targets), metadata

    commands, aseg_nii, cleanup_paths = _fs_mask_command_plan(seg_path, mask_dir)
    env = _fs_command_env()
    required_bins = {entry["cmd"][0] for entry in commands}
    missing_bins = [tool for tool in required_bins if shutil.which(tool, path=env.get("PATH", "")) is None]
    if missing_bins:
        metadata["status"] = "missing_tools"
        metadata["missing"] = missing_bins
        return outputs, metadata

    executed: list[str] = []
    for entry in commands:
        label = entry.get("label", entry["cmd"][0])
        outputs_exist = all(os.path.isfile(path) for path in entry.get("outputs", []))
        if outputs_exist and not _FS_MASK_FORCE_REGEN:
            continue
        _run_mask_command(entry["cmd"], label=label, env=env, cwd=mask_dir)
        executed.append(str(label))

    if not _FS_MASK_KEEP_TEMP:
        for temp_path in cleanup_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    metadata["status"] = "generated"
    metadata["commands_run"] = executed
    metadata["aseg_nii"] = aseg_nii
    existing = {key: path for key, path in final_targets.items() if os.path.isfile(path)}
    metadata["available"] = sorted(existing.keys())

    return existing, metadata


def _prepare_anatomical_strategy(
    nifti_directory: str,
    analysis_directory: str,
    reference_img: nib.Nifti1Image,
    tracking_config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    requested_act = bool(tracking_config.get("act_enabled", False))
    requested_pft = bool(tracking_config.get("pft_enabled", False))
    strategy: dict[str, object] = {
        "use_act": requested_act,
        "use_pft": requested_pft,
        "act": None,
        "cmc": None,
        "tissue_maps": None,
        "pft_kwargs": {
            "step_size": float(tracking_config.get("pft_step_size", 0.2)),
            "max_angle": float(tracking_config.get("pft_max_angle", 20.0)),
            "pmf_threshold": float(tracking_config.get("pft_pmf_threshold", 0.1)),
            "particle_count": int(tracking_config.get("pft_particle_count", 15) or 15),
            "back_tracking_dist": float(tracking_config.get("pft_backtrack_distance", 2.0)),
            "front_tracking_dist": float(tracking_config.get("pft_front_distance", 1.0)),
            "max_trials": int(tracking_config.get("pft_max_trials", 20) or 20),
            "seed_buffer_fraction": float(tracking_config.get("pft_seed_buffer_fraction", 1.0)),
            "min_wm_pve": float(tracking_config.get("pft_min_wm_pve", 0.0)),
            "voxel_size": None,
        },
    }
    debug: dict[str, object] = {
        "act_requested": requested_act,
        "pft_requested": requested_pft,
    }

    voxel_sizes: Optional[np.ndarray]
    try:
        voxel_sizes = _voxel_sizes_from_affine(reference_img.affine)
    except Exception as exc:
        voxel_sizes = None
        debug["voxel_size_error"] = str(exc)
    else:
        strategy["pft_kwargs"]["voxel_size"] = tuple(float(x) for x in np.asarray(voxel_sizes).ravel()[:3])
        debug["voxel_size_mm"] = strategy["pft_kwargs"]["voxel_size"]

    if not (requested_act or requested_pft):
        return strategy, debug

    tissue_maps, metadata = _gather_tissue_probability_maps(
        nifti_directory,
        analysis_directory,
        reference_img,
    )
    debug.update(metadata)
    required = ("wm", "gm", "csf")
    if not all(key in tissue_maps for key in required):
        debug["anatomical_disabled"] = "missing_tissue_maps"
        strategy["use_act"] = False
        strategy["use_pft"] = False
        return strategy, debug

    strategy["tissue_maps"] = tissue_maps
    try:
        if strategy["use_act"]:
            strategy["act"] = ActStoppingCriterion.from_pve(
                tissue_maps["wm"], tissue_maps["gm"], tissue_maps["csf"]
            )
    except Exception as exc:
        debug["act_error"] = str(exc)
        strategy["use_act"] = False

    try:
        if strategy["use_pft"]:
            cmc_kwargs: dict[str, object] = {}
            cmc_step = strategy["pft_kwargs"].get("step_size")
            if cmc_step:
                cmc_kwargs["step_size"] = float(cmc_step)
            if voxel_sizes is not None:
                cmc_kwargs["average_voxel_size"] = float(np.mean(voxel_sizes))
            strategy["cmc"] = CmcStoppingCriterion.from_pve(
                tissue_maps["wm"],
                tissue_maps["gm"],
                tissue_maps["csf"],
                **cmc_kwargs,
            )
    except Exception as exc:
        debug["pft_error"] = str(exc)
        strategy["use_pft"] = False

    debug["act_enabled"] = bool(strategy["use_act"])
    debug["pft_enabled"] = bool(strategy["use_pft"])
    return strategy, debug


def _select_stopping_criterion(
    stop_threshold: float,
    fa_volume: np.ndarray,
    strategy: Optional[dict[str, object]],
):
    if strategy:
        if strategy.get("use_pft") and strategy.get("cmc") is not None:
            return strategy["cmc"]
        if strategy.get("use_act") and strategy.get("act") is not None:
            return strategy["act"]
    return ThresholdStoppingCriterion(fa_volume, float(stop_threshold))


def _voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
    arr = np.asarray(affine, dtype=np.float64)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return np.ones(3, dtype=np.float64)
    linear = arr[:3, :3]
    return np.sqrt(np.sum(linear * linear, axis=0))


def _run_particle_filtering_tracking(
    peaks,
    seeds: np.ndarray,
    voxel_to_world: np.ndarray,
    *,
    stopping_criterion,
    config: dict[str, object],
    rng_seed: Optional[int],
):
    step_size = float(config.get("step_size", 0.2) or 0.2)
    max_angle = float(config.get("max_angle", 20.0) or 20.0)
    pmf_threshold = float(config.get("pmf_threshold", 0.1) or 0.1)
    particle_count = int(config.get("particle_count", 15) or 15)
    backtrack = float(config.get("back_tracking_dist", 2.0) or 2.0)
    front = float(config.get("front_tracking_dist", 1.0) or 1.0)
    max_trials = int(config.get("max_trials", 20) or 20)
    seed_buffer = float(config.get("seed_buffer_fraction", 1.0) or 1.0)
    min_wm_pve = float(config.get("min_wm_pve", 0.0) or 0.0)

    voxel_size = config.get("voxel_size")
    if voxel_size is None:
        voxel_size = _voxel_sizes_from_affine(voxel_to_world)
    voxel_size = tuple(float(x) for x in np.asarray(voxel_size).ravel()[:3])

    sh_coeffs = getattr(peaks, "shm_coeff", None)
    kwargs = {
        "seed_positions": seeds,
        "sc": stopping_criterion,
        "affine": voxel_to_world,
        "step_size": step_size,
        "voxel_size": voxel_size,
        "max_angle": max_angle,
        "pmf_threshold": pmf_threshold,
        "particle_count": particle_count,
        "pft_back_tracking_dist": backtrack,
        "pft_front_tracking_dist": front,
        "pft_max_trial": max_trials,
        "seed_buffer_fraction": seed_buffer,
        "min_wm_pve_before_stopping": min_wm_pve,
    }
    if sh_coeffs is not None:
        kwargs["sh"] = sh_coeffs
    else:
        kwargs["pam"] = peaks
    if rng_seed is not None:
        kwargs["random_seed"] = int(np.uint32(rng_seed))

    streamlines = pft_tracking(**kwargs)
    return Streamlines(streamlines)

def _dbg_print(msg: str) -> None:
    print(msg, flush=True)


def _dbg(msg: str) -> None:
    if _DBG:
        _dbg_print(f"[tracks][dbg] {msg}")


def _log_streamline_shape(msg: str) -> None:
    global _STREAMLINE_SHAPE_LOG_COUNT
    if not (_DBG and _DBG_STREAMLINE_SHAPES):
        return
    if _DBG_STREAMLINE_SHAPES_LIMIT and _STREAMLINE_SHAPE_LOG_COUNT >= _DBG_STREAMLINE_SHAPES_LIMIT:
        return
    _STREAMLINE_SHAPE_LOG_COUNT += 1
    _dbg_print(msg)


if _TRACK_OPACITY is not None and _DBG:
    _dbg(f"track opacity forced to {_TRACK_OPACITY}")


def _safe_summary_array(a, name: str, max_elems: int = 6) -> dict:
    try:
        shp = tuple(int(x) for x in np.shape(a))
    except Exception:
        shp = "<unknown>"
    preview = None
    try:
        flat = np.ravel(a)
        if flat.size > 0:
            preview = [float(flat[i]) for i in range(min(flat.size, max_elems))]
    except Exception:
        preview = "<unavailable>"
    return {"name": name, "shape": shp, "preview": preview}


def _dump_debug_json(path: str, payload: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=lambda o: str(o))
    except Exception as _:
        pass


def _print_affine(label: str, A: np.ndarray) -> None:
    if not _DBG:
        return
    _dbg_print(f"[tracks][dbg] {label} shape={getattr(A,'shape',None)}")
    _dbg_print(f"[tracks][dbg] {label}=\n{np.array(A)}")

def _canonical_affine(affine: np.ndarray) -> np.ndarray:
    """Return a 4x4 voxel-to-world affine derived from ``affine``."""

    arr = np.asarray(affine, dtype=np.float64)
    if arr.shape == (4, 4):
        out = np.array(arr, copy=True)
        _print_affine("canonical affine 4x4 ok", out)
        return out

    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 4:
        raise ValueError("Expected affine with at least 4x4 elements")

    canonical = np.eye(4, dtype=arr.dtype)
    canonical[:3, :3] = arr[:3, :3]
    canonical[:3, 3] = arr[:3, 3]
    _print_affine("canonical affine coerced", canonical)
    return canonical

@dataclass
class TractographyOutputs:
    """Container describing generated tractography artefacts."""

    tract_path: str
    render_path: Optional[str] = None
    montage_path: Optional[str] = None
    density_path: Optional[str] = None
    animation_path: Optional[str] = None


@dataclass(frozen=True)
class DiffusionDataset:
    """Resolved diffusion inputs used for tractography."""

    volume_path: str
    bval_path: str
    bvec_path: str
    label: str = "unknown"
    default_model: str = "DTI"


@dataclass
class OrientationFit:
    """Describes an orientation model fit plus supporting metadata."""

    requested: str
    resolved: str
    model: object
    fa_volume: np.ndarray
    details: dict[str, object]
    peaks_kwargs: dict[str, object]


def _fa_stats(volume: np.ndarray) -> dict[str, float]:
    """Return quick FA distribution statistics for debugging output."""

    stats = {
        "min": float(np.min(volume)) if volume.size else 0.0,
        "max": float(np.max(volume)) if volume.size else 0.0,
        "mean": float(np.mean(volume)) if volume.size else 0.0,
    }
    return stats


def _resolve_diffusion_dataset(
    nifti_directory: str,
    diffusion_filename: Optional[str],
) -> DiffusionDataset:
    """Return a single, validated diffusion dataset from the filesystem."""

    if _find_dwi_files is None:
        raise ImportError(
            "modules.opt08_fa.find_dwi_files is unavailable – diffusion discovery requires opt08_fa."
        )

    preferred: Optional[tuple[str, ...]] = None
    if diffusion_filename:
        preferred = (diffusion_filename,)

    acquisition = _find_dwi_files(
        nifti_directory,
        preferred_filenames=preferred,
        include_metadata=True,
    )
    if acquisition is None:
        raise FileNotFoundError(
            "No diffusion dataset found – ensure parameters.py lists a diffusion filename."
        )

    if (
        DiffusionAcquisitionRuntime is not None
        and isinstance(acquisition, DiffusionAcquisitionRuntime)
    ):
        dataset = DiffusionDataset(
            volume_path=acquisition.volume_path,
            bval_path=acquisition.bval_path,
            bvec_path=acquisition.bvec_path,
            label=acquisition.label or "unknown",
            default_model=(acquisition.model or "DTI").upper(),
        )
    else:
        volume_path, bval_path, bvec_path = acquisition
        dataset = DiffusionDataset(
            volume_path=volume_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
        )

    missing = [
        path
        for path in (dataset.volume_path, dataset.bval_path, dataset.bvec_path)
        if not path or not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Diffusion dataset incomplete – missing files: " + ", ".join(missing)
        )

    return dataset


def _infer_csd_sh_order(
    gtab,
    *,
    default: int = 4,
    min_order: int = 2,
    max_order: int = 12,
    safety_margin: int = 4,
) -> tuple[int, dict[str, object]]:
    """Return a feasible SH order for CSD along with diagnostics."""

    diagnostics: dict[str, object] = {
        "default": int(default),
        "min_order": int(min_order),
        "max_order": int(max_order),
        "safety_margin": int(safety_margin),
    }

    try:
        bvals = np.asarray(getattr(gtab, "bvals", []), dtype=np.float32)
        bvecs = np.asarray(getattr(gtab, "bvecs", []), dtype=np.float32)
    except Exception:
        diagnostics["reason"] = "missing_gradient_table"
        selected = default if default % 2 == 0 else default + 1
        selected = max(2, min(selected, max_order))
        diagnostics["selected_order"] = int(selected)
        diagnostics["auto"] = True
        return selected, diagnostics

    if bvals.ndim == 0:
        bvals = bvals.reshape(1)
    gradient_mask = bvals > 50.0
    nonzero_gradients = int(np.count_nonzero(gradient_mask))
    diagnostics["nonzero_gradients"] = nonzero_gradients
    diagnostics["b0_count"] = int(bvals.size - nonzero_gradients)

    unique_directions = 0
    if nonzero_gradients > 0 and bvecs.size:
        try:
            grad_vecs = bvecs[gradient_mask]
            if grad_vecs.ndim != 2 or grad_vecs.shape[1] != 3:
                grad_vecs = np.reshape(grad_vecs, (-1, 3))
            norms = np.linalg.norm(grad_vecs, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            normed = grad_vecs / norms[:, None]
            quantized = np.round(normed, 3)
            unique = {tuple(vec.tolist()) for vec in quantized if np.all(np.isfinite(vec))}
            unique_directions = len(unique)
        except Exception:
            unique_directions = 0

    direction_count = max(nonzero_gradients, unique_directions)
    diagnostics["unique_directions"] = unique_directions or None
    diagnostics["direction_count"] = direction_count

    min_even = max(2, int(min_order))
    if min_even % 2:
        min_even += 1
    best_order = 2 if direction_count < 6 else min_even
    margin = max(0, int(safety_margin))

    order_requirements: list[dict[str, int]] = []
    for order in range(best_order, int(max_order) + 1, 2):
        coeffs = (order + 1) * (order + 2) // 2
        order_requirements.append({"order": order, "coefficients": coeffs})
        if direction_count >= coeffs + margin:
            best_order = order
        else:
            break
    diagnostics["order_requirements"] = order_requirements

    if direction_count >= (default + 1) * (default + 2) // 2 + margin:
        best_order = max(best_order, default if default % 2 == 0 else default + 1)

    best_order = min(max_order, max(2, best_order))
    diagnostics["selected_order"] = int(best_order)
    diagnostics["auto"] = True
    return best_order, diagnostics


def _fit_orientation_model(
    data: np.ndarray,
    gtab,
    requested_model: Optional[str],
    tracking_config: dict[str, float],
) -> OrientationFit:
    """Fit the requested orientation model, falling back to tensors when needed."""

    requested = (requested_model or "DTI").upper()
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(data)
    fa_volume = tensor_fit.fa.astype(np.float32)
    fa_volume = np.nan_to_num(fa_volume, nan=0.0, posinf=0.0, neginf=0.0)

    base_details = {"fa_stats": _fa_stats(fa_volume)}

    def _result(
        model: object,
        resolved: str,
        extra: Optional[dict[str, object]] = None,
        peaks_kwargs: Optional[dict[str, object]] = None,
    ) -> OrientationFit:
        details = dict(base_details)
        if extra:
            details.update(extra)
        return OrientationFit(
            requested=requested,
            resolved=resolved,
            model=model,
            fa_volume=fa_volume,
            details=details,
            peaks_kwargs=dict(peaks_kwargs or {}),
        )

    normalized_requested = requested.replace("-", "_")
    if normalized_requested == "CSD" and _FORCE_MT_CSD:
        normalized_requested = "MT_CSD"
        base_details["csd_force_mt"] = True
    if normalized_requested in {"DTI", "TENSOR"}:
        return _result(tensor_model, "TENSOR")

    if normalized_requested in {"MT_CSD", "MSMT_CSD", "MSMT"}:
        csd_roi = int(tracking_config.get("csd_response_roi") or 8)
        csd_fa_thr = float(tracking_config.get("csd_response_fa") or 0.7)
        gm_fa_thr = max(0.2, csd_fa_thr * 0.5)
        csf_fa_thr = 0.15
        csd_sh_override = tracking_config.get("csd_sh_order")
        response_tol_cfg = tracking_config.get("csd_response_tol")
        response_tol = 20
        if response_tol_cfg not in (None, ""):
            try:
                response_tol = max(1, int(response_tol_cfg))
            except Exception:
                pass
        sh_order = int(csd_sh_override) if csd_sh_override else 8
        if sh_order % 2:
            sh_order += 1
        try:
            response_wm, response_gm, response_csf = auto_response_msmt(
                gtab,
                data,
                roi_radii=csd_roi,
                wm_fa_thr=csd_fa_thr,
                gm_fa_thr=gm_fa_thr,
                csf_fa_thr=csf_fa_thr,
                tol=response_tol,
            )
            unique_bvals = unique_bvals_tolerance(gtab.bvals, tol=response_tol)
            response_obj = multi_shell_fiber_response(
                sh_order,
                unique_bvals,
                response_wm,
                response_gm,
                response_csf,
            )
            iso_compartments = int(getattr(response_obj, "iso", 3) or 3)
            iso_override_cfg = tracking_config.get("csd_iso_compartments")
            if iso_override_cfg not in (None, ""):
                try:
                    iso_compartments = max(2, int(iso_override_cfg))
                except Exception:
                    pass
            mt_csd_model = MultiShellDeconvModel(
                gtab,
                response_obj,
                sh_order_max=sh_order,
                iso=iso_compartments,
            )
        except Exception as exc:
            reason = f"MT_CSD init failed: {exc}"
            return _result(
                tensor_model,
                "TENSOR",
                {"fallback_reason": reason, "failed_model": "MT_CSD"},
            )

        response_info = {
            "roi_radius": int(csd_roi),
            "wm_fa_threshold": float(csd_fa_thr),
            "gm_fa_threshold": float(gm_fa_thr),
            "csf_fa_threshold": float(csf_fa_thr),
            "sh_order": sh_order,
            "response_type": "MSMT",
            "iso_compartments": int(iso_compartments),
            "response_tol": int(response_tol),
        }
        return _result(
            mt_csd_model,
            "MT_CSD",
            {"response": response_info},
            {"sh_order": sh_order},
        )

    if normalized_requested == "CSD":
        csd_roi = int(tracking_config.get("csd_response_roi") or 8)
        csd_fa_thr = float(tracking_config.get("csd_response_fa") or 0.7)
        csd_sh_override = tracking_config.get("csd_sh_order")
        csd_sh_order: int
        csd_sh_diagnostics: Optional[dict[str, object]] = None
        if csd_sh_override in (None, 0):
            csd_sh_order, csd_sh_diagnostics = _infer_csd_sh_order(gtab, default=6)
        else:
            csd_sh_order = int(csd_sh_override)
            if csd_sh_order % 2:
                csd_sh_order += 1
            csd_sh_diagnostics = {"auto": False, "selected_order": int(csd_sh_order)}
        try:
            response, ratio = auto_response_ssst(
                gtab,
                data,
                roi_radii=csd_roi,
                fa_thr=csd_fa_thr,
            )
            csd_model = ConstrainedSphericalDeconvModel(
                gtab,
                response,
                sh_order=csd_sh_order,
            )
        except Exception as exc:  # noqa: BLE001 - propagate context via fallback
            reason = f"CSD init failed: {exc}"
            return _result(
                tensor_model,
                "TENSOR",
                {"fallback_reason": reason, "failed_model": "CSD"},
            )

        response_info = {
            "roi_radius": int(csd_roi),
            "fa_threshold": float(csd_fa_thr),
            "ratio": float(ratio),
            "sh_order": int(getattr(csd_model, "sh_order", csd_sh_order)),
        }
        if csd_sh_diagnostics:
            response_info["sh_order_selection"] = csd_sh_diagnostics
        return _result(
            csd_model,
            "CSD",
            {"response": response_info},
            {"sh_order": response_info["sh_order"]},
        )

    if normalized_requested == "QBALL":
        sh_order = int(tracking_config.get("qball_sh_order") or 6)
        if sh_order % 2:
            sh_order += 1
        try:
            qball_model = QballModel(gtab, sh_order=sh_order)
        except Exception as exc:  # noqa: BLE001 - propagate context via fallback
            reason = f"QBALL init failed: {exc}"
            return _result(
                tensor_model,
                "TENSOR",
                {"fallback_reason": reason, "failed_model": "QBALL"},
            )

        return _result(
            qball_model,
            "QBALL",
            {"sh_order": sh_order},
            {"sh_order": sh_order},
        )

    if normalized_requested == "GQI":
        sampling_length = float(tracking_config.get("gqi_sampling_length") or 1.2)
        normalize_peaks = bool(tracking_config.get("gqi_normalize_peaks", False))
        try:
            gqi_model = GeneralizedQSamplingModel(
                gtab,
                method="gqi2",
                sampling_length=sampling_length,
                normalize_peaks=normalize_peaks,
            )
        except Exception as exc:  # noqa: BLE001 - propagate context via fallback
            reason = f"GQI init failed: {exc}"
            return _result(
                tensor_model,
                "TENSOR",
                {"fallback_reason": reason, "failed_model": "GQI"},
            )

        return _result(
            gqi_model,
            "GQI",
            {
                "sampling_length": sampling_length,
                "normalize_peaks": normalize_peaks,
            },
        )

    return _result(
        tensor_model,
        "TENSOR",
        {"fallback_reason": f"Unknown model '{requested}'", "failed_model": requested},
    )


# ---------------------------------------------------------------------------
# DWI normalisation helpers
# ---------------------------------------------------------------------------


def _as_4d_dwi(data: np.ndarray, bvals: np.ndarray) -> np.ndarray:
    """
    Return a 4-D array shaped (X, Y, Z, N) from potentially 5-D inputs.
    We pick the gradient dimension by matching the axis whose length equals
    len(bvals). Any remaining singleton axes beyond Z are squeezed.
    """

    if _DBG:
        _dbg_print(
            f"[tracks][dbg] entering _as_4d_dwi: data.ndim={np.ndim(data)} "
            f"raw_shape={getattr(data,'shape',None)} bvals.size={int(bvals.size)}"
        )

    arr = np.asarray(data)
    if arr.ndim == 4:
        _dbg(f"DWI already 4-D: shape={arr.shape}")
        return arr
    if arr.ndim < 4:
        raise ValueError(f"DWI must be >=4-D, got {arr.ndim}D")

    grad_len = int(bvals.size)
    grad_axis = None
    for ax in range(3, arr.ndim):
        if arr.shape[ax] == grad_len:
            grad_axis = ax
            break
    if grad_axis is None:
        grad_axis = arr.ndim - 1

    arr = np.moveaxis(arr, grad_axis, -1)
    _dbg(f"DWI grad axis chosen: axis={grad_axis}, shape after move={arr.shape}")

    if arr.ndim > 4:
        to_squeeze: list[int] = []
        for ax in range(3, arr.ndim - 1):
            if arr.shape[ax] == 1:
                to_squeeze.append(ax)
        if to_squeeze:
            arr = np.squeeze(arr, axis=tuple(to_squeeze))
            _dbg(f"Squeezed singleton axes {to_squeeze}; new shape={arr.shape}")

    if arr.ndim != 4:
        raise ValueError(
            f"Failed to coerce DWI to 4-D. Shape after coercion: {arr.shape}"
        )
    _dbg(f"DWI final 4-D shape={arr.shape}")
    return arr


def _shape_str(a: np.ndarray) -> str:
    try:
        return "x".join(map(str, a.shape))
    except Exception:
        return "<unknown>"


# ---------------------------------------------------------------------------
# Streamline sanitisation helpers
# ---------------------------------------------------------------------------


def _to_xyz(points: np.ndarray) -> np.ndarray:
    """
    Return Nx3 float32 coordinates from points with possible extra columns.
    If a 4th column exists, treat it as homogeneous w and dehomogenise.
    """

    if _DBG_STREAMLINE_SHAPES:
        _log_streamline_shape(
            f"[tracks][dbg] _to_xyz incoming shape={getattr(points,'shape',None)} "
            f"dtype={getattr(points,'dtype',None)}"
        )

    P = np.asarray(points)
    if P.ndim != 2:
        if _DBG_STREAMLINE_SHAPES:
            _log_streamline_shape(
                f"[tracks][dbg] _to_xyz passthrough ndim={P.ndim}"
            )
        return P
    if P.shape[1] == 3:
        return P.astype(np.float32)
    if P.shape[1] >= 4:
        w = P[:, 3]
        w = np.where((w == 0) | (~np.isfinite(w)), 1.0, w)
        out = (P[:, :3] / w[:, None]).astype(np.float32)
        if _DBG_STREAMLINE_SHAPES:
            _log_streamline_shape("[tracks][dbg] _to_xyz dehomogenised >=4 cols")
        return out
    # fewer than 3 columns, pass through
    if _DBG_STREAMLINE_SHAPES:
        _log_streamline_shape(f"[tracks][dbg] _to_xyz short column shape={P.shape}")
    return P.astype(np.float32)


def _coerce_streamlines_xyz(
    sls: Streamlines,
    *,
    progress_desc: Optional[str] = None,
) -> Streamlines:
    """Ensure every streamline is Nx3 and has at least two points."""

    out = Streamlines()
    idx = 0
    total = 0
    progress = None
    if progress_desc:
        try:
            total = len(sls)
        except TypeError:
            total = 0
        progress = _create_progress_bar(
            total,
            desc=progress_desc,
            unit="sl",
        )

    for sl in sls:
        xyz = _to_xyz(sl)
        if xyz.shape[0] >= 2 and xyz.shape[1] == 3:
            out.append(xyz)
            if _DBG and idx < 3:
                _dbg(f"kept streamline[{idx}] shape={xyz.shape}, "
                     f"head={np.array2string(xyz[:2], precision=3)}")
                idx += 1
        else:
            _dbg(f"dropped streamline shape={xyz.shape}")
        if progress:
            progress.update(1)
    if progress:
        progress.close()
    return out


def _streamline_length(sl: np.ndarray) -> float:
    if sl.shape[0] < 2:
        return 0.0
    diffs = np.diff(sl.astype(np.float32), axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


def _length_stats(lengths: np.ndarray) -> dict[str, Optional[float]]:
    if lengths.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": int(lengths.size),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
    }


def _postprocess_streamlines(
    streamlines: Iterable[np.ndarray],
    *,
    min_length: Optional[float],
    max_length: Optional[float],
    subsample_stride: int,
    subsample_min_count: int = 0,
    debug_blob: Optional[dict] = None,
) -> Streamlines:
    sl_list = Streamlines(streamlines)
    lengths = np.array([_streamline_length(sl) for sl in sl_list], dtype=np.float32)

    if debug_blob is not None:
        debug_blob["raw_count"] = int(len(sl_list))
        debug_blob["raw_length_stats"] = _length_stats(lengths)

    if min_length is not None:
        keep = lengths >= float(min_length)
    else:
        keep = np.ones(len(sl_list), dtype=bool)
    if max_length is not None:
        keep &= lengths <= float(max_length)

    if keep.size and not np.all(keep):
        sl_list = Streamlines([sl for sl, flag in zip(sl_list, keep) if flag])
        lengths = lengths[keep]
        if _DBG:
            _dbg(
                f"length filter applied: kept={len(sl_list)} min={min_length} max={max_length}"
            )

    stride = max(1, int(subsample_stride))
    min_count = max(0, int(subsample_min_count))
    if len(sl_list) <= min_count:
        stride = 1
        if debug_blob is not None and min_count:
            debug_blob["subsample_skipped"] = {
                "count": len(sl_list),
                "threshold": min_count,
            }
    if stride > 1 and len(sl_list) > 0:
        sl_list = Streamlines(sl_list[::stride])
        lengths = lengths[::stride]
        if _DBG:
            _dbg(f"subsampled streamlines with stride={stride}; count={len(sl_list)}")

    if debug_blob is not None:
        debug_blob["filtered_count"] = int(len(sl_list))
        debug_blob["filtered_length_stats"] = _length_stats(lengths)

    return sl_list


def _run_tracking_attempt(
    *,
    label: str,
    peaks,
    seeds: np.ndarray,
    voxel_to_world: np.ndarray,
    fa_volume: np.ndarray,
    stop_threshold: float,
    filter_config: dict[str, Optional[float]],
    tracking_strategy: Optional[dict[str, object]] = None,
    rng_seed: Optional[int] = None,
) -> tuple[Streamlines, dict]:
    attempt: dict = {
        "label": label,
        "stop_threshold": float(stop_threshold),
        "seed_count": int(len(seeds)) if hasattr(seeds, "__len__") else 0,
        "filter": {
            "min_length": filter_config.get("min_length"),
            "max_length": filter_config.get("max_length"),
            "subsample_stride": filter_config.get("subsample_stride"),
        },
        "backend": "pft" if tracking_strategy and tracking_strategy.get("use_pft") else "local",
    }

    stopping_criterion = _select_stopping_criterion(
        stop_threshold,
        fa_volume,
        tracking_strategy,
    )
    try:
        if tracking_strategy and tracking_strategy.get("use_pft"):
            streamline_generator = _run_particle_filtering_tracking(
                peaks,
                seeds,
                voxel_to_world,
                stopping_criterion=stopping_criterion,
                config=tracking_strategy.get("pft_kwargs", {}),
                rng_seed=rng_seed,
            )
        else:
            streamline_generator = LocalTracking(
                peaks,
                stopping_criterion,
                seeds,
                affine=voxel_to_world,
                step_size=0.5,
                return_all=False,
            )
    except Exception:
        if _DBG:
            _dbg_print("[tracks][err] Tracking backend construction failed")
            _dbg_print(f"[tracks][dbg] voxel_to_world shape={voxel_to_world.shape}")
            try:
                _dbg_print(
                    f"[tracks][dbg] seeds dtype/shape={getattr(seeds, 'dtype', None)}/{getattr(seeds, 'shape', None)}"
                )
            except Exception:
                pass
            try:
                _dbg_print(
                    f"[tracks][dbg] stopping_criterion data shape={getattr(stopping_criterion, 'data', None).shape}"
                )
            except Exception:
                pass
            _dbg_print(traceback.format_exc())
        raise
    raw_streamlines = _coerce_streamlines_xyz(Streamlines(streamline_generator))

    processed = _postprocess_streamlines(
        raw_streamlines,
        min_length=filter_config.get("min_length"),
        max_length=filter_config.get("max_length"),
        subsample_stride=int(filter_config.get("subsample_stride", 1) or 1),
        subsample_min_count=int(filter_config.get("subsample_min_count", 0) or 0),
        debug_blob=attempt,
    )

    return processed, attempt


def _orientation_colour_from_vector(vec: np.ndarray) -> np.ndarray:
    if not np.any(np.isfinite(vec)):
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    if vec.size < 3:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    angle = float(np.arctan2(vec[1], vec[0]))
    hue = (angle + np.pi) / (2 * np.pi)
    rgb = np.array(colorsys.hsv_to_rgb(hue, 0.85, 1.0), dtype=np.float32)
    return rgb


def _streamline_orientation_colours(streamlines: Streamlines) -> list[np.ndarray]:
    colours: list[np.ndarray] = []
    for sl in streamlines:
        if sl.shape[0] < 2:
            rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            vec = sl[-1] - sl[0]
            rgb = _orientation_colour_from_vector(vec)
        colours.append(np.repeat(rgb[None, :], sl.shape[0], axis=0))
    return colours


def _streamline_fraction_inside(
    streamlines: Streamlines,
    affine: Optional[np.ndarray],
    shape: Optional[Sequence[int]],
) -> float:
    """Return the fraction of points lying within the provided ``shape``."""

    if affine is None or shape is None:
        return 0.0

    try:
        inv_affine = np.linalg.inv(_canonical_affine(affine))
    except Exception:
        return 0.0

    if len(shape) > 3:
        shape = shape[:3]
    bounds = np.array(shape, dtype=np.float64) - 0.5

    inside = 0
    total = 0
    for sl in streamlines:
        if sl.shape[0] == 0:
            continue
        pts = nib.affines.apply_affine(inv_affine, sl)
        total += pts.shape[0]
        mask = (
            (pts[:, 0] >= -0.5)
            & (pts[:, 0] <= bounds[0])
            & (pts[:, 1] >= -0.5)
            & (pts[:, 1] <= bounds[1])
            & (pts[:, 2] >= -0.5)
            & (pts[:, 2] <= bounds[2])
        )
        inside += int(np.count_nonzero(mask))

    if total == 0:
        return 0.0

    return inside / float(total)


def _load_streamlines_world(
    path: str,
    *,
    expected_affine: Optional[np.ndarray] = None,
    expected_shape: Optional[Sequence[int]] = None,
    progress_desc: Optional[str] = None,
) -> Optional[Streamlines]:
    """Load streamlines ensuring they are expressed in world space.

    Older releases wrote world-space streamlines while also recording the
    diffusion affine in the tractogram header. ``tractogram.to_world`` would
    therefore apply the affine twice, pushing fibres far outside the field of
    view. To remain backward compatible we evaluate both the raw stored
    coordinates and the transformed-to-world version, selecting whichever
    overlaps the diffusion volume best.
    """

    try:
        tractogram_file = nib_streamlines.load(path)
    except Exception:
        return None

    tractogram = tractogram_file.tractogram

    raw_streamlines = _coerce_streamlines_xyz(
        Streamlines(tractogram.streamlines),
        progress_desc=progress_desc,
    )
    transformed_streamlines: Optional[Streamlines] = None

    try:
        transformed = tractogram.to_world(lazy=False)
    except Exception:
        if _DBG:
            _dbg_print(
                "[tracks][dbg] tractogram.to_world() failed; proceeding without transform"
            )
    else:
        transformed_streamlines = _coerce_streamlines_xyz(
            Streamlines(transformed.streamlines),
            progress_desc=(
                f"{progress_desc} (world)" if progress_desc else None
            ),
        )

    if expected_affine is not None and expected_shape is not None:
        candidates: list[tuple[str, Streamlines]] = [("raw", raw_streamlines)]
        if transformed_streamlines is not None:
            candidates.append(("world", transformed_streamlines))

        best_label = "raw"
        best_streamlines = raw_streamlines
        best_fraction = _streamline_fraction_inside(
            raw_streamlines, expected_affine, expected_shape
        )

        for label, candidate in candidates[1:]:
            fraction = _streamline_fraction_inside(
                candidate, expected_affine, expected_shape
            )
            if fraction > best_fraction + 1e-6:
                best_fraction = fraction
                best_streamlines = candidate
                best_label = label

        if _DBG:
            _dbg(
                f"streamline load candidate={best_label} inside={best_fraction:.3f}"
            )
        return best_streamlines

    if transformed_streamlines is not None:
        return transformed_streamlines

    return raw_streamlines


def _load_background_volume(
    nifti_path: str,
    reference_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Return a background volume resampled into ``reference_img`` space."""

    img = nib.load(nifti_path)
    if img.shape[:3] == reference_img.shape[:3]:
        return img

    if resample_from_to is None:
        raise RuntimeError(
            "nibabel.processing.resample_from_to is required for anatomical overlays"
        )

    spatial_shape = reference_img.shape
    if len(spatial_shape) > 3:
        spatial_shape = spatial_shape[:3]
    target = (tuple(int(dim) for dim in spatial_shape), _canonical_affine(reference_img.affine))
    resampled = resample_from_to(img, target, order=1)
    return resampled


def _seed_points_from_mask(
    mask: np.ndarray,
    voxel_to_world: np.ndarray,
    *,
    density: Union[int, Sequence[int]] = 1,
    max_points: Optional[int] = None,
    jitter_mm: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return evenly distributed seed points from ``mask`` with optional subsampling."""

    affine = _canonical_affine(voxel_to_world)
    seeds = np.asarray(seeds_from_mask(mask, affine, density=density), dtype=np.float32)
    if seeds.size == 0:
        return seeds

    if max_points is not None and max_points > 0 and len(seeds) > max_points:
        generator = rng or np.random.default_rng()
        idx = generator.choice(len(seeds), size=max_points, replace=False)
        seeds = seeds[idx]

    if jitter_mm and jitter_mm > 0:
        generator = rng or np.random.default_rng()
        noise = generator.uniform(-jitter_mm, jitter_mm, size=seeds.shape)
        seeds = seeds + noise.astype(np.float32)

    return seeds


def _sanitize_seed_points(
    seeds: np.ndarray,
    voxel_to_world: np.ndarray,
    volume_shape: Sequence[int],
    *,
    hard_margin_vox: float = 0.5,
    safety_margin_vox: float = 0.1,
) -> tuple[np.ndarray, int, int]:
    """Keep / nudge seeds so they remain inside ``volume_shape`` bounds."""

    if seeds.size == 0:
        return seeds, 0, 0

    try:
        canonical_affine = _canonical_affine(voxel_to_world)
        inv_affine = np.linalg.inv(canonical_affine)
    except np.linalg.LinAlgError:
        return seeds, 0, 0

    coords = nib.affines.apply_affine(inv_affine, seeds)
    shape = np.asarray(volume_shape[:3], dtype=np.float32)

    hard_lower = -0.5 - float(hard_margin_vox)
    hard_upper = shape - 0.5 + float(hard_margin_vox)
    hard_mask = (
        (coords[:, 0] >= hard_lower)
        & (coords[:, 0] <= hard_upper[0])
        & (coords[:, 1] >= hard_lower)
        & (coords[:, 1] <= hard_upper[1])
        & (coords[:, 2] >= hard_lower)
        & (coords[:, 2] <= hard_upper[2])
    )

    filtered = coords[hard_mask]
    dropped = int(coords.shape[0] - filtered.shape[0])
    if filtered.size == 0:
        return np.empty((0, 3), dtype=np.float32), dropped, 0

    safety_lower = -0.5 + float(safety_margin_vox)
    safety_upper = shape - 0.5 - float(safety_margin_vox)
    adjusted = 0
    clipped = filtered.copy()
    for axis in range(3):
        upper_axis = safety_upper[axis]
        if upper_axis <= safety_lower:
            continue
        clipped_axis = np.clip(clipped[:, axis], safety_lower, upper_axis)
        adjusted += int(np.count_nonzero(clipped_axis != clipped[:, axis]))
        clipped[:, axis] = clipped_axis

    sanitized = nib.affines.apply_affine(canonical_affine, clipped)
    return sanitized.astype(np.float32), dropped, adjusted


def _build_tracking_attempt_plan(
    tracking_config: dict[str, float],
    filter_defaults: dict[str, Optional[float]],
) -> list[dict[str, object]]:
    """Return a list of adaptive tracking attempts, honouring env overrides."""

    custom_plan = os.environ.get("P_BRAIN_TRACK_ATTEMPTS")
    if custom_plan:
        try:
            parsed = json.loads(custom_plan)
        except Exception:
            _dbg("failed to parse P_BRAIN_TRACK_ATTEMPTS; using defaults")
        else:
            attempts: list[dict[str, object]] = []
            for idx, entry in enumerate(parsed if isinstance(parsed, list) else []):
                if not isinstance(entry, dict):
                    continue
                cfg = dict(entry)
                cfg.setdefault("label", f"custom_{idx+1}")
                cfg.setdefault("seed_fa_threshold", tracking_config["wm_threshold"])
                cfg.setdefault("seed_density", tracking_config["seed_density"])
                cfg.setdefault("stop_threshold", tracking_config["stop_threshold"])
                attempts.append(cfg)
            if attempts:
                return attempts

    subsample_stride = max(1, int(filter_defaults.get("subsample_stride", 1) or 1))
    subsample_min_count = max(0, int(filter_defaults.get("subsample_min_count") or 0))
    min_length = filter_defaults.get("min_length")
    max_length = filter_defaults.get("max_length")
    fallback_min_length = tracking_config.get("fallback_min_length")
    fallback_stride = tracking_config.get("fallback_subsample_stride")

    seed_density = max(1, int(tracking_config.get("seed_density", 1)))

    plan: list[dict[str, object]] = [
        {
            "label": "core_high_fa",
            "seed_fa_threshold": tracking_config["wm_threshold"],
            "seed_density": seed_density,
            "stop_threshold": tracking_config["stop_threshold"],
            "min_length": min_length,
            "max_length": max_length,
            "subsample_stride": subsample_stride,
            "subsample_min_count": subsample_min_count,
            "mask_dilation_iters": 0,
            "seed_jitter_mm": 0.0,
            "max_seeds": None,
        },
        {
            "label": "boost_mid_fa",
            "seed_fa_threshold": max(0.08, tracking_config["wm_threshold"] * 0.85),
            "seed_density": seed_density,
            "stop_threshold": max(0.08, tracking_config["stop_threshold"] * 0.9),
            "min_length": min_length,
            "max_length": max_length,
            "subsample_stride": max(1, subsample_stride // 2),
            "subsample_min_count": subsample_min_count // 2,
            "mask_dilation_iters": 1,
            "seed_jitter_mm": 0.35,
            "max_seeds": tracking_config.get("max_seed_points"),
        },
        {
            "label": "explore_low_fa",
            "seed_fa_threshold": max(0.06, tracking_config["wm_threshold"] * 0.65),
            "seed_density": seed_density,
            "stop_threshold": tracking_config["fallback_stop_threshold"],
            "min_length": fallback_min_length,
            "max_length": max_length,
            "subsample_stride": int(fallback_stride) if fallback_stride else subsample_stride,
            "subsample_min_count": 0,
            "mask_dilation_iters": 2,
            "seed_jitter_mm": 0.75,
            "max_seeds": tracking_config.get("max_seed_points"),
        },
    ]
    return plan


def _seed_mask_for_threshold(
    fa_volume: np.ndarray,
    threshold: float,
    dilation_iters: int,
    cache: dict[tuple[float, int], np.ndarray],
) -> np.ndarray:
    """Return / cache a boolean mask for the provided FA threshold."""

    key = (float(threshold), int(max(0, dilation_iters)))
    cached = cache.get(key)
    if cached is not None:
        return cached

    mask = np.asarray(fa_volume > float(threshold), dtype=bool)
    iterations = key[1]
    if iterations > 0:
        if binary_dilation is None:
            global _DILATION_WARNING_EMITTED
            if not _DILATION_WARNING_EMITTED and _DBG:
                _dbg("scipy unavailable – seed mask dilation skipped")
                _DILATION_WARNING_EMITTED = True
        else:
            mask = binary_dilation(mask, iterations=iterations)

    cache[key] = mask
    return mask


def _ensure_image_directory(
    image_directory: Optional[str], analysis_directory: str
) -> Optional[str]:
    """Return a usable image directory, falling back to ``analysis_directory``."""

    if image_directory:
        os.makedirs(image_directory, exist_ok=True)
        return image_directory

    parent = os.path.dirname(os.path.abspath(analysis_directory))
    fallback = os.path.join(parent, "Images")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _render_with_fury(
    streamlines: Streamlines,
    output_path: str,
    *,
    background_color: tuple[float, float, float] = (0.02, 0.02, 0.05),
) -> None:
    """Render colourful streamlines using ``fury`` for rich shading and lighting."""

    streamlines = Streamlines(streamlines)
    scene = dipy_window.Scene()
    if _DBG:
        _dbg_print(
            f"[tracks][dbg] FURY scene created. streamlines_count={len(streamlines)}"
        )
    scene.background(background_color)

    try:
        colours = _streamline_orientation_colours(streamlines)
        stream_actor = dipy_actor.streamtube(
            streamlines,
            colors=colours,
            linewidth=0.3,
        )
        _set_actor_opacity(stream_actor)
        scene.add(stream_actor)
    except Exception:
        import traceback as _tb
        try:
            stream_actor = dipy_actor.line(
                streamlines,
                colors=colours,
                linewidth=1.0,
            )
            _set_actor_opacity(stream_actor)
            scene.add(stream_actor)
        except Exception:
            try:
                fallback_colours = dipy_colormap.line_colors(streamlines)
                stream_actor = dipy_actor.line(
                    streamlines,
                    colors=fallback_colours,
                    linewidth=1.0,
                )
                _set_actor_opacity(stream_actor)
                scene.add(stream_actor)
            except Exception:
                raise RuntimeError("FURY line actor build failed:\n" + _tb.format_exc())

    # Gently rotate the scene so fibres are rendered with depth cues.
    scene.reset_camera()
    scene.pitch(-15)
    scene.yaw(20)
    scene.zoom(1.2)

    snapshot_kwargs = {"fname": output_path, "size": (1600, 1600)}
    try:
        signature = inspect.signature(dipy_window.snapshot)
    except (TypeError, ValueError):  # pragma: no cover - rare introspection issues
        signature = None
    if signature is not None and "offscreen" in signature.parameters:
        snapshot_kwargs["offscreen"] = True

    try:
        dipy_window.snapshot(scene, **snapshot_kwargs)
    except Exception:
        import traceback as _tb
        raise RuntimeError("FURY snapshot failed:\n" + _tb.format_exc())
    finally:
        clear = getattr(scene, "clear", None)
        if callable(clear):  # pragma: no branch - safety guard for older fury
            clear()


def _render_streamline_animation_fury(
    streamlines: list[np.ndarray],
    output_path: str,
    *,
    frame_count: int,
    rotation_degrees: float,
    fps: int,
    progressive: bool,
    elevation_degrees: float,
    ffmpeg_binary: Optional[str],
    orbit_center: np.ndarray,
    orbit_radius: float,
    background_color: tuple[float, float, float] = (0.02, 0.02, 0.05),
) -> bool:
    if not streamlines:
        return False

    size = (1200, 1200)
    total = len(streamlines)
    base_yaw = 20.0
    center = np.asarray(orbit_center[:3], dtype=np.float32)
    radius = float(max(1.0, orbit_radius))
    elev_rad = np.deg2rad(float(elevation_degrees))
    sin_elev = np.sin(elev_rad)
    cos_elev = np.cos(elev_rad)

    with _open_animation_writer(output_path, fps, ffmpeg_binary=ffmpeg_binary) as writer:
        for frame_idx in range(frame_count):
            frac = (frame_idx + 1) / float(frame_count)
            reveal_fraction = frac if progressive else 1.0
            reveal_count = max(1, int(total * reveal_fraction))
            subset = Streamlines(streamlines[:reveal_count])

            scene = dipy_window.Scene()
            scene.background(background_color)
            colours = _streamline_orientation_colours(subset)
            try:
                actor = dipy_actor.streamtube(subset, colors=colours, linewidth=0.3)
            except Exception:
                actor = dipy_actor.line(subset, colors=colours, linewidth=1.0)
            _set_actor_opacity(actor)
            scene.add(actor)

            yaw_rad = np.deg2rad(base_yaw + rotation_degrees * frac)
            cam_pos = np.array(
                [
                    center[0] + radius * np.cos(yaw_rad) * cos_elev,
                    center[1] + radius * np.sin(yaw_rad) * cos_elev,
                    center[2] + radius * sin_elev,
                ],
                dtype=np.float32,
            )
            camera = scene.camera()
            try:
                camera.SetFocalPoint(float(center[0]), float(center[1]), float(center[2]))
                camera.SetPosition(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
                camera.SetViewUp(0.0, 0.0, 1.0)
            except Exception:
                scene.reset_camera()
                scene.pitch(float(elevation_degrees))
                scene.yaw(base_yaw + rotation_degrees * frac)
            reset_clip = getattr(scene, "reset_clipping_range", None)
            if callable(reset_clip):
                reset_clip()

            frame = dipy_window.snapshot(
                scene,
                size=size,
                offscreen=True,
            )
            writer.append_data(np.asarray(frame, dtype=np.uint8))
            clear = getattr(scene, "clear", None)
            if callable(clear):
                clear()

    return True


def _fury_render_worker(
    tract_path: str,
    output_path: str,
    background_color: tuple[float, float, float],
    result_queue,
) -> None:
    """Helper executed in a subprocess to isolate FURY crashes."""

    try:
        tractogram = nib_streamlines.load(tract_path).tractogram
        streamlines = _coerce_streamlines_xyz(Streamlines(tractogram.streamlines))
        if _DBG:
            print(f"[tracks][dbg] FURY worker loaded {len(streamlines)} streamlines", flush=True)
        _render_with_fury(
            streamlines,
            output_path,
            background_color=background_color,
        )
    except KeyboardInterrupt:
        raise
    except Exception:
        import traceback as _tb
        tb = _tb.format_exc()
        # include a tiny bit of state; affine shapes etc are printed earlier already
        msg = "[worker traceback follows]\n" + tb
        result_queue.put((False, msg))
    else:
        result_queue.put((True, None))


def _env_flag_disabled(value: str) -> bool:
    """Return ``True`` when ``value`` represents a disabled boolean flag."""

    return value.strip().lower() in {"0", "false", "no", "off"}


def _should_use_fury() -> bool:
    """Determine whether FURY rendering should be attempted.

    Default OFF on Apple Silicon macOS due to VTK/Metal offscreen quirks
    that yield transform shape mismatches. Allow explicit opt-in via
    P_BRAIN_ENABLE_FURY=1. Explicit disable via P_BRAIN_DISABLE_FURY=1.
    """

    if not _FURY_AVAILABLE:
        return False

    flag = os.environ.get("P_BRAIN_ENABLE_FURY", "")
    if flag:
        return not _env_flag_disabled(flag)

    disable_flag = os.environ.get("P_BRAIN_DISABLE_FURY", "")
    if disable_flag:
        return not disable_flag.strip().lower() in {"1", "true", "yes", "on"}

    # Platform default: avoid FURY on Apple Silicon macOS
    try:
        if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
            if _DBG:
                _dbg_print("[tracks][dbg] FURY default OFF on macOS arm64; use P_BRAIN_ENABLE_FURY=1 to force ON")
            return False
    except Exception:
        pass
    return True


def _render_with_fury_isolated(
    tract_path: str,
    output_path: str,
    *,
    background_color: tuple[float, float, float] = (0.02, 0.02, 0.05),
) -> bool:
    """Render streamlines via FURY inside a subprocess, returning success."""

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_fury_render_worker,
        args=(tract_path, output_path, background_color, result_queue),
    )
    process.start()
    process.join()

    success = False
    message: Optional[str] = None

    if process.exitcode == 0:
        try:
            success, message = result_queue.get_nowait()
        except queue.Empty:  # pragma: no cover - unexpected but tolerable
            success = True
    else:
        success = False
        if process.exitcode is not None:
            if process.exitcode < 0:
                message = f"terminated by signal {-process.exitcode}"
            elif process.exitcode > 0:
                message = f"exited with status {process.exitcode}"

    if not success:
        prefix = "[tracks] FURY rendering failed"
        if message:
            # message already includes traceback from worker
            print(f"{prefix}:\n{message}")
        else:
            print(f"{prefix} – falling back to matplotlib")

    return success


def _render_with_matplotlib(
    streamlines: Streamlines,
    output_path: str,
) -> None:
    """Fallback renderer relying on matplotlib when ``fury`` is unavailable."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - side effect enables 3D plots

    fig = plt.figure(figsize=(10, 10))
    if _DBG:
        _dbg_print(
            f"[tracks][dbg] matplotlib fallback renderer. "
            f"streamlines_count={sum(1 for _ in streamlines)}"
        )
        # reset iterator by re-wrapping (Streamlines is re-iterable)
        streamlines = Streamlines(streamlines)
    ax = fig.add_subplot(111, projection="3d")

    for sl in streamlines:
        sl = _to_xyz(sl)
        if len(sl) < 2:
            continue
        segments = np.diff(sl, axis=0)
        norms = np.linalg.norm(segments, axis=1, keepdims=True)
        directions = np.zeros_like(segments)
        valid = norms[:, 0] > 0
        directions[valid] = segments[valid] / norms[valid]
        colour = np.abs(directions.mean(axis=0)) if valid.any() else np.array([0.5, 0.5, 0.5])
        colour = np.clip(colour, 0.1, 1.0)
        ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], color=colour, linewidth=0.6, alpha=0.8)

    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.15)

    ax.view_init(elev=75, azim=90)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor="black")
    plt.close(fig)


class _FFmpegPipeWriter:
    def __init__(self, output_path: str, fps: int, ffmpeg_binary: str):
        self._output_path = output_path
        self._fps = max(1, int(fps))
        self._ffmpeg_binary = ffmpeg_binary
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    def _ensure_started(self, width: int, height: int) -> None:
        if self._process is not None:
            return
        cmd = [
            self._ffmpeg_binary,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self._fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            self._output_path,
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._width = width
        self._height = height

    def append_data(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("FFmpeg writer expects RGB images")
        height, width, _ = frame.shape
        if self._width is not None and (self._width != width or self._height != height):
            raise ValueError("Frame size changed during FFmpeg encoding")
        self._ensure_started(width, height)
        assert self._process and self._process.stdin is not None
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process is None:
            return
        stdin = self._process.stdin
        if stdin and not stdin.closed:
            stdin.close()
            # subprocess.communicate() flushes stdin when present; avoid flushing a closed pipe.
            try:
                self._process.stdin = None
            except Exception:
                pass
        stdout, stderr = self._process.communicate()
        if self._process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {self._process.returncode}: {stderr.decode(errors='ignore')[:400]}"
            )
        self._process = None


@contextmanager
def _open_animation_writer(
    output_path: str,
    fps: int,
    *,
    ffmpeg_binary: Optional[str],
) -> Iterator[object]:
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp4" and not _IMAGEIO_HAS_FFMPEG:
        if not ffmpeg_binary:
            raise RuntimeError(
                "MP4 animation export requires ffmpeg. Install imageio-ffmpeg or set P_BRAIN_TRACK_ANIMATION_FORMAT=gif."
            )
        writer = _FFmpegPipeWriter(output_path, fps, ffmpeg_binary)
        try:
            yield writer
        finally:
            writer.close()
        return

    if imageio is None:
        raise RuntimeError("imageio is required for animation export")

    writer_kwargs = _imageio_writer_kwargs(output_path, fps)
    with imageio.get_writer(output_path, **writer_kwargs) as writer:
        yield writer


def _imageio_writer_kwargs(output_path: str, fps: int) -> dict[str, object]:
    """Infer writer arguments compatible with the chosen container."""

    fps = max(1, int(fps))
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        duration_ms = max(1, int(round(1000.0 / fps)))
        return {"duration": duration_ms, "loop": 0}
    return {"fps": fps}


def _compute_streamline_framing(streamlines: Sequence[np.ndarray]) -> Optional[tuple[np.ndarray, float]]:
    min_coords: Optional[np.ndarray] = None
    max_coords: Optional[np.ndarray] = None
    for sl in streamlines:
        arr = np.asarray(sl, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        arr = arr[:, :3]
        local_min = arr.min(axis=0)
        local_max = arr.max(axis=0)
        if min_coords is None:
            min_coords = local_min
            max_coords = local_max
        else:
            min_coords = np.minimum(min_coords, local_min)
            max_coords = np.maximum(max_coords, local_max)
    if min_coords is None or max_coords is None:
        return None

    center = (min_coords + max_coords) / 2.0
    span = np.max(max_coords - min_coords)
    radius = max(1.0, float(span) * 0.65)
    return center.astype(np.float32), radius


def _render_streamline_animation_matplotlib(
    streamlines: list[np.ndarray],
    output_path: str,
    *,
    frame_count: int,
    rotation_degrees: float,
    fps: int,
    progressive: bool,
    elevation_degrees: float,
    ffmpeg_binary: Optional[str],
    orbit_center: np.ndarray,
    orbit_radius: float,
) -> bool:
    if not streamlines:
        return False

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    total = len(streamlines)
    base_azim = 90.0

    with _open_animation_writer(output_path, fps, ffmpeg_binary=ffmpeg_binary) as writer:
        for frame_idx in range(frame_count):
            frac = (frame_idx + 1) / float(frame_count)
            reveal_fraction = frac if progressive else 1.0
            reveal_count = max(1, int(total * reveal_fraction))
            subset = streamlines[:reveal_count]

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("black")
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            radius = float(max(1.0, orbit_radius))
            cx, cy, cz = map(float, orbit_center[:3])
            ax.set_xlim(cx - radius, cx + radius)
            ax.set_ylim(cy - radius, cy + radius)
            ax.set_zlim(cz - radius, cz + radius)

            for sl in subset:
                if len(sl) < 2:
                    continue
                sl = _to_xyz(sl)
                segments = np.diff(sl, axis=0)
                norms = np.linalg.norm(segments, axis=1, keepdims=True)
                valid = norms[:, 0] > 0
                directions = np.zeros_like(segments)
                directions[valid] = segments[valid] / norms[valid]
                colour = np.abs(directions.mean(axis=0)) if valid.any() else np.array([0.5, 0.5, 0.5])
                colour = np.clip(colour, 0.1, 1.0)
                ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], color=colour, linewidth=0.6, alpha=0.85)

            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
                axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.15)

            ax.view_init(elev=float(elevation_degrees), azim=base_azim + rotation_degrees * frac)
            ax.set_axis_off()
            fig.tight_layout()

            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(height, width, 3)
            writer.append_data(frame)
            plt.close(fig)

    return True


def _render_streamline_animation(
    streamlines: Streamlines,
    output_path: str,
    *,
    frame_count: int,
    rotation_degrees: float,
    fps: int,
    progressive: bool,
    elevation_degrees: float,
    ffmpeg_binary: Optional[str],
    rng_seed: Optional[int] = None,
) -> bool:
    if imageio is None:
        raise RuntimeError("imageio is not available for animation rendering")

    streamlines = Streamlines(streamlines)
    streamline_list = [np.asarray(sl) for sl in streamlines]
    if not streamline_list:
        return False

    framing = _compute_streamline_framing(streamline_list)
    if framing is None:
        return False
    orbit_center, orbit_radius = framing

    order = np.arange(len(streamline_list))
    if rng_seed is not None:
        rng = np.random.default_rng(int(rng_seed) & 0xFFFFFFFF)
        rng.shuffle(order)
    ordered_streamlines = [streamline_list[idx] for idx in order]

    use_fury = _should_use_fury()
    try:
        if use_fury:
            return _render_streamline_animation_fury(
                ordered_streamlines,
                output_path,
                frame_count=frame_count,
                rotation_degrees=rotation_degrees,
                fps=fps,
                progressive=progressive,
                elevation_degrees=elevation_degrees,
                ffmpeg_binary=ffmpeg_binary,
                orbit_center=orbit_center,
                orbit_radius=orbit_radius,
            )
        return _render_streamline_animation_matplotlib(
            ordered_streamlines,
            output_path,
            frame_count=frame_count,
            rotation_degrees=rotation_degrees,
            fps=fps,
            progressive=progressive,
            elevation_degrees=elevation_degrees,
            ffmpeg_binary=ffmpeg_binary,
            orbit_center=orbit_center,
            orbit_radius=orbit_radius,
        )
    except RuntimeError as err:
        raise err
    except Exception:
        if _DBG:
            _dbg_print("[tracks][err] animation rendering failed")
            _dbg_print(traceback.format_exc())
        raise


def _streamline_voxel_stats(
    streamlines: Iterable[np.ndarray],
    affine: np.ndarray,
    volume_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return accumulated colour sums and densities for streamline segments."""

    aff4 = _canonical_affine(affine)
    inv_affine = np.linalg.inv(aff4)
    _dbg(f"overlay inv_affine shape={inv_affine.shape}")
    counts = np.zeros(volume_shape, dtype=np.float32)
    colour_sum = np.zeros(volume_shape + (3,), dtype=np.float32)

    for streamline in streamlines:
        if streamline.shape[0] < 2:
            continue
        streamline = _to_xyz(streamline)

        world_a = streamline[:-1]
        world_b = streamline[1:]
        midpoints_world = 0.5 * (world_a + world_b)

        directions = world_b - world_a
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        valid = norms[:, 0] > 0
        if not np.any(valid):
            continue

        directions = directions[valid] / norms[valid]
        midpoints_world = midpoints_world[valid]

        colours = np.abs(directions)

        midpoints_vox = nib.affines.apply_affine(inv_affine, midpoints_world)
        indices = np.round(midpoints_vox).astype(int)

        valid_mask = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < volume_shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < volume_shape[1])
            & (indices[:, 2] >= 0)
            & (indices[:, 2] < volume_shape[2])
        )

        if not np.any(valid_mask):
            continue

        indices = indices[valid_mask]
        colours = colours[valid_mask]

        counts[indices[:, 0], indices[:, 1], indices[:, 2]] += 1.0
        colour_sum[indices[:, 0], indices[:, 1], indices[:, 2]] += colours

    return colour_sum, counts


def _aggregate_streamline_colours(
    streamlines: Iterable[np.ndarray],
    affine: np.ndarray,
    volume_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-voxel colour averages and streamline densities."""

    colour_sum, counts = _streamline_voxel_stats(streamlines, affine, volume_shape)

    with np.errstate(invalid="ignore", divide="ignore"):
        colour_avg = np.zeros_like(colour_sum)
        mask = counts > 0
        colour_avg[mask] = colour_sum[mask] / counts[mask][..., None]

    return colour_avg, counts


_MONTAGE_ROWS = 2
_MONTAGE_COLS = 5
_MONTAGE_ROTATE = 1
_MONTAGE_PAD = 3


def _brain_mask_from_tissue_maps(
    tissue_maps: Optional[dict[str, np.ndarray]],
    *,
    keys: Optional[Sequence[str]] = None,
) -> np.ndarray | None:
    """Return a boolean brain mask derived from tissue probability volumes."""

    if not tissue_maps:
        return None

    if keys is None:
        keys = ("wm", "gm", "csf")

    volumes: list[np.ndarray] = []
    for key in keys:
        vol = tissue_maps.get(key)
        if vol is None:
            continue
        arr = np.asarray(vol, dtype=np.float32)
        if arr.ndim != 3:
            continue
        volumes.append(arr)

    if not volumes:
        return None

    reference_shape = volumes[0].shape
    mask = np.zeros(reference_shape, dtype=bool)
    for arr in volumes:
        if arr.shape != reference_shape:
            return None
        mask |= arr > 0.05

    return mask if mask.any() else None


def _tissue_map_coverage_stats(
    target_mask: np.ndarray,
    tissue_maps: Optional[dict[str, np.ndarray]],
    *,
    keys: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Return coverage metrics for ``target_mask`` within anatomical tissue maps."""

    stats: dict[str, object] = {
        "target_voxels": 0,
        "supported_voxels": 0,
        "fraction_supported": 0.0,
    }
    if tissue_maps is None:
        return stats

    target = np.asarray(target_mask, dtype=bool)
    total = int(np.count_nonzero(target))
    stats["target_voxels"] = total
    if total == 0:
        return stats

    union_mask = _brain_mask_from_tissue_maps(tissue_maps, keys=keys)
    if union_mask is None or union_mask.shape != target.shape:
        return stats

    supported_mask = np.asarray(union_mask, dtype=bool)
    supported = int(np.count_nonzero(target & supported_mask))
    stats["supported_voxels"] = supported
    stats["fraction_supported"] = float(supported / total) if total else 0.0

    axis_profiles: list[dict[str, int]] = []
    axes = (0, 1, 2)
    for axis in axes:
        other_axes = tuple(idx for idx in axes if idx != axis)
        target_profile = np.any(target, axis=other_axes)
        supported_profile = np.any(target & supported_mask, axis=other_axes)
        axis_profiles.append(
            {
                "axis": axis,
                "target_slices": int(np.count_nonzero(target_profile)),
                "supported_slices": int(np.count_nonzero(supported_profile)),
            }
        )
    stats["axis_profiles"] = axis_profiles
    stats["unsupported_voxels"] = total - supported
    return stats


def _save_mask_qc_volumes(
    diffusion_dir: str,
    voxel_to_world: np.ndarray,
    wm_mask: np.ndarray,
    tissue_maps: Optional[dict[str, np.ndarray]],
) -> None:
    """Persist simple NIfTI masks to inspect ACT/PFT coverage.

    When ``P_BRAIN_TRACK_SAVE_MASK_QC=1`` is set, this helper writes
    three volumes into ``diffusion_dir``:

    - ``fa_wm_mask.nii.gz``: binary FA-based WM seed mask.
    - ``anatomical_union_mask.nii.gz``: union of anatomical tissue maps.
    - ``fa_wm_supported_mask.nii.gz``: intersection between the two.

    These masks live on the diffusion grid and share the DWI affine so
    they can be overlaid directly in external viewers.
    """

    if not _env_bool("P_BRAIN_TRACK_SAVE_MASK_QC", False):
        return
    if tissue_maps is None:
        return

    try:
        os.makedirs(diffusion_dir, exist_ok=True)
        wm_bool = np.asarray(wm_mask, dtype=bool)
        union_mask = _brain_mask_from_tissue_maps(tissue_maps)
        if union_mask is None or union_mask.shape != wm_bool.shape:
            return
        union_bool = np.asarray(union_mask, dtype=bool)
        supported = wm_bool & union_bool

        def _save(name: str, data: np.ndarray) -> None:
            img = nib.Nifti1Image(
                data.astype(np.uint8),
                _canonical_affine(voxel_to_world),
            )
            nib.save(img, os.path.join(diffusion_dir, name))

        _save("fa_wm_mask.nii.gz", wm_bool)
        _save("anatomical_union_mask.nii.gz", union_bool)
        _save("fa_wm_supported_mask.nii.gz", supported)
    except Exception:
        if _DBG:
            _dbg_print("[tracks][dbg] failed to write mask QC volumes")
            _dbg_print(traceback.format_exc())


def _rotate_points_for_display(
    points: np.ndarray,
    slice_shape: Sequence[int],
    rotations: int,
) -> np.ndarray:
    """Rotate (row, col) coordinates similar to ``np.rot90``."""

    if points.size == 0:
        return points

    rotations = rotations % 4
    rotated = np.asarray(points, dtype=np.float32)
    rows, cols = int(slice_shape[0]), int(slice_shape[1])

    for _ in range(rotations):
        row = rotated[:, 0]
        col = rotated[:, 1]
        new_row = cols - 1 - col
        new_col = row
        rotated = np.stack([new_row, new_col], axis=1)
        rows, cols = cols, rows

    return rotated


def _collect_slice_polylines(
    streamlines: Streamlines,
    reference_img: nib.Nifti1Image,
    slice_indices: Sequence[int],
    *,
    slab_half_thickness: float = _MONTAGE_SLICE_THICKNESS,
    max_polylines_per_slice: int = _MONTAGE_MAX_POLYLINES,
) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Return per-slice polylines and colours from streamline slabs."""

    unique_indices = sorted({int(idx) for idx in slice_indices if idx is not None})
    if not unique_indices:
        return {}

    aff4 = _canonical_affine(reference_img.affine)
    inv_affine = np.linalg.inv(aff4)

    slice_polylines: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        idx: [] for idx in unique_indices
    }
    rng = np.random.default_rng(0x7139CAFEBABE1234)

    for streamline in streamlines:
        if streamline.shape[0] < 2:
            continue
        world = _to_xyz(streamline)
        vox = nib.affines.apply_affine(inv_affine, world)
        if vox.shape[0] < 2:
            continue
        z_values = vox[:, 2]

        for slice_idx in unique_indices:
            mask = np.abs(z_values - slice_idx) <= slab_half_thickness
            if np.count_nonzero(mask) < 2:
                continue

            start = None
            for idx, keep in enumerate(mask):
                if keep and start is None:
                    start = idx
                elif not keep and start is not None:
                    if idx - start >= 2:
                        run_vox = vox[start:idx, :2]
                        run_world = world[start:idx]
                        colour = _orientation_colour_from_vector(
                            run_world[-1] - run_world[0]
                        )
                        slice_polylines[slice_idx].append((run_vox, colour))
                    start = None
            if start is not None and mask[-1] and len(vox) - start >= 2:
                run_vox = vox[start:, :2]
                run_world = world[start:]
                colour = _orientation_colour_from_vector(run_world[-1] - run_world[0])
                slice_polylines[slice_idx].append((run_vox, colour))

    if max_polylines_per_slice > 0:
        for slice_idx, polylines in slice_polylines.items():
            if len(polylines) <= max_polylines_per_slice:
                continue
            choices = rng.choice(
                len(polylines), size=max_polylines_per_slice, replace=False
            )
            choices.sort()
            slice_polylines[slice_idx] = [polylines[i] for i in choices]

    return slice_polylines


def _tight_bbox_from_mask(mask2d: np.ndarray, pad: int = 3) -> tuple[int, int, int, int]:
    if not np.any(mask2d):
        return 0, mask2d.shape[0], 0, mask2d.shape[1]

    rows = np.any(mask2d, axis=1)
    cols = np.any(mask2d, axis=0)
    r0 = np.argmax(rows)
    r1 = len(rows) - np.argmax(rows[::-1])
    c0 = np.argmax(cols)
    c1 = len(cols) - np.argmax(cols[::-1])

    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(mask2d.shape[0], r1 + pad)
    c1 = min(mask2d.shape[1], c1 + pad)

    height = r1 - r0
    width = c1 - c0
    if height <= 0 or width <= 0:
        return r0, r1, c0, c1

    if height < width:
        diff = width - height
        extra_top = diff // 2
        extra_bottom = diff - extra_top
        r0 = max(0, r0 - extra_top)
        r1 = min(mask2d.shape[0], r1 + extra_bottom)
        shortfall = width - (r1 - r0)
        if shortfall > 0:
            if r0 > 0:
                shift = min(shortfall, r0)
                r0 -= shift
                shortfall -= shift
            if shortfall > 0 and r1 < mask2d.shape[0]:
                r1 = min(mask2d.shape[0], r1 + shortfall)
    elif width < height:
        diff = height - width
        extra_left = diff // 2
        extra_right = diff - extra_left
        c0 = max(0, c0 - extra_left)
        c1 = min(mask2d.shape[1], c1 + extra_right)
        shortfall = height - (c1 - c0)
        if shortfall > 0:
            if c0 > 0:
                shift = min(shortfall, c0)
                c0 -= shift
                shortfall -= shift
            if shortfall > 0 and c1 < mask2d.shape[1]:
                c1 = min(mask2d.shape[1], c1 + shortfall)

    return r0, r1, c0, c1


def _spaced_unique_indices(zmin: int, zmax: int, k: int) -> np.ndarray:
    if zmax < zmin:
        return np.zeros(k, dtype=int)
    xs = np.linspace(zmin, zmax, num=k)
    idx = np.rint(xs).astype(int)
    idx = np.clip(idx, zmin, zmax)
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = min(zmax, idx[i - 1] + 1)
    if idx.size < k:
        idx = np.pad(idx, (0, k - idx.size), mode="edge")
    return idx


def _slice_valid_bounds(mask: np.ndarray | None) -> tuple[int, int]:
    if isinstance(mask, np.ndarray) and mask.ndim == 1 and mask.size:
        indices = np.flatnonzero(mask)
        if indices.size:
            return int(indices[0]), int(indices[-1])
    return 0, max(0, int(mask.size - 1) if isinstance(mask, np.ndarray) else 0)


def _render_montage(
    streamlines: Streamlines,
    reference_img: nib.Nifti1Image,
    background_img: nib.Nifti1Image,
    output_path: str,
    *,
    title: str,
    brain_mask: np.ndarray | None = None,
) -> None:
    """Create a montage with a T1 (or FA) underlay and streamline overlays."""

    import matplotlib.pyplot as plt

    background = np.asarray(background_img.get_fdata(), dtype=np.float32)
    if background.ndim > 3:
        background = background[..., 0]
    background = np.nan_to_num(background)

    _, counts = _aggregate_streamline_colours(
        streamlines, _canonical_affine(reference_img.affine), background.shape
    )

    brain_mask_bool = None
    if brain_mask is not None and np.shape(brain_mask) == background.shape:
        brain_mask_bool = np.asarray(brain_mask, dtype=bool)

    preferred_mask = brain_mask_bool
    if preferred_mask is None or not preferred_mask.any():
        fallback = counts > 0
        preferred_mask = fallback if np.any(fallback) else np.isfinite(background)

    ref_mask = preferred_mask
    union_xy = np.any(ref_mask, axis=2)
    if not union_xy.any():
        union_xy = np.any(np.isfinite(background), axis=2)
    union_xy_r = np.rot90(union_xy, _MONTAGE_ROTATE)
    r0, r1, c0, c1 = _tight_bbox_from_mask(union_xy_r, pad=_MONTAGE_PAD)

    slice_valid = np.any(ref_mask, axis=(0, 1))
    zmin, zmax = _slice_valid_bounds(slice_valid)
    if zmax <= zmin:
        zmin, zmax = 0, background.shape[2] - 1
    z_indices = _spaced_unique_indices(zmin, zmax, _MONTAGE_ROWS * _MONTAGE_COLS)
    slice_polylines = _collect_slice_polylines(streamlines, reference_img, z_indices)

    masked_values = background[ref_mask]
    if masked_values.size:
        bg_min, bg_max = np.percentile(masked_values, (2.0, 98.0))
    else:
        finite = background[np.isfinite(background)]
        bg_min, bg_max = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)
    if not np.isfinite(bg_min) or not np.isfinite(bg_max) or bg_min == bg_max:
        bg_min, bg_max = float(np.min(background)), float(np.max(background) or 1.0)
    if bg_max <= bg_min:
        bg_max = bg_min + 1.0

    fig, axes = plt.subplots(
        _MONTAGE_ROWS,
        _MONTAGE_COLS,
        figsize=(_MONTAGE_COLS * 2.2, _MONTAGE_ROWS * 2.2),
    )
    axes = axes.ravel()

    for ax, z_index in zip(axes, z_indices):
        slice_bg = background[:, :, z_index]
        display_bg = np.rot90(slice_bg, _MONTAGE_ROTATE)[r0:r1, c0:c1]
        ax.imshow(
            display_bg,
            cmap="gray",
            vmin=bg_min,
            vmax=bg_max,
            interpolation="bicubic",
        )

        height = r1 - r0
        width = c1 - c0
        polylines = slice_polylines.get(int(z_index), [])
        offset = np.array([[r0, c0]], dtype=np.float32)
        if polylines:
            for coords, colour in polylines:
                if coords.shape[0] < 2:
                    continue
                rotated_coords = _rotate_points_for_display(
                    coords,
                    slice_bg.shape,
                    _MONTAGE_ROTATE,
                )
                shifted = rotated_coords - offset
                mask = (
                    (shifted[:, 0] >= 0)
                    & (shifted[:, 0] < height)
                    & (shifted[:, 1] >= 0)
                    & (shifted[:, 1] < width)
                )
                if np.count_nonzero(mask) < 2:
                    continue
                clipped = shifted[mask]
                ax.plot(
                    clipped[:, 1],
                    clipped[:, 0],
                    color=colour,
                    linewidth=0.9,
                    alpha=0.9,
                )

        ax.set_title(f"z = {int(z_index)}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _streamline_density_image(
    streamlines: Streamlines,
    reference_img: nib.Nifti1Image,
    *,
    normalise: bool = False,
) -> nib.Nifti1Image:
    """Return a NIfTI image whose voxels encode streamline densities."""

    spatial_shape = reference_img.shape[:3]
    if not all(dim > 0 for dim in spatial_shape):  # pragma: no cover - defensive
        raise ValueError("reference image must provide 3-D spatial shape")

    colour_sum, counts = _streamline_voxel_stats(
        streamlines, reference_img.affine, spatial_shape
    )
    _ = colour_sum  # Colour sum unused – shared helper returns both

    data = counts
    if normalise and np.max(counts) > 0:
        data = counts / float(np.max(counts))

    header = reference_img.header.copy()
    img = nib.Nifti1Image(
        data.astype(np.float32),
        _canonical_affine(reference_img.affine),
        header,
    )
    return img


def _header_debug(img: nib.Nifti1Image) -> dict:
    hdr = img.header
    out = {}
    try:
        out = {
            "dim": tuple(int(x) for x in hdr.get("dim", ())),
            "qform_code": int(hdr["qform_code"]), "sform_code": int(hdr["sform_code"]),
        }
    except Exception:
        pass
    return out


def _load_tractography_meta(meta_path: str) -> Optional[dict[str, object]]:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _canonical_orientation_model_name(model: Optional[str]) -> str:
    name = (model or "DTI").strip().upper()
    if not name:
        return "DTI"
    if name in {"DT", "DTI", "TENSOR"}:
        return "DTI"
    return name


def _tractography_meta_matches(meta: Optional[dict[str, object]], requested_model: str) -> bool:
    """Return True when saved metadata indicates cached streamlines match request."""

    requested = _canonical_orientation_model_name(requested_model)
    if not meta:
        # Legacy tractography.trk files had no meta; only reuse for default tensor/DTI.
        return requested == "DTI"
    saved = meta.get("diffusion_model")
    if isinstance(saved, str) and saved.strip():
        return _canonical_orientation_model_name(saved) == requested
    return requested == "DTI"


def generate_tractography(
    nifti_directory: str,
    analysis_directory: str,
    image_directory: Optional[str] = None,
    diffusion_filename: Optional[str] = None,
    diffusion_model: Optional[str] = None,
    *,
    anatomical_overlay: Optional[str] = None,
    create_montage: bool = False,
    montage_title: str = "Tractography overlay",
    render_only: bool = False,
    force_regenerate: bool = False,
    enable_act: Optional[bool] = None,
    enable_pft: Optional[bool] = None,
) -> TractographyOutputs:
    """Compute deterministic tractography and associated visualisations.

    When ``render_only`` is ``True`` precomputed tractography is required and
    streamlines will not be regenerated; only downstream outputs are refreshed.
    ``force_regenerate`` overrides reuse heuristics to ensure fresh
    tractography (useful after changing orientation models).

    ``enable_act`` and ``enable_pft`` explicitly toggle anatomically constrained
    tractography and particle-filtering tracking respectively. When left as
    ``None`` the environment defaults from ``_tracking_config_defaults`` apply.
    """

    diffusion_dir = os.path.join(analysis_directory, "diffusion")
    os.makedirs(diffusion_dir, exist_ok=True)
    debug_json_path = os.path.join(diffusion_dir, "tractography_debug.json")
    debug_blob = {
        "ts": datetime.datetime.now().isoformat(),
        "numpy_version": getattr(np, "__version__", None),
        "dipy_version": None,
        "env": {k: v for k, v in os.environ.items() if k.startswith("P_BRAIN") or k.startswith("OMP_")},
    }
    render_only = bool(render_only)
    force_regenerate = bool(force_regenerate)
    if force_regenerate and render_only:
        _dbg(
            "force_regenerate requested – overriding render_only to ensure streamlines are rebuilt"
        )
        render_only = False

    debug_blob["render_only"] = render_only
    debug_blob["force_regenerate"] = force_regenerate

    filter_config = _streamline_filter_defaults()
    render_filter_override: Optional[dict[str, object]] = None
    if render_only and _RENDER_ONLY_DISABLE_SUBSAMPLE:
        render_filter_override = {
            "reason": "render_only",
            "original_stride": filter_config.get("subsample_stride"),
            "original_min_count": filter_config.get("subsample_min_count"),
        }
        filter_config["subsample_stride"] = 1
        filter_config["subsample_min_count"] = 0
    debug_blob["streamline_filter"] = {
        "min_length": filter_config["min_length"],
        "max_length": filter_config["max_length"],
        "subsample_stride": filter_config["subsample_stride"],
        "subsample_min_count": filter_config.get("subsample_min_count"),
    }
    if render_filter_override:
        debug_blob["streamline_filter_override"] = render_filter_override

    tracking_config = _tracking_config_defaults(filter_config)
    anatomical_overrides: dict[str, bool] = {}
    if enable_act is not None:
        tracking_config["act_enabled"] = bool(enable_act)
        anatomical_overrides["act_enabled"] = bool(enable_act)
    if enable_pft is not None:
        tracking_config["pft_enabled"] = bool(enable_pft)
        anatomical_overrides["pft_enabled"] = bool(enable_pft)
    if anatomical_overrides:
        debug_blob["anatomical_override_flags"] = anatomical_overrides
    debug_blob["tracking_config"] = tracking_config
    try:
        import dipy

        debug_blob["dipy_version"] = getattr(dipy, "__version__", None)
    except Exception:
        pass

    tract_path = os.path.join(diffusion_dir, "tractography.trk")
    tract_meta_path = tract_path + ".meta.json"
    debug_blob["tract_meta_path"] = tract_meta_path
    if render_only and not os.path.exists(tract_path):
        raise FileNotFoundError(
            "Existing tractography streamlines not found – run without --tracks_only first."
        )

    forced_backup_path: Optional[str] = None
    if force_regenerate and os.path.exists(tract_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        forced_backup_path = f"{tract_path}.forced_backup_{timestamp}"
        try:
            shutil.move(tract_path, forced_backup_path)
            try:
                if os.path.exists(tract_meta_path):
                    shutil.move(tract_meta_path, forced_backup_path + ".meta.json")
            except Exception:
                debug_blob["forced_backup_meta_error"] = traceback.format_exc()
            _dbg(
                f"force_regenerate moved existing tractography to backup: {forced_backup_path}"
            )
            debug_blob["forced_backup_path"] = forced_backup_path
        except Exception:
            debug_blob["forced_backup_error"] = traceback.format_exc()

    diffusion_dataset = _resolve_diffusion_dataset(
        nifti_directory,
        diffusion_filename=diffusion_filename,
    )

    dwi_path = diffusion_dataset.volume_path
    bval_path = diffusion_dataset.bval_path
    bvec_path = diffusion_dataset.bvec_path
    diffusion_label = diffusion_dataset.label
    dataset_default_model = diffusion_dataset.default_model or "DTI"
    diffusion_model_choice = _canonical_orientation_model_name(
        (diffusion_model or dataset_default_model)
    )

    debug_blob["diffusion_acquisition"] = {
        "label": diffusion_label,
        "model": dataset_default_model,
        "default_model": dataset_default_model,
        "requested_model": diffusion_model_choice,
        "volume": os.path.basename(dwi_path),
    }

    img = nib.load(dwi_path)
    # Force a 4x4 voxel->world affine even if header carries higher-dim transforms.
    voxel_to_world = _canonical_affine(getattr(img, "affine", np.eye(4)))
    if _DBG:
        _dbg_print(f"[tracks][dbg] DWI file: {dwi_path}")
        _dbg_print(f"[tracks][dbg] Header dim/qform/sform: {_header_debug(img)}")
        _print_affine("img.affine (voxel->mm)", voxel_to_world)
        try:
            xyzt = img.header.get_xyzt_units()
            _dbg_print(f"[tracks][dbg] xyzt units={xyzt}")
        except Exception:
            pass

    montage_brain_mask: Optional[np.ndarray] = None
    if render_only:
        tracking_config_disabled = dict(tracking_config)
        tracking_config_disabled["act_enabled"] = False
        tracking_config_disabled["pft_enabled"] = False
        anatomical_strategy, _ = _prepare_anatomical_strategy(
            nifti_directory,
            analysis_directory,
            img,
            tracking_config_disabled,
        )
        anatomical_debug = {
            "act_requested": bool(tracking_config.get("act_enabled", False)),
            "pft_requested": bool(tracking_config.get("pft_enabled", False)),
            "anatomical_disabled": "render_only",
        }
    else:
        anatomical_strategy, anatomical_debug = _prepare_anatomical_strategy(
            nifti_directory,
            analysis_directory,
            img,
            tracking_config,
        )
        montage_brain_mask = _brain_mask_from_tissue_maps(anatomical_strategy.get("tissue_maps"))
    debug_blob["anatomical_tracking"] = anatomical_debug

    min_streamlines_target = int(
        tracking_config.get("target_streamlines")
        or tracking_config.get("min_streamlines")
        or 0
    )
    streamlines: Optional[Streamlines] = None
    regeneration_reason: Optional[str] = None
    reuse_existing = True if render_only else _env_bool("P_BRAIN_TRACK_REUSE_EXISTING", True)
    if force_regenerate:
        reuse_existing = False
        regeneration_reason = "forced"

    if reuse_existing and os.path.exists(tract_path):
        cached_meta = _load_tractography_meta(tract_meta_path)
        meta_ok = _tractography_meta_matches(cached_meta, diffusion_model_choice)
        debug_blob["tract_meta"] = cached_meta
        debug_blob["tract_meta_matches"] = meta_ok
        if not meta_ok:
            if render_only:
                raise RuntimeError(
                    "Existing tractography.trk was computed with a different orientation model; "
                    "rerun without --tracks_only or pass --tracks_force."
                )
            reuse_existing = False
            regeneration_reason = "diffusion_model_changed"

    debug_blob["reuse_existing"] = reuse_existing
    if reuse_existing and os.path.exists(tract_path):
        if _DBG:
            _dbg_print(f"[tracks][dbg] pre-existing tract file found: {tract_path}")
        load_progress_desc = "Loading streamlines" if render_only else None
        streamlines = _load_streamlines_world(
            tract_path,
            expected_affine=voxel_to_world,
            expected_shape=img.shape,
            progress_desc=load_progress_desc,
        )

    if render_only and streamlines is None:
        raise RuntimeError(
            "Tractography streamlines could not be loaded; rerun without --tracks_only."
        )

    fa_volume: Optional[np.ndarray] = None
    stream_stats: dict[str, object] = {}
    density_enabled = _env_bool("P_BRAIN_TRACK_SAVE_DENSITY", True)
    debug_blob["density_enabled"] = density_enabled
    density_path: Optional[str] = None

    ffmpeg_binary = _resolve_ffmpeg_binary()
    if ffmpeg_binary:
        debug_blob["ffmpeg_binary"] = ffmpeg_binary

    animation_enabled = _env_bool("P_BRAIN_TRACK_SAVE_ANIMATION", True)
    animation_frames = _env_int("P_BRAIN_TRACK_ANIMATION_FRAMES", 180)
    animation_fps = _env_int("P_BRAIN_TRACK_ANIMATION_FPS", 12)
    animation_rotation = _env_float("P_BRAIN_TRACK_ANIMATION_ROTATION_DEG", 360.0) or 360.0
    animation_progressive = _env_bool("P_BRAIN_TRACK_ANIMATION_PROGRESSIVE", True)
    animation_elevation = _env_float("P_BRAIN_TRACK_ANIMATION_ELEV_DEG", 0.0) or 0.0
    requested_animation_format = os.environ.get("P_BRAIN_TRACK_ANIMATION_FORMAT", "mp4").strip().lower() or "mp4"
    animation_format = requested_animation_format
    animation_notes: dict[str, object] = {}
    if animation_enabled and animation_format == "mp4" and not _IMAGEIO_HAS_FFMPEG:
        if ffmpeg_binary:
            animation_notes["format_via_system_ffmpeg"] = ffmpeg_binary
        else:
            animation_notes["format_fallback"] = "mp4 requires imageio-ffmpeg or ffmpeg binary"
            animation_format = "gif"
    animation_path: Optional[str] = None
    animation_info = {
        "enabled": animation_enabled,
        "frames": animation_frames,
        "fps": animation_fps,
        "rotation_deg": animation_rotation,
        "progressive": animation_progressive,
        "elevation_deg": animation_elevation,
        "format": animation_format,
        "requested_format": requested_animation_format,
        **animation_notes,
    }
    debug_blob["animation"] = animation_info

    def _write_debug_snapshot() -> None:
        try:
            debug_blob["voxel_to_world"] = np.array(voxel_to_world).tolist()
            debug_blob["tract_exists"] = os.path.exists(tract_path)
            if density_path:
                debug_blob["density_path"] = density_path
            if animation_path:
                debug_blob["animation_path"] = animation_path
            _dump_debug_json(debug_json_path, debug_blob)
            if _DBG:
                _dbg_print(f"[tracks][dbg] wrote debug snapshot: {debug_json_path}")
        except Exception:
            if _DBG:
                _dbg_print("[tracks][dbg] failed to write debug snapshot")
                _dbg_print(traceback.format_exc())

    if streamlines is not None:
        fa_candidates = (
            os.path.join(diffusion_dir, "fa_map_native_debug.nii.gz"),
            os.path.join(diffusion_dir, "fa_map.nii.gz"),
        )
        for candidate in fa_candidates:
            if not os.path.isfile(candidate):
                continue
            try:
                fa_img = nib.load(candidate)
            except Exception:
                continue
            fa_data = np.asarray(fa_img.get_fdata(), dtype=np.float32)
            if fa_data.shape[:3] == img.shape[:3]:
                fa_volume = fa_data
                break
        # Streamlines loaded from file can carry 4 columns in rare cases
        loaded_streamlines = _load_streamlines_world(
            tract_path,
            expected_affine=voxel_to_world,
            expected_shape=img.shape,
        )
        if loaded_streamlines is not None:
            streamlines = loaded_streamlines
        stream_stats["source"] = "loaded"
        if streamlines is None:
            raise RuntimeError("Failed to load existing tractography streamlines")
        streamlines = _postprocess_streamlines(
            streamlines,
            min_length=filter_config["min_length"],
            max_length=filter_config["max_length"],
            subsample_stride=filter_config["subsample_stride"],
            subsample_min_count=int(filter_config.get("subsample_min_count", 0) or 0),
            debug_blob=stream_stats,
        )
        filtered_count = int(stream_stats.get("filtered_count", len(streamlines)))
        if len(streamlines) == 0:
            raise RuntimeError(
                "Existing tractography streamlines were removed by length filtering"
            )
        if (
            not render_only
            and min_streamlines_target
            and filtered_count < min_streamlines_target
        ):
            debug_blob["regenerated_sparse_tracks"] = {
                "filtered_count": filtered_count,
                "threshold": min_streamlines_target,
            }
            backup_path = tract_path + ".sparse_backup"
            try:
                shutil.move(tract_path, backup_path)
                debug_blob["sparse_backup_path"] = backup_path
            except Exception:
                debug_blob["sparse_backup_error"] = traceback.format_exc()
            streamlines = None
            regeneration_reason = "sparse_loaded"
            if _DBG:
                _dbg(
                    "Loaded tractography below min_streamlines threshold; regenerating."
                )
            if os.path.exists(tract_path):
                try:
                    os.remove(tract_path)
                except OSError:
                    pass
            stream_stats = {}

    seed_rng_seed: Optional[int] = None

    if streamlines is None:
        if render_only:
            raise RuntimeError(
                "render_only requested but no streamlines available to render"
            )
        stream_stats["source"] = "regenerated" if regeneration_reason else "generated"
        raw = img.get_fdata(dtype=np.float32)
        bvals = np.loadtxt(bval_path)
        bvecs = np.loadtxt(bvec_path)
        if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        data = _as_4d_dwi(raw, bvals)

        if data.shape[-1] != bvals.size or bvecs.shape != (bvals.size, 3):
            raise ValueError(
                f"bvals/bvecs mismatch. volumes={data.shape[-1]}, "
                f"bvals={bvals.size}, bvecs={bvecs.shape}"
            )
        if _DBG:
            _dbg_print(f"[tracks][dbg] DWI shape={data.shape}")
            _dbg_print(
                f"[tracks][dbg] bvals size={bvals.size} unique="
                f"{sorted(set(int(x) for x in bvals.tolist()))}"
            )
            _dbg_print(
                f"[tracks][dbg] bvecs shape={bvecs.shape} norms≈"
                f"{[float(f'{n:.3f}') for n in np.linalg.norm(bvecs,axis=1)[:6]]}"
            )
            _dbg_print(
                f"[tracks][dbg] affine(vox->mm) shape={voxel_to_world.shape}"
            )
            debug_blob["dwi_shape"] = tuple(int(x) for x in data.shape)
            debug_blob["bvals_len"] = int(bvals.size)
            debug_blob["bvecs_shape"] = tuple(int(x) for x in bvecs.shape)

        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        orientation_fit = _fit_orientation_model(
            data,
            gtab,
            requested_model=diffusion_model_choice,
            tracking_config=tracking_config,
        )
        orientation_model = orientation_fit.model
        fa_volume = orientation_fit.fa_volume
        if _DBG:
            stats = orientation_fit.details.get("fa_stats", {})
            _dbg_print(
                "[tracks][dbg] FA stats: min={min:.4f} max={max:.4f} mean={mean:.4f}".format(
                    min=float(stats.get("min", 0.0)),
                    max=float(stats.get("max", 0.0)),
                    mean=float(stats.get("mean", 0.0)),
                )
            )

        orientation_debug: dict[str, object] = {
            "requested": orientation_fit.requested,
            "actual": orientation_fit.resolved,
            "details": orientation_fit.details,
        }
        response_info = orientation_fit.details.get("response")
        if response_info:
            orientation_debug["response"] = response_info
        debug_blob["orientation_model"] = orientation_debug
        if _DBG:
            fallback_reason = orientation_fit.details.get("fallback_reason")
            note = f"; fallback={fallback_reason}" if fallback_reason else ""
            _dbg(
                f"orientation model resolved to {orientation_fit.resolved} (requested {orientation_fit.requested}){note}"
            )

        require_requested_orientation = _env_bool(
            "P_BRAIN_TRACK_REQUIRE_REQUESTED_MODEL",
            True,
        )
        if (
            require_requested_orientation
            and orientation_fit.requested
            and orientation_fit.resolved != orientation_fit.requested
        ):
            fallback_details = orientation_fit.details.get("fallback_reason") or "unspecified"
            raise RuntimeError(
                "Requested orientation model {req} but resolved to {actual}: {detail}".format(
                    req=orientation_fit.requested,
                    actual=orientation_fit.resolved,
                    detail=fallback_details,
                )
            )

        attempt_plan = _build_tracking_attempt_plan(tracking_config, filter_config)
        if not attempt_plan:
            raise RuntimeError("No tracking attempts configured")
        debug_blob["attempt_plan"] = attempt_plan
        mask_thresholds = [
            float(cfg.get("seed_fa_threshold", tracking_config["wm_threshold"]))
            for cfg in attempt_plan
        ]
        if not mask_thresholds:
            mask_thresholds = [tracking_config["wm_threshold"]]
        peaks_mask_threshold = float(min(mask_thresholds))
        debug_blob["peaks_mask_threshold"] = peaks_mask_threshold

        wm_mask = fa_volume > peaks_mask_threshold
        if not np.any(wm_mask):
            relaxed_mask_thr = max(0.05, peaks_mask_threshold * 0.75)
            wm_mask = fa_volume > relaxed_mask_thr
            tracking_config["wm_threshold_relaxed"] = relaxed_mask_thr
        if not np.any(wm_mask):
            raise RuntimeError(
                "Diffusion data does not contain voxels above FA thresholds"
            )
        if _DBG:
            _dbg_print(
                f"[tracks][dbg] WM mask voxels={int(np.count_nonzero(wm_mask))}"
            )

        if (
            anatomical_strategy
            and anatomical_strategy.get("tissue_maps")
            and (
                anatomical_strategy.get("use_act")
                or anatomical_strategy.get("use_pft")
            )
        ):
            tissue_maps = anatomical_strategy.get("tissue_maps") or {}
            # Prefer FreeSurfer-derived diffusion masks when available; fall
            # back to generic WM/GM/CSF union otherwise.
            fs_keys = [
                key
                for key in tissue_maps.keys()
                if key.startswith("fs_")
            ]
            coverage_keys: Optional[Sequence[str]]
            if fs_keys:
                coverage_keys = tuple(sorted(fs_keys))
            else:
                coverage_keys = None

            _save_mask_qc_volumes(
                diffusion_dir,
                voxel_to_world,
                wm_mask,
                tissue_maps,
            )
            coverage_stats = _tissue_map_coverage_stats(
                wm_mask,
                tissue_maps,
                keys=coverage_keys,
            )
            anatomical_debug["tissue_coverage"] = coverage_stats
            min_fraction = float(tracking_config.get("anatomical_min_coverage") or 0.0)
            min_fraction = _clamp01(min_fraction)
            frac_supported = float(coverage_stats.get("fraction_supported", 0.0))
            target_voxels = int(coverage_stats.get("target_voxels", 0))
            if target_voxels and frac_supported < min_fraction:
                if _DBG:
                    _dbg(
                        "anatomical masks cover {:.1f}% of FA WM mask (<{:.0f}%) – disabling ACT/PFT".format(
                            frac_supported * 100.0,
                            min_fraction * 100.0,
                        )
                    )
                anatomical_strategy["use_act"] = False
                anatomical_strategy["act"] = None
                anatomical_strategy["use_pft"] = False
                anatomical_strategy["cmc"] = None
                anatomical_debug["anatomical_disabled"] = "insufficient_tissue_coverage"
                anatomical_debug["coverage_min_fraction"] = min_fraction
                if montage_brain_mask is not None:
                    montage_brain_mask = None
        debug_blob["anatomical_tracking"] = anatomical_debug

        peaks_kwargs = {
            "relative_peak_threshold": 0.5,
            "min_separation_angle": 25,
            "mask": wm_mask,
            "return_sh": True,
        }
        orientation_specific = dict(orientation_fit.peaks_kwargs)
        orientation_specific.pop("mask", None)
        for key, value in orientation_specific.items():
            if value is not None:
                peaks_kwargs[key] = value
        try:
            peaks = peaks_from_model(
                orientation_model,
                data,
                default_sphere,
                **peaks_kwargs,
            )
        except Exception:
            if _DBG:
                _dbg_print("[tracks][err] peaks_from_model failed")
                _dbg_print(traceback.format_exc())
            raise
        if _DBG:
            try:
                _dbg_print(
                    f"[tracks][dbg] peaks.peak_dirs shape={getattr(peaks,'peak_dirs',None).shape}"
                )
                if hasattr(peaks, "affine"):
                    _print_affine("peaks.affine", peaks.affine)
            except Exception:
                pass

        rng_seed = np.uint32(abs(hash(os.path.abspath(diffusion_dir))) & 0xFFFFFFFF)
        seed_rng_seed = int(rng_seed)
        debug_blob["seed_rng_seed"] = int(rng_seed)
        rng = np.random.default_rng(int(rng_seed))

        mask_cache: dict[tuple[float, int], np.ndarray] = {}
        tracking_attempts: list[dict] = []
        combined_streamlines: list[np.ndarray] = []
        max_attempts = int(tracking_config.get("max_attempts") or len(attempt_plan) or 1)
        target_streamlines = int(tracking_config.get("target_streamlines") or 0)
        max_seed_points = int(tracking_config.get("max_seed_points") or 0)
        progress_streamline_mode = target_streamlines > 0
        progress_total = target_streamlines if progress_streamline_mode else max_attempts
        progress_bar = _create_progress_bar(
            progress_total,
            desc="Tractography",
            unit="streamline" if progress_streamline_mode else "attempt",
        )

        worker_limit = min(max_attempts, max(1, _TRACKING_WORKER_REQUEST))
        executor, executor_backend = _create_tracking_executor(worker_limit)
        if executor is None:
            worker_limit = 1
        debug_blob["tracking_workers"] = {
            "requested": _TRACKING_WORKER_REQUEST,
            "effective": worker_limit,
            "backend": executor_backend,
        }

        pending_futures: dict = {}
        pending_results: dict[int, tuple[Streamlines, dict, dict]] = {}
        next_sequence_to_apply = 0
        target_sequence_limit: Optional[int] = None
        target_met = False

        def _apply_attempt_result(seq: int, processed: Streamlines, attempt_record: dict, context: dict) -> None:
            nonlocal target_met, target_sequence_limit

            attempt_record.update(context["attempt_meta"])
            attempt_record["sequence"] = seq
            attempt_record["combined_before"] = len(combined_streamlines)

            if target_sequence_limit is not None and seq > target_sequence_limit:
                attempt_record["skipped_after_target"] = True
                tracking_attempts.append(attempt_record)
            else:
                combined_streamlines.extend(
                    [np.asarray(sl, dtype=np.float32) for sl in processed]
                )
                attempt_record["combined_after"] = len(combined_streamlines)
                tracking_attempts.append(attempt_record)

                if progress_bar:
                    if progress_streamline_mode:
                        progress_value = min(len(combined_streamlines), progress_total)
                        delta = max(0, progress_value - progress_bar.n)
                        if delta:
                            progress_bar.update(delta)
                        progress_bar.set_postfix(
                            {
                                "attempt": seq + 1,
                                "streams": len(combined_streamlines),
                            },
                            refresh=True,
                        )
                    else:
                        progress_bar.update(1)

                if (
                    target_streamlines
                    and len(combined_streamlines) >= target_streamlines
                    and target_sequence_limit is None
                ):
                    target_sequence_limit = seq
                    target_met = True

        def _drain_ready_results(block: bool) -> None:
            nonlocal next_sequence_to_apply
            while True:
                if next_sequence_to_apply in pending_results:
                    processed, attempt_record, ctx = pending_results.pop(next_sequence_to_apply)
                    _apply_attempt_result(next_sequence_to_apply, processed, attempt_record, ctx)
                    next_sequence_to_apply += 1
                    continue

                if not pending_futures or executor is None:
                    break

                if block:
                    done, _ = wait(
                        pending_futures.keys(),
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    done, _ = wait(
                        pending_futures.keys(),
                        timeout=0.0,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        break

                for future in done:
                    context = pending_futures.pop(future)
                    try:
                        processed, attempt_record = future.result()
                    except Exception:
                        for remaining in pending_futures:
                            remaining.cancel()
                        if executor is not None:
                            executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    pending_results[context["sequence"]] = (processed, attempt_record, context)

                if not block:
                    break

        attempt_index = 0
        try:
            while attempt_index < max_attempts and not target_met:
                cfg = attempt_plan[min(attempt_index, len(attempt_plan) - 1)]
                label = str(cfg.get("label", f"attempt_{attempt_index + 1}"))
                if attempt_index >= len(attempt_plan):
                    label = f"{label}_repeat{attempt_index - len(attempt_plan) + 1}"

                seed_threshold = float(
                    cfg.get("seed_fa_threshold", tracking_config["wm_threshold"])
                )
                dilation_iters = int(cfg.get("mask_dilation_iters") or 0)
                mask = _seed_mask_for_threshold(
                    fa_volume, seed_threshold, dilation_iters, mask_cache
                )
                mask_voxels = int(np.count_nonzero(mask))
                if mask_voxels == 0:
                    tracking_attempts.append(
                        {
                            "label": label,
                            "seed_fa_threshold": seed_threshold,
                            "mask_voxels": 0,
                            "skipped": "empty_mask",
                        }
                    )
                    attempt_index += 1
                    if progress_bar and not progress_streamline_mode:
                        progress_bar.update(1)
                    continue

                density_cfg = cfg.get("seed_density", tracking_config["seed_density"])
                if isinstance(density_cfg, (list, tuple)):
                    density_param: Union[int, Sequence[int]] = tuple(
                        max(1, int(max(1, d))) for d in density_cfg
                    )
                else:
                    density_param = max(1, int(density_cfg))

                jitter_mm = float(cfg.get("seed_jitter_mm") or 0.0)
                max_points_override = cfg.get("max_seeds")
                candidate_limit = (
                    int(max_points_override)
                    if max_points_override not in (None, 0)
                    else max_seed_points
                )
                max_points = int(candidate_limit) if candidate_limit else None

                seeds = _seed_points_from_mask(
                    mask,
                    voxel_to_world,
                    density=density_param,
                    max_points=max_points,
                    jitter_mm=jitter_mm,
                    rng=rng,
                )
                original_seed_count = int(seeds.shape[0])
                seeds, dropped_seeds, adjusted_seeds = _sanitize_seed_points(
                    seeds, voxel_to_world, fa_volume.shape
                )
                if _DBG and (dropped_seeds or adjusted_seeds):
                    _dbg(
                        f"sanitized seeds for attempt {label}: dropped={dropped_seeds} adjusted={adjusted_seeds}"
                    )

                if seeds.size == 0:
                    tracking_attempts.append(
                        {
                            "label": label,
                            "seed_fa_threshold": seed_threshold,
                            "mask_voxels": mask_voxels,
                            "skipped": "no_seeds",
                            "seeds_requested": original_seed_count,
                            "seeds_dropped_outside": int(dropped_seeds),
                            "seeds_adjusted_inside": int(adjusted_seeds),
                        }
                    )
                    attempt_index += 1
                    if progress_bar and not progress_streamline_mode:
                        progress_bar.update(1)
                    continue

                filter_override = {
                    "min_length": cfg.get("min_length", filter_config["min_length"]),
                    "max_length": cfg.get("max_length", filter_config["max_length"]),
                    "subsample_stride": cfg.get(
                        "subsample_stride", filter_config["subsample_stride"]
                    ),
                    "subsample_min_count": cfg.get(
                        "subsample_min_count", filter_config.get("subsample_min_count", 0)
                    ),
                }
                stop_threshold = float(
                    cfg.get("stop_threshold", tracking_config["stop_threshold"])
                )

                context = {
                    "sequence": attempt_index,
                    "attempt_meta": {
                        "seed_fa_threshold": seed_threshold,
                        "mask_voxels": mask_voxels,
                        "seed_density": density_param,
                        "seed_jitter_mm": jitter_mm,
                        "max_seed_points": max_points,
                        "seeds_used": int(len(seeds)),
                        "seeds_requested": original_seed_count,
                        "seeds_dropped_outside": int(dropped_seeds),
                        "seeds_adjusted_inside": int(adjusted_seeds),
                    },
                }

                if executor is None:
                    processed, attempt_record = _run_tracking_attempt(
                        label=label,
                        peaks=peaks,
                        seeds=seeds,
                        voxel_to_world=voxel_to_world,
                        fa_volume=fa_volume,
                        stop_threshold=stop_threshold,
                        filter_config=filter_override,
                        tracking_strategy=anatomical_strategy,
                        rng_seed=seed_rng_seed,
                    )
                    pending_results[context["sequence"]] = (processed, attempt_record, context)
                    _drain_ready_results(block=False)
                else:
                    future = executor.submit(
                        _run_tracking_attempt,
                        label=label,
                        peaks=peaks,
                        seeds=seeds,
                        voxel_to_world=voxel_to_world,
                        fa_volume=fa_volume,
                        stop_threshold=stop_threshold,
                        filter_config=filter_override,
                        tracking_strategy=anatomical_strategy,
                        rng_seed=seed_rng_seed,
                    )
                    pending_futures[future] = context
                    while len(pending_futures) >= worker_limit:
                        _drain_ready_results(block=True)

                attempt_index += 1
                _drain_ready_results(block=False)

        finally:
            _drain_ready_results(block=True)
            if progress_bar:
                progress_bar.close()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

        if len(combined_streamlines) == 0:
            raise RuntimeError("Tractography produced no streamlines")

        streamlines = Streamlines(combined_streamlines)

        max_output = int(tracking_config.get("max_output_streamlines") or 0)
        if max_output and len(streamlines) > max_output:
            rng_trim = np.random.default_rng(int(rng_seed) ^ 0xA5A5A5A5)
            keep_idx = np.sort(
                rng_trim.choice(len(streamlines), size=max_output, replace=False)
            )
            streamlines = Streamlines([streamlines[i] for i in keep_idx])
            stream_stats["output_trimmed_to"] = max_output

        if target_streamlines and len(streamlines) < target_streamlines:
            stream_stats["target_unmet"] = {
                "target": target_streamlines,
                "actual": len(streamlines),
            }

        final_lengths = np.array(
            [_streamline_length(sl) for sl in streamlines], dtype=np.float32
        )

        total_raw = sum(int(attempt.get("raw_count", 0)) for attempt in tracking_attempts)
        stream_stats.update(
            {
                "attempts": tracking_attempts,
                "raw_count": total_raw,
                "filtered_count": len(streamlines),
                "combined_streamlines": len(streamlines),
                "target_streamlines": target_streamlines,
                "filtered_length_stats": _length_stats(final_lengths),
            }
        )

        try:
            saved = False
            if StatefulTractogram is not None and Space is not None:
                try:
                    sft = StatefulTractogram(streamlines, img, Space.RASMM)
                    nib_streamlines.save(sft, tract_path)
                    saved = True
                except Exception:
                    debug_blob["stateful_tractogram_error"] = traceback.format_exc()
                    if _DBG:
                        _dbg_print(
                            "[tracks][warn] StatefulTractogram save failed; falling back to legacy tractogram"
                        )
            if not saved:
                tractogram = nib_streamlines.Tractogram(
                    streamlines,
                    affine_to_rasmm=np.eye(4),
                )
                nib_streamlines.save(tractogram, tract_path)
            _dbg(f"saved tractogram: {tract_path}")

            try:
                meta_payload = {
                    "ts": datetime.datetime.now().isoformat(),
                    "diffusion_model": diffusion_model_choice,
                    "dwi": os.path.basename(dwi_path) if isinstance(dwi_path, str) else None,
                }
                with open(tract_meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_payload, f, indent=2, sort_keys=True)
            except Exception:
                debug_blob["tract_meta_write_error"] = traceback.format_exc()
        except Exception:
            if _DBG:
                _dbg_print("[tracks][err] Saving tractogram failed")
                _dbg_print(traceback.format_exc())
            raise


    debug_blob["streamline_stats"] = stream_stats
    debug_blob["regeneration_reason"] = regeneration_reason

    if density_enabled:
        try:
            density_img = _streamline_density_image(streamlines, img)
            density_path = os.path.join(diffusion_dir, "tractography_density.nii.gz")
            nib.save(density_img, density_path)
            _dbg(f"saved density volume: {density_path}")
        except Exception:
            density_path = None
            debug_blob["density_error"] = traceback.format_exc()

    image_directory = _ensure_image_directory(image_directory, analysis_directory)
    tract_image_dir = os.path.join(image_directory, "tractography")
    os.makedirs(tract_image_dir, exist_ok=True)

    if _DBG:
        try:
            _dbg_print(f"[tracks][dbg] platform={platform.system()} arch={platform.machine()}")
        except Exception:
            pass
    _write_debug_snapshot()

    render_path = os.path.join(tract_image_dir, "tractography_render.png")
    if _should_use_fury():
        if not _render_with_fury_isolated(tract_path, render_path):
            _render_with_matplotlib(streamlines, render_path)
    else:  # pragma: no cover - fallback depends on runtime environment
        _render_with_matplotlib(streamlines, render_path)

    if animation_enabled:
        animation_candidates: list[tuple[str, str]] = []
        animation_filename = f"tractography_rotation.{animation_format}"
        animation_file_path = os.path.join(tract_image_dir, animation_filename)
        animation_candidates.append((animation_format, animation_file_path))
        if animation_format != "gif":
            gif_path = os.path.join(tract_image_dir, "tractography_rotation.gif")
            animation_candidates.append(("gif", gif_path))
        animation_seed = seed_rng_seed
        if animation_seed is None:
            try:
                animation_seed = int(
                    np.uint32(
                        abs(
                            hash(
                                (
                                    tract_path,
                                    os.path.getmtime(tract_path)
                                    if os.path.exists(tract_path)
                                    else 0,
                                )
                            )
                        )
                        & 0xFFFFFFFF
                    )
                )
            except Exception:
                animation_seed = None
        animation_errors: list[str] = []
        for fmt, candidate_path in animation_candidates:
            try:
                _render_streamline_animation(
                    streamlines,
                    candidate_path,
                    frame_count=animation_frames,
                    rotation_degrees=float(animation_rotation),
                    fps=animation_fps,
                    progressive=animation_progressive,
                    elevation_degrees=float(animation_elevation),
                    ffmpeg_binary=ffmpeg_binary,
                    rng_seed=animation_seed,
                )
            except Exception:
                animation_errors.append(traceback.format_exc())
                continue
            else:
                animation_path = candidate_path
                animation_info["actual_format"] = fmt
                break
        if animation_path is None and animation_errors:
            debug_blob["animation_error"] = animation_errors
        else:
            debug_blob["animation_error"] = None
        _write_debug_snapshot()

    montage_path: Optional[str] = None
    if create_montage:
        background_path = anatomical_overlay
        background_source = None
        if background_path is None and _AUTO_ANATOMICAL_OVERLAY:
            background_path = _find_anatomical_overlay_path(nifti_directory)
            if background_path:
                background_source = "auto_anatomical"
        elif background_path is not None:
            background_source = "provided"

        background_img: Optional[nib.Nifti1Image] = None
        if background_path:
            try:
                background_img = _load_background_volume(background_path, img)
                rel_background = background_path
                try:
                    rel_background = os.path.relpath(
                        background_path,
                        start=nifti_directory,
                    )
                except Exception:
                    rel_background = background_path
                bg_meta = debug_blob.setdefault("montage_background", {})
                bg_meta["path"] = rel_background
                if background_source:
                    bg_meta["source"] = background_source
            except Exception:
                background_img = None
                debug_blob["montage_background_error"] = traceback.format_exc()

        if background_img is None:
            if fa_volume is None:
                # When tractography was precomputed we may not have FA in-memory.
                raw = img.get_fdata(dtype=np.float32)
                bvals = np.loadtxt(bval_path)
                bvecs = np.loadtxt(bvec_path)
                if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
                    bvecs = bvecs.T
                data = _as_4d_dwi(raw, bvals)
                gtab = gradient_table(bvals=bvals, bvecs=bvecs)
                tensor_model = TensorModel(gtab)
                tensor_fit = tensor_model.fit(data)
                fa_volume = tensor_fit.fa.astype(np.float32)
                fa_volume = np.nan_to_num(fa_volume, nan=0.0, posinf=0.0, neginf=0.0)
            background_img = nib.Nifti1Image(
                fa_volume, voxel_to_world, img.header
            )
        montage_path = os.path.join(tract_image_dir, "tractography_montage.png")
        try:
            _render_montage(
                streamlines,
                img,
                background_img,
                montage_path,
                title=montage_title,
                brain_mask=montage_brain_mask,
            )
        except Exception as e:
            _dbg(f"*** FATAL in section: render montage | {type(e).__name__}: {e}")
            raise

    return TractographyOutputs(
        tract_path=tract_path,
        render_path=render_path,
        montage_path=montage_path,
        density_path=density_path,
        animation_path=animation_path,
    )

