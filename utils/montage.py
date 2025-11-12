"""Utilities for rendering parametric map montages in native DCE space.

Supports external head/ICV masks (e.g., FSL/FreeSurfer). If present, these are
preferentially used to remove air outside the head while retaining skull/scalp,
and to constrain colour overlays to the intracranial volume."""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import shutil
import subprocess
import tempfile
from matplotlib.colors import LinearSegmentedColormap
from nibabel.processing import resample_from_to
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    label,
    distance_transform_edt,
)
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.morphology import (
    ball,
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    remove_small_objects,
)

ROWS = 2
COLS = 5
NUM_TILES = ROWS * COLS
ROT90 = 1
BBOX_PADDING = 3
DPI = 300
EPS = 1e-8
FEATHER_MM = 3.0  # width of soft edge for T1 underlay
HEAD_DILATE_MM = 8.0      # grow brain mask to include skull+scalp
HEAD_EXTRA_MM = 2.0       # tiny extra cushion
HEAD_MIN_VOXELS = 10_000  # drop tiny islands from T1 envelope
ICV_ERODE_MM = 3.0        # approximate skull thickness to peel head -> ICV
MASK_CANDIDATES_HEAD = (
    "head_mask_in_DCE.nii.gz",
    "head_mask.nii.gz",
    "mask_head.nii.gz",
    "head_in_DCE.nii.gz",
    "skull_mask_in_DCE.nii.gz",
    "skull_mask.nii.gz",
)
MASK_CANDIDATES_ICV = (
    "icv_mask_in_DCE.nii.gz",
    "icv_mask.nii.gz",
    "mask_icv.nii.gz",
    "brainmask_in_DCE.nii.gz",
    "brainmask.nii.gz",
    "brainmask.mgz",
)
FREESURFER_BRAINMASK = ("brainmask.mgz", "brainmask.nii.gz")


@dataclass
class MapJob:
    """Configuration for rendering a single parametric map montage."""

    base: str
    output_base: str
    vmin: float | None = None
    vmax: float | None = None
    cmap_name: str = "specthl"
    mask_zero: bool = False
    output_ext: str = ".png"
    patterns: Sequence[str] = field(default_factory=tuple)
    search_directories: Sequence[str] = ("",)
    metric: str | None = None

    def candidate_patterns(self) -> Sequence[str]:
        if self.patterns:
            return self.patterns
        return (
            f"{self.base}.nii.gz",
            f"{self.base}.nii",
            f"{self.base}_*.nii.gz",
            f"{self.base}_*.nii",
        )


MAP_JOBS: Sequence[MapJob] = (
    MapJob("CBF_per_voxel_tikhonov", "cbf_montage"),
    MapJob("CBF_tikhonov_map_atlas", "cbf_parcel_montage"),
    MapJob("mtt_map", "mtt_montage"),
    MapJob("MTT_tikhonov_map_atlas", "mtt_parcel_montage"),
    MapJob("cth_map", "cth_montage"),
    MapJob("CTH_tikhonov_map_atlas", "cth_parcel_montage"),
    MapJob("Ki_per_voxel", "ki_voxel_montage"),
    MapJob("Ki_map_atlas", "ki_atlas_montage"),
    MapJob("vp_map_atlas", "vp_atlas_montage"),
    MapJob("vp_per_voxel", "vp_per_voxel", mask_zero=True, output_ext=".png"),
    MapJob(
        "fa_map",
        "fa_montage",
        vmin=0.0,
        vmax=1.0,
        search_directories=("", "diffusion"),
        metric="fa",
    ),
    MapJob(
        "fa_map_atlas",
        "fa_parcel_montage",
        vmin=0.0,
        vmax=1.0,
        search_directories=("", "diffusion"),
        metric="fa",
    ),
    MapJob("md_map", "md_montage", search_directories=("", "diffusion")),
    MapJob("md_map_atlas", "md_parcel_montage", search_directories=("", "diffusion")),
    MapJob("ad_map", "ad_montage", search_directories=("", "diffusion")),
    MapJob("ad_map_atlas", "ad_parcel_montage", search_directories=("", "diffusion")),
    MapJob("rd_map", "rd_montage", search_directories=("", "diffusion")),
    MapJob("rd_map_atlas", "rd_parcel_montage", search_directories=("", "diffusion")),
    MapJob("mo_map", "mo_montage", search_directories=("", "diffusion")),
    MapJob("mo_map_atlas", "mo_parcel_montage", search_directories=("", "diffusion")),
    MapJob(
        "tensor_residual_map",
        "tensor_residual_montage",
        search_directories=("", "diffusion"),
    ),
    MapJob(
        "tensor_residual_map_atlas",
        "tensor_residual_parcel_montage",
        search_directories=("", "diffusion"),
    ),
)

MAP_JOB_LOOKUP: Dict[str, MapJob] = {job.base: job for job in MAP_JOBS}

PROJECTION_TARGETS: Dict[str, str] = {
    "Ki_map_atlas": "ki_projection_parcel",
    "vp_map_atlas": "vp_projection_parcel",
    "CBF_tikhonov_map_atlas": "cbf_projection_parcel",
    "CTH_tikhonov_map_atlas": "cth_projection_parcel",
    "MTT_tikhonov_map_atlas": "mtt_projection_parcel",
    "fa_map_atlas": "fa_projection_parcel",
    "md_map_atlas": "md_projection_parcel",
    "ad_map_atlas": "ad_projection_parcel",
    "rd_map_atlas": "rd_projection_parcel",
    "mo_map_atlas": "mo_projection_parcel",
    "tensor_residual_map_atlas": "tensor_residual_projection_parcel",
}

ParcelStatistics = Dict[Tuple[str, str], Dict[int, float]]

_ATLAS_SEGMENTATION_PATH = (
    "segmentation",
    "segmentation",
    "mri",
    "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz",
)


def _atlas_segmentation_path(nifti_directory: str) -> str:
    return os.path.join(nifti_directory, *_ATLAS_SEGMENTATION_PATH)


def _load_atlas_segmentation(nifti_directory: str) -> tuple[np.ndarray, np.ndarray]:
    atlas_path = _atlas_segmentation_path(nifti_directory)
    if not os.path.isfile(atlas_path):
        raise FileNotFoundError(atlas_path)

    atlas_img = nib.load(atlas_path)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    if atlas_data.ndim != 3:
        raise ValueError("Atlas segmentation is not 3D")

    atlas_labels = np.unique(atlas_data)
    atlas_labels = atlas_labels[atlas_labels != 0]
    if atlas_labels.size == 0:
        raise ValueError("Atlas segmentation contains no labelled parcels")

    return atlas_data, atlas_labels


def generate_parametric_montages(
    analysis_directory: str,
    image_directory: str,
    dce_path: str,
    *,
    anatomical_overlay: str | None = None,
    segmentation_path: str | None = None,
    head_mask_path: str | None = None,
    icv_mask_path: str | None = None,
    rows: int = ROWS,
    cols: int = COLS,
    dpi: int = DPI,
) -> None:
    """Render PNG montages for available parametric maps.

    Parameters
    ----------
    analysis_directory:
        Directory where the parametric NIfTI maps are stored.
    image_directory:
        Root ``Images`` directory for the current subject.
    dce_path:
        Path to the native-space DCE reference volume used to select slices.
    anatomical_overlay:
        Optional path to a T1-weighted anatomical volume already aligned to the
        DCE reference. When provided, montage values will be rendered atop the
        grayscale anatomical background.
    segmentation_path:
        Optional atlas segmentation aligned to the DCE reference. When provided,
        montage rendering will restrict overlays to the labelled brain voxels.
    head_mask_path:
        Optional explicit path to a head mask aligned to the DCE reference. If
        not provided, common filenames will be searched near the DCE volume.
    icv_mask_path:
        Optional explicit path to an intracranial volume mask aligned to the DCE
        reference. If not provided, common filenames will be searched near the
        DCE volume.
    rows, cols:
        Layout of the montage grid.
    dpi:
        Resolution of the saved PNG files.
    """

    if not os.path.isdir(analysis_directory):
        return
    if not os.path.isfile(dce_path):
        print(f"[montage] DCE reference not found – skipping montage rendering: {dce_path}")
        return

    reference = _load_reference_volume(dce_path)
    if reference is None:
        print("[montage] Unable to load DCE reference volume – skipping montages.")
        return

    try:
        raw_reference_img = nib.load(dce_path)
    except Exception as exc:  # noqa: BLE001 - surface helpful context to CLI users
        print(f"[montage] Unable to load DCE reference volume – skipping montages: {exc}")
        return

    reference_img = nib.Nifti1Image(
        reference,
        np.array(raw_reference_img.affine, copy=True),
        raw_reference_img.header.copy() if raw_reference_img.header is not None else None,
    )

    brain_mask: np.ndarray | None = None
    segmentation_img: nib.Nifti1Image | None = None
    if segmentation_path:
        try:
            segmentation_img = nib.load(segmentation_path)
            segmentation_data = np.asarray(segmentation_img.get_fdata(), dtype=np.float32)
            if segmentation_data.ndim != 3:
                raise ValueError("Segmentation volume is not 3D")
            target = (reference.shape, reference_img.affine)
            if (
                segmentation_img.shape != reference.shape
                or not np.allclose(segmentation_img.affine, reference_img.affine)
            ):
                segmentation_img = resample_from_to(segmentation_img, target, order=0)
            segmentation_resampled = np.asarray(
                segmentation_img.get_fdata(), dtype=np.float32
            )
            brain_mask = np.isfinite(segmentation_resampled) & (segmentation_resampled > 0.5)
            if not brain_mask.any():
                brain_mask = None
        except FileNotFoundError:
            segmentation_img = None
        except Exception as exc:  # noqa: BLE001 - continue without mask when issues arise
            print(
                "[montage] Failed to load segmentation mask – continuing without "
                f"brain mask: {exc}"
            )
            segmentation_img = None
            brain_mask = None

    # Try to load external masks (FSL/FreeSurfer outputs) and prefer them
    ext_head_mask, ext_icv_mask = _load_external_masks(
        dce_path,
        reference_img,
        explicit_head=head_mask_path,
        explicit_icv=icv_mask_path,
    )

    # Load T1 underlay first so we can build a HEAD mask for cropping
    overlay = None
    head_mask: np.ndarray | None = None
    if anatomical_overlay:
        overlay, overlay_error = _prepare_anatomical_overlay(
            anatomical_overlay,
            reference_img,
            segmentation_img,
            head_mask_override=ext_head_mask,
            icv_mask_override=ext_icv_mask,
        )
        if overlay_error:
            print(f"[montage] {overlay_error} – continuing without anatomical overlay.")
            overlay = None
        # Fallback brain mask from T1 if atlas mask is absent
        if brain_mask is None and overlay is not None and isinstance(overlay.get("mask_brain"), np.ndarray):
            brain_mask = overlay["mask_brain"].astype(bool)
        # Head mask for cropping the tiles and masking the underlay
        if overlay is not None and isinstance(overlay.get("mask_head"), np.ndarray):
            head_mask = overlay["mask_head"].astype(bool)
    else:
        # Even without an anatomical underlay we can still crop by external head mask
        if ext_head_mask is not None:
            head_mask = ext_head_mask
        if brain_mask is None and ext_icv_mask is not None:
            brain_mask = ext_icv_mask

    # Build reference using head mask when available, else brain
    ref_info = _build_reference(reference, rows * cols, mask=head_mask if head_mask is not None else brain_mask)
    if ref_info is None:
        print("[montage] Reference volume contained no finite voxels – skipping montages.")
        return

    out_dir = os.path.join(image_directory, "AI", "Montages")
    os.makedirs(out_dir, exist_ok=True)

    generated_any = False
    for job in MAP_JOBS:
        for suffix, map_path in _find_available_maps(job, analysis_directory).items():
            try:
                output_name = job.output_base + suffix + job.output_ext
                out_path = os.path.join(out_dir, output_name)
                _render_montage(
                    map_path,
                    out_path,
                    job,
                    ref_info,
                    reference_img=reference_img,
                    rows=rows,
                    cols=cols,
                    dpi=dpi,
                    overlay=overlay,
                    brain_mask=brain_mask,
                    segmentation_img=segmentation_img,
                )
                generated_any = True
                print(f"[montage] Saved {os.path.relpath(out_path, start=image_directory)}")
            except Exception as exc:
                print(f"[montage] Failed to render {map_path}: {exc}")

    if not generated_any:
        print("[montage] No parametric maps found for montage rendering.")


def generate_projection_montages(
    analysis_directory: str,
    image_directory: str,
    nifti_directory: str,
    dce_path: str,
    *,
    rows: int = ROWS,
    cols: int = COLS,
    dpi: int = DPI,
    population_stats: ParcelStatistics | None = None,
) -> bool:
    """Render parcel-level projection montages for atlas-based metrics."""

    if not os.path.isdir(analysis_directory):
        return False
    if not os.path.isdir(nifti_directory):
        print(f"[projection] NIfTI directory missing – skipping: {nifti_directory}")
        return False
    if not os.path.isfile(dce_path):
        print(f"[projection] DCE reference not found – skipping projection rendering: {dce_path}")
        return False

    try:
        atlas_data, atlas_labels = _load_atlas_segmentation(nifti_directory)
    except FileNotFoundError as exc:
        print(f"[projection] Atlas segmentation missing – skipping: {exc}")
        return False
    except ValueError as exc:
        print(f"[projection] {exc} – skipping: {_atlas_segmentation_path(nifti_directory)}")
        return False

    reference = _load_reference_volume(dce_path)
    if reference is None:
        print("[projection] Unable to load DCE reference volume – skipping projections.")
        return False

    brain_mask = np.isfinite(atlas_data) & (atlas_data > 0)
    ref_info = _build_reference(reference, rows * cols, mask=brain_mask)
    if ref_info is None:
        print("[projection] Reference volume contained no finite voxels – skipping projections.")
        return False

    out_dir = os.path.join(image_directory, "AI", "Montages")
    os.makedirs(out_dir, exist_ok=True)

    generated_any = False
    stats_lookup: Mapping[Tuple[str, str], Dict[int, float]] = population_stats or {}

    for base, output_base in PROJECTION_TARGETS.items():
        job = MAP_JOB_LOOKUP.get(base)
        if job is None:
            continue
        available_maps = _find_available_maps(job, analysis_directory)
        suffixes = set(available_maps)
        if stats_lookup:
            suffixes.update(suffix for stat_base, suffix in stats_lookup if stat_base == base)

        for suffix in sorted(suffixes):
            map_path = available_maps.get(suffix)
            projected = None
            label_means = stats_lookup.get((base, suffix)) if stats_lookup else None
            if label_means is None and suffix and stats_lookup:
                label_means = stats_lookup.get((base, ""))

            try:
                if label_means:
                    projected = _projection_from_label_means(atlas_data, label_means)
                if projected is None and map_path:
                    projected = _parcel_mean_projection(map_path, atlas_data, atlas_labels)
                if projected is None:
                    continue

                output_name = output_base + suffix + job.output_ext
                out_path = os.path.join(out_dir, output_name)
                _render_projection_montage(
                    projected,
                    ref_info,
                    job,
                    out_path,
                    rows=rows,
                    cols=cols,
                    dpi=dpi,
                )
                generated_any = True
                print(f"[projection] Saved {os.path.relpath(out_path, start=image_directory)}")
            except Exception as exc:
                target = map_path if map_path else f"population statistics for {base}{suffix}"
                print(f"[projection] Failed to render {target}: {exc}")

    if not generated_any:
        print("[projection] No atlas maps found for projection rendering.")

    return generated_any


def _projection_from_label_means(
    atlas_data: np.ndarray, label_means: Mapping[int, float]
) -> np.ndarray | None:
    projected = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    filled_any = False

    for label, value in label_means.items():
        mask = atlas_data == int(label)
        if not np.any(mask):
            continue
        projected[mask] = np.float32(value)
        filled_any = True

    if not filled_any:
        return None
    return projected


def _parcel_label_means(
    map_path: str, atlas_data: np.ndarray, atlas_labels: np.ndarray
) -> Dict[int, float]:
    img = nib.load(map_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")
    if data.shape != atlas_data.shape:
        raise ValueError("Atlas segmentation and parametric map shapes do not match")

    label_means: Dict[int, float] = {}
    for label in atlas_labels:
        mask = atlas_data == int(label)
        if not np.any(mask):
            continue
        values = data[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        label_means[int(label)] = float(np.mean(values, dtype=np.float32))

    return label_means


def _collect_dataset_parcel_means(
    analysis_directory: str, atlas_data: np.ndarray, atlas_labels: np.ndarray
) -> Dict[Tuple[str, str], Dict[int, float]]:
    dataset_means: Dict[Tuple[str, str], Dict[int, float]] = {}

    for base in PROJECTION_TARGETS:
        job = MAP_JOB_LOOKUP.get(base)
        if job is None:
            continue
        for suffix, map_path in _find_available_maps(job, analysis_directory).items():
            label_means = _parcel_label_means(map_path, atlas_data, atlas_labels)
            if label_means:
                dataset_means[(base, suffix)] = label_means

    return dataset_means


def _iter_population_dataset_dirs(
    data_root: str, include_controls: bool
) -> Sequence[str]:
    if not os.path.isdir(data_root):
        return []

    dataset_dirs = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if not os.path.isdir(path):
            continue
        if name == "controls":
            continue
        dataset_dirs.append(path)

    if include_controls:
        controls_root = os.path.join(data_root, "controls")
        if os.path.isdir(controls_root):
            for name in sorted(os.listdir(controls_root)):
                path = os.path.join(controls_root, name)
                if os.path.isdir(path):
                    dataset_dirs.append(path)

    return dataset_dirs


def build_population_projection_stats(
    data_root: str, *, include_controls: bool = False
) -> ParcelStatistics:
    dataset_dirs = _iter_population_dataset_dirs(data_root, include_controls)
    if not dataset_dirs:
        print("[projection] No datasets available for population aggregation.")
        return {}

    aggregates: defaultdict[Tuple[str, str], defaultdict[int, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0])
    )

    contributing_datasets = 0
    for dataset_dir in dataset_dirs:
        analysis_directory = os.path.join(dataset_dir, "Analysis")
        nifti_directory = os.path.join(dataset_dir, "NIfTI")
        if not os.path.isdir(analysis_directory) or not os.path.isdir(nifti_directory):
            continue

        try:
            atlas_data, atlas_labels = _load_atlas_segmentation(nifti_directory)
        except (FileNotFoundError, ValueError):
            continue

        try:
            dataset_means = _collect_dataset_parcel_means(
                analysis_directory, atlas_data, atlas_labels
            )
        except Exception as exc:  # noqa: BLE001 - surface helpful context
            print(f"[projection] Failed to collect parcel means for {dataset_dir}: {exc}")
            continue

        if not dataset_means:
            continue

        contributing_datasets += 1
        for key, label_means in dataset_means.items():
            stats_for_key = aggregates[key]
            for label, value in label_means.items():
                bucket = stats_for_key[label]
                bucket[0] += float(value)
                bucket[1] += 1

    population_stats: ParcelStatistics = {}
    for key, label_summaries in aggregates.items():
        means = {
            label: total / count for label, (total, count) in label_summaries.items() if count
        }
        if means:
            population_stats[key] = means

    if population_stats:
        print(
            f"[projection] Aggregated parcel means from {contributing_datasets} dataset(s)."
        )
    else:
        print("[projection] No parcel statistics available for population aggregation.")

    return population_stats


def _mk_specthl() -> LinearSegmentedColormap:
    anchors = [
        (0.00, (0, 0, 0)),
        (0.10, (0, 0, 40)),
        (0.22, (0, 0, 120)),
        (0.35, (60, 0, 170)),
        (0.50, (130, 0, 180)),
        (0.62, (200, 0, 120)),
        (0.73, (230, 30, 60)),
        (0.83, (255, 120, 0)),
        (0.92, (255, 200, 0)),
        (1.00, (255, 255, 255)),
    ]
    xs, cols = zip(*anchors)
    cols = np.array(cols, dtype=float) / 255.0
    return LinearSegmentedColormap.from_list("specthl", list(zip(xs, cols)), N=256)


def _get_cmap(name: str | None) -> mpl.colors.Colormap:
    key = name or "specthl"
    try:
        if key.lower() == "specthl":
            return _mk_specthl()
        return mpl.colormaps[key].copy()
    except Exception:
        # last-ditch fallback so a bad/None name never kills the montage
        return mpl.colormaps["viridis"].copy()


def _load_reference_volume(dce_path: str) -> np.ndarray | None:
    img = nib.load(dce_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim == 4:
        data = np.nanmean(data, axis=-1)
    if data.ndim != 3:
        return None
    return data


def _prepare_anatomical_overlay(
    overlay_path: str,
    reference_img: nib.Nifti1Image,
    segmentation_img: nib.Nifti1Image | None = None,
    *,
    head_mask_override: np.ndarray | None = None,
    icv_mask_override: np.ndarray | None = None,
) -> tuple[Dict[str, Any] | None, str | None]:
    try:
        overlay_img = nib.load(overlay_path)
    except FileNotFoundError:
        return None, f"Anatomical overlay not found: {overlay_path}"
    except Exception as exc:  # noqa: BLE001 - report to CLI users
        return None, f"Failed to load anatomical overlay {overlay_path}: {exc}"

    overlay_data = np.asarray(overlay_img.get_fdata(), dtype=np.float32)
    if overlay_data.ndim == 4:
        overlay_data = np.nanmean(overlay_data, axis=-1, dtype=np.float32)
    if overlay_data.ndim != 3:
        return None, f"Anatomical overlay {overlay_path} is not a 3D volume"

    if not np.isfinite(overlay_data).any():
        return None, "Anatomical overlay has no finite voxels"

    # Prefer explicit/external masks if provided; otherwise derive from T1 (+atlas)
    mask_head = head_mask_override
    mask_brain = icv_mask_override

    if mask_head is None and mask_brain is None:
        ref_filename = reference_img.get_filename() if hasattr(reference_img, "get_filename") else None
        if ref_filename:
            fs_icv = _try_freesurfer_icv_from_nearby(ref_filename, reference_img)
            if fs_icv is not None:
                mask_brain = fs_icv
        if mask_brain is None and isinstance(overlay_path, str) and os.path.isfile(overlay_path):
            fsl_head, fsl_icv = _try_fsl_bet_masks(overlay_path, overlay_img)
            if fsl_icv is not None:
                mask_brain = fsl_icv
            if fsl_head is not None:
                mask_head = fsl_head

    if mask_head is not None or mask_brain is not None:
        try:
            if mask_head is None and mask_brain is not None:
                r = _voxel_radius(overlay_img, ICV_ERODE_MM)
                # approximate skull thickness back outwards
                mask_head = binary_dilation(mask_brain, ball(max(1, r)))
            if mask_brain is None and mask_head is not None:
                r = _voxel_radius(overlay_img, ICV_ERODE_MM)
                mask_brain = binary_erosion(mask_head, ball(max(1, r)))
                mask_brain = binary_fill_holes(mask_brain)
        except Exception:
            pass

    if mask_head is None or mask_brain is None:
        try:
            derived_head, derived_brain = _build_head_mask(
                overlay_img,
                segmentation_img,
                dilate_mm=HEAD_DILATE_MM,
                erode_mm=ICV_ERODE_MM,
            )
            if mask_head is None:
                mask_head = derived_head
            if mask_brain is None:
                mask_brain = derived_brain
        except Exception as exc:  # noqa: BLE001 - keep rendering with degraded mask
            if mask_head is None:
                mask_head = np.isfinite(overlay_data) & (overlay_data > 0)
            if mask_brain is None:
                mask_brain = None
            print(
                "[montage] Failed to build anatomical head mask – falling back to "
                "finite voxels only:",
                exc,
            )

    if mask_head is None or not np.any(mask_head):
        mask_head = np.isfinite(overlay_data) & (overlay_data > 0)

    # Use head mask for the underlay, so air is gone though head tissue remains
    masked_overlay = np.array(overlay_data, copy=True)
    masked_overlay[~mask_head] = 0.0
    overlay_for_range = np.array(overlay_data, copy=True)
    overlay_for_range[~mask_head] = np.nan
    vmin, vmax = _estimate_intensity_range(overlay_for_range)

    alpha_map = _alpha_feather_from_mask(mask_head, overlay_img, FEATHER_MM)

    clean_overlay_img = nib.Nifti1Image(
        masked_overlay,
        np.array(overlay_img.affine, copy=True),
        overlay_img.header.copy() if overlay_img.header is not None else None,
    )

    return {
        "volume": clean_overlay_img,
        "alpha": 0.65,
        "vmin": vmin,
        "vmax": vmax,
        "mask_head": mask_head,    # for underlay display and cropping
        "alpha_map": alpha_map,    # per-pixel alpha to feather the rim
        "mask_brain": mask_brain,  # for parametric overlays
        "mask": mask_head,         # keep legacy key pointing to head mask
        "affine": np.array(clean_overlay_img.affine, copy=True),
        "header": clean_overlay_img.header.copy()
        if clean_overlay_img.header is not None
        else None,
    }, None


def _voxel_radius(img: nib.Nifti1Image, mm: float) -> int:
    zoom = np.array(img.header.get_zooms()[:3], dtype=float)
    r = int(np.ceil(mm / max(1e-6, min(zoom))))
    return max(1, r)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = label(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    if counts.size <= 1:
        return mask.astype(bool)
    counts[0] = 0
    idx = int(np.argmax(counts))
    return lab == idx


def _t1_envelope(img: nib.Nifti1Image) -> np.ndarray:
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    finite = np.isfinite(vol)
    vals = vol[finite]
    if vals.size == 0:
        return finite
    try:
        thr = float(threshold_otsu(vals))
    except Exception:
        thr = float(np.percentile(vals, 2.0))
    soft = finite & (vol > thr)
    soft = binary_closing(soft, ball(1))
    soft = binary_fill_holes(soft)
    soft = remove_small_objects(soft, min_size=HEAD_MIN_VOXELS, connectivity=2)
    soft = _largest_component(soft)
    soft = binary_dilation(soft, ball(_voxel_radius(img, HEAD_EXTRA_MM)))
    return soft.astype(bool)


def _build_head_mask(
    overlay_img: nib.Nifti1Image,
    segmentation_img: nib.Nifti1Image | None,
    *,
    dilate_mm: float = 8.0,
    erode_mm: float = 3.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    # Envelope of all head tissues (no air)
    head = _t1_envelope(overlay_img)

    icv_prior: np.ndarray | None = None
    if segmentation_img is not None:
        seg_img = segmentation_img
        if seg_img.shape != overlay_img.shape or not np.allclose(
            seg_img.affine, overlay_img.affine
        ):
            seg_img = resample_from_to(seg_img, overlay_img, order=0)
        seg_data = np.asarray(seg_img.get_fdata(), dtype=np.float32)
        brain = np.isfinite(seg_data) & (seg_data > 0.5)
        if brain.any():
            brain = binary_fill_holes(brain)
            brain = _largest_component(brain)
            icv_prior = brain.astype(bool)
            r = _voxel_radius(overlay_img, dilate_mm)
            grown = binary_dilation(icv_prior, ball(r))
            head = np.asarray(head | grown, dtype=bool)
    r_erode = _voxel_radius(overlay_img, erode_mm)
    icv_from_head = binary_erosion(head, ball(r_erode))
    icv_from_head = binary_fill_holes(icv_from_head)
    icv_from_head = _largest_component(icv_from_head)

    if icv_prior is not None:
        from skimage.morphology import binary_closing as _bclose

        icv = _bclose(icv_from_head | icv_prior, ball(1))
    else:
        icv = icv_from_head

    head = binary_fill_holes(head)
    head = _largest_component(head)
    return head.astype(bool), icv.astype(bool)


def _alpha_feather_from_mask(
    head_mask: np.ndarray, img: nib.Nifti1Image, feather_mm: float
) -> np.ndarray:
    """Per-pixel alpha in [0,1] that ramps up inside the head over ``feather_mm``."""

    mask = np.asarray(head_mask, dtype=bool)
    if mask.shape != img.shape:
        raise ValueError("alpha feather mask shape mismatch")

    r = max(1, _voxel_radius(img, float(feather_mm)))
    # distance to boundary inside the mask
    d_in = distance_transform_edt(mask)
    alpha = np.clip(d_in / float(r), 0.0, 1.0).astype(np.float32)
    alpha[~mask] = 0.0
    # light smoothing keeps the ramp silky
    alpha = gaussian_filter(alpha, sigma=0.6, mode="nearest")
    return alpha


def _load_binary_mask(path: str, reference_img: nib.Nifti1Image) -> np.ndarray | None:
    """Load a binary mask from disk, resample to the reference grid, return bool array."""

    try:
        img = nib.load(path)
    except Exception:
        return None
    if img.ndim != 3 and (hasattr(img, "shape") and len(img.shape) != 3):
        return None

    try:
        if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine):
            img = resample_from_to(img, (reference_img.shape, reference_img.affine), order=0)
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        mask = np.isfinite(data) & (data > 0.5)
        if not mask.any():
            # sometimes masks are 0/1 but smoothed; open+fill to revive thin rims
            mask = data > 0.1
            mask = binary_opening(mask, ball(1))
        mask = binary_fill_holes(mask)
        mask = _largest_component(mask)
        return mask.astype(bool)
    except Exception:
        return None


def _search_nearby(paths: list[str], names: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for root in paths:
        for name in names:
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate):
                out.append(candidate)
    return out


def _load_external_masks(
    dce_path: str,
    reference_img: nib.Nifti1Image,
    *,
    explicit_head: str | None = None,
    explicit_icv: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Locate and load external head/ICV masks if available.

    Search order:
      1) Explicit paths if provided.
      2) Same folder as DCE.
      3) Siblings commonly present in this repository layout:
         - ../NIfTI/
         - ../segmentation/segmentation/mri/
    """

    dce_dir = os.path.abspath(os.path.dirname(dce_path))
    siblings = [
        dce_dir,
        os.path.abspath(os.path.join(dce_dir, "..", "NIfTI")),
        os.path.abspath(os.path.join(dce_dir, "..", "segmentation", "segmentation", "mri")),
    ]

    head_candidates: list[str] = []
    icv_candidates: list[str] = []
    if explicit_head:
        head_candidates.append(explicit_head)
    if explicit_icv:
        icv_candidates.append(explicit_icv)
    head_candidates.extend(_search_nearby(siblings, MASK_CANDIDATES_HEAD))
    icv_candidates.extend(_search_nearby(siblings, MASK_CANDIDATES_ICV))

    head_mask = None
    icv_mask = None
    head_source = None
    icv_source = None
    for path in head_candidates:
        candidate = _load_binary_mask(path, reference_img)
        if candidate is not None:
            head_mask = candidate
            head_source = path
            break
    for path in icv_candidates:
        candidate = _load_binary_mask(path, reference_img)
        if candidate is not None:
            icv_mask = candidate
            icv_source = path
            break

    if head_mask is not None:
        print(f"[montage] Using external head mask: {head_source}")
    else:
        print("[montage] No external head mask engaged.")
    if icv_mask is not None:
        print(f"[montage] Using external ICV mask: {icv_source}")
    else:
        print("[montage] No external ICV mask engaged.")

    return head_mask, icv_mask


def _has_cmd(cmd: str) -> bool:
    try:
        return shutil.which(cmd) is not None
    except Exception:
        return False


def _try_freesurfer_icv_from_nearby(
    dce_path: str, reference_img: nib.Nifti1Image
) -> np.ndarray | None:
    """Look for FreeSurfer brainmask.* near the DCE and load as ICV."""

    dce_dir = os.path.abspath(os.path.dirname(dce_path))
    candidates: list[str] = []
    siblings = [
        dce_dir,
        os.path.abspath(os.path.join(dce_dir, "..", "NIfTI")),
        os.path.abspath(os.path.join(dce_dir, "..", "segmentation", "segmentation", "mri")),
    ]
    for root in siblings:
        for name in FREESURFER_BRAINMASK:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                candidates.append(path)
    if candidates:
        print("[montage] FreeSurfer brainmask candidates:", candidates)
    for path in candidates:
        try:
            img = nib.load(path)
            if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine):
                img = resample_from_to(img, (reference_img.shape, reference_img.affine), order=0)
            data = np.asarray(img.get_fdata(), dtype=np.float32)
            mask = np.isfinite(data) & (data > 0.5)
            mask = binary_fill_holes(mask)
            mask = _largest_component(mask)
            if mask.any():
                print("[montage] Using FreeSurfer ICV:", path)
                return mask.astype(bool)
        except Exception:
            continue
    return None


def _try_fsl_bet_masks(
    overlay_path: str, overlay_img: nib.Nifti1Image
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Use FSL BET on the underlay to get ICV, then grow to head."""

    if not _has_cmd("bet"):
        return None, None
    try:
        with tempfile.TemporaryDirectory(prefix="pbrain_bet_") as td:
            prefix = os.path.join(td, "ovl")
            cmd = ["bet", overlay_path, prefix, "-m", "-R", "-f", "0.20"]
            print("[montage] Running FSL BET:", " ".join(cmd))
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print("[montage] FSL BET failed:", result.stderr.strip()[:240])
                return None, None
            mask_path = prefix + "_mask.nii.gz"
            if not os.path.isfile(mask_path):
                print("[montage] FSL BET produced no _mask.nii.gz")
                return None, None
            mask_img = nib.load(mask_path)
            if mask_img.shape != overlay_img.shape or not np.allclose(mask_img.affine, overlay_img.affine):
                mask_img = resample_from_to(mask_img, overlay_img, order=0)
            mask = np.asarray(mask_img.get_fdata(), dtype=np.float32) > 0.5
            radius = _voxel_radius(overlay_img, ICV_ERODE_MM if ICV_ERODE_MM > 0 else 1.0)
            head = binary_dilation(mask, ball(max(1, radius)))
            head = binary_fill_holes(head)
            head = _largest_component(head)
            print("[montage] Using FSL BET-derived ICV and grown HEAD.")
            return head.astype(bool), mask.astype(bool)
    except Exception as exc:  # noqa: BLE001 - keep rendering with degraded mask
        print("[montage] FSL BET path failed:", exc)
    return None, None


def _estimate_intensity_range(volume: np.ndarray) -> tuple[float | None, float | None]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return None, None

    vmin, vmax = np.percentile(finite, (2.0, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None

    if np.isclose(vmin, vmax):
        delta = np.abs(vmin) if vmin else 1.0
        vmax = vmin + delta

    return float(vmin), float(vmax)


def _build_reference(
    volume: np.ndarray, tiles: int, *, mask: np.ndarray | None = None
) -> Dict[str, np.ndarray] | None:
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != volume.shape:
            raise ValueError("Brain mask shape does not match reference volume")
        mask = mask & np.isfinite(volume)
    else:
        mask = np.isfinite(volume) & (np.abs(volume) > EPS)
    if not mask.any():
        return None

    union_xy = np.any(mask, axis=2)
    union_xy_r = np.rot90(union_xy, ROT90)
    r0, r1, c0, c1 = _tight_bbox_from_mask(union_xy_r, pad=BBOX_PADDING)

    bbox_fracs = {
        "r0_frac": r0 / max(1, union_xy_r.shape[0]),
        "r1_frac": r1 / max(1, union_xy_r.shape[0]),
        "c0_frac": c0 / max(1, union_xy_r.shape[1]),
        "c1_frac": c1 / max(1, union_xy_r.shape[1]),
    }

    z_indices = _spaced_unique_indices(0, volume.shape[2] - 1, tiles)
    z_fracs = (
        z_indices / max(1, volume.shape[2] - 1)
        if volume.shape[2] > 1
        else np.zeros_like(z_indices)
    )

    return {
        "bbox_fracs": bbox_fracs,
        "z_fracs": z_fracs,
        "rotate": ROT90,
    }


def _tight_bbox_from_mask(mask2d: np.ndarray, pad: int = 3) -> tuple[int, int, int, int]:
    if not mask2d.any():
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
        # Expand the vertical bounds so the background extent matches the
        # horizontal padding when rendering montages. This keeps the montage
        # tiles the same size while balancing the surrounding background.
        diff = width - height
        extra_top = diff // 2
        extra_bottom = diff - extra_top
        r0 = max(0, r0 - extra_top)
        r1 = min(mask2d.shape[0], r1 + extra_bottom)
        # If we were clipped by the image boundaries, compensate on the
        # opposite side to preserve the requested padding.
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


def _map_bbox_from_ref(ref_bbox: Dict[str, float], shape_rot: Sequence[int]) -> tuple[int, int, int, int]:
    hx, hy = shape_rot
    r0 = int(np.floor(ref_bbox["r0_frac"] * hx))
    r1 = int(np.ceil(ref_bbox["r1_frac"] * hx))
    c0 = int(np.floor(ref_bbox["c0_frac"] * hy))
    c1 = int(np.ceil(ref_bbox["c1_frac"] * hy))
    r0 = max(0, min(r0, hx - 1))
    r1 = max(r0 + 1, min(r1, hx))
    c0 = max(0, min(c0, hy - 1))
    c1 = max(c0 + 1, min(c1, hy))
    return r0, r1, c0, c1


def _map_z_from_ref(z_fracs: np.ndarray, nz: int) -> np.ndarray:
    z = np.asarray(z_fracs, dtype=np.float64)
    if nz <= 0 or z.size == 0:
        return np.zeros(z.size, dtype=np.intp)

    idx = np.rint(z * (nz - 1)).astype(np.intp, copy=False)
    return np.clip(idx, 0, nz - 1)


def _resize_mask(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    target_shape = (int(target_shape[0]), int(target_shape[1]))
    if mask.shape == target_shape:
        return mask.astype(bool)

    resized = resize(
        mask.astype(np.float32),
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    if resized.size:
        resized = gaussian_filter(resized, sigma=0.5, mode="reflect")
    return resized >= 0.5


def _find_available_maps(job: MapJob, analysis_directory: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    search_dirs = job.search_directories or ("",)
    for rel_dir in search_dirs:
        base_dir = analysis_directory if not rel_dir else os.path.join(analysis_directory, rel_dir)
        for pattern in job.candidate_patterns():
            for path in sorted(glob.glob(os.path.join(base_dir, pattern))):
                suffix = _extract_suffix(path, job.base)
                if suffix not in found or path.endswith(".nii.gz"):
                    found[suffix] = path
    return found


def _parcel_mean_projection(
    map_path: str, atlas_data: np.ndarray, atlas_labels: np.ndarray
) -> np.ndarray | None:
    img = nib.load(map_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")
    if data.shape != atlas_data.shape:
        raise ValueError("Atlas segmentation and parametric map shapes do not match")

    projected = np.full_like(data, np.nan, dtype=np.float32)
    for label in atlas_labels:
        mask = atlas_data == label
        if not np.any(mask):
            continue
        values = data[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        projected[mask] = np.mean(values, dtype=np.float32)

    if not np.isfinite(projected).any():
        return None
    return projected


def _extract_suffix(path: str, base: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name == base:
        return ""
    if name.startswith(base):
        return name[len(base) :]
    return ""


def _render_montage(
    map_path: str,
    out_path: str,
    job: MapJob,
    ref_info: Dict[str, np.ndarray],
    *,
    reference_img: nib.Nifti1Image,
    rows: int,
    cols: int,
    dpi: int,
    overlay: Dict[str, Any] | None = None,
    brain_mask: np.ndarray | None = None,
    segmentation_img: nib.Nifti1Image | None = None,
) -> None:
    img = nib.load(map_path)
    # Resample every map to the DCE reference grid for slice parity
    from nibabel.processing import resample_from_to

    target = (reference_img.shape, reference_img.affine)
    if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine):
        img = resample_from_to(img, target, order=1)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    # Light display smoothing for coarse diffusion grids
    try:
        zoom_src = np.array(img.header.get_zooms()[:3], dtype=float)
        zoom_ref = np.array(reference_img.header.get_zooms()[:3], dtype=float)
        ratio = max(zoom_src[0] / zoom_ref[0], zoom_src[1] / zoom_ref[1])
        if ratio > 1.4:
            data = gaussian_filter(data, sigma=(0.6, 0.6, 0.0), mode="nearest")
    except Exception:
        pass
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    if brain_mask is not None:
        mask_data = np.asarray(brain_mask, dtype=bool)
        if mask_data.shape != data.shape:
            if segmentation_img is None:
                raise ValueError("Brain mask shape does not match parametric map")
            from nibabel.processing import resample_from_to

            target = (data.shape, img.affine)
            seg_img = segmentation_img
            if (
                segmentation_img.shape != data.shape
                or not np.allclose(segmentation_img.affine, img.affine)
            ):
                try:
                    seg_img = resample_from_to(segmentation_img, target, order=0)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        "Failed to resample brain mask to parametric map space"
                    ) from exc
            seg_data = np.asarray(seg_img.get_fdata(), dtype=np.float32)
            mask_data = np.isfinite(seg_data) & (seg_data > 0.5)
        brain_mask = mask_data
        valmask3d = np.isfinite(data) & mask_data
    else:
        valmask3d = np.isfinite(data) & (np.abs(data) > EPS)
    union_xy = np.any(valmask3d, axis=2)
    union_xy_r = np.rot90(union_xy, ref_info["rotate"])

    r0, r1, c0, c1 = _map_bbox_from_ref(ref_info["bbox_fracs"], union_xy_r.shape)
    z_indices = _map_z_from_ref(ref_info["z_fracs"], data.shape[2])
    if z_indices.size < rows * cols:
        pad_value = z_indices[-1] if z_indices.size else 0
        z_indices = np.pad(z_indices, (0, rows * cols - z_indices.size), constant_values=pad_value)
    else:
        z_indices = z_indices[: rows * cols]

    cmap = _get_cmap(job.cmap_name)
    norm_data = data if brain_mask is None else np.where(brain_mask, data, np.nan)
    norm, tick_values = _build_normalizer(
        norm_data, job, mask_zero_override=False if brain_mask is not None else None
    )
    if (getattr(job, "metric", None) or "").lower() == "fa":
        vmax = float(getattr(norm, "vmax", 1.0))
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=False)
    # Keep sub-vmin faint so low values remain visible; NaNs remain invisible
    under_rgba = list(cmap(0.0)); under_rgba[-1] = 0.45
    cmap = cmap.with_extremes(bad=(0, 0, 0, 0), under=tuple(under_rgba))

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.2, rows * 2.2), facecolor=(0.0, 0.0, 0.0, 0.0)
    )
    fig.patch.set_alpha(0.0)
    axes = axes.ravel()

    overlay_volume = overlay.get("volume") if overlay else None
    overlay_mask_volume = overlay.get("mask_head") if overlay else None
    overlay_affine = overlay.get("affine") if overlay else None
    overlay_header = overlay.get("header") if overlay else None
    overlay_alpha = float(overlay.get("alpha", 0.65)) if overlay else 1.0
    overlay_alpha_map = overlay.get("alpha_map") if overlay else None
    overlay_vmin = overlay.get("vmin") if overlay else None
    overlay_vmax = overlay.get("vmax") if overlay else None
    overlay_data = None
    overlay_mask_data = None
    overlay_alpha_data = None
    overlay_z_indices = None
    if overlay_volume is not None:
        ovl_img = None
        if isinstance(overlay_volume, str):
            ovl_img = nib.load(overlay_volume)
        elif hasattr(overlay_volume, "shape") and hasattr(overlay_volume, "affine"):
            ovl_img = overlay_volume
        elif isinstance(overlay_volume, np.ndarray):
            affine = overlay_affine if overlay_affine is not None else img.affine
            header = overlay_header if overlay_header is not None else img.header
            ovl_img = nib.Nifti1Image(overlay_volume, affine, header)
        if ovl_img is not None:
            from nibabel.processing import resample_from_to

            target = (data.shape[:3], img.affine)  # already equals reference grid
            ovl_affine = ovl_img.affine
            alpha_img = None
            if overlay_alpha_map is not None:
                alpha_img = nib.Nifti1Image(
                    np.asarray(overlay_alpha_map, dtype=np.float32),
                    overlay_affine if overlay_affine is not None else ovl_affine,
                )
            if ovl_img.shape[:3] != data.shape[:3] or not np.allclose(ovl_affine, img.affine):
                ovl_img = resample_from_to(ovl_img, target, order=1)
            overlay_data = np.asarray(ovl_img.get_fdata(), dtype=np.float32)
            if alpha_img is not None:
                if alpha_img.shape[:3] != data.shape[:3] or not np.allclose(
                    alpha_img.affine, img.affine
                ):
                    alpha_img = resample_from_to(alpha_img, target, order=1)
                overlay_alpha_data = np.asarray(alpha_img.get_fdata(), dtype=np.float32)
            if overlay_mask_volume is not None:
                mask_img = None
                if isinstance(overlay_mask_volume, str):
                    mask_img = nib.load(overlay_mask_volume)
                elif hasattr(overlay_mask_volume, "shape") and hasattr(
                    overlay_mask_volume, "affine"
                ):
                    mask_img = overlay_mask_volume
                elif isinstance(overlay_mask_volume, np.ndarray):
                    mask_affine = overlay_affine if overlay_affine is not None else ovl_img.affine
                    mask_img = nib.Nifti1Image(overlay_mask_volume.astype(np.float32), mask_affine)
                if mask_img is not None:
                    if mask_img.shape[:3] != data.shape[:3] or not np.allclose(
                        mask_img.affine, img.affine
                    ):
                        mask_img = resample_from_to(mask_img, target, order=0)
                    overlay_mask_data = (
                        np.asarray(mask_img.get_fdata(), dtype=np.float32) > 0.5
                    )
            if overlay_mask_data is None and overlay_data is not None:
                overlay_mask_data = np.isfinite(overlay_data)
            if overlay_data is not None:
                overlay_z_indices = _map_z_from_ref(
                    ref_info["z_fracs"], overlay_data.shape[2]
                )

    nz = data.shape[2]
    if nz == 0:
        raise ValueError("volume has zero z-extent")
    if z_indices.size and z_indices.max() >= nz:
        raise RuntimeError(f"z mapping produced {int(z_indices.max())} with nz={nz}")

    if overlay_data is not None and overlay_z_indices is not None:
        if overlay_z_indices.size < rows * cols:
            pad_value = (
                overlay_z_indices[-1] if overlay_z_indices.size else 0
            )
            overlay_z_indices = np.pad(
                overlay_z_indices,
                (0, rows * cols - overlay_z_indices.size),
                constant_values=pad_value,
            )
        else:
            overlay_z_indices = overlay_z_indices[: rows * cols]

    for tile_index, (ax, z) in enumerate(zip(axes, z_indices)):
        zi = int(np.clip(z, 0, data.shape[2] - 1))
        overlay_zi = None
        if overlay_z_indices is not None and tile_index < overlay_z_indices.size:
            overlay_zi = int(
                np.clip(overlay_z_indices[tile_index], 0, overlay_data.shape[2] - 1)
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_facecolor("#e0e0e0")
        for spine in ax.spines.values():
            spine.set_visible(False)

        sl = data[:, :, zi]
        slr = np.rot90(sl, ref_info["rotate"])
        slc = slr[r0:r1, c0:c1]
        if np.isfinite(slc).sum() == 0:
            print(f"[montage] z={zi}: all NaN after mask/crop")

        union_crop = union_xy_r[r0:r1, c0:c1]
        extent = (0.0, 1.0, 1.0, 0.0)
        overlay_union_crop = union_crop
        render_shape = slc.shape
        if overlay_data is not None:
            index_for_overlay = zi if overlay_zi is None else overlay_zi
            overlay_slice = overlay_data[:, :, index_for_overlay]
            overlay_rot = np.rot90(overlay_slice, ref_info["rotate"])
            overlay_r0, overlay_r1, overlay_c0, overlay_c1 = _map_bbox_from_ref(
                ref_info["bbox_fracs"], overlay_rot.shape
            )
            overlay_crop = overlay_rot[overlay_r0:overlay_r1, overlay_c0:overlay_c1]
            # Underlay visibility follows HEAD mask, not the brain union
            overlay_alpha_crop = None
            if overlay_mask_data is not None:
                mask_slice_raw = overlay_mask_data[:, :, index_for_overlay]
                mask_rot = np.rot90(mask_slice_raw, ref_info["rotate"])
                overlay_union = mask_rot.astype(bool)
            else:
                overlay_union = union_xy_r
            overlay_union_crop = overlay_union[
                overlay_r0:overlay_r1, overlay_c0:overlay_c1
            ]
            if overlay_alpha_data is not None:
                alpha_slice = overlay_alpha_data[:, :, index_for_overlay]
                alpha_rot = np.rot90(alpha_slice, ref_info["rotate"])
                overlay_alpha_crop = alpha_rot[
                    overlay_r0:overlay_r1, overlay_c0:overlay_c1
                ]
                overlay_alpha_crop = overlay_alpha_crop * overlay_union_crop.astype(np.float32)
                overlay_alpha_crop = np.clip(overlay_alpha_crop, 0.0, 1.0)
            overlay_mask_slice = None
            if overlay_mask_data is not None:
                mask_slice_raw = overlay_mask_data[:, :, index_for_overlay]
                mask_rot = np.rot90(mask_slice_raw, ref_info["rotate"])
                overlay_mask_slice = mask_rot[overlay_r0:overlay_r1, overlay_c0:overlay_c1]
            overlay_mask = (~overlay_union_crop) | (~np.isfinite(overlay_crop))
            if overlay_mask_slice is not None:
                overlay_mask |= ~overlay_mask_slice
            overlay_arr = np.ma.array(overlay_crop, mask=overlay_mask)
            ax.imshow(
                overlay_arr,
                cmap="gray",
                # underlay: keep sharp-ish skull edges; avoid “holes” by preferring bilinear
                interpolation="bilinear",
                origin="upper",
                vmin=overlay_vmin,
                vmax=overlay_vmax,
                extent=extent,
                alpha=(
                    overlay_alpha * overlay_alpha_crop
                    if overlay_alpha_crop is not None
                    else overlay_alpha
                ),
            )
            render_shape = overlay_crop.shape

        mask_slice = valmask3d[:, :, zi]
        mask_slice_rot = np.rot90(mask_slice, ref_info["rotate"])
        mask_slice_crop = mask_slice_rot[r0:r1, c0:c1]

        if brain_mask is None and job.mask_zero:
            finite_vals = slc[np.isfinite(slc) & (slc > 0)]
            if finite_vals.size:
                cutoff = np.percentile(finite_vals, 0.1)
                eps_dyn = max(cutoff, 1e-6)
            else:
                eps_dyn = 1e-6
            mask_slice_crop &= slc > eps_dyn

        slc_filled = np.array(slc, copy=True)
        slc_filled[~mask_slice_crop] = 0.0
        slc_render = slc_filled.astype(np.float32)
        mask_render = mask_slice_crop
        union_render = union_crop
        if render_shape != slc.shape:
            slc_render = resize(
                slc_render,
                render_shape,
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            mask_render = _resize_mask(mask_slice_crop, render_shape)
            union_render = _resize_mask(union_crop, render_shape)
        # seal tiny pinholes from resampling
        try:
            from skimage.morphology import binary_closing as _bclose

            valid_mask = _bclose(union_render & mask_render, footprint=np.ones((2, 2), bool))
        except Exception:
            valid_mask = union_render & mask_render
        arr = np.ma.array(
            slc_render,
            mask=(~valid_mask) | (~np.isfinite(slc_render)),
        )
        # If the slice is entirely below global vmin (e.g., FA in cortex),
        # use a local min/max so low-but-real values remain visible.
        finite_vals = arr.compressed() if hasattr(arr, "compressed") else np.asarray([], dtype=float)
        if finite_vals.size and np.nanmax(finite_vals) <= float(getattr(norm, "vmin", 0.0)):
            loc_vmin = float(np.nanmin(finite_vals))
            loc_vmax = float(np.nanpercentile(finite_vals, 99.0))
            if loc_vmax > loc_vmin:
                local_norm = mcolors.Normalize(vmin=loc_vmin, vmax=loc_vmax, clip=False)
                ax.imshow(
                    arr,
                    cmap=cmap,
                    norm=local_norm,
                    interpolation="bilinear",
                    origin="upper",
                    alpha=overlay_alpha,
                    extent=extent,
                )
            else:
                ax.imshow(
                    arr,
                    cmap=cmap,
                    norm=norm,
                    interpolation="bilinear",
                    origin="upper",
                    alpha=overlay_alpha,
                    extent=extent,
                )
        else:
            ax.imshow(
                arr,
                cmap=cmap,
                norm=norm,
                interpolation="bilinear",
                origin="upper",
                alpha=overlay_alpha,
                extent=extent,
            )

    # Hide any unused axes when there are fewer slices than tiles
    for ax in axes[len(z_indices) :]:
        ax.axis("off")

    cax = fig.add_axes([0.93, 0.12, 0.015, 0.3])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    if tick_values:
        cb.set_ticks(tick_values)
        cb.set_ticklabels([f"{val:.2g}" for val in tick_values])
    cb.ax.tick_params(labelsize=8, colors="black")
    for spine in cb.ax.spines.values():
        spine.set_edgecolor("black")

    plt.subplots_adjust(left=0.02, right=0.9, top=0.96, bottom=0.02, wspace=0.02, hspace=0.02)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _render_projection_montage(
    data: np.ndarray,
    ref_info: Dict[str, np.ndarray],
    job: MapJob,
    out_path: str,
    *,
    rows: int,
    cols: int,
    dpi: int,
) -> None:
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    finite_mask = np.isfinite(data)
    if job.mask_zero:
        finite_mask &= np.abs(data) > EPS
    if not finite_mask.any():
        raise ValueError("Projection map contains no finite values")

    union_xy = np.any(finite_mask, axis=2)
    union_xy_r = np.rot90(union_xy, ref_info["rotate"])

    r0, r1, c0, c1 = _map_bbox_from_ref(ref_info["bbox_fracs"], union_xy_r.shape)
    z_indices = _map_z_from_ref(ref_info["z_fracs"], data.shape[2])
    if z_indices.size < rows * cols:
        pad_value = z_indices[-1] if z_indices.size else 0
        z_indices = np.pad(z_indices, (0, rows * cols - z_indices.size), constant_values=pad_value)
    else:
        z_indices = z_indices[: rows * cols]

    cmap = _get_cmap(job.cmap_name)
    norm, tick_values = _build_projection_normalizer(data, job)
    if (getattr(job, "metric", None) or "").lower() == "fa":
        vmax = float(getattr(norm, "vmax", 1.0))
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=False)
    under_rgba = list(cmap(0.0)); under_rgba[-1] = 0.45
    cmap = cmap.with_extremes(bad=(0, 0, 0, 0), under=tuple(under_rgba))

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.2, rows * 2.2), facecolor=(0.0, 0.0, 0.0, 0.0)
    )
    fig.patch.set_alpha(0.0)
    axes = axes.ravel()

    nz = data.shape[2]
    if nz == 0:
        raise ValueError("volume has zero z-extent")
    if z_indices.size and z_indices.max() >= nz:
        raise RuntimeError(f"z mapping produced {int(z_indices.max())} with nz={nz}")

    for ax, z in zip(axes, z_indices):
        zi = int(np.clip(z, 0, data.shape[2] - 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_facecolor("#e0e0e0")
        for spine in ax.spines.values():
            spine.set_visible(False)

        sl = data[:, :, zi]
        slr = np.rot90(sl, ref_info["rotate"])
        slc = slr[r0:r1, c0:c1]
        if np.isfinite(slc).sum() == 0:
            print(f"[montage] z={zi}: all NaN after mask/crop")

        union_crop = union_xy_r[r0:r1, c0:c1]
        if job.mask_zero:
            finite_vals = slc[np.isfinite(slc) & (slc > 0)]
            if finite_vals.size:
                cutoff = np.percentile(finite_vals, 0.1)
                eps_dyn = max(cutoff, 1e-6)
            else:
                eps_dyn = 1e-6
            mask_slice = np.isfinite(slc) & (slc > eps_dyn)
        else:
            mask_slice = np.isfinite(slc)

        arr = np.ma.array(slc, mask=(~union_crop) | (~mask_slice))
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")

    for ax in axes[len(z_indices) :]:
        ax.axis("off")

    cax = fig.add_axes([0.93, 0.12, 0.015, 0.3])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    if tick_values:
        cb.set_ticks(tick_values)
        cb.set_ticklabels([f"{val:.2g}" for val in tick_values])
    cb.ax.tick_params(labelsize=8, colors="black")
    for spine in cb.ax.spines.values():
        spine.set_edgecolor("black")

    plt.subplots_adjust(left=0.02, right=0.9, top=0.96, bottom=0.02, wspace=0.02, hspace=0.02)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _build_normalizer(
    data: np.ndarray,
    job: MapJob,
    *,
    mask_zero_override: bool | None = None,
) -> tuple[mpl.colors.Normalize, list[float]]:
    vmin = float(job.vmin) if job.vmin is not None else np.nan
    vmax = float(job.vmax) if job.vmax is not None else np.nan

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        mask = np.isfinite(data)
        mask_zero = job.mask_zero if mask_zero_override is None else mask_zero_override
        if mask_zero:
            mask &= data > EPS
        finite_vals = data[mask]
        if finite_vals.size:
            vmin, vmax = _robust_bounds(finite_vals)
        else:
            vmin, vmax = 0.0, 1.0

    if vmax <= vmin:
        padding = abs(vmin) if vmin != 0 else 1.0
        vmax = vmin + padding

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    return norm, _default_ticks(vmin, vmax)


def _build_projection_normalizer(
    data: np.ndarray, job: MapJob
) -> tuple[mpl.colors.Normalize, list[float]]:
    mask = np.isfinite(data)
    if job.mask_zero:
        mask &= np.abs(data) > EPS

    finite_vals = data[mask]
    if finite_vals.size == 0:
        raise ValueError("Projection map contains no finite values for colour scaling")

    vmin, vmax = _robust_bounds(finite_vals)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    return norm, _default_ticks(vmin, vmax)


def _robust_bounds(values: np.ndarray, lower_q: float = 2.0, upper_q: float = 98.0) -> tuple[float, float]:
    """Return percentile-based limits that are resilient to outliers."""

    lo = float(np.nanpercentile(values, lower_q))
    hi = float(np.nanpercentile(values, upper_q))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = lo
        padding = abs(center) if center != 0 else 1.0
        lo = center - padding * 0.5
        hi = center + padding * 0.5

    return lo, hi


def _round_bounds(lo: float, hi: float) -> tuple[float, float]:
    span = hi - lo
    if span <= 0:
        return float(lo), float(hi if hi > lo else lo + 1.0)

    if span >= 1.0:
        lo_r = np.floor(lo)
        hi_r = np.ceil(hi)
        if lo_r == hi_r:
            hi_r = lo_r + 1.0
        return float(lo_r), float(hi_r)

    decimals = int(np.ceil(-np.log10(span))) + 1
    factor = 10 ** decimals
    lo_r = np.floor(lo * factor) / factor
    hi_r = np.ceil(hi * factor) / factor
    if lo_r == hi_r:
        hi_r = lo_r + 1.0 / factor
    return float(lo_r), float(hi_r)


def _default_ticks(lo: float, hi: float) -> list[float]:
    lo_r, hi_r = _round_bounds(lo, hi)
    if hi_r <= lo_r:
        return [lo_r]
    steps = np.linspace(lo_r, hi_r, 5)
    return [float(x) for x in steps]
