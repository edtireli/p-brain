import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Header
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from utils import parameters as parameter_config

try:  # Matplotlib is optional in some deployment setups.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is always available in tests
    plt = None

try:
    from nibabel.processing import resample_from_to
except ImportError:  # pragma: no cover - nibabel ships the helper in supported envs
    resample_from_to = None


_ATLAS_SEGMENTATION_CANDIDATES = (
    ("segmentation", "mri", "aparc.DKTatlas+aseg.deep.nii.gz"),
    ("segmentation", "mri", "aparc.DKTatlas+aseg.deep.mgz"),
    (
        "segmentation",
        "segmentation",
        "mri",
        "aparc.DKTatlas+aseg.deep.nii.gz",
    ),
    (
        "segmentation",
        "segmentation",
        "mri",
        "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz",
    ),
)


@dataclass(frozen=True)
class DiffusionAcquisition:
    volume_path: str
    bval_path: str
    bvec_path: str
    label: str
    model: str


# ---------------------------------------------------------------------------
# Tissue mask discovery and handling
# ---------------------------------------------------------------------------

_TISSUE_PATTERNS: Dict[str, Dict[str, Iterable[str]]] = {
    "white_matter": {
        "json_label": "white_matter",
        "plot_label": "White matter",
        "patterns": (
            "wm.nii",
            "wm.nii.gz",
            "_wm.nii",
            "_wm.nii.gz",
        ),
    },
    "cortical_gm": {
        "json_label": "cortical_gm",
        "plot_label": "Cortical GM",
        "patterns": (
            "cortical_gm.nii",
            "cortical_gm.nii.gz",
            "_cortical_gm.nii",
            "_cortical_gm.nii.gz",
        ),
    },
    "subcortical_gm": {
        "json_label": "subcortical_gm",
        "plot_label": "Subcortical GM",
        "patterns": (
            "subcortical_gm.nii",
            "subcortical_gm.nii.gz",
            "_subcortical_gm.nii",
            "_subcortical_gm.nii.gz",
        ),
    },
    "gm_brainstem": {
        "json_label": "gm_brainstem",
        "plot_label": "Brainstem GM",
        "patterns": (
            "gm_brainstem.nii",
            "gm_brainstem.nii.gz",
            "brainstem_gm.nii",
            "brainstem_gm.nii.gz",
        ),
    },
    "gm_cerebellum": {
        "json_label": "gm_cerebellum",
        "plot_label": "Cerebellar GM",
        "patterns": (
            "gm_cerebellum.nii",
            "gm_cerebellum.nii.gz",
            "cerebellum_gm.nii",
            "cerebellum_gm.nii.gz",
        ),
    },
    "wm_cerebellum": {
        "json_label": "wm_cerebellum",
        "plot_label": "Cerebellar WM",
        "patterns": (
            "wm_cerebellum.nii",
            "wm_cerebellum.nii.gz",
            "cerebellum_wm.nii",
            "cerebellum_wm.nii.gz",
        ),
    },
    "wm_cc": {
        "json_label": "wm_cc",
        "plot_label": "Corpus callosum",
        "patterns": (
            "wm_cc.nii",
            "wm_cc.nii.gz",
            "corpus_callosum_wm.nii",
            "corpus_callosum_wm.nii.gz",
        ),
    },
    "brainstem": {
        "json_label": "brainstem",
        "plot_label": "Brainstem",
        "patterns": (
            "brainstem.nii",
            "brainstem.nii.gz",
            "brainstem_wm.nii",
            "brainstem_wm.nii.gz",
        ),
    },
    "csf": {
        "json_label": "csf",
        "plot_label": "CSF",
        "patterns": (
            "csf.nii",
            "csf.nii.gz",
            "pve_csf.nii",
            "pve_csf.nii.gz",
            "csf_mask.nii",
            "csf_mask.nii.gz",
        ),
    },
}


def _find_file_with_patterns(search_roots: Iterable[str], patterns: Iterable[str]) -> Optional[str]:
    """Return the first file found under ``search_roots`` ending with any pattern."""

    normalized_patterns = tuple(p.lower() for p in patterns)
    visited = set()
    for root in search_roots:
        if not root or root in visited or not os.path.isdir(root):
            continue
        visited.add(root)
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                lower = filename.lower()
                if any(lower.endswith(pattern) for pattern in normalized_patterns):
                    return os.path.join(dirpath, filename)
    return None


ResampleTarget = Tuple[Tuple[int, ...], np.ndarray, Optional[Nifti1Header]]


def _reference_grid(img: nib.Nifti1Image) -> ResampleTarget:
    """Return a resampling target limited to the spatial dimensions of ``img``."""

    shape = img.shape
    if len(shape) > 3:
        shape = shape[:3]
    header: Optional[Nifti1Header]
    try:
        header = img.header.copy()
    except AttributeError:  # pragma: no cover - fallback for header-less objects
        header = None
    return shape, np.array(img.affine, copy=True), header


def _dwi_signal_mask(
    dwi_4d: np.ndarray, bvals_1d: np.ndarray, *, b0_thresh: float = 50.0
) -> np.ndarray:
    """Return a mask capturing intrinsic diffusion signal, prioritising b0 volumes."""

    b0_idx = bvals_1d < b0_thresh
    if np.count_nonzero(b0_idx):
        b0 = dwi_4d[..., b0_idx].mean(axis=-1)
    else:
        b0 = dwi_4d.mean(axis=-1)
    nz = b0[b0 > 0]
    if nz.size:
        thr = float(max(np.percentile(nz, 5.0), np.finfo(np.float32).eps))
    else:
        thr = 0.0
    return b0 > thr


def _heal_metric_mask_edges(
    metric_mask: np.ndarray,
    signal_mask: Optional[np.ndarray],
    raw_metric_data: np.ndarray,
) -> None:
    """Ensure outermost slices survive masking when they contain diffusion signal."""

    if metric_mask.size == 0 or metric_mask.ndim < 3:
        return

    for edge_index, label in ((0, "first"), (-1, "final")):
        slice_mask = metric_mask[..., edge_index]
        if np.any(slice_mask):
            continue

        if signal_mask is not None:
            signal_slice = signal_mask[..., edge_index]
            if np.any(signal_slice):
                metric_mask[..., edge_index] |= signal_slice
                print(
                    f"[!] Expanding metric mask on {label} z-slice based on diffusion signal"
                )
                continue

        finite_slice = np.isfinite(raw_metric_data[..., edge_index])
        if np.any(finite_slice):
            metric_mask[..., edge_index] |= finite_slice
            print(
                f"[!] Expanding metric mask on {label} z-slice based on raw diffusion metrics"
            )


def _maybe_resample_to_dce(
    img: nib.Nifti1Image,
    target: Optional[ResampleTarget],
    label: str,
) -> nib.Nifti1Image:
    """Resample ``img`` to ``target`` when provided, preserving float32 storage."""

    if target is None:
        return img

    original_data = np.asarray(img.get_fdata(dtype=np.float32))
    finite_original = np.isfinite(original_data)
    if np.any(finite_original):
        original_signal = float(np.nanmax(np.abs(original_data[finite_original])))
    else:
        original_signal = 0.0

    # ``resample_from_to`` performs spline interpolation which does not handle
    # ``NaN`` values gracefully – even a single ``NaN`` can propagate through
    # the neighbourhood and collapse the entire resampled volume to ``NaN``.
    # The diffusion metrics purposefully store ``NaN`` outside their valid
    # tissue masks, so temporarily replace them with zeros during resampling
    # while separately tracking which voxels contained real signal.  The mask
    # is later re-applied in the resampled space to restore ``NaN`` padding.
    has_nans = np.logical_not(finite_original).any()
    if has_nans:
        filled_data = np.where(finite_original, original_data, 0.0).astype(
            np.float32
        )
        filled_header = img.header.copy()
        filled_header.set_data_dtype(np.float32)
        try:
            filled_header.set_slope_inter(1.0, 0.0)
        except AttributeError:
            with np.errstate(invalid="ignore"):
                filled_header["scl_slope"] = 1.0
                filled_header["scl_inter"] = 0.0
        resample_source = nib.Nifti1Image(filled_data, img.affine, filled_header)
        resample_source.set_data_dtype(np.float32)
        nan_mask_img = nib.Nifti1Image(
            finite_original.astype(np.float32), img.affine, img.header
        )
        nan_mask_img.set_data_dtype(np.float32)
    else:
        resample_source = img
        nan_mask_img = None

    ref_shape, ref_affine, ref_header = target
    original_shape = tuple(int(dim) for dim in original_data.shape[: len(ref_shape)])
    target_shape = tuple(int(dim) for dim in ref_shape)
    if any(t_dim < o_dim for t_dim, o_dim in zip(target_shape, original_shape)):
        print(
            "[!] DCE reference has a smaller field of view – keeping diffusion-space geometry for",
            label,
        )
        return img
    if img.shape == ref_shape and np.allclose(img.affine, ref_affine):
        return img

    if resample_from_to is None:
        print(
            f"[!] Cannot resample {label} to DCE space – nibabel.processing unavailable"
        )
        return img

    try:
        resampled = resample_from_to(resample_source, (ref_shape, ref_affine))
    except Exception as exc:  # noqa: BLE001 - expose runtime issues to CLI users
        print(f"[!] Failed to resample {label} to DCE space: {exc}")
        return img

    data = resampled.get_fdata(dtype=np.float32)
    if nan_mask_img is not None:
        try:
            resampled_mask = resample_from_to(
                nan_mask_img, (ref_shape, ref_affine), order=0
            )
            mask_data = resampled_mask.get_fdata(dtype=np.float32)
            valid_mask = mask_data > 0.5
            data = np.where(valid_mask, data, np.float32(np.nan))
        except Exception as exc:  # noqa: BLE001 - expose runtime issues to CLI users
            print(
                "[!] Failed to resample diffusion validity mask; falling back to "
                f"diffusion-space geometry for {label}: {exc}"
            )
            return img
    finite_resampled = np.isfinite(data)
    has_resampled_signal = np.any(finite_resampled)
    if has_resampled_signal:
        resampled_signal = float(np.nanmax(np.abs(data[finite_resampled])))
    else:
        resampled_signal = 0.0

    signal_floor = np.finfo(np.float32).eps
    if original_signal > signal_floor and (
        resampled_signal <= signal_floor or not has_resampled_signal
    ):
        print(
            f"[!] Resampling {label} collapsed non-zero diffusion values – keeping "
            "diffusion-space geometry."
        )
        fallback_header = img.header.copy()
        fallback_header.set_data_dtype(np.float32)
        try:
            fallback_header.set_slope_inter(1.0, 0.0)
        except AttributeError:
            with np.errstate(invalid="ignore"):
                fallback_header["scl_slope"] = 1.0
                fallback_header["scl_inter"] = 0.0
        fallback_img = nib.Nifti1Image(
            np.asarray(original_data, dtype=np.float32, order="C"),
            img.affine,
            fallback_header,
        )
        fallback_img.set_data_dtype(np.float32)
        return fallback_img

    header = resampled.header.copy()
    header.set_data_dtype(np.float32)
    try:
        header.set_slope_inter(1.0, 0.0)
    except AttributeError:
        with np.errstate(invalid="ignore"):
            header["scl_slope"] = 1.0
            header["scl_inter"] = 0.0
    if ref_header is not None:
        try:
            qform, qcode = ref_header.get_qform(coded=True)
        except AttributeError:
            qform = None
            qcode = 0
        else:
            if qcode:
                try:
                    header.set_qform(qform, int(qcode))
                except Exception:  # pragma: no cover - best-effort geometry sync
                    pass
        try:
            sform, scode = ref_header.get_sform(coded=True)
        except AttributeError:
            sform = None
            scode = 0
        else:
            if scode:
                try:
                    header.set_sform(sform, int(scode))
                except Exception:  # pragma: no cover - best-effort geometry sync
                    pass

    resampled = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32, order="C"),
        np.array(resampled.affine, copy=True),
        header,
    )
    resampled.set_data_dtype(np.float32)
    return resampled


def _load_mask(mask_path: str, reference_img: nib.Nifti1Image) -> Optional[np.ndarray]:
    if mask_path is None:
        return None

    mask_img = nib.load(mask_path)
    mask_shape = mask_img.shape
    if len(mask_shape) > 3:
        # Discard non-spatial dimensions (e.g., singleton time/channel axes).
        data = mask_img.get_fdata()
        spatial = data[..., 0]
        spatial = np.squeeze(spatial)
        mask_img = nib.Nifti1Image(spatial, mask_img.affine)
        mask_shape = mask_img.shape

    ref_shape, ref_affine, _ = _reference_grid(reference_img)
    if mask_shape != ref_shape or not np.allclose(mask_img.affine, ref_affine):
        if resample_from_to is None:
            print(f"[!] Cannot resample {mask_path} – nibabel.processing unavailable")
            return None
        try:
            mask_img = resample_from_to(mask_img, (ref_shape, ref_affine), order=0)
            print(f"[!] Resampled mask {os.path.basename(mask_path)} to diffusion grid")
        except Exception as exc:
            print(f"[!] Failed to resample mask {mask_path}: {exc}")
            return None
    return (mask_img.get_fdata() > 0.5).astype(bool)


def _collect_tissue_masks(
    nifti_directory: str, analysis_directory: str, reference_img: nib.Nifti1Image
) -> Dict[str, Tuple[np.ndarray, Dict[str, str]]]:
    """Load available tissue masks and resample them to ``reference_img``."""

    search_roots = list(
        dict.fromkeys(
            filter(
                None,
                (
                    analysis_directory,
                    os.path.dirname(analysis_directory),
                    nifti_directory,
                    os.path.dirname(nifti_directory),
                ),
            )
        )
    )

    masks: Dict[str, Tuple[np.ndarray, Dict[str, str]]] = {}
    for tissue, info in _TISSUE_PATTERNS.items():
        mask_path = _find_file_with_patterns(search_roots, info["patterns"])
        if not mask_path:
            continue
        mask = _load_mask(mask_path, reference_img)
        if mask is None:
            continue
        masks[tissue] = (mask, info)

    atlas = _load_atlas_segmentation(nifti_directory, reference_img)
    if atlas is not None:
        atlas_data, atlas_labels = atlas
        label_lookup = _load_label_lookup()
        derived_masks = _derive_tissue_masks_from_atlas(
            atlas_data,
            atlas_labels,
            label_lookup,
        )
        for tissue, derived_mask in derived_masks.items():
            info = _TISSUE_PATTERNS.get(tissue)
            if info is None:
                continue
            derived_bool = np.asarray(derived_mask, dtype=bool)
            if tissue in masks:
                existing_mask, existing_info = masks[tissue]
                combined = np.asarray(existing_mask, dtype=bool) | derived_bool
                masks[tissue] = (combined, existing_info)
            else:
                masks[tissue] = (derived_bool, info)
    return masks


def _classify_atlas_label(label_name: str) -> Optional[str]:
    """Return the canonical tissue key for an atlas label name."""

    if not label_name:
        return None

    lower = label_name.lower()

    if (
        "ventricle" in lower
        or lower.endswith("vent")
        or "choroid" in lower
        or "plexus" in lower
        or lower.endswith("csf")
        or "csf" in lower
    ):
        return "csf"

    if "cerebellum" in lower:
        if "white" in lower:
            return "wm_cerebellum"
        if "cortex" in lower or lower.startswith("ctx-"):
            return "gm_cerebellum"
    if "corpus" in lower or "callosum" in lower:
        return "wm_cc"
    if "brain-stem" in lower or "brainstem" in lower:
        return "brainstem"
    if "white" in lower or lower.endswith("wm") or "white-matter" in lower:
        return "white_matter"
    if lower.startswith("ctx-") or "cortex" in lower:
        return "cortical_gm"
    if any(token in lower for token in ("thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala", "accumbens", "ventraldc", "ventraldc", "nucleus", "globus", "hypothalamus")):
        return "subcortical_gm"
    if "gm" in lower and "brainstem" in lower:
        return "gm_brainstem"
    return None


def _derive_tissue_masks_from_atlas(
    atlas_data: np.ndarray,
    atlas_labels: np.ndarray,
    label_lookup: Dict[int, str],
) -> Dict[str, np.ndarray]:
    """Generate tissue masks using an aligned atlas segmentation."""

    tissue_masks: Dict[str, np.ndarray] = {}
    for label in atlas_labels:
        name = label_lookup.get(int(label), "")
        tissue_key = _classify_atlas_label(name)
        if tissue_key is None:
            continue

        mask = atlas_data == int(label)
        if not np.any(mask):
            continue

        if tissue_key in tissue_masks:
            tissue_masks[tissue_key] |= mask
        else:
            tissue_masks[tissue_key] = mask.astype(bool)

    # Derive a brainstem GM mask when only a combined brainstem label exists.
    if "brainstem" in tissue_masks and "gm_brainstem" not in tissue_masks:
        tissue_masks["gm_brainstem"] = tissue_masks["brainstem"].copy()

    return tissue_masks


def _atlas_segmentation_path(nifti_directory: str) -> Optional[str]:
    for rel_parts in _ATLAS_SEGMENTATION_CANDIDATES:
        candidate = os.path.join(nifti_directory, *rel_parts)
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_label_lookup(lut_path: Optional[str] = None) -> Dict[int, str]:
    """Return a mapping from atlas indices to region names."""

    if lut_path is None:
        freesurfer_home = os.environ.get("FREESURFER_HOME")
        if freesurfer_home:
            lut_path = os.path.join(freesurfer_home, "FreeSurferColorLUT.txt")

    lookup: Dict[int, str] = {}
    if lut_path and os.path.exists(lut_path):
        try:
            with open(lut_path, "r") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = re.split(r"\s+", line)
                    if len(parts) >= 2 and parts[0].isdigit():
                        lookup[int(parts[0])] = parts[1]
        except Exception:
            # Fall back to numeric labels if the LUT is unreadable.
            pass
    return lookup


def _load_atlas_segmentation(
    nifti_directory: str, reference_img: nib.Nifti1Image
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the atlas segmentation in diffusion space.

    Returns ``None`` when the atlas cannot be found or aligned to the reference
    diffusion image.
    """

    atlas_path = _atlas_segmentation_path(nifti_directory)
    if not atlas_path or not os.path.isfile(atlas_path):
        return None

    atlas_img = nib.load(atlas_path)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.float32)

    if atlas_data.ndim > 3:
        atlas_data = np.squeeze(atlas_data)
        if atlas_data.ndim > 3:
            return None
        atlas_img = nib.Nifti1Image(atlas_data, atlas_img.affine)

    if atlas_data.ndim != 3:
        return None

    ref_shape, ref_affine, _ = _reference_grid(reference_img)

    if atlas_img.shape != ref_shape or not np.allclose(
        atlas_img.affine, ref_affine
    ):
        if resample_from_to is None:
            print(
                f"[!] Cannot resample atlas segmentation – nibabel.processing unavailable: {atlas_path}"
            )
            return None
        try:
            atlas_img = resample_from_to(
                atlas_img, (ref_shape, ref_affine), order=0
            )
            atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.float32)
            print("[!] Resampled atlas segmentation to diffusion grid")
        except Exception as exc:
            print(f"[!] Failed to resample atlas segmentation: {exc}")
            return None

    atlas_labels = np.unique(atlas_data)
    atlas_labels = atlas_labels[atlas_labels != 0]
    if atlas_labels.size == 0:
        return None

    return atlas_data.astype(np.int32), atlas_labels.astype(np.int32)


def _load_atlas_segmentation_dce(
    nifti_directory: str,
) -> Optional[Tuple[np.ndarray, nib.Nifti1Image]]:
    """Load the atlas segmentation in DCE space without resampling."""

    atlas_path = _atlas_segmentation_path(nifti_directory)
    if not atlas_path or not os.path.isfile(atlas_path):
        return None

    atlas_img = nib.load(atlas_path)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.float32)

    if atlas_data.ndim > 3:
        atlas_data = np.squeeze(atlas_data)
        if atlas_data.ndim > 3:
            return None
        atlas_img = nib.Nifti1Image(atlas_data, atlas_img.affine, atlas_img.header)

    if atlas_data.ndim != 3:
        return None

    return atlas_data.astype(np.int32), atlas_img


def _parcel_metadata(
    atlas_data: np.ndarray,
    atlas_labels: np.ndarray,
    label_lookup: Dict[int, str],
    wm_mask: Optional[np.ndarray],
) -> Dict[int, Dict[str, object]]:
    metadata: Dict[int, Dict[str, object]] = {}
    for label in atlas_labels:
        mask = atlas_data == int(label)
        indices = np.where(mask)
        voxel_count = int(indices[0].size)
        if voxel_count == 0:
            continue

        name = label_lookup.get(int(label), str(int(label)))
        if wm_mask is not None:
            overlap = int(np.count_nonzero(wm_mask[indices]))
            wm_fraction = overlap / voxel_count if voxel_count else 0.0
            is_wm = wm_fraction >= 0.5
        else:
            lower_name = name.lower()
            is_wm = any(
                token in lower_name for token in ("white", "wm", "callosum", "corpus")
            )

        metadata[int(label)] = {
            "indices": indices,
            "name": name,
            "voxel_count": voxel_count,
            "is_wm": is_wm,
        }

    return metadata


def _is_white_matter_label(name: str) -> bool:
    lower_name = name.lower()
    return any(
        token in lower_name for token in ("white", "wm", "callosum", "corpus")
    )


def _parcel_means_dce(
    metric_data: np.ndarray,
    atlas_data: np.ndarray,
    label_lookup: Dict[int, str],
    *,
    restrict_to_wm: bool = False,
    fallback_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    parcel_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    parcels: Dict[str, Dict[str, float]] = {}

    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        mask = atlas_data == int(label)
        values = metric_data[mask]
        values = values[np.isfinite(values)]
        if values.size == 0 and fallback_data is not None:
            fallback_values = fallback_data[mask]
            fallback_values = fallback_values[np.isfinite(fallback_values)]
            values = fallback_values
        if values.size == 0:
            continue

        name = label_lookup.get(int(label), str(int(label)))
        if restrict_to_wm and not _is_white_matter_label(name):
            continue

        mean_value = float(np.mean(values, dtype=np.float32))
        parcel_map[mask] = np.float32(mean_value)
        parcels[name] = {
            "label": int(label),
            "mean": mean_value,
            "voxel_count": int(mask.sum()),
        }

    return parcel_map, parcels


def _parcel_means(
    metric_data: np.ndarray,
    metadata: Dict[int, Dict[str, object]],
    *,
    restrict_to_wm: bool = False,
    fallback_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    parcel_map = np.full(metric_data.shape, np.nan, dtype=np.float32)
    parcels: Dict[str, Dict[str, float]] = {}

    for label, info in metadata.items():
        if restrict_to_wm and not info["is_wm"]:
            continue

        indices = info["indices"]
        values = metric_data[indices]
        values = values[np.isfinite(values)]
        if values.size == 0 and fallback_data is not None:
            fallback_values = fallback_data[indices]
            fallback_values = fallback_values[np.isfinite(fallback_values)]
            values = fallback_values
        if values.size == 0:
            continue

        mean_value = float(np.mean(values, dtype=np.float32))
        parcel_map[indices] = np.float32(mean_value)
        parcels[info["name"]] = {
            "label": int(label),
            "mean": mean_value,
            "voxel_count": int(info["voxel_count"]),
        }

    return parcel_map, parcels


def _ensure_image_directory(image_directory: Optional[str], analysis_directory: str) -> Optional[str]:
    if image_directory:
        return image_directory
    analysis_parent = os.path.dirname(analysis_directory)
    if analysis_parent:
        fallback = os.path.join(analysis_parent, "Images")
        if os.path.isdir(fallback) or not os.path.exists(fallback):
            return fallback
    return None


def _plot_metric_histogram(
    metric_name: str,
    metric_data: np.ndarray,
    masks: Dict[str, Tuple[np.ndarray, Dict[str, str]]],
    image_directory: Optional[str],
):
    if plt is None or not image_directory:
        return

    save_dir = os.path.join(image_directory, "Diffusion")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plotted = False
    max_height = 0.0

    for tissue, (mask, info) in masks.items():
        values = metric_data[mask]
        values = values[~np.isnan(values)]
        if not values.size:
            continue
        hist, *_ = plt.hist(
            values, bins=60, alpha=0.5, label=info["plot_label"], density=True
        )
        if hist.size:
            max_height = max(max_height, float(np.max(hist)))
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel(metric_name.upper())
    plt.ylabel("Density")
    plt.title(f"Distribution of {metric_name.upper()} across tissues")
    plt.legend()
    if max_height:
        plt.ylim(top=max_height * 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric_name.lower()}_histogram.png"))
    plt.close()


def find_wm_mask(nifti_directory):
    """Return path to a white matter mask within ``nifti_directory`` if found."""

    patterns = ("wm.nii", "wm.nii.gz", "_wm.nii", "_wm.nii.gz")
    for root, _, files in os.walk(nifti_directory):
        for file in files:
            lower = file.lower()
            if any(lower.endswith(pattern) for pattern in patterns):
                return os.path.join(root, file)
    return None


def find_dwi_files(
    nifti_directory: str,
    preferred_filenames: Optional[Sequence[str]] = None,
    *,
    include_metadata: bool = False,
):
    """Locate a diffusion NIfTI and its ``.bval``/``.bvec`` files.

    When ``include_metadata`` is ``True`` a :class:`DiffusionAcquisition`
    describing the file label and preferred reconstruction model is returned.
    Otherwise the legacy ``(nifti, bval, bvec)`` tuple is preserved.
    """

    file_groups = parameter_config.diffusion_file_groups()
    group_priority = parameter_config.diffusion_file_priority()
    model_map = parameter_config.diffusion_model_map()

    def _label_for_path(path: str) -> str:
        lower = os.path.basename(path).lower()
        for group, patterns in file_groups.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if pattern_lower and lower.endswith(pattern_lower):
                    return group
        if "dti" in lower:
            return "dti"
        if "dwi" in lower:
            return "dwi"
        return "unknown"

    def _default_model(label: str) -> str:
        if "dwi" in label:
            return "CSD"
        return "DTI"

    def candidate_from(path: str) -> Optional[tuple[str, str, str]]:
        if not path.lower().endswith((".nii", ".nii.gz")):
            return None
        if not os.path.exists(path):
            return None
        base = os.path.splitext(path)[0]
        if path.lower().endswith(".nii.gz"):
            base = os.path.splitext(base)[0]
        bval = base + ".bval"
        bvec = base + ".bvec"
        if os.path.exists(bval) and os.path.exists(bvec):
            return path, bval, bvec
        return None

    def _build_acquisition(path: str) -> Optional[DiffusionAcquisition]:
        candidate = candidate_from(path)
        if not candidate:
            return None
        label = _label_for_path(path)
        model = model_map.get(label, _default_model(label))
        return DiffusionAcquisition(*candidate, label=label, model=model)

    def _preferred_paths(filename: str) -> Iterable[str]:
        if not filename:
            return

        if os.path.isabs(filename):
            base_path = filename
        else:
            base_path = os.path.join(nifti_directory, filename)

        candidates = [base_path]
        lower = base_path.lower()

        if lower.endswith(".nii.gz"):
            pass
        elif lower.endswith(".nii"):
            candidates.append(base_path + ".gz")
        else:
            candidates.extend((base_path + ".nii", base_path + ".nii.gz"))

        seen = set()
        for candidate_path in candidates:
            if not candidate_path:
                continue
            name = os.path.basename(candidate_path)
            if name.startswith("._"):
                continue
            if candidate_path in seen:
                continue
            seen.add(candidate_path)
            yield candidate_path

    def _format_return(acquisition: DiffusionAcquisition):
        if include_metadata:
            return acquisition
        return acquisition.volume_path, acquisition.bval_path, acquisition.bvec_path

    if preferred_filenames:
        for filename in preferred_filenames:
            for candidate_path in _preferred_paths(filename):
                acquisition = _build_acquisition(candidate_path)
                if acquisition:
                    return _format_return(acquisition)

    # Prepare candidate buckets honouring configured priority plus fallbacks.
    priority_labels = list(dict.fromkeys(group_priority + ("dti", "dwi", "unknown")))
    diffusion_candidates: Dict[str, List[DiffusionAcquisition]] = {
        label: [] for label in priority_labels
    }

    for root, _, files in os.walk(nifti_directory):
        for file in files:
            if file.startswith("._"):
                continue
            if not (file.endswith(".nii") or file.endswith(".nii.gz")):
                continue

            acquisition = _build_acquisition(os.path.join(root, file))
            if acquisition is None:
                continue
            diffusion_candidates.setdefault(acquisition.label, []).append(acquisition)

    for label in priority_labels:
        candidates = diffusion_candidates.get(label) or []
        if not candidates:
            continue
        selected = sorted(candidates, key=lambda item: item.volume_path)[0]
        return _format_return(selected)

    return (None, None, None) if not include_metadata else None


def _brain_union_mask(masks: Dict[str, Tuple[np.ndarray, Dict[str, str]]]) -> Optional[np.ndarray]:
    union = None
    for mask, _ in masks.values():
        union = mask if union is None else (union | mask)
    return union


def _union_of_tissues(
    masks: Dict[str, Tuple[np.ndarray, Dict[str, str]]], tissues: Iterable[str]
) -> Optional[np.ndarray]:
    union = None
    for tissue in tissues:
        if tissue not in masks:
            continue
        mask = np.asarray(masks[tissue][0], dtype=bool)
        union = mask if union is None else (union | mask)
    return union


def _compute_statistics(
    metric_name: str,
    metric_data: np.ndarray,
    masks: Dict[str, Tuple[np.ndarray, Dict[str, str]]],
) -> Dict[str, Dict[str, float]]:
    return _compute_statistics_with_options(metric_name, metric_data, masks)


def _compute_statistics_with_options(
    metric_name: str,
    metric_data: np.ndarray,
    masks: Dict[str, Tuple[np.ndarray, Dict[str, str]]],
    *,
    mask_selection: Optional[Iterable[str]] = None,
    tissue_groups: Optional[Dict[str, Iterable[str]]] = None,
    union_label: str = "brain",
    union_tissues: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}

    selected = tuple(mask_selection) if mask_selection is not None else tuple(masks.keys())
    for tissue in selected:
        if tissue not in masks:
            continue
        mask, info = masks[tissue]
        voxels = metric_data[mask]
        voxels = voxels[~np.isnan(voxels)]
        key = f"{info['json_label']}_median_total"
        stats[key] = {
            "median": float(np.nanmedian(voxels)) if voxels.size else float("nan"),
            "mean": float(np.nanmean(voxels)) if voxels.size else float("nan"),
            "voxel_count": int(mask.sum()),
        }

    if tissue_groups:
        for group, tissues in tissue_groups.items():
            union_mask = _union_of_tissues(masks, tissues)
            if union_mask is None:
                continue
            voxels = metric_data[union_mask]
            voxels = voxels[~np.isnan(voxels)]
            stats[f"{group}_median_total"] = {
                "median": float(np.nanmedian(voxels)) if voxels.size else float("nan"),
                "mean": float(np.nanmean(voxels)) if voxels.size else float("nan"),
                "voxel_count": int(np.sum(union_mask)),
            }

    union_mask = None
    if union_tissues is not None:
        union_mask = _union_of_tissues(masks, union_tissues)
    if union_mask is None:
        union_mask = _brain_union_mask(masks)

    if union_mask is not None:
        voxels = metric_data[union_mask]
        voxels = voxels[~np.isnan(voxels)]
        stats[f"{union_label}_median_total"] = {
            "median": float(np.nanmedian(voxels)) if voxels.size else float("nan"),
            "mean": float(np.nanmean(voxels)) if voxels.size else float("nan"),
            "voxel_count": int(np.sum(union_mask)),
        }

    return stats


_WM_TISSUES = ("white_matter", "wm_cerebellum", "wm_cc", "brainstem")
_BRAIN_TISSUE_GROUPS = {
    "white_matter": _WM_TISSUES,
    "gray_matter": ("cortical_gm", "subcortical_gm"),
    "cerebellum": ("gm_cerebellum", "wm_cerebellum"),
    "brainstem": ("brainstem", "gm_brainstem"),
}


def compute_fa(
    nifti_directory,
    analysis_directory,
    image_directory=None,
    diffusion_filename=None,
    *,
    dce_path: Optional[str] = None,
):
    preferred = (diffusion_filename,) if diffusion_filename else None
    acquisition = find_dwi_files(
        nifti_directory,
        preferred_filenames=preferred,
        include_metadata=True,
    )
    if acquisition is None:
        print(
            "[!] No diffusion volume found; update utils/parameters.py with the correct filename."
        )
        return

    if isinstance(acquisition, DiffusionAcquisition):
        dwi_path = acquisition.volume_path
        bval_path = acquisition.bval_path
        bvec_path = acquisition.bvec_path
        acquisition_label = acquisition.label or "unknown"
        acquisition_model = acquisition.model or "DTI"
    else:
        dwi_path, bval_path, bvec_path = acquisition
        acquisition_label = "unknown"
        acquisition_model = "DTI"

    print(
        f"[!] Computing FA from {os.path.basename(dwi_path)}"
        f" using {acquisition_model} workflow ({acquisition_label})"
    )
    img = nib.load(dwi_path)
    data = img.get_fdata()

    dce_resample_target: Optional[Tuple[Tuple[int, ...], np.ndarray]] = None
    if dce_path:
        if os.path.isfile(dce_path):
            try:
                dce_img = nib.load(dce_path)
                dce_resample_target = _reference_grid(dce_img)
            except Exception as exc:  # noqa: BLE001 - expose runtime issues to CLI users
                print(f"[!] Failed to load DCE reference for diffusion resampling: {exc}")
        else:
            print(
                "[!] DCE reference for diffusion resampling missing – proceeding without resampling: "
                f"{dce_path}"
            )

    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    signal_mask = _dwi_signal_mask(data, bvals)

    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)

    fa = tenfit.fa.astype(np.float32)
    md = tenfit.md.astype(np.float32)
    ad = tenfit.ad.astype(np.float32)
    rd = tenfit.rd.astype(np.float32)
    mode = tenfit.mode.astype(np.float32)

    predicted = tenfit.predict(gtab)
    residual = np.sqrt(np.nanmean((data - predicted) ** 2, axis=-1)).astype(np.float32)

    metrics = {
        "fa": {
            "data": fa,
            "units": "unitless",
            "tissue_groups": _BRAIN_TISSUE_GROUPS,
            "include_parcels": True,
            "restrict_parcels_to_wm": False,
            "parcel_selection": "all",
        },
        "md": {
            "data": md,
            "units": "mm^2/s",
            "tissue_groups": _BRAIN_TISSUE_GROUPS,
            "include_parcels": True,
            "restrict_parcels_to_wm": False,
            "parcel_selection": "all",
        },
        "ad": {
            "data": ad,
            "units": "mm^2/s",
            "tissue_groups": _BRAIN_TISSUE_GROUPS,
            "include_parcels": True,
            "restrict_parcels_to_wm": False,
            "parcel_selection": "all",
        },
        "rd": {
            "data": rd,
            "units": "mm^2/s",
            "tissue_groups": _BRAIN_TISSUE_GROUPS,
            "include_parcels": True,
            "restrict_parcels_to_wm": False,
            "parcel_selection": "all",
        },
        "mo": {
            "data": mode,
            "units": "unitless",
            "tissue_groups": _BRAIN_TISSUE_GROUPS,
            "include_parcels": True,
            "restrict_parcels_to_wm": False,
            "parcel_selection": "all",
        },
        "tensor_residual": {
            "data": residual,
            "units": "signal rms",
            "include_parcels": False,
            "compute_stats": False,
        },
    }

    os.makedirs(analysis_directory, exist_ok=True)

    diffusion_dir = os.path.join(analysis_directory, "diffusion")
    os.makedirs(diffusion_dir, exist_ok=True)

    masks = _collect_tissue_masks(nifti_directory, analysis_directory, img)
    if not masks:
        wm_mask_path = find_wm_mask(nifti_directory)
        if wm_mask_path:
            wm_mask = _load_mask(wm_mask_path, img)
            if wm_mask is not None:
                masks["white_matter"] = (wm_mask, _TISSUE_PATTERNS["white_matter"])

    image_directory = _ensure_image_directory(image_directory, analysis_directory)

    stats_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    atlas_summary: Dict[str, Dict[str, object]] = {}

    label_lookup = _load_label_lookup()

    atlas_loaded = _load_atlas_segmentation(nifti_directory, img)
    atlas_loaded_dce = _load_atlas_segmentation_dce(nifti_directory)
    atlas_dce_data: Optional[np.ndarray] = None
    atlas_dce_img: Optional[nib.Nifti1Image] = None
    if atlas_loaded_dce is not None:
        atlas_dce_data, atlas_dce_img = atlas_loaded_dce
        atlas_dce_data = np.asarray(atlas_dce_data, dtype=np.int32)
    parcel_metadata: Dict[int, Dict[str, object]] = {}
    atlas_brain_mask: Optional[np.ndarray] = None
    if atlas_loaded is not None:
        atlas_data, atlas_labels = atlas_loaded
        atlas_brain_mask = atlas_data > 0
        wm_mask = None
        if "white_matter" in masks:
            wm_mask = np.asarray(masks["white_matter"][0], dtype=bool)
        derived_tissues = _derive_tissue_masks_from_atlas(atlas_data, atlas_labels, label_lookup)
        for tissue, derived_mask in derived_tissues.items():
            info = _TISSUE_PATTERNS.get(tissue)
            if info is None:
                continue
            if tissue in masks:
                existing_mask, existing_info = masks[tissue]
                combined = np.asarray(existing_mask, dtype=bool) | np.asarray(derived_mask, dtype=bool)
                masks[tissue] = (combined, existing_info)
            else:
                masks[tissue] = (np.asarray(derived_mask, dtype=bool), info)
        parcel_metadata = _parcel_metadata(atlas_data, atlas_labels, label_lookup, wm_mask)
        if not parcel_metadata:
            parcel_metadata = {}

    brain_mask = _brain_union_mask(masks)
    if atlas_brain_mask is not None:
        atlas_brain_mask = np.asarray(atlas_brain_mask, dtype=bool)
        if brain_mask is None:
            brain_mask = atlas_brain_mask
        else:
            brain_mask = np.asarray(brain_mask, dtype=bool) | atlas_brain_mask

    if brain_mask is None:
        brain_mask = signal_mask
    else:
        brain_mask = np.logical_or(brain_mask, signal_mask)

    for metric_name, payload in metrics.items():
        raw_metric_data = payload["data"]
        voxel_mask_tissues = payload.get("voxel_mask_tissues")

        # Choose mask
        metric_mask = (
            _union_of_tissues(masks, voxel_mask_tissues)
            if payload.get("voxel_mask_tissues")
            else brain_mask
        )
        metric_mask = (
            np.asarray(metric_mask, dtype=bool)
            if metric_mask is not None
            else brain_mask
        )

        _heal_metric_mask_edges(metric_mask, signal_mask, raw_metric_data)

        # Apply masking
        metric_data = np.where(metric_mask, raw_metric_data, np.float32(np.nan))

        metric_data = np.asarray(metric_data, dtype=np.float32)

        # Native-space debug artifact to inspect last-slice coverage when needed
        native_dbg = os.path.join(
            diffusion_dir, f"{metric_name}_map_native_debug.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(metric_data.astype(np.float32), img.affine, img.header),
            native_dbg,
        )
        print(
            f"[dbg] {metric_name}: last-slice raw finite = {int(np.isfinite(raw_metric_data[..., -1]).sum())} "
            f"| masked finite = {int(np.isfinite(metric_data[..., -1]).sum())}"
        )

        metric_img = nib.Nifti1Image(metric_data, img.affine, img.header)
        metric_img.set_data_dtype(np.float32)
        metric_img = _maybe_resample_to_dce(
            metric_img,
            dce_resample_target,
            f"{metric_name.upper()} map",
        )

        raw_metric_img = nib.Nifti1Image(
            np.asarray(raw_metric_data, dtype=np.float32), img.affine, img.header
        )
        raw_metric_img.set_data_dtype(np.float32)
        raw_metric_img = _maybe_resample_to_dce(
            raw_metric_img,
            dce_resample_target,
            f"{metric_name.upper()} raw diffusion map",
        )
        map_path = os.path.join(diffusion_dir, f"{metric_name}_map.nii.gz")
        nib.save(metric_img, map_path)
        print(f"[!] Saved {metric_name.upper()} map to {map_path}")

        # Backwards compatibility for legacy FA outputs
        if metric_name == "fa":
            legacy_path = os.path.join(analysis_directory, "FA_map.nii.gz")
            nib.save(metric_img, legacy_path)

        if payload.get("compute_stats", True):
            stats_kwargs = {
                "mask_selection": payload.get("stats_mask_selection"),
                "tissue_groups": payload.get("tissue_groups"),
                "union_label": payload.get("stats_union_label", "brain"),
                "union_tissues": payload.get("stats_union_tissues"),
            }
            stats_summary[metric_name] = {
                "units": payload["units"],
                **_compute_statistics_with_options(
                    metric_name, metric_data, masks, **stats_kwargs
                ),
            }
        else:
            stats_summary[metric_name] = {"units": payload["units"]}

        _plot_metric_histogram(metric_name, metric_data, masks, image_directory)

        if payload.get("include_parcels", True):
            restrict = payload.get("restrict_parcels_to_wm", False)
            atlas_img = None
            parcels: Optional[Dict[str, Dict[str, float]]] = None

            if (
                atlas_dce_data is not None
                and dce_resample_target is not None
                and metric_img.shape[:3] == atlas_dce_data.shape
            ):
                metric_resampled = metric_img.get_fdata(dtype=np.float32)
                raw_metric_resampled = raw_metric_img.get_fdata(dtype=np.float32)
                parcel_map, dce_parcels = _parcel_means_dce(
                    metric_resampled,
                    atlas_dce_data,
                    label_lookup,
                    restrict_to_wm=restrict,
                    fallback_data=raw_metric_resampled,
                )
                if dce_parcels:
                    header = (
                        atlas_dce_img.header.copy()
                        if atlas_dce_img is not None
                        else None
                    )
                    affine = (
                        atlas_dce_img.affine
                        if atlas_dce_img is not None
                        else metric_img.affine
                    )
                    atlas_img = nib.Nifti1Image(
                        parcel_map.astype(np.float32), affine, header
                    )
                    atlas_img.set_data_dtype(np.float32)
                    parcels = dce_parcels

            if parcels is None and parcel_metadata:
                parcel_map, parcels = _parcel_means(
                    metric_data,
                    parcel_metadata,
                    restrict_to_wm=restrict,
                    fallback_data=raw_metric_data,
                )
                if parcels:
                    atlas_img = nib.Nifti1Image(
                        parcel_map.astype(np.float32), img.affine, img.header
                    )
                    atlas_img.set_data_dtype(np.float32)
                    atlas_img = _maybe_resample_to_dce(
                        atlas_img,
                        dce_resample_target,
                        f"{metric_name.upper()} atlas map",
                    )

            if parcels:
                atlas_path = os.path.join(
                    diffusion_dir, f"{metric_name}_map_atlas.nii.gz"
                )
                nib.save(atlas_img, atlas_path)
                print(f"[!] Saved {metric_name.upper()} atlas map to {atlas_path}")
                atlas_summary[metric_name] = {
                    "units": payload["units"],
                    "parcel_selection": payload.get("parcel_selection", "all"),
                    "parcels": parcels,
                }

    stats_path = os.path.join(diffusion_dir, "diffusion_values_median_total.json")
    with open(stats_path, "w") as fp:
        json.dump(stats_summary, fp, indent=4)
    print(f"[!] Wrote diffusion statistics to {stats_path}")

    if atlas_summary:
        atlas_stats_path = os.path.join(diffusion_dir, "diffusion_values_atlas.json")
        with open(atlas_stats_path, "w") as fp:
            json.dump(atlas_summary, fp, indent=4)
        print(f"[!] Wrote diffusion atlas statistics to {atlas_stats_path}")

    brain_union_mask = brain_mask if brain_mask is not None else _brain_union_mask(masks)
    if brain_union_mask is not None:
        valid = np.asarray(brain_union_mask, dtype=bool) & signal_mask & np.isfinite(fa)
        mean_fa = float(np.mean(fa[valid], dtype=np.float32)) if np.any(valid) else float("nan")
    else:
        valid = signal_mask & np.isfinite(fa)
        mean_fa = float(np.mean(fa[valid], dtype=np.float32)) if np.any(valid) else float("nan")
    wm_union_mask = _union_of_tissues(masks, _WM_TISSUES)
    with open(os.path.join(diffusion_dir, "fa_mean.txt"), "w") as f:
        f.write(f"{mean_fa}\n")
    print(f"[!] Mean FA: {mean_fa:.4f}")

    if "white_matter" in masks:
        wm_mask = np.asarray(masks["white_matter"][0], dtype=bool)

        wm_valid = wm_mask & signal_mask & np.isfinite(fa)
        wm_n = int(wm_valid.sum())

        if wm_n > 0:
            mean_fa_wm = float(np.mean(fa[wm_valid], dtype=np.float32))
            with open(os.path.join(diffusion_dir, "fa_mean_wm.txt"), "w") as f:
                f.write(f"{mean_fa_wm}\n")

            print(f"[!] Mean WM FA (WM∩signal): {mean_fa_wm:.4f} (n={wm_n})")
        else:
            print("[!] WM mask present but no valid WM voxels after signal/finite filtering")
            print(f"[!] Mean WM FA: {mean_fa_wm:.4f}")
            bad = wm_mask & (~signal_mask)
            print(f"[dbg] WM outside signal: {int(bad.sum())} / {int(wm_mask.sum())}")

            fa_wm = np.where(wm_mask & signal_mask & np.isfinite(fa), fa, np.nan)
            fa_wm_img = nib.Nifti1Image(
                fa_wm.astype(np.float32), img.affine, img.header
            )
            fa_wm_img.set_data_dtype(np.float32)
            fa_wm_img = _maybe_resample_to_dce(
                fa_wm_img, dce_resample_target, "WM FA map"
            )
            wm_out_path = os.path.join(diffusion_dir, "fa_wm_map.nii.gz")
            nib.save(fa_wm_img, wm_out_path)
            legacy_wm_path = os.path.join(analysis_directory, "FA_WM_map.nii.gz")
            nib.save(fa_wm_img, legacy_wm_path)
            print(f"[!] Saved WM FA map to {wm_out_path}")
