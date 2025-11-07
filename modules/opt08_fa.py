import json
import os
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

try:  # Matplotlib is optional in some deployment setups.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is always available in tests
    plt = None

try:
    from nibabel.processing import resample_from_to
except ImportError:  # pragma: no cover - nibabel ships the helper in supported envs
    resample_from_to = None


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


def _load_mask(mask_path: str, reference_img: nib.Nifti1Image) -> Optional[np.ndarray]:
    if mask_path is None:
        return None

    mask_img = nib.load(mask_path)
    if mask_img.shape != reference_img.shape or not np.allclose(mask_img.affine, reference_img.affine):
        if resample_from_to is None:
            print(f"[!] Cannot resample {mask_path} – nibabel.processing unavailable")
            return None
        try:
            mask_img = resample_from_to(mask_img, (reference_img.shape, reference_img.affine), order=0)
            print(f"[!] Resampled mask {os.path.basename(mask_path)} to diffusion grid")
        except Exception as exc:
            print(f"[!] Failed to resample mask {mask_path}: {exc}")
            return None
    return mask_img.get_fdata() > 0.5


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
    return masks


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


def find_dwi_files(nifti_directory):
    """Locate a DWI NIfTI file and its corresponding ``.bval`` and ``.bvec`` files.

    The conversion step may create ``.nii`` *or* ``.nii.gz`` files.  This
    function therefore checks for both extensions and strips them correctly when
    forming the paths to the gradient files.
    """

    for root, _, files in os.walk(nifti_directory):
        for file in files:
            if re.search(r"dwi", file, re.IGNORECASE) and (
                file.endswith(".nii") or file.endswith(".nii.gz")
            ):
                # Handle both .nii and .nii.gz extensions when deriving the base
                base = os.path.splitext(os.path.join(root, file))[0]
                if file.endswith(".nii.gz"):
                    base = os.path.splitext(base)[0]

                bval = base + ".bval"
                bvec = base + ".bvec"

                if os.path.exists(bval) and os.path.exists(bvec):
                    return os.path.join(root, file), bval, bvec

    return None, None, None


def _brain_union_mask(masks: Dict[str, Tuple[np.ndarray, Dict[str, str]]]) -> Optional[np.ndarray]:
    union = None
    for mask, _ in masks.values():
        union = mask if union is None else (union | mask)
    return union


def _compute_statistics(
    metric_name: str,
    metric_data: np.ndarray,
    masks: Dict[str, Tuple[np.ndarray, Dict[str, str]]],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for tissue, (mask, info) in masks.items():
        voxels = metric_data[mask]
        voxels = voxels[~np.isnan(voxels)]
        key = f"{info['json_label']}_median_total"
        stats[key] = {
            "median": float(np.nanmedian(voxels)) if voxels.size else float("nan"),
            "mean": float(np.nanmean(voxels)) if voxels.size else float("nan"),
            "voxel_count": int(mask.sum()),
        }

    union_mask = _brain_union_mask(masks)
    if union_mask is not None:
        voxels = metric_data[union_mask]
        voxels = voxels[~np.isnan(voxels)]
        stats["brain_median_total"] = {
            "median": float(np.nanmedian(voxels)) if voxels.size else float("nan"),
            "mean": float(np.nanmean(voxels)) if voxels.size else float("nan"),
            "voxel_count": int(np.sum(union_mask)),
        }
    return stats


def compute_fa(nifti_directory, analysis_directory, image_directory=None):
    dwi_path, bval_path, bvec_path = find_dwi_files(nifti_directory)
    if dwi_path is None:
        print("[!] No DWI data found; skipping FA computation")
        return

    print(f"[!] Computing FA from {os.path.basename(dwi_path)}")
    img = nib.load(dwi_path)
    data = img.get_fdata()

    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    gtab = gradient_table(bvals, bvecs)
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
        "fa": {"data": fa, "units": "unitless"},
        "md": {"data": md, "units": "mm^2/s"},
        "ad": {"data": ad, "units": "mm^2/s"},
        "rd": {"data": rd, "units": "mm^2/s"},
        "mo": {"data": mode, "units": "unitless"},
        "tensor_residual": {"data": residual, "units": "signal rms"},
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

    for metric_name, payload in metrics.items():
        metric_data = payload["data"]
        metric_img = nib.Nifti1Image(metric_data, img.affine, img.header)
        metric_img.set_data_dtype(np.float32)
        map_path = os.path.join(diffusion_dir, f"{metric_name}_map.nii.gz")
        nib.save(metric_img, map_path)
        print(f"[!] Saved {metric_name.upper()} map to {map_path}")

        # Backwards compatibility for legacy FA outputs
        if metric_name == "fa":
            legacy_path = os.path.join(analysis_directory, "FA_map.nii.gz")
            nib.save(metric_img, legacy_path)

        stats_summary[metric_name] = {
            "units": payload["units"],
            **_compute_statistics(metric_name, metric_data, masks),
        }

        _plot_metric_histogram(metric_name, metric_data, masks, image_directory)

    stats_path = os.path.join(diffusion_dir, "diffusion_values_median_total.json")
    with open(stats_path, "w") as fp:
        json.dump(stats_summary, fp, indent=4)
    print(f"[!] Wrote diffusion statistics to {stats_path}")

    mean_fa = float(np.nanmean(fa))
    with open(os.path.join(diffusion_dir, "fa_mean.txt"), "w") as f:
        f.write(f"{mean_fa}\n")
    print(f"[!] Mean FA: {mean_fa:.4f}")

    if "white_matter" in masks:
        wm_mask = masks["white_matter"][0]
        wm_values = fa[wm_mask]
        if wm_values.size:
            mean_fa_wm = float(np.nanmean(wm_values))
            with open(os.path.join(diffusion_dir, "fa_mean_wm.txt"), "w") as f:
                f.write(f"{mean_fa_wm}\n")
            print(f"[!] Mean WM FA: {mean_fa_wm:.4f}")

            fa_wm = fa * wm_mask
            fa_wm_img = nib.Nifti1Image(fa_wm.astype(np.float32), img.affine, img.header)
            wm_out_path = os.path.join(diffusion_dir, "fa_wm_map.nii.gz")
            nib.save(fa_wm_img, wm_out_path)
            legacy_wm_path = os.path.join(analysis_directory, "FA_WM_map.nii.gz")
            nib.save(fa_wm_img, legacy_wm_path)
            print(f"[!] Saved WM FA map to {wm_out_path}")
