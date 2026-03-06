"""
Manual tissue ROI — interactive tissue segmentation alternative.

When ``--tissue-roi manual`` is selected, the user draws ROIs directly on
DCE axial slices instead of running brain segmentation (FastSurfer / SynthSeg).
The hand-drawn ROIs replace the automatic tissue masks and feed into the
same downstream CTC + PK modelling pipeline.

CLI flow (matplotlib):
    1.  Load the DCE first frame (or T1 in DCE space if available).
    2.  Present the ``ROISelector_tissue`` GUI for interactive drawing.
    3.  User draws one or more ROIs, labels each as WM / GM / mixed.
    4.  Voxels inside each ROI become binary masks in DCE voxel space.
    5.  Feed masks into ``compute_and_plot_ctcs_median``.

Web flow:
    The web backend saves tissue ROI voxels under
    ``Analysis/Tissue ROI Data/<type>/ROI_voxels_slice_<N>.npy``
    and sets ``P_BRAIN_TISSUE_ROI_METHOD=manual`` before re-running the
    ``tissue_ctc`` / ``modelling`` stages.
"""

from __future__ import annotations

import os
import sys
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

import utils.settings as settings

logger = logging.getLogger(__name__)

turbo_mode = True  # Suppress interactive plots when True

# ── Tissue type helpers ──────────────────────────────────────────────

TISSUE_LABELS = {
    "wm":  "White Matter",
    "gm":  "Grey Matter",
    "mixed": "Mixed",
    "lesion": "Lesion",
}


def _is_headless() -> bool:
    """Return True when running without a display (e.g. inside the web backend)."""
    if os.environ.get("PBRAIN_HEADLESS", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if os.environ.get("MPLBACKEND", "").strip().lower() == "agg":
        return True
    # On macOS/Linux, check DISPLAY (X11) or WAYLAND_DISPLAY.
    if sys.platform != "win32":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            # macOS doesn't always set DISPLAY; check for native backend.
            if sys.platform != "darwin":
                return True
    return False


def _load_saved_tissue_rois(
    analysis_directory: str,
) -> Dict[str, np.ndarray]:
    """Load tissue ROI masks saved by the web UI.

    Returns a dict mapping tissue-type keys (``wm``, ``gm``, ``mixed``)
    to per-slice voxel dicts.  Returns an empty dict when no saved ROI
    data is found.

    The web backend writes via the generic ROI save endpoint:
        Analysis/ROI Data/Tissue/<subtype>/ROI_voxels_slice_<N>.npy
    where ``<subtype>`` is "White Matter", "Grey Matter", or "Mixed".
    Each file contains an (M, 2) int array of ``[row, col]`` voxel
    coordinates in *p-brain's rotated* in-plane frame.
    """
    import re

    # The web backend uses "ROI Data/Tissue/<subtype>".
    root = os.path.join(analysis_directory, "ROI Data", "Tissue")
    if not os.path.isdir(root):
        # Fallback: check legacy "Tissue ROI Data" path.
        root = os.path.join(analysis_directory, "Tissue ROI Data")
        if not os.path.isdir(root):
            return {}

    result: Dict[str, Dict[int, np.ndarray]] = {}

    for tissue_type in os.listdir(root):
        tissue_dir = os.path.join(root, tissue_type)
        if not os.path.isdir(tissue_dir):
            continue
        key = tissue_type.lower().replace(" ", "_")
        # Normalise common synonyms.
        if key in {"white_matter", "white matter", "wm"}:
            key = "wm"
        elif key in {"grey_matter", "gray_matter", "grey matter", "gray matter", "gm"}:
            key = "gm"
        elif key in {"mixed", "mixed_matter", "mixed matter"}:
            key = "mixed"
        elif key in {"lesion", "lesions"}:
            key = "lesion"

        for fname in os.listdir(tissue_dir):
            m = re.search(r"ROI_voxels_slice_(\d+)\.npy$", fname)
            if not m:
                continue
            slice_idx = int(m.group(1)) - 1  # on-disk is 1-based
            voxels = np.load(os.path.join(tissue_dir, fname))
            if voxels.size == 0:
                continue
            result.setdefault(key, {})[slice_idx] = voxels

    return result


def _build_masks_from_drawn_rois(
    selected_voxels: Dict[int, List[np.ndarray]],
    shape_3d: Tuple[int, int, int],
) -> np.ndarray:
    """Convert ``ROISelector_tissue`` output into a 3-D boolean mask.

    ``selected_voxels`` is ``{slice_index: [array_of_(row,col), ...]}``.
    Coordinates are in raw NIfTI voxel space (same frame as the DCE 4-D
    array) because the selector operates on un-rotated data.

    Returns a boolean mask of shape ``shape_3d``.
    """
    mask = np.zeros(shape_3d, dtype=bool)
    for slice_idx, roi_list in selected_voxels.items():
        if slice_idx < 0 or slice_idx >= shape_3d[2]:
            continue
        for voxels in roi_list:
            coords = np.asarray(voxels)
            for (r, c) in coords:
                if 0 <= r < shape_3d[0] and 0 <= c < shape_3d[1]:
                    mask[r, c, slice_idx] = True
    return mask


def _build_masks_from_saved_rois(
    saved_slices: Dict[int, np.ndarray],
    shape_3d: Tuple[int, int, int],
) -> np.ndarray:
    """Build a 3-D boolean mask from web-saved ROI voxel files.

    ``saved_slices`` maps 0-based slice index → (M, 2) int array of
    ``[rot_r, rot_c]`` pairs in **p-brain's rotated** in-plane frame
    (the web backend applies ``np.rot90(k=-1)`` before saving).

    This function inverse-rotates back to native NIfTI voxel space so
    the mask aligns with the raw ``data_4d`` array used downstream.

    The forward transform (native → rotated) is:
        rot_r = native_col, rot_c = X - 1 - native_row

    The inverse is:
        native_row = X - 1 - rot_c, native_col = rot_r

    where X = ``shape_3d[0]`` (the first spatial dim of the NIfTI).
    """
    mask = np.zeros(shape_3d, dtype=bool)
    x_dim = shape_3d[0]
    for slice_idx, voxels in saved_slices.items():
        if slice_idx < 0 or slice_idx >= shape_3d[2]:
            continue
        for (rot_r, rot_c) in np.asarray(voxels):
            native_row = x_dim - 1 - int(rot_c)
            native_col = int(rot_r)
            if 0 <= native_row < shape_3d[0] and 0 <= native_col < shape_3d[1]:
                mask[native_row, native_col, slice_idx] = True
    return mask


# ── Interactive CLI entry point (matplotlib) ─────────────────────────

def _interactive_draw_tissue_rois(
    dce_path: str,
    t2_path: Optional[str] = None,
    overlay_path: Optional[str] = None,
) -> Dict[str, Dict[int, List[np.ndarray]]]:
    """Show the matplotlib ROI selector and collect tissue ROIs.

    Returns ``{"wm": selected_voxels_dict, "gm": ...}`` where each value
    is the ``ROISelector_tissue.roi_slices`` dict (slice_idx → [voxels]).

    The display uses **raw NIfTI voxel space** (no rotation) so the drawn
    voxel coordinates map directly to the DCE 4-D array indices expected by
    ``compute_and_plot_ctcs_median``.

    ``overlay_path`` selects the background volume for interactive drawing.
    When ``None`` (or the volume cannot be loaded / shape-matched), the
    first frame of the DCE 4-D is used.
    """
    from termcolor import colored
    from .opt04_tissue_function import ROISelector_tissue
    import matplotlib.pyplot as plt

    # Load DCE first frame as the default display volume.
    dce_4d = nib.load(dce_path).get_fdata()
    dce_shape = dce_4d.shape[:3]
    display_3d = dce_4d[:, :, :, 0] if dce_4d.ndim == 4 else dce_4d
    overlay_label = "DCE (frame 0)"

    # Try to use a structural overlay instead, if requested.
    if overlay_path and os.path.isfile(overlay_path):
        try:
            ov_img = nib.load(overlay_path)
            ov_data = ov_img.get_fdata()
            if ov_data.ndim == 4:
                ov_data = ov_data[:, :, :, 0]
            if ov_data.shape == dce_shape:
                display_3d = ov_data
                overlay_label = os.path.basename(overlay_path)
            else:
                # Attempt resample to DCE space.
                try:
                    from nibabel.processing import resample_from_to
                    dce_img = nib.load(dce_path)
                    ref_shape = dce_img.shape[:3]
                    ref_affine = dce_img.affine
                    resampled = resample_from_to(
                        nib.Nifti1Image(ov_data, ov_img.affine),
                        (ref_shape, ref_affine),
                    )
                    display_3d = np.asarray(resampled.dataobj, dtype=np.float32)
                    overlay_label = f"{os.path.basename(overlay_path)} (resampled)"
                except Exception as exc:
                    print(f"[manual-tissue-roi] Could not resample overlay to DCE space: {exc}")
                    print("[manual-tissue-roi] Falling back to DCE first frame.")
        except Exception as exc:
            print(f"[manual-tissue-roi] Could not load overlay {overlay_path}: {exc}")
            print("[manual-tissue-roi] Falling back to DCE first frame.")

    print(f"[manual-tissue-roi] Overlay: {overlay_label}")

    collected: Dict[str, Dict[int, List[np.ndarray]]] = {}

    for tissue_key, tissue_name in TISSUE_LABELS.items():
        print()
        print(colored("=" * 60, "white"))
        print(colored(f"  Draw ROI for: {tissue_name}", "cyan"))
        print(colored("=" * 60, "white"))
        print("1. Left " + colored("click", "cyan") + " to add ROI points.")
        print("2. Press " + colored("shift", "cyan") + " to close the polygon.")
        print("3. Press " + colored("enter", "cyan") + " to save the current ROI.")
        print("4. Use " + colored("left/right", "cyan") + " arrows to change slices.")
        print("5. Press " + colored("z", "cyan") + " to zoom.")
        print("6. Press " + colored("Esc", "red") + " to finish this tissue type.")
        print()

        skip = input(
            f"[?] Draw ROI for {tissue_name}? (y/n, default y): "
        ).strip().lower()
        if skip == "n":
            continue

        selector = ROISelector_tissue(display_3d)
        # The module-level turbo_mode in opt04 suppresses plt.show().
        # We must call it ourselves so the window blocks until user presses Esc.
        plt.show()
        voxels = selector.get_selected_voxels()

        n_rois = sum(len(v) for v in voxels.values())
        if n_rois == 0:
            print(f"[!] No ROIs drawn for {tissue_name}, skipping.")
            continue

        print(f"[✓] {n_rois} ROI(s) saved for {tissue_name}.")
        collected[tissue_key] = dict(voxels)

    return collected


# ── Core: manual tissue analysis ─────────────────────────────────────

def _run_manual_tissue_analysis(
    analysis_directory: str,
    nifti_directory: str,
    image_directory: str,
    filenames: tuple,
    parameters: tuple,
) -> None:
    """Run tissue CTC + PK modelling using manual ROI masks.

    This function:
    1.  Obtains tissue masks (interactive CLI draw or web-saved files).
    2.  Loads DCE, T1/M0, time points.
    3.  Passes masks to ``compute_and_plot_ctcs_median``.
    """
    import pickle

    from utils.loading import (
        load_dce_4d,
        resolve_flip_angle_deg,
        resolve_dce_time_step_s,
        build_time_points_s,
    )

    from .AI_tissue_functions import compute_and_plot_ctcs_median

    (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    ) = filenames

    t2_path = os.path.join(nifti_directory, axial_t2_2D_filename) if axial_t2_2D_filename else None
    dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else None

    if not dce_path or not os.path.exists(dce_path):
        raise RuntimeError("Missing DCE NIfTI.")

    # ── Resolve overlay for interactive drawing ────
    # The user may request a specific structural background via --overlay.
    overlay_pref = getattr(settings, "MANUAL_ROI_OVERLAY", "auto").strip().lower()

    # Build a lookup of available structural volumes.
    _struct_candidates: Dict[str, Optional[str]] = {}
    for _label, _fname in (
        ("t1", t1_3D_filename),
        ("t2", t2_3D_filename if t2_3D_filename else axial_t2_2D_filename),
        ("flair", flair_3D_filename),
    ):
        if _fname:
            _fpath = os.path.join(nifti_directory, _fname)
            _struct_candidates[_label] = _fpath if os.path.isfile(_fpath) else None
        else:
            _struct_candidates[_label] = None

    overlay_path: Optional[str] = None
    if overlay_pref == "dce":
        overlay_path = None  # will use DCE first frame
    elif overlay_pref in ("t1", "t2", "flair"):
        overlay_path = _struct_candidates.get(overlay_pref)
        if not overlay_path:
            print(f"[manual-tissue-roi] Requested overlay '{overlay_pref}' not found; "
                  f"falling back to auto.")
            overlay_pref = "auto"
    if overlay_pref == "auto":
        # Prefer T1 > T2 > FLAIR > DCE.
        for _k in ("t1", "t2", "flair"):
            if _struct_candidates.get(_k):
                overlay_path = _struct_candidates[_k]
                print(f"[manual-tissue-roi] Auto-selected overlay: {_k} "
                      f"({os.path.basename(overlay_path)})")
                break
        if overlay_path is None:
            print("[manual-tissue-roi] No structural volume found; "
                  "using DCE first frame as overlay.")

    flip_angle_deg = resolve_flip_angle_deg(dce_path, default=None)

    # ── Load DCE 4D ────
    ref_img, data_4d = load_dce_4d(dce_path, prefer_complex_mag=True, dtype=np.float32)
    data_4d = np.asarray(data_4d)
    ref_affine = ref_img.affine
    ref_header = ref_img.header.copy()
    dce_shape = data_4d.shape[:3]

    # ── Load T1/M0 ────
    T1_matrix = None
    M0_matrix = None
    t1_pkl = os.path.join(analysis_directory, "Fitting", "voxel_T1_matrix.pkl")
    m0_pkl = os.path.join(analysis_directory, "Fitting", "voxel_M0_matrix.pkl")
    if os.path.isfile(t1_pkl) and os.path.isfile(m0_pkl):
        with open(t1_pkl, "rb") as f:
            T1_matrix = pickle.load(f)
        with open(m0_pkl, "rb") as f:
            M0_matrix = pickle.load(f)

    # ── Time points ────
    time_path = os.path.join(analysis_directory, "Fitting", "time_points_s.npy")
    time_points_s = None
    if os.path.isfile(time_path):
        try:
            time_points_s = np.load(time_path)
        except Exception:
            pass
    if time_points_s is None:
        dt_s = resolve_dce_time_step_s(dce_path, default=None)
        time_points_s = build_time_points_s(data_4d.shape[-1], dt_s)
        os.makedirs(os.path.dirname(time_path), exist_ok=True)
        np.save(time_path, time_points_s)

    # ── Obtain masks ────
    # Priority: saved web ROI files → interactive CLI drawing → exit 42 for web.
    saved = _load_saved_tissue_rois(analysis_directory)

    wm_mask_dce = None
    gm_mask_dce = None
    lesion_mask_dce = None

    if saved:
        print("[manual-tissue-roi] Using saved tissue ROI voxels.")
        if "wm" in saved:
            wm_mask_dce = _build_masks_from_saved_rois(saved["wm"], dce_shape)
        if "gm" in saved:
            gm_mask_dce = _build_masks_from_saved_rois(saved["gm"], dce_shape)
        if "mixed" in saved:
            # For mixed: merge into both GM and WM so downstream gets curves
            # for both tissue types from the same region.
            mixed_mask = _build_masks_from_saved_rois(saved["mixed"], dce_shape)
            if wm_mask_dce is None:
                wm_mask_dce = mixed_mask
            else:
                wm_mask_dce = wm_mask_dce | mixed_mask
            if gm_mask_dce is None:
                gm_mask_dce = mixed_mask
            else:
                gm_mask_dce = gm_mask_dce | mixed_mask
        if "lesion" in saved:
            lesion_mask_dce = _build_masks_from_saved_rois(saved["lesion"], dce_shape)
    elif _is_headless():
        # Running inside the web backend (no display) — signal the UI
        # to show the tissue ROI drawing panel by exiting with code 42.
        print("[manual-tissue-roi] No saved tissue ROIs found.  "
              "Exiting with code 42 so the web UI can present the drawing panel.")
        sys.exit(42)
    else:
        # Interactive CLI drawing.
        drawn = _interactive_draw_tissue_rois(dce_path, t2_path=t2_path,
                                              overlay_path=overlay_path)
        if not drawn:
            print("[manual-tissue-roi] No ROIs drawn. Falling back to voxelwise-only.")

        if "wm" in drawn:
            wm_mask_dce = _build_masks_from_drawn_rois(drawn["wm"], dce_shape)
        if "gm" in drawn:
            gm_mask_dce = _build_masks_from_drawn_rois(drawn["gm"], dce_shape)
        if "mixed" in drawn:
            mixed_mask = _build_masks_from_drawn_rois(drawn["mixed"], dce_shape)
            if wm_mask_dce is None:
                wm_mask_dce = mixed_mask
            else:
                wm_mask_dce = wm_mask_dce | mixed_mask
            if gm_mask_dce is None:
                gm_mask_dce = mixed_mask
            else:
                gm_mask_dce = gm_mask_dce | mixed_mask
        if "lesion" in drawn:
            lesion_mask_dce = _build_masks_from_drawn_rois(drawn["lesion"], dce_shape)

    # ── Decide mode ────
    has_any_mask = (
        (wm_mask_dce is not None and np.any(wm_mask_dce))
        or (gm_mask_dce is not None and np.any(gm_mask_dce))
        or (lesion_mask_dce is not None and np.any(lesion_mask_dce))
    )
    voxelwise_only = not has_any_mask

    if voxelwise_only:
        print("[manual-tissue-roi] No tissue masks available — running voxelwise-only.")

    # For manual ROI we treat the drawn WM as "wm" and drawn GM as
    # "cortical_gm".  We pass ``subcortical_gm`` and extra masks as None
    # (the function handles None gracefully).
    model = (settings.KINETIC_MODEL or "patlak").strip().lower()
    compute_ki = model in {"patlak", "both"}
    compute_cbf = model in {"tikhonov", "both"}

    _, _, _, boundary, _, _, _ = parameters

    # Use the T2 volume for visualisation if available.
    t2_img = None
    if t2_path and os.path.isfile(t2_path):
        t2_img = nib.load(t2_path).get_fdata()

    # Also use WM mask for T2 space overlay (same space since user drew on
    # DCE-resolution images, and T2 is used only for plotting).
    wm_mask_t2 = wm_mask_dce
    cortical_gm_mask_t2 = gm_mask_dce
    # Lesion ROI is passed through the subcortical_gm slot (unused in manual
    # mode) so it gets its own per-slice CTC curve in the downstream analysis.
    subcortical_gm_mask_t2 = lesion_mask_dce

    compute_and_plot_ctcs_median(
        data_4d,
        t2_img,
        wm_mask_t2,
        cortical_gm_mask_t2,
        subcortical_gm_mask_t2,
        wm_mask_dce,
        gm_mask_dce,
        lesion_mask_dce,  # lesion ROI via subcortical_gm_mask_dce slot
        T1_matrix,
        M0_matrix,
        analysis_directory,
        time_points_s,
        image_directory,
        dce_path=dce_path,
        ref_affine=ref_affine,
        ref_header=ref_header,
        boundary=boundary,
        compute_per_voxel_Ki=compute_ki,
        compute_per_voxel_CBF=compute_cbf,
        flip_angle_deg=flip_angle_deg,
        voxelwise_only=voxelwise_only,
    )

    print("[manual-tissue-roi] Tissue analysis completed (manual ROI).")


# ── Public entry points ──────────────────────────────────────────────

def tissue_function_manual(
    analysis_directory: str,
    nifti_directory: str,
    image_directory: str,
    filenames: tuple,
    parameters: tuple,
    *,
    compute_diffusion: bool = False,
) -> None:
    """Drop-in replacement for ``tissue_function_AI`` using manual ROIs.

    Call signature mirrors ``tissue_function_AI`` so ``main.py`` can
    switch transparently based on ``TISSUE_ROI_METHOD``.
    """
    _run_manual_tissue_analysis(
        analysis_directory,
        nifti_directory,
        image_directory,
        filenames,
        parameters,
    )

    if compute_diffusion:
        diffusion_filename = filenames[-2] if filenames else None
        if diffusion_filename:
            try:
                from . import opt08_fa
            except ImportError as exc:
                print(f"[diffusion] Unable to import diffusion workflow: {exc}")
            else:
                dce_filename = filenames[-1] if filenames else None
                dce_path = (
                    os.path.join(nifti_directory, dce_filename) if dce_filename else None
                )
                try:
                    print("[diffusion] Computing diffusion metrics")
                    opt08_fa.compute_fa(
                        nifti_directory,
                        analysis_directory,
                        image_directory,
                        diffusion_filename=diffusion_filename,
                        dce_path=dce_path,
                    )
                except Exception as exc:
                    print(f"[diffusion] Failed: {exc}")
        else:
            print("[diffusion] No diffusion filename; skipping.")

    # Montage rendering.
    dce_filename = filenames[-1] if filenames else None
    if dce_filename:
        dce_path = os.path.join(nifti_directory, dce_filename)
        if os.path.exists(dce_path):
            try:
                from utils.montage import generate_parametric_montages
                generate_parametric_montages(
                    analysis_directory, image_directory, dce_path,
                    segmentation_path=None,
                )
            except Exception as exc:
                print(f"[montage] Error: {exc}")
