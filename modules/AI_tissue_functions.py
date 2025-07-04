turbo_mode = False  # Set to True to suppress all plots
force_recreate_masks = True  # If True: recreate all masks regardless of existence
# If True, drop detection is performed on tissue CTCs and the ignored
# regions are excluded from the Patlak fit.  When False, every sample
# is used for the Patlak analysis.
correct_signal_jumps = False


# When True, a two-step FLIRT registration is used when aligning
# segmentation masks to DCE and T2 images.  When False (default), the
# previous one-step "-applyxfm -usesqform" approach is retained.
use_flirt_registration = False

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import multiprocessing
import shutil
import utils.settings as settings
from utils.fonts import *
from utils.loading import *
from utils.plotting import *
from .kinetic_models import two_compartment_fit
from skimage.transform import resize
import json
from scipy.ndimage import binary_dilation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import re

def patlak_total(C_t, C_a, t):
    """Optional drop correction then Patlak fit."""
    if C_t.size == 0:
        return (np.nan, np.nan, np.nan)          # Ki, λ, SD_Ki
    if correct_signal_jumps:
        _, bad, _ = mask_problematic(C_t)        # <- same length as C_t
    else:
        bad = None
    Ki, lam, SD, *_ = patlak_with_exclusions(C_t, C_a, t, bad_mask=bad)
    return Ki, lam, SD

def mask_problematic(ctc, *, tail_start: int = 100, thresh_factor: float = 0.5):
    """
    Replace “bad” (post-tail drop) samples in *ctc* with NaN and return
    (ctc_masked, bad_mask, drop_idxs).

    ── NEW: now **always** returns 3 values ──
    """
    ctc = np.asarray(ctc, dtype=float)

    # ── EARLY EXIT ────────────────────────────────────────────────
    if ctc.size <= tail_start + 1:
        # too short to analyse — nothing is “bad”
        return (
            ctc.astype(float),                       # ctc_masked
            np.zeros_like(ctc, dtype=bool),          # bad_mask
            np.array([], dtype=int)                  # drop_idxs
        )

    # identify the drop points
    drop_idxs, *_ = identify_drop_points(ctc, tail_start, thresh_factor)

    # boolean mask of the dropped samples
    bad_mask = np.zeros_like(ctc, dtype=bool)
    if drop_idxs.size:
        bad_mask[drop_idxs] = True

    # masked copy of the curve
    ctc_masked = ctc.copy()
    ctc_masked[bad_mask] = np.nan

    return ctc_masked, bad_mask, drop_idxs


# -------------- patlak_analysis.py (new version) --------------
def patlak_with_exclusions(C_t, C_a, t, bad_mask=None):
    """
    Patlak that *plots everything* but fits only the good samples.
    bad_mask == True  →  point is shown hollow and excluded from fit
    """
    n = min(len(C_t), len(C_a), len(t))
    C_t, C_a, t = C_t[:n], C_a[:n], t[:n]
    if bad_mask is not None:
        bad_mask = bad_mask[:n]

    C_t, C_a, t = map(np.asarray, (C_t, C_a, t))
    if bad_mask is None:
        bad_mask = np.zeros_like(C_t, dtype=bool)

    # --- Patlak co-ordinates (never introduce NaNs here) --------
    dt = np.diff(t)
    x = np.concatenate(([0], np.cumsum(C_a[:-1] * dt))) / C_a
    y = C_t / C_a

    # points that *could* be used
    finite   = np.isfinite(x) & np.isfinite(y) & (C_a != 0)
    # classic ⅓-to-⅔ window
    w        = (x >= x[finite].max() / 3) & (x <= x[finite].max())
    include  = finite & w & (~bad_mask)

    # bail-out if <2 points
    if include.sum() < 2:
        return np.nan, np.nan, np.nan, x, y, include

    xm, ym   = x[include].mean(), y[include].mean()
    Ki       = ((x[include]-xm)*(y[include]-ym)).sum() / ((x[include]-xm)**2).sum()
    lam      = ym - Ki*xm
    resid    = y[include] - (lam + Ki*x[include])
    SD_Ki    = np.sqrt((resid**2).sum() / ((x[include]-xm)**2).sum() / (include.sum()-2))

    return Ki*6000, lam*100, SD_Ki*6000, x, y, include


def identify_drop_points(signal, tail_start: int = 100, threshold_factor: float = 0.5):
    """
    Detect “drop” samples occurring at/after *tail_start* where the curve
    falls more than *threshold_factor*·σ below a fitted linear tail trend.

    Parameters
    ----------
    signal : 1-D array-like
        Concentration-time curve.
    tail_start : int, default 100
        Index that marks the beginning of the tail region.
    threshold_factor : float, default 0.5
        Multiplier for the residual standard deviation that sets the
        drop threshold.

    Returns
    -------
    drop_idxs : np.ndarray (int)
        Indices judged to be ‘bad’ (empty if none).
    trend : np.ndarray | None
        The fitted linear trend across the *entire* signal,
        or *None* when the curve is too short to fit.
    thresh : float | None
        The absolute drop threshold, or *None* when no fit is done.
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size

    # ── EARLY EXIT ────────────────────────────────────────────────
    # need ≥2 points after tail_start to fit a line
    if n <= tail_start + 1:
        return np.array([], dtype=int), None, None
    # also bail if everything is NaN
    if np.all(np.isnan(signal)):
        return np.array([], dtype=int), None, None

    # standard path
    x = np.arange(tail_start, n)
    y = signal[tail_start:]

    # handle NaNs in the tail region
    good_tail = ~np.isnan(y)
    if good_tail.sum() < 2:                     # not enough data to fit
        return np.array([], dtype=int), None, None

    # fit linear trend to *clean* tail samples
    m, b = np.polyfit(x[good_tail], y[good_tail], 1)
    trend = m * np.arange(n) + b

    # residuals & threshold
    resid = signal - trend
    mu, sigma = np.nanmean(resid), np.nanstd(resid)
    thresh = mu - threshold_factor * sigma

    # flag all drops beyond threshold, only in tail
    drop_idxs = np.where((resid < thresh) & (np.arange(n) >= tail_start))[0]
    return drop_idxs.astype(int), trend, thresh

# -- Helper functions for compute_Ki_from_atlas -----------------------------

# These globals are populated in ``_init_compute_Ki`` and used by
# ``_process_label``.  They allow child processes spawned by
# ``multiprocessing`` to access the large numpy arrays without needing to
# pickle and send them with every task.
_atlas_data = None
_data_4d = None
_T1_matrix = None
_M0_matrix = None
_time_points_s = None
_C_a_full = None
_compute_CTC = None
_find_baseline_point_advanced = None
_custom_shifter = None
_patlak_analysis_plotting = None


def _load_label_lookup(lut_path=None):
    """Return a dict mapping segmentation indices to region names."""
    if lut_path is None:
        fs_home = os.environ.get("FREESURFER_HOME")
        if fs_home:
            lut_path = os.path.join(fs_home, "FreeSurferColorLUT.txt")
    lookup = {}
    if lut_path and os.path.exists(lut_path):
        try:
            with open(lut_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = re.split(r"\s+", line)
                    if len(parts) >= 2 and parts[0].isdigit():
                        lookup[int(parts[0])] = parts[1]
        except Exception:
            pass
    return lookup


def _init_compute_Ki(atlas_data, data_4d, T1_matrix, M0_matrix, time_points_s,
                     C_a_full, compute_CTC, find_baseline_point_advanced,
                     custom_shifter, patlak_analysis_plotting):
    """Initialise global read-only data for compute_Ki_from_atlas workers."""

    global _atlas_data, _data_4d, _T1_matrix, _M0_matrix, _time_points_s
    global _C_a_full, _compute_CTC, _find_baseline_point_advanced
    global _custom_shifter, _patlak_analysis_plotting

    _atlas_data = atlas_data
    _data_4d = data_4d
    _T1_matrix = T1_matrix
    _M0_matrix = M0_matrix
    _time_points_s = time_points_s
    _C_a_full = C_a_full
    _compute_CTC = compute_CTC
    _find_baseline_point_advanced = find_baseline_point_advanced
    _custom_shifter = custom_shifter
    _patlak_analysis_plotting = patlak_analysis_plotting


def _process_label(lbl):
    """Worker for ``compute_Ki_from_atlas`` to process a single label."""

    mask = (_atlas_data == lbl)
    indices = np.argwhere(mask)
    if len(indices) < 1:
        return (lbl, np.nan, np.nan, np.nan, 0)

    curves_for_label = []
    for (x, y, z) in indices:
        voxel_time_course = _data_4d[x, y, z, :]
        if np.isnan(voxel_time_course).any():
            continue
        T1 = _T1_matrix[x, y, z]
        M0 = _M0_matrix[x, y, z]
        c_t_0 = _compute_CTC(voxel_time_course, T1, m0=M0)
        if np.isnan(c_t_0).any():
            continue
        baseline_point = _find_baseline_point_advanced(c_t_0)
        c_t = _custom_shifter(c_t_0, baseline_point)
        if np.isnan(c_t).any():
            continue
        if np.all(c_t == 0.0):
            continue
        curves_for_label.append(c_t)

    if len(curves_for_label) == 0:
        return (lbl, np.nan, np.nan, np.nan, 0)

    curves_for_label = np.array(curves_for_label)
    median_ct = np.median(curves_for_label, axis=0)

    min_len = min(len(median_ct), len(_C_a_full))
    C_t_label = median_ct[:min_len]
    C_a_label = _C_a_full[:min_len]
    t_label = _time_points_s[:min_len]

    Ki, lam, SD_Ki, _, _, _ = _patlak_analysis_plotting(
        C_t_label, C_a_label, t_label)
    return (lbl, Ki, SD_Ki, lam, len(indices))


def compute_Ki_from_atlas(
    atlas_path,
    data_4d,
    T1_matrix,
    M0_matrix,
    time_points_s,
    C_a_full,
    affine,
    output_directory,
    compute_CTC,
    find_baseline_point_advanced,
    custom_shifter,
    patlak_analysis_plotting
):

    # Load the atlas and find unique labels
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)

    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels != 0]  # exclude background label=0 if present

    label_lookup = _load_label_lookup()

    # Prepare empty 3D volumes for Ki, SD(Ki), and vp
    Ki_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    SD_Ki_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    vp_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)

    # Dictionary to keep numerical results per label for JSON output
    atlas_results = {}

    # Initialise worker globals for multiprocessing or direct execution
    _init_compute_Ki(
        atlas_data,
        data_4d,
        T1_matrix,
        M0_matrix,
        time_points_s,
        C_a_full,
        compute_CTC,
        find_baseline_point_advanced,
        custom_shifter,
        patlak_analysis_plotting,
    )

    if settings.MULTIPROCESSING:
        with multiprocessing.Pool(
            settings.NUMBER_OF_CORES,
            initializer=_init_compute_Ki,
            initargs=(
                atlas_data,
                data_4d,
                T1_matrix,
                M0_matrix,
                time_points_s,
                C_a_full,
                compute_CTC,
                find_baseline_point_advanced,
                custom_shifter,
                patlak_analysis_plotting,
            ),
        ) as pool:
            results = pool.map(_process_label, unique_labels)
    else:
        results = [_process_label(lbl) for lbl in unique_labels]

    for lbl, Ki, SD_Ki, lam, voxel_count in results:
        if np.isnan(Ki):
            continue
        mask = (atlas_data == lbl)
        Ki_map[mask] = Ki
        SD_Ki_map[mask] = SD_Ki
        vp_map[mask] = lam
        label_key = label_lookup.get(int(lbl), str(lbl))
        atlas_results[label_key] = {
            "Ki": float(Ki),
            "SD_Ki": float(SD_Ki),
            "vp": float(lam),
            "voxel_count": int(voxel_count)
        }

    # Save results as NIfTI
    os.makedirs(output_directory, exist_ok=True)

    Ki_nii = nib.Nifti1Image(Ki_map, affine)
    SD_Ki_nii = nib.Nifti1Image(SD_Ki_map, affine)
    vp_nii = nib.Nifti1Image(vp_map, affine)

    nib.save(Ki_nii, os.path.join(output_directory, 'Ki_map_atlas.nii.gz'))
    nib.save(SD_Ki_nii, os.path.join(output_directory, 'SD_Ki_map_atlas.nii.gz'))
    nib.save(vp_nii, os.path.join(output_directory, 'vp_map_atlas.nii.gz'))

    # Save numerical results to JSON
    json_path = os.path.join(output_directory, 'Ki_values_atlas.json')
    with open(json_path, 'w') as jf:
        json.dump(atlas_results, jf, indent=4)

    print("Done. Wrote:")
    print("  Ki_map_atlas.nii.gz")
    print("  SD_Ki_map_atlas.nii.gz")
    print("  vp_map_atlas.nii.gz")


def construct_convolution_matrix(C_a, delta_t):
    n = len(C_a)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, :i+1] = C_a[i::-1] * delta_t
    return A


from scipy.linalg import solve

def tikhonov_regularization(A, C_t, lambd):
    n = A.shape[1]
    L = np.eye(n)  # Using identity matrix for regularization
    ATA = A.T @ A
    LTL = L.T @ L
    regularized_matrix = ATA + lambd * LTL
    ATC_t = A.T @ C_t
    R = np.linalg.solve(regularized_matrix, ATC_t)
    return R

def plot_predictions_with_masks(image, wm_mask, cortical_gm_mask, subcortical_gm_mask, gm_brainstem_mask, gm_cerebellum_mask, wm_cerebellum_mask, wm_cc_mask, image_directory):
    n_slices = image.shape[2]
    n_cols = 5
    n_rows = (n_slices + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

    for i in range(n_slices):
        row = i // n_cols
        col = i % n_cols

        image_slice = np.rot90(image[:, :, i])
        wm_slice = np.rot90(wm_mask[:, :, i])
        cortical_gm_slice = np.rot90(cortical_gm_mask[:, :, i])
        subcortical_gm_slice = np.rot90(subcortical_gm_mask[:, :, i])
        gm_brainstem_slice = np.rot90(gm_brainstem_mask[:, :, i])
        gm_cerebellum_slice = np.rot90(gm_cerebellum_mask[:, :, i])
        wm_cerebellum_slice = np.rot90(wm_cerebellum_mask[:, :, i])
        wm_cc_slice = np.rot90(wm_cc_mask[:, :, i])

        color_overlay = np.zeros((*image_slice.shape, 3))
        color_overlay[:, :, 2][wm_slice == 1] = 1.0  # Blue channel for white matter

        # Assign bright red to cortical gray matter
        color_overlay[:, :, 0][cortical_gm_slice == 1] = 1.0  # Bright red

        # Assign dark red to subcortical gray matter
        color_overlay[:, :, 0][subcortical_gm_slice == 1] = 0.5  # Darker red

        # Assign orange to brainstem (red + green)
        color_overlay[:, :, 0][gm_brainstem_slice == 1] = 1.0  # Red channel
        color_overlay[:, :, 1][gm_brainstem_slice == 1] = 0.5  # Green channel

        # Assign yellow to cerebellum GM (red + green)
        color_overlay[:, :, 0][gm_cerebellum_slice == 1] = 1.0  # Red channel
        color_overlay[:, :, 1][gm_cerebellum_slice == 1] = 1.0  # Green channel

        # Assign cyan to cerebellum WM (green + blue)
        color_overlay[:, :, 1][wm_cerebellum_slice == 1] = 1.0  # Green channel
        color_overlay[:, :, 2][wm_cerebellum_slice == 1] = 1.0  # Blue channel

        # Assign magenta to corpus callosum WM (red + blue)
        color_overlay[:, :, 0][wm_cc_slice == 1] = 1.0  # Red channel
        color_overlay[:, :, 2][wm_cc_slice == 1] = 1.0  # Blue channel

        ax = axes[row, col]
        ax.imshow(image_slice, cmap='gray')
        ax.imshow(color_overlay, alpha=0.5)
        ax.set_title(f'Slice {i+1}')

        ax.grid(False)
        ax.axis("off")

    # Remove empty subplots
    for j in range(n_slices, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    os.makedirs(os.path.join(image_directory, 'AI', 'Segmentation'), exist_ok=True)
    plt.savefig(os.path.join(image_directory, 'AI', 'Segmentation', 'T2_WM_GM_masks.png'))
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay(3, fig)
        plt.show()
    else:
        plt.close(fig)

def segmentation(
    fastsurfer_path,
    seg_mgz_path,
    t1_path,
    output_dir,
    sid,
    apple_metal=True,
    rerun=False,
    method="fastsurfer",
):
    method = (method or "fastsurfer").lower()

    if method == "fastsurfer":
        # Check if FastSurfer is installed
        if not os.path.exists(fastsurfer_path):
            raise Exception(
                "FastSurfer not found, ensure correct installation and configuration of path."
            )

        # Run FastSurfer if the segmentation file doesn't exist or rerun is forced
        if rerun or not os.path.exists(seg_mgz_path):
            if os.path.exists(seg_mgz_path):
                print("Rerunning FastSurfer segmentation...")
            else:
                print("Segmentation file not found, running FastSurfer...")
            if apple_metal:
                command = (
                    f"export PYTORCH_ENABLE_MPS_FALLBACK=1 && "
                    f"{fastsurfer_path} --seg_only --device mps "
                    f"--t1 {t1_path} "
                    f"--sid {sid} "
                    f"--sd {output_dir}"
                )
            else:
                command = (
                    f"{fastsurfer_path} --seg_only "
                    f"--t1 {t1_path} "
                    f"--sid {sid} "
                    f"--sd {output_dir}"
                )
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(seg_mgz_path):
                print("FastSurfer segmentation failed, attempting run with vox_size 1 ...")
                if apple_metal:
                    command = (
                        f"export PYTORCH_ENABLE_MPS_FALLBACK=1 && "
                        f"{fastsurfer_path} --seg_only --device mps "
                        f"--t1 {t1_path} "
                        f"--vox_size 1 "
                        f"--sid {sid} "
                        f"--sd {output_dir} "
                        f"--no_cereb"
                    )
                else:
                    command = (
                        f"{fastsurfer_path} --seg_only "
                        f"--t1 {t1_path} "
                        f"--vox_size 1 "
                        f"--sid {sid} "
                        f"--sd {output_dir} "
                        f"--no_cereb"
                    )
                subprocess.run(command, shell=True)
                if not os.path.exists(seg_mgz_path):
                    raise RuntimeError("FastSurfer segmentation failed even with vox_size 1")
        else:
            print("Segmentation file already exists, skipping FastSurfer segmentation.")
    else:
        print(f"Segmentation method '{method}' selected. Skipping FastSurfer execution.")
        if not os.path.exists(seg_mgz_path):
            raise FileNotFoundError(
                f"Segmentation file not found: {seg_mgz_path}. Provide your own segmentation before running."
            )

    aseg_mgz_path = seg_mgz_path

    # Convert aseg.mgz to aseg.nii if needed
    aseg_nii_path = aseg_mgz_path.replace('.mgz', '.nii.gz')
    if not os.path.exists(aseg_nii_path):
        print(f"Converting {aseg_mgz_path} to {aseg_nii_path}...")
        subprocess.run(['mri_convert', aseg_mgz_path, aseg_nii_path])
    else:
        print(f"{aseg_nii_path} already exists, skipping conversion.")

    # Paths for the masks
    cortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'cortical_gm.nii.gz')
    subcortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'subcortical_gm.nii.gz')
    wm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'wm.nii.gz')

    # Create masks using predefined flags
    # White Matter Mask
    temp_wm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'temp_wm.nii.gz')
    if force_recreate_masks or not os.path.exists(temp_wm_mask_path):
        wm_command = f"mri_binarize --i {aseg_nii_path} --all-wm --o {temp_wm_mask_path}"
        subprocess.run(wm_command, shell=True)
    else:
        print("Temporary WM mask already exists, skipping mri_binarize for temp WM.")

    # Subcortical Gray Matter Mask
    temp_subcortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'temp_subcortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(temp_subcortical_gm_mask_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_nii_path} --subcort-gm --o {temp_subcortical_gm_mask_path}"
        subprocess.run(subcortical_gm_command, shell=True)
    else:
        print("Temporary subcortical GM mask already exists, skipping mri_binarize for temp subcortical GM.")

    # Cortical Gray Matter Mask
    # Create overall gray matter mask
    gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'gm.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_mask_path):
        gm_command = f"mri_binarize --i {aseg_nii_path} --gm --o {gm_mask_path}"
        subprocess.run(gm_command, shell=True)
    else:
        print("GM mask already exists, skipping mri_binarize for GM.")

    # Create gm_brainstem_mask
    gm_brainstem_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'gm_brainstem.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_brainstem_mask_path):
        gm_brainstem_command = f"mri_binarize --i {aseg_nii_path} --match 16 --o {gm_brainstem_mask_path}"
        subprocess.run(gm_brainstem_command, shell=True)
    else:
        print("Brainstem GM mask already exists, skipping mri_binarize for brainstem GM.")

    # Create gm_cerebellum_mask
    gm_cerebellum_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'gm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_cerebellum_mask_path):
        gm_cerebellum_command = f"mri_binarize --i {aseg_nii_path} --match 8 47 --o {gm_cerebellum_mask_path}"
        subprocess.run(gm_cerebellum_command, shell=True)
    else:
        print("Cerebellum GM mask already exists, skipping mri_binarize for cerebellum GM.")

    # Create wm_cerebellum_mask
    wm_cerebellum_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'wm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cerebellum_mask_path):
        wm_cerebellum_command = f"mri_binarize --i {aseg_nii_path} --match 7 46 --o {wm_cerebellum_mask_path}"
        subprocess.run(wm_cerebellum_command, shell=True)
    else:
        print("Cerebellum WM mask already exists, skipping mri_binarize for cerebellum WM.")

    # Create wm_cc_mask
    wm_cc_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'wm_cc.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cc_mask_path):
        wm_cc_command = f"mri_binarize --i {aseg_nii_path} --match 251 252 253 254 255 --o {wm_cc_mask_path}"
        subprocess.run(wm_cc_command, shell=True)
    else:
        print("Corpus Callosum WM mask already exists, skipping mri_binarize for corpus callosum WM.")

    # Create cortical gray matter mask by subtracting subcortical gray matter, brainstem, and cerebellum from total gray matter
    if force_recreate_masks or not os.path.exists(cortical_gm_mask_path):
        cortical_gm_command = f"fslmaths {gm_mask_path} -sub {temp_subcortical_gm_mask_path} -sub {gm_brainstem_mask_path} -sub {gm_cerebellum_mask_path} -thr 0.5 -bin {cortical_gm_mask_path}"
        subprocess.run(cortical_gm_command, shell=True)
    else:
        print("Cortical GM mask already exists, skipping creation.")

    # Create subcortical gray matter mask by subtracting brainstem and cerebellum from the temp subcortical GM mask
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_path):
        subcortical_gm_command = f"fslmaths {temp_subcortical_gm_mask_path} -sub {gm_brainstem_mask_path} -sub {gm_cerebellum_mask_path} -thr 0.5 -bin {subcortical_gm_mask_path}"
        subprocess.run(subcortical_gm_command, shell=True)
    else:
        print("Subcortical GM mask already exists, skipping creation.")

    # Create white matter mask by subtracting cerebellar WM and corpus callosum from the temp WM mask
    if force_recreate_masks or not os.path.exists(wm_mask_path):
        wm_command = f"fslmaths {temp_wm_mask_path} -sub {wm_cerebellum_mask_path} -sub {wm_cc_mask_path} -thr 0.5 -bin {wm_mask_path}"
        subprocess.run(wm_command, shell=True)
    else:
        print("WM mask already exists, skipping creation.")

    # Optionally, remove temporary files
    if os.path.exists(temp_wm_mask_path):
        os.remove(temp_wm_mask_path)
    if os.path.exists(temp_subcortical_gm_mask_path):
        os.remove(temp_subcortical_gm_mask_path)
    if os.path.exists(gm_mask_path):
        os.remove(gm_mask_path)

def plot_total_ct_and_patlak(time_points, C_t_total, C_a,
                             Ki, lam, SD_Ki, tissue_name,
                             save_path=None):
    """
    Re-written to drop the black cross markers on the Patlak panel.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    if C_t_total.size == 0 or C_a.size == 0:
        return

    if correct_signal_jumps:
        drop_idxs, trend, thresh = identify_drop_points(C_t_total)
    else:
        drop_idxs = np.array([], dtype=int)

    # Patlak co-ordinates
    dt = np.diff(time_points)
    x_patlak = np.zeros_like(C_a)
    y_patlak = np.zeros_like(C_a)
    for i in range(1, len(C_a)):
        if C_a[i] == 0:
            continue
        x_patlak[i] = np.sum(C_a[:i]*dt[:i]) / C_a[i]
        y_patlak[i] = C_t_total[i] / C_a[i]
    valid = (C_a!=0) & (x_patlak!=0) & (y_patlak!=0)
    x_pat, y_pat = x_patlak[valid], y_patlak[valid]
    idx_valid = np.where(valid)[0]
    bad_mask_pat = np.isin(idx_valid, drop_idxs)

    # ------------- figure ----------------
    fig = plt.figure(figsize=(12,5))
    gs  = plt.GridSpec(1,2,width_ratios=[2,1], wspace=0.35)

    # ---- CTC panel
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_points, C_t_total, color='blue', lw=2, label=f'{tissue_name} C(t)')
    if drop_idxs.size:
        t0, t1 = time_points[drop_idxs[0]], time_points[drop_idxs[-1]]
        ax1.axvspan(t0, t1, color='grey', alpha=0.3)
        ax1.scatter(time_points[drop_idxs], C_t_total[drop_idxs],
                    facecolors='none', edgecolors='black', s=50,
                    label='Ignored')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tissue C(t)')
    ax1.grid(True)
    ax1.legend(loc='upper right')

    ax1_a = ax1.twinx()
    ax1_a.plot(time_points, C_a, color='red', ls='--', lw=2, label='AIF')
    ax1_a.set_ylabel('C_a(t)')
    ax1_a.tick_params(axis='y', labelcolor='red')
    # merge legends
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax1_a.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')

    # ---- Patlak panel
    ax2 = fig.add_subplot(gs[1])
    keep = ~bad_mask_pat
    ax2.scatter(x_pat[keep], y_pat[keep],
                color='blue', s=25, marker='o',
                label='Used in fit')

    # now explicitly plot *all* the dropped points hollow:
    ax2.scatter(x_pat[bad_mask_pat], y_pat[bad_mask_pat],
                facecolors='none', edgecolors='blue', s=40,
                label='Ignored')


    if not np.isnan(Ki):
        ax2.plot(x_pat, lam/100 + (Ki/6000)*x_pat,
                 color='green', ls='--', label='Patlak fit')

    ax2.set_xlabel('∫C_a dt / C_a')
    ax2.set_ylabel('C_t / C_a')
    ax2.set_title(f"{tissue_name} | Ki={Ki:.4f}, λ={lam:.4f}")
    ax2.grid(True)
    ax2.legend(loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)




def coregistration(seg_mgz_path, dce_path, t2_path):
    import subprocess
    import nibabel as nib
    import numpy as np
    import os

    # Step 1: Convert segmentation file from .mgz to .nii.gz format
    aseg_nii_path = seg_mgz_path.replace('.mgz', '.nii.gz')
    if not os.path.exists(aseg_nii_path):
        print(f"Converting {seg_mgz_path} to {aseg_nii_path}...")
        result = subprocess.run(['mri_convert', seg_mgz_path, aseg_nii_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_convert failed with error:\n{result.stderr}")
            raise RuntimeError("mri_convert command failed.")
    else:
        print(f"{aseg_nii_path} already exists, skipping conversion.")

    # Ensure the converted file exists
    if not os.path.exists(aseg_nii_path):
        raise FileNotFoundError(f"Converted segmentation file not found: {aseg_nii_path}")

    # Step 2: Align the segmentation image to the DCE space
    aseg_in_dce_path = aseg_nii_path.replace('.nii.gz', '_in_DCE.nii.gz')
    if not os.path.exists(aseg_in_dce_path):
        if use_flirt_registration:
            mat_dce = aseg_nii_path.replace('.nii.gz', '_to_DCE.mat')
            flirt_reg_dce = [
                'flirt', '-in', aseg_nii_path, '-ref', dce_path,
                '-omat', mat_dce, '-dof', '6'
            ]
            print(f"Running FLIRT registration for DCE: {' '.join(flirt_reg_dce)}")
            result = subprocess.run(flirt_reg_dce, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT registration failed for DCE with error:\n{result.stderr}")
                raise RuntimeError("FLIRT registration for DCE failed.")

            flirt_apply_dce = [
                'flirt', '-in', aseg_nii_path, '-ref', dce_path,
                '-applyxfm', '-init', mat_dce, '-interp', 'nearestneighbour',
                '-out', aseg_in_dce_path
            ]
            print(f"Applying transform for DCE: {' '.join(flirt_apply_dce)}")
            result = subprocess.run(flirt_apply_dce, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT applyxfm failed for DCE with error:\n{result.stderr}")
                raise RuntimeError("FLIRT applyxfm for DCE failed.")
        else:
            flirt_cmd_dce = [
                'flirt', '-in', aseg_nii_path, '-ref', dce_path,
                '-applyxfm', '-usesqform', '-interp', 'nearestneighbour',
                '-out', aseg_in_dce_path
            ]
            print(f"Running FLIRT command for DCE: {' '.join(flirt_cmd_dce)}")
            result = subprocess.run(flirt_cmd_dce, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT failed for DCE with error:\n{result.stderr}")
                raise RuntimeError("FLIRT command for DCE failed.")
    else:
        print(f"Aligned segmentation to DCE already exists at {aseg_in_dce_path}.")

    # Ensure the output file was created
    if not os.path.exists(aseg_in_dce_path):
        raise FileNotFoundError(f"Expected output not found: {aseg_in_dce_path}")

    # Step 3: Align the segmentation image to the T2 space
    aseg_in_t2_path = aseg_nii_path.replace('.nii.gz', '_in_T2.nii.gz')
    if not os.path.exists(aseg_in_t2_path):
        if use_flirt_registration:
            mat_t2 = aseg_nii_path.replace('.nii.gz', '_to_T2.mat')
            flirt_reg_t2 = [
                'flirt', '-in', aseg_nii_path, '-ref', t2_path,
                '-omat', mat_t2, '-dof', '6'
            ]
            print(f"Running FLIRT registration for T2: {' '.join(flirt_reg_t2)}")
            result = subprocess.run(flirt_reg_t2, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT registration failed for T2 with error:\n{result.stderr}")
                raise RuntimeError("FLIRT registration for T2 failed.")

            flirt_apply_t2 = [
                'flirt', '-in', aseg_nii_path, '-ref', t2_path,
                '-applyxfm', '-init', mat_t2, '-interp', 'nearestneighbour',
                '-out', aseg_in_t2_path
            ]
            print(f"Applying transform for T2: {' '.join(flirt_apply_t2)}")
            result = subprocess.run(flirt_apply_t2, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT applyxfm failed for T2 with error:\n{result.stderr}")
                raise RuntimeError("FLIRT applyxfm for T2 failed.")
        else:
            flirt_cmd_t2 = [
                'flirt', '-in', aseg_nii_path, '-ref', t2_path,
                '-applyxfm', '-usesqform', '-interp', 'nearestneighbour',
                '-out', aseg_in_t2_path
            ]
            print(f"Running FLIRT command for T2: {' '.join(flirt_cmd_t2)}")
            result = subprocess.run(flirt_cmd_t2, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT failed for T2 with error:\n{result.stderr}")
                raise RuntimeError("FLIRT command for T2 failed.")
    else:
        print(f"Aligned segmentation to T2 already exists at {aseg_in_t2_path}.")

    # Ensure the output file was created
    if not os.path.exists(aseg_in_t2_path):
        raise FileNotFoundError(f"Expected output not found: {aseg_in_t2_path}")

    # Step 4: Create masks from the aligned segmentation images
    # For DCE space
    wm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_wm.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_mask_dce_path):
        wm_command = f"mri_binarize --i {aseg_in_dce_path} --all-wm --o {wm_mask_dce_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for WM in DCE space failed.")
    else:
        print("WM mask in DCE space already exists, skipping mri_binarize for WM.")

    subcortical_gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_subcortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_dce_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_in_dce_path} --subcort-gm --o {subcortical_gm_mask_dce_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for subcortical GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for subcortical GM in DCE space failed.")
    else:
        print("Subcortical GM mask in DCE space already exists, skipping mri_binarize.")

    gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_mask_dce_path):
        gm_command = f"mri_binarize --i {aseg_in_dce_path} --gm --o {gm_mask_dce_path}"
        print(f"Running command: {gm_command}")
        result = subprocess.run(gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for GM in DCE space failed.")
    else:
        print("GM mask in DCE space already exists, skipping mri_binarize.")

    # Create gm_brainstem_mask in DCE space
    gm_brainstem_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_gm_brainstem.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_brainstem_mask_dce_path):
        gm_brainstem_command = f"mri_binarize --i {aseg_in_dce_path} --match 16 --o {gm_brainstem_mask_dce_path}"
        print(f"Running command: {gm_brainstem_command}")
        result = subprocess.run(gm_brainstem_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for brainstem GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for brainstem GM in DCE space failed.")
    else:
        print("Brainstem GM mask in DCE space already exists, skipping mri_binarize.")

    # Create gm_cerebellum_mask in DCE space
    gm_cerebellum_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_gm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_cerebellum_mask_dce_path):
        gm_cerebellum_command = f"mri_binarize --i {aseg_in_dce_path} --match 8 47 --o {gm_cerebellum_mask_dce_path}"
        print(f"Running command: {gm_cerebellum_command}")
        result = subprocess.run(gm_cerebellum_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for cerebellum GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for cerebellum GM in DCE space failed.")
    else:
        print("Cerebellum GM mask in DCE space already exists, skipping mri_binarize.")

    # Create wm_cerebellum_mask in DCE space
    wm_cerebellum_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_wm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cerebellum_mask_dce_path):
        wm_cerebellum_command = f"mri_binarize --i {aseg_in_dce_path} --match 7 46 --o {wm_cerebellum_mask_dce_path}"
        print(f"Running command: {wm_cerebellum_command}")
        result = subprocess.run(wm_cerebellum_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for cerebellum WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for cerebellum WM in DCE space failed.")
    else:
        print("Cerebellum WM mask in DCE space already exists, skipping mri_binarize.")

    # Create wm_cc_mask in DCE space
    wm_cc_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_wm_cc.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cc_mask_dce_path):
        wm_cc_command = f"mri_binarize --i {aseg_in_dce_path} --match 251 252 253 254 255 --o {wm_cc_mask_dce_path}"
        print(f"Running command: {wm_cc_command}")
        result = subprocess.run(wm_cc_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for corpus callosum WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for corpus callosum WM in DCE space failed.")
    else:
        print("Corpus Callosum WM mask in DCE space already exists, skipping mri_binarize.")

    # Create cortical gray matter mask by subtracting subcortical GM, brainstem, and cerebellum from total GM
    cortical_gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_cortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(cortical_gm_mask_dce_path):
        cortical_gm_command = f"fslmaths {gm_mask_dce_path} -sub {subcortical_gm_mask_dce_path} -sub {gm_brainstem_mask_dce_path} -sub {gm_cerebellum_mask_dce_path} -thr 0.5 -bin {cortical_gm_mask_dce_path}"
        print(f"Running command: {cortical_gm_command}")
        result = subprocess.run(cortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for cortical GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for cortical GM in DCE space failed.")
    else:
        print("Cortical GM mask in DCE space already exists, skipping creation.")

    # Create subcortical GM mask by subtracting brainstem and cerebellum from subcortical GM
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_dce_path):
        subcortical_gm_command = f"fslmaths {subcortical_gm_mask_dce_path} -sub {gm_brainstem_mask_dce_path} -sub {gm_cerebellum_mask_dce_path} -thr 0.5 -bin {subcortical_gm_mask_dce_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for subcortical GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for subcortical GM in DCE space failed.")
    else:
        print("Subcortical GM mask in DCE space already exists, skipping creation.")

    # Create WM mask by subtracting cerebellar WM and corpus callosum from total WM
    if force_recreate_masks or not os.path.exists(wm_mask_dce_path):
        wm_command = f"fslmaths {wm_mask_dce_path} -sub {wm_cerebellum_mask_dce_path} -sub {wm_cc_mask_dce_path} -thr 0.5 -bin {wm_mask_dce_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for WM in DCE space failed.")
    else:
        print("WM mask in DCE space already exists, skipping creation.")

    # Similarly for T2 space
    wm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_wm.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_mask_t2_path):
        wm_command = f"mri_binarize --i {aseg_in_t2_path} --all-wm --o {wm_mask_t2_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for WM in T2 space failed.")
    else:
        print("WM mask in T2 space already exists, skipping mri_binarize for WM.")

    subcortical_gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_subcortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_t2_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_in_t2_path} --subcort-gm --o {subcortical_gm_mask_t2_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for subcortical GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for subcortical GM in T2 space failed.")
    else:
        print("Subcortical GM mask in T2 space already exists, skipping mri_binarize.")

    gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_mask_t2_path):
        gm_command = f"mri_binarize --i {aseg_in_t2_path} --gm --o {gm_mask_t2_path}"
        print(f"Running command: {gm_command}")
        result = subprocess.run(gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for GM in T2 space failed.")
    else:
        print("GM mask in T2 space already exists, skipping mri_binarize.")

    # Create gm_brainstem_mask in T2 space
    gm_brainstem_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_gm_brainstem.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_brainstem_mask_t2_path):
        gm_brainstem_command = f"mri_binarize --i {aseg_in_t2_path} --match 16 --o {gm_brainstem_mask_t2_path}"
        print(f"Running command: {gm_brainstem_command}")
        result = subprocess.run(gm_brainstem_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for brainstem GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for brainstem GM in T2 space failed.")
    else:
        print("Brainstem GM mask in T2 space already exists, skipping mri_binarize.")

    # Create gm_cerebellum_mask in T2 space
    gm_cerebellum_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_gm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_cerebellum_mask_t2_path):
        gm_cerebellum_command = f"mri_binarize --i {aseg_in_t2_path} --match 8 47 --o {gm_cerebellum_mask_t2_path}"
        print(f"Running command: {gm_cerebellum_command}")
        result = subprocess.run(gm_cerebellum_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for cerebellum GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for cerebellum GM in T2 space failed.")
    else:
        print("Cerebellum GM mask in T2 space already exists, skipping mri_binarize.")

    # Create wm_cerebellum_mask in T2 space
    wm_cerebellum_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_wm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cerebellum_mask_t2_path):
        wm_cerebellum_command = f"mri_binarize --i {aseg_in_t2_path} --match 7 46 --o {wm_cerebellum_mask_t2_path}"
        print(f"Running command: {wm_cerebellum_command}")
        result = subprocess.run(wm_cerebellum_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for cerebellum WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for cerebellum WM in T2 space failed.")
    else:
        print("Cerebellum WM mask in T2 space already exists, skipping mri_binarize.")

    # Create wm_cc_mask in T2 space
    wm_cc_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_wm_cc.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cc_mask_t2_path):
        wm_cc_command = f"mri_binarize --i {aseg_in_t2_path} --match 251 252 253 254 255 --o {wm_cc_mask_t2_path}"
        print(f"Running command: {wm_cc_command}")
        result = subprocess.run(wm_cc_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for corpus callosum WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for corpus callosum WM in T2 space failed.")
    else:
        print("Corpus Callosum WM mask in T2 space already exists, skipping mri_binarize.")

    # Create cortical gray matter mask by subtracting subcortical GM, brainstem, and cerebellum from total GM
    cortical_gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_cortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(cortical_gm_mask_t2_path):
        cortical_gm_command = f"fslmaths {gm_mask_t2_path} -sub {subcortical_gm_mask_t2_path} -sub {gm_brainstem_mask_t2_path} -sub {gm_cerebellum_mask_t2_path} -thr 0.5 -bin {cortical_gm_mask_t2_path}"
        print(f"Running command: {cortical_gm_command}")
        result = subprocess.run(cortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for cortical GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for cortical GM in T2 space failed.")
    else:
        print("Cortical GM mask in T2 space already exists, skipping creation.")

    # Create subcortical GM mask by subtracting brainstem and cerebellum from subcortical GM
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_t2_path):
        subcortical_gm_command = f"fslmaths {subcortical_gm_mask_t2_path} -sub {gm_brainstem_mask_t2_path} -sub {gm_cerebellum_mask_t2_path} -thr 0.5 -bin {subcortical_gm_mask_t2_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for subcortical GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for subcortical GM in T2 space failed.")
    else:
        print("Subcortical GM mask in T2 space already exists, skipping creation.")

    # Create WM mask by subtracting cerebellar WM and corpus callosum from total WM
    if force_recreate_masks or not os.path.exists(wm_mask_t2_path):
        wm_command = f"fslmaths {wm_mask_t2_path} -sub {wm_cerebellum_mask_t2_path} -sub {wm_cc_mask_t2_path} -thr 0.5 -bin {wm_mask_t2_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for WM in T2 space failed.")
    else:
        print("WM mask in T2 space already exists, skipping creation.")

    # Ensure all mask files exist before loading
    required_files = [
        wm_mask_dce_path, cortical_gm_mask_dce_path, subcortical_gm_mask_dce_path,
        gm_brainstem_mask_dce_path, gm_cerebellum_mask_dce_path, wm_cerebellum_mask_dce_path, wm_cc_mask_dce_path,
        wm_mask_t2_path, cortical_gm_mask_t2_path, subcortical_gm_mask_t2_path,
        gm_brainstem_mask_t2_path, gm_cerebellum_mask_t2_path, wm_cerebellum_mask_t2_path, wm_cc_mask_t2_path
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required mask file not found: {file_path}")

    # Load the masks
    wm_mask_dce = nib.load(wm_mask_dce_path).get_fdata().astype(bool)
    cortical_gm_mask_dce = nib.load(cortical_gm_mask_dce_path).get_fdata().astype(bool)
    subcortical_gm_mask_dce = nib.load(subcortical_gm_mask_dce_path).get_fdata().astype(bool)
    gm_brainstem_mask_dce = nib.load(gm_brainstem_mask_dce_path).get_fdata().astype(bool)
    gm_cerebellum_mask_dce = nib.load(gm_cerebellum_mask_dce_path).get_fdata().astype(bool)
    wm_cerebellum_mask_dce = nib.load(wm_cerebellum_mask_dce_path).get_fdata().astype(bool)
    wm_cc_mask_dce = nib.load(wm_cc_mask_dce_path).get_fdata().astype(bool)

    wm_mask_t2 = nib.load(wm_mask_t2_path).get_fdata().astype(bool)
    cortical_gm_mask_t2 = nib.load(cortical_gm_mask_t2_path).get_fdata().astype(bool)
    subcortical_gm_mask_t2 = nib.load(subcortical_gm_mask_t2_path).get_fdata().astype(bool)
    gm_brainstem_mask_t2 = nib.load(gm_brainstem_mask_t2_path).get_fdata().astype(bool)
    gm_cerebellum_mask_t2 = nib.load(gm_cerebellum_mask_t2_path).get_fdata().astype(bool)
    wm_cerebellum_mask_t2 = nib.load(wm_cerebellum_mask_t2_path).get_fdata().astype(bool)
    wm_cc_mask_t2 = nib.load(wm_cc_mask_t2_path).get_fdata().astype(bool)

    return (wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce,
            subcortical_gm_mask_t2, subcortical_gm_mask_dce, gm_brainstem_mask_t2,
            gm_brainstem_mask_dce, gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
            wm_cerebellum_mask_t2, wm_cerebellum_mask_dce, wm_cc_mask_t2, wm_cc_mask_dce)


def patlak_analysis_plotting(c_tissue, c_input, time):
    """
    Patlak fit that *ignores* any x or y that is NaN.
    All maths identical otherwise.
    Returns: Ki, lam, SD_Ki, x_full, y_full, included_mask
    where included_mask is True for points used in the fit.
    """
    if len(time) < 2:
        return (np.nan,)*3 + (np.array([]),)*3

    delta_t = np.diff(time)
    y = c_tissue / c_input
    x = np.concatenate(([0], np.cumsum(c_input[:-1]*delta_t))) / c_input
    good = (~np.isnan(x)) & (~np.isnan(y)) & (c_input != 0)

    # 1/3–2/3 Patlak window
    x_max = np.nanmax(x[good]) if good.any() else np.nan
    w = (x >= x_max/3) & (x <= x_max)
    good &= w

    if good.sum() < 2:
        return (np.nan,)*3 + (x, y, good)

    xm, ym = x[good].mean(), y[good].mean()
    Ki_raw = ((x[good]-xm)*(y[good]-ym)).sum() / ((x[good]-xm)**2).sum()
    lam_raw = ym - Ki_raw*xm

    resid = y[good] - (lam_raw + Ki_raw*x[good])
    SD_raw = np.sqrt((resid**2).sum() / ((x[good]-xm)**2).sum() / (good.sum()-2))

    return Ki_raw*6000, lam_raw*100, SD_raw*6000, x, y, good



def find_baseline_point_advanced(y_data, fs=15, cutoff=4.0, order=3, radius=10):
    """
    Finds the baseline point in the given 1D array of y-values based on advanced filtering and gradient analysis.
    """
    # Ignore the first point
    y_data = y_data[1:]
    
    # Apply the low-pass filter
    y_filtered = butter_lowpass_filter(y_data, cutoff, fs, order)
    
    # Compute the gradient of the filtered data
    gradient_filtered = np.gradient(y_filtered)
    
    # Find the major peaks in the gradient
    major_peaks_gradient = find_major_peaks(gradient_filtered, radius)
    
    # Find the baseline points as the points right before the major peaks in the gradient
    baseline_points_gradient = [peak - 1 for peak in major_peaks_gradient]
    
    # Select the baseline point with the smaller index
    baseline_point = min(baseline_points_gradient) if baseline_points_gradient else None
    
    # Adjust the index due to ignoring the first point
    if baseline_point is not None:
        baseline_point += 1
    
    return baseline_point

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.transform import resize

def plot_ctcs_and_patlak(
    t2_img_slice, dce_img_slice,
    wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
    wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
    avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
    x_patlak_wm, y_patlak_wm, Ki_wm, lambda_wm,
    x_patlak_cortical_gm, y_patlak_cortical_gm, Ki_cortical_gm, lambda_cortical_gm,
    x_patlak_subcortical_gm, y_patlak_subcortical_gm, Ki_subcortical_gm, lambda_subcortical_gm,
    slice_idx, save_path=None, boundary_mask=None, boundary_ctc=None,
    x_patlak_boundary=None, y_patlak_boundary=None, Ki_boundary=None, lambda_boundary=None,
    included_wm=None, included_cortical_gm=None, included_subcortical_gm=None, included_boundary=None,
    gm_brainstem_ctc=None, x_patlak_gm_brainstem=None, y_patlak_gm_brainstem=None, Ki_gm_brainstem=None, lambda_gm_brainstem=None, included_gm_brainstem=None,
    gm_cerebellum_ctc=None, x_patlak_gm_cerebellum=None, y_patlak_gm_cerebellum=None, Ki_gm_cerebellum=None, lambda_gm_cerebellum=None, included_gm_cerebellum=None,
    wm_cerebellum_ctc=None, x_patlak_wm_cerebellum=None, y_patlak_wm_cerebellum=None, Ki_wm_cerebellum=None, lambda_wm_cerebellum=None, included_wm_cerebellum=None,
    wm_cc_ctc=None, x_patlak_wm_cc=None, y_patlak_wm_cc=None, Ki_wm_cc=None, lambda_wm_cc=None, included_wm_cc=None,
    gm_brainstem_mask_t2=None, gm_brainstem_mask_dce=None,
    gm_cerebellum_mask_t2=None, gm_cerebellum_mask_dce=None,
    wm_cerebellum_mask_t2=None, wm_cerebellum_mask_dce=None,
    wm_cc_mask_t2=None, wm_cc_mask_dce=None,
    bad_wm=None, bad_cortical_gm=None, bad_subcortical_gm=None,
    bad_gm_brainstem=None, bad_gm_cerebellum=None, bad_wm_cerebellum=None,
    bad_wm_cc=None, bad_boundary=None,
    model_fits=None
):
    """
    Re-written version that draws **one** grey band for each continuous
    union-segment of bad samples.  Black ‘×’ on Patlak removed.
    All args identical to the original signature.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from skimage.transform import resize

    # --------------- helper(s) -----------------
    def resize_and_binarize(mask, target_shape):
        from skimage.transform import resize
        m = resize(mask.astype(float), target_shape, order=0,
                   preserve_range=True, anti_aliasing=False)
        return (m > 0.5).astype(float)

    def overlay_mask(ax, mask, rgba):
        if mask.any():
            rgba_img = np.zeros((*mask.shape, 4))
            rgba_img[..., :3] = rgba[:3]
            rgba_img[..., 3] = rgba[3] * mask
            ax.imshow(np.rot90(rgba_img), interpolation='none')

    # --------------- figure set-up --------------
    fig  = plt.figure(figsize=(14, 18))
    gs   = GridSpec(3, 2, figure=fig,
                    height_ratios=[1, 1, 1], width_ratios=[1, 1])
    gs.update(hspace=0.4)

    ax_t2  = fig.add_subplot(gs[0, 0])
    ax_dce = fig.add_subplot(gs[0, 1])
    ax_ctc = fig.add_subplot(gs[1, :])
    ax_pat = fig.add_subplot(gs[2, :])

    # ---------------- T2 / DCE panels ----------------
    t2_vmin, t2_vmax   = np.percentile(t2_img_slice, (1, 99))
    dce_vmin, dce_vmax = np.percentile(dce_img_slice, (1, 99))
    t2_norm  = (np.clip(t2_img_slice,  t2_vmin,  t2_vmax)-t2_vmin )/(t2_vmax -t2_vmin)
    dce_norm = (np.clip(dce_img_slice, dce_vmin, dce_vmax)-dce_vmin)/(dce_vmax-dce_vmin)

    ax_t2.imshow(np.rot90(t2_norm),  cmap='gray', vmin=0, vmax=1)
    ax_dce.imshow(np.rot90(dce_norm), cmap='gray', vmin=0, vmax=1)

    # colour scheme
    col = dict(
        white_matter  =[0,0,1,0.5],
        cortical_gm   =[1,0,0,0.5],
        subcortical_gm=[0.5,0,0,0.5],
        gm_brainstem  =[1,0.5,0,0.5],
        gm_cerebellum =[1,1,0,0.5],
        wm_cerebellum =[0,1,1,0.5],
        wm_cc         =[1,0,1,0.5],
        boundary      =[0,1,0,0.5]
    )

    # --- resize masks for display
    wm_mask_t2_r          = resize_and_binarize(wm_mask_t2,          t2_img_slice.shape)
    cortical_gm_mask_t2_r = resize_and_binarize(cortical_gm_mask_t2, t2_img_slice.shape)
    subcortical_gm_mask_t2_r = resize_and_binarize(subcortical_gm_mask_t2, t2_img_slice.shape)
    wm_mask_dce_r         = resize_and_binarize(wm_mask_dce,         dce_img_slice.shape)
    cortical_gm_mask_dce_r   = resize_and_binarize(cortical_gm_mask_dce, dce_img_slice.shape)
    subcortical_gm_mask_dce_r = resize_and_binarize(subcortical_gm_mask_dce, dce_img_slice.shape)
    # optional masks
    gm_brainstem_mask_t2_r   = resize_and_binarize(gm_brainstem_mask_t2,   t2_img_slice.shape) if gm_brainstem_mask_t2 is not None else np.zeros_like(t2_img_slice)
    gm_brainstem_mask_dce_r  = resize_and_binarize(gm_brainstem_mask_dce,  dce_img_slice.shape) if gm_brainstem_mask_dce is not None else np.zeros_like(dce_img_slice)
    gm_cerebellum_mask_t2_r  = resize_and_binarize(gm_cerebellum_mask_t2,  t2_img_slice.shape) if gm_cerebellum_mask_t2 is not None else np.zeros_like(t2_img_slice)
    gm_cerebellum_mask_dce_r = resize_and_binarize(gm_cerebellum_mask_dce, dce_img_slice.shape) if gm_cerebellum_mask_dce is not None else np.zeros_like(dce_img_slice)
    wm_cerebellum_mask_t2_r  = resize_and_binarize(wm_cerebellum_mask_t2,  t2_img_slice.shape) if wm_cerebellum_mask_t2 is not None else np.zeros_like(t2_img_slice)
    wm_cerebellum_mask_dce_r = resize_and_binarize(wm_cerebellum_mask_dce, dce_img_slice.shape) if wm_cerebellum_mask_dce is not None else np.zeros_like(dce_img_slice)
    wm_cc_mask_t2_r          = resize_and_binarize(wm_cc_mask_t2,          t2_img_slice.shape) if wm_cc_mask_t2 is not None else np.zeros_like(t2_img_slice)
    wm_cc_mask_dce_r         = resize_and_binarize(wm_cc_mask_dce,         dce_img_slice.shape) if wm_cc_mask_dce is not None else np.zeros_like(dce_img_slice)
    boundary_t2_r  = resize_and_binarize(boundary_mask,  t2_img_slice.shape)  if boundary_mask  is not None else None
    boundary_dce_r = resize_and_binarize(boundary_mask,  dce_img_slice.shape) if boundary_mask  is not None else None

    # apply overlays
    overlay_pairs = [
        (wm_mask_t2_r,          col['white_matter']),
        (cortical_gm_mask_t2_r, col['cortical_gm']),
        (subcortical_gm_mask_t2_r, col['subcortical_gm']),
        (gm_brainstem_mask_t2_r, col['gm_brainstem']),
        (gm_cerebellum_mask_t2_r,col['gm_cerebellum']),
        (wm_cerebellum_mask_t2_r,col['wm_cerebellum']),
        (wm_cc_mask_t2_r,       col['wm_cc']),
    ]
    for m,c in overlay_pairs:
        overlay_mask(ax_t2, m, c)
    if boundary_t2_r is not None:
        overlay_mask(ax_t2, boundary_t2_r, col['boundary'])
    ax_t2.set_title(f'T2 Slice {slice_idx} with masks'); ax_t2.axis('off')

    overlay_pairs_dce = [
        (wm_mask_dce_r,          col['white_matter']),
        (cortical_gm_mask_dce_r, col['cortical_gm']),
        (subcortical_gm_mask_dce_r,col['subcortical_gm']),
        (gm_brainstem_mask_dce_r, col['gm_brainstem']),
        (gm_cerebellum_mask_dce_r,col['gm_cerebellum']),
        (wm_cerebellum_mask_dce_r,col['wm_cerebellum']),
        (wm_cc_mask_dce_r,       col['wm_cc']),
    ]
    for m,c in overlay_pairs_dce:
        overlay_mask(ax_dce, m, c)
    if boundary_dce_r is not None:
        overlay_mask(ax_dce, boundary_dce_r, col['boundary'])
    ax_dce.set_title(f'DCE Slice {slice_idx} with masks'); ax_dce.axis('off')

    # ---------------- CTC panel -----------------
    ax_ctc.set_facecolor('#f7f7f7')

    # helper to add curve, points & build bad-mask list
    union_len = 0
    bad_masks = []

    def add_curve(ctc, label, colour, bad_mask):
        nonlocal union_len
        if ctc is None or not ctc.size:
            return
        ax_ctc.plot(ctc, label=label, color=colour)
        if bad_mask is not None and bad_mask.any():
            ax_ctc.scatter(np.where(bad_mask), ctc[bad_mask],
                           facecolors='none', edgecolors='black', s=50)
            bad_masks.append(bad_mask.copy())
            union_len = max(union_len, bad_mask.size)

    add_curve(avg_wm_ctc,              'Cortical WM',   'blue',      bad_wm)
    add_curve(avg_cortical_gm_ctc,     'Cortical GM',   'red',       bad_cortical_gm)
    add_curve(avg_subcortical_gm_ctc,  'Subcortical GM','darkred',   bad_subcortical_gm)
    add_curve(gm_brainstem_ctc,        'Brainstem',     'orange',    bad_gm_brainstem)
    add_curve(gm_cerebellum_ctc,       'Cerebellar GM', 'yellow',    bad_gm_cerebellum)
    add_curve(wm_cerebellum_ctc,       'Cerebellar WM', 'cyan',      bad_wm_cerebellum)
    add_curve(wm_cc_ctc,               'WM Corpus Callosum', 'magenta', bad_wm_cc)
    add_curve(boundary_ctc,            'Boundary',      'green',     bad_boundary)

    # ----------- compute & shade UNION of bad regions -----------
    if bad_masks:
        # pad masks so they all have equal length
        unified = np.zeros(union_len, dtype=bool)
        for m in bad_masks:
            pad = union_len - m.size
            if pad > 0:
                m = np.pad(m, (0, pad), constant_values=False)
            unified |= m

        # find contiguous True blocks
        idx = np.where(unified)[0]
        if idx.size:
            boundaries = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for block in boundaries:
                ax_ctc.axvspan(block[0], block[-1], color='grey', alpha=0.3)

    ax_ctc.set_title('Concentration functions')
    ax_ctc.legend(loc='upper right')
    ax_ctc.grid(True)


    # ---------------- Patlak panel (bottom) ------------------
    ax_pat.set_facecolor('#f7f7f7')

    if settings.KINETIC_MODEL.lower() == 'two_compartment' and model_fits is not None:
        def add_fit(ctc, fit, colour, label):
            if ctc is None or fit is None:
                return
            ax_pat.plot(ctc, color=colour, label=f"{label} data")
            ax_pat.plot(fit, '--', color=colour, label=f"{label} fit")

        add_fit(avg_wm_ctc,             model_fits.get('wm'),            'blue',    'Cortical WM')
        add_fit(avg_cortical_gm_ctc,    model_fits.get('cortical_gm'),   'red',     'Cortical GM')
        add_fit(avg_subcortical_gm_ctc, model_fits.get('subcortical_gm'), 'darkred','Subcortical GM')
        add_fit(gm_brainstem_ctc,       model_fits.get('gm_brainstem'),  'orange',  'Brainstem')
        add_fit(gm_cerebellum_ctc,      model_fits.get('gm_cerebellum'), 'gold',    'Cerebellar GM')
        add_fit(wm_cerebellum_ctc,      model_fits.get('wm_cerebellum'), 'cyan',    'Cerebellar WM')
        add_fit(wm_cc_ctc,              model_fits.get('wm_cc'),         'magenta', 'WM CC')
        add_fit(boundary_ctc,           model_fits.get('boundary'),      'green',   'Boundary')

        ax_pat.set_title('Two-compartment fit')
        ax_pat.set_xlabel('Time (frames)')
        ax_pat.set_ylabel('Concentration (mM)')
        ax_pat.grid(True)
        ax_pat.legend(loc='upper right')
    else:
        included_y_values = []

        def add_patlak(xp, yp, inc_mask, Ki, lam, colour, label):
            if xp.size == 0 or np.isnan(Ki):
                return
            ax_pat.scatter(xp[inc_mask], yp[inc_mask],
                           color=colour, marker='o', s=25, label=label)
            included_y_values.extend(yp[inc_mask].tolist())
            excl = ~inc_mask & np.isfinite(xp) & np.isfinite(yp)
            ax_pat.scatter(xp[excl], yp[excl],
                           facecolors='none', edgecolors=colour, s=40)
            ax_pat.plot(xp, lam/100 + (Ki/6000)*xp,
                        color=colour, linestyle='--')

        add_patlak(x_patlak_wm,             y_patlak_wm,             included_wm,
                  Ki_wm,             lambda_wm,             'blue',    'Cortical WM')
        add_patlak(x_patlak_cortical_gm,    y_patlak_cortical_gm,    included_cortical_gm,    Ki_cortical_gm,    lambda_cortical_gm,    'red',     'Cortical GM')
        add_patlak(x_patlak_subcortical_gm, y_patlak_subcortical_gm, included_subcortical_gm, Ki_subcortical_gm, lambda_subcortical_gm, 'darkred', 'Subcortical GM')
        add_patlak(x_patlak_gm_brainstem,   y_patlak_gm_brainstem,   included_gm_brainstem,   Ki_gm_brainstem,   lambda_gm_brainstem,   'orange',  'Brainstem')
        add_patlak(x_patlak_gm_cerebellum,  y_patlak_gm_cerebellum,  included_gm_cerebellum,  Ki_gm_cerebellum,  lambda_gm_cerebellum,  'gold',    'Cerebellar GM')
        add_patlak(x_patlak_wm_cerebellum,  y_patlak_wm_cerebellum,  included_wm_cerebellum,  Ki_wm_cerebellum,  lambda_wm_cerebellum,  'cyan',    'Cerebellar WM')
        add_patlak(x_patlak_wm_cc,          y_patlak_wm_cc,          included_wm_cc,          Ki_wm_cc,          lambda_wm_cc,          'magenta', 'WM CC')
        add_patlak(x_patlak_boundary,       y_patlak_boundary,       included_boundary,       Ki_boundary,       lambda_boundary,       'green',   'Boundary')

        if included_y_values:
            ymin, ymax = min(included_y_values), max(included_y_values)
            ax_pat.set_ylim(ymin, ymax)

        ax_pat.set_title('Patlak fit')
        ax_pat.set_xlim(0, 800)
        ax_pat.set_xlabel('∫C_a dt / C_a')
        ax_pat.set_ylabel('C_t / C_a')
        ax_pat.grid(True)
        ax_pat.legend(loc='lower left')
    

    plt.suptitle(f"Slice {slice_idx}", y=0.98)
    fit_text = ""
    if not np.isnan(Ki_wm):
        fit_text += f"Cortical WM:    Ki = {Ki_wm:.5f} ml/100g/min, λ = {lambda_wm:.5f} ml/100g\n"
    if not np.isnan(Ki_cortical_gm):
        fit_text += f"Cortical GM:    Ki = {Ki_cortical_gm:.5f} ml/100g/min, λ = {lambda_cortical_gm:.5f} ml/100g\n"
    if not np.isnan(Ki_subcortical_gm):
        fit_text += f"Subcortical GM: Ki = {Ki_subcortical_gm:.5f} ml/100g/min, λ = {lambda_subcortical_gm:.5f} ml/100g\n"
    if gm_brainstem_ctc is not None and not np.isnan(Ki_gm_brainstem):
        fit_text += f"Brainstem:      Ki = {Ki_gm_brainstem:.5f} ml/100g/min, λ = {lambda_gm_brainstem:.5f} ml/100g\n"
    if gm_cerebellum_ctc is not None and not np.isnan(Ki_gm_cerebellum):
        fit_text += f"Cerebellar GM:  Ki = {Ki_gm_cerebellum:.5f} ml/100g/min, λ = {lambda_gm_cerebellum:.5f} ml/100g\n"
    if wm_cerebellum_ctc is not None and not np.isnan(Ki_wm_cerebellum):
        fit_text += f"Cerebellar WM:  Ki = {Ki_wm_cerebellum:.5f} ml/100g/min, λ = {lambda_wm_cerebellum:.5f} ml/100g\n"
    if boundary_ctc is not None and not np.isnan(Ki_boundary):
        fit_text += f"Boundary:       Ki = {Ki_boundary:.5f} ml/100g/min, λ = {lambda_boundary:.5f} ml/100g"

    ax_pat.text(0.5, -0.23, fit_text.strip(),
                transform=ax_pat.transAxes, fontsize=10,
                ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.75))

    plt.tight_layout()
    if save_path:
        # Ensure the destination directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



def compute_and_plot_ctcs_median(
    data_4d, t2_img,
    wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
    wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
    T1_matrix, M0_matrix, analysis_directory, time_points_s, image_directory,
    dce_path, boundary=False, compute_per_voxel_Ki=False, compute_per_voxel_CBF=False,
    gm_brainstem_mask_t2=None, gm_brainstem_mask_dce=None,
    gm_cerebellum_mask_t2=None, gm_cerebellum_mask_dce=None,
    wm_cerebellum_mask_t2=None, wm_cerebellum_mask_dce=None,
    wm_cc_mask_t2=None, wm_cc_mask_dce=None
):
    """
    Computes median CTCs for different tissue types across slices, performs Patlak analysis,
    saves the results, and generates plots. Also computes the total median for the entire tissue volume.
    Optionally computes K_i and CBF per voxel and generates overlay images and NIfTI files.
    CBF values are scaled to millilitres per 100 grams of tissue per minute (ml/100g/min).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from scipy.ndimage import binary_dilation

    n_slices = t2_img.shape[2]

    # Load C_a once
    max_folder = os.path.join(analysis_directory, 'TSCC Data', 'Max')
    npy_files = [f for f in os.listdir(max_folder) if f.endswith('.npy')]

    if len(npy_files) != 1:
        raise ValueError(f"Expected exactly one .npy file in {max_folder}, but found {len(npy_files)}.")

    ca_file = npy_files[0]
    C_a_full = np.load(os.path.join(max_folder, ca_file))

    all_patlak_data = []
    Ki_wm_list = []
    Ki_cortical_gm_list = []
    Ki_subcortical_gm_list = []
    Ki_gm_brainstem_list = []
    Ki_gm_cerebellum_list = []
    Ki_wm_cerebellum_list = []
    Ki_wm_cc_list = []
    Ki_boundary_list = []

    # Lists to collect T1 and M0 values for each tissue across slices
    T1_wm_vals,           M0_wm_vals           = [], []
    T1_cortical_gm_vals,  M0_cortical_gm_vals  = [], []
    T1_subcortical_gm_vals, M0_subcortical_gm_vals = [], []
    T1_gm_brainstem_vals, M0_gm_brainstem_vals = [], []
    T1_gm_cerebellum_vals, M0_gm_cerebellum_vals = [], []
    T1_wm_cerebellum_vals, M0_wm_cerebellum_vals = [], []
    T1_wm_cc_vals,        M0_wm_cc_vals        = [], []
    if boundary:
        T1_boundary_vals, M0_boundary_vals = [], []

    # Initialize lists to collect all valid CTCs across slices
    wm_ctcs_total = []
    cortical_gm_ctcs_total = []
    subcortical_gm_ctcs_total = []
    gm_brainstem_ctcs_total = []
    gm_cerebellum_ctcs_total = []
    wm_cerebellum_ctcs_total = []
    wm_cc_ctcs_total = []
    boundary_ctcs_total = []

    

    # Initialize empty 3D arrays to store K_i values per voxel
    Ki_wm_image = np.full(data_4d.shape[:3], np.nan)
    Ki_cortical_gm_image = np.full(data_4d.shape[:3], np.nan)
    Ki_subcortical_gm_image = np.full(data_4d.shape[:3], np.nan)
    Ki_gm_brainstem_image = np.full(data_4d.shape[:3], np.nan)
    Ki_gm_cerebellum_image = np.full(data_4d.shape[:3], np.nan)
    Ki_wm_cerebellum_image = np.full(data_4d.shape[:3], np.nan)
    Ki_wm_cc_image = np.full(data_4d.shape[:3], np.nan)
    if boundary:
        Ki_boundary_image = np.full(data_4d.shape[:3], np.nan)

    # Initialize per-voxel Ki and CBF arrays if needed
    if compute_per_voxel_Ki:
        Ki_per_voxel = np.full(data_4d.shape[:3], np.nan)
    if compute_per_voxel_CBF:
        CBF_per_voxel = np.full(data_4d.shape[:3], np.nan)

    # Add tqdm progress bar to the loop
    for i in tqdm(range(n_slices), desc="Processing slices"):
        # Extract relevant masks for the current slice
        wm_slice_t2 = wm_mask_t2[:, :, i]
        cortical_gm_slice_t2 = cortical_gm_mask_t2[:, :, i]
        subcortical_gm_slice_t2 = subcortical_gm_mask_t2[:, :, i]
        gm_brainstem_slice_t2 = gm_brainstem_mask_t2[:, :, i]
        gm_cerebellum_slice_t2 = gm_cerebellum_mask_t2[:, :, i]
        wm_cerebellum_slice_t2 = wm_cerebellum_mask_t2[:, :, i]
        wm_cc_slice_t2 = wm_cc_mask_t2[:, :, i]

        wm_slice_dce = wm_mask_dce[:, :, i]
        cortical_gm_slice_dce = cortical_gm_mask_dce[:, :, i]
        subcortical_gm_slice_dce = subcortical_gm_mask_dce[:, :, i]
        gm_brainstem_slice_dce = gm_brainstem_mask_dce[:, :, i]
        gm_cerebellum_slice_dce = gm_cerebellum_mask_dce[:, :, i]
        wm_cerebellum_slice_dce = wm_cerebellum_mask_dce[:, :, i]
        wm_cc_slice_dce = wm_cc_mask_dce[:, :, i]

        # Combine cortical and subcortical GM masks for boundary calculation
        gm_slice_dce = np.logical_or(cortical_gm_slice_dce, subcortical_gm_slice_dce)

        # Compute the boundary mask if required
        if boundary:
            wm_dilated = binary_dilation(wm_slice_dce, iterations=1)
            gm_dilated = binary_dilation(gm_slice_dce, iterations=1)
            boundary_mask = np.logical_and(wm_dilated, gm_dilated)
            boundary_indices = np.argwhere(boundary_mask)
        else:
            boundary_mask = None
            boundary_indices = []

        # Find voxel indices for each tissue type in the slice
        wm_indices = np.argwhere(wm_slice_dce)
        cortical_gm_indices = np.argwhere(cortical_gm_slice_dce)
        subcortical_gm_indices = np.argwhere(subcortical_gm_slice_dce)
        gm_brainstem_indices = np.argwhere(gm_brainstem_slice_dce)
        gm_cerebellum_indices = np.argwhere(gm_cerebellum_slice_dce)
        wm_cerebellum_indices = np.argwhere(wm_cerebellum_slice_dce)
        wm_cc_indices = np.argwhere(wm_cc_slice_dce)

        # Collect T1 and M0 values for each tissue type
        T1_wm_vals.extend(T1_matrix[:, :, i][wm_slice_dce].ravel())
        M0_wm_vals.extend(M0_matrix[:, :, i][wm_slice_dce].ravel())
        T1_cortical_gm_vals.extend(T1_matrix[:, :, i][cortical_gm_slice_dce].ravel())
        M0_cortical_gm_vals.extend(M0_matrix[:, :, i][cortical_gm_slice_dce].ravel())
        T1_subcortical_gm_vals.extend(T1_matrix[:, :, i][subcortical_gm_slice_dce].ravel())
        M0_subcortical_gm_vals.extend(M0_matrix[:, :, i][subcortical_gm_slice_dce].ravel())
        T1_gm_brainstem_vals.extend(T1_matrix[:, :, i][gm_brainstem_slice_dce].ravel())
        M0_gm_brainstem_vals.extend(M0_matrix[:, :, i][gm_brainstem_slice_dce].ravel())
        T1_gm_cerebellum_vals.extend(T1_matrix[:, :, i][gm_cerebellum_slice_dce].ravel())
        M0_gm_cerebellum_vals.extend(M0_matrix[:, :, i][gm_cerebellum_slice_dce].ravel())
        T1_wm_cerebellum_vals.extend(T1_matrix[:, :, i][wm_cerebellum_slice_dce].ravel())
        M0_wm_cerebellum_vals.extend(M0_matrix[:, :, i][wm_cerebellum_slice_dce].ravel())
        T1_wm_cc_vals.extend(T1_matrix[:, :, i][wm_cc_slice_dce].ravel())
        M0_wm_cc_vals.extend(M0_matrix[:, :, i][wm_cc_slice_dce].ravel())
        if boundary_mask is not None:
            T1_boundary_vals.extend(T1_matrix[:, :, i][boundary_mask].ravel())
            M0_boundary_vals.extend(M0_matrix[:, :, i][boundary_mask].ravel())

        # Initialize lists to store valid CTCs
        wm_ctcs = []
        cortical_gm_ctcs = []
        subcortical_gm_ctcs = []
        gm_brainstem_ctcs = []
        gm_cerebellum_ctcs = []
        wm_cerebellum_ctcs = []
        wm_cc_ctcs = []
        boundary_ctcs = []

        # Function to process CTCs for a given set of indices
        def process_ctcs(indices):
            ctcs = []
            for (x, y) in indices:
                voxel_time_course = data_4d[x, y, i, :]
                T1 = T1_matrix[x, y, i]
                M0 = M0_matrix[x, y, i]
                C_t_0 = compute_CTC(voxel_time_course, T1, m0=M0)
                baseline_point = find_baseline_point_advanced(C_t_0)
                C_t = custom_shifter(C_t_0, baseline_point)

                # Exclude CTCs with NaNs or zeros
                if np.isnan(C_t).any() or np.all(C_t == 0):
                    continue
                ctcs.append(C_t)
            return ctcs

        # Process CTCs for each tissue type
        wm_ctcs = process_ctcs(wm_indices)
        cortical_gm_ctcs = process_ctcs(cortical_gm_indices)
        subcortical_gm_ctcs = process_ctcs(subcortical_gm_indices)
        gm_brainstem_ctcs = process_ctcs(gm_brainstem_indices)
        gm_cerebellum_ctcs = process_ctcs(gm_cerebellum_indices)
        wm_cerebellum_ctcs = process_ctcs(wm_cerebellum_indices)
        wm_cc_ctcs = process_ctcs(wm_cc_indices)

        # Process CTCs for boundary if required
        if boundary and len(boundary_indices) > 0:
            boundary_ctcs = process_ctcs(boundary_indices)

        # Add the valid CTCs from this slice to the total lists
        wm_ctcs_total.extend(wm_ctcs)
        cortical_gm_ctcs_total.extend(cortical_gm_ctcs)
        subcortical_gm_ctcs_total.extend(subcortical_gm_ctcs)
        gm_brainstem_ctcs_total.extend(gm_brainstem_ctcs)
        gm_cerebellum_ctcs_total.extend(gm_cerebellum_ctcs)
        wm_cerebellum_ctcs_total.extend(wm_cerebellum_ctcs)
        wm_cc_ctcs_total.extend(wm_cc_ctcs)
        if boundary and boundary_ctcs:
            boundary_ctcs_total.extend(boundary_ctcs)

        # Compute median CTCs if valid CTCs are available
        avg_wm_ctc = np.median(wm_ctcs, axis=0) if wm_ctcs else np.array([])
        avg_cortical_gm_ctc = np.median(cortical_gm_ctcs, axis=0) if cortical_gm_ctcs else np.array([])
        avg_subcortical_gm_ctc = np.median(subcortical_gm_ctcs, axis=0) if subcortical_gm_ctcs else np.array([])
        avg_gm_brainstem_ctc = np.median(gm_brainstem_ctcs, axis=0) if gm_brainstem_ctcs else np.array([])
        avg_gm_cerebellum_ctc = np.median(gm_cerebellum_ctcs, axis=0) if gm_cerebellum_ctcs else np.array([])
        avg_wm_cerebellum_ctc = np.median(wm_cerebellum_ctcs, axis=0) if wm_cerebellum_ctcs else np.array([])
        avg_wm_cc_ctc = np.median(wm_cc_ctcs, axis=0) if wm_cc_ctcs else np.array([])
        if boundary and boundary_ctcs:
            avg_boundary_ctc = np.median(boundary_ctcs, axis=0)
        else:
            avg_boundary_ctc = np.array([])

        if correct_signal_jumps:
            avg_wm_ctc,               bad_wm,_               = mask_problematic(avg_wm_ctc)
            avg_cortical_gm_ctc,      bad_cortical_gm,_      = mask_problematic(avg_cortical_gm_ctc)
            avg_subcortical_gm_ctc,   bad_subcortical_gm,_   = mask_problematic(avg_subcortical_gm_ctc)

            avg_gm_brainstem_ctc,  bad_gm_brainstem,  _ = mask_problematic(avg_gm_brainstem_ctc)
            avg_gm_cerebellum_ctc, bad_gm_cerebellum, _ = mask_problematic(avg_gm_cerebellum_ctc)
            avg_wm_cerebellum_ctc, bad_wm_cerebellum, _ = mask_problematic(avg_wm_cerebellum_ctc)
            avg_wm_cc_ctc,         bad_wm_cc,         _ = mask_problematic(avg_wm_cc_ctc)
            if boundary and avg_boundary_ctc.size:
                avg_boundary_ctc, bad_boundary,_ = mask_problematic(avg_boundary_ctc)
            else:
                bad_boundary = None
        else:
            bad_wm = bad_cortical_gm = bad_subcortical_gm = None
            bad_gm_brainstem = bad_gm_cerebellum = bad_wm_cerebellum = None
            bad_wm_cc = bad_boundary = None
        # Save the tissue concentration curves as .npy files
        save_dir_ctc = os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'AI')
        os.makedirs(save_dir_ctc, exist_ok=True)

        np.save(os.path.join(save_dir_ctc, f'wm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_wm_ctc)
        np.save(os.path.join(save_dir_ctc, f'cortical_gm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_cortical_gm_ctc)
        np.save(os.path.join(save_dir_ctc, f'subcortical_gm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_subcortical_gm_ctc)
        np.save(os.path.join(save_dir_ctc, f'gm_brainstem_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_gm_brainstem_ctc)
        np.save(os.path.join(save_dir_ctc, f'gm_cerebellum_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_gm_cerebellum_ctc)
        np.save(os.path.join(save_dir_ctc, f'wm_cerebellum_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_wm_cerebellum_ctc)
        np.save(os.path.join(save_dir_ctc, f'wm_cc_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_wm_cc_ctc)
        if boundary and avg_boundary_ctc.size > 0:
            np.save(os.path.join(save_dir_ctc, f'boundary_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_boundary_ctc)

        # Ensure the CTCs and C_a_full have the same length
        min_length = len(C_a_full)
        ctc_list = [
            avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
            avg_gm_brainstem_ctc, avg_gm_cerebellum_ctc, avg_wm_cerebellum_ctc, avg_wm_cc_ctc,
            avg_boundary_ctc
        ]
        for ctc in ctc_list:
            if ctc.size > 0:
                min_length = min(min_length, ctc.size)

        C_a_slice = C_a_full[:min_length]
        time_points = time_points_s[:min_length]

        # Truncate CTCs to match length
        C_t_wm = avg_wm_ctc[:min_length] if avg_wm_ctc.size > 0 else np.array([])
        C_t_cortical_gm = avg_cortical_gm_ctc[:min_length] if avg_cortical_gm_ctc.size > 0 else np.array([])
        C_t_subcortical_gm = avg_subcortical_gm_ctc[:min_length] if avg_subcortical_gm_ctc.size > 0 else np.array([])
        C_t_gm_brainstem = avg_gm_brainstem_ctc[:min_length] if avg_gm_brainstem_ctc.size > 0 else np.array([])
        C_t_gm_cerebellum = avg_gm_cerebellum_ctc[:min_length] if avg_gm_cerebellum_ctc.size > 0 else np.array([])
        C_t_wm_cerebellum = avg_wm_cerebellum_ctc[:min_length] if avg_wm_cerebellum_ctc.size > 0 else np.array([])
        C_t_wm_cc = avg_wm_cc_ctc[:min_length] if avg_wm_cc_ctc.size > 0 else np.array([])
        if boundary and avg_boundary_ctc.size > 0:
            C_t_boundary = avg_boundary_ctc[:min_length]
        else:
            C_t_boundary = np.array([])

        # Perform kinetic model fit for each tissue type
        def perform_model_fit(C_t):
            if C_t.size == 0:
                return (np.nan, np.nan, np.nan, None, np.array([], dtype=bool))
            if settings.KINETIC_MODEL.lower() == 'two_compartment':
                Ki, lam, SD_Ki, fit_curve = two_compartment_fit(C_a_slice, C_t, time_points)
                return Ki, lam, SD_Ki, fit_curve, np.array([], dtype=bool)
            else:
                Ki, lam, SD_Ki, x_patlak, y_patlak, included = patlak_analysis_plotting(C_t, C_a_slice, time_points)
                return Ki, lam, SD_Ki, (x_patlak, y_patlak), included

        Ki_wm, lambda_wm, SD_Ki_wm, curve_wm, included_wm = perform_model_fit(C_t_wm)
        Ki_cortical_gm, lambda_cortical_gm, SD_Ki_cortical_gm, curve_cortical_gm, included_cortical_gm = perform_model_fit(C_t_cortical_gm)
        Ki_subcortical_gm, lambda_subcortical_gm, SD_Ki_subcortical_gm, curve_subcortical_gm, included_subcortical_gm = perform_model_fit(C_t_subcortical_gm)
        Ki_gm_brainstem, lambda_gm_brainstem, SD_Ki_gm_brainstem, curve_gm_brainstem, included_gm_brainstem = perform_model_fit(C_t_gm_brainstem)
        Ki_gm_cerebellum, lambda_gm_cerebellum, SD_Ki_gm_cerebellum, curve_gm_cerebellum, included_gm_cerebellum = perform_model_fit(C_t_gm_cerebellum)
        Ki_wm_cerebellum, lambda_wm_cerebellum, SD_Ki_wm_cerebellum, curve_wm_cerebellum, included_wm_cerebellum = perform_model_fit(C_t_wm_cerebellum)
        Ki_wm_cc, lambda_wm_cc, SD_Ki_wm_cc, curve_wm_cc, included_wm_cc = perform_model_fit(C_t_wm_cc)
        if boundary and C_t_boundary.size > 0:
            Ki_boundary, lambda_boundary, SD_Ki_boundary, curve_boundary, included_boundary = perform_model_fit(C_t_boundary)
        else:
            Ki_boundary = np.nan
            lambda_boundary = np.nan
            SD_Ki_boundary = np.nan
            curve_boundary = None
            included_boundary = np.array([], dtype=bool)

        # Collect Ki values for plotting
        Ki_wm_list.append(Ki_wm)
        Ki_cortical_gm_list.append(Ki_cortical_gm)
        Ki_subcortical_gm_list.append(Ki_subcortical_gm)
        Ki_gm_brainstem_list.append(Ki_gm_brainstem)
        Ki_gm_cerebellum_list.append(Ki_gm_cerebellum)
        Ki_wm_cerebellum_list.append(Ki_wm_cerebellum)
        Ki_wm_cc_list.append(Ki_wm_cc)
        if boundary:
            Ki_boundary_list.append(Ki_boundary)

        # Assign Ki values to the masks in the current slice
        Ki_wm_image[:, :, i][wm_slice_dce] = Ki_wm
        Ki_cortical_gm_image[:, :, i][cortical_gm_slice_dce] = Ki_cortical_gm
        Ki_subcortical_gm_image[:, :, i][subcortical_gm_slice_dce] = Ki_subcortical_gm
        Ki_gm_brainstem_image[:, :, i][gm_brainstem_slice_dce] = Ki_gm_brainstem
        Ki_gm_cerebellum_image[:, :, i][gm_cerebellum_slice_dce] = Ki_gm_cerebellum
        Ki_wm_cerebellum_image[:, :, i][wm_cerebellum_slice_dce] = Ki_wm_cerebellum
        Ki_wm_cc_image[:, :, i][wm_cc_slice_dce] = Ki_wm_cc
        if boundary:
            Ki_boundary_image[:, :, i][boundary_mask] = Ki_boundary

        # Plot the results for the current slice. Images are always written
        # under ``AI/Tissue functions`` so that ``_rename_model_outputs`` can
        # move the entire directory to ``AI_patlak`` or ``AI_tikhonov`` after
        # the model run completes.
        fit_curves = {
            'wm': curve_wm,
            'cortical_gm': curve_cortical_gm,
            'subcortical_gm': curve_subcortical_gm,
            'gm_brainstem': curve_gm_brainstem,
            'gm_cerebellum': curve_gm_cerebellum,
            'wm_cerebellum': curve_wm_cerebellum,
            'wm_cc': curve_wm_cc,
            'boundary': curve_boundary
        }

        plot_ctcs_and_patlak(
            t2_img[:, :, i], data_4d[:, :, i, 20],
            wm_slice_t2, cortical_gm_slice_t2, subcortical_gm_slice_t2,
            wm_slice_dce, cortical_gm_slice_dce, subcortical_gm_slice_dce,
            avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
            np.array([]), np.array([]), Ki_wm, lambda_wm,
            np.array([]), np.array([]), Ki_cortical_gm, lambda_cortical_gm,
            np.array([]), np.array([]), Ki_subcortical_gm, lambda_subcortical_gm,
            slice_idx=i+1,
            save_path=os.path.join(image_directory, 'AI', 'Tissue functions', f"AI_Tissue_slice_{i+1}_segmented_median.png"),
            boundary_mask=boundary_mask,
            boundary_ctc=avg_boundary_ctc,
            x_patlak_boundary=np.array([]), y_patlak_boundary=np.array([]),
            Ki_boundary=Ki_boundary, lambda_boundary=lambda_boundary,
            included_wm=included_wm,
            included_cortical_gm=included_cortical_gm,
            included_subcortical_gm=included_subcortical_gm,
            included_boundary=included_boundary,
            gm_brainstem_ctc=avg_gm_brainstem_ctc,
            x_patlak_gm_brainstem=np.array([]),
            y_patlak_gm_brainstem=np.array([]),
            Ki_gm_brainstem=Ki_gm_brainstem,
            lambda_gm_brainstem=lambda_gm_brainstem,
            included_gm_brainstem=included_gm_brainstem,
            gm_cerebellum_ctc=avg_gm_cerebellum_ctc,
            x_patlak_gm_cerebellum=np.array([]),
            y_patlak_gm_cerebellum=np.array([]),
            Ki_gm_cerebellum=Ki_gm_cerebellum,
            lambda_gm_cerebellum=lambda_gm_cerebellum,
            included_gm_cerebellum=included_gm_cerebellum,
            wm_cerebellum_ctc=avg_wm_cerebellum_ctc,
            x_patlak_wm_cerebellum=np.array([]),
            y_patlak_wm_cerebellum=np.array([]),
            Ki_wm_cerebellum=Ki_wm_cerebellum,
            lambda_wm_cerebellum=lambda_wm_cerebellum,
            included_wm_cerebellum=included_wm_cerebellum,
            wm_cc_ctc=avg_wm_cc_ctc,
            x_patlak_wm_cc=np.array([]),
            y_patlak_wm_cc=np.array([]),
            Ki_wm_cc=Ki_wm_cc,
            lambda_wm_cc=lambda_wm_cc,
            included_wm_cc=included_wm_cc,
            model_fits=fit_curves,
            gm_brainstem_mask_t2=gm_brainstem_slice_t2,
            gm_brainstem_mask_dce=gm_brainstem_slice_dce,
            gm_cerebellum_mask_t2=gm_cerebellum_slice_t2,
            gm_cerebellum_mask_dce=gm_cerebellum_slice_dce,
            wm_cerebellum_mask_t2=wm_cerebellum_slice_t2,          
            wm_cerebellum_mask_dce=wm_cerebellum_slice_dce,        
            wm_cc_mask_t2=wm_cc_slice_t2,                          
            wm_cc_mask_dce=wm_cc_slice_dce,
            bad_wm=bad_wm,
            bad_cortical_gm=bad_cortical_gm,
            bad_subcortical_gm=bad_subcortical_gm,
            bad_gm_brainstem=bad_gm_brainstem,
            bad_gm_cerebellum=bad_gm_cerebellum,
            bad_wm_cerebellum=bad_wm_cerebellum,
            bad_wm_cc=bad_wm_cc,
            bad_boundary=bad_boundary                      
    )

        # Collect data for JSON output
        patlak_data = {
            'slice': i + 1,
            'white_matter_median': {
                'Ki': Ki_wm,
                'SD_Ki': SD_Ki_wm,
                'lambda': lambda_wm,
                'voxel_count': int(np.sum(wm_slice_dce))
            },
            'cortical_gray_matter_median': {
                'Ki': Ki_cortical_gm,
                'SD_Ki': SD_Ki_cortical_gm,
                'lambda': lambda_cortical_gm,
                'voxel_count': int(np.sum(cortical_gm_slice_dce))
            },
            'subcortical_gray_matter_median': {
                'Ki': Ki_subcortical_gm,
                'SD_Ki': SD_Ki_subcortical_gm,
                'lambda': lambda_subcortical_gm,
                'voxel_count': int(np.sum(subcortical_gm_slice_dce))
            },
            'gm_brainstem_median': {
                'Ki': Ki_gm_brainstem,
                'SD_Ki': SD_Ki_gm_brainstem,
                'lambda': lambda_gm_brainstem,
                'voxel_count': int(np.sum(gm_brainstem_slice_dce))
            },
            'gm_cerebellum_median': {
                'Ki': Ki_gm_cerebellum,
                'SD_Ki': SD_Ki_gm_cerebellum,
                'lambda': lambda_gm_cerebellum,
                'voxel_count': int(np.sum(gm_cerebellum_slice_dce))
            },
            'wm_cerebellum_median': {
                'Ki': Ki_wm_cerebellum,
                'SD_Ki': SD_Ki_wm_cerebellum,
                'lambda': lambda_wm_cerebellum,
                'voxel_count': int(np.sum(wm_cerebellum_slice_dce))
            },
            'wm_cc_median': {
                'Ki': Ki_wm_cc,
                'SD_Ki': SD_Ki_wm_cc,
                'lambda': lambda_wm_cc,
                'voxel_count': int(np.sum(wm_cc_slice_dce))
            }
        }

        if boundary and avg_boundary_ctc.size > 0:
            patlak_data['boundary_median'] = {
                'Ki': Ki_boundary,
                'SD_Ki': SD_Ki_boundary,
                'lambda': lambda_boundary,
                'voxel_count': int(np.sum(boundary_mask))
            }

        all_patlak_data.append(patlak_data)

        # Compute K_i and/or CBF per voxel if enabled
        if compute_per_voxel_Ki or compute_per_voxel_CBF:
            # Combine WM and GM masks for the current slice
            gm_slice_dce = np.logical_or(cortical_gm_slice_dce, subcortical_gm_slice_dce)
            brain_mask_slice = np.logical_or(wm_slice_dce, gm_slice_dce)
            brain_indices = np.argwhere(brain_mask_slice)

            # Initialize K_i and CBF slice arrays
            if compute_per_voxel_Ki:
                Ki_slice = np.full(brain_mask_slice.shape, np.nan)
            if compute_per_voxel_CBF:
                CBF_slice = np.full(brain_mask_slice.shape, np.nan)

            # For each voxel in the brain mask, compute K_i and/or CBF
            for (x, y) in brain_indices:
                voxel_time_course = data_4d[x, y, i, :]
                T1 = T1_matrix[x, y, i]
                M0 = M0_matrix[x, y, i]
                C_t_0 = compute_CTC(voxel_time_course, T1, m0=M0)
                baseline_point = find_baseline_point_advanced(C_t_0)
                C_t = custom_shifter(C_t_0, baseline_point)

                # Exclude CTCs with NaNs or zeros
                if np.isnan(C_t).any() or np.all(C_t == 0):
                    continue

                # Ensure C_t and C_a_full have the same length
                min_length_voxel = min(len(C_t), len(C_a_full))
                C_t_voxel = C_t[:min_length_voxel]
                C_a_voxel = C_a_full[:min_length_voxel]
                time_points_voxel = time_points_s[:min_length_voxel]

                if compute_per_voxel_Ki:
                    # Perform Patlak analysis
                    Ki_voxel, _, _, _, _, _ = patlak_analysis_plotting(C_t_voxel, C_a_voxel, time_points_voxel)
                    Ki_slice[x, y] = Ki_voxel

                if compute_per_voxel_CBF:
                    delta_t = np.diff(time_points_voxel)[0]
                    A = construct_convolution_matrix(C_a_voxel, delta_t)
                    lambd = 0.1  # Adjust as needed

                    # Solve for the residue function
                    try:
                        R_estimated = tikhonov_regularization(A, C_t_voxel, lambd)
                        # R[0] represents flow in 1/s. Scale to ml/100g/min.
                        CBF_voxel = R_estimated[0] * 6000
                        CBF_slice[x, y] = CBF_voxel
                    except np.linalg.LinAlgError:
                        continue  # Skip if the matrix is singular

            # Store the K_i and/or CBF slice in the 3D arrays
            if compute_per_voxel_Ki:
                Ki_per_voxel[:, :, i] = Ki_slice
            if compute_per_voxel_CBF:
                CBF_per_voxel[:, :, i] = CBF_slice

    # Save Ki images as NIfTI files
    affine = nib.load(dce_path).affine

    Ki_wm_nii = nib.Nifti1Image(Ki_wm_image, affine)
    Ki_wm_path = os.path.join(analysis_directory, 'Ki_wm.nii.gz')
    nib.save(Ki_wm_nii, Ki_wm_path)
    print(f"K_i WM saved to {Ki_wm_path}")

    Ki_cortical_gm_nii = nib.Nifti1Image(Ki_cortical_gm_image, affine)
    Ki_cortical_gm_path = os.path.join(analysis_directory, 'Ki_cortical_gm.nii.gz')
    nib.save(Ki_cortical_gm_nii, Ki_cortical_gm_path)
    print(f"K_i Cortical GM saved to {Ki_cortical_gm_path}")

    Ki_subcortical_gm_nii = nib.Nifti1Image(Ki_subcortical_gm_image, affine)
    Ki_subcortical_gm_path = os.path.join(analysis_directory, 'Ki_subcortical_gm.nii.gz')
    nib.save(Ki_subcortical_gm_nii, Ki_subcortical_gm_path)
    print(f"K_i Subcortical GM saved to {Ki_subcortical_gm_path}")

    if boundary:
        Ki_boundary_nii = nib.Nifti1Image(Ki_boundary_image, affine)
        Ki_boundary_path = os.path.join(analysis_directory, 'Ki_boundary.nii.gz')
        nib.save(Ki_boundary_nii, Ki_boundary_path)
        print(f"K_i Boundary saved to {Ki_boundary_path}")

    # Compute global min and max for K_i
    if compute_per_voxel_Ki:
        global_Ki_min = np.nanmin(Ki_per_voxel)
        global_Ki_max = np.nanmax(Ki_per_voxel)
        print(f"Global K_i min: {global_Ki_min}, max: {global_Ki_max}")

        # Generate overlay images for K_i
        for i in range(n_slices):
            Ki_slice = Ki_per_voxel[:, :, i]
            if np.isnan(Ki_slice).all():
                continue  # Skip slices without valid K_i values

            save_dir_overlay = os.path.join(image_directory, 'AI', 'Ki Overlays')
            os.makedirs(save_dir_overlay, exist_ok=True)
            save_path_overlay = os.path.join(save_dir_overlay, f"Ki_overlay_slice_{i+1}.png")
            plot_Ki_overlay(
                data_4d[:, :, i, 20], Ki_slice, slice_idx=i+1, save_path=save_path_overlay,
                vmin=global_Ki_min, vmax=global_Ki_max
            )

        # Save Ki_per_voxel as a .nii file
        Ki_per_voxel_nii = nib.Nifti1Image(Ki_per_voxel, affine=nib.load(dce_path).affine)
        Ki_per_voxel_path = os.path.join(analysis_directory, 'Ki_per_voxel.nii.gz')
        nib.save(Ki_per_voxel_nii, Ki_per_voxel_path)
        print(f"K_i per voxel saved to {Ki_per_voxel_path}")

    # Compute global min and max for CBF
    if compute_per_voxel_CBF:
        global_CBF_min = np.nanmin(CBF_per_voxel)
        global_CBF_max = np.nanmax(CBF_per_voxel)
        print(f"Global CBF min: {global_CBF_min}, max: {global_CBF_max}")

        # Generate overlay images for CBF
        for i in range(n_slices):
            CBF_slice = CBF_per_voxel[:, :, i]
            if np.isnan(CBF_slice).all():
                continue  # Skip slices without valid CBF values

            save_dir_overlay = os.path.join(image_directory, 'AI', 'CBF Overlays')
            os.makedirs(save_dir_overlay, exist_ok=True)
            save_path_overlay = os.path.join(save_dir_overlay, f"CBF_overlay_slice_{i+1}.png")
            plot_CBF_overlay(
                data_4d[:, :, i, 20], CBF_slice, slice_idx=i+1, save_path=save_path_overlay,
                vmin=global_CBF_min, vmax=global_CBF_max
            )

        # Save CBF_per_voxel as a .nii file
        CBF_per_voxel_nii = nib.Nifti1Image(CBF_per_voxel, affine=nib.load(dce_path).affine)
        CBF_per_voxel_path = os.path.join(analysis_directory, 'CBF_per_voxel.nii.gz')
        nib.save(CBF_per_voxel_nii, CBF_per_voxel_path)
        print(f"CBF per voxel saved to {CBF_per_voxel_path}")

    # Save all Patlak data to JSON file after processing all slices
    json_file_path = os.path.join(analysis_directory, "AI_values_median.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(all_patlak_data, json_file, indent=4)

    # Plot Ki values as a function of slice number
    if Ki_wm_list:
        num_processed_slices = len(Ki_wm_list)
        slice_numbers = range(1, num_processed_slices + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(slice_numbers, Ki_wm_list, label='White Matter Ki', marker='o')
        plt.plot(slice_numbers, Ki_cortical_gm_list, label='Cortical Gray Matter Ki', marker='o')
        plt.plot(slice_numbers, Ki_subcortical_gm_list, label='Subcortical Gray Matter Ki', marker='o')
        if boundary and Ki_boundary_list:
            plt.plot(slice_numbers, Ki_boundary_list, label='Boundary Ki', marker='o')
        plt.xlabel('Slice Number')
        plt.ylabel('K_i')
        plt.title('K_i values across Slices')
        plt.legend()
        plt.grid(True)

        # Ensure the directory exists
        save_dir = os.path.join(image_directory, 'AI', 'Tissue functions')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'Ki_vs_slice_median.png'))
        plt.close()
    else:
        print("No Ki values were computed; skipping Ki plot.")

    # ----------------------------------------------------------------------------- 
    # 1) Build overall-median CTCs (already trimmed to the common min_length)
    # -----------------------------------------------------------------------------
    avg_wm_ctc_total            = np.median(wm_ctcs_total,            axis=0) if wm_ctcs_total            else np.array([])
    avg_cortical_gm_ctc_total   = np.median(cortical_gm_ctcs_total,   axis=0) if cortical_gm_ctcs_total   else np.array([])
    avg_subcortical_gm_ctc_total= np.median(subcortical_gm_ctcs_total,axis=0) if subcortical_gm_ctcs_total else np.array([])
    avg_boundary_ctc_total      = np.median(boundary_ctcs_total,      axis=0) if boundary_ctcs_total      else np.array([])
    avg_gm_brainstem_ctc_total  = np.median(gm_brainstem_ctcs_total,  axis=0) if gm_brainstem_ctcs_total  else np.array([])
    avg_gm_cerebellum_ctc_total = np.median(gm_cerebellum_ctcs_total, axis=0) if gm_cerebellum_ctcs_total else np.array([])
    avg_wm_cerebellum_ctc_total = np.median(wm_cerebellum_ctcs_total, axis=0) if wm_cerebellum_ctcs_total else np.array([])
    avg_wm_cc_ctc_total         = np.median(wm_cc_ctcs_total,         axis=0) if wm_cc_ctcs_total         else np.array([])

    # find common length
    min_length = len(C_a_full)
    for ctc in (avg_wm_ctc_total, avg_cortical_gm_ctc_total, avg_subcortical_gm_ctc_total,
                avg_boundary_ctc_total, avg_gm_brainstem_ctc_total, avg_gm_cerebellum_ctc_total,
                avg_wm_cerebellum_ctc_total, avg_wm_cc_ctc_total):
        if ctc.size:
            min_length = min(min_length, ctc.size)

    C_a_total         = C_a_full[:min_length]
    time_points_total = time_points_s[:min_length]

    C_t_wm_total             = avg_wm_ctc_total[:min_length]
    C_t_cortical_gm_total    = avg_cortical_gm_ctc_total[:min_length]
    C_t_subcortical_gm_total = avg_subcortical_gm_ctc_total[:min_length]
    C_t_boundary_total       = avg_boundary_ctc_total[:min_length]
    C_t_gm_brainstem_total   = avg_gm_brainstem_ctc_total[:min_length]
    C_t_gm_cerebellum_total  = avg_gm_cerebellum_ctc_total[:min_length]
    C_t_wm_cerebellum_total  = avg_wm_cerebellum_ctc_total[:min_length]
    C_t_wm_cc_total          = avg_wm_cc_ctc_total[:min_length]

    # ----------------------------------------------------------------------------- 
    # 2) Helper: run mask_problematic on the *trimmed* curve, then Patlak
    # -----------------------------------------------------------------------------
    def patlak_total(C_t):
        if not C_t.size:
            return np.nan, np.nan, np.nan
        if settings.KINETIC_MODEL.lower() == 'two_compartment':
            Ki, lam, SD, _ = two_compartment_fit(C_a_total, C_t, time_points_total)
            return Ki, lam, SD
        if correct_signal_jumps:
            _, bad, _ = mask_problematic(C_t)
        else:
            bad = None
        Ki, lam, SD, *_ = patlak_with_exclusions(C_t, C_a_total, time_points_total, bad_mask=bad)
        return Ki, lam, SD

    # ----------------------------------------------------------------------------- 
    # 3) Patlak for every tissue
    # -----------------------------------------------------------------------------
    Ki_wm_total,           lambda_wm_total,           SD_Ki_wm_total            = patlak_total(C_t_wm_total)
    Ki_cortical_gm_total,  lambda_cortical_gm_total,  SD_Ki_cortical_gm_total   = patlak_total(C_t_cortical_gm_total)
    Ki_subcortical_gm_total,lambda_subcortical_gm_total,SD_Ki_subcortical_gm_total = patlak_total(C_t_subcortical_gm_total)
    Ki_boundary_total,     lambda_boundary_total,     SD_Ki_boundary_total      = patlak_total(C_t_boundary_total)
    Ki_gm_brainstem_total, lambda_gm_brainstem_total, SD_Ki_gm_brainstem_total  = patlak_total(C_t_gm_brainstem_total)
    Ki_gm_cerebellum_total,lambda_gm_cerebellum_total,SD_Ki_gm_cerebellum_total = patlak_total(C_t_gm_cerebellum_total)
    Ki_wm_cerebellum_total,lambda_wm_cerebellum_total,SD_Ki_wm_cerebellum_total = patlak_total(C_t_wm_cerebellum_total)
    Ki_wm_cc_total,        lambda_wm_cc_total,        SD_Ki_wm_cc_total         = patlak_total(C_t_wm_cc_total)

    # ----------------------------------------------------------------------------- 
    # 4) Collect everything for JSON and plotting
    # -----------------------------------------------------------------------------
    tissue_results = {
        "white_matter":      dict(C_t=C_t_wm_total,           Ki=Ki_wm_total,           lam=lambda_wm_total,           SD_Ki=SD_Ki_wm_total,          vox=len(wm_ctcs_total)),
        "cortical_gm":       dict(C_t=C_t_cortical_gm_total,  Ki=Ki_cortical_gm_total,  lam=lambda_cortical_gm_total,  SD_Ki=SD_Ki_cortical_gm_total, vox=len(cortical_gm_ctcs_total)),
        "subcortical_gm":    dict(C_t=C_t_subcortical_gm_total,Ki=Ki_subcortical_gm_total,lam=lambda_subcortical_gm_total,SD_Ki=SD_Ki_subcortical_gm_total,vox=len(subcortical_gm_ctcs_total)),
        "gm_brainstem":      dict(C_t=C_t_gm_brainstem_total, Ki=Ki_gm_brainstem_total, lam=lambda_gm_brainstem_total, SD_Ki=SD_Ki_gm_brainstem_total,vox=len(gm_brainstem_ctcs_total)),
        "gm_cerebellum":     dict(C_t=C_t_gm_cerebellum_total,Ki=Ki_gm_cerebellum_total,lam=lambda_gm_cerebellum_total,SD_Ki=SD_Ki_gm_cerebellum_total,vox=len(gm_cerebellum_ctcs_total)),
        "wm_cerebellum":     dict(C_t=C_t_wm_cerebellum_total,Ki=Ki_wm_cerebellum_total,lam=lambda_wm_cerebellum_total,SD_Ki=SD_Ki_wm_cerebellum_total,vox=len(wm_cerebellum_ctcs_total)),
        "wm_cc":             dict(C_t=C_t_wm_cc_total,        Ki=Ki_wm_cc_total,        lam=lambda_wm_cc_total,        SD_Ki=SD_Ki_wm_cc_total,       vox=len(wm_cc_ctcs_total)),
    }

    if boundary and C_t_boundary_total.size:
        tissue_results["boundary"] = dict(C_t=C_t_boundary_total, Ki=Ki_boundary_total,
                                        lam=lambda_boundary_total, SD_Ki=SD_Ki_boundary_total,
                                        vox=len(boundary_ctcs_total))

    # ----------------------------------------------------------------------
    # Compute global median T1 and M0 values for each tissue
    # ----------------------------------------------------------------------
    def median_or_nan(vals):
        return float(np.median(vals)) if vals else float('nan')

    t1_m0_results = {
        "white_matter_median_total": {
            "T1": median_or_nan(T1_wm_vals),
            "M0": median_or_nan(M0_wm_vals),
            "voxel_count": len(T1_wm_vals)
        },
        "cortical_gm_median_total": {
            "T1": median_or_nan(T1_cortical_gm_vals),
            "M0": median_or_nan(M0_cortical_gm_vals),
            "voxel_count": len(T1_cortical_gm_vals)
        },
        "subcortical_gm_median_total": {
            "T1": median_or_nan(T1_subcortical_gm_vals),
            "M0": median_or_nan(M0_subcortical_gm_vals),
            "voxel_count": len(T1_subcortical_gm_vals)
        },
        "gm_brainstem_median_total": {
            "T1": median_or_nan(T1_gm_brainstem_vals),
            "M0": median_or_nan(M0_gm_brainstem_vals),
            "voxel_count": len(T1_gm_brainstem_vals)
        },
        "gm_cerebellum_median_total": {
            "T1": median_or_nan(T1_gm_cerebellum_vals),
            "M0": median_or_nan(M0_gm_cerebellum_vals),
            "voxel_count": len(T1_gm_cerebellum_vals)
        },
        "wm_cerebellum_median_total": {
            "T1": median_or_nan(T1_wm_cerebellum_vals),
            "M0": median_or_nan(M0_wm_cerebellum_vals),
            "voxel_count": len(T1_wm_cerebellum_vals)
        },
        "wm_cc_median_total": {
            "T1": median_or_nan(T1_wm_cc_vals),
            "M0": median_or_nan(M0_wm_cc_vals),
            "voxel_count": len(T1_wm_cc_vals)
        },
    }

    if boundary and T1_boundary_vals:
        t1_m0_results["boundary_median_total"] = {
            "T1": median_or_nan(T1_boundary_vals),
            "M0": median_or_nan(M0_boundary_vals),
            "voxel_count": len(T1_boundary_vals)
        }

    json_file_path_t1m0 = os.path.join(analysis_directory, "T1_M0_values_median_total.json")
    with open(json_file_path_t1m0, "w") as jf:
        json.dump(t1_m0_results, jf, indent=4)

    # Write JSON
    json_file_path_total = os.path.join(analysis_directory, "AI_values_median_total.json")
    with open(json_file_path_total, "w") as jf:
        json.dump({
            t+"_median_total": {
                "Ki":   d["Ki"],
                "SD_Ki":d["SD_Ki"],
                "lambda":d["lam"],
                "voxel_count":d["vox"]
            } for t,d in tissue_results.items()
        }, jf, indent=4)

    # ----------------------------------------------------------------------------- 
    # 5) Create one PNG per tissue
    # -----------------------------------------------------------------------------
    for tissue_name, vals in tissue_results.items():
        save_path = os.path.join(image_directory, "AI", "Tissue functions",
                                f"{tissue_name}_total_CT_and_patlak.png")
        plot_total_ct_and_patlak(
            time_points=time_points_total,
            C_t_total  = vals["C_t"],
            C_a        = C_a_total,
            Ki         = vals["Ki"],
            lam        = vals["lam"],
            SD_Ki      = vals["SD_Ki"],
            tissue_name= tissue_name.replace('_', ' ').title(),
            save_path  = save_path
        )
        
def plot_Ki_overlay(dce_slice, Ki_slice, slice_idx, save_path, vmin, vmax):
    """
    Plots the DCE image slice with an overlay of K_i values.

    Parameters:
    - dce_slice: 2D numpy array of the DCE image at a specific time point.
    - Ki_slice: 2D numpy array of K_i values for the slice.
    - slice_idx: Integer indicating the slice number.
    - save_path: Path to save the overlay image.
    - vmin: Global minimum K_i value for consistent color scaling.
    - vmax: Global maximum K_i value for consistent color scaling.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy.ma as ma

    # Mask out NaN values in K_i
    Ki_masked = ma.masked_invalid(Ki_slice)

    # Set up the figure and axis
    plt.figure(figsize=(8, 8))
    plt.imshow(np.rot90(dce_slice), cmap='gray', interpolation='nearest')

    # Overlay K_i values using a colormap
    plt.imshow(np.rot90(Ki_masked), cmap='jet', interpolation='nearest', alpha=0.6,
               norm=Normalize(vmin=vmin, vmax=vmax))

    plt.colorbar(label='K$_i$ (ml/100g/min)')
    plt.title(f'Blood Brain Barrier permeability (K$_i$) per voxel - Slice {slice_idx}')
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_CBF_overlay(dce_slice, CBF_slice, slice_idx, save_path, vmin, vmax):
    """
    Plots the DCE image slice with an overlay of CBF values.

    Parameters:
    - dce_slice: 2D numpy array of the DCE image at a specific time point.
    - CBF_slice: 2D numpy array of CBF values for the slice.
    - slice_idx: Integer indicating the slice number.
    - save_path: Path to save the overlay image.
    - vmin: Global minimum CBF value for consistent color scaling.
    - vmax: Global maximum CBF value for consistent color scaling.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy.ma as ma

    # Mask out NaN values in CBF
    CBF_masked = ma.masked_invalid(CBF_slice)

    # Set up the figure and axis
    plt.figure(figsize=(8, 8))
    plt.imshow(np.rot90(dce_slice), cmap='gray', interpolation='nearest')

    # Overlay CBF values using a colormap
    plt.imshow(np.rot90(CBF_masked), cmap='viridis', interpolation='nearest', alpha=0.6,
               norm=Normalize(vmin=vmin, vmax=vmax))

    plt.colorbar(label='CBF (ml/100g/min)')
    plt.title(f'Cerebral Blood Flow (CBF) per voxel - Slice {slice_idx}')
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()


def _rename_model_outputs(analysis_directory, image_directory, suffix, boundary=False):
    """Rename analysis outputs with a model-specific suffix."""
    import shutil

    files = [
        'Ki_wm.nii.gz',
        'Ki_cortical_gm.nii.gz',
        'Ki_subcortical_gm.nii.gz',
        'Ki_per_voxel.nii.gz',
        'CBF_per_voxel.nii.gz',
        'AI_values_median.json',
        'Ki_vs_slice_median.png',
        'AI_values_median_total.json',
        'T1_M0_values_median_total.json',
        'Ki_map_atlas.nii.gz',
        'SD_Ki_map_atlas.nii.gz',
        'vp_map_atlas.nii.gz',
        'Ki_values_atlas.json',
    ]
    if boundary:
        files.append('Ki_boundary.nii.gz')

    for fname in files:
        src = os.path.join(analysis_directory, fname)
        if os.path.exists(src):
            base, ext = os.path.splitext(fname)
            if ext == '.gz':
                base2, ext2 = os.path.splitext(base)
                dst = os.path.join(analysis_directory, f'{base2}{suffix}{ext2}.gz')
            else:
                dst = os.path.join(analysis_directory, f'{base}{suffix}{ext}')
            os.rename(src, dst)

    ai_dir = os.path.join(image_directory, 'AI')
    if os.path.exists(ai_dir):
        dst_dir = os.path.join(image_directory, f'AI{suffix}')
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.rename(ai_dir, dst_dir)


def _tissue_function_AI(model, analysis_directory, nifti_directory, image_directory, filenames, parameters):
    """Run tissue function analysis for a single kinetic model."""
    settings.KINETIC_MODEL = model
    t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename,\
        flair_3D_filename, axial_flair_3D_filename, axial_t2_2D_filename, dce_filename = filenames

    IsVFA, IsIR, apple_metal, boundary, RERUN_SEGMENTATION, SEGMENTATION_METHOD, _ = parameters

    # Automatically enable jump correction when requested via a JSON file
    global correct_signal_jumps
    jumpfix_file = os.path.join(os.path.dirname(analysis_directory), 'apply_jumpfix.json')
    if os.path.exists(jumpfix_file):
        print('[!] apply_jumpfix.json detected – enabling signal jump correction')
        correct_signal_jumps = True

    # Allow optional FLIRT-based coregistration via an environment variable
    global use_flirt_registration
    if os.getenv('USE_FLIRT_REGISTRATION') == '1':
        print('[!] USE_FLIRT_REGISTRATION=1 – enabling FLIRT-based coregistration')
        use_flirt_registration = True

    fastsurfer_path = '/Users/edt/FastSurfer/run_fastsurfer.sh'
    t1_path = os.path.join(nifti_directory, t1_3D_filename)
    seg_dir = os.path.join(nifti_directory, 'segmentation')
    sid = 'segmentation'  # Define the subject ID
    seg_mgz_path = os.path.join(seg_dir, sid, 'mri', 'aparc.DKTatlas+aseg.deep.mgz')
    t2_path = os.path.join(nifti_directory, axial_t2_2D_filename)
    dce_path = os.path.join(nifti_directory, dce_filename)

    # Ensure segmentation directory exists
    os.makedirs(seg_dir, exist_ok=True)

    # Run segmentation and create masks
    segmentation(
        fastsurfer_path,
        seg_mgz_path,
        t1_path,
        seg_dir,
        sid,
        apple_metal,
        RERUN_SEGMENTATION,
        SEGMENTATION_METHOD,
    )

    # Paths to masks in the same directory as aparc.DKTatlas+aseg.deep.mgz
    mask_dir = os.path.dirname(seg_mgz_path)
    cortical_gm_mask_path = os.path.join(mask_dir, 'cortical_gm.nii.gz')
    subcortical_gm_mask_path = os.path.join(mask_dir, 'subcortical_gm.nii.gz')
    wm_mask_path = os.path.join(mask_dir, 'wm.nii.gz')

    print('[!] Coregistering GM/WM masks onto T2 and DCE space')
    (wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce,
     subcortical_gm_mask_t2, subcortical_gm_mask_dce, gm_brainstem_mask_t2,
     gm_brainstem_mask_dce, gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
     wm_cerebellum_mask_t2, wm_cerebellum_mask_dce, wm_cc_mask_t2, wm_cc_mask_dce) = coregistration(
        seg_mgz_path=seg_mgz_path,
        dce_path=dce_path,
        t2_path=t2_path
    )

    # Load the T2 image for visualization
    t2_img = nib.load(t2_path).get_fdata()

    # Plot the predictions with gray and white matter masks on T2
    plot_predictions_with_masks(t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
                                gm_brainstem_mask_t2, gm_cerebellum_mask_t2, wm_cerebellum_mask_t2,
                                wm_cc_mask_t2, image_directory)

    # Continue with the rest of your processing, ensuring to include the new masks in your analysis and plotting

    # Load the DCE 4D data
    data_4d = np.array(nib.load(dce_path).get_fdata())

    # Load T1 and M0 matrices
    T1_matrix = load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl'))
    M0_matrix = load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl'))

    # Compute time_points_s
    TR = nib.load(dce_path).header.get_zooms()[-1]
    num_volumes = data_4d.shape[-1]
    total_scan_duration = TR * num_volumes
    time_points_s = np.linspace(0, total_scan_duration, num_volumes)

    # Update compute_and_plot_ctcs_median function to include new tissue types
    compute_and_plot_ctcs_median(
        data_4d, t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
        wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
        T1_matrix, M0_matrix, analysis_directory, time_points_s, image_directory,
        dce_path=dce_path, boundary=boundary, compute_per_voxel_Ki=True, compute_per_voxel_CBF=True,
        gm_brainstem_mask_t2=gm_brainstem_mask_t2, gm_brainstem_mask_dce=gm_brainstem_mask_dce,
        gm_cerebellum_mask_t2=gm_cerebellum_mask_t2, gm_cerebellum_mask_dce=gm_cerebellum_mask_dce,
        wm_cerebellum_mask_t2=wm_cerebellum_mask_t2, wm_cerebellum_mask_dce=wm_cerebellum_mask_dce,
        wm_cc_mask_t2=wm_cc_mask_t2, wm_cc_mask_dce=wm_cc_mask_dce
    )

    # The atlas is the segmentation in DCE space
    atlas_path = os.path.join(
        nifti_directory, 
        'segmentation', 
        'segmentation', 
        'mri', 
        'aparc.DKTatlas+aseg.deep_in_DCE.nii.gz'
    )

    max_folder = os.path.join(analysis_directory, 'TSCC Data', 'Max')
    npy_files = [f for f in os.listdir(max_folder) if f.endswith('.npy')]
    ca_file = npy_files[0]
    C_a_full = np.load(os.path.join(max_folder, ca_file))

    output_dir = analysis_directory

    compute_Ki_from_atlas(
        atlas_path=atlas_path,
        data_4d=data_4d,
        T1_matrix=T1_matrix,
        M0_matrix=M0_matrix,
        time_points_s=time_points_s,
        C_a_full=C_a_full,  
        affine=nib.load(dce_path).affine,
        output_directory=output_dir,
        compute_CTC=compute_CTC,
        find_baseline_point_advanced=find_baseline_point_advanced,
        custom_shifter=custom_shifter,
        patlak_analysis_plotting=patlak_analysis_plotting
    )

    _rename_model_outputs(analysis_directory, image_directory, f"_{model}", boundary)


def tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters):
    """Run tissue function analysis using the configured kinetic model."""
    model_setting = settings.KINETIC_MODEL.lower()
    models = ['patlak', 'tikhonov'] if model_setting == 'both' else [model_setting]
    ai_base = os.path.join(image_directory, 'AI')
    for m in models:
        print(f'[!] Running {m} model')
        if os.path.exists(ai_base):
            shutil.rmtree(ai_base)
        os.makedirs(ai_base, exist_ok=True)
        _tissue_function_AI(m, analysis_directory, nifti_directory, image_directory, filenames, parameters)
