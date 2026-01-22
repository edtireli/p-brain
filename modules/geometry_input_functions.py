import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import utils.settings as settings
from utils.mapping import choice2type
from utils.plotting import plot_time_intensity_curves_AI, plot_time_intensity_curves_and_CTC_AI


@dataclass(frozen=True)
class GeometryRoiConfig:
    rica_radius: int
    lica_radius: int
    sss_radius: int
    rica_slices: int
    lica_slices: int
    sss_slices: int
    rica_z_range: str
    lica_z_range: str
    sss_z_range: str
    sss_midline_band: int
    baseline_frames: int


def _parse_z_range(spec: str, n_slices: int, default: tuple[int, int]) -> tuple[int, int]:
    if not spec:
        return default
    if ":" not in spec:
        return default
    left, right = spec.split(":", 1)
    try:
        start = int(left.strip())
        end = int(right.strip())
    except ValueError:
        return default

    start = max(0, min(n_slices - 1, start))
    end = max(0, min(n_slices - 1, end))
    if end < start:
        start, end = end, start
    return start, end


def _disk_voxels(cx: int, cy: int, radius: int, width: int, height: int) -> np.ndarray:
    radius = max(1, int(radius))
    y, x = np.ogrid[:height, :width]
    mask = (x - cy) ** 2 + (y - cx) ** 2 <= radius ** 2
    coords = np.argwhere(mask)
    return coords.astype(int)


def _compute_peak_map(dce4d: np.ndarray, baseline_frames: int) -> np.ndarray:
    baseline_frames = max(1, int(baseline_frames))
    baseline = dce4d[..., :baseline_frames].mean(axis=-1)
    peak = dce4d.max(axis=-1)
    return (peak - baseline).astype(np.float32)


def _compute_peak_and_ttp(
    dce4d: np.ndarray, baseline_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    baseline_frames = max(1, int(baseline_frames))
    baseline = dce4d[..., :baseline_frames].mean(axis=-1)
    sig = (dce4d - baseline[..., None]).astype(np.float32)
    peak = sig.max(axis=-1)
    ttp = sig.argmax(axis=-1).astype(np.int16)
    return peak, ttp


def _robust_ttp_target(peak_map: np.ndarray, ttp_map: np.ndarray, brain_mask: np.ndarray) -> int:
    m = brain_mask & np.isfinite(peak_map) & (peak_map > 0)
    if not np.any(m):
        return int(np.median(ttp_map))

    vals = peak_map[m]
    thr = np.percentile(vals, 99.5) if vals.size else 0.0
    m2 = m & (peak_map >= thr)
    if not np.any(m2):
        return int(np.median(ttp_map[m]))

    ttp_vals = ttp_map[m2].astype(np.int32)
    if ttp_vals.size == 0:
        return int(np.median(ttp_map[m]))
    return int(np.median(ttp_vals))


def _brain_mask_from_mean(dce4d: np.ndarray) -> np.ndarray:
    mean_img = dce4d.mean(axis=-1)
    thr = np.percentile(mean_img, 60)
    return mean_img > thr


def _best_centers_by_slice(
    score3d: np.ndarray,
    allowed_mask3d: np.ndarray,
    z_range: tuple[int, int],
    k_slices: int,
) -> list[tuple[int, int, int, float]]:
    z0, z1 = z_range
    candidates: list[tuple[int, int, int, float]] = []
    for z in range(z0, z1 + 1):
        m = allowed_mask3d[:, :, z]
        if not np.any(m):
            continue
        slice_scores = np.where(m, score3d[:, :, z], -np.inf)
        flat_idx = int(np.nanargmax(slice_scores))
        x, y = np.unravel_index(flat_idx, slice_scores.shape)
        s = float(slice_scores[x, y])
        if not np.isfinite(s):
            continue
        candidates.append((int(x), int(y), int(z), s))

    candidates.sort(key=lambda t: t[3], reverse=True)
    if k_slices <= 0:
        return []
    return candidates[: min(k_slices, len(candidates))]


def _save_roi_outputs(
    *,
    roi_type: str,
    roi_subtype: str,
    centers: list[tuple[int, int, int, float]],
    radius: int,
    dce4d: np.ndarray,
    ref_img: nib.Nifti1Image,
    analysis_dir: str,
    image_dir: str,
    nifti_dir: str,
    time_points_s: np.ndarray,
    filenames,
    is_vfa: bool,
):
    width, height, n_slices = dce4d.shape[1], dce4d.shape[0], dce4d.shape[2]

    mask3d = np.zeros((height, width, n_slices), dtype=np.uint8)

    roi_data_dir = os.path.join(analysis_dir, "ROI Data", roi_type, roi_subtype)
    frame_data_dir = os.path.join(analysis_dir, "Frame Data", roi_type, roi_subtype)
    os.makedirs(roi_data_dir, exist_ok=True)
    os.makedirs(frame_data_dir, exist_ok=True)

    # Also keep the conventional ITC/CTC outputs to stay compatible with the pipeline.
    for x, y, z, _score in centers:
        vox = _disk_voxels(x, y, radius, width=width, height=height)
        mask3d[vox[:, 0], vox[:, 1], z] = 1

        # Store ROI voxels per slice (AI-compatible)
        np.save(
            os.path.join(roi_data_dir, f"ROI_voxels_slice_{z+1}.npy"),
            vox,
        )

        # Peak frame for the center voxel (used downstream as reference)
        tc = dce4d[x, y, z, :]
        baseline = tc[: settings.ROI_DCE_BASELINE_FRAMES].mean() if tc.size else 0.0
        peak_frame = int(np.argmax(tc - baseline)) if tc.size else 0
        np.save(
            os.path.join(frame_data_dir, f"frame_index_slice_{z+1}.npy"),
            peak_frame,
        )

        plot_time_intensity_curves_AI(
            dce4d,
            vox,
            z,
            peak_frame,
            time_points_s,
            analysis_dir,
            image_dir,
            type=roi_type,
            subtype=roi_subtype,
        )
        plot_time_intensity_curves_and_CTC_AI(
            dce4d,
            peak_frame,
            vox,
            z,
            peak_frame,
            time_points_s,
            analysis_dir,
            image_dir,
            nifti_dir,
            type=roi_type,
            subtype=roi_subtype,
            IsVFA=is_vfa,
            filenames=filenames,
        )

    out_dir = os.path.join(analysis_dir, "ROI NIfTI")
    os.makedirs(out_dir, exist_ok=True)
    safe_type = roi_type.replace(" ", "_")
    safe_subtype = roi_subtype.replace(" ", "_")
    out_path = os.path.join(out_dir, f"{safe_type}__{safe_subtype}__mask.nii.gz")

    out_img = nib.Nifti1Image(mask3d.astype(np.uint8), affine=ref_img.affine, header=ref_img.header)
    nib.save(out_img, out_path)


def _save_radius_overlay(
    *,
    dce4d: np.ndarray,
    centers_by_label: dict[str, list[tuple[int, int, int, float]]],
    radii_by_label: dict[str, int],
    image_dir: str,
):
    mean_img = dce4d.mean(axis=-1)

    panels: list[tuple[str, tuple[int, int, int, float]]] = []
    for label, centers in centers_by_label.items():
        for c in centers:
            panels.append((label, c))

    if not panels:
        return

    cols = min(4, len(panels))
    rows = int(np.ceil(len(panels) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, (label, (x, y, z, _score)) in enumerate(panels):
        r = radii_by_label[label]
        ax = axes[idx // cols, idx % cols]
        ax.imshow(mean_img[:, :, z], cmap="gray", origin="lower")
        circ = plt.Circle((y, x), r, fill=False, color="red", linewidth=2)
        ax.add_patch(circ)
        ax.set_title(f"{label} z={z+1} r={r}px")
        ax.axis("off")

    # Hide unused axes
    for idx in range(len(panels), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.join(image_dir, "AI"), exist_ok=True)
    plt.savefig(os.path.join(image_dir, "AI", "DCE_geometry_roi_radii.png"), dpi=300)
    plt.close(fig)


def input_function_geometry(analysis_directory, nifti_directory, image_directory, filenames, parameters):
    (
        _t1_3D_filename,
        _axial_t1_3D_filename,
        _t2_3D_filename,
        _axial_t2_3D_filename,
        _flair_3D_filename,
        _axial_flair_3D_filename,
        _axial_t2_2D_filename,
        _diffusion_filename,
        dce_filename,
    ) = filenames

    is_vfa, _is_ir, _apple_metal, _boundary, *_rest = parameters

    dce_path = os.path.join(nifti_directory, dce_filename)
    ref_img = nib.load(dce_path)
    dce4d = np.rot90(ref_img.get_fdata(), k=-1, axes=(0, 1))

    tr_s = float(ref_img.header.get_zooms()[-1])
    n_volumes = int(dce4d.shape[-1])
    total_scan_duration = tr_s * n_volumes
    time_points_s = np.linspace(0, total_scan_duration, n_volumes)
    os.makedirs(os.path.join(analysis_directory, "Fitting"), exist_ok=True)
    np.save(os.path.join(analysis_directory, "Fitting", "time_points_s.npy"), time_points_s)

    cfg = GeometryRoiConfig(
        rica_radius=settings.ROI_RICA_RADIUS,
        lica_radius=settings.ROI_LICA_RADIUS,
        sss_radius=settings.ROI_SSS_RADIUS,
        rica_slices=settings.ROI_RICA_SLICES,
        lica_slices=settings.ROI_LICA_SLICES,
        sss_slices=settings.ROI_SSS_SLICES,
        rica_z_range=settings.ROI_RICA_Z_RANGE,
        lica_z_range=settings.ROI_LICA_Z_RANGE,
        sss_z_range=settings.ROI_SSS_Z_RANGE,
        sss_midline_band=settings.ROI_SSS_MIDLINE_BAND,
        baseline_frames=settings.ROI_DCE_BASELINE_FRAMES,
    )

    height, width, n_slices = dce4d.shape[0], dce4d.shape[1], dce4d.shape[2]
    mid_y = width // 2

    # Dynamic defaults: scale slice-count expectations to volume depth.
    # For classic 10-slice 2D DCE, scale==1. For thicker 3D volumes the
    # slice counts expand proportionally.
    scale = max(1, int(round(n_slices / 10)))
    rica_slices_eff = max(1, int(cfg.rica_slices) * scale)
    lica_slices_eff = max(1, int(cfg.lica_slices) * scale)
    sss_slices_eff = max(1, int(cfg.sss_slices) * scale)

    # Default z search windows (inclusive), expressed as fractions of z depth.
    # Empirically this is more robust than a hard "mid-slice" split:
    # - ICA: superior ~30% of slices
    # - SSS: mid ~30-70% of slices
    # These match typical r/l ICA low-z and SSS mid-z selections in 10-slice DCE.
    z_ica_end = max(0, min(n_slices - 1, int(np.floor(0.3 * n_slices)) - 1))
    z_sss_start = max(0, min(n_slices - 1, int(np.floor(0.3 * n_slices))))
    z_sss_end = max(z_sss_start, min(n_slices - 1, int(np.floor(0.7 * n_slices)) - 1))

    default_rica = (0, z_ica_end)
    default_lica = (0, z_ica_end)
    default_sss = (z_sss_start, z_sss_end)

    rica_z = _parse_z_range(cfg.rica_z_range, n_slices, default_rica)
    lica_z = _parse_z_range(cfg.lica_z_range, n_slices, default_lica)
    sss_z = _parse_z_range(cfg.sss_z_range, n_slices, default_sss)

    brain_mask = _brain_mask_from_mean(dce4d)
    peak_map, ttp_map = _compute_peak_and_ttp(dce4d, cfg.baseline_frames)

    # Mild time-to-peak prior for arteries: prefer voxels near the global bolus mode,
    # which helps avoid late venous structures and very-early noise spikes.
    ttp_target = _robust_ttp_target(peak_map, ttp_map, brain_mask)
    beta = 0.25
    artery_score = peak_map / (1.0 + beta * np.abs(ttp_map.astype(np.float32) - float(ttp_target)))

    allowed_rica = brain_mask.copy()
    allowed_rica[:, :mid_y, :] = False

    allowed_lica = brain_mask.copy()
    allowed_lica[:, mid_y:, :] = False

    allowed_sss = brain_mask.copy()
    band = max(1, int(cfg.sss_midline_band))
    y0 = max(0, mid_y - band)
    y1 = min(width, mid_y + band)
    allowed_sss[:, :y0, :] = False
    allowed_sss[:, y1:, :] = False

    # Explicitly exclude the SSS midline band from ICA candidates.
    # This prevents the ICA selector from snapping onto SSS/transverse sinus endings.
    allowed_rica[:, : (mid_y + band), :] = False
    allowed_lica[:, (mid_y - band) :, :] = False

    rica_centers = _best_centers_by_slice(artery_score, allowed_rica, rica_z, rica_slices_eff)
    lica_centers = _best_centers_by_slice(artery_score, allowed_lica, lica_z, lica_slices_eff)
    sss_centers = _best_centers_by_slice(peak_map, allowed_sss, sss_z, sss_slices_eff)

    # Persist outputs in the same on-disk format as the existing AI pipeline.
    _save_roi_outputs(
        roi_type="Artery",
        roi_subtype="Right Interior Carotid",
        centers=rica_centers,
        radius=cfg.rica_radius,
        dce4d=dce4d,
        ref_img=ref_img,
        analysis_dir=analysis_directory,
        image_dir=image_directory,
        nifti_dir=nifti_directory,
        time_points_s=time_points_s,
        filenames=filenames,
        is_vfa=bool(is_vfa),
    )

    _save_roi_outputs(
        roi_type="Artery",
        roi_subtype="Left Interior Carotid",
        centers=lica_centers,
        radius=cfg.lica_radius,
        dce4d=dce4d,
        ref_img=ref_img,
        analysis_dir=analysis_directory,
        image_dir=image_directory,
        nifti_dir=nifti_directory,
        time_points_s=time_points_s,
        filenames=filenames,
        is_vfa=bool(is_vfa),
    )

    _save_roi_outputs(
        roi_type="Vein",
        roi_subtype="Sinus Sagittalis",
        centers=sss_centers,
        radius=cfg.sss_radius,
        dce4d=dce4d,
        ref_img=ref_img,
        analysis_dir=analysis_directory,
        image_dir=image_directory,
        nifti_dir=nifti_directory,
        time_points_s=time_points_s,
        filenames=filenames,
        is_vfa=bool(is_vfa),
    )

    _save_radius_overlay(
        dce4d=dce4d,
        centers_by_label={
            "rICA": rica_centers,
            "lICA": lica_centers,
            "SSS": sss_centers,
        },
        radii_by_label={
            "rICA": int(cfg.rica_radius),
            "lICA": int(cfg.lica_radius),
            "SSS": int(cfg.sss_radius),
        },
        image_dir=image_directory,
    )

    # Return simple metadata for debugging (not used by the existing pipeline).
    return {
        "rICA": rica_centers,
        "lICA": lica_centers,
        "SSS": sss_centers,
        "radii": {
            "rICA": int(cfg.rica_radius),
            "lICA": int(cfg.lica_radius),
            "SSS": int(cfg.sss_radius),
        },
        "z_ranges": {
            "rICA": rica_z,
            "lICA": lica_z,
            "SSS": sss_z,
        },
        "ttp_target": int(ttp_target),
    }
