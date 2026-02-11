import os
import warnings
from dataclasses import dataclass

import nibabel as nib
from nibabel import orientations as nio
import numpy as np
import matplotlib.pyplot as plt

try:
    from nibabel.processing import resample_from_to
except Exception:  # pragma: no cover
    resample_from_to = None

from scipy import ndimage

import utils.settings as settings
import modules.time_shifting as tscc
from utils.mapping import choice2type
from utils.loading import (
    build_time_points_s,
    resolve_dce_time_step_s,
    resolve_flip_angle_deg,
    resolve_turboflash_tr_s,
    load_dce_4d,
)
from utils.plotting import turboflash, plot_time_intensity_curves_AI, plot_time_intensity_curves_and_CTC_AI


@dataclass(frozen=True)
class GeometryRoiConfig:
    rica_slices: int
    lica_slices: int
    sss_slices: int
    rica_z_range: str
    lica_z_range: str
    sss_z_range: str
    sss_midline_band: int
    baseline_frames: int


def _getenv_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = str(raw).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


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


def _default_z_bands(n_slices: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return default (rICA, lICA, SSS) z-ranges.

    Heuristic: ICA typically lives in the superior third; SSS below mid-slices.
    """

    if n_slices <= 1:
        return (0, 0), (0, 0), (0, 0)

    hi = n_slices - 1
    rica_end = int(np.floor(hi * 0.30))
    rica_end = max(0, min(hi, rica_end))

    sss_start = int(np.floor(hi * 0.35))
    sss_end = int(np.floor(hi * 0.70))
    sss_start = max(0, min(hi, sss_start))
    sss_end = max(0, min(hi, sss_end))
    if sss_end < sss_start:
        sss_start, sss_end = sss_end, sss_start

    return (0, rica_end), (0, rica_end), (sss_start, sss_end)


def _compute_peak_map(dce4d: np.ndarray, baseline_frames: int) -> np.ndarray:
    baseline_frames = max(1, int(baseline_frames))
    baseline = dce4d[..., :baseline_frames].mean(axis=-1)
    peak = dce4d.max(axis=-1)
    return (peak - baseline).astype(np.float32)


def _compute_ttp_map(dce4d: np.ndarray, baseline_frames: int) -> np.ndarray:
    """Return time-to-peak index map (frame indices)."""

    baseline_frames = max(1, int(baseline_frames))
    baseline = dce4d[..., :baseline_frames].mean(axis=-1, keepdims=True)
    signal = dce4d - baseline
    signal[..., :baseline_frames] = -np.inf
    return np.argmax(signal, axis=-1).astype(np.int32)


def _derive_ttp_targets(
    peak_amp: np.ndarray,
    ttp_idx: np.ndarray,
    allowed_mask3d: np.ndarray,
    baseline_frames: int,
) -> tuple[int, int, int]:
    """Return (artery_target, vein_target, sigma_frames)."""

    baseline_frames = max(1, int(baseline_frames))
    vals = peak_amp[allowed_mask3d]
    if vals.size < 64:
        artery_t = baseline_frames + 5
        vein_t = baseline_frames + 20
        sigma = 6
        return artery_t, vein_t, sigma

    thr = float(np.percentile(vals, 99.5))
    strong = allowed_mask3d & (peak_amp >= thr)
    ttps = ttp_idx[strong]
    if ttps.size < 16:
        ttps = ttp_idx[allowed_mask3d]
    if ttps.size < 16:
        artery_t = baseline_frames + 5
        vein_t = baseline_frames + 20
        sigma = 6
        return artery_t, vein_t, sigma

    artery_t = int(np.percentile(ttps, 15))
    vein_t = int(np.percentile(ttps, 85))
    artery_t = max(baseline_frames, artery_t)
    vein_t = max(artery_t + 1, vein_t)
    sigma = max(4, int(0.03 * peak_amp.shape[-1]) if peak_amp.ndim == 4 else 6)
    # note: peak_amp is 3D, so above branch is mainly defensive
    sigma = 6 if sigma <= 0 else sigma
    return artery_t, vein_t, sigma


def _gaussian_weight(idx: np.ndarray, target: int, sigma: int) -> np.ndarray:
    sigma = max(1, int(sigma))
    return np.exp(-0.5 * ((idx.astype(np.float32) - float(target)) / float(sigma)) ** 2).astype(
        np.float32
    )


def _ap_prior_weights(height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (anterior_weight, posterior_weight) broadcastable to (x,y,z).

    Assumes axis-0 (rows) runs roughly anterior (top) -> posterior (bottom).
    Set P_BRAIN_ROI_AP_FLIP=1 to invert if the dataset is flipped.
    """

    height = max(1, int(height))
    x = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    if _getenv_bool("P_BRAIN_ROI_AP_FLIP", default=False):
        x = 1.0 - x
    gamma = float(os.getenv("P_BRAIN_ROI_AP_GAMMA", "1.5") or "1.5")
    gamma = 1.0 if not np.isfinite(gamma) or gamma <= 0 else gamma
    anterior = np.power(1.0 - x, gamma, dtype=np.float32)
    posterior = np.power(x, gamma, dtype=np.float32)
    return anterior.astype(np.float32), posterior.astype(np.float32)


def _logistic_split_weight(ttp_idx: np.ndarray, t_mid: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft split: returns (early_weight, late_weight) in [0,1]."""

    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = 2.0
    z = (ttp_idx.astype(np.float32) - float(t_mid)) / float(scale)
    early = 1.0 / (1.0 + np.exp(z))
    late = 1.0 - early
    return early.astype(np.float32), late.astype(np.float32)


def _infer_lr_ap_axes_from_affine(affine: np.ndarray) -> dict:
    """Infer which voxel axes correspond to LR and AP from the NIfTI affine.

    Returns a dict with keys:
      - lr_axis: int | None
      - lr_code: 'L'|'R'|None
      - ap_axis: int | None
      - ap_code: 'A'|'P'|None
    """

    try:
        ax = nio.aff2axcodes(affine)
    except Exception:
        ax = (None, None, None)

    def _find(c1: str, c2: str) -> tuple[int | None, str | None]:
        for i, c in enumerate(ax):
            if c in {c1, c2}:
                return i, c
        return None, None

    lr_axis, lr_code = _find("L", "R")
    ap_axis, ap_code = _find("A", "P")
    return {
        "lr_axis": lr_axis,
        "lr_code": lr_code,
        "ap_axis": ap_axis,
        "ap_code": ap_code,
    }


def _split_lr_masks(mask3d: np.ndarray, *, lr_axis: int, lr_code: str | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (right_mask, left_mask, mid_index) split by LR axis."""

    lr_axis = int(lr_axis)
    mid = int(mask3d.shape[lr_axis] // 2)

    # Determine which side of the index axis corresponds to anatomical right.
    # If axis increases toward 'R', right side is indices >= mid; if toward 'L', it's indices < mid.
    right_is_high = (lr_code == "R")

    right = mask3d.copy()
    left = mask3d.copy()
    if lr_axis == 0:
        if right_is_high:
            right[:mid, :, :] = False
            left[mid:, :, :] = False
        else:
            right[mid:, :, :] = False
            left[:mid, :, :] = False
    elif lr_axis == 1:
        if right_is_high:
            right[:, :mid, :] = False
            left[:, mid:, :] = False
        else:
            right[:, mid:, :] = False
            left[:, :mid, :] = False
    else:
        # LR axis not in-plane; fallback split on axis 1.
        mid = int(mask3d.shape[1] // 2)
        right[:, :mid, :] = False
        left[:, mid:, :] = False
    return right, left, mid


def _midline_band_mask(shape3d: tuple[int, int, int], *, lr_axis: int, mid: int, band: int) -> np.ndarray:
    band = max(1, int(band))
    m = np.zeros(shape3d, dtype=bool)
    if lr_axis == 0:
        a0 = max(0, mid - band)
        a1 = min(shape3d[0], mid + band + 1)
        m[a0:a1, :, :] = True
    elif lr_axis == 1:
        a0 = max(0, mid - band)
        a1 = min(shape3d[1], mid + band + 1)
        m[:, a0:a1, :] = True
    else:
        # fallback: assume axis1 is LR
        a0 = max(0, mid - band)
        a1 = min(shape3d[1], mid + band + 1)
        m[:, a0:a1, :] = True
    return m


def _middle_slice_mask(shape3d: tuple[int, int, int]) -> np.ndarray:
    """Mask keeping only the middle fraction of slices along z.

    Controlled by P_BRAIN_ROI_MID_Z_FRAC (default 0.30).
    """

    xdim, ydim, zdim = (int(shape3d[0]), int(shape3d[1]), int(shape3d[2]))
    frac = float(os.getenv("P_BRAIN_ROI_MID_Z_FRAC", "0.30") or "0.30")
    frac = 0.30 if not np.isfinite(frac) else float(frac)
    frac = min(1.0, max(0.0, frac))
    if zdim <= 1 or frac >= 0.999:
        return np.ones((xdim, ydim, zdim), dtype=bool)
    half = 0.5 * frac
    z0 = int(np.floor((0.5 - half) * zdim))
    z1 = int(np.ceil((0.5 + half) * zdim)) - 1
    z0 = max(0, min(zdim - 1, z0))
    z1 = max(0, min(zdim - 1, z1))
    if z1 < z0:
        z0, z1 = z1, z0
    m = np.zeros((xdim, ydim, zdim), dtype=bool)
    m[:, :, z0 : z1 + 1] = True
    return m


def _pca_score_per_slice(dce4d: np.ndarray, mask3d: np.ndarray, baseline_frames: int) -> np.ndarray:
    """Compute a simple per-slice PCA 'vascularness' score.

    Returns a 3D array aligned with (x,y,z) where higher magnitude indicates
    stronger contribution to the dominant temporal mode.
    """

    baseline_frames = max(1, int(baseline_frames))
    xdim, ydim, zdim, tdim = dce4d.shape
    out = np.zeros((xdim, ydim, zdim), dtype=np.float32)

    for z in range(zdim):
        m = mask3d[:, :, z]
        if not np.any(m):
            continue
        vox = dce4d[:, :, z, :][m]
        if vox.shape[0] < 64:
            continue
        baseline = vox[:, :baseline_frames].mean(axis=1, keepdims=True)
        X = vox - baseline
        X[:, :baseline_frames] = 0.0
        X = X - X.mean(axis=0, keepdims=True)
        # SVD: X = U S Vt; voxel scores along PC1 are U[:,0]*S[0]
        try:
            U, S, _Vt = np.linalg.svd(X, full_matrices=False)
        except Exception:
            continue
        if S.size == 0:
            continue
        scores = (U[:, 0] * S[0]).astype(np.float32)
        # normalize robustly
        scale = float(np.percentile(np.abs(scores), 95)) if scores.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            continue
        scores = scores / scale
        tmp = np.zeros((xdim, ydim), dtype=np.float32)
        tmp[m] = np.abs(scores)
        out[:, :, z] = tmp
    return out


def _brain_mask_from_mean(dce4d: np.ndarray) -> np.ndarray:
    mean_img = dce4d.mean(axis=-1)
    thr = np.percentile(mean_img, 60)
    return mean_img > thr


def _segmentation_candidates_in_dce(nifti_directory: str) -> list[str]:
    return [
        os.path.join(
            nifti_directory,
            "segmentation",
            "segmentation",
            "mri",
            "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz",
        ),
        os.path.join(
            nifti_directory,
            "segmentation",
            "mri",
            "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz",
        ),
        os.path.join(nifti_directory, "segmentation", "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz"),
    ]


def _tissue_mask_candidates_in_dce(nifti_directory: str) -> list[str]:
    """Union of tissue masks in DCE space (WM/GM), used to constrain ICA search."""

    base = os.path.join(nifti_directory, "segmentation", "segmentation", "mri")
    return [
        os.path.join(base, "aparc.DKTatlas+aseg.deep_in_DCE_wm.nii.gz"),
        os.path.join(base, "aparc.DKTatlas+aseg.deep_in_DCE_gm.nii.gz"),
        os.path.join(base, "aparc.DKTatlas+aseg.deep_in_DCE_subcortical_gm.nii.gz"),
        os.path.join(base, "aparc.DKTatlas+aseg.deep_in_DCE_cortical_gm.nii.gz"),
        os.path.join(base, "wm.nii.gz"),
        os.path.join(base, "gm.nii.gz"),
    ]


def _load_union_mask(
    mask_paths: list[str], reference_img: nib.Nifti1Image
) -> np.ndarray | None:
    any_loaded = False
    union: np.ndarray | None = None
    for path in mask_paths:
        if not os.path.isfile(path):
            continue
        try:
            img = nib.load(path)
            data = np.asarray(img.get_fdata(), dtype=np.float32)
            if data.ndim != 3:
                continue
            if img.shape != reference_img.shape[:3] or not np.allclose(img.affine, reference_img.affine):
                if resample_from_to is None:
                    continue
                target = (reference_img.shape[:3], reference_img.affine)
                img = resample_from_to(img, target, order=0)
                data = np.asarray(img.get_fdata(), dtype=np.float32)
                if data.shape != reference_img.shape[:3]:
                    continue
            m = data > 0
            if union is None:
                union = m
            else:
                union = union | m
            any_loaded = True
        except Exception:
            continue
    if not any_loaded:
        return None
    return union


def _dilate_bool_mask(mask3d: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations == 0:
        return mask3d
    structure = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_dilation(mask3d, structure=structure, iterations=iterations)


def _sample_voxel_indices(mask3d: np.ndarray, scores: np.ndarray, max_voxels: int) -> np.ndarray:
    idx = np.argwhere(mask3d)
    if idx.size == 0:
        return idx
    max_voxels = int(max_voxels)
    if max_voxels <= 0 or idx.shape[0] <= max_voxels:
        return idx

    # Prefer high-score voxels for PCA.
    vals = scores[mask3d]
    if vals.size != idx.shape[0]:
        return idx[:max_voxels]

    k = min(max_voxels, idx.shape[0])
    pick = np.argpartition(vals, -k)[-k:]
    return idx[pick]


def _adaptive_percentile_mask(
    score: np.ndarray,
    mask: np.ndarray,
    *,
    start_pct: float,
    min_pct: float,
    step: float,
    min_vox: int,
) -> tuple[np.ndarray, float]:
    """Threshold `score` within `mask` at a percentile, relaxing until enough voxels."""

    start_pct = float(start_pct)
    min_pct = float(min_pct)
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        step = 0.5
    min_pct = min(start_pct, min_pct)
    min_vox = max(1, int(min_vox))

    vals = score[mask]
    if vals.size == 0:
        return (mask & np.isfinite(score) & (score > 0)), float(min_pct)

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (mask & np.isfinite(score) & (score > 0)), float(min_pct)

    p = float(start_pct)
    while p >= float(min_pct) - 1e-6:
        thr = float(np.percentile(vals, p))
        if not np.isfinite(thr):
            p -= step
            continue
        m = mask & np.isfinite(score) & (score >= thr)
        if int(m.sum()) >= min_vox:
            return m, float(p)
        p -= step

    thr = float(np.percentile(vals, float(min_pct)))
    if not np.isfinite(thr):
        return (mask & np.isfinite(score) & (score > 0)), float(min_pct)
    return (mask & np.isfinite(score) & (score >= thr)), float(min_pct)


def _principal_axis_alignment(
    coords_xyz: np.ndarray,
    *,
    spacing: tuple[float, float, float] | None = None,
) -> tuple[float, float, np.ndarray]:
    """Return (abs(z_alignment), elongation, principal_axis_unit_vector)."""

    if coords_xyz.ndim != 2 or coords_xyz.shape[1] != 3 or coords_xyz.shape[0] < 8:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return 1.0, 1.0, v

    c = coords_xyz.astype(np.float32)
    if spacing is not None:
        sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        if np.isfinite(sx) and np.isfinite(sy) and np.isfinite(sz) and sx > 0 and sy > 0 and sz > 0:
            c = c * np.array([sx, sy, sz], dtype=np.float32)[None, :]
    c = c - c.mean(axis=0, keepdims=True)
    cov = np.cov(c.T)
    try:
        w, V = np.linalg.eigh(cov)
    except Exception:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return 1.0, 1.0, v

    order = np.argsort(w)
    w = np.maximum(w[order], 1e-6)
    V = V[:, order]
    v = V[:, -1].astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    z_align = float(abs(v[2]))
    elong = float(w[-1] / w[0])
    return z_align, elong, v


def _iter_3d_components(mask3d: np.ndarray) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(3, 1)
    lab, nlab = ndimage.label(mask3d.astype(bool), structure=structure)
    return lab, int(nlab)


def _score_component_3d(
    *,
    coords: np.ndarray,
    score3d: np.ndarray,
    dist_inside: np.ndarray | None,
    spacing: tuple[float, float, float] | None = None,
) -> dict:
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    total = float(score3d[xs, ys, zs].sum())
    z_span = int(zs.max() - zs.min()) if zs.size else 0
    z_count = int(np.unique(zs).size)
    cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())
    z_align, elong, axis = _principal_axis_alignment(np.stack([xs, ys, zs], axis=1), spacing=spacing)

    boundary_contact = 0.0
    median_dist = 0.0
    if dist_inside is not None:
        d = dist_inside[xs, ys, zs]
        if d.size:
            boundary_contact = float(np.count_nonzero(d <= 1.0)) / float(max(1, d.size))
            median_dist = float(np.median(d))

    return {
        "total": total,
        "n": int(coords.shape[0]),
        "centroid": (cx, cy, cz),
        "z_span": z_span,
        "z_count": z_count,
        "z_align": float(z_align),
        "elong": float(elong),
        "axis": axis,
        "boundary_contact": float(boundary_contact),
        "median_dist": float(median_dist),
    }


def _pca_global_scores(
    dce4d: np.ndarray,
    mask3d: np.ndarray,
    peak_amp: np.ndarray,
    baseline_frames: int,
    max_voxels: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (pc_timecourses, voxel_score_maps).

    - pc_timecourses: (k, t)
    - voxel_score_maps: (k, x, y, z)
    """

    baseline_frames = max(1, int(baseline_frames))
    n_components = max(1, int(n_components))
    max_voxels = max(256, int(max_voxels))

    pts = _sample_voxel_indices(mask3d, peak_amp, max_voxels=max_voxels)
    if pts.shape[0] < 512:
        return None

    # Extract and normalize timecourses.
    X = dce4d[pts[:, 0], pts[:, 1], pts[:, 2], :].astype(np.float32)
    base = X[:, :baseline_frames].mean(axis=1, keepdims=True)
    X = X - base
    X[:, :baseline_frames] = 0.0
    X = X - X.mean(axis=0, keepdims=True)

    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except Exception:
        return None
    if S.size == 0:
        return None

    k = min(n_components, Vt.shape[0])
    pc_time = Vt[:k, :].astype(np.float32)

    # Voxel scores along PCs.
    scores = (U[:, :k] * S[:k]).astype(np.float32)  # (n_vox, k)

    # Map back to volume.
    xdim, ydim, zdim = mask3d.shape
    maps = np.zeros((k, xdim, ydim, zdim), dtype=np.float32)
    for i in range(k):
        tmp = np.zeros((xdim, ydim, zdim), dtype=np.float32)
        tmp[pts[:, 0], pts[:, 1], pts[:, 2]] = np.abs(scores[:, i])
        # robust normalize
        m = tmp[mask3d]
        scale = float(np.percentile(m, 99)) if m.size else 1.0
        if np.isfinite(scale) and scale > 0:
            tmp = tmp / scale
        maps[i] = tmp

    return pc_time, maps


def _pick_pc_by_peak(pc_time: np.ndarray) -> tuple[int, int]:
    """Return (early_pc_index, late_pc_index) based on time-to-peak of PCs."""

    peaks = []
    for i in range(pc_time.shape[0]):
        v = pc_time[i]
        p = int(np.argmax(np.abs(v)))
        peaks.append(p)
    early = int(np.argmin(peaks))
    late = int(np.argmax(peaks))
    return early, late


def _select_sss_centers_from_components(
    *,
    score3d: np.ndarray,
    allowed3d: np.ndarray,
    z_range: tuple[int, int],
    k_slices: int,
    lr_axis_inplane: int,
    mid_lr: int,
    band: int,
    dist_inside: np.ndarray,
    ap_axis_inplane: int | None,
    ap_code: str | None,
) -> list[tuple[int, int, int, float]]:
    """Choose SSS centers by connected components to avoid transverse sinus."""

    z0, z1 = z_range
    results: list[tuple[int, int, int, float]] = []
    k_slices = max(1, int(k_slices))

    for z in range(z0, z1 + 1):
        allowed = allowed3d[:, :, z]
        if not np.any(allowed):
            continue
        s = np.where(allowed, score3d[:, :, z], 0.0)
        vals = s[allowed]
        if vals.size < 32:
            continue
        thr = float(np.percentile(vals, 99.5))
        cand = allowed & (s >= thr)
        if not np.any(cand):
            continue

        lab, nlab = ndimage.label(cand)
        if nlab <= 0:
            continue

        best = None
        best_score = -np.inf

        # Precompute an in-slice boundary mask: voxels right at the brain boundary.
        boundary = dist_inside[:, :, z] <= 1.0

        for li in range(1, nlab + 1):
            comp = lab == li
            if comp.sum() < 8:
                continue

            comp_n = int(comp.sum())
            max_area = int(os.getenv("P_BRAIN_SSS_MAX_AREA", "220") or "220")
            if max_area > 0 and comp_n > max_area:
                # Big blobs/rims are not SSS cross-sections.
                continue
            comp_scores = s[comp]
            total = float(comp_scores.sum())
            coords = np.argwhere(comp)
            cx = float(coords[:, 0].mean())
            cy = float(coords[:, 1].mean())
            # Distance to midline in the LR direction (in-plane axis).
            lr_cent = cy if int(lr_axis_inplane) == 1 else cx
            dist_mid = abs(lr_cent - float(mid_lr))
            if dist_mid > max(3.0, float(band) * 1.2):
                continue

            # Reject deep intracranial dots: SSS should be close to the skull.
            dvals = dist_inside[:, :, z][comp]
            if dvals.size:
                max_inner = float(os.getenv("P_BRAIN_SSS_MAX_DIST_INNER", "5.0") or "5.0")
                if np.isfinite(max_inner) and float(np.median(dvals)) > max_inner:
                    continue

            # Reject transverse-sinus-like rims: too much boundary contact.
            if boundary is not None:
                contact = float(np.count_nonzero(comp & boundary)) / float(max(1, comp_n))
                max_contact = float(os.getenv("P_BRAIN_SSS_MAX_BOUNDARY_CONTACT", "0.55") or "0.55")
                if np.isfinite(max_contact) and contact > max_contact:
                    continue

            # Reject components spanning too wide laterally (LR span).
            lr_vals = coords[:, 1] if int(lr_axis_inplane) == 1 else coords[:, 0]
            lr_span = int(lr_vals.max() - lr_vals.min()) if lr_vals.size else 0
            max_lr_span = int(os.getenv("P_BRAIN_SSS_MAX_LR_SPAN", "18") or "18")
            if max_lr_span > 0 and lr_span > max_lr_span:
                continue

            # Elongation via covariance eigenvalues (prefer roundish cross-sections).
            cov = np.cov(coords.T)
            try:
                w = np.linalg.eigvalsh(cov)
            except Exception:
                w = np.array([1.0, 1.0])
            w = np.sort(np.maximum(w, 1e-6))
            elong = float(w[-1] / w[0])
            elong = min(elong, 50.0)

            max_elong = float(os.getenv("P_BRAIN_SSS_MAX_ELONG", "6.0") or "6.0")
            if np.isfinite(max_elong) and elong > max_elong:
                continue

            # Posterior preference (helps pick the posterior branch when SSS splits).
            post_bonus = 0.0
            if ap_axis_inplane is not None and ap_code in {"A", "P"}:
                ap_cent = cy if int(ap_axis_inplane) == 1 else cx
                ap_dim = float(score3d.shape[int(ap_axis_inplane)])
                if ap_dim > 1:
                    a = float(ap_cent) / (ap_dim - 1.0)
                    # Make p in [0,1] where 1 is posterior.
                    p = a if ap_code == "P" else (1.0 - a)
                    post_bonus = float(os.getenv("P_BRAIN_SSS_POSTERIOR_BONUS", "0.5") or "0.5") * p

            # Penalize midline offset and elongation.
            elong_pen = 1.0 + float(os.getenv("P_BRAIN_SSS_ELONG_PEN", "1.0") or "1.0") * max(0.0, elong - 1.0)
            penalty = 1.0 + (dist_mid / max(1.0, float(band)))
            comp_score = (total * (1.0 + post_bonus)) / (penalty * elong_pen)
            if comp_score > best_score:
                best_score = comp_score
                best = (int(round(cx)), int(round(cy)))

        if best is None or not np.isfinite(best_score):
            continue
        results.append((best[0], best[1], int(z), float(best_score)))

    results.sort(key=lambda t: t[3], reverse=True)
    return results[: min(k_slices, len(results))]


def _select_sss_roi_voxels_by_slice(
    *,
    score3d: np.ndarray,
    allowed3d: np.ndarray,
    z_range: tuple[int, int],
    k_slices: int,
    lr_axis_inplane: int,
    mid_lr: int,
    band: int,
    dist_inside: np.ndarray,
    ap_axis_inplane: int | None,
    ap_code: str | None,
) -> list[tuple[int, np.ndarray, float]]:
    """Choose SSS ROI regions by connected components (returns per-slice voxels)."""

    z0, z1 = z_range
    out: list[tuple[int, np.ndarray, float]] = []
    k_slices = max(1, int(k_slices))

    for z in range(z0, z1 + 1):
        allowed = allowed3d[:, :, z]
        if not np.any(allowed):
            continue
        s2 = np.where(allowed, score3d[:, :, z], 0.0)
        vals = s2[allowed]
        if vals.size < 32:
            continue

        thr = float(np.percentile(vals, 99.5))
        cand = allowed & (s2 >= thr)
        if not np.any(cand):
            continue

        lab, nlab = ndimage.label(cand)
        if nlab <= 0:
            continue

        boundary = dist_inside[:, :, z] <= 1.0
        best_vox = None
        best_score = -np.inf

        for li in range(1, nlab + 1):
            comp = lab == li
            comp_n = int(comp.sum())
            if comp_n < 8:
                continue

            max_area = int(os.getenv("P_BRAIN_SSS_MAX_AREA", "220") or "220")
            if max_area > 0 and comp_n > max_area:
                continue

            coords = np.argwhere(comp)
            cx = float(coords[:, 0].mean())
            cy = float(coords[:, 1].mean())

            lr_cent = cy if int(lr_axis_inplane) == 1 else cx
            dist_mid = abs(lr_cent - float(mid_lr))
            if dist_mid > max(3.0, float(band) * 1.2):
                continue

            # Prefer close-to-skull components; reject deep dots.
            dvals = dist_inside[:, :, z][comp]
            if dvals.size:
                max_inner = float(os.getenv("P_BRAIN_SSS_MAX_DIST_INNER", "5.0") or "5.0")
                if np.isfinite(max_inner) and float(np.median(dvals)) > max_inner:
                    continue

            # Boundary-contact heuristics: reject rim-like arcs; optionally require some contact.
            contact = float(np.count_nonzero(comp & boundary)) / float(max(1, comp_n))
            max_contact = float(os.getenv("P_BRAIN_SSS_MAX_BOUNDARY_CONTACT", "0.55") or "0.55")
            if np.isfinite(max_contact) and contact > max_contact:
                continue
            min_contact = float(os.getenv("P_BRAIN_SSS_MIN_BOUNDARY_CONTACT", "0.00") or "0.00")
            if np.isfinite(min_contact) and min_contact > 0 and contact < min_contact:
                continue

            # Reject components spanning too wide laterally (LR span).
            lr_vals = coords[:, 1] if int(lr_axis_inplane) == 1 else coords[:, 0]
            lr_span = int(lr_vals.max() - lr_vals.min()) if lr_vals.size else 0
            max_lr_span = int(os.getenv("P_BRAIN_SSS_MAX_LR_SPAN", "18") or "18")
            if max_lr_span > 0 and lr_span > max_lr_span:
                continue

            # Elongation (prefer roundish cross-sections).
            cov = np.cov(coords.T)
            try:
                w = np.linalg.eigvalsh(cov)
            except Exception:
                w = np.array([1.0, 1.0])
            w = np.sort(np.maximum(w, 1e-6))
            elong = float(w[-1] / w[0])
            elong = min(elong, 50.0)
            max_elong = float(os.getenv("P_BRAIN_SSS_MAX_ELONG", "6.0") or "6.0")
            if np.isfinite(max_elong) and elong > max_elong:
                continue

            # Posterior preference (helps pick posterior branch when SSS splits).
            post_bonus = 0.0
            if ap_axis_inplane is not None and ap_code in {"A", "P"}:
                ap_cent = cy if int(ap_axis_inplane) == 1 else cx
                ap_dim = float(score3d.shape[int(ap_axis_inplane)])
                if ap_dim > 1:
                    a = float(ap_cent) / (ap_dim - 1.0)
                    p = a if ap_code == "P" else (1.0 - a)
                    post_bonus = float(os.getenv("P_BRAIN_SSS_POSTERIOR_BONUS", "0.5") or "0.5") * p

            comp_scores = s2[comp]
            total = float(comp_scores.sum())
            elong_pen = 1.0 + float(os.getenv("P_BRAIN_SSS_ELONG_PEN", "1.0") or "1.0") * max(0.0, elong - 1.0)
            penalty = 1.0 + (dist_mid / max(1.0, float(band)))
            comp_score = (total * (1.0 + post_bonus)) / (penalty * elong_pen)

            if comp_score > best_score:
                best_score = comp_score
                best_vox = coords.astype(int)

        if best_vox is None or not np.isfinite(best_score):
            continue
        out.append((int(z), best_vox, float(best_score)))

    out.sort(key=lambda t: t[2], reverse=True)
    return out[: min(k_slices, len(out))]


def _select_basilar_roi_voxels_by_slice(
    *,
    score3d: np.ndarray,
    allowed3d: np.ndarray,
    z_range: tuple[int, int],
    k_slices: int,
    dist_inside: np.ndarray | None = None,
    min_dist_inside: float | None = None,
) -> list[tuple[int, np.ndarray, float]]:
    """Select basilar artery ROI regions from a midline corridor (per-slice voxels)."""

    z0, z1 = z_range
    out: list[tuple[int, np.ndarray, float]] = []
    k_slices = max(1, int(k_slices))

    pct = float(os.getenv("P_BRAIN_ROI_BASILAR_PCT", "99.0") or "99.0")
    min_pct = float(os.getenv("P_BRAIN_ROI_BASILAR_MIN_PCT", "97.0") or "97.0")
    step = float(os.getenv("P_BRAIN_ROI_BASILAR_PCT_STEP", "0.5") or "0.5")
    step = 0.5 if (not np.isfinite(step) or step <= 0) else step
    min_area = int(os.getenv("P_BRAIN_ROI_BASILAR_MIN_AREA", "6") or "6")
    max_area = int(os.getenv("P_BRAIN_ROI_BASILAR_MAX_AREA", "400") or "400")

    for z in range(z0, z1 + 1):
        allowed = allowed3d[:, :, z]
        if not np.any(allowed):
            continue
        s2 = np.where(allowed, score3d[:, :, z], 0.0)
        vals = s2[allowed]
        if vals.size < 64:
            continue

        cand = None
        p = float(pct)
        while p >= float(min_pct) - 1e-6:
            thr = float(np.percentile(vals, p))
            cand = allowed & (s2 >= thr)
            if cand is not None and int(cand.sum()) >= int(min_area):
                break
            p -= float(step)
        if cand is None or not np.any(cand):
            continue

        lab, nlab = ndimage.label(cand)
        if nlab <= 0:
            continue

        best_vox = None
        best_score = -np.inf
        for li in range(1, nlab + 1):
            comp = lab == li
            n = int(comp.sum())
            if n < int(min_area) or n > int(max_area):
                continue
            if min_dist_inside is not None and dist_inside is not None:
                d = dist_inside[:, :, z][comp]
                if d.size and float(np.median(d)) < float(min_dist_inside):
                    continue
            total = float(s2[comp].sum())
            if total > best_score:
                best_score = total
                best_vox = np.argwhere(comp).astype(int)

        if best_vox is None or not np.isfinite(best_score):
            continue
        out.append((int(z), best_vox, float(best_score)))

    out.sort(key=lambda t: t[2], reverse=True)
    return out[: min(k_slices, len(out))]


def _load_segmentation_brain_mask(
    nifti_directory: str, reference_img: nib.Nifti1Image
) -> np.ndarray | None:
    """Load atlas segmentation aligned to DCE and return a boolean brain mask."""

    for path in _segmentation_candidates_in_dce(nifti_directory):
        if not os.path.isfile(path):
            continue
        try:
            seg_img = nib.load(path)
            seg_data = np.asarray(seg_img.get_fdata(), dtype=np.float32)
            if seg_data.ndim != 3:
                continue

            if seg_img.shape != reference_img.shape[:3] or not np.allclose(
                seg_img.affine, reference_img.affine
            ):
                if resample_from_to is None:
                    continue
                target = (reference_img.shape[:3], reference_img.affine)
                seg_img = resample_from_to(seg_img, target, order=0)
                seg_data = np.asarray(seg_img.get_fdata(), dtype=np.float32)
                if seg_data.shape != reference_img.shape[:3]:
                    continue

            return seg_data != 0
        except Exception:
            continue

    return None


def _load_fitting_t1_m0_maps(
    analysis_directory: str, reference_img: nib.Nifti1Image
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load `t1_map.nii.gz` and `m0_map.nii.gz` from Analysis/Fitting.

    Returns (t1_ms, m0) as float32 arrays in DCE space.
    """

    fitting_dir = os.path.join(analysis_directory, "Fitting")
    t1_path = os.path.join(fitting_dir, "t1_map.nii.gz")
    m0_path = os.path.join(fitting_dir, "m0_map.nii.gz")
    if not os.path.isfile(t1_path) or not os.path.isfile(m0_path):
        return None

    try:
        t1_img = nib.load(t1_path)
        m0_img = nib.load(m0_path)
    except Exception:
        return None

    if t1_img.shape != reference_img.shape[:3] or not np.allclose(t1_img.affine, reference_img.affine):
        if resample_from_to is None:
            return None
        target = (reference_img.shape[:3], reference_img.affine)
        try:
            t1_img = resample_from_to(t1_img, target, order=1)
            m0_img = resample_from_to(m0_img, target, order=1)
        except Exception:
            return None

    try:
        t1 = np.asarray(t1_img.get_fdata(), dtype=np.float32)
        m0 = np.asarray(m0_img.get_fdata(), dtype=np.float32)
    except Exception:
        return None

    if t1.ndim != 3 or m0.ndim != 3:
        return None
    if t1.shape != reference_img.shape[:3] or m0.shape != reference_img.shape[:3]:
        return None

    return t1, m0


def _ctc_model_params(dce_path: str) -> dict:
    """Return kwargs for `compute_CTC` based on settings + DCE sidecar."""

    # Validator parity: TurboFLASH is the only supported conversion.
    ctc_model = (getattr(settings, "CTC_MODEL", "turboflash") or "turboflash").strip().lower()
    if ctc_model in {"advanced", "method4", "validated_method4"}:
        ctc_model = "turboflash"
    if ctc_model != "turboflash":
        raise ValueError(
            f"Unsupported CTC_MODEL={ctc_model!r}. Validator parity requires 'turboflash'."
        )

    flip_angle_deg = resolve_flip_angle_deg(dce_path, default=None)
    # The validated closed-form TurboFLASH conversion does not require TR/nph.
    tr_s = None
    nph = None

    return {
        "ctc_model": ctc_model,
        "flip_angle_deg": flip_angle_deg,
        "tr_s": tr_s,
        "nph": nph,
    }


def _write_brain_concentration_nifti(
    *,
    dce4d: np.ndarray,
    ref_img: nib.Nifti1Image,
    dce_path: str,
    analysis_directory: str,
    brain_mask: np.ndarray,
    baseline_frames: int,
    output_path: str,
    batch_voxels: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute concentration curves within brain mask and optionally write 4D NIfTI.

    Returns:
      - peak_amp_map (3D, float32) in concentration space
      - ttp_idx_map (3D, int32) in concentration space
    """

    fit = _load_fitting_t1_m0_maps(analysis_directory, ref_img)
    if fit is None:
        raise FileNotFoundError(
            "Missing T1/M0 maps in Analysis/Fitting (expected t1_map.nii.gz and m0_map.nii.gz). "
            "Run T1 fitting first."
        )
    t1_map, m0_map = fit

    params = _ctc_model_params(dce_path)
    if params["ctc_model"] == "turboflash":
        raise NotImplementedError(
            "Voxelwise 4D concentration export is not supported for CTC_MODEL=turboflash (too slow). "
            "Set P_BRAIN_CTC_MODEL=saturation to export a brain concentration NIfTI."
        )

    baseline_frames = max(1, int(baseline_frames))
    batch_voxels = max(256, int(batch_voxels))

    mask = brain_mask.astype(bool)
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise ValueError("Brain mask is empty; cannot compute concentration.")

    peak_amp = np.zeros(mask.shape, dtype=np.float32)
    ttp_idx = np.zeros(mask.shape, dtype=np.int32)
    write = bool(output_path)
    ctc4d = None
    mm_path = None
    if write:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mm_path = output_path + ".memmap"
        try:
            ctc4d = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=dce4d.shape)
        except Exception as exc:
            raise RuntimeError(f"Failed to allocate memmap for concentration export: {exc}")

    # Compute in batches to keep memory bounded.
    for start in range(0, idx.shape[0], batch_voxels):
        chunk = idx[start : start + batch_voxels]
        xs, ys, zs = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        S = dce4d[xs, ys, zs, :].astype(np.float32)
        # Ensure per-voxel scalars broadcast over time.
        T1 = t1_map[xs, ys, zs].astype(np.float32)[:, None]
        M0 = m0_map[xs, ys, zs].astype(np.float32)[:, None]

        # turboflash supports broadcasting for (vox, t) + (vox,)
        Ct = turboflash(
            S,
            T1,
            m0=M0,
            prints=False,
            flip_angle_deg=params["flip_angle_deg"],
            ctc_model=params["ctc_model"],
            tr_s=params["tr_s"],
            nph=params["nph"],
        ).astype(np.float32)

        base = Ct[:, :baseline_frames].mean(axis=1, keepdims=True)
        Ct0 = Ct - base
        Ct0[:, :baseline_frames] = -np.inf
        ttp = np.argmax(Ct0, axis=1).astype(np.int32)
        amp = (np.max(Ct, axis=1) - base[:, 0]).astype(np.float32)

        peak_amp[xs, ys, zs] = amp
        ttp_idx[xs, ys, zs] = ttp
        if ctc4d is not None:
            ctc4d[xs, ys, zs, :] = np.where(np.isfinite(Ct), Ct, 0.0)

    if ctc4d is not None:
        try:
            ctc4d.flush()
        except Exception:
            pass
        header = ref_img.header.copy()
        try:
            header.set_data_dtype(np.float32)
        except Exception:
            pass
        out_img = nib.Nifti1Image(ctc4d, affine=ref_img.affine, header=header)
        nib.save(out_img, output_path)
        try:
            del ctc4d
        except Exception:
            pass
        if mm_path and os.path.exists(mm_path):
            try:
                os.remove(mm_path)
            except Exception:
                pass

    return peak_amp, ttp_idx


def _kmeans(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    max_iter: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple k-means for small diagnostic clustering (numpy only).

    Returns (labels, centers).
    """

    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    k = int(k)
    if k <= 1 or n == 0:
        return np.zeros((n,), dtype=np.int32), X[:1].copy() if n else np.zeros((1, d), dtype=np.float32)

    # init with random unique points
    pick = rng.choice(n, size=min(k, n), replace=False)
    centers = X[pick].copy()
    if centers.shape[0] < k:
        pad = np.repeat(centers[-1:], k - centers.shape[0], axis=0)
        centers = np.concatenate([centers, pad], axis=0)

    labels = np.zeros((n,), dtype=np.int32)
    for _ in range(int(max_iter)):
        # assign
        # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T
        dist2 = x2 + c2 - 2.0 * (X @ centers.T)
        new_labels = np.argmin(dist2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update
        for i in range(k):
            mask = labels == i
            if not np.any(mask):
                centers[i] = X[rng.integers(0, n)]
            else:
                centers[i] = X[mask].mean(axis=0)

    return labels, centers


def _write_debug_slice_png(volume3d: np.ndarray, out_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(volume3d, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary_images(
    *,
    analysis_directory: str,
    image_directory: str,
    ref_img: nib.Nifti1Image,
    brain_mask: np.ndarray,
    peak_amp: np.ndarray,
    artery_score: np.ndarray,
    vein_score: np.ndarray,
) -> None:
    """Write the two requested summary plots into the dataset Images/ directory.

    1) A/V from the initial PCA-derived artery_score/vein_score
    2) Final rICA/lICA/SSS ROIs loaded from Analysis/ROI NIfTI
    """

    out_dir = os.path.join(image_directory, "deterministic")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)

    bm = brain_mask.astype(bool)
    peak_amp_f = np.nan_to_num(peak_amp, nan=0.0, posinf=0.0, neginf=0.0)
    artery_f = np.nan_to_num(artery_score, nan=0.0, posinf=0.0, neginf=0.0)
    vein_f = np.nan_to_num(vein_score, nan=0.0, posinf=0.0, neginf=0.0)

    def _rot90_ccw(a: np.ndarray) -> np.ndarray:
        return np.rot90(a, 1)

    def _adaptive_score_mask(score: np.ndarray, mask: np.ndarray, *, start_pct: float, min_pct: float, min_vox: int) -> np.ndarray:
        m, _ = _adaptive_percentile_mask(
            score,
            mask,
            start_pct=float(start_pct),
            min_pct=float(min_pct),
            step=0.5,
            min_vox=int(min_vox),
        )
        return m

    # Candidate vessel voxels then split by winner (artery vs vein).
    vessel_mask, _ = _adaptive_percentile_mask(
        peak_amp_f,
        bm,
        start_pct=float(os.getenv("P_BRAIN_ROI_VESSEL_PCT", "99.0") or "99.0"),
        min_pct=float(os.getenv("P_BRAIN_ROI_VESSEL_MIN_PCT", "95.0") or "95.0"),
        step=0.5,
        min_vox=int(os.getenv("P_BRAIN_ROI_VESSEL_MIN_VOX", "1200") or "1200"),
    )
    seed_pct = float(os.getenv("P_BRAIN_ROI_AV_SEED_PCT", "99.5") or "99.5")
    seed_min_pct = float(os.getenv("P_BRAIN_ROI_AV_SEED_MIN_PCT", "95.0") or "95.0")
    seed_min_vox = int(os.getenv("P_BRAIN_ROI_AV_SEED_MIN_VOX", "200") or "200")
    artery_mask = (
        _adaptive_score_mask(artery_f, bm, start_pct=seed_pct, min_pct=seed_min_pct, min_vox=seed_min_vox)
        & vessel_mask
        & (artery_f >= vein_f)
    )
    vein_mask = (
        _adaptive_score_mask(vein_f, bm, start_pct=seed_pct, min_pct=seed_min_pct, min_vox=seed_min_vox)
        & vessel_mask
        & (vein_f > artery_f)
    )

    zdim = int(peak_amp.shape[2])
    sel = [z for z in range(zdim) if np.any(vessel_mask[:, :, z])]
    if not sel:
        sel = list(range(zdim))
    max_slices = int(os.getenv("P_BRAIN_ROI_DEBUG_MONTAGE_MAX_SLICES", "12") or "12")
    sel = sel[: max(1, min(max_slices, len(sel)))]

    cols = 4
    fine_cols = cols * 2
    rows = int(np.ceil(len(sel) / cols))

    def _make_centered_montage_axes(fig: plt.Figure, n_panels: int) -> list[plt.Axes]:
        gs = fig.add_gridspec(rows, fine_cols)
        axes: list[plt.Axes] = []
        last_row_count = n_panels - (rows - 1) * cols
        last_row_count = max(1, min(cols, last_row_count))
        last_row_start = (fine_cols - last_row_count * 2) // 2

        for i in range(n_panels):
            r = i // cols
            c = i % cols
            if r == rows - 1:
                start = last_row_start + c * 2
            else:
                start = c * 2
            axes.append(fig.add_subplot(gs[r, start : start + 2]))
        return axes

    legend_handles = [
        plt.Line2D([0], [0], color="red", lw=3, label="Artery"),
        plt.Line2D([0], [0], color="blue", lw=3, label="Vein"),
    ]
    title_fs = int(os.getenv("P_BRAIN_ROI_SUMMARY_FONTSIZE", "14") or "14")
    legend_fs = int(os.getenv("P_BRAIN_ROI_SUMMARY_FONTSIZE", "14") or "14")

    # 1) Artery/Vein montage
    fig = plt.figure(figsize=(4.2 * cols, 4.2 * rows))
    axes = _make_centered_montage_axes(fig, len(sel))
    for ax_i, z in enumerate(sel):
        ax = axes[ax_i]
        base = _rot90_ccw(peak_amp_f[:, :, z])
        ax.imshow(base, cmap="gray")
        a = _rot90_ccw(artery_mask[:, :, z])
        v = _rot90_ccw(vein_mask[:, :, z])
        if np.any(a):
            rgba = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.float32)
            rgba[..., 0] = 1.0
            rgba[..., 3] = a.astype(np.float32) * 0.70
            ax.imshow(rgba)
            try:
                ax.contour(a.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=2.0)
            except Exception:
                pass
        if np.any(v):
            rgba = np.zeros((v.shape[0], v.shape[1], 4), dtype=np.float32)
            rgba[..., 2] = 1.0
            rgba[..., 3] = v.astype(np.float32) * 0.70
            ax.imshow(rgba)
            try:
                ax.contour(v.astype(np.uint8), levels=[0.5], colors=["blue"], linewidths=2.0)
            except Exception:
                pass
        ax.set_title(f"z={z+1}", fontsize=title_fs)
        ax.axis("off")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=legend_fs,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(os.path.join(out_dir, "pca_artery_vein.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2) Final ROI montage (rICA/lICA/SSS) from Analysis/ROI NIfTI
    roi_dir = os.path.join(analysis_directory, "ROI NIfTI")
    sss_candidates = [
        os.path.join(roi_dir, "Vein__Sinus_Sagittalis__mask.nii.gz"),
        os.path.join(roi_dir, "Vein__Superior_Sagittal_Sinus__mask.nii.gz"),
    ]
    sss_path = next((p for p in sss_candidates if os.path.isfile(p)), sss_candidates[0])
    roi_paths = {
        "rica": os.path.join(roi_dir, "Artery__Right_Interior_Carotid__mask.nii.gz"),
        "lica": os.path.join(roi_dir, "Artery__Left_Interior_Carotid__mask.nii.gz"),
        "sss": sss_path,
    }
    roi_masks: dict[str, np.ndarray] = {}
    for k, p in roi_paths.items():
        if os.path.isfile(p):
            try:
                roi_masks[k] = (nib.load(p).get_fdata() > 0.5)
            except Exception:
                pass

    if roi_masks:
        final_legend_handles = [
            plt.Line2D([0], [0], color="red", lw=3, label="Right ICA"),
            plt.Line2D([0], [0], color="orange", lw=3, label="Left ICA"),
            plt.Line2D([0], [0], color="blue", lw=3, label="SSS"),
        ]
        fig = plt.figure(figsize=(4.2 * cols, 4.2 * rows))
        axes = _make_centered_montage_axes(fig, len(sel))
        for ax_i, z in enumerate(sel):
            ax = axes[ax_i]
            base = _rot90_ccw(peak_amp_f[:, :, z])
            ax.imshow(base, cmap="gray")
            if "rica" in roi_masks and z < roi_masks["rica"].shape[2] and np.any(roi_masks["rica"][:, :, z]):
                m = _rot90_ccw(roi_masks["rica"][:, :, z])
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
                rgba[..., 0] = 1.0
                rgba[..., 1] = 0.55
                rgba[..., 3] = m.astype(np.float32) * 0.75
                ax.imshow(rgba)
                try:
                    ax.contour(m.astype(np.uint8), levels=[0.5], colors=["orange"], linewidths=2.0)
                except Exception:
                    pass
            if "lica" in roi_masks and z < roi_masks["lica"].shape[2] and np.any(roi_masks["lica"][:, :, z]):
                m = _rot90_ccw(roi_masks["lica"][:, :, z])
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
                rgba[..., 0] = 1.0
                rgba[..., 3] = m.astype(np.float32) * 0.75
                ax.imshow(rgba)
                try:
                    ax.contour(m.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=2.0)
                except Exception:
                    pass
            if "sss" in roi_masks and z < roi_masks["sss"].shape[2] and np.any(roi_masks["sss"][:, :, z]):
                m = _rot90_ccw(roi_masks["sss"][:, :, z])
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
                rgba[..., 2] = 1.0
                rgba[..., 3] = m.astype(np.float32) * 0.75
                ax.imshow(rgba)
                try:
                    ax.contour(m.astype(np.uint8), levels=[0.5], colors=["blue"], linewidths=2.0)
                except Exception:
                    pass
            ax.set_title(f"z={z+1}", fontsize=title_fs)
            ax.axis("off")
        fig.legend(
            handles=final_legend_handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=legend_fs,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(os.path.join(out_dir, "final_rica_lica_sss.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_debug_outputs(
    *,
    analysis_directory: str,
    image_directory: str,
    ref_img: nib.Nifti1Image,
    dce_path: str,
    dce4d: np.ndarray,
    brain_mask: np.ndarray,
    cfg: GeometryRoiConfig,
    artery_score: np.ndarray,
    vein_score: np.ndarray,
    peak_amp: np.ndarray,
    ttp_idx: np.ndarray,
) -> None:
    """Write PCA/curve-family and vesselness debug artifacts."""

    debug_dir = os.path.join(analysis_directory, "ROI Debug")
    os.makedirs(debug_dir, exist_ok=True)

    # 1) Concentration-derived summaries: max and AUC (sum over time) per slice + 3D NIfTIs.
    # Prefer existing brain concentration 4D, otherwise compute quickly using saturation model.
    ctc_path = os.path.join(analysis_directory, "CTC Data", "Tissue", "brain_concentration_4d.nii.gz")
    use_existing = os.path.isfile(ctc_path)
    ctc_img = None
    if use_existing:
        try:
            ctc_img = nib.load(ctc_path)
        except Exception:
            ctc_img = None

    if ctc_img is None:
        # Compute a debug concentration export using saturation model even if the run uses turboflash.
        tmp_path = os.path.join(debug_dir, "brain_concentration_4d_debug_saturation.nii.gz")
        _write_brain_concentration_nifti(
            dce4d=dce4d,
            ref_img=ref_img,
            dce_path=dce_path,
            analysis_directory=analysis_directory,
            brain_mask=brain_mask,
            baseline_frames=cfg.baseline_frames,
            output_path=tmp_path,
            batch_voxels=int(os.getenv("P_BRAIN_CTC_BATCH_VOXELS", "20000") or "20000"),
        )
        ctc_img = nib.load(tmp_path)
        ctc_path = tmp_path

    ctc_dataobj = ctc_img.dataobj  # may be mmap
    xdim, ydim, zdim, tdim = ctc_img.shape
    max3d = np.zeros((xdim, ydim, zdim), dtype=np.float32)
    auc3d = np.zeros((xdim, ydim, zdim), dtype=np.float32)
    for z in range(zdim):
        slab = np.asarray(ctc_dataobj[:, :, z, :], dtype=np.float32)
        slab *= brain_mask[:, :, z][:, :, None].astype(np.float32)
        max2d = np.max(slab, axis=-1)
        auc2d = np.sum(slab, axis=-1)
        max3d[:, :, z] = max2d
        auc3d[:, :, z] = auc2d
        _write_debug_slice_png(
            max2d,
            os.path.join(debug_dir, f"ctc_max_z{z+1:02d}.png"),
            f"CTC max (z={z+1})",
        )
        _write_debug_slice_png(
            auc2d,
            os.path.join(debug_dir, f"ctc_auc_z{z+1:02d}.png"),
            f"CTC AUC/sum (z={z+1})",
        )

    header = ref_img.header.copy()
    try:
        header.set_data_dtype(np.float32)
    except Exception:
        pass
    nib.save(
        nib.Nifti1Image(max3d.astype(np.float32), affine=ref_img.affine, header=header),
        os.path.join(debug_dir, "brain_concentration_max_3d.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(auc3d.astype(np.float32), affine=ref_img.affine, header=header),
        os.path.join(debug_dir, "brain_concentration_auc_3d.nii.gz"),
    )

    # 2) Save artery/vein score maps and thresholded vessel-candidate masks.
    def _thr_mask(score: np.ndarray, mask: np.ndarray, pct: float) -> np.ndarray:
        vals = score[mask]
        if vals.size < 128:
            return mask & (score > 0)
        thr = float(np.percentile(vals, pct))
        return mask & (score >= thr)

    def _adaptive_thr_mask(
        score: np.ndarray,
        mask: np.ndarray,
        *,
        start_pct: float,
        min_pct: float,
        min_vox: int,
    ) -> tuple[np.ndarray, float]:
        pct = float(start_pct)
        while pct >= float(min_pct):
            m = _thr_mask(score, mask, pct)
            if int(m.sum()) >= int(min_vox):
                return m, pct
            pct -= 0.5
        m = _thr_mask(score, mask, float(min_pct))
        return m, float(min_pct)

    # Candidate vessel voxels for PCA: start strict, relax until we have enough samples.
    pca_min_vox = int(os.getenv("P_BRAIN_ROI_PCA_MIN_VOXELS", "1500") or "1500")
    vessel_mask, vessel_pct = _adaptive_thr_mask(
        peak_amp,
        brain_mask,
        start_pct=float(os.getenv("P_BRAIN_ROI_VESSEL_PCT", "99.0") or "99.0"),
        min_pct=float(os.getenv("P_BRAIN_ROI_VESSEL_MIN_PCT", "95.0") or "95.0"),
        min_vox=pca_min_vox,
    )
    # Competing A/V masks: restrict to strong vascular voxels then split by which score wins.
    artery_mask = _thr_mask(artery_score, brain_mask, 99.5) & vessel_mask & (artery_score >= vein_score)

    # For the *debug* vein mask, suppress boundary-adjacent "pools" that aren't useful vessels.
    # (SSS selection uses its own component logic and is not driven by this mask.)
    dist3d = ndimage.distance_transform_edt(brain_mask.astype(bool)).astype(np.float32)
    vein_min_dist = float(os.getenv("P_BRAIN_DEBUG_VEIN_MIN_DIST_INNER", "2.0") or "2.0")
    core = dist3d >= vein_min_dist if np.isfinite(vein_min_dist) and vein_min_dist > 0 else brain_mask
    vein_mask = _thr_mask(vein_score, brain_mask, 99.5) & vessel_mask & core & (vein_score > artery_score)

    nib.save(
        nib.Nifti1Image(artery_score.astype(np.float32), affine=ref_img.affine, header=header),
        os.path.join(debug_dir, "artery_score_3d.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(vein_score.astype(np.float32), affine=ref_img.affine, header=header),
        os.path.join(debug_dir, "vein_score_3d.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(vessel_mask.astype(np.uint8), affine=ref_img.affine, header=ref_img.header),
        os.path.join(debug_dir, "vessel_candidate_mask.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(artery_mask.astype(np.uint8), affine=ref_img.affine, header=ref_img.header),
        os.path.join(debug_dir, "artery_candidate_mask.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(vein_mask.astype(np.uint8), affine=ref_img.affine, header=ref_img.header),
        os.path.join(debug_dir, "vein_candidate_mask.nii.gz"),
    )

    # Per-slice overlays (candidate masks on max concentration).
    for z in range(zdim):
        base = max3d[:, :, z]
        for name, m in (
            ("vessel", vessel_mask[:, :, z]),
            ("artery", artery_mask[:, :, z]),
            ("vein", vein_mask[:, :, z]),
        ):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(base, cmap="gray")
            ax.imshow(np.where(m, 1.0, np.nan), cmap="autumn", alpha=0.55)
            ax.set_title(f"{name} mask on CTC max (z={z+1})")
            ax.axis("off")
            fig.savefig(os.path.join(debug_dir, f"{name}_mask_overlay_z{z+1:02d}.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    # 2b) Montage overlays: arteries (red) + veins (blue) on selected slices.
    # Selected slices default to ones with any vessel candidates.
    sel = [z for z in range(zdim) if np.any(vessel_mask[:, :, z])]
    if not sel:
        sel = list(range(zdim))
    max_slices = int(os.getenv("P_BRAIN_ROI_DEBUG_MONTAGE_MAX_SLICES", "12") or "12")
    sel = sel[: max(1, min(max_slices, len(sel)))]

    cols = min(4, len(sel))
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax_i, z in enumerate(sel):
        ax = axes[ax_i]
        base = max3d[:, :, z]
        ax.imshow(base, cmap="gray")
        a = artery_mask[:, :, z]
        v = vein_mask[:, :, z]
        # Explicit RGBA overlays to guarantee visible coloring.
        if np.any(a):
            rgba = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.float32)
            rgba[..., 0] = 1.0
            rgba[..., 3] = a.astype(np.float32) * 0.70
            ax.imshow(rgba)
            try:
                ax.contour(a.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=2.0)
            except Exception:
                pass
        if np.any(v):
            rgba = np.zeros((v.shape[0], v.shape[1], 4), dtype=np.float32)
            rgba[..., 2] = 1.0
            rgba[..., 3] = v.astype(np.float32) * 0.70
            ax.imshow(rgba)
            try:
                ax.contour(v.astype(np.uint8), levels=[0.5], colors=["blue"], linewidths=2.0)
            except Exception:
                pass
        ax.set_title(f"A(red)/V(blue) z={z+1}")
        ax.axis("off")
    for j in range(len(sel), axes.size):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(debug_dir, "montage_artery_vein_masks.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2c) ROI montage: rICA/lICA/SSS masks over CTC max on the same slices.
    # These are written to Analysis/ROI NIfTI by the ROI saver; try to load if present.
    roi_dir = os.path.join(analysis_directory, "ROI NIfTI")
    sss_candidates = [
        os.path.join(roi_dir, "Vein__Sinus_Sagittalis__mask.nii.gz"),
        os.path.join(roi_dir, "Vein__Superior_Sagittal_Sinus__mask.nii.gz"),
    ]
    sss_path = next((p for p in sss_candidates if os.path.isfile(p)), sss_candidates[0])
    roi_paths = {
        "rica": os.path.join(roi_dir, "Artery__Right_Interior_Carotid__mask.nii.gz"),
        "lica": os.path.join(roi_dir, "Artery__Left_Interior_Carotid__mask.nii.gz"),
        "sss": sss_path,
        "basilar": os.path.join(roi_dir, "Artery__Basilar__mask.nii.gz"),
    }
    roi_masks = {}
    for k, p in roi_paths.items():
        if os.path.isfile(p):
            try:
                roi_masks[k] = (nib.load(p).get_fdata() > 0.5)
            except Exception:
                pass

    if roi_masks:
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax_i, z in enumerate(sel):
            ax = axes[ax_i]
            base = max3d[:, :, z]
            ax.imshow(base, cmap="gray")
            if "rica" in roi_masks and z < roi_masks["rica"].shape[2] and np.any(roi_masks["rica"][:, :, z]):
                m = roi_masks["rica"][:, :, z]
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
                rgba[..., 0] = 1.0
                rgba[..., 3] = m.astype(np.float32) * 0.75
                ax.imshow(rgba)
                try:
                    ax.contour(m.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=2.0)
                except Exception:
                    pass
            if "lica" in roi_masks and z < roi_masks["lica"].shape[2] and np.any(roi_masks["lica"][:, :, z]):
                m = roi_masks["lica"][:, :, z]
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
                rgba[..., 0] = 1.0
                rgba[..., 1] = 0.55
                rgba[..., 3] = m.astype(np.float32) * 0.75
                ax.imshow(rgba)
                try:
                    ax.contour(m.astype(np.uint8), levels=[0.5], colors=["orange"], linewidths=2.0)
                except Exception:
                    pass
            if "sss" in roi_masks and z < roi_masks["sss"].shape[2] and np.any(roi_masks["sss"][:, :, z]):
                m = roi_masks["sss"][:, :, z]
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
                rgba[..., 2] = 1.0
                rgba[..., 3] = m.astype(np.float32) * 0.75
                ax.imshow(rgba)
                try:
                    ax.contour(m.astype(np.uint8), levels=[0.5], colors=["blue"], linewidths=2.0)
                except Exception:
                    pass
            if "basilar" in roi_masks and z < roi_masks["basilar"].shape[2] and np.any(roi_masks["basilar"][:, :, z]):
                m = roi_masks["basilar"][:, :, z]
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
                rgba[..., 1] = 1.0
                rgba[..., 3] = m.astype(np.float32) * 0.75
                ax.imshow(rgba)
                try:
                    ax.contour(m.astype(np.uint8), levels=[0.5], colors=["lime"], linewidths=2.0)
                except Exception:
                    pass
            ax.set_title(f"ROIs (rICA/ lICA/ SSS) z={z+1}")
            ax.axis("off")
        for j in range(len(sel), axes.size):
            axes[j].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(debug_dir, "montage_rois.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3) PCA curve families on a vascular candidate subset.
    do_pca = _getenv_bool("P_BRAIN_ROI_DEBUG_PCA", default=True)
    if not do_pca:
        return

    # Use vessel candidates, but cap count.
    max_vox = int(os.getenv("P_BRAIN_ROI_PCA_MAX_VOXELS", "20000") or "20000")
    # Restrict curve-family sampling to the middle z-slab to avoid edge slices.
    pca_mid_only = _getenv_bool("P_BRAIN_ROI_DEBUG_PCA_MIDZ_ONLY", default=True)
    vmask_for_pca = vessel_mask
    if pca_mid_only:
        vmask_for_pca = vessel_mask & _middle_slice_mask(vessel_mask.shape)
    pts = _sample_voxel_indices(vmask_for_pca, peak_amp, max_voxels=max_vox)
    if pts.shape[0] < 256:
        warnings.warn(
            f"Not enough vascular voxels for PCA debug (n={pts.shape[0]} @ pct={vessel_pct}).",
            RuntimeWarning,
        )
        return

    # Extract concentration curves for sampled voxels.
    # nibabel ArrayProxy does not support fancy indexing with arrays; sample per-voxel.
    pts = np.asarray(pts, dtype=np.int64)
    t_len = int(ctc_img.shape[3])
    curves = np.empty((pts.shape[0], t_len), dtype=np.float32)
    for i, (x, y, z) in enumerate(pts):
        curves[i, :] = np.asarray(ctc_dataobj[int(x), int(y), int(z), :], dtype=np.float32)
    base = curves[:, : cfg.baseline_frames].mean(axis=1, keepdims=True)
    X = curves - base
    X[:, : cfg.baseline_frames] = 0.0
    # Normalize per-voxel to focus on shape.
    denom = np.percentile(np.abs(X), 95, axis=1, keepdims=True)
    denom = np.where(denom <= 1e-6, 1.0, denom)
    Xn = X / denom
    Xn = Xn - Xn.mean(axis=0, keepdims=True)

    try:
        U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    except Exception as exc:
        warnings.warn(f"PCA SVD failed: {exc}", RuntimeWarning)
        return

    k_pc = int(os.getenv("P_BRAIN_ROI_PCA_COMPONENTS", "3") or "3")
    k_pc = max(1, min(k_pc, Vt.shape[0]))
    pc_time = Vt[:k_pc].astype(np.float32)
    scores = (U[:, :k_pc] * S[:k_pc]).astype(np.float32)

    # Cluster the PCA scores to get curve families.
    k_clusters = int(os.getenv("P_BRAIN_ROI_PCA_N_CLUSTERS", "6") or "6")
    k_clusters = max(2, min(k_clusters, 12))
    labels, centers = _kmeans(scores, k_clusters, seed=int(os.getenv("P_BRAIN_ROI_PCA_SEED", "0") or "0"))

    # Compute mean curves per cluster.
    cluster_means = []
    cluster_ttps = []
    for ci in range(k_clusters):
        m = labels == ci
        if not np.any(m):
            cluster_means.append(None)
            cluster_ttps.append(10**9)
            continue
        mean_curve = curves[m].mean(axis=0)
        cluster_means.append(mean_curve)
        # time-to-peak after baseline
        mc = mean_curve.copy()
        mc[: cfg.baseline_frames] = -np.inf
        cluster_ttps.append(int(np.argmax(mc)))

    # Plot PCs + families.
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i in range(k_pc):
        axes[0].plot(pc_time[i], label=f"PC{i+1}")
    axes[0].set_title("PCA components (timecourses)")
    axes[0].legend(loc="best")
    for ci, mean_curve in enumerate(cluster_means):
        if mean_curve is None:
            continue
        axes[1].plot(mean_curve, label=f"C{ci} (ttp={cluster_ttps[ci]})")
    axes[1].set_title("Curve families (cluster mean concentration curves)")
    axes[1].legend(loc="best", ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(debug_dir, "pca_curve_families.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # PCA score-space plot (cluster scatter).
    if scores.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap_name = "tab20" if k_clusters > 10 else "tab10"
        cmap = plt.cm.get_cmap(cmap_name, k_clusters)
        sc = ax.scatter(
            scores[:, 0],
            scores[:, 1],
            c=labels,
            cmap=cmap,
            s=14,
            alpha=0.85,
            linewidths=0,
        )
        if isinstance(centers, np.ndarray) and centers.shape[0] == k_clusters and centers.shape[1] >= 2:
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c=np.arange(k_clusters),
                cmap=cmap,
                s=140,
                marker="X",
                edgecolors="black",
                linewidths=0.6,
            )
        ax.set_title("PCA score space (k-means clusters)")
        ax.set_xlabel("PC1 score")
        ax.set_ylabel("PC2 score")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(k_clusters), pad=0.01, fraction=0.05)
        cbar.set_label("Cluster")
        fig.tight_layout()

        fig.savefig(os.path.join(debug_dir, "pca_space_clusters.png"), dpi=300, bbox_inches="tight")

        if image_directory:
            out_dir = os.path.join(image_directory, "deterministic")
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, "pca_space_clusters.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Choose artery/vein families from cluster TTP.
    valid = [(ci, t) for ci, t in enumerate(cluster_ttps) if t < 10**9]
    if len(valid) >= 2:
        artery_ci = min(valid, key=lambda t: t[1])[0]
        vein_ci = max(valid, key=lambda t: t[1])[0]
    else:
        artery_ci = 0
        vein_ci = 1 if k_clusters > 1 else 0

    # Create 3D masks for each cluster (within vessel_mask only) + artery/vein family masks.
    family_masks = np.zeros(brain_mask.shape + (k_clusters,), dtype=np.uint8)
    for i, (x, y, z) in enumerate(pts):
        family_masks[x, y, z, labels[i]] = 1

    for ci in range(k_clusters):
        nib.save(
            nib.Nifti1Image(family_masks[..., ci], affine=ref_img.affine, header=ref_img.header),
            os.path.join(debug_dir, f"pca_family_mask_cluster_{ci}.nii.gz"),
        )

    nib.save(
        nib.Nifti1Image(family_masks[..., artery_ci], affine=ref_img.affine, header=ref_img.header),
        os.path.join(debug_dir, "pca_family_mask_artery.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(family_masks[..., vein_ci], affine=ref_img.affine, header=ref_img.header),
        os.path.join(debug_dir, "pca_family_mask_vein.nii.gz"),
    )


def _normalize_curve_for_pca(
    curve: np.ndarray,
    *,
    baseline_frames: int,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_curve, normalized_curve) used by PCA-style plots.

    When ``normalize`` is enabled, normalization matches the PCA debug logic:
    subtract baseline mean, zero out baseline frames, and scale by the 95th
    percentile of absolute deviation.
    """

    curve = np.asarray(curve, dtype=np.float32).copy()
    baseline_frames = max(1, int(baseline_frames))
    if curve.size == 0:
        return curve, curve
    if not normalize:
        return curve, curve
    base = float(np.mean(curve[:baseline_frames]))
    centered = curve - base
    centered[:baseline_frames] = 0.0
    denom = float(np.percentile(np.abs(centered), 95)) if centered.size else 1.0
    if not np.isfinite(denom) or denom <= 1e-6:
        denom = 1.0
    normalized = centered / denom
    return curve, normalized.astype(np.float32)


def _write_normalization_demo_plot(
    *,
    out_path: str,
    curve: np.ndarray,
    baseline_frames: int,
    title: str,
    normalize: bool = True,
    time_s: np.ndarray | None = None,
    x_label: str = "Time (s)",
    y_label: str = "",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw, norm = _normalize_curve_for_pca(curve, baseline_frames=baseline_frames, normalize=normalize)
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    x = np.asarray(time_s, dtype=np.float32) if time_s is not None else np.arange(raw.size, dtype=np.float32)
    ax.plot(
        x,
        raw,
        color="#9e9e9e",
        alpha=0.25,
        linewidth=1.2,
        linestyle="--",
        label="Raw" if normalize else "Raw (normalization disabled)",
    )
    if normalize:
        ax.plot(x, norm, color="red", alpha=0.95, linewidth=1.3, label="Normalized")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_pca_and_normalization_plots(
    *,
    analysis_directory: str,
    image_directory: str,
    ref_img: nib.Nifti1Image,
    dce_path: str,
    dce4d: np.ndarray,
    brain_mask: np.ndarray,
    cfg: GeometryRoiConfig,
    peak_amp: np.ndarray,
) -> None:
    """Write PCA plots + normalization demos to Images/deterministic.

    This is intentionally lightweight (no 3D NIfTIs / per-slice overlays) so it can
    run alongside the standard summary montage.
    """

    if not image_directory:
        return
    out_dir = os.path.join(image_directory, "deterministic")
    os.makedirs(out_dir, exist_ok=True)

    baseline_frames = int(cfg.baseline_frames)
    zdim = int(dce4d.shape[2])

    normalize_curves = _getenv_bool("P_BRAIN_ROI_NORMALIZE_CURVES", default=True)

    tr_s = 1.0
    try:
        zooms = ref_img.header.get_zooms()
        if zooms and len(zooms) >= 4 and np.isfinite(float(zooms[-1])):
            tr_s = float(zooms[-1])
    except Exception:
        tr_s = 1.0

    # --- PCA plots (signal-space by default, optional concentration-space).
    pca_space = (os.getenv("P_BRAIN_ROI_PCA_PLOTS_SPACE", "signal") or "signal").strip().lower()
    if pca_space not in {"signal", "concentration", "ctc"}:
        pca_space = "signal"

    # Candidate vessel voxels for PCA.
    bm = brain_mask.astype(bool)
    vessel_mask, _ = _adaptive_percentile_mask(
        np.nan_to_num(peak_amp, nan=0.0, posinf=0.0, neginf=0.0),
        bm,
        start_pct=float(os.getenv("P_BRAIN_ROI_VESSEL_PCT", "99.0") or "99.0"),
        min_pct=float(os.getenv("P_BRAIN_ROI_VESSEL_MIN_PCT", "95.0") or "95.0"),
        step=0.5,
        min_vox=int(os.getenv("P_BRAIN_ROI_PCA_MIN_VOXELS", "1500") or "1500"),
    )

    max_vox = int(os.getenv("P_BRAIN_ROI_PCA_MAX_VOXELS", "20000") or "20000")
    pca_mid_only = _getenv_bool("P_BRAIN_ROI_DEBUG_PCA_MIDZ_ONLY", default=True)
    vmask_for_pca = vessel_mask & (_middle_slice_mask(vessel_mask.shape) if pca_mid_only else True)
    pts = _sample_voxel_indices(vmask_for_pca, peak_amp, max_voxels=max_vox)
    pts = np.asarray(pts, dtype=np.int64)
    if pts.shape[0] >= 256:
        curves = None
        y_label = "MR Signal (a.u.)"

        if pca_space in {"concentration", "ctc"}:
            ctc_path = os.path.join(analysis_directory, "CTC Data", "Tissue", "brain_concentration_4d.nii.gz")
            ctc_img = None
            if os.path.isfile(ctc_path):
                try:
                    ctc_img = nib.load(ctc_path)
                except Exception:
                    ctc_img = None
            if ctc_img is None:
                # Best-effort compute using saturation model if fitting maps exist.
                try:
                    _write_brain_concentration_nifti(
                        dce4d=dce4d,
                        ref_img=ref_img,
                        dce_path=dce_path,
                        analysis_directory=analysis_directory,
                        brain_mask=brain_mask,
                        baseline_frames=baseline_frames,
                        output_path=ctc_path,
                        batch_voxels=int(os.getenv("P_BRAIN_CTC_BATCH_VOXELS", "20000") or "20000"),
                    )
                    ctc_img = nib.load(ctc_path)
                except Exception:
                    ctc_img = None

            if ctc_img is not None:
                ctc_dataobj = ctc_img.dataobj
                t_len = int(ctc_img.shape[3])
                curves = np.empty((pts.shape[0], t_len), dtype=np.float32)
                for i, (x, y, z) in enumerate(pts):
                    curves[i, :] = np.asarray(ctc_dataobj[int(x), int(y), int(z), :], dtype=np.float32)
                y_label = "Concentration (mmol/100g/min)"

        if curves is None:
            # Signal curves from DCE directly (always available).
            t_len = int(dce4d.shape[3])
            curves = np.empty((pts.shape[0], t_len), dtype=np.float32)
            for i, (x, y, z) in enumerate(pts):
                curves[i, :] = np.asarray(dce4d[int(x), int(y), int(z), :], dtype=np.float32)

        t_len = int(curves.shape[1])
        time_s = (np.arange(t_len, dtype=np.float32) * float(tr_s)).astype(np.float32)

        if normalize_curves:
            base = curves[:, :baseline_frames].mean(axis=1, keepdims=True)
            X = curves - base
            X[:, :baseline_frames] = 0.0
            denom = np.percentile(np.abs(X), 95, axis=1, keepdims=True)
            denom = np.where(denom <= 1e-6, 1.0, denom)
            Xn = X / denom
        else:
            Xn = curves.astype(np.float32)
        Xn = Xn - Xn.mean(axis=0, keepdims=True)

        try:
            U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
            k_pc = int(os.getenv("P_BRAIN_ROI_PCA_COMPONENTS", "3") or "3")
            k_pc = max(1, min(k_pc, Vt.shape[0]))
            pc_time = Vt[:k_pc].astype(np.float32)
            scores = (U[:, :k_pc] * S[:k_pc]).astype(np.float32)

            # Dominant PC membership per sampled voxel (by absolute score magnitude).
            pc_membership = np.argmax(np.abs(scores), axis=1).astype(np.int32) if scores.size else np.zeros((scores.shape[0],), dtype=np.int32)
            pc_colors = ["red", "green", "blue", "purple", "orange", "cyan"]
            pc_colors = pc_colors[: max(1, int(k_pc))]

            k_clusters = int(os.getenv("P_BRAIN_ROI_PCA_N_CLUSTERS", "6") or "6")
            k_clusters = max(2, min(k_clusters, 12))
            labels, centers = _kmeans(scores, k_clusters, seed=int(os.getenv("P_BRAIN_ROI_PCA_SEED", "0") or "0"))

            # Cluster mean curves and TTPs.
            cluster_means = []
            cluster_ttps = []
            for ci in range(k_clusters):
                m = labels == ci
                if not np.any(m):
                    cluster_means.append(None)
                    cluster_ttps.append(10**9)
                    continue
                mean_curve = curves[m].mean(axis=0)
                cluster_means.append(mean_curve)
                mc = mean_curve.copy()
                mc[:baseline_frames] = -np.inf
                cluster_ttps.append(int(np.argmax(mc)))

            # PCA components + family means.
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for i in range(k_pc):
                axes[0].plot(time_s, pc_time[i], label=f"PC{i+1}", linewidth=1.2)
            axes[0].set_title("PCA components (timecourses)")
            axes[0].set_ylabel("Component amplitude (a.u.)")
            axes[0].legend(loc="best")
            axes[0].grid(True, which="major", alpha=0.35, linewidth=0.8)
            for ci, mean_curve in enumerate(cluster_means):
                if mean_curve is None:
                    continue
                axes[1].plot(time_s, mean_curve, label=f"C{ci} (ttp={cluster_ttps[ci]})", linewidth=1.2)
            axes[1].set_title("Curve families (cluster mean curves)")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel(y_label)
            axes[1].legend(loc="best", ncol=2)
            axes[1].grid(True, which="major", alpha=0.35, linewidth=0.8)

            try:
                from matplotlib.ticker import MaxNLocator

                for ax in axes:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            except Exception:
                pass

            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "pca_curve_families.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            # PCA score space.
            if scores.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                cmap_name = "tab20" if k_clusters > 10 else "tab10"
                cmap = plt.cm.get_cmap(cmap_name, k_clusters)
                sc = ax.scatter(
                    scores[:, 0],
                    scores[:, 1],
                    c=labels,
                    cmap=cmap,
                    s=14,
                    alpha=0.85,
                    linewidths=0,
                )
                if isinstance(centers, np.ndarray) and centers.shape[0] == k_clusters and centers.shape[1] >= 2:
                    ax.scatter(
                        centers[:, 0],
                        centers[:, 1],
                        c=np.arange(k_clusters),
                        cmap=cmap,
                        s=140,
                        marker="X",
                        edgecolors="black",
                        linewidths=0.6,
                    )
                ax.set_title("PCA score space (k-means clusters)")
                ax.set_xlabel("PC1 score")
                ax.set_ylabel("PC2 score")
                ax.grid(True, alpha=0.25)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(k_clusters), pad=0.01, fraction=0.05)
                cbar.set_label("Cluster")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "pca_space_clusters.png"), dpi=300, bbox_inches="tight")
                plt.close(fig)

                # PC-membership score-space plot (same colors used by the montage below).
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(int(k_pc)):
                    m = pc_membership == i
                    if not np.any(m):
                        continue
                    ax.scatter(
                        scores[m, 0],
                        scores[m, 1],
                        s=12,
                        alpha=0.85,
                        linewidths=0,
                        c=pc_colors[i],
                        label=f"PC{i+1}",
                    )
                ax.set_title("PCA score space (dominant PC membership)")
                ax.set_xlabel("PC1 score")
                ax.set_ylabel("PC2 score")
                ax.grid(True, alpha=0.25)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(loc="best", frameon=False)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "pca_space_pc_membership.png"), dpi=300, bbox_inches="tight")
                plt.close(fig)

            # Slice montage showing dominant PC membership (same colors as score-space).
            try:
                sel = [z for z in range(zdim) if np.any(vessel_mask[:, :, z])]
                if not sel:
                    sel = list(range(zdim))
                max_slices = int(os.getenv("P_BRAIN_ROI_DEBUG_MONTAGE_MAX_SLICES", "12") or "12")
                sel = sel[: max(1, min(max_slices, len(sel)))]

                cols = min(4, len(sel))
                rows = int(np.ceil(len(sel) / cols))
                fig = plt.figure(figsize=(4.2 * cols, 4.2 * rows))
                gs = fig.add_gridspec(rows, cols)
                axes = []
                for i in range(len(sel)):
                    r = i // cols
                    c = i % cols
                    axes.append(fig.add_subplot(gs[r, c]))

                pts_arr = np.asarray(pts, dtype=np.int64)
                for ax_i, z in enumerate(sel):
                    ax = axes[ax_i]
                    base2d = peak_amp[:, :, z]
                    ax.imshow(base2d, cmap="gray")

                    in_slice = pts_arr[:, 2] == int(z)
                    if np.any(in_slice):
                        xs = pts_arr[in_slice, 0]
                        ys = pts_arr[in_slice, 1]
                        memb = pc_membership[in_slice]
                        for pi in range(int(k_pc)):
                            m = memb == pi
                            if not np.any(m):
                                continue
                            ax.scatter(
                                ys[m],
                                xs[m],
                                s=8,
                                c=pc_colors[pi],
                                alpha=0.75,
                                linewidths=0,
                            )

                    ax.set_title(f"Dominant PC (z={z+1})")
                    ax.axis("off")

                # Hide unused axes.
                for j in range(len(sel), len(axes)):
                    axes[j].axis("off")

                handles = [plt.Line2D([0], [0], color=pc_colors[i], lw=3, label=f"PC{i+1}") for i in range(int(k_pc))]
                fig.legend(handles=handles, loc="upper center", ncol=int(k_pc), frameon=False, bbox_to_anchor=(0.5, 1.02))
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
                fig.savefig(os.path.join(out_dir, "pca_pc_membership_montage.png"), dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                warnings.warn(f"Failed to write PC membership montage: {exc}", RuntimeWarning)

            # 3D PCA score-space plot when available.
            if scores.shape[1] >= 3 and int(k_pc) >= 3:
                try:
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                    fig = plt.figure(figsize=(8.5, 6.5))
                    ax = fig.add_subplot(111, projection="3d")
                    cmap_name = "tab20" if k_clusters > 10 else "tab10"
                    cmap = plt.cm.get_cmap(cmap_name, k_clusters)
                    ax.scatter(
                        scores[:, 0],
                        scores[:, 1],
                        scores[:, 2],
                        c=labels,
                        cmap=cmap,
                        s=10,
                        alpha=0.85,
                        linewidths=0,
                    )
                    if isinstance(centers, np.ndarray) and centers.shape[0] == k_clusters and centers.shape[1] >= 3:
                        ax.scatter(
                            centers[:, 0],
                            centers[:, 1],
                            centers[:, 2],
                            c=np.arange(k_clusters),
                            cmap=cmap,
                            s=80,
                            marker="X",
                            edgecolors="black",
                            linewidths=0.6,
                        )
                    ax.set_title("PCA score space (k-means clusters, 3D)")
                    ax.set_xlabel("PC1 score")
                    ax.set_ylabel("PC2 score")
                    ax.set_zlabel("PC3 score")
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, "pca_space_clusters_3d.png"), dpi=300, bbox_inches="tight")
                    plt.close(fig)
                except Exception as exc:
                    warnings.warn(f"Failed to write 3D PCA plot: {exc}", RuntimeWarning)
        except Exception as exc:
            warnings.warn(f"Failed to write PCA plots: {exc}", RuntimeWarning)

    # --- Normalization demo plots.
    if _getenv_bool("P_BRAIN_ROI_WRITE_NORMALIZATION_DEMO", default=True):
        # Choose one vessel voxel and one non-vessel tissue voxel with large peak amplitude.
        vessel_pts = np.argwhere(vessel_mask)
        if vessel_pts.size:
            idx = int(np.argmax(peak_amp[vessel_mask]))
            vx = vessel_pts[idx]
        else:
            vx = None

        tissue_mask = bm & (~vessel_mask)
        tissue_pts = np.argwhere(tissue_mask)
        tx = None
        if tissue_pts.size:
            # Pick the strongest tissue voxel (large normalization effect).
            idx = int(np.argmax(peak_amp[tissue_mask]))
            tx = tissue_pts[idx]

        # For CTC demos, prefer existing brain concentration 4D; otherwise fall back to signal.
        ctc_img = None
        ctc_path = os.path.join(analysis_directory, "CTC Data", "Tissue", "brain_concentration_4d.nii.gz")
        if os.path.isfile(ctc_path):
            try:
                ctc_img = nib.load(ctc_path)
            except Exception:
                ctc_img = None

        if ctc_img is None:
            try:
                _write_brain_concentration_nifti(
                    dce4d=dce4d,
                    ref_img=ref_img,
                    dce_path=dce_path,
                    analysis_directory=analysis_directory,
                    brain_mask=brain_mask,
                    baseline_frames=baseline_frames,
                    output_path=ctc_path,
                    batch_voxels=int(os.getenv("P_BRAIN_CTC_BATCH_VOXELS", "20000") or "20000"),
                )
                ctc_img = nib.load(ctc_path)
            except Exception:
                ctc_img = None

        use_ctc = ctc_img is not None
        dataobj = ctc_img.dataobj if use_ctc else None
        for label, pt in (("vessel", vx), ("tissue", tx)):
            if pt is None:
                continue
            x, y, z = (int(pt[0]), int(pt[1]), int(pt[2]))
            t_len = int(dce4d.shape[3])
            time_s = (np.arange(t_len, dtype=np.float32) * float(tr_s)).astype(np.float32)
            if use_ctc:
                curve = np.asarray(dataobj[x, y, z, :], dtype=np.float32)
                out_path = os.path.join(out_dir, f"ctc_normalization_demo_{label}.png")
                title = "Normalization of Concentration time curves"
                ylab = "Concentration (mmol/100g/min)"
            else:
                curve = np.asarray(dce4d[x, y, z, :], dtype=np.float32)
                out_path = os.path.join(out_dir, f"signal_normalization_demo_{label}.png")
                title = "Normalization of MR signal time curves"
                ylab = "MR Signal (a.u.)"
            _write_normalization_demo_plot(
                out_path=out_path,
                curve=curve,
                baseline_frames=baseline_frames,
                title=title,
                normalize=normalize_curves,
                time_s=time_s,
                x_label="Time (s)",
                y_label=ylab,
            )


def _write_global_pc_membership_plots(
    *,
    image_directory: str,
    ref_img: nib.Nifti1Image,
    dce4d: np.ndarray,
    brain_mask: np.ndarray,
    peak_amp: np.ndarray,
    baseline_frames: int,
    pc_time: np.ndarray,
) -> None:
    """Visualize dominant PC membership for the *global* PCA used for A/V scoring."""

    if not image_directory:
        return
    out_dir = os.path.join(image_directory, "deterministic")
    os.makedirs(out_dir, exist_ok=True)

    k_pc = int(pc_time.shape[0])
    if k_pc < 2:
        return

    baseline_frames = max(1, int(baseline_frames))

    # Project *all* intracranial voxels for membership visualization.
    mask3d = brain_mask.astype(bool)
    idx = np.argwhere(mask3d)
    if idx.size == 0:
        return

    # Centering term should match the PCA training: subtract mean across voxels per timepoint.
    # We approximate using a capped random subset for stability.
    rng = np.random.default_rng(0)
    cap = int(os.getenv("P_BRAIN_ROI_GLOBAL_PC_MEAN_CAP", "5000") or "5000")
    if idx.shape[0] > cap:
        pick = rng.choice(idx.shape[0], size=cap, replace=False)
        idx_mean = idx[pick]
    else:
        idx_mean = idx

    Xs = dce4d[idx_mean[:, 0], idx_mean[:, 1], idx_mean[:, 2], :].astype(np.float32)
    base = Xs[:, :baseline_frames].mean(axis=1, keepdims=True)
    Xs = Xs - base
    Xs[:, :baseline_frames] = 0.0
    mean_time = Xs.mean(axis=0, keepdims=True).astype(np.float32)  # (1,t)

    # Project all voxels in vessel_mask onto PCs in batches.
    membership = np.full(brain_mask.shape, fill_value=-1, dtype=np.int8)

    tr_s = 1.0
    try:
        zooms = ref_img.header.get_zooms()
        if zooms and len(zooms) >= 4 and np.isfinite(float(zooms[-1])):
            tr_s = float(zooms[-1])
    except Exception:
        tr_s = 1.0

    # Score-space scatter: subsample to keep it readable.
    scatter_cap = int(os.getenv("P_BRAIN_ROI_GLOBAL_PC_SCATTER_CAP", "15000") or "15000")
    scatter_pick = None
    if idx.shape[0] > scatter_cap:
        scatter_pick = rng.choice(idx.shape[0], size=scatter_cap, replace=False)
    else:
        scatter_pick = np.arange(idx.shape[0])

    scatter_scores = np.empty((scatter_pick.shape[0], min(3, k_pc)), dtype=np.float32)
    scatter_membership = np.empty((scatter_pick.shape[0],), dtype=np.int32)

    batch = int(os.getenv("P_BRAIN_ROI_GLOBAL_PC_BATCH", "25000") or "25000")
    batch = max(512, int(batch))
    V = pc_time[:k_pc].T.astype(np.float32)  # (t,k)
    si = 0
    scatter_map = {int(scatter_pick[i]): i for i in range(int(scatter_pick.shape[0]))}

    for start in range(0, idx.shape[0], batch):
        chunk = idx[start : start + batch]
        X = dce4d[chunk[:, 0], chunk[:, 1], chunk[:, 2], :].astype(np.float32)
        base = X[:, :baseline_frames].mean(axis=1, keepdims=True)
        X = X - base
        X[:, :baseline_frames] = 0.0
        X = X - mean_time
        scores = X @ V  # (n,k)
        memb = np.argmax(np.abs(scores), axis=1).astype(np.int32)
        membership[chunk[:, 0], chunk[:, 1], chunk[:, 2]] = memb.astype(np.int8)

        # Fill scatter arrays for indices inside this chunk.
        for j in range(chunk.shape[0]):
            global_i = start + j
            if global_i in scatter_map:
                out_i = scatter_map[global_i]
                scatter_membership[out_i] = memb[j]
                take = min(3, k_pc)
                scatter_scores[out_i, :take] = scores[j, :take]

    # --- Score-space plot colored by dominant PC.
    if k_pc >= 2:
        pc_colors = ["red", "green", "blue", "purple", "orange", "cyan"]
        pc_colors = pc_colors[:k_pc]
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(k_pc):
            m = scatter_membership == i
            if not np.any(m):
                continue
            ax.scatter(
                scatter_scores[m, 0],
                scatter_scores[m, 1],
                s=10,
                alpha=0.75,
                linewidths=0,
                c=pc_colors[i],
                label=f"PC{i+1}",
            )
        ax.set_title("Global PCA score space (dominant PC membership)")
        ax.set_xlabel("PC1 score")
        ax.set_ylabel("PC2 score")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pca_global_space_pc_membership.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Slice montage with membership overlay (same colors).
    try:
        zdim = int(membership.shape[2])
        sel = [z for z in range(zdim) if np.any(mask3d[:, :, z])]
        if not sel:
            sel = list(range(zdim))
        max_slices = int(os.getenv("P_BRAIN_ROI_DEBUG_MONTAGE_MAX_SLICES", "12") or "12")
        sel = sel[: max(1, min(max_slices, len(sel)))]

        cols = min(4, len(sel))
        rows = int(np.ceil(len(sel) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
        axes = np.atleast_1d(axes).ravel()

        pc_colors = ["red", "green", "blue", "purple", "orange", "cyan"]
        pc_colors = pc_colors[:k_pc]

        for ax_i, z in enumerate(sel):
            ax = axes[ax_i]
            ax.imshow(peak_amp[:, :, z], cmap="gray")
            m = membership[:, :, z]
            for pi in range(k_pc):
                mask = m == pi
                if not np.any(mask):
                    continue
                rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
                col = np.array(plt.matplotlib.colors.to_rgba(pc_colors[pi]), dtype=np.float32)
                rgba[..., 0:3] = col[0:3]
                rgba[..., 3] = mask.astype(np.float32) * 0.35
                ax.imshow(rgba)
            ax.set_title(f"Dominant global PC (z={z+1})")
            ax.axis("off")
        for j in range(len(sel), axes.size):
            axes[j].axis("off")

        handles = [plt.Line2D([0], [0], color=pc_colors[i], lw=3, label=f"PC{i+1}") for i in range(k_pc)]
        fig.legend(handles=handles, loc="upper center", ncol=k_pc, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(os.path.join(out_dir, "pca_global_pc_membership_montage.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        warnings.warn(f"Failed to write global PC membership montage: {exc}", RuntimeWarning)


def _maybe_run_segmentation_for_geometry(
    *,
    nifti_directory: str,
    filenames,
    parameters,
):
    """Attempt to generate the DCE-aligned atlas segmentation (FastSurfer path).

    Best-effort: if tools aren't installed, the caller should fall back.
    """

    try:
        from modules.AI_tissue_functions import segmentation as _segmentation
        from modules.AI_tissue_functions import coregistration as _coregistration
    except Exception:
        return

    (
        t1_3D_filename,
        _axial_t1_3D_filename,
        _t2_3D_filename,
        _axial_t2_3D_filename,
        _flair_3D_filename,
        _axial_flair_3D_filename,
        axial_t2_2D_filename,
        _diffusion_filename,
        dce_filename,
    ) = filenames

    (
        _is_vfa,
        _is_ir,
        apple_metal,
        _boundary,
        rerun_segmentation,
        segmentation_method,
        *_rest,
    ) = parameters

    fastsurfer_path = os.getenv(
        "P_BRAIN_FASTSURFER_PATH", "/Users/edt/FastSurfer/run_fastsurfer.sh"
    )
    t1_path = os.path.join(nifti_directory, t1_3D_filename)
    t2_path = os.path.join(nifti_directory, axial_t2_2D_filename)
    dce_path = os.path.join(nifti_directory, dce_filename)

    seg_dir = os.path.join(nifti_directory, "segmentation")
    sid = "segmentation"
    seg_mgz_path = os.path.join(seg_dir, sid, "mri", "aparc.DKTatlas+aseg.deep.mgz")

    if not os.path.isfile(t1_path) or not os.path.isfile(t2_path) or not os.path.isfile(dce_path):
        return

    os.makedirs(seg_dir, exist_ok=True)

    _segmentation(
        fastsurfer_path,
        seg_mgz_path,
        t1_path,
        seg_dir,
        sid,
        bool(apple_metal),
        bool(rerun_segmentation),
        segmentation_method,
    )
    _coregistration(seg_mgz_path=seg_mgz_path, dce_path=dce_path, t2_path=t2_path)


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


def _component_voxels_around_peak(
    score3d: np.ndarray,
    allowed3d: np.ndarray,
    z: int,
    *,
    pct: float = 99.5,
    min_area: int = 6,
    max_area: int = 500,
    dist_inside2d: np.ndarray | None = None,
    min_dist_inside: float | None = None,
) -> np.ndarray | None:
    """Return 2D voxel coords (x,y) for the best component around the slice peak."""

    allowed = allowed3d[:, :, z]
    if not np.any(allowed):
        return None
    s2 = np.where(allowed, score3d[:, :, z], 0.0)
    vals = s2[allowed]
    if vals.size < 32:
        return None
    thr = float(np.percentile(vals, float(pct)))
    cand = allowed & (s2 >= thr)
    if not np.any(cand):
        return None

    # Peak voxel within allowed area.
    peak_flat = int(np.argmax(np.where(allowed, score3d[:, :, z], -np.inf)))
    px, py = np.unravel_index(peak_flat, score3d[:, :, z].shape)

    lab, nlab = ndimage.label(cand)
    if nlab <= 0:
        return None

    peak_label = int(lab[px, py])
    order = [peak_label] + [i for i in range(1, nlab + 1) if i != peak_label and i != 0]
    best = None
    best_score = -np.inf
    for li in order:
        if li == 0:
            continue
        comp = lab == li
        n = int(comp.sum())
        if n < int(min_area) or n > int(max_area):
            continue
        if min_dist_inside is not None and dist_inside2d is not None:
            d = dist_inside2d[comp]
            if d.size and float(np.median(d)) < float(min_dist_inside):
                continue
        total = float(s2[comp].sum())
        if total > best_score:
            best_score = total
            best = comp
        # if peak component passes, prefer it immediately
        if li == peak_label and best is not None:
            break

    if best is None:
        return None
    return np.argwhere(best).astype(int)


def _component_voxels_near_seed(
    score3d: np.ndarray,
    allowed3d: np.ndarray,
    z: int,
    *,
    seed_xy: tuple[int, int],
    search_radius: int = 30,
    pct: float = 99.5,
    min_pct: float = 95.0,
    pct_step: float = 0.5,
    min_area: int = 6,
    max_area: int = 800,
    dist_inside2d: np.ndarray | None = None,
    min_dist_inside: float | None = None,
    max_boundary_contact: float = 0.20,
    edge_margin: int = 4,
) -> tuple[np.ndarray, float] | None:
    """Return (vox2d, score) for the best component near a seed point."""

    allowed = allowed3d[:, :, z]
    if not np.any(allowed):
        return None

    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    if search_radius > 0:
        x0 = max(0, sx - int(search_radius))
        x1 = min(allowed.shape[0] - 1, sx + int(search_radius))
        y0 = max(0, sy - int(search_radius))
        y1 = min(allowed.shape[1] - 1, sy + int(search_radius))
        win = np.zeros_like(allowed, dtype=bool)
        win[x0 : x1 + 1, y0 : y1 + 1] = True
        allowed = allowed & win

    if not np.any(allowed):
        return None

    s2 = np.where(allowed, score3d[:, :, z], 0.0)
    vals = s2[allowed]
    if vals.size < 32:
        return None

    p = float(pct)
    min_pct = min(float(p), float(min_pct))
    step = float(pct_step)
    if not np.isfinite(step) or step <= 0:
        step = 0.5

    boundary = (dist_inside2d <= 1.0) if dist_inside2d is not None else None
    best_vox = None
    best_score = -np.inf
    best_dist = float("inf")

    while p >= float(min_pct) - 1e-6:
        thr = float(np.percentile(vals, p))
        cand = allowed & (s2 >= thr)
        if not np.any(cand):
            p -= step
            continue
        lab, nlab = ndimage.label(cand)
        if nlab <= 0:
            p -= step
            continue

        for li in range(1, nlab + 1):
            comp = lab == li
            n = int(comp.sum())
            if n < int(min_area) or n > int(max_area):
                continue
            if min_dist_inside is not None and dist_inside2d is not None:
                d = dist_inside2d[comp]
                if d.size and float(np.median(d)) < float(min_dist_inside):
                    continue
            if boundary is not None and np.isfinite(max_boundary_contact) and max_boundary_contact > 0:
                contact = float(np.count_nonzero(comp & boundary)) / float(max(1, n))
                if contact > float(max_boundary_contact):
                    continue
            coords = np.argwhere(comp)
            if coords.size == 0:
                continue
            cx = float(coords[:, 0].mean())
            cy = float(coords[:, 1].mean())
            if edge_margin > 0:
                if (
                    cx < edge_margin
                    or cy < edge_margin
                    or cx > (allowed.shape[0] - 1 - edge_margin)
                    or cy > (allowed.shape[1] - 1 - edge_margin)
                ):
                    continue
            dist = float((cx - sx) ** 2 + (cy - sy) ** 2)
            total = float(s2[comp].sum())
            if not np.isfinite(total) or total <= 0:
                continue

            # Prefer the closest component to the seed; break ties by score.
            if dist < best_dist - 1e-6 or (abs(dist - best_dist) <= 1e-6 and total > best_score):
                best_dist = dist
                best_score = total
                best_vox = coords.astype(int)

        if best_vox is not None:
            break
        p -= step

    if best_vox is None or not np.isfinite(best_score):
        return None
    return best_vox, float(best_score)


def _select_ica_roi_voxels_by_slice(
    *,
    score3d: np.ndarray,
    allowed3d: np.ndarray,
    z_range: tuple[int, int],
    k_slices: int,
    dist_inside: np.ndarray | None = None,
    min_dist_inside: float | None = None,
    lr_axis_inplane: int | None = None,
    mid_lr: int | None = None,
    min_lr_from_midline: float | None = None,
) -> list[tuple[int, np.ndarray, float]]:
    """Return list of (z, vox2d[x,y], score) for ICA-like ROIs.

    Unlike the SSS, ICA ROIs may contain multiple disconnected components per slice.
    This routine unions the top components per slice after rejecting boundary pools
    and (optionally) midline-adjacent components (to avoid basilar).
    """

    z0, z1 = z_range
    out: list[tuple[int, np.ndarray, float]] = []
    k_slices = max(1, int(k_slices))
    pct = float(os.getenv("P_BRAIN_ROI_ICA_COMPONENT_PCT", "99.5") or "99.5")
    min_pct = float(os.getenv("P_BRAIN_ROI_ICA_COMPONENT_MIN_PCT", "95.0") or "95.0")
    step = float(os.getenv("P_BRAIN_ROI_ICA_COMPONENT_PCT_STEP", "0.5") or "0.5")
    min_pct = min(pct, min_pct)
    step = 0.5 if (not np.isfinite(step) or step <= 0) else step
    min_area = int(os.getenv("P_BRAIN_ROI_ICA_MIN_AREA", "8") or "8")
    max_area = int(os.getenv("P_BRAIN_ROI_ICA_MAX_AREA", "800") or "800")
    max_components = int(os.getenv("P_BRAIN_ROI_ICA_MAX_COMPONENTS_PER_SLICE", "3") or "3")
    max_components = max(1, min(10, max_components))
    max_contact = float(os.getenv("P_BRAIN_ROI_ICA_MAX_BOUNDARY_CONTACT", "0.20") or "0.20")
    edge_margin = int(os.getenv("P_BRAIN_ROI_ICA_EDGE_MARGIN", "4") or "4")
    edge_margin = max(0, min(40, edge_margin))

    # If not explicitly provided, default midline exclusion to a modest band.
    if min_lr_from_midline is None:
        min_lr_from_midline = float(os.getenv("P_BRAIN_ROI_ICA_MIN_LR_FROM_MIDLINE", "0") or "0")

    for z in range(z0, z1 + 1):
        allowed = allowed3d[:, :, z]
        if not np.any(allowed):
            continue

        # Adaptive thresholding to get candidate voxels.
        s2 = np.where(allowed, score3d[:, :, z], 0.0)
        vals = s2[allowed]
        if vals.size < 64:
            continue

        cand = None
        p = float(pct)
        while p >= float(min_pct) - 1e-6:
            thr = float(np.percentile(vals, p))
            cand = allowed & (s2 >= thr)
            if cand is not None and int(cand.sum()) >= int(min_area):
                break
            p -= float(step)
        if cand is None or not np.any(cand):
            continue

        lab, nlab = ndimage.label(cand)
        if nlab <= 0:
            continue

        boundary = None
        if dist_inside is not None:
            boundary = dist_inside[:, :, z] <= 1.0

        comps: list[tuple[float, np.ndarray]] = []
        for li in range(1, nlab + 1):
            comp = lab == li
            n = int(comp.sum())
            if n < int(min_area) or n > int(max_area):
                continue
            if min_dist_inside is not None and dist_inside is not None:
                d = dist_inside[:, :, z][comp]
                if d.size and float(np.median(d)) < float(min_dist_inside):
                    continue
            if boundary is not None and np.isfinite(max_contact) and max_contact > 0:
                contact = float(np.count_nonzero(comp & boundary)) / float(max(1, n))
                if contact > float(max_contact):
                    continue
            if (
                lr_axis_inplane is not None
                and mid_lr is not None
                and min_lr_from_midline is not None
                and float(min_lr_from_midline) > 0
            ):
                coords = np.argwhere(comp)
                if coords.size:
                    lr_cent = float(coords[:, 1].mean()) if int(lr_axis_inplane) == 1 else float(coords[:, 0].mean())
                    if abs(lr_cent - float(mid_lr)) < float(min_lr_from_midline):
                        continue
            # Reject components too close to image edges (common for peripheral pools).
            if edge_margin > 0:
                coords = np.argwhere(comp)
                if coords.size:
                    cx = float(coords[:, 0].mean())
                    cy = float(coords[:, 1].mean())
                    if (
                        cx < edge_margin
                        or cy < edge_margin
                        or cx > (score3d.shape[0] - 1 - edge_margin)
                        or cy > (score3d.shape[1] - 1 - edge_margin)
                    ):
                        continue

            total = float(s2[comp].sum())
            if np.isfinite(total) and total > 0:
                comps.append((total, np.argwhere(comp).astype(int)))

        if not comps:
            continue

        comps.sort(key=lambda t: t[0], reverse=True)
        picked = comps[: min(max_components, len(comps))]
        vox = np.concatenate([v for _s, v in picked], axis=0)
        # unique rows
        vox = np.unique(vox, axis=0)
        s = float(sum(_s for _s, _v in picked))
        out.append((int(z), vox, s))

    out.sort(key=lambda t: t[2], reverse=True)
    return out[: min(k_slices, len(out))]


def _save_roi_outputs(
    *,
    roi_type: str,
    roi_subtype: str,
    centers: list[tuple[int, int, int, float]],
    voxels_by_slice: dict[int, np.ndarray],
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

    # Clear stale per-slice files from previous runs.
    try:
        for fn in os.listdir(roi_data_dir):
            if fn.startswith("ROI_voxels_slice_") and fn.endswith(".npy"):
                os.remove(os.path.join(roi_data_dir, fn))
        for fn in os.listdir(frame_data_dir):
            if fn.startswith("frame_index_slice_") and fn.endswith(".npy"):
                os.remove(os.path.join(frame_data_dir, fn))
    except Exception:
        pass

    # Also keep the conventional ITC/CTC outputs to stay compatible with the pipeline.
    # The ROI geometry is the selected connected vessel region per slice, then dilated by N pixels.
    dilate_px = int(os.getenv("P_BRAIN_ROI_DILATE_PIXELS", "2") or "2")
    dilate_px = max(0, min(10, dilate_px))
    for x, y, z, _score in centers:
        vox0 = np.asarray(voxels_by_slice.get(int(z), np.empty((0, 2), dtype=int)), dtype=int)
        if vox0.ndim != 2 or vox0.shape[1] != 2 or vox0.shape[0] == 0:
            # Deterministic geometry ROIs must come from the connected regions found by
            # PCA/connected-component analysis (then dilated). Avoid circular fallbacks.
            continue

        m2 = np.zeros((height, width), dtype=bool)
        m2[vox0[:, 0], vox0[:, 1]] = True
        if dilate_px > 0:
            m2 = ndimage.binary_dilation(m2, iterations=dilate_px)
        vox = np.argwhere(m2).astype(int)
        mask3d[vox[:, 0], vox[:, 1], z] = 1

        # Store ROI voxels per slice (AI-compatible)
        np.save(
            os.path.join(roi_data_dir, f"ROI_voxels_slice_{z+1}.npy"),
            np.asarray(vox, dtype=np.int16),
        )

        # Peak frame for the selected ROI curve (used downstream as reference)
        # Keep consistent with utils.plotting.{plot_time_intensity_curves_AI, plot_time_intensity_curves_and_CTC_AI}.
        curve_method = (getattr(settings, "VASCULAR_ROI_CURVE_METHOD", "max") or "max").strip().lower()
        if curve_method not in {"max", "mean", "median"}:
            curve_method = "max"
        adaptive_enabled = bool(getattr(settings, "VASCULAR_ROI_ADAPTIVE_MAX", False))

        tc = dce4d[vox[:, 0], vox[:, 1], z, :]
        if tc.ndim == 2 and tc.shape[0] > 0:
            if curve_method == "mean":
                sel_tc = tc.mean(axis=0)
            elif curve_method == "median":
                sel_tc = np.median(tc, axis=0)
            elif curve_method == "max" and adaptive_enabled:
                # Adaptive max voxel: can be a different voxel per frame.
                sel_tc = tc.max(axis=0)
            else:
                # Fixed max-voxel curve: pick the voxel with the highest peak.
                idx = int(np.argmax(tc.max(axis=1)))
                sel_tc = tc[idx]
        else:
            sel_tc = dce4d[x, y, z, :]

        baseline_frames = int(getattr(settings, "ROI_DCE_BASELINE_FRAMES", 5))
        baseline = sel_tc[:baseline_frames].mean() if sel_tc.size else 0.0
        peak_frame = int(np.argmax(sel_tc - baseline)) if sel_tc.size else 0
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
            rotate_ac=False,
        )

    out_dir = os.path.join(analysis_dir, "ROI NIfTI")
    os.makedirs(out_dir, exist_ok=True)
    safe_type = roi_type.replace(" ", "_")
    safe_subtype = roi_subtype.replace(" ", "_")
    out_path = os.path.join(out_dir, f"{safe_type}__{safe_subtype}__mask.nii.gz")

    out_img = nib.Nifti1Image(mask3d.astype(np.uint8), affine=ref_img.affine, header=ref_img.header)
    nib.save(out_img, out_path)


def _write_input_function_rois_screenshot(
    *,
    analysis_directory: str,
    image_directory: str,
    peak_amp: np.ndarray,
) -> None:
    """Write the ROI overview PNG expected by p-brain-web.

    p-brain-web looks for `Images/AI/AI_input_function_ROIs.png`.
    Deterministic/geometry ROI selection historically did not write it, which
    caused the UI to show a stale (AI) screenshot or nothing.
    """

    if not image_directory:
        return

    out_dir = os.path.join(image_directory, "AI")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "AI_input_function_ROIs.png")

    roi_dir = os.path.join(analysis_directory, "ROI NIfTI")
    if not os.path.isdir(roi_dir):
        return

    # Load final masks (already dilated/cleaned) if present.
    sss_candidates = [
        os.path.join(roi_dir, "Vein__Sinus_Sagittalis__mask.nii.gz"),
        os.path.join(roi_dir, "Vein__Superior_Sagittal_Sinus__mask.nii.gz"),
    ]
    sss_path = next((p for p in sss_candidates if os.path.isfile(p)), None)
    roi_paths = {
        "rICA": os.path.join(roi_dir, "Artery__Right_Interior_Carotid__mask.nii.gz"),
        "lICA": os.path.join(roi_dir, "Artery__Left_Interior_Carotid__mask.nii.gz"),
        "SSS": sss_path,
        "Basilar": os.path.join(roi_dir, "Artery__Basilar__mask.nii.gz"),
    }

    masks: dict[str, np.ndarray] = {}
    for name, path in roi_paths.items():
        if not path or not os.path.isfile(path):
            continue
        try:
            m = nib.load(path).get_fdata() > 0.5
        except Exception:
            continue
        if m.ndim == 3:
            masks[name] = m.astype(bool)

    if not masks:
        return

    base = np.nan_to_num(peak_amp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    base = np.maximum(base, 0.0)

    # Pick slices that actually contain ROIs.
    zdim = int(base.shape[2])
    any_roi = np.zeros((zdim,), dtype=bool)
    for m in masks.values():
        if m.shape[2] == zdim:
            any_roi |= np.any(m, axis=(0, 1))
    sel = [z for z in range(zdim) if any_roi[z]]
    if not sel:
        sel = [int(zdim // 2)] if zdim > 0 else [0]

    max_slices = int(os.getenv("P_BRAIN_ROI_SCREENSHOT_MAX_SLICES", "12") or "12")
    sel = sel[: max(1, min(max_slices, len(sel)))]

    cols = min(4, len(sel))
    rows = int(np.ceil(len(sel) / cols))

    def _rot90_ccw(a: np.ndarray) -> np.ndarray:
        return np.rot90(a, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_i, z in enumerate(sel):
        ax = axes[ax_i]
        sl = _rot90_ccw(base[:, :, z])
        ax.imshow(sl, cmap="gray")

        def _overlay(mask2d: np.ndarray, rgba: tuple[float, float, float], alpha: float = 0.75) -> None:
            if not np.any(mask2d):
                return
            ov = np.zeros((mask2d.shape[0], mask2d.shape[1], 4), dtype=np.float32)
            ov[..., 0] = float(rgba[0])
            ov[..., 1] = float(rgba[1])
            ov[..., 2] = float(rgba[2])
            ov[..., 3] = mask2d.astype(np.float32) * float(alpha)
            ax.imshow(ov)

        # Match the deterministic summary palette.
        if "rICA" in masks and z < masks["rICA"].shape[2]:
            _overlay(_rot90_ccw(masks["rICA"][:, :, z]), (1.0, 0.0, 0.0))
        if "lICA" in masks and z < masks["lICA"].shape[2]:
            _overlay(_rot90_ccw(masks["lICA"][:, :, z]), (1.0, 0.55, 0.0))
        if "SSS" in masks and z < masks["SSS"].shape[2]:
            _overlay(_rot90_ccw(masks["SSS"][:, :, z]), (0.0, 0.0, 1.0))
        if "Basilar" in masks and z < masks["Basilar"].shape[2]:
            _overlay(_rot90_ccw(masks["Basilar"][:, :, z]), (0.0, 1.0, 0.0))

        ax.set_title(f"Input ROIs z={z+1}")
        ax.axis("off")

    for j in range(len(sel), axes.size):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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

    is_vfa, _is_ir, apple_metal, _boundary, rerun_segmentation, segmentation_method, *_rest = (
        parameters
    )

    if not dce_filename:
        try:
            from utils.parameters import _infer_dce_filename  # type: ignore

            dce_filename = _infer_dce_filename(nifti_directory)
        except Exception:
            dce_filename = None

    if not dce_filename:
        raise ValueError(
            f"Could not determine DCE NIfTI in {nifti_directory}. "
            "Expected a 4D NIfTI (x,y,z,t)."
        )

    dce_path = os.path.join(nifti_directory, dce_filename)

    # This file was produced by an older radius-based visualization step.
    # The deterministic geometry ROIs are region-based (2px dilated connected regions),
    # so remove any stale output to avoid confusion.
    try:
        stale = os.path.join(image_directory or "", "AI", "DCE_geometry_roi_radii.png")
        if stale and os.path.isfile(stale):
            os.remove(stale)
    except Exception:
        pass

    ref_img, dce4d = load_dce_4d(dce_path, prefer_complex_mag=True, dtype=np.float32)

    if dce4d.ndim != 4:
        raise ValueError(
            f"Expected DCE to be 4D (x,y,z,t); got shape {dce4d.shape} from {dce_path}"
        )

    n_volumes = int(dce4d.shape[-1])
    dt_s = resolve_dce_time_step_s(dce_path, default=None)
    time_points_s = build_time_points_s(n_volumes, dt_s)
    os.makedirs(os.path.join(analysis_directory, "Fitting"), exist_ok=True)
    np.save(os.path.join(analysis_directory, "Fitting", "time_points_s.npy"), time_points_s)

    cfg = GeometryRoiConfig(
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
    axes = _infer_lr_ap_axes_from_affine(ref_img.affine)

    # For in-plane LR/AP heuristics we only support axes 0/1; fallback to common axial layout.
    lr_axis_inplane = axes["lr_axis"] if axes["lr_axis"] in {0, 1} else 1
    lr_code = axes["lr_code"]
    ap_axis_inplane = axes["ap_axis"] if axes["ap_axis"] in {0, 1} else None
    ap_code = axes["ap_code"]

    default_rica, default_lica, default_sss = _default_z_bands(n_slices)

    rica_z = _parse_z_range(cfg.rica_z_range, n_slices, default_rica)
    lica_z = _parse_z_range(cfg.lica_z_range, n_slices, default_lica)
    sss_z = _parse_z_range(cfg.sss_z_range, n_slices, default_sss)

    # Skull stripping: prefer segmentation-derived intracranial mask.
    # Keep a *strict* mask for ROI selection (avoid peripheral pools), and a slightly
    # dilated mask for scoring (avoid excluding vessels at the boundary).
    brain_mask_strict = _load_segmentation_brain_mask(nifti_directory, ref_img)
    if brain_mask_strict is None:
        try:
            _maybe_run_segmentation_for_geometry(
                nifti_directory=nifti_directory,
                filenames=filenames,
                parameters=parameters,
            )
        except Exception:
            pass
        brain_mask_strict = _load_segmentation_brain_mask(nifti_directory, ref_img)

    if brain_mask_strict is None:
        brain_mask_strict = _brain_mask_from_mean(dce4d)

    # Dilate slightly to avoid excluding vessels at the boundary.
    dil_iters = int(os.getenv("P_BRAIN_ROI_SKULLSTRIP_DILATE", "2") or "2")
    brain_mask = _dilate_bool_mask(brain_mask_strict.astype(bool), dil_iters)

    # Hard skull-strip the DCE for all downstream scoring.
    dce4d = dce4d * brain_mask[..., None].astype(dce4d.dtype)

    # Optional: export a brain-only 4D concentration NIfTI and/or score in concentration space.
    score_space = (os.getenv("P_BRAIN_ROI_SCORE_SPACE", "signal") or "signal").strip().lower()
    write_ctc = _getenv_bool("P_BRAIN_WRITE_BRAIN_CTC_NIFTI", default=False)
    ctc_out_path = os.getenv("P_BRAIN_BRAIN_CTC_NIFTI_PATH")
    if not ctc_out_path:
        ctc_out_path = os.path.join(
            analysis_directory,
            "CTC Data",
            "Tissue",
            "brain_concentration_4d.nii.gz",
        )

    peak_amp = None
    ttp_idx = None
    if write_ctc or score_space == "concentration":
        try:
            peak_amp_ctc, ttp_idx_ctc = _write_brain_concentration_nifti(
                dce4d=dce4d,
                ref_img=ref_img,
                dce_path=dce_path,
                analysis_directory=analysis_directory,
                brain_mask=brain_mask,
                baseline_frames=cfg.baseline_frames,
                output_path=ctc_out_path if write_ctc else "",
                batch_voxels=int(os.getenv("P_BRAIN_CTC_BATCH_VOXELS", "20000") or "20000"),
            )
            if score_space == "concentration":
                peak_amp = peak_amp_ctc
                ttp_idx = ttp_idx_ctc
        except NotImplementedError as exc:
            if score_space == "concentration":
                raise
            warnings.warn(str(exc), RuntimeWarning)
        except Exception as exc:
            if score_space == "concentration":
                raise
            warnings.warn(f"Failed to export brain concentration NIfTI: {exc}", RuntimeWarning)

    if peak_amp is None or ttp_idx is None:
        peak_amp = _compute_peak_map(dce4d, cfg.baseline_frames)
        ttp_idx = _compute_ttp_map(dce4d, cfg.baseline_frames)

    # If tissue masks exist in DCE space, use them to constrain ICA search.
    # IMPORTANT: vessels may not be inside GM/WM masks; use a small dilation when available.
    _tissue_mask = _load_union_mask(_tissue_mask_candidates_in_dce(nifti_directory), ref_img)
    # NOTE: vessels are often not included in GM/WM tissue masks; use the intracranial mask
    # for ICA hemisphere splitting to avoid dropping true ICA candidates.
    ica_base_mask = brain_mask

    artery_t, vein_t, sigma = _derive_ttp_targets(peak_amp, ttp_idx, brain_mask, cfg.baseline_frames)
    # Enforce arteries earlier than veins by at least a couple frames.
    tdim = int(dce4d.shape[-1])
    min_sep = int(os.getenv("P_BRAIN_ROI_AV_MIN_SEP_FRAMES", "2") or "2")
    min_sep = max(1, min(50, min_sep))
    if vein_t < artery_t + min_sep:
        vein_t = min(max(artery_t + min_sep, artery_t + 1), max(artery_t + 1, tdim - 1))
    early_w = _gaussian_weight(ttp_idx, artery_t, sigma)
    late_w = _gaussian_weight(ttp_idx, vein_t, sigma)

    # Global PCA over high-amplitude intracranial voxels.
    use_pca = _getenv_bool("P_BRAIN_ROI_USE_PCA", default=True)
    pca_max_vox = int(os.getenv("P_BRAIN_ROI_PCA_MAX_VOXELS", "20000") or "20000")
    pca_k = int(os.getenv("P_BRAIN_ROI_PCA_COMPONENTS", "3") or "3")
    pca = _pca_global_scores(
        dce4d,
        brain_mask,
        peak_amp,
        cfg.baseline_frames,
        max_voxels=pca_max_vox,
        n_components=pca_k,
    ) if use_pca else None

    if pca is not None:
        pc_time, pc_maps = pca
        early_pc, late_pc = _pick_pc_by_peak(pc_time)
        artery_score = (pc_maps[early_pc] * peak_amp) * early_w
        vein_score = (pc_maps[late_pc] * peak_amp) * late_w

        if _getenv_bool("P_BRAIN_ROI_WRITE_GLOBAL_PC_MEMBERSHIP", default=True):
            try:
                _write_global_pc_membership_plots(
                    image_directory=image_directory,
                    ref_img=ref_img,
                    dce4d=dce4d,
                    brain_mask=brain_mask,
                    peak_amp=peak_amp,
                    baseline_frames=cfg.baseline_frames,
                    pc_time=pc_time,
                )
            except Exception as exc:
                warnings.warn(f"Failed to write global PC membership plots: {exc}", RuntimeWarning)
    else:
        # Fallback: amplitude * timing weights
        artery_score = peak_amp * early_w
        vein_score = peak_amp * late_w

    # Additional separation: arteries peak slightly earlier than veins.
    t_mid = 0.5 * (float(artery_t) + float(vein_t))
    split_scale = float(os.getenv("P_BRAIN_ROI_AV_TTP_LOGISTIC_SCALE", "2.0") or "2.0")
    w_early, w_late = _logistic_split_weight(ttp_idx, t_mid=t_mid, scale=split_scale)
    artery_score = artery_score * w_early
    vein_score = vein_score * w_late

    # Spatial prior: arteries anterior, veins posterior (use AP axis from affine when available).
    if ap_axis_inplane is not None and ap_code in {"A", "P"}:
        ap_dim = int(dce4d.shape[int(ap_axis_inplane)])
        a = np.linspace(0.0, 1.0, ap_dim, dtype=np.float32)
        # normalize so a1 is anterior
        a1 = a if ap_code == "A" else (1.0 - a)
        gamma = float(os.getenv("P_BRAIN_ROI_AP_GAMMA", "1.5") or "1.5")
        gamma = 1.0 if not np.isfinite(gamma) or gamma <= 0 else gamma
        anterior_1d = np.power(a1, gamma, dtype=np.float32)
        posterior_1d = np.power(1.0 - a1, gamma, dtype=np.float32)
        if int(ap_axis_inplane) == 0:
            anterior_w = anterior_1d[:, None, None]
            posterior_w = posterior_1d[:, None, None]
        else:
            anterior_w = anterior_1d[None, :, None]
            posterior_w = posterior_1d[None, :, None]
        artery_score = artery_score * anterior_w
        vein_score = vein_score * posterior_w
    else:
        anterior_w, posterior_w = _ap_prior_weights(int(dce4d.shape[0]))
        artery_score = artery_score * anterior_w
        vein_score = vein_score * posterior_w

    # Optional: write diagnostic PCA/curve-family and vesselness outputs.
    if _getenv_bool("P_BRAIN_ROI_DEBUG", default=False):
        strict = _getenv_bool("P_BRAIN_ROI_DEBUG_STRICT", default=False)
        if strict:
            _save_debug_outputs(
                analysis_directory=analysis_directory,
                image_directory=image_directory,
                ref_img=ref_img,
                dce_path=dce_path,
                dce4d=dce4d,
                brain_mask=brain_mask,
                cfg=cfg,
                artery_score=artery_score,
                vein_score=vein_score,
                peak_amp=peak_amp,
                ttp_idx=ttp_idx,
            )
        else:
            try:
                _save_debug_outputs(
                    analysis_directory=analysis_directory,
                    image_directory=image_directory,
                    ref_img=ref_img,
                    dce_path=dce_path,
                    dce4d=dce4d,
                    brain_mask=brain_mask,
                    cfg=cfg,
                    artery_score=artery_score,
                    vein_score=vein_score,
                    peak_amp=peak_amp,
                    ttp_idx=ttp_idx,
                )
            except Exception as exc:
                warnings.warn(f"ROI debug output failed: {exc}", RuntimeWarning)

    # Distance-to-boundary inside the (slightly dilated) intracranial mask.
    dist_inside = ndimage.distance_transform_edt(brain_mask.astype(bool)).astype(np.float32)

    band = max(1, int(cfg.sss_midline_band))

    right_mask, left_mask, mid_lr = _split_lr_masks(ica_base_mask.astype(bool), lr_axis=int(lr_axis_inplane), lr_code=lr_code)
    allowed_rica = right_mask
    allowed_lica = left_mask

    # Keep the basilar (midline artery) out of ICA ROIs.
    ica_mid_exclude = os.getenv("P_BRAIN_ROI_ICA_MIDLINE_EXCLUDE_BAND")
    if ica_mid_exclude is None or str(ica_mid_exclude).strip() == "":
        # Keep this independent of the SSS midline band (which may be wide).
        ica_mid_exclude_n = 12
    else:
        ica_mid_exclude_n = int(ica_mid_exclude)
    ica_mid_exclude_n = max(0, min(40, ica_mid_exclude_n))
    if ica_mid_exclude_n > 0:
        not_mid = ~_midline_band_mask(
            brain_mask_strict.shape,
            lr_axis=int(lr_axis_inplane),
            mid=int(mid_lr),
            band=int(ica_mid_exclude_n),
        )
        allowed_rica = allowed_rica & not_mid
        allowed_lica = allowed_lica & not_mid

    use_3d = _getenv_bool("P_BRAIN_ROI_USE_3D_COMPONENTS", default=True)
    keep_z_oriented = _getenv_bool("P_BRAIN_ROI_KEEP_Z_ORIENTED", default=True)
    z_align_min = float(os.getenv("P_BRAIN_ROI_Z_ALIGN_MIN", "0.55") or "0.55")
    z_align_min = 0.0 if (not np.isfinite(z_align_min) or z_align_min < 0) else min(1.0, z_align_min)

    # Filter out near-skull pools for ICA search.
    ica_min_dist = int(os.getenv("P_BRAIN_ROI_ICA_MIN_DIST_INNER", "3") or "3")
    ica_min_dist = max(0, min(20, ica_min_dist))
    if ica_min_dist > 0:
        allowed_rica = allowed_rica & (dist_inside >= float(ica_min_dist))
        allowed_lica = allowed_lica & (dist_inside >= float(ica_min_dist))

    # SSS must be near midline in LR, close to skull (distance small), and venous-like.
    midline = _midline_band_mask(brain_mask.shape, lr_axis=int(lr_axis_inplane), mid=int(mid_lr), band=int(band))
    allowed_sss = brain_mask.astype(bool) & midline

    rica_comps: list[tuple[int, np.ndarray, float]] = []
    lica_comps: list[tuple[int, np.ndarray, float]] = []
    basilar_comps: list[tuple[int, np.ndarray, float]] = []
    sss_comps: list[tuple[int, np.ndarray, float]] = []

    if use_3d:
        debug3d = _getenv_bool("P_BRAIN_ROI_DEBUG_3D", default=False)
        try:
            _zooms = tuple(float(z) for z in ref_img.header.get_zooms()[:3])
            if len(_zooms) != 3 or not all(np.isfinite(z) and z > 0 for z in _zooms):
                _zooms = (1.0, 1.0, 1.0)
        except Exception:
            _zooms = (1.0, 1.0, 1.0)

        peak_amp_f = np.nan_to_num(peak_amp, nan=0.0, posinf=0.0, neginf=0.0)
        artery_score_f = np.nan_to_num(artery_score, nan=0.0, posinf=0.0, neginf=0.0)
        vein_score_f = np.nan_to_num(vein_score, nan=0.0, posinf=0.0, neginf=0.0)

        # Build the same artery/vein candidate masks as the PCA visualization stage.
        # IMPORTANT: the final rICA/lICA/SSS selection must be a subset of these masks.
        vessel_mask, _v_pct = _adaptive_percentile_mask(
            peak_amp_f,
            brain_mask.astype(bool),
            start_pct=float(os.getenv("P_BRAIN_ROI_VESSEL_PCT", "99.0") or "99.0"),
            min_pct=float(os.getenv("P_BRAIN_ROI_VESSEL_MIN_PCT", "95.0") or "95.0"),
            step=0.5,
            min_vox=int(os.getenv("P_BRAIN_ROI_VESSEL_MIN_VOX", "1200") or "1200"),
        )

        seed_pct = float(os.getenv("P_BRAIN_ROI_AV_SEED_PCT", "99.5") or "99.5")
        seed_min_pct = float(os.getenv("P_BRAIN_ROI_AV_SEED_MIN_PCT", "95.0") or "95.0")
        seed_min_vox = int(os.getenv("P_BRAIN_ROI_AV_SEED_MIN_VOX", "200") or "200")

        artery_strength, _ = _adaptive_percentile_mask(
            artery_score_f,
            brain_mask.astype(bool),
            start_pct=seed_pct,
            min_pct=seed_min_pct,
            step=0.5,
            min_vox=seed_min_vox,
        )
        vein_strength, _ = _adaptive_percentile_mask(
            vein_score_f,
            brain_mask.astype(bool),
            start_pct=seed_pct,
            min_pct=seed_min_pct,
            step=0.5,
            min_vox=seed_min_vox,
        )
        artery_seed = artery_strength & vessel_mask & (artery_score_f >= vein_score_f)
        vein_seed = vein_strength & vessel_mask & (vein_score_f > artery_score_f)

        # Broader arterial candidate mask used for optional adjacent-slice ICA augmentation.
        # Less strict than artery_seed to allow small arterial voxels in z0+1..z0+2.
        artery_like = vessel_mask & (artery_score_f >= vein_score_f) & brain_mask.astype(bool)

        if debug3d:
            print(
                f"[ROI3D] vessel_pct={_v_pct:.1f} vessel_vox={int(vessel_mask.sum())} "
                f"artery_vox={int(artery_seed.sum())} vein_vox={int(vein_seed.sum())} zooms={_zooms}",
                flush=True,
            )

        # PCA directionality on component geometry: keep components whose principal axis has
        # sufficient alignment with z (slice direction). This rejects transverse arcs.
        def _pick_best_components(
            cand3d: np.ndarray,
            seed3d: np.ndarray,
            side: str,
            *,
            k: int,
            require_midline: bool,
            mid_band: int,
            min_lr_from_mid: int,
            min_z_span: int,
            max_boundary_contact: float,
            z_align_min_local: float,
        ) -> list[tuple[int, np.ndarray, float]]:
            lab, nlab = _iter_3d_components(cand3d)
            if debug3d:
                print(
                    f"[ROI3D] side={side} cand_vox={int(cand3d.sum())} comps={int(nlab)} "
                    f"mid_req={require_midline} mid_band={mid_band} min_lr_from_mid={min_lr_from_mid} "
                    f"min_z_span={min_z_span} z_align_min={z_align_min:.2f} max_bc={max_boundary_contact}",
                    flush=True,
                )
            comps = []
            raw = []
            for li in range(1, nlab + 1):
                # Score directionality on the full (dilated) component, but keep output voxels
                # strictly from the PCA-selected seed mask.
                coords_all = np.argwhere(lab == li).astype(int)
                if coords_all.shape[0] < int(os.getenv("P_BRAIN_ROI_COMP3D_MIN_VOX", "40") or "40"):
                    continue
                coords = np.argwhere((lab == li) & seed3d).astype(int)
                if coords.shape[0] == 0:
                    continue
                feat = _score_component_3d(
                    coords=coords_all,
                    score3d=(artery_score_f if side != "sss" else vein_score_f),
                    dist_inside=dist_inside,
                    spacing=_zooms,
                )
                if not np.isfinite(feat["total"]) or feat["total"] <= 0:
                    continue
                lr_cent = float(feat["centroid"][int(lr_axis_inplane)])
                dist_mid = abs(lr_cent - float(mid_lr))
                raw.append(
                    (
                        float(feat["total"]),
                        int(coords.shape[0]),
                        float(feat["z_align"]),
                        int(feat["z_span"]),
                        float(feat["boundary_contact"]),
                        float(dist_mid),
                    )
                )
                # LR position constraint
                if require_midline:
                    if dist_mid > float(mid_band):
                        continue
                else:
                    if dist_mid < float(min_lr_from_mid):
                        continue
                # Z-extent constraint
                if int(feat["z_span"]) < int(min_z_span):
                    continue
                # Boundary/pool rejection
                if np.isfinite(max_boundary_contact) and float(feat["boundary_contact"]) > float(max_boundary_contact):
                    continue
                # Directionality filter
                if keep_z_oriented and float(feat["z_align"]) < float(z_align_min_local):
                    continue
                comps.append((feat["total"], coords))

            if debug3d:
                if raw:
                    top_raw = sorted(raw, key=lambda t: t[0], reverse=True)[:5]
                    msg = ", ".join(
                        [
                            f"tot={t[0]:.1f} n={t[1]} zA={t[2]:.2f} zS={t[3]} bc={t[4]:.2f} dMid={t[5]:.1f}"
                            for t in top_raw
                        ]
                    )
                    print(f"[ROI3D] side={side} top_raw: {msg}", flush=True)
                print(f"[ROI3D] side={side} kept={len(comps)}", flush=True)

            comps.sort(key=lambda t: t[0], reverse=True)
            picked = comps[: min(int(k), len(comps))]
            out = []
            for _s, coords in picked:
                # break into per-slice voxel sets
                for z in np.unique(coords[:, 2]):
                    vox2 = coords[coords[:, 2] == z][:, :2]
                    out.append((int(z), vox2.astype(int), float(_s)))
            # merge duplicates by z (union voxels)
            byz: dict[int, list[np.ndarray]] = {}
            score_byz: dict[int, float] = {}
            for z, vox2, s in out:
                byz.setdefault(int(z), []).append(vox2)
                score_byz[int(z)] = max(float(score_byz.get(int(z), 0.0)), float(s))
            merged = []
            for z, lst in byz.items():
                vv = np.unique(np.concatenate(lst, axis=0), axis=0)
                merged.append((int(z), vv.astype(int), float(score_byz[z])))
            merged.sort(key=lambda t: t[2], reverse=True)
            return merged

        def _augment_adjacent_slices_from_lowest(
            comps: list[tuple[int, np.ndarray, float]],
            cand3d: np.ndarray,
            score3d: np.ndarray,
            *,
            brain_mask3d: np.ndarray,
            lr_axis_inplane: int,
            mid_lr: int,
            lr_code: str | None,
            side: str,
            max_above: int = 2,
        ) -> list[tuple[int, np.ndarray, float]]:
            """Add ICA voxels in slices above the lowest confirmed slice.

            For z in [z0+1, z0+max_above], if arterial candidates exist, pick the
            2D connected component whose centroid is closest to the centroid of the
            confirmed slice at z0. Candidates are restricted to the same LR side.

            If the candidate mask is empty in a slice, fall back to a high-percentile
            mask derived from the arterial score within the brain mask.
            """

            if not comps:
                return comps

            z0 = int(min(int(z) for z, _vox, _s in comps))
            base_vox = None
            base_score = None
            for z, vox, s in comps:
                if int(z) == z0:
                    base_vox = vox
                    base_score = float(s)
                    break
            if base_vox is None or base_vox.size == 0:
                return comps

            base_cx = float(base_vox[:, 0].mean())
            base_cy = float(base_vox[:, 1].mean())
            base_score = float(base_score) if base_score is not None else 0.0

            lr_axis_inplane = int(lr_axis_inplane)
            mid_lr = int(mid_lr)
            side = (side or "").strip().lower()
            want_right = side in {"rica", "right", "r"}
            right_is_high = (lr_code == "R")
            dbg_adj = _getenv_bool("P_BRAIN_ROI_DEBUG_ICA_ADJ", default=False)

            def _is_on_side(lr_cent: float) -> bool:
                if want_right:
                    return (lr_cent >= float(mid_lr)) if right_is_high else (lr_cent < float(mid_lr))
                return (lr_cent < float(mid_lr)) if right_is_high else (lr_cent >= float(mid_lr))

            present = {int(z) for z, _vox, _s in comps}
            out = list(comps)
            zmax = int(cand3d.shape[2])

            for dz in range(1, int(max_above) + 1):
                z = int(z0 + dz)
                if z < 0 or z >= zmax:
                    continue
                if z in present:
                    continue

                slab = cand3d[:, :, z].astype(bool)
                if not np.any(slab):
                    bm2 = brain_mask3d[:, :, z].astype(bool)
                    if np.any(bm2):
                        vals = score3d[:, :, z][bm2]
                        if vals.size > 0:
                            pct = float(os.getenv("P_BRAIN_ROI_ICA_ADJ_SCORE_PCT", "99.5") or "99.5")
                            pct = 99.5 if (not np.isfinite(pct)) else min(99.9, max(90.0, pct))
                            thr = float(np.percentile(vals, pct))
                            slab = (score3d[:, :, z] >= thr) & bm2

                if not np.any(slab):
                    if dbg_adj:
                        print(f"[ICA_ADJ] side={side} z={z} empty candidates", flush=True)
                    continue

                lab, nlab = ndimage.label(slab.astype(np.uint8))
                if int(nlab) <= 0:
                    if dbg_adj:
                        print(f"[ICA_ADJ] side={side} z={z} nlab=0", flush=True)
                    continue

                best_vox2 = None
                best_dist2 = None
                best_mean_score = None
                for li in range(1, int(nlab) + 1):
                    coords = np.argwhere(lab == li).astype(int)
                    if coords.shape[0] == 0:
                        continue
                    cx = float(coords[:, 0].mean())
                    cy = float(coords[:, 1].mean())

                    lr_cent = float(cx if lr_axis_inplane == 0 else cy)
                    if not _is_on_side(lr_cent):
                        continue

                    d2 = (cx - base_cx) ** 2 + (cy - base_cy) ** 2
                    if best_dist2 is None or d2 < best_dist2:
                        best_dist2 = float(d2)
                        best_vox2 = coords[:, :2].astype(int)
                        try:
                            best_mean_score = float(np.mean(score3d[coords[:, 0], coords[:, 1], z]))
                        except Exception:
                            best_mean_score = None

                if best_vox2 is None or best_vox2.size == 0:
                    if dbg_adj:
                        print(f"[ICA_ADJ] side={side} z={z} no component on-side", flush=True)
                    continue

                s = best_mean_score if best_mean_score is not None else base_score
                out.append((int(z), best_vox2, float(s)))
                present.add(int(z))
                if dbg_adj:
                    print(
                        f"[ICA_ADJ] side={side} added z={z} n={int(best_vox2.shape[0])} "
                        f"base=({base_cx:.1f},{base_cy:.1f})",
                        flush=True,
                    )

            return out

        # Basilar: midline artery, z-oriented
        basilar_band = int(os.getenv("P_BRAIN_ROI_BASILAR_MIDLINE_BAND", str(max(3, band))) or str(max(3, band)))
        basilar_band = max(1, min(40, basilar_band))
        bas_seed = artery_seed & brain_mask.astype(bool) & _midline_band_mask(
            brain_mask.shape, lr_axis=int(lr_axis_inplane), mid=int(mid_lr), band=int(basilar_band)
        )
        basilar3d = bas_seed
        basilar_comps = _pick_best_components(
            basilar3d,
            bas_seed,
            "basilar",
            k=int(os.getenv("P_BRAIN_ROI_BASILAR_SLICES", "1") or "1"),
            require_midline=True,
            mid_band=basilar_band,
            min_lr_from_mid=0,
            min_z_span=int(os.getenv("P_BRAIN_ROI_BASILAR_MIN_ZSPAN", "1") or "1"),
            max_boundary_contact=float(os.getenv("P_BRAIN_ROI_BASILAR_MAX_BOUNDARY_CONTACT", "0.50") or "0.50"),
            z_align_min_local=float(os.getenv("P_BRAIN_ROI_Z_ALIGN_MIN_BASILAR", str(z_align_min)) or str(z_align_min)),
        )

        # Rigidly prevent basilar voxels from being selected as ICA.
        # Basilar and ICA can otherwise overlap when the basilar centroid is a bit off-midline
        # (e.g. within basilar_band but outside ICA's smaller midline-exclusion band).
        basilar_exclude = None
        if basilar_comps:
            basilar_exclude = np.zeros(brain_mask.shape, dtype=bool)
            for z_b, vox2_b, _s_b in basilar_comps:
                vv = np.asarray(vox2_b, dtype=int)
                if vv.ndim == 2 and vv.shape[1] == 2 and vv.shape[0] > 0:
                    z_i = int(z_b)
                    if 0 <= z_i < basilar_exclude.shape[2]:
                        basilar_exclude[vv[:, 0], vv[:, 1], z_i] = True

        # ICAs: must come from PCA-selected artery seed; use a lightly dilated copy only
        # to connect across slices, then intersect back to the seed for voxel selection.
        ica_dilate = int(os.getenv("P_BRAIN_ROI_ICA_DILATE3D", "1") or "1")
        ica_dilate = max(0, min(3, ica_dilate))

        rica_seed = artery_seed & allowed_rica & brain_mask.astype(bool)
        lica_seed = artery_seed & allowed_lica & brain_mask.astype(bool)
        if basilar_exclude is not None:
            rica_seed = rica_seed & (~basilar_exclude)
            lica_seed = lica_seed & (~basilar_exclude)
        if ica_dilate > 0:
            st = ndimage.generate_binary_structure(3, 1)
            rica3d = ndimage.binary_dilation(rica_seed, structure=st, iterations=ica_dilate)
            lica3d = ndimage.binary_dilation(lica_seed, structure=st, iterations=ica_dilate)
        else:
            rica3d = rica_seed
            lica3d = lica_seed
        ica_zmin = float(os.getenv("P_BRAIN_ROI_Z_ALIGN_MIN_ICA", "0.0") or "0.0")
        ica_zmin = 0.0 if (not np.isfinite(ica_zmin) or ica_zmin < 0) else min(1.0, ica_zmin)
        rica_comps = _pick_best_components(
            rica3d,
            rica_seed,
            "rica",
            k=int(cfg.rica_slices),
            require_midline=False,
            mid_band=0,
            min_lr_from_mid=ica_mid_exclude_n,
            min_z_span=int(os.getenv("P_BRAIN_ROI_ICA_MIN_ZSPAN", "1") or "1"),
            max_boundary_contact=float(os.getenv("P_BRAIN_ROI_ICA_MAX_BOUNDARY_CONTACT", "0.20") or "0.20"),
            z_align_min_local=ica_zmin,
        )
        lica_comps = _pick_best_components(
            lica3d,
            lica_seed,
            "lica",
            k=int(cfg.lica_slices),
            require_midline=False,
            mid_band=0,
            min_lr_from_mid=ica_mid_exclude_n,
            min_z_span=int(os.getenv("P_BRAIN_ROI_ICA_MIN_ZSPAN", "1") or "1"),
            max_boundary_contact=float(os.getenv("P_BRAIN_ROI_ICA_MAX_BOUNDARY_CONTACT", "0.20") or "0.20"),
            z_align_min_local=ica_zmin,
        )

        # Optional augmentation: include arterial voxels in up to 2 slices above
        # the lowest confirmed ICA slice, if arterial signal is present there.
        if _getenv_bool("P_BRAIN_ROI_ICA_ADD_ADJACENT", default=True):
            # Default: add up to 20% of z-slices above the lowest confirmed ICA slice.
            # For zdim=10 this yields 2.
            raw_slices = os.getenv("P_BRAIN_ROI_ICA_ADJACENT_SLICES")
            if raw_slices is not None and str(raw_slices).strip() != "":
                max_above = int(raw_slices)
            else:
                frac = float(os.getenv("P_BRAIN_ROI_ICA_ADJACENT_FRAC", "0.2") or "0.2")
                frac = 0.2 if (not np.isfinite(frac) or frac <= 0) else min(1.0, frac)
                max_above = int(np.ceil(frac * float(int(brain_mask.shape[2]))))
            max_above = max(0, min(int(brain_mask.shape[2]) - 1, max_above))
            if max_above > 0:
                rica_comps = _augment_adjacent_slices_from_lowest(
                    rica_comps,
                    artery_like,
                    artery_score_f,
                    brain_mask3d=brain_mask.astype(bool),
                    lr_axis_inplane=int(lr_axis_inplane),
                    mid_lr=int(mid_lr),
                    lr_code=lr_code,
                    side="rica",
                    max_above=max_above,
                )
                lica_comps = _augment_adjacent_slices_from_lowest(
                    lica_comps,
                    artery_like,
                    artery_score_f,
                    brain_mask3d=brain_mask.astype(bool),
                    lr_axis_inplane=int(lr_axis_inplane),
                    mid_lr=int(mid_lr),
                    lr_code=lr_code,
                    side="lica",
                    max_above=max_above,
                )

        # SSS: venous midline component, z-oriented; still uses earlier transverse-reject logic.
        sss_seed = vein_seed & allowed_sss
        sss3d = sss_seed
        sss_zmin = float(os.getenv("P_BRAIN_ROI_Z_ALIGN_MIN_SSS", str(z_align_min)) or str(z_align_min))
        sss_zmin = 0.0 if (not np.isfinite(sss_zmin) or sss_zmin < 0) else min(1.0, sss_zmin)
        sss_comps = _pick_best_components(
            sss3d,
            sss_seed,
            "sss",
            k=int(cfg.sss_slices),
            require_midline=True,
            mid_band=band,
            min_lr_from_mid=0,
            min_z_span=int(os.getenv("P_BRAIN_ROI_SSS_MIN_ZSPAN", "2") or "2"),
            max_boundary_contact=float(os.getenv("P_BRAIN_SSS_MAX_BOUNDARY_CONTACT", "0.55") or "0.55"),
            z_align_min_local=sss_zmin,
        )

    else:
        # Legacy per-slice selection path.
        rica_comps = _select_ica_roi_voxels_by_slice(
            score3d=artery_score,
            allowed3d=allowed_rica,
            z_range=rica_z,
            k_slices=cfg.rica_slices,
            dist_inside=dist_inside,
            min_dist_inside=float(os.getenv("P_BRAIN_ROI_ICA_MIN_DIST_INNER", "3") or "3"),
            lr_axis_inplane=int(lr_axis_inplane),
            mid_lr=int(mid_lr),
            min_lr_from_midline=float(ica_mid_exclude_n),
        )
        lica_comps = _select_ica_roi_voxels_by_slice(
            score3d=artery_score,
            allowed3d=allowed_lica,
            z_range=lica_z,
            k_slices=cfg.lica_slices,
            dist_inside=dist_inside,
            min_dist_inside=float(os.getenv("P_BRAIN_ROI_ICA_MIN_DIST_INNER", "3") or "3"),
            lr_axis_inplane=int(lr_axis_inplane),
            mid_lr=int(mid_lr),
            min_lr_from_midline=float(ica_mid_exclude_n),
        )

        # SSS legacy component-by-slice selector
        sss_comps = _select_sss_roi_voxels_by_slice(
            score3d=vein_score,
            allowed3d=allowed_sss,
            z_range=sss_z,
            k_slices=cfg.sss_slices,
            lr_axis_inplane=int(lr_axis_inplane),
            mid_lr=int(mid_lr),
            band=band,
            dist_inside=dist_inside,
            ap_axis_inplane=ap_axis_inplane,
            ap_code=ap_code,
        )

        basilar_band = int(os.getenv("P_BRAIN_ROI_BASILAR_MIDLINE_BAND", str(max(3, band))) or str(max(3, band)))
        basilar_band = max(1, min(40, basilar_band))
        allowed_basilar = brain_mask.astype(bool) & _midline_band_mask(
            brain_mask.shape,
            lr_axis=int(lr_axis_inplane),
            mid=int(mid_lr),
            band=int(basilar_band),
        )
        basilar_slices = int(os.getenv("P_BRAIN_ROI_BASILAR_SLICES", "1") or "1")
        basilar_slices = max(1, min(10, basilar_slices))
        basilar_comps = _select_basilar_roi_voxels_by_slice(
            score3d=artery_score,
            allowed3d=allowed_basilar,
            z_range=rica_z,
            k_slices=basilar_slices,
            dist_inside=dist_inside,
            min_dist_inside=float(os.getenv("P_BRAIN_ROI_BASILAR_MIN_DIST_INNER", "2") or "2"),
        )

    # Convert component masks to (x,y,z,score) centers for downstream plotting and ROI saving.
    rica_centers = []
    rica_voxels_by_slice: dict[int, np.ndarray] = {}
    for z, vox, s in rica_comps:
        rica_voxels_by_slice[int(z)] = vox
        cx = int(round(float(vox[:, 0].mean())))
        cy = int(round(float(vox[:, 1].mean())))
        rica_centers.append((cx, cy, int(z), float(s)))
    lica_centers = []
    lica_voxels_by_slice: dict[int, np.ndarray] = {}
    for z, vox, s in lica_comps:
        lica_voxels_by_slice[int(z)] = vox
        cx = int(round(float(vox[:, 0].mean())))
        cy = int(round(float(vox[:, 1].mean())))
        lica_centers.append((cx, cy, int(z), float(s)))

    sss_centers = []
    sss_voxels_by_slice: dict[int, np.ndarray] = {}
    for z, vox, s in sss_comps:
        sss_voxels_by_slice[int(z)] = vox
        cx = int(round(float(vox[:, 0].mean())))
        cy = int(round(float(vox[:, 1].mean())))
        sss_centers.append((cx, cy, int(z), float(s)))
    basilar_centers = []
    basilar_voxels_by_slice: dict[int, np.ndarray] = {}
    for z, vox, s in basilar_comps:
        basilar_voxels_by_slice[int(z)] = vox
        cx = int(round(float(vox[:, 0].mean())))
        cy = int(round(float(vox[:, 1].mean())))
        basilar_centers.append((cx, cy, int(z), float(s)))

    # Persist outputs in the same on-disk format as the existing AI pipeline.
    _save_roi_outputs(
        roi_type="Artery",
        roi_subtype="Right Interior Carotid",
        centers=rica_centers,
        voxels_by_slice=rica_voxels_by_slice,
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
        voxels_by_slice=lica_voxels_by_slice,
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
        voxels_by_slice=sss_voxels_by_slice,
        dce4d=dce4d,
        ref_img=ref_img,
        analysis_dir=analysis_directory,
        image_dir=image_directory,
        nifti_dir=nifti_directory,
        time_points_s=time_points_s,
        filenames=filenames,
        is_vfa=bool(is_vfa),
    )

    if basilar_centers:
        _save_roi_outputs(
            roi_type="Artery",
            roi_subtype="Basilar",
            centers=basilar_centers,
            voxels_by_slice=basilar_voxels_by_slice,
            dce4d=dce4d,
            ref_img=ref_img,
            analysis_dir=analysis_directory,
            image_dir=image_directory,
            nifti_dir=nifti_directory,
            time_points_s=time_points_s,
            filenames=filenames,
            is_vfa=bool(is_vfa),
        )

    # Write the screenshot used by p-brain-web for the ROI stage.
    try:
        _write_input_function_rois_screenshot(
            analysis_directory=analysis_directory,
            image_directory=image_directory,
            peak_amp=peak_amp,
        )
    except Exception:
        pass

    if _getenv_bool("P_BRAIN_ROI_WRITE_SUMMARY_PLOTS", default=True):
        try:
            if _getenv_bool("P_BRAIN_ROI_WRITE_PCA_PLOTS", default=True):
                _write_pca_and_normalization_plots(
                    analysis_directory=analysis_directory,
                    image_directory=image_directory,
                    ref_img=ref_img,
                    dce_path=dce_path,
                    dce4d=dce4d,
                    brain_mask=brain_mask,
                    cfg=cfg,
                    peak_amp=peak_amp,
                )
            _write_summary_images(
                analysis_directory=analysis_directory,
                image_directory=image_directory,
                ref_img=ref_img,
                brain_mask=brain_mask,
                peak_amp=peak_amp,
                artery_score=artery_score,
                vein_score=vein_score,
            )
        except Exception as exc:
            warnings.warn(f"Failed to write Images/ summary plots: {exc}", RuntimeWarning)

    # Propagate final ROIs into the next step: TSCC generation (SSS time shift + rescale)
    # so the modelling pipeline can use SSS-derived AIF when enabled.
    if bool(getattr(settings, "INPUT_FUNCTION_USE_SSS", True)):
        try:
            tscc.time_shifting(
                analysis_directory,
                nifti_directory,
                image_directory,
                artery=getattr(settings, "INPUT_FUNCTION_ARTERY", "RICA"),
            )
        except Exception as exc:
            warnings.warn(f"Time shifting failed: {exc}", RuntimeWarning)

    # Return simple metadata for debugging (not used by the existing pipeline).
    return {
        "rICA": rica_centers,
        "lICA": lica_centers,
        "SSS": sss_centers,
        "z_ranges": {
            "rICA": rica_z,
            "lICA": lica_z,
            "SSS": sss_z,
        },
    }
