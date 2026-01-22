import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import label as cc_label
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import binary_erosion
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import label as cc_label_nd
except Exception:  # pragma: no cover
    distance_transform_edt = None
    cc_label = None
    binary_dilation = None
    binary_erosion = None
    gaussian_filter = None
    cc_label_nd = None

try:
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
except Exception:  # pragma: no cover
    hessian_matrix = None
    hessian_matrix_eigvals = None

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


def _compute_upslope_and_late(
    dce4d: np.ndarray, baseline_frames: int, late_frames: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    baseline_frames = max(1, int(baseline_frames))
    late_frames = max(1, int(late_frames))
    baseline = dce4d[..., :baseline_frames].mean(axis=-1)
    sig = (dce4d - baseline[..., None]).astype(np.float32)

    # Max positive frame-to-frame change
    diff = np.diff(sig, axis=-1)
    upslope = np.maximum(0.0, diff.max(axis=-1)).astype(np.float32)

    # Late tail magnitude
    late = sig[..., -late_frames:].mean(axis=-1).astype(np.float32)
    return upslope, late


def _compute_bat_and_auc(
    dce4d: np.ndarray,
    baseline_frames: int,
    *,
    bat_mad_k: float = 3.0,
    bat_persist: int = 2,
    early_window: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bolus arrival time (BAT) and early-fraction map.

    BAT is the first frame where enhancement exceeds baseline + k*MAD for
    `bat_persist` consecutive frames. AUC_early is integrated from BAT to
    BAT+early_window. early_fraction = AUC_early / AUC_total.
    """

    baseline_frames = max(3, int(baseline_frames))
    bat_persist = max(1, int(bat_persist))
    early_window = max(2, int(early_window))

    base = dce4d[..., :baseline_frames].astype(np.float32)
    baseline = np.median(base, axis=-1)
    mad = np.median(np.abs(base - baseline[..., None]), axis=-1) + 1e-6
    enh = (dce4d.astype(np.float32) - baseline[..., None])

    thr = baseline + (float(bat_mad_k) * mad)
    above = enh > (thr[..., None] - baseline[..., None])  # compare in enhancement domain

    # Persistence: require N consecutive True values.
    # Compute running sum over time of above-mask.
    run = np.zeros_like(enh, dtype=np.int16)
    run[..., 0] = above[..., 0].astype(np.int16)
    for t in range(1, above.shape[-1]):
        run[..., t] = (run[..., t - 1] + 1) * above[..., t].astype(np.int16)
    arrived = run >= bat_persist

    # BAT index: first time arrived becomes True, else sentinel = last frame.
    bat = np.full(enh.shape[:-1], enh.shape[-1] - 1, dtype=np.int16)
    any_arr = arrived.any(axis=-1)
    if np.any(any_arr):
        bat_idx = np.argmax(arrived, axis=-1).astype(np.int16)
        bat[any_arr] = bat_idx[any_arr]

    # AUC total and early AUC (BAT-relative)
    enh_pos = np.maximum(0.0, enh).astype(np.float32)
    csum = np.cumsum(enh_pos, axis=-1, dtype=np.float32)
    auc_total = csum[..., -1].astype(np.float32)

    T = int(enh_pos.shape[-1])
    bat_i = bat.astype(np.int32)
    t0 = np.clip(bat_i, 0, T - 1)
    t1 = np.clip(bat_i + early_window, 1, T)

    idx1 = (t1 - 1)[..., None]
    s1 = np.take_along_axis(csum, idx1, axis=-1)[..., 0]

    idx0 = np.clip(t0 - 1, 0, T - 1)[..., None]
    s0 = np.take_along_axis(csum, idx0, axis=-1)[..., 0]
    s0 = np.where(t0 > 0, s0, 0.0).astype(np.float32)

    auc_early = (s1 - s0).astype(np.float32)
    early_frac = (auc_early / (auc_total + 1e-6)).astype(np.float32)
    return bat.astype(np.int16), auc_total, early_frac


def _blobness2d(img2d: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Blobness for bright blob-like cross-sections.

    Uses Hessian eigenvalues; returns 0..1-ish.
    """

    x = img2d.astype(np.float32)
    if hessian_matrix is not None and hessian_matrix_eigvals is not None:
        H = hessian_matrix(x, sigma=sigma, order="rc", use_gaussian_derivatives=False)
        l1, l2 = hessian_matrix_eigvals(H)
        l1 = l1.astype(np.float32)
        l2 = l2.astype(np.float32)
    elif gaussian_filter is not None:
        # Fallback: compute Hessian via gaussian_filter derivatives
        Ixx = gaussian_filter(x, sigma=sigma, order=(2, 0))
        Iyy = gaussian_filter(x, sigma=sigma, order=(0, 2))
        Ixy = gaussian_filter(x, sigma=sigma, order=(1, 1))
        tr = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        disc = np.maximum(0.0, tr * tr - 4.0 * det)
        sdisc = np.sqrt(disc)
        l1 = 0.5 * (tr + sdisc)
        l2 = 0.5 * (tr - sdisc)
    else:
        return np.zeros_like(x, dtype=np.float32)

    # For bright blobs in a dark background, both eigenvalues tend negative.
    l1n = -l1
    l2n = -l2
    lpos = (l1n > 0) & (l2n > 0)
    # Similar magnitude => blob, not line
    ratio = np.minimum(l1n, l2n) / (np.maximum(l1n, l2n) + 1e-6)
    strength = np.sqrt(l1n * l2n)
    out = np.zeros_like(x, dtype=np.float32)
    out[lpos] = (ratio[lpos] * strength[lpos]).astype(np.float32)
    # Normalize within slice
    m = np.isfinite(out)
    if np.any(m):
        hi = float(np.percentile(out[m], 99.5))
        if hi > 0:
            out = np.clip(out / hi, 0.0, 1.0)
    return out.astype(np.float32)


@dataclass(frozen=True)
class _Component3D:
    id: int
    mass: float
    nvox: int
    centroid_ijk: tuple[float, float, float]
    centroid_ras_mm: tuple[float, float, float]
    mean_bat: float
    mean_early_frac: float
    mean_blob: float
    elongation: float
    z_min: int
    z_max: int
    voxels: np.ndarray  # (N,3) int ijk


def _axis_world_coords(affine: np.ndarray, shape3: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 1D world-coordinate axes sampled along each index axis.

    Uses the volume center for the other two axes so the result is stable
    for near-axial acquisitions.
    """

    nx, ny, nz = (int(shape3[0]), int(shape3[1]), int(shape3[2]))
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    cz = (nz - 1) / 2.0

    xs = np.zeros(nx, dtype=np.float32)
    ys = np.zeros(ny, dtype=np.float32)
    zs = np.zeros(nz, dtype=np.float32)

    for i in range(nx):
        p = affine @ np.array([float(i), cy, cz, 1.0], dtype=np.float32)
        xs[i] = float(p[0])
    for j in range(ny):
        p = affine @ np.array([cx, float(j), cz, 1.0], dtype=np.float32)
        ys[j] = float(p[1])
    for k in range(nz):
        p = affine @ np.array([cx, cy, float(k), 1.0], dtype=np.float32)
        zs[k] = float(p[2])

    return xs, ys, zs


def _cluster_top_components(
    score3d: np.ndarray,
    allowed3d: np.ndarray,
    affine: np.ndarray,
    bat_map: np.ndarray,
    early_frac: np.ndarray,
    blob3d: np.ndarray,
    *,
    q: float = 99.7,
    max_components: int = 30,
    min_vox: int = 20,
) -> list[_Component3D]:
    if cc_label_nd is None:
        return []
    m = allowed3d & np.isfinite(score3d) & (score3d > 0)
    if not np.any(m):
        return []
    vals = score3d[m]
    thr = float(np.percentile(vals, np.clip(q, 50.0, 99.99)))
    bw = m & (score3d >= thr)
    if not np.any(bw):
        return []

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lbl, n = cc_label_nd(bw.astype(np.uint8), structure=structure)
    if n <= 0:
        return []

    comps: list[_Component3D] = []
    for cid in range(1, n + 1):
        vox = np.argwhere(lbl == cid)
        if vox.size == 0:
            continue
        nvox = int(vox.shape[0])
        if nvox < int(min_vox):
            continue

        w = score3d[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.float32)
        wsum = float(w.sum())
        if not np.isfinite(wsum) or wsum <= 0:
            continue

        ijk = vox.astype(np.float32)
        cijk = (ijk * w[:, None]).sum(axis=0) / max(1e-6, wsum)
        cras = (affine @ np.array([float(cijk[0]), float(cijk[1]), float(cijk[2]), 1.0], dtype=np.float32))[:3]

        mbat = float((bat_map[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.float32) * w).sum() / max(1e-6, wsum))
        mef = float((early_frac[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.float32) * w).sum() / max(1e-6, wsum))
        mbl = float((blob3d[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.float32) * w).sum() / max(1e-6, wsum))

        cen = ijk - cijk[None, :]
        cov = (cen * w[:, None]).T @ cen / max(1e-6, wsum)
        try:
            evals = np.linalg.eigvalsh(cov.astype(np.float64))
            evals = np.sort(np.maximum(evals, 1e-9))
            elong = float(np.sqrt(evals[-1] / evals[0]))
        except Exception:
            elong = 1.0

        zmin = int(vox[:, 2].min())
        zmax = int(vox[:, 2].max())

        comps.append(
            _Component3D(
                id=int(cid),
                mass=float(wsum),
                nvox=nvox,
                centroid_ijk=(float(cijk[0]), float(cijk[1]), float(cijk[2])),
                centroid_ras_mm=(float(cras[0]), float(cras[1]), float(cras[2])),
                mean_bat=mbat,
                mean_early_frac=mef,
                mean_blob=mbl,
                elongation=elong,
                z_min=zmin,
                z_max=zmax,
                voxels=vox.astype(np.int32),
            )
        )

    comps.sort(key=lambda c: c.mass, reverse=True)
    return comps[: max(1, int(max_components))]


def _centers_from_component(
    comp: _Component3D,
    score3d: np.ndarray,
    *,
    k_slices: int,
) -> list[tuple[int, int, int, float]]:
    if k_slices <= 0:
        return []
    vox = comp.voxels
    zs = vox[:, 2]
    out: list[tuple[int, int, int, float]] = []
    uniq = np.unique(zs)
    slice_items: list[tuple[int, float]] = []
    for z in uniq:
        m = zs == z
        vv = vox[m]
        s = score3d[vv[:, 0], vv[:, 1], vv[:, 2]].astype(np.float32)
        slice_items.append((int(z), float(s.sum())))
    slice_items.sort(key=lambda t: t[1], reverse=True)
    h, w, _nz = score3d.shape
    for z, smass in slice_items[: min(len(slice_items), int(k_slices))]:
        m = zs == z
        vv = vox[m]
        s = score3d[vv[:, 0], vv[:, 1], vv[:, 2]].astype(np.float32)
        wsum = float(s.sum())
        if wsum <= 0:
            continue
        cx = float((vv[:, 0].astype(np.float32) * s).sum() / wsum)
        cy = float((vv[:, 1].astype(np.float32) * s).sum() / wsum)
        xi = int(np.clip(int(round(cx)), 0, h - 1))
        yi = int(np.clip(int(round(cy)), 0, w - 1))
        out.append((xi, yi, int(z), float(smass)))
    out.sort(key=lambda t: t[3], reverse=True)
    return out


def _voxels_by_z_from_component(
    comp: _Component3D,
    *,
    z_keep: set[int],
) -> dict[int, np.ndarray]:
    vox = comp.voxels
    out: dict[int, np.ndarray] = {}
    for z in z_keep:
        m = vox[:, 2] == int(z)
        if not np.any(m):
            continue
        out[int(z)] = vox[m][:, :2].astype(int)
    return out


def _robust_minmax01(x: np.ndarray, mask: np.ndarray, p_low: float = 5.0, p_high: float = 99.0) -> np.ndarray:
    m = mask & np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[m]
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    out = (x.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _component_candidates(
    prob2d: np.ndarray,
    allowed2d: np.ndarray,
    *,
    q: float,
    min_area: int,
    max_area: int,
    ecc_ratio_max: float,
    max_candidates: int = 8,
) -> list[tuple[int, int, float, int]]:
    """Return candidate centers from small, roughly-round high-probability blobs.

    Each candidate is (x, y, score, area) where score is probability mass.
    """

    if cc_label is None:
        return []
    m = allowed2d & np.isfinite(prob2d) & (prob2d > 0)
    if not np.any(m):
        return []

    vals = prob2d[m]
    thr = float(np.percentile(vals, np.clip(q, 50.0, 99.99)))
    bw = m & (prob2d >= thr)
    if not np.any(bw):
        return []

    lbl, n = cc_label(bw.astype(np.uint8))
    if n <= 0:
        return []

    min_area = max(1, int(min_area))
    max_area = max(min_area, int(max_area))
    ecc_ratio_max = float(max(1.0, ecc_ratio_max))

    cands: list[tuple[int, int, float, int]] = []
    for i in range(1, n + 1):
        coords = np.argwhere(lbl == i)
        if coords.size == 0:
            continue
        area = int(coords.shape[0])
        if area < min_area or area > max_area:
            continue

        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        dx = int(x1 - x0 + 1)
        dy = int(y1 - y0 + 1)
        if min(dx, dy) <= 0:
            continue
        ratio = float(max(dx, dy) / max(1, min(dx, dy)))
        if ratio > ecc_ratio_max:
            continue

        # Center of mass (rounded) and probability mass as score.
        cx = int(np.round(coords[:, 0].mean()))
        cy = int(np.round(coords[:, 1].mean()))
        cx = int(np.clip(cx, 0, prob2d.shape[0] - 1))
        cy = int(np.clip(cy, 0, prob2d.shape[1] - 1))
        score = float(prob2d[coords[:, 0], coords[:, 1]].sum())
        cands.append((cx, cy, score, area))

    cands.sort(key=lambda t: t[2], reverse=True)
    return cands[: max(1, int(max_candidates))]


def _perimeter_area_ratio(bw: np.ndarray) -> float:
    if binary_erosion is None:
        return 0.0
    bw = bw.astype(bool)
    if bw.sum() == 0:
        return 0.0
    # Approximate perimeter via boundary pixels
    eroded = binary_erosion(bw)
    boundary = bw & (~eroded)
    per = float(boundary.sum())
    area = float(bw.sum())
    return per / max(1.0, area)


def _component_voxels_from_threshold(
    prob2d: np.ndarray,
    allowed2d: np.ndarray,
    *,
    q: float,
    pick: tuple[int, int],
    dilate_r: int = 0,
) -> np.ndarray:
    """Return voxel coords for the connected component containing `pick`.

    Falls back to a small disk if CC labelling isn't available.
    """
    x, y = int(pick[0]), int(pick[1])
    if cc_label is None:
        return _disk_voxels(x, y, max(1, dilate_r), width=prob2d.shape[1], height=prob2d.shape[0])

    m = allowed2d & np.isfinite(prob2d) & (prob2d > 0)
    if not np.any(m):
        return _disk_voxels(x, y, max(1, dilate_r), width=prob2d.shape[1], height=prob2d.shape[0])
    vals = prob2d[m]
    thr = float(np.percentile(vals, np.clip(q, 50.0, 99.99)))
    bw = m & (prob2d >= thr)
    if not np.any(bw):
        return _disk_voxels(x, y, max(1, dilate_r), width=prob2d.shape[1], height=prob2d.shape[0])

    lbl, n = cc_label(bw.astype(np.uint8))
    if n <= 0:
        return _disk_voxels(x, y, max(1, dilate_r), width=prob2d.shape[1], height=prob2d.shape[0])
    comp_id = int(lbl[x, y])
    if comp_id <= 0:
        # pick the max-probability component if click missed
        flat = int(np.argmax(np.where(bw, prob2d, -1.0)))
        x2, y2 = np.unravel_index(flat, prob2d.shape)
        comp_id = int(lbl[int(x2), int(y2)])
        if comp_id <= 0:
            return _disk_voxels(x, y, max(1, dilate_r), width=prob2d.shape[1], height=prob2d.shape[0])

    comp = lbl == comp_id
    if dilate_r > 0 and binary_dilation is not None:
        r = int(dilate_r)
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        se = (xx**2 + yy**2) <= (r**2)
        comp = binary_dilation(comp, structure=se)
    coords = np.argwhere(comp)
    return coords.astype(int)


def _select_coherent_centers(
    candidates_by_z: dict[int, list[tuple[int, int, float, int]]],
    *,
    k_slices: int,
    coherence_px: float,
) -> list[tuple[int, int, int, float]]:
    if k_slices <= 0:
        return []
    all_cands: list[tuple[int, int, int, float, int]] = []
    for z, cands in candidates_by_z.items():
        for x, y, s, area in cands:
            all_cands.append((x, y, int(z), float(s), int(area)))
    if not all_cands:
        return []

    # Seed with the globally best candidate.
    all_cands.sort(key=lambda t: t[3], reverse=True)
    sx, sy, sz, ss, _sa = all_cands[0]
    selected: dict[int, tuple[int, int, int, float]] = {int(sz): (int(sx), int(sy), int(sz), float(ss))}

    # Grow by taking best candidates in other slices within coherence distance.
    coh = float(max(5.0, coherence_px))
    for _ in range(3):
        ref_x = float(np.median([v[0] for v in selected.values()]))
        ref_y = float(np.median([v[1] for v in selected.values()]))
        for z, cands in candidates_by_z.items():
            if z in selected:
                continue
            best = None
            best_s = -1.0
            for x, y, s, _area in cands:
                d = float(np.hypot(float(x) - ref_x, float(y) - ref_y))
                if d <= coh and s > best_s:
                    best_s = float(s)
                    best = (int(x), int(y), int(z), float(s))
            if best is not None:
                selected[int(z)] = best
        if len(selected) >= k_slices:
            break
        coh *= 1.6

    # If still not enough, fill with highest-score remaining candidates (unique slices).
    if len(selected) < k_slices:
        for x, y, z, s, _area in all_cands:
            if z in selected:
                continue
            selected[int(z)] = (int(x), int(y), int(z), float(s))
            if len(selected) >= k_slices:
                break

    out = list(selected.values())
    out.sort(key=lambda t: t[3], reverse=True)
    return out[:k_slices]


def _centrality_map01(brain_mask2d: np.ndarray) -> np.ndarray:
    if distance_transform_edt is None:
        return np.ones(brain_mask2d.shape, dtype=np.float32)
    dist = distance_transform_edt(brain_mask2d.astype(bool)).astype(np.float32)
    if dist.max() <= 0:
        return np.ones(brain_mask2d.shape, dtype=np.float32)
    return np.clip(dist / dist.max(), 0.0, 1.0)


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


def _slice_probability(
    logits2d: np.ndarray,
    allowed2d: np.ndarray,
    centrality2d: np.ndarray | None = None,
    gamma: float = 0.0,
) -> np.ndarray:
    m = allowed2d & np.isfinite(logits2d)
    if not np.any(m):
        return np.zeros_like(logits2d, dtype=np.float32)
    w = np.exp(np.clip(logits2d.astype(np.float32), -20.0, 20.0))
    if centrality2d is not None and gamma > 0:
        w = w * np.power(np.clip(centrality2d, 0.0, 1.0) + 1e-6, float(gamma))
    w = np.where(m, w, 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.zeros_like(logits2d, dtype=np.float32)
    return (w / s).astype(np.float32)


def _radius_from_probability_mass(
    prob2d: np.ndarray,
    cx: int,
    cy: int,
    r_min: int,
    r_max: int,
    target_mass: float,
) -> int:
    r_min = max(1, int(r_min))
    r_max = max(r_min, int(r_max))
    target_mass = float(np.clip(target_mass, 0.01, 0.95))
    height, width = prob2d.shape
    for r in range(r_min, r_max + 1):
        coords = _disk_voxels(cx, cy, r, width=width, height=height)
        mass = float(prob2d[coords[:, 0], coords[:, 1]].sum())
        if mass >= target_mass:
            return int(r)
    return int(r_max)


def _save_roi_outputs(
    *,
    roi_type: str,
    roi_subtype: str,
    centers: list[tuple[int, int, int, float]],
    radius: int,
    radius_by_z: dict[int, int] | None = None,
    voxels_by_z: dict[int, np.ndarray] | None = None,
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
        if voxels_by_z and int(z) in voxels_by_z:
            vox = voxels_by_z[int(z)]
        else:
            r_eff = int(radius_by_z.get(int(z), radius)) if radius_by_z else int(radius)
            vox = _disk_voxels(x, y, r_eff, width=width, height=height)
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


def _save_probability_overlay(
    *,
    dce4d: np.ndarray,
    centers_by_label: dict[str, list[tuple[int, int, int, float]]],
    radii_by_z_by_label: dict[str, dict[int, int]],
    prob3d_by_label: dict[str, np.ndarray],
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

    for idx, (label, (x, y, z, score)) in enumerate(panels):
        ax = axes[idx // cols, idx % cols]
        ax.imshow(mean_img[:, :, z], cmap="gray", origin="lower")
        p = prob3d_by_label[label][:, :, z]
        ax.imshow(p, cmap="magma", origin="lower", alpha=0.55, vmin=0.0, vmax=float(np.percentile(p[p > 0], 99)) if np.any(p > 0) else 1.0)
        r = int(radii_by_z_by_label.get(label, {}).get(int(z), 0))
        if r > 0:
            circ = plt.Circle((y, x), r, fill=False, color="cyan", linewidth=2)
            ax.add_patch(circ)
        ax.set_title(f"{label} z={z+1} p={score:.3g}")
        ax.axis("off")

    for idx in range(len(panels), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.join(image_dir, "AI"), exist_ok=True)
    plt.savefig(os.path.join(image_dir, "AI", "DCE_geometry_roi_probability.png"), dpi=300)
    plt.close(fig)


def _save_ttp_overlay(
    *,
    dce4d: np.ndarray,
    ttp_map: np.ndarray,
    ttp_ref: int,
    centers_by_label: dict[str, list[tuple[int, int, int, float]]],
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
        ax = axes[idx // cols, idx % cols]
        ax.imshow(mean_img[:, :, z], cmap="gray", origin="lower")
        dt = (ttp_map[:, :, z].astype(np.float32) - float(ttp_ref))
        vmax = float(np.percentile(np.abs(dt[np.isfinite(dt)]), 99)) if np.any(np.isfinite(dt)) else 1.0
        vmax = max(1.0, vmax)
        ax.imshow(dt, cmap="coolwarm", origin="lower", alpha=0.55, vmin=-vmax, vmax=vmax)
        ax.scatter([y], [x], s=40, c="yellow")
        ax.set_title(f"{label} z={z+1} ΔTTP={int(dt[x,y])}")
        ax.axis("off")

    for idx in range(len(panels), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.join(image_dir, "AI"), exist_ok=True)
    plt.savefig(os.path.join(image_dir, "AI", "DCE_geometry_roi_ttp.png"), dpi=300)
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
    dce4d = ref_img.get_fdata().astype(np.float32)

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

    xs_mm, ys_mm, zs_mm = _axis_world_coords(ref_img.affine, (height, width, n_slices))
    x_mid = float(xs_mm[int(round((height - 1) / 2.0))])
    y_min, y_max = float(np.min(ys_mm)), float(np.max(ys_mm))

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
    upslope_map, late_map = _compute_upslope_and_late(dce4d, cfg.baseline_frames, late_frames=10)

    bat_map, auc_total, early_frac = _compute_bat_and_auc(
        dce4d,
        cfg.baseline_frames,
        bat_mad_k=float(getattr(settings, "ROI_GEOM_BAT_MAD_K", 3.0)),
        bat_persist=int(getattr(settings, "ROI_GEOM_BAT_PERSIST", 2)),
        early_window=int(getattr(settings, "ROI_GEOM_EARLY_WINDOW", 6)),
    )

    # Keep a debug scalar: approximate arterial bolus timing.
    ttp_target = _robust_ttp_target(peak_map, ttp_map, brain_mask)

    allowed_rica = brain_mask.copy()
    allowed_lica = brain_mask.copy()
    allowed_sss = brain_mask.copy()

    right_i = xs_mm > x_mid
    left_i = xs_mm < x_mid
    allowed_rica &= right_i[:, None, None]
    allowed_lica &= left_i[:, None, None]

    band = max(1, int(cfg.sss_midline_band))
    i_mid = int(round((height - 1) / 2.0))
    i0 = max(0, i_mid - band)
    i1 = min(height, i_mid + band + 1)
    allowed_sss[:i0, :, :] = False
    allowed_sss[i1:, :, :] = False

    # Exclude midline band from ICA.
    allowed_rica[:i1, :, :] = False
    allowed_lica[i0:, :, :] = False

    use_prob = bool(getattr(settings, "ROI_GEOM_USE_PROBABILITY", True))
    dyn_radius = bool(getattr(settings, "ROI_GEOM_DYNAMIC_RADIUS", True))
    r_min = int(getattr(settings, "ROI_GEOM_RADIUS_MIN", 3))
    mass_frac = float(getattr(settings, "ROI_GEOM_PROB_MASS_FRACTION", 0.20))
    gamma = float(getattr(settings, "ROI_GEOM_CENTRALITY_GAMMA", 1.0))
    min_centrality = float(getattr(settings, "ROI_GEOM_MIN_CENTRALITY", 0.12))
    coherence_px = float(getattr(settings, "ROI_GEOM_COHERENCE_PX", 35.0))
    comp_q = float(getattr(settings, "ROI_GEOM_COMPONENT_Q", 99.6))
    comp_min_area = int(getattr(settings, "ROI_GEOM_COMPONENT_MIN_AREA", 6))
    comp_max_area = int(getattr(settings, "ROI_GEOM_COMPONENT_MAX_AREA", 250))
    comp_ecc_max = float(getattr(settings, "ROI_GEOM_COMPONENT_ECC_RATIO_MAX", 3.0))
    use_component_roi = bool(getattr(settings, "ROI_GEOM_USE_COMPONENT_ROI", True))
    artery_perim_area_max = float(getattr(settings, "ROI_GEOM_ARTERY_PERIM_AREA_MAX", 1.2))

    # Precompute per-slice centrality maps (distance-to-edge within brain mask).
    centrality = np.ones((height, width, n_slices), dtype=np.float32)
    if gamma > 0 and distance_transform_edt is not None:
        for z in range(n_slices):
            centrality[:, :, z] = _centrality_map01(brain_mask[:, :, z])

    # Hard exclude the most peripheral rim to suppress transverse sinus / edge vessels.
    if distance_transform_edt is not None and min_centrality > 0:
        cmask = centrality >= float(min_centrality)
        allowed_rica &= cmask
        allowed_lica &= cmask
        allowed_sss &= cmask

    prob_rica = np.zeros((height, width, n_slices), dtype=np.float32)
    prob_lica = np.zeros((height, width, n_slices), dtype=np.float32)
    prob_sss = np.zeros((height, width, n_slices), dtype=np.float32)

    # Global (whole-volume) scoring (no per-slice normalization).
    auc01 = _robust_minmax01(auc_total, brain_mask)
    ef01 = _robust_minmax01(early_frac, brain_mask, p_low=5.0, p_high=99.0)
    bat01 = _robust_minmax01(bat_map.astype(np.float32), brain_mask, p_low=5.0, p_high=95.0)
    peak01 = _robust_minmax01(peak_map, brain_mask)
    upslope01 = _robust_minmax01(upslope_map, brain_mask)
    late01 = _robust_minmax01(late_map, brain_mask)

    blob3d = np.zeros((height, width, n_slices), dtype=np.float32)
    blob_sigma = float(getattr(settings, "ROI_GEOM_BLOB_SIGMA", 1.2))
    for z in range(n_slices):
        blob3d[:, :, z] = _blobness2d(auc01[:, :, z], sigma=blob_sigma)

    artery_score = (
        1.7 * auc01
        + 1.8 * ef01
        + 1.3 * (1.0 - bat01)
        + 0.9 * upslope01
        + 0.6 * peak01
        - 0.8 * late01
    ).astype(np.float32)
    vein_score = (
        1.5 * auc01
        + 1.2 * (1.0 - ef01)
        + 1.0 * bat01
        + 0.9 * late01
        + 0.3 * peak01
        - 0.4 * upslope01
    ).astype(np.float32)

    # Blobness suppresses in-plane tubular/arc structures (e.g. transverse sinus).
    artery_score *= (0.35 + 0.65 * blob3d)
    vein_score *= (0.35 + 0.65 * blob3d)

    # Centrality weighting.
    if gamma > 0:
        artery_score *= np.power(np.clip(centrality, 0.0, 1.0) + 1e-6, float(gamma))
        vein_score *= np.power(np.clip(centrality, 0.0, 1.0) + 1e-6, float(gamma))

    # Posterior-lateral transverse sinus suppression for arteries.
    if y_max > y_min:
        y_frac = (ys_mm[None, :, None] - y_min) / max(1e-6, (y_max - y_min))
        artery_score *= (0.65 + 0.35 * np.clip(y_frac, 0.0, 1.0))

    score_rica = np.where(allowed_rica, artery_score, 0.0).astype(np.float32)
    score_lica = np.where(allowed_lica, artery_score, 0.0).astype(np.float32)
    score_sss = np.where(allowed_sss, vein_score, 0.0).astype(np.float32)

    # Restrict scoring to configured z windows.
    zmask_rica = np.zeros(n_slices, dtype=bool)
    zmask_rica[rica_z[0] : rica_z[1] + 1] = True
    zmask_lica = np.zeros(n_slices, dtype=bool)
    zmask_lica[lica_z[0] : lica_z[1] + 1] = True
    zmask_sss = np.zeros(n_slices, dtype=bool)
    zmask_sss[sss_z[0] : sss_z[1] + 1] = True
    score_rica *= zmask_rica[None, None, :]
    score_lica *= zmask_lica[None, None, :]
    score_sss *= zmask_sss[None, None, :]

    comp_q3d = float(getattr(settings, "ROI_GEOM_GLOBAL_Q", 99.7))
    rica_comps = _cluster_top_components(score_rica, allowed_rica, ref_img.affine, bat_map, early_frac, blob3d, q=comp_q3d)
    lica_comps = _cluster_top_components(score_lica, allowed_lica, ref_img.affine, bat_map, early_frac, blob3d, q=comp_q3d)
    sss_comps = _cluster_top_components(score_sss, allowed_sss, ref_img.affine, bat_map, early_frac, blob3d, q=comp_q3d)

    def _obj(rc: _Component3D, lc: _Component3D, sc: _Component3D) -> float:
        rx, ry, rz = rc.centroid_ras_mm
        lx, ly, lz = lc.centroid_ras_mm

        dr = abs(rx - x_mid)
        dl = abs(lx - x_mid)
        sym = abs(dr - dl) + 0.25 * (abs(ry - ly) + abs(rz - lz))

        art_bat = 0.5 * (rc.mean_bat + lc.mean_bat)
        art_ef = 0.5 * (rc.mean_early_frac + lc.mean_early_frac)
        ven_pen = 0.0
        if sc.mean_bat < art_bat + 2.0:
            ven_pen += (art_bat + 2.0 - sc.mean_bat) * 2.0
        if sc.mean_early_frac > art_ef - 0.05:
            ven_pen += (sc.mean_early_frac - (art_ef - 0.05)) * 30.0

        shape_pen = max(0.0, rc.elongation - 3.0) * 2.0 + max(0.0, lc.elongation - 3.0) * 2.0

        mass = np.log(rc.mass + 1.0) + np.log(lc.mass + 1.0) + 0.8 * np.log(sc.mass + 1.0)
        return float(mass - 0.25 * sym - ven_pen - shape_pen)

    best = None
    best_val = -1e18
    rica_list = rica_comps[:10] if rica_comps else []
    lica_list = lica_comps[:10] if lica_comps else []
    sss_list = sss_comps[:10] if sss_comps else []
    for rc in rica_list:
        for lc in lica_list:
            for sc in sss_list:
                v = _obj(rc, lc, sc)
                if v > best_val:
                    best_val = v
                    best = (rc, lc, sc)

    rica_vox_by_z: dict[int, np.ndarray] = {}
    lica_vox_by_z: dict[int, np.ndarray] = {}
    sss_vox_by_z: dict[int, np.ndarray] = {}

    if best is None:
        rica_centers = _best_centers_by_slice(score_rica, allowed_rica, rica_z, rica_slices_eff)
        lica_centers = _best_centers_by_slice(score_lica, allowed_lica, lica_z, lica_slices_eff)
        sss_centers = _best_centers_by_slice(score_sss, allowed_sss, sss_z, sss_slices_eff)
    else:
        rc, lc, sc = best
        rica_centers = _centers_from_component(rc, score_rica, k_slices=rica_slices_eff)
        lica_centers = _centers_from_component(lc, score_lica, k_slices=lica_slices_eff)
        sss_centers = _centers_from_component(sc, score_sss, k_slices=sss_slices_eff)

        rica_keep = {z for _x, _y, z, _s in rica_centers}
        lica_keep = {z for _x, _y, z, _s in lica_centers}
        sss_keep = {z for _x, _y, z, _s in sss_centers}
        rica_vox_by_z = _voxels_by_z_from_component(rc, z_keep=rica_keep)
        lica_vox_by_z = _voxels_by_z_from_component(lc, z_keep=lica_keep)
        sss_vox_by_z = _voxels_by_z_from_component(sc, z_keep=sss_keep)

    # For overlays and dynamic radii, normalize scores to 0..1-ish maps.
    def _norm01(s: np.ndarray) -> np.ndarray:
        m = np.isfinite(s) & (s > 0)
        if not np.any(m):
            return np.zeros_like(s, dtype=np.float32)
        hi = float(np.percentile(s[m], 99.5))
        if hi <= 0:
            return np.zeros_like(s, dtype=np.float32)
        return np.clip((s / hi).astype(np.float32), 0.0, 1.0)

    prob_rica = _norm01(score_rica)
    prob_lica = _norm01(score_lica)
    prob_sss = _norm01(score_sss)

    rica_radius_by_z: dict[int, int] = {}
    lica_radius_by_z: dict[int, int] = {}
    sss_radius_by_z: dict[int, int] = {}
    if dyn_radius:
        for x, y, z, _p in rica_centers:
            rica_radius_by_z[int(z)] = _radius_from_probability_mass(
                prob_rica[:, :, z], x, y, r_min=r_min, r_max=int(cfg.rica_radius), target_mass=mass_frac
            )
        for x, y, z, _p in lica_centers:
            lica_radius_by_z[int(z)] = _radius_from_probability_mass(
                prob_lica[:, :, z], x, y, r_min=r_min, r_max=int(cfg.lica_radius), target_mass=mass_frac
            )
        for x, y, z, _p in sss_centers:
            sss_radius_by_z[int(z)] = _radius_from_probability_mass(
                prob_sss[:, :, z], x, y, r_min=r_min, r_max=int(cfg.sss_radius), target_mass=mass_frac
            )

    # Persist outputs in the same on-disk format as the existing AI pipeline.
    _save_roi_outputs(
        roi_type="Artery",
        roi_subtype="Right Interior Carotid",
        centers=rica_centers,
        radius=cfg.rica_radius,
        radius_by_z=rica_radius_by_z if rica_radius_by_z else None,
        voxels_by_z=rica_vox_by_z if rica_vox_by_z else None,
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
        radius_by_z=lica_radius_by_z if lica_radius_by_z else None,
        voxels_by_z=lica_vox_by_z if lica_vox_by_z else None,
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
        radius_by_z=sss_radius_by_z if sss_radius_by_z else None,
        voxels_by_z=sss_vox_by_z if sss_vox_by_z else None,
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

    _save_probability_overlay(
        dce4d=dce4d,
        centers_by_label={
            "rICA": rica_centers,
            "lICA": lica_centers,
            "SSS": sss_centers,
        },
        radii_by_z_by_label={
            "rICA": rica_radius_by_z,
            "lICA": lica_radius_by_z,
            "SSS": sss_radius_by_z,
        },
        prob3d_by_label={
            "rICA": prob_rica,
            "lICA": prob_lica,
            "SSS": prob_sss,
        },
        image_dir=image_directory,
    )

    _save_ttp_overlay(
        dce4d=dce4d,
        ttp_map=ttp_map,
        ttp_ref=int(ttp_target),
        centers_by_label={
            "rICA": rica_centers,
            "lICA": lica_centers,
            "SSS": sss_centers,
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
        "dynamic_radii_by_z": {
            "rICA": rica_radius_by_z,
            "lICA": lica_radius_by_z,
            "SSS": sss_radius_by_z,
        },
        "z_ranges": {
            "rICA": rica_z,
            "lICA": lica_z,
            "SSS": sss_z,
        },
        "ttp_target": int(ttp_target),
    }
