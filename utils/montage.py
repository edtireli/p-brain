"""Utilities for rendering parametric map montages in native DCE space."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROWS = 2
COLS = 5
NUM_TILES = ROWS * COLS
ROT90 = 1
BBOX_PADDING = 3
DPI = 300
EPS = 1e-8


@dataclass
class MapJob:
    """Configuration for rendering a single parametric map montage."""

    base: str
    output_base: str
    vmin: float
    vmax: float
    cmap_name: str = "specthl"
    mask_zero: bool = False
    output_ext: str = ".png"
    patterns: Sequence[str] = field(default_factory=tuple)

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
    MapJob("CBF_per_voxel_tikhonov", "cbf_montage", 0.0, 30.0),
    MapJob("CBF_tikhonov_map_atlas", "cbf_parcel_montage", 0.0, 30.0),
    MapJob("mtt_map", "mtt_montage", 0.0, 0.02),
    MapJob("MTT_tikhonov_map_atlas", "mtt_parcel_montage", 0.0, 0.02),
    MapJob("cth_map", "cth_montage", 0.0, 3.0),
    MapJob("CTH_tikhonov_map_atlas", "cth_parcel_montage", 0.0, 3.0),
    MapJob("Ki_per_voxel", "ki_voxel_montage", -0.1, 0.15),
    MapJob("Ki_map_atlas", "ki_atlas_montage", -0.1, 0.15),
    MapJob("vp_map_atlas", "vp_atlas_montage", 0.0, 3.0),
    MapJob("vp_per_voxel", "vp_per_voxel", 0.0, 3.0, mask_zero=True, output_ext=".png"),
)


def generate_parametric_montages(
    analysis_directory: str,
    image_directory: str,
    dce_path: str,
    *,
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

    ref_info = _build_reference(reference, rows * cols)
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
                    rows=rows,
                    cols=cols,
                    dpi=dpi,
                )
                generated_any = True
                print(f"[montage] Saved {os.path.relpath(out_path, start=image_directory)}")
            except Exception as exc:
                print(f"[montage] Failed to render {map_path}: {exc}")

    if not generated_any:
        print("[montage] No parametric maps found for montage rendering.")


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


def _get_cmap(name: str) -> mpl.colors.Colormap:
    if name.lower() == "specthl":
        return _mk_specthl()
    return mpl.colormaps[name].copy()


def _load_reference_volume(dce_path: str) -> np.ndarray | None:
    img = nib.load(dce_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim == 4:
        data = np.nanmean(data, axis=-1)
    if data.ndim != 3:
        return None
    return data


def _build_reference(volume: np.ndarray, tiles: int) -> Dict[str, np.ndarray] | None:
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
    if nz <= 1:
        return np.zeros_like(z_fracs, dtype=int)
    z = np.rint(z_fracs * (nz - 1)).astype(int)
    z = np.clip(z, 0, nz - 1)
    for i in range(1, len(z)):
        if z[i] <= z[i - 1] and nz > 1:
            z[i] = min(nz - 1, z[i - 1] + 1)
    return z


def _find_available_maps(job: MapJob, analysis_directory: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for pattern in job.candidate_patterns():
        for path in sorted(glob.glob(os.path.join(analysis_directory, pattern))):
            suffix = _extract_suffix(path, job.base)
            if suffix not in found or path.endswith(".nii.gz"):
                found[suffix] = path
    return found


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
    rows: int,
    cols: int,
    dpi: int,
) -> None:
    img = nib.load(map_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

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
    norm, tick_values = _build_normalizer(data, job)
    cmap = cmap.with_extremes(bad=(0, 0, 0, 0), under=(0, 0, 0, 0))

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.2, rows * 2.2), facecolor=(0.0, 0.0, 0.0, 0.0)
    )
    fig.patch.set_alpha(0.0)
    axes = axes.ravel()

    for ax, z in zip(axes, z_indices):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_facecolor("#e0e0e0")
        for spine in ax.spines.values():
            spine.set_visible(False)

        sl = data[:, :, int(z)]
        slr = np.rot90(sl, ref_info["rotate"])
        slc = slr[r0:r1, c0:c1]

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

    # Hide any unused axes when there are fewer slices than tiles
    for ax in axes[len(z_indices) :]:
        ax.axis("off")

    cax = fig.add_axes([0.93, 0.12, 0.015, 0.3])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    if tick_values:
        cb.set_ticks(tick_values)
        cb.set_ticklabels([f"{val:g}" for val in tick_values])
    cb.ax.tick_params(labelsize=8, colors="black")
    for spine in cb.ax.spines.values():
        spine.set_edgecolor("black")

    plt.subplots_adjust(left=0.02, right=0.9, top=0.96, bottom=0.02, wspace=0.02, hspace=0.02)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _build_normalizer(
    data: np.ndarray, job: MapJob
) -> tuple[mpl.colors.Normalize, list[float]]:
    vmin = float(job.vmin)
    vmax = float(job.vmax)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        mask = np.isfinite(data)
        if job.mask_zero:
            mask &= data > EPS
        finite_vals = data[mask]
        if finite_vals.size:
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))
        else:
            vmin, vmax = 0.0, 1.0

    if vmax <= vmin:
        vmax = vmin + (abs(vmin) if vmin != 0 else 1.0)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    return norm, _default_ticks(vmin, vmax)


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
