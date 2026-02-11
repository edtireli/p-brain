"""Utilities for rendering parametric map montages in native DCE space.

Supports external head/ICV masks (e.g., FSL/FreeSurfer). If present, these are
preferentially used to remove air outside the head while retaining skull/scalp,
and to constrain colour overlays to the intracranial volume."""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Sequence, Tuple

# Ensure a non-interactive backend for reproducible, headless rendering.
# This also makes PNG transparency behave consistently on macOS.
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib as mpl

try:  # pragma: no cover - backend selection is environment-dependent
    mpl.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
except Exception:
    pass

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
    remove_small_holes,
    remove_small_objects,
)

try:  # optional; used for non-matplotlib PNG rendering
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


def _display_orient2d(a2: np.ndarray) -> np.ndarray:
    """Orient a 2D slice consistently for PNG rendering.

    By default, this matches the validator/MATLAB viewing convention for the
    Hemisure datasets: rotate in-plane by +90° (np.rot90(k=1)).

    Override via env vars:
    - PBRAIN_DISPLAY_ROT90_K: integer k for np.rot90 (default: 1)
    - PBRAIN_DISPLAY_FLIP_LR: 0/1 (default: 0)
    - PBRAIN_DISPLAY_FLIP_UD: 0/1 (default: 0)
    """

    out = np.asarray(a2)
    try:
        k = int((os.environ.get("PBRAIN_DISPLAY_ROT90_K") or "1").strip()) % 4
    except Exception:
        k = 1
    if k:
        out = np.rot90(out, k=k)

    flr = (os.environ.get("PBRAIN_DISPLAY_FLIP_LR") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    fud = (os.environ.get("PBRAIN_DISPLAY_FLIP_UD") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if flr:
        out = np.fliplr(out)
    if fud:
        out = np.flipud(out)
    return out


def _load_pillow_font(size: int) -> Any:
    """Best-effort font loader for Pillow rendering."""
    if ImageFont is None:
        return None
    try:
        # Common on many Python installs.
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        pass
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ):
        try:
            if os.path.isfile(path):
                return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _units_for_job(job: MapJob) -> str | None:
    base = (job.base or "").lower()
    metric = (getattr(job, "metric", "") or "").lower()
    # Match manuscript captions (human-readable equivalents of siunitx strings).
    if metric == "ki" or base.startswith("ki_"):
        return "mL/100g/min"
    if "cbf" in base:
        return "mL/100g/min"
    if base.startswith("vp_") or metric == "vp":
        return "mL/100g"
    if base.startswith("mtt") or "mtt" in base:
        return "seconds"
    if base.startswith("cth") or "cth" in base:
        return "seconds"
    # Diffusion captions in article.tex don't specify units, so omit by default.
    if metric == "fa" or base.startswith("fa_"):
        return None
    if base.startswith(("md_", "ad_", "rd_", "mo_")):
        return None
    if base.startswith("tensor_residual"):
        return None
    if base.startswith("t1_") or base == "t1_map" or "t1_map" in base:
        return "ms"
    # M0 is typically in arbitrary units; omit by default.
    if base.startswith("m0_") or base == "m0_map" or "m0_map" in base:
        return None
    return None


# Pillow colorbar defaults (can be overridden via env vars)
PILLOW_COLORBAR_TICK_FONT_SIZE = int(os.environ.get("PBRAIN_COLORBAR_TICK_FONT_SIZE", "24"))
PILLOW_COLORBAR_UNITS_FONT_SIZE = int(os.environ.get("PBRAIN_COLORBAR_UNITS_FONT_SIZE", "28"))
PILLOW_COLORBAR_UNITS_POSITION = os.environ.get("PBRAIN_COLORBAR_UNITS_POSITION", "right_rot").lower()
PILLOW_COLORBAR_UNITS_GAP_PX = int(os.environ.get("PBRAIN_COLORBAR_UNITS_GAP_PX", "10"))
PILLOW_OUTPUT_DPI = int(os.environ.get("PBRAIN_PNG_DPI", "300"))
PILLOW_TILE_INNER_MARGIN_PX = int(os.environ.get("PBRAIN_TILE_INNER_MARGIN_PX", "10"))
PILLOW_TILE_GAP_PX = int(os.environ.get("PBRAIN_TILE_GAP_PX", "18"))
PILLOW_OUTER_MARGIN_PX = int(os.environ.get("PBRAIN_OUTER_MARGIN_PX", "0"))
PILLOW_TRANSPARENT_GUTTERS = os.environ.get("PBRAIN_TRANSPARENT_GUTTERS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)
PILLOW_TRANSPARENT_COLORBAR_BG = os.environ.get("PBRAIN_TRANSPARENT_COLORBAR_BG", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)


def _pillow_paint_grid_background(
    canvas: Any,
    *,
    x0: int,
    y0: int,
    rows: int,
    cols: int,
    tile_w: int,
    tile_h: int,
    gap: int,
    color: tuple[int, int, int, int],
) -> None:
    if ImageDraw is None:
        return
    w = cols * tile_w + (cols - 1) * gap
    h = rows * tile_h + (rows - 1) * gap
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=color)


def _pillow_punch_transparent_gutters(
    canvas: Any,
    *,
    grid_x0: int,
    grid_y0: int,
    rows: int,
    cols: int,
    tile_w: int,
    tile_h: int,
    gap: int,
    include_grid_to_colorbar_gap: bool,
) -> None:
    """Make the inter-tile gaps transparent while keeping tiles opaque."""
    if ImageDraw is None or gap <= 0:
        return

    grid_w = cols * tile_w + (cols - 1) * gap
    grid_h = rows * tile_h + (rows - 1) * gap
    alpha = canvas.getchannel("A")
    draw = ImageDraw.Draw(alpha)

    # Vertical gaps between tile columns.
    for c in range(cols - 1):
        x = grid_x0 + (c + 1) * tile_w + c * gap
        draw.rectangle([x, grid_y0, x + gap - 1, grid_y0 + grid_h - 1], fill=0)

    # Horizontal gaps between tile rows.
    for r in range(rows - 1):
        y = grid_y0 + (r + 1) * tile_h + r * gap
        draw.rectangle([grid_x0, y, grid_x0 + grid_w - 1, y + gap - 1], fill=0)

    # Gap separating the tile grid from the colorbar.
    if include_grid_to_colorbar_gap:
        x = grid_x0 + grid_w
        draw.rectangle([x, grid_y0, x + gap - 1, grid_y0 + grid_h - 1], fill=0)

    canvas.putalpha(alpha)


def _pillow_colorbar_required_width(
    *,
    norm: mcolors.Normalize,
    tick_values: list[float] | None,
    units: str | None,
    units_position: str,
    units_gap_px: int = 12,
    tick_font_size: int,
    units_font_size: int,
) -> int:
    """Compute a safe colorbar slot width so tick labels/units never clip."""
    if Image is None or ImageDraw is None:
        return 150

    # Layout constants (must match _draw_colorbar_pillow).
    left_pad = 10
    right_pad = 12
    tick_line = 10
    tick_pad = 14
    bar_w = 32
    units_gap = int(max(0, units_gap_px))

    font = _load_pillow_font(int(tick_font_size))
    units_font = _load_pillow_font(int(units_font_size))

    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    ticks = tick_values or _default_ticks(float(norm.vmin), float(norm.vmax))
    finite_ticks = [float(tv) for tv in ticks if np.isfinite(tv)]
    tick_labels = _format_tick_labels(finite_ticks)
    max_tick_w = 0
    for label in tick_labels:
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            w = int(bbox[2] - bbox[0])
        except Exception:
            w = len(label) * max(6, int(tick_font_size * 0.6))
        max_tick_w = max(max_tick_w, w)

    label_w = max_tick_w
    pos = (units_position or "above").lower().strip()
    if units and pos in {"beside", "right", "right_rot", "right-rot", "beside_rot", "beside-rot"}:
        try:
            bbox = draw.textbbox((0, 0), units, font=units_font)
            tw = int(bbox[2] - bbox[0])
            th = int(bbox[3] - bbox[1])
        except Exception:
            tw = len(units) * max(6, int(units_font_size * 0.6))
            th = max(10, int(units_font_size))

        # Units are placed to the right of tick labels. For rotated variants, measure the
        # actual rotated image width so we don't underestimate and cause overlap/clipping.
        if pos in {"right_rot", "right-rot", "beside_rot", "beside-rot"}:
            try:
                bbox = draw.textbbox((0, 0), units, font=units_font)
                x0b, y0b, x1b, y1b = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                pad = 2
                tw = max(1, x1b - x0b)
                th = max(1, y1b - y0b)
                txt_img = Image.new(
                    "RGBA",
                    (max(1, tw + pad * 2), max(1, th + pad * 2)),
                    (0, 0, 0, 0),
                )
                txt_draw = ImageDraw.Draw(txt_img)
                txt_draw.text((pad - x0b, pad - y0b), units, fill=(0, 0, 0, 255), font=units_font)
                rot = txt_img.rotate(-90, expand=True)
                units_w = int(rot.size[0])
            except Exception:
                units_w = int(max(th, 10))
        else:
            units_w = int(tw)
        label_w = max_tick_w + units_gap + units_w

    needed = left_pad + bar_w + tick_line + tick_pad + label_w + right_pad
    return int(max(150, needed))


def _pillow_required_colorbar_width_for_projection(data: np.ndarray, job: MapJob) -> int:
    """Compute required Pillow colorbar width for a projection montage."""
    cmap = _get_cmap(job.cmap_name)
    norm, tick_values = _build_projection_normalizer(data, job)
    if (getattr(job, "metric", None) or "").lower() == "fa":
        vmax = float(getattr(norm, "vmax", 1.0))
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=False)
    units = _units_for_job(job)
    return _pillow_colorbar_required_width(
        norm=norm,
        tick_values=tick_values,
        units=units,
        units_position=PILLOW_COLORBAR_UNITS_POSITION,
        units_gap_px=PILLOW_COLORBAR_UNITS_GAP_PX,
        tick_font_size=PILLOW_COLORBAR_TICK_FONT_SIZE,
        units_font_size=PILLOW_COLORBAR_UNITS_FONT_SIZE,
    )


def _pillow_required_colorbar_width_for_montage(
    map_path: str,
    job: MapJob,
    *,
    reference_img: nib.Nifti1Image,
    overlay: Dict[str, Any] | None,
    brain_mask: np.ndarray | None,
    segmentation_img: nib.Nifti1Image | None,
) -> int:
    """Compute required Pillow colorbar width for a parametric montage."""

    is_atlas = job.base.endswith("_map_atlas")
    is_diffusion = _is_diffusion_job(job)

    img = nib.load(map_path)
    target = (reference_img.shape, reference_img.affine)
    if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine):
        img = resample_from_to(img, target, order=0 if is_atlas else 1)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    if not is_atlas:
        try:
            zoom_src = np.array(img.header.get_zooms()[:3], dtype=float)
            zoom_ref = np.array(reference_img.header.get_zooms()[:3], dtype=float)
            ratio = max(zoom_src[0] / zoom_ref[0], zoom_src[1] / zoom_ref[1])
            if ratio > 1.4:
                data = gaussian_filter(data, sigma=(0.6, 0.6, 0.0), mode="nearest")
        except Exception:
            pass
    else:
        data = _fill_empty_slices_nearest(data)

    if is_diffusion and not is_atlas:
        inpaint_domain = None
        if brain_mask is not None and brain_mask.shape == data.shape:
            inpaint_domain = np.asarray(brain_mask, dtype=bool)
        else:
            finite = np.isfinite(data)
            try:
                from scipy.ndimage import binary_closing

                inpaint_domain = binary_closing(finite, structure=np.ones((3, 3, 3), bool))
            except Exception:
                inpaint_domain = finite
        data = _inpaint_nans_nearest(data, inside_mask=inpaint_domain)

    # Display-domain mask (matches _render_montage_pillow)
    if brain_mask is not None and not is_atlas:
        mask_data = np.asarray(brain_mask, dtype=bool)
        if mask_data.shape != data.shape:
            if segmentation_img is None:
                raise ValueError("Brain mask shape does not match parametric map")
            target2 = (data.shape, img.affine)
            seg_img = segmentation_img
            if (
                segmentation_img.shape != data.shape
                or not np.allclose(segmentation_img.affine, img.affine)
            ):
                seg_img = resample_from_to(segmentation_img, target2, order=0)
            seg_data = np.asarray(seg_img.get_fdata(), dtype=np.float32)
            mask_data = np.isfinite(seg_data) & (seg_data > 0.5)
        brain_mask = mask_data

    head_support_3d = None
    if overlay is not None and isinstance(overlay.get("mask_head"), np.ndarray):
        head_support_3d = overlay["mask_head"].astype(bool)
        if head_support_3d.shape != data.shape:
            head_img = nib.Nifti1Image(head_support_3d.astype(np.float32), reference_img.affine)
            head_img = resample_from_to(head_img, (data.shape, img.affine), order=0)
            head_support_3d = np.asarray(head_img.get_fdata(), dtype=np.float32) > 0.5

    norm_data = data if brain_mask is None else np.where(brain_mask, data, np.nan)
    if is_atlas and head_support_3d is not None:
        norm_data = np.where(head_support_3d, norm_data, np.nan)

    focus_data = None
    if (getattr(job, "metric", "") or "").lower() == "ki":
        core_mask = None
        if brain_mask is not None and brain_mask.shape == data.shape:
            core_mask = _mask_erode_mm(brain_mask, img, mm=2.0)
        elif head_support_3d is not None and head_support_3d.shape == data.shape:
            core_mask = _mask_erode_mm(head_support_3d, img, mm=2.0)
        if core_mask is not None and core_mask.any():
            focus_data = np.where(core_mask, data, np.nan)
        else:
            focus_data = norm_data

    norm, tick_values = _build_normalizer(
        norm_data,
        job,
        mask_zero_override=False if brain_mask is not None else None,
        focus_data=focus_data,
    )
    if (getattr(job, "metric", None) or "").lower() == "fa":
        vmax = float(getattr(norm, "vmax", 1.0))
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=False)

    units = _units_for_job(job)
    return _pillow_colorbar_required_width(
        norm=norm,
        tick_values=tick_values,
        units=units,
        units_position=PILLOW_COLORBAR_UNITS_POSITION,
        units_gap_px=PILLOW_COLORBAR_UNITS_GAP_PX,
        tick_font_size=PILLOW_COLORBAR_TICK_FONT_SIZE,
        units_font_size=PILLOW_COLORBAR_UNITS_FONT_SIZE,
    )


def _draw_colorbar_pillow(
    canvas: Any,
    *,
    norm: mcolors.Normalize,
    cmap: mpl.colors.Colormap,
    tick_values: list[float] | None,
    transparent_background: bool,
    units: str | None,
    units_position: str = "above",
    units_gap_px: int = 10,
    x0: int,
    y0: int,
    w: int,
    h: int,
    tick_font_size: int = 18,
    units_font_size: int = 18,
) -> None:
    if Image is None or ImageDraw is None:
        return

    lut = cmap(np.linspace(0, 1, 256)).astype(np.float32)
    lut[:, 3] = 1.0

    # Deterministic layout so required width can be computed reliably.
    left_pad = 10
    right_pad = 12
    bar_w = 32
    bar_x0 = x0 + left_pad
    cb_y0 = y0
    cb_h = h

    grad = np.linspace(1.0, 0.0, cb_h, dtype=np.float32)
    idx = (grad * 255.0 + 0.5).astype(np.int32)
    bar_rgb = (lut[idx, :3] * 255.0 + 0.5).astype(np.uint8)
    bar_img = np.zeros((cb_h, bar_w, 4), dtype=np.uint8)
    bar_img[..., :3] = bar_rgb[:, None, :]
    bar_img[..., 3] = 255
    bar = Image.fromarray(bar_img, mode="RGBA")
    canvas.alpha_composite(bar, (bar_x0, cb_y0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [bar_x0, cb_y0, bar_x0 + bar_w - 1, cb_y0 + cb_h - 1],
        outline=(0, 0, 0, 255),
        width=2,
    )

    ticks = tick_values or _default_ticks(float(norm.vmin), float(norm.vmax))
    font = _load_pillow_font(int(tick_font_size))
    units_font = _load_pillow_font(int(units_font_size))

    # Measure max tick label width for unit placement to the right.
    finite_ticks = [float(tv) for tv in ticks if np.isfinite(tv)]
    tick_labels = _format_tick_labels(finite_ticks)
    max_tick_w = 0
    for label in tick_labels:
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            w_lbl = int(bbox[2] - bbox[0])
        except Exception:
            w_lbl = len(label) * max(6, int(tick_font_size * 0.6))
        max_tick_w = max(max_tick_w, w_lbl)

    # Tick labels normally sit to the right of the bar, but clamp them so they
    # can never be pushed off-canvas if width estimation differs across fonts.
    tick_text_x = bar_x0 + bar_w + 14
    max_tick_text_x = x0 + w - right_pad - max_tick_w
    if max_tick_text_x < tick_text_x:
        tick_text_x = max(x0, int(max_tick_text_x))

    units_position = (units_position or "above").lower().strip()
    if units:
        label = units
        try:
            bbox = draw.textbbox((0, 0), label, font=units_font)
            tw = int(bbox[2] - bbox[0])
            th = int(bbox[3] - bbox[1])
        except Exception:
            tw = len(label) * max(6, int(units_font_size * 0.6))
            th = max(10, int(units_font_size))

        if units_position in {"right_rot", "right-rot", "beside_rot", "beside-rot"}:
            # Rotated 90° clockwise, centered on bar height, placed to the right of tick labels.
            if Image is not None:
                try:
                    bbox = draw.textbbox((0, 0), label, font=units_font)
                    x0b, y0b, x1b, y1b = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                except Exception:
                    x0b, y0b, x1b, y1b = 0, 0, tw, th
                pad = 2
                tw_img = max(1, int(x1b - x0b))
                th_img = max(1, int(y1b - y0b))
                txt_img = Image.new(
                    "RGBA",
                    (max(1, tw_img + pad * 2), max(1, th_img + pad * 2)),
                    (0, 0, 0, 0),
                )
                txt_draw = ImageDraw.Draw(txt_img)
                # Use bbox offsets so descenders/negative font bearings never clip.
                txt_draw.text((pad - x0b, pad - y0b), label, fill=(0, 0, 0, 255), font=units_font)
                rot = txt_img.rotate(-90, expand=True)
                rw, rh = rot.size
                min_x_units = tick_text_x + max_tick_w + int(max(0, units_gap_px))
                max_x_units = x0 + w - right_pad - rw
                if max_x_units < x0:
                    x_units = x0
                elif max_x_units < min_x_units:
                    # Not enough room to keep a gap; keep it visible even if it overlaps.
                    x_units = max_x_units
                else:
                    # Right-align within the allocated slot.
                    x_units = max_x_units
                y_units = cb_y0 + (cb_h - rh) // 2
                canvas.alpha_composite(rot, (x_units, int(np.clip(y_units, cb_y0, cb_y0 + cb_h - rh))))
        elif units_position in {"right"}:
            # Unrotated, centered vertically, to the right of tick labels.
            min_x_units = tick_text_x + max_tick_w + int(max(0, units_gap_px))
            max_x_units = x0 + w - right_pad - tw
            if max_x_units < x0:
                x_units = x0
            elif max_x_units < min_x_units:
                x_units = max_x_units
            else:
                x_units = max_x_units
            y_units = cb_y0 + (cb_h - th) // 2
            draw.text((x_units, int(np.clip(y_units, cb_y0, cb_y0 + cb_h - th))), label, fill=(0, 0, 0, 255), font=units_font)
        elif units_position == "beside":
            # Legacy: to the right of the bar, near the top.
            draw.text(
                (bar_x0 + bar_w + 12, max(0, cb_y0 - th // 2)),
                label,
                fill=(0, 0, 0, 255),
                font=units_font,
            )
        elif units_position == "on":
            # Put on top of the bar itself; add a tiny outline for readability.
            tx = bar_x0 + (bar_w - tw) // 2
            ty = cb_y0 + 6
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((tx + dx, ty + dy), label, fill=(255, 255, 255, 255), font=units_font)
            draw.text((tx, ty), label, fill=(0, 0, 0, 255), font=units_font)
        else:
            # Default: above the bar.
            draw.text(
                (bar_x0 + (bar_w - tw) // 2, max(0, cb_y0 - th - int(units_gap_px))),
                label,
                fill=(0, 0, 0, 255),
                font=units_font,
            )

    last_y: int | None = None
    min_sep = max(8, int(tick_font_size * 1.05))
    for tv, label in zip(finite_ticks, tick_labels):
        t = (float(tv) - float(norm.vmin)) / max(1e-12, float(norm.vmax) - float(norm.vmin))
        t = float(np.clip(t, 0.0, 1.0))
        y = cb_y0 + int(round((1.0 - t) * (cb_h - 1)))
        if last_y is not None and abs(y - last_y) < min_sep:
            continue
        draw.line(
            [bar_x0 + bar_w, y, bar_x0 + bar_w + 10, y],
            fill=(0, 0, 0, 255),
            width=2,
        )
        # Use a compact significant-figure formatter (avoids all-zero labels for small ranges).
        y_text = y - int(tick_font_size * 0.45)
        y_text = int(np.clip(y_text, cb_y0, cb_y0 + cb_h - tick_font_size))
        # Clamp X per label to prevent complete off-canvas rendering when font
        # measurements differ across environments.
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            w_lbl = int(bbox[2] - bbox[0])
        except Exception:
            w_lbl = len(label) * max(6, int(tick_font_size * 0.6))
        max_x_lbl = x0 + w - right_pad - w_lbl
        x_lbl = int(np.clip(tick_text_x, x0, max_x_lbl if max_x_lbl >= x0 else x0))
        draw.text((x_lbl, y_text), label, fill=(0, 0, 0, 255), font=font)
        last_y = y

ROWS = 2
COLS = 5
NUM_TILES = ROWS * COLS
ROT90 = 1
BBOX_PADDING = 3
DPI = 300
EPS = 1e-8
FEATHER_MM = 1.0           # soft rim for T1 only
MAP_EDGE_FEATHER_MM = 0.0  # hard edge for parametric overlays
HEAD_DILATE_MM = 8.0      # grow brain mask to include skull+scalp
HEAD_EXTRA_MM = 2.0       # tiny extra cushion
HEAD_MIN_VOXELS = 10_000  # drop tiny islands from T1 envelope
ICV_ERODE_MM = 3.0        # approximate skull thickness to peel head -> ICV
MASK_SMOOTH_MM = 1.5      # closing radius to soften ragged mask edges
MASK_HOLE_VOXELS = 4_000  # fill interior voids below this volume
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
    MapJob("mtt_map", "mtt_montage", vmin=0.0),
    MapJob("MTT_tikhonov_map_atlas", "mtt_parcel_montage", vmin=0.0),
    MapJob("cth_map", "cth_montage", vmin=0.0),
    MapJob("CTH_tikhonov_map_atlas", "cth_parcel_montage", vmin=0.0),
    MapJob("Ki_per_voxel", "ki_voxel_montage", metric="ki"),
    MapJob("Ki_map_atlas", "ki_atlas_montage", metric="ki"),
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
    MapJob(
        "t1_map",
        "t1_montage",
        vmin=0.0,
        search_directories=("", "Fitting"),
        cmap_name="specthl",
    ),
    MapJob(
        "m0_map",
        "m0_montage",
        search_directories=("", "Fitting"),
        cmap_name="specthl",
    ),
)

MAP_JOB_LOOKUP: Dict[str, MapJob] = {job.base: job for job in MAP_JOBS}

# Diffusion detection helpers
_DIFFUSION_BASE_PREFIXES = ("fa_", "md_", "ad_", "rd_", "mo_", "tensor_residual_")


def _is_diffusion_job(job: MapJob) -> bool:
    name = (job.base or "").lower()
    if any(name.startswith(p) for p in _DIFFUSION_BASE_PREFIXES):
        return True
    return "diffusion" in tuple((job.search_directories or ()))


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


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def _is_appledouble(name: str) -> bool:
    # macOS resource fork files (AppleDouble) start with '._' and often look like
    # real files (e.g., '._Ki_map_atlas_patlak.nii.gz') but are not NIfTI.
    return name.startswith("._")


def _best_map_job_base(stem: str) -> tuple[str, str]:
    """Return (base, suffix) by matching the longest known MapJob base prefix."""
    best_base = ""
    for base in MAP_JOB_LOOKUP:
        if stem == base or stem.startswith(base):
            if len(base) > len(best_base):
                best_base = base
    if best_base:
        return best_base, stem[len(best_base) :]
    return stem, ""


def _discover_atlas_maps(analysis_directory: str) -> Dict[str, Dict[str, str]]:
    """Discover all atlas-based NIfTI maps under an Analysis directory."""
    discovered: Dict[str, Dict[str, str]] = defaultdict(dict)
    if not os.path.isdir(analysis_directory):
        return {}

    for root, _, files in os.walk(analysis_directory):
        for fname in files:
            if _is_appledouble(fname) or fname.startswith("."):
                continue
            lower = fname.lower()
            if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
                continue
            stem = _strip_nii_suffix(fname)
            if "atlas" not in stem.lower():
                continue
            base, suffix = _best_map_job_base(stem)
            path = os.path.join(root, fname)
            existing = discovered[base].get(suffix)
            if existing is None or (existing.endswith(".nii") and path.endswith(".nii.gz")):
                discovered[base][suffix] = path

    return dict(discovered)


def _default_projection_job(base: str) -> MapJob:
    # Use robust bounds by default; most atlas projections use NaNs for background.
    return MapJob(base=base, output_base=f"{base}_projection", cmap_name="specthl")

ParcelStatistics = Dict[Tuple[str, str], Dict[int, float]]

_ATLAS_SEGMENTATION_PATH = (
    "segmentation",
    "segmentation",
    "mri",
    "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz",
)


def _atlas_segmentation_path(nifti_directory: str) -> str:
    return os.path.join(nifti_directory, *_ATLAS_SEGMENTATION_PATH)


def _load_atlas_segmentation_img(
    nifti_directory: str,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
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

    return atlas_img, atlas_data, atlas_labels


def _load_atlas_segmentation(nifti_directory: str) -> tuple[np.ndarray, np.ndarray]:
    _, atlas_data, atlas_labels = _load_atlas_segmentation_img(nifti_directory)
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
    transparent_background: bool = False,
    fixed_colorbar_width: int | None = None,
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
    transparent_background:
        When True, montages are rendered without anatomical underlays and the
        saved PNGs use a transparent background.
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

    # Build reference using head mask for atlas jobs and the brain/union mask for others
    ref_info_head = _build_reference(
        reference, rows * cols, mask=head_mask if head_mask is not None else brain_mask
    )
    ref_info_union = _build_reference(reference, rows * cols, mask=brain_mask)
    ref_info = _combine_reference_info(ref_info_head, ref_info_union)
    if ref_info is None:
        print("[montage] Reference volume contained no finite voxels – skipping montages.")
        return

    out_dir = os.path.join(image_directory, "AI", "Montages")
    os.makedirs(out_dir, exist_ok=True)

    generated_any = False

    # First, discover what we will render so we can precompute a fixed width.
    render_jobs: list[tuple[MapJob, str, str, str, np.ndarray | None, Dict[str, Any] | None]] = []
    for job in MAP_JOBS:
        # Keep atlas maps for perfusion/BBB metrics.
        # Skip only diffusion *_map_atlas* montages. Parcel projections still render below.
        if _is_diffusion_job(job) and job.base.endswith("_map_atlas"):
            continue

        for suffix, map_path in _find_available_maps(job, analysis_directory).items():
            output_name = job.output_base + suffix + job.output_ext
            out_path = os.path.join(out_dir, output_name)
            job_brain_mask = None if job.base.endswith("_map_atlas") else brain_mask
            render_overlay = None if transparent_background else overlay
            render_jobs.append((job, suffix, map_path, out_path, job_brain_mask, render_overlay))

    if not render_jobs:
        print("[montage] No parametric maps found for montage rendering.")
        return

    # Enforce identical PNG widths across all montages by using the maximum required
    # Pillow colorbar width for this batch.
    fixed_cb_w = None
    if Image is not None:
        if fixed_colorbar_width is not None and int(fixed_colorbar_width) > 0:
            fixed_cb_w = int(fixed_colorbar_width)
        else:
            max_w = 0
            for job, _suffix, map_path, _out_path, job_brain_mask, render_overlay in render_jobs:
                try:
                    needed = _pillow_required_colorbar_width_for_montage(
                        map_path,
                        job,
                        reference_img=reference_img,
                        overlay=render_overlay,
                        brain_mask=job_brain_mask,
                        segmentation_img=segmentation_img,
                    )
                    max_w = max(max_w, int(needed))
                except Exception:
                    continue
            fixed_cb_w = int(max(150, max_w)) if max_w else None

    for job, _suffix, map_path, out_path, job_brain_mask, render_overlay in render_jobs:
        try:
            _render_montage(
                map_path,
                out_path,
                job,
                ref_info,
                reference_img=reference_img,
                rows=rows,
                cols=cols,
                dpi=dpi,
                overlay=render_overlay,
                brain_mask=job_brain_mask,
                segmentation_img=segmentation_img,
                transparent_background=transparent_background,
                fixed_colorbar_width=fixed_cb_w,
            )
            generated_any = True
            print(f"[montage] Saved {os.path.relpath(out_path, start=image_directory)}")
        except Exception as exc:
            print(f"[montage] Failed to render {map_path}: {exc}")

    # Optional: export per-slice diagnostics for all available maps.
    if (os.environ.get("P_BRAIN_WRITE_SLICE_DIAGNOSTICS") or "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            _export_per_slice_diagnostics(
                analysis_directory,
                image_directory,
                reference_img=reference_img,
                overlay=overlay,
                segmentation_img=segmentation_img,
                brain_mask=brain_mask,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[montage] Slice diagnostics export failed: {exc}")

    if not generated_any:
        print("[montage] No parametric maps found for montage rendering.")


def _export_per_slice_diagnostics(
    analysis_directory: str,
    image_directory: str,
    *,
    reference_img: nib.Nifti1Image,
    overlay: Dict[str, Any] | None,
    segmentation_img: nib.Nifti1Image | None,
    brain_mask: np.ndarray | None,
) -> None:
    """Write per-slice diagnostic PNGs for each available map.

    Output layout (relative to `image_directory`):
      Images/AI/SliceDiagnostics/<map_base><suffix>/slice_XXX.png
    """

    out_root = os.path.join(image_directory, "AI", "SliceDiagnostics")
    os.makedirs(out_root, exist_ok=True)

    # Reuse the same discovery logic as montage rendering.
    for job in MAP_JOBS:
        if _is_diffusion_job(job) and job.base.endswith("_map_atlas"):
            continue

        found = _find_available_maps(job, analysis_directory)
        if not found:
            continue

        for suffix, map_path in found.items():
            series_name = f"{job.base}{suffix}" if suffix else job.base
            out_dir = os.path.join(out_root, series_name)
            os.makedirs(out_dir, exist_ok=True)
            try:
                _render_map_per_slice(
                    map_path,
                    out_dir,
                    job,
                    reference_img=reference_img,
                    overlay=overlay,
                    segmentation_img=segmentation_img,
                    brain_mask=None if job.base.endswith("_map_atlas") else brain_mask,
                )
                print(f"[montage] Slice diagnostics: wrote {os.path.relpath(out_dir, start=image_directory)}")
            except Exception as exc:
                print(f"[montage] Slice diagnostics failed for {map_path}: {exc}")


def _render_map_per_slice(
    map_path: str,
    out_dir: str,
    job: MapJob,
    *,
    reference_img: nib.Nifti1Image,
    overlay: Dict[str, Any] | None,
    segmentation_img: nib.Nifti1Image | None,
    brain_mask: np.ndarray | None,
) -> None:
    """Render a single parametric map as one PNG per axial slice."""

    img = nib.load(map_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    # Resample to reference grid when needed (keep order=1 for continuous maps).
    if data.shape != reference_img.shape or not np.allclose(np.asarray(img.affine), np.asarray(reference_img.affine)):
        try:
            resampled = resample_from_to(img, reference_img, order=1)
            data = np.asarray(resampled.get_fdata(), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Unable to resample map to reference grid") from exc

    # Build underlay for consistent slice selection/cropping.
    underlay = None
    if overlay is not None and isinstance(overlay.get("t1"), np.ndarray):
        underlay = np.asarray(overlay["t1"], dtype=np.float32)

    # Masking rules.
    mask = None
    if segmentation_img is not None and job.base.endswith("_map_atlas"):
        seg = np.asarray(segmentation_img.get_fdata(), dtype=np.float32)
        mask = np.isfinite(seg) & (seg > 0.5)
    elif brain_mask is not None:
        mask = np.asarray(brain_mask, dtype=bool)

    if mask is not None and mask.shape == data.shape:
        data = np.where(mask, data, np.nan)

    if job.mask_zero:
        data = np.where(data == 0, np.nan, data)

    finite = np.isfinite(data)
    if not np.any(finite):
        return

    vmin = float(job.vmin) if job.vmin is not None else float(np.nanpercentile(data[finite], 2))
    vmax = float(job.vmax) if job.vmax is not None else float(np.nanpercentile(data[finite], 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(data[finite]))
        vmax = float(np.nanmax(data[finite]))
        if vmin == vmax:
            vmax = vmin + 1.0

    cmap = plt.get_cmap(job.cmap_name)
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    try:
        cmap.set_bad("black")
    except Exception:
        pass

    units = _units_for_job(job)

    z = int(data.shape[2])
    for k in range(z):
        sl = data[:, :, k]
        if not np.isfinite(sl).any():
            continue

        sl = _display_orient2d(sl)

        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=160)
        ax.set_title(f"{job.base} slice {k+1}", fontsize=10)

        if underlay is not None and underlay.shape == data.shape:
            bg = _display_orient2d(underlay[:, :, k])
            ax.imshow(bg, cmap="gray", origin="lower")

        im = ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", alpha=0.85 if underlay is not None else 1.0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if units:
            cb.set_label(units, fontsize=9)

        h, w = sl.shape
        ax.set_xticks([0, int(w // 2), int(w - 1)])
        ax.set_yticks([0, int(h // 2), int(h - 1)])
        ax.tick_params(labelsize=7)

        out_path = os.path.join(out_dir, f"slice_{k+1:03d}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def compute_fixed_colorbar_width(
    analysis_directory: str,
    nifti_directory: str,
    dce_path: str,
    *,
    anatomical_overlay: str | None = None,
    segmentation_path: str | None = None,
    head_mask_path: str | None = None,
    icv_mask_path: str | None = None,
    rows: int = ROWS,
    cols: int = COLS,
    dpi: int = DPI,
    population_stats: ParcelStatistics | None = None,
    transparent_background: bool = False,
    include_projections: bool = True,
) -> int | None:
    """Return a single fixed Pillow colorbar width for all outputs.

    The returned value is intended to be passed as ``fixed_colorbar_width`` to both
    ``generate_parametric_montages`` and ``generate_projection_montages`` so every
    rendered PNG shares the same final pixel width.
    """

    if Image is None:
        return None
    if not os.path.isdir(analysis_directory):
        return None
    if not os.path.isfile(dce_path):
        return None

    reference = _load_reference_volume(dce_path)
    if reference is None:
        return None

    try:
        raw_reference_img = nib.load(dce_path)
    except Exception:
        return None

    reference_img = nib.Nifti1Image(
        reference,
        np.array(raw_reference_img.affine, copy=True),
        raw_reference_img.header.copy() if raw_reference_img.header is not None else None,
    )

    segmentation_img = None
    brain_mask: np.ndarray | None = None
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
            segmentation_resampled = np.asarray(segmentation_img.get_fdata(), dtype=np.float32)
            brain_mask = np.isfinite(segmentation_resampled) & (segmentation_resampled > 0.5)
            if not brain_mask.any():
                brain_mask = None
        except FileNotFoundError:
            segmentation_img = None
        except Exception:
            segmentation_img = None
            brain_mask = None

    ext_head_mask, ext_icv_mask = _load_external_masks(
        dce_path,
        reference_img,
        explicit_head=head_mask_path,
        explicit_icv=icv_mask_path,
    )

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
            overlay = None
        if brain_mask is None and overlay is not None and isinstance(overlay.get("mask_brain"), np.ndarray):
            brain_mask = overlay["mask_brain"].astype(bool)
        if overlay is not None and isinstance(overlay.get("mask_head"), np.ndarray):
            head_mask = overlay["mask_head"].astype(bool)
    else:
        if ext_head_mask is not None:
            head_mask = ext_head_mask
        if brain_mask is None and ext_icv_mask is not None:
            brain_mask = ext_icv_mask

    render_overlay = None if transparent_background else overlay

    max_w = 0
    # Parametric montages.
    for job in MAP_JOBS:
        if _is_diffusion_job(job) and job.base.endswith("_map_atlas"):
            continue

        for _suffix, map_path in _find_available_maps(job, analysis_directory).items():
            try:
                job_brain_mask = None if job.base.endswith("_map_atlas") else brain_mask
                needed = _pillow_required_colorbar_width_for_montage(
                    map_path,
                    job,
                    reference_img=reference_img,
                    overlay=render_overlay,
                    brain_mask=job_brain_mask,
                    segmentation_img=segmentation_img,
                )
                max_w = max(max_w, int(needed))
            except Exception:
                continue

    # Projection montages.
    if include_projections and os.path.isdir(nifti_directory):
        try:
            atlas_img, atlas_data, atlas_labels = _load_atlas_segmentation_img(nifti_directory)
        except Exception:
            atlas_img = None
            atlas_data = None
            atlas_labels = None

        if atlas_data is not None and atlas_labels is not None:
            stats_lookup: Mapping[Tuple[str, str], Dict[int, float]] = population_stats or {}
            discovered = _discover_atlas_maps(analysis_directory)
            targets: Dict[str, str] = dict(PROJECTION_TARGETS)
            for base in discovered:
                if base not in targets:
                    targets[base] = f"{base.lower()}_projection_parcel"

            render_items: list[tuple[np.ndarray, MapJob]] = []
            for base in sorted(targets):
                job = MAP_JOB_LOOKUP.get(base) or _default_projection_job(base)
                if _is_diffusion_job(job) and job.base.endswith("_map_atlas") and not job.mask_zero:
                    job = replace(job, mask_zero=True)
                available_maps = discovered.get(base, {})
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
                        if projected is None and map_path and atlas_img is not None:
                            projected = _parcel_mean_projection(
                                map_path,
                                atlas_data,
                                atlas_labels,
                                atlas_img=atlas_img,
                            )
                        if projected is None:
                            continue
                        render_items.append((projected, job))
                    except Exception:
                        continue

            try:
                seg_job = MapJob(
                    base="atlas_segmentation",
                    output_base="atlas_segmentation_projection",
                    vmin=float(np.min(atlas_labels)),
                    vmax=float(np.max(atlas_labels)),
                    cmap_name="tab20",
                    mask_zero=True,
                )
                render_items.append((atlas_data.astype(np.float32), seg_job))
            except Exception:
                pass

            for arr, job in render_items:
                try:
                    needed = _pillow_required_colorbar_width_for_projection(arr, job)
                    max_w = max(max_w, int(needed))
                except Exception:
                    continue

    return int(max(150, max_w)) if max_w else None


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
    transparent_background: bool = False,
    fixed_colorbar_width: int | None = None,
) -> bool:
    """Render parcel-level projection montages for atlas-based metrics.

    When ``transparent_background`` is True the saved PNGs omit the default
    grey canvas to simplify downstream compositing.
    """

    if not os.path.isdir(analysis_directory):
        return False
    if not os.path.isdir(nifti_directory):
        print(f"[projection] NIfTI directory missing – skipping: {nifti_directory}")
        return False
    if not os.path.isfile(dce_path):
        print(f"[projection] DCE reference not found – skipping projection rendering: {dce_path}")
        return False

    try:
        atlas_img, atlas_data, atlas_labels = _load_atlas_segmentation_img(nifti_directory)
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

    try:
        raw_reference_img = nib.load(dce_path)
    except Exception as exc:  # noqa: BLE001 - propagate context to CLI users
        print(f"[projection] Unable to load DCE reference volume – skipping projections: {exc}")
        return False

    reference_img = nib.Nifti1Image(
        reference,
        np.array(raw_reference_img.affine, copy=True),
        raw_reference_img.header.copy() if raw_reference_img.header is not None else None,
    )

    brain_mask = np.isfinite(atlas_data) & (atlas_data > 0)
    ref_info = _build_reference(reference, rows * cols, mask=brain_mask)
    if ref_info is None:
        print("[projection] Reference volume contained no finite voxels – skipping projections.")
        return False

    out_dir = os.path.join(image_directory, "AI", "Montages")
    os.makedirs(out_dir, exist_ok=True)

    generated_any = False
    stats_lookup: Mapping[Tuple[str, str], Dict[int, float]] = population_stats or {}

    discovered = _discover_atlas_maps(analysis_directory)
    targets: Dict[str, str] = dict(PROJECTION_TARGETS)
    for base in discovered:
        if base not in targets:
            targets[base] = f"{base.lower()}_projection_parcel"

    # Build a render plan first so we can enforce identical widths.
    render_items: list[tuple[np.ndarray, MapJob, str]] = []
    for base in sorted(targets):
        output_base = targets[base]
        job = MAP_JOB_LOOKUP.get(base) or _default_projection_job(base)
        if _is_diffusion_job(job) and job.base.endswith("_map_atlas") and not job.mask_zero:
            job = replace(job, mask_zero=True)
        available_maps = discovered.get(base, {})
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
                    projected = _parcel_mean_projection(
                        map_path,
                        atlas_data,
                        atlas_labels,
                        atlas_img=atlas_img,
                    )
                if projected is None:
                    continue

                output_name = output_base + suffix + job.output_ext
                out_path = os.path.join(out_dir, output_name)
                render_items.append((projected, job, out_path))
            except Exception as exc:
                target = map_path if map_path else f"population statistics for {base}{suffix}"
                print(f"[projection] Failed to render {target}: {exc}")

    # Always render the atlas segmentation itself as a reference projection.
    try:
        seg_job = MapJob(
            base="atlas_segmentation",
            output_base="atlas_segmentation_projection",
            vmin=float(np.min(atlas_labels)),
            vmax=float(np.max(atlas_labels)),
            cmap_name="tab20",
            mask_zero=True,
        )
        seg_out = os.path.join(out_dir, "atlas_segmentation_projection.png")
        render_items.append((atlas_data.astype(np.float32), seg_job, seg_out))
    except Exception as exc:
        print(f"[projection] Failed to render atlas segmentation projection: {exc}")

    if not render_items:
        if not generated_any:
            print("[projection] No atlas maps found for projection rendering.")
        return generated_any

    fixed_cb_w = None
    if Image is not None:
        if fixed_colorbar_width is not None and int(fixed_colorbar_width) > 0:
            fixed_cb_w = int(fixed_colorbar_width)
        else:
            max_w = 0
            for arr, job, _out_path in render_items:
                try:
                    needed = _pillow_required_colorbar_width_for_projection(arr, job)
                    max_w = max(max_w, int(needed))
                except Exception:
                    continue
            fixed_cb_w = int(max(150, max_w)) if max_w else None

    for arr, job, out_path in render_items:
        try:
            _render_projection_montage(
                arr,
                ref_info,
                job,
                out_path,
                rows=rows,
                cols=cols,
                dpi=dpi,
                reference_img=reference_img,
                transparent_background=transparent_background,
                fixed_colorbar_width=fixed_cb_w,
            )
            generated_any = True
            print(f"[projection] Saved {os.path.relpath(out_path, start=image_directory)}")
        except Exception as exc:
            print(f"[projection] Failed to render {out_path}: {exc}")

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
    map_path: str,
    atlas_data: np.ndarray,
    atlas_labels: np.ndarray,
    *,
    atlas_img: nib.Nifti1Image | None = None,
) -> Dict[int, float]:
    img = nib.load(map_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    atlas_for_map = atlas_data
    if atlas_img is not None and (
        data.shape != atlas_data.shape
        or not np.allclose(np.asarray(atlas_img.affine), np.asarray(img.affine))
    ):
        try:
            from nibabel.processing import resample_from_to

            resampled = resample_from_to(atlas_img, img, order=0)
            atlas_for_map = np.asarray(resampled.get_fdata(), dtype=np.int32)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Atlas segmentation and parametric map shapes do not match"
            ) from exc
    elif data.shape != atlas_data.shape:
        raise ValueError("Atlas segmentation and parametric map shapes do not match")

    label_means: Dict[int, float] = {}
    for label in atlas_labels:
        mask = atlas_for_map == int(label)
        if not np.any(mask):
            continue
        values = data[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        label_means[int(label)] = float(np.mean(values, dtype=np.float32))

    return label_means


def _collect_dataset_parcel_means(
    analysis_directory: str,
    atlas_data: np.ndarray,
    atlas_labels: np.ndarray,
    *,
    atlas_img: nib.Nifti1Image | None = None,
) -> Dict[Tuple[str, str], Dict[int, float]]:
    dataset_means: Dict[Tuple[str, str], Dict[int, float]] = {}

    discovered = _discover_atlas_maps(analysis_directory)
    for base, suffix_map in discovered.items():
        for suffix, map_path in suffix_map.items():
            try:
                label_means = _parcel_label_means(
                    map_path,
                    atlas_data,
                    atlas_labels,
                    atlas_img=atlas_img,
                )
            except Exception:  # noqa: BLE001 - skip unreadable/corrupt files
                continue
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
            atlas_img, atlas_data, atlas_labels = _load_atlas_segmentation_img(
                nifti_directory
            )
        except (FileNotFoundError, ValueError):
            continue

        try:
            dataset_means = _collect_dataset_parcel_means(
                analysis_directory,
                atlas_data,
                atlas_labels,
                atlas_img=atlas_img,
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


def _dim_rgba(rgba: tuple[float, float, float, float], gain: float = 0.55) -> tuple[float, float, float, float]:
    """Darken an RGBA color without touching alpha."""
    r, g, b, a = rgba
    return (r * gain, g * gain, b * gain, a)


def _opaque_for_image(cmap: mpl.colors.Colormap) -> mpl.colors.Colormap:
    """
    For image tiles: force alpha=1 for all colors so values never blend with the
    axes facecolor. Keep NaN ('bad') fully transparent so the anatomical underlay
    shows through.
    """
    lut = cmap(np.linspace(0, 1, getattr(cmap, "N", 256)))
    lut[:, -1] = 1.0
    out = mpl.colors.ListedColormap(lut, name=getattr(cmap, "name", "cm") + "_imgopaque")
    # preserve your endpoint choices and make them opaque
    under = list(cmap(0.0)); under[-1] = 1.0
    over  = list(cmap(1.0)); over[-1]  = 1.0
    out.set_under(tuple(under))
    out.set_over(tuple(over))
    # NaNs transparent only in tiles
    out.set_bad((0, 0, 0, 0))
    return out
def _opaque_colormap_for_colorbar(cmap: mpl.colors.Colormap) -> mpl.colors.Colormap:
    """
    For colorbars: use the full untruncated range [0, 1] with alpha=1 everywhere,
    same endpoints as the tiles, and no transparency. This guarantees the first
    color of the bar equals the vmin color seen in the image.
    """
    base = cmap
    lut = base(np.linspace(0, 1, getattr(base, "N", 256)))
    lut[:, -1] = 1.0
    out = mpl.colors.ListedColormap(lut, name=getattr(base, "name", "cm") + "_cbopaque")
    # match endpoints exactly
    under = list(base(0.0)); under[-1] = 1.0
    over  = list(base(1.0)); over[-1]  = 1.0
    out.set_under(tuple(under))
    out.set_over(tuple(over))
    # never transparent inside the bar
    out.set_bad(tuple(under))  # harmless, ScalarMappable will not feed NaN to the bar
    return out

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

    if mask_head is not None:
        mask_head = _polish_mask(
            mask_head,
            overlay_img,
            min_size=HEAD_MIN_VOXELS,
            closing_mm=MASK_SMOOTH_MM,
            hole_voxels=MASK_HOLE_VOXELS,
        )
    if mask_brain is not None:
        mask_brain = _polish_mask(
            mask_brain,
            overlay_img,
            min_size=None,
            closing_mm=MASK_SMOOTH_MM,
            hole_voxels=MASK_HOLE_VOXELS,
        )

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
        "alpha": 0.85,
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


def _polish_mask(
    mask: np.ndarray,
    img: nib.Nifti1Image,
    *,
    min_size: int | None = None,
    closing_mm: float | None = MASK_SMOOTH_MM,
    hole_voxels: int | None = MASK_HOLE_VOXELS,
    fill_holes: bool = True,
) -> np.ndarray:
    """Regularise a binary mask to avoid ragged rims and voids."""

    polished = np.asarray(mask, dtype=bool)
    if not polished.any():
        return polished

    if fill_holes:
        polished = binary_fill_holes(polished)

    if hole_voxels is not None and hole_voxels > 0:
        try:
            polished = remove_small_holes(
                polished, area_threshold=int(hole_voxels), connectivity=2
            )
        except Exception:  # noqa: BLE001 - keep the best effort result
            pass

    if closing_mm is not None and closing_mm > 0:
        try:
            polished = binary_closing(polished, ball(_voxel_radius(img, closing_mm)))
        except Exception:  # noqa: BLE001 - keep the best effort result
            pass

    if min_size is not None and min_size > 0:
        try:
            polished = remove_small_objects(polished, min_size=min_size, connectivity=2)
        except Exception:  # noqa: BLE001
            pass

    polished = _largest_component(polished)
    return polished.astype(bool)


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
    soft = _polish_mask(
        soft,
        img,
        min_size=HEAD_MIN_VOXELS,
        closing_mm=MASK_SMOOTH_MM,
        hole_voxels=MASK_HOLE_VOXELS,
    )
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
            brain = _polish_mask(
                brain,
                overlay_img,
                min_size=HEAD_MIN_VOXELS // 2,
                closing_mm=MASK_SMOOTH_MM,
                hole_voxels=MASK_HOLE_VOXELS,
            )
            icv_prior = brain.astype(bool)
            r = _voxel_radius(overlay_img, dilate_mm)
            grown = binary_dilation(icv_prior, ball(r))
            head = np.asarray(head | grown, dtype=bool)
    r_erode = _voxel_radius(overlay_img, erode_mm)
    icv_from_head = binary_erosion(head, ball(r_erode))
    icv_from_head = _polish_mask(
        icv_from_head,
        overlay_img,
        min_size=HEAD_MIN_VOXELS // 2,
        closing_mm=MASK_SMOOTH_MM,
        hole_voxels=MASK_HOLE_VOXELS,
    )

    if icv_prior is not None:
        icv = _polish_mask(
            icv_from_head | icv_prior,
            overlay_img,
            min_size=HEAD_MIN_VOXELS // 2,
            closing_mm=MASK_SMOOTH_MM,
            hole_voxels=MASK_HOLE_VOXELS,
        )
    else:
        icv = icv_from_head

    head = _polish_mask(
        head,
        overlay_img,
        min_size=HEAD_MIN_VOXELS,
        closing_mm=MASK_SMOOTH_MM,
        hole_voxels=MASK_HOLE_VOXELS,
    )
    return head.astype(bool), icv.astype(bool)


def _mask_erode_mm(mask: np.ndarray, img: nib.Nifti1Image, mm: float) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if not source.any() or mm <= 0:
        return source.astype(bool)
    try:
        radius = max(1, _voxel_radius(img, mm))
        eroded = binary_erosion(source, ball(radius))
        if not eroded.any():
            return source.astype(bool)
        return eroded.astype(bool)
    except Exception:
        return source.astype(bool)


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


def _alpha_feather_slice(
    mask: np.ndarray,
    zoom_xy: Sequence[float],
    *,
    scale_y: float = 1.0,
    scale_x: float = 1.0,
    feather_mm: float = MAP_EDGE_FEATHER_MM,
) -> np.ndarray:
    mask2d = np.asarray(mask, dtype=bool)
    if mask2d.ndim != 2:
        raise ValueError("Slice feather mask must be 2D")
    if not mask2d.any():
        return np.zeros(mask2d.shape, dtype=np.float32)

    zoom_y, zoom_x = float(zoom_xy[0]), float(zoom_xy[1])
    pixel_y = zoom_y / max(scale_y, 1e-6)
    pixel_x = zoom_x / max(scale_x, 1e-6)
    sampling = (max(pixel_y, 1e-6), max(pixel_x, 1e-6))

    dist = distance_transform_edt(mask2d, sampling=sampling)
    alpha = np.clip(dist / max(feather_mm, 1e-6), 0.0, 1.0).astype(np.float32)
    alpha[~mask2d] = 0.0
    if alpha.size:
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
        mask = _polish_mask(
            mask,
            reference_img,
            min_size=HEAD_MIN_VOXELS // 2,
            closing_mm=MASK_SMOOTH_MM,
            hole_voxels=MASK_HOLE_VOXELS,
        )
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
            mask = _polish_mask(
                mask,
                reference_img,
                min_size=HEAD_MIN_VOXELS // 2,
                closing_mm=MASK_SMOOTH_MM,
                hole_voxels=MASK_HOLE_VOXELS,
            )
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
            mask = _polish_mask(
                mask,
                overlay_img,
                min_size=HEAD_MIN_VOXELS // 2,
                closing_mm=MASK_SMOOTH_MM,
                hole_voxels=MASK_HOLE_VOXELS,
            )
            radius = _voxel_radius(overlay_img, ICV_ERODE_MM if ICV_ERODE_MM > 0 else 1.0)
            head = binary_dilation(mask, ball(max(1, radius)))
            head = _polish_mask(
                head,
                overlay_img,
                min_size=HEAD_MIN_VOXELS,
                closing_mm=MASK_SMOOTH_MM,
                hole_voxels=MASK_HOLE_VOXELS,
            )
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


def _combine_reference_info(
    primary: Dict[str, np.ndarray] | None,
    secondary: Dict[str, np.ndarray] | None,
) -> Dict[str, np.ndarray] | None:
    """Merge reference metadata so every montage shares the same framing."""

    refs = [ref for ref in (primary, secondary) if ref is not None]
    if not refs:
        return None

    rotate = refs[0]["rotate"]
    z_fracs = np.array(refs[0]["z_fracs"], copy=True)
    bbox = dict(refs[0]["bbox_fracs"])

    for ref in refs[1:]:
        if ref["rotate"] != rotate:
            raise ValueError("Reference rotations do not match")
        bbox_ref = ref["bbox_fracs"]
        bbox["r0_frac"] = min(bbox["r0_frac"], bbox_ref["r0_frac"])
        bbox["r1_frac"] = max(bbox["r1_frac"], bbox_ref["r1_frac"])
        bbox["c0_frac"] = min(bbox["c0_frac"], bbox_ref["c0_frac"])
        bbox["c1_frac"] = max(bbox["c1_frac"], bbox_ref["c1_frac"])
        if ref["z_fracs"].size > z_fracs.size:
            z_fracs = np.array(ref["z_fracs"], copy=True)

    return {
        "bbox_fracs": bbox,
        "z_fracs": z_fracs,
        "rotate": rotate,
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


def _map_z_from_ref(
    z_fracs: np.ndarray,
    nz: int,
    *,
    zmin: int | None = None,
    zmax: int | None = None,
) -> np.ndarray:
    z = np.asarray(z_fracs, dtype=np.float64)
    if nz <= 0 or z.size == 0:
        return np.zeros(z.size, dtype=np.intp)

    if zmin is None:
        zmin = 0
    if zmax is None:
        zmax = nz - 1

    zmin = int(max(0, min(zmin, nz - 1)))
    zmax = int(max(zmin, min(zmax, nz - 1)))
    span = max(1, zmax - zmin)

    idx = np.rint(z * span).astype(np.intp, copy=False) + zmin
    idx = np.clip(idx, zmin, zmax)

    if idx.size == 0:
        return idx

    idx[0] = zmin
    idx[-1] = zmax

    if idx.size > 1:
        for i in range(1, idx.size):
            if idx[i] <= idx[i - 1]:
                idx[i] = min(zmax, idx[i - 1] + 1)

    return idx


def _slice_valid_bounds(
    primary: np.ndarray | None, fallback: np.ndarray | None
) -> tuple[int | None, int | None]:
    p = np.flatnonzero(primary) if isinstance(primary, np.ndarray) and primary.size else np.array([], dtype=int)
    f = np.flatnonzero(fallback) if isinstance(fallback, np.ndarray) and fallback.size else np.array([], dtype=int)
    if p.size or f.size:
        lo = int(min(p[0] if p.size else f[0], f[0] if f.size else p[0]))
        hi = int(max(p[-1] if p.size else f[-1], f[-1] if f.size else p[-1]))
        return lo, hi
    return None, None


def _resize_mask(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    target_shape = (int(target_shape[0]), int(target_shape[1]))
    if mask.shape == target_shape:
        return mask.astype(bool)
    # Masks require categorical resampling. Linear + AA creates pinholes.
    resized = resize(
        mask.astype(np.uint8),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)
    # Seal tiny artifacts introduced by raster changes
    try:
        from scipy.ndimage import binary_closing, binary_fill_holes
        resized = binary_fill_holes(binary_closing(resized, structure=np.ones((3,3), bool)))
    except Exception:
        pass
    return resized.astype(bool)


def _fill_empty_slices_nearest(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        return data

    slice_has_data = np.isfinite(data).any(axis=(0, 1))
    if slice_has_data.all():
        return data

    valid_indices = np.flatnonzero(slice_has_data)
    if valid_indices.size == 0:
        return data

    filled = np.array(data, copy=True)
    for idx in np.flatnonzero(~slice_has_data):
        nearest = valid_indices[np.argmin(np.abs(valid_indices - idx))]
        filled[:, :, idx] = filled[:, :, nearest]
    return filled


def _inpaint_nans_nearest(
    volume: np.ndarray,
    inside_mask: np.ndarray | None = None,
) -> np.ndarray:
    """3D nearest-neighbour inpaint of NaN islands. Only fills NaNs inside `inside_mask` if provided."""
    if volume.ndim != 3:
        return volume
    vol = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(vol)
    if inside_mask is None:
        valid = finite
        target = ~finite
    else:
        dom = np.asarray(inside_mask, dtype=bool)
        valid = finite & dom
        target = (~finite) & dom
    if not np.any(target) or not np.any(valid):
        return vol
    _, idx = distance_transform_edt(~valid, return_indices=True)
    out = vol.copy()
    out[target] = vol[tuple(idx[d][target] for d in range(3))]
    return out


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
    map_path: str,
    atlas_data: np.ndarray,
    atlas_labels: np.ndarray,
    *,
    atlas_img: nib.Nifti1Image | None = None,
) -> np.ndarray | None:
    img = nib.load(map_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    atlas_for_map = atlas_data
    if atlas_img is not None and (
        data.shape != atlas_data.shape
        or not np.allclose(np.asarray(atlas_img.affine), np.asarray(img.affine))
    ):
        try:
            from nibabel.processing import resample_from_to

            resampled = resample_from_to(atlas_img, img, order=0)
            atlas_for_map = np.asarray(resampled.get_fdata(), dtype=np.int32)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Atlas segmentation and parametric map shapes do not match"
            ) from exc
    elif data.shape != atlas_data.shape:
        raise ValueError("Atlas segmentation and parametric map shapes do not match")

    projected = np.full_like(data, np.nan, dtype=np.float32)
    for label in atlas_labels:
        mask = atlas_for_map == label
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
    transparent_background: bool = False,
    fixed_colorbar_width: int | None = None,
) -> None:
    # Prefer deterministic Pillow rendering (no Matplotlib figure/colorbar).
    # Fall back to the Matplotlib renderer only when Pillow is unavailable.
    if Image is not None:
        _render_montage_pillow(
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
            transparent_background=transparent_background,
            fixed_colorbar_width=fixed_colorbar_width,
        )
        return

    is_atlas = job.base.endswith("_map_atlas")
    is_diffusion = _is_diffusion_job(job)
    img = nib.load(map_path)
    # Resample to DCE grid. Atlas maps use nearest neighbour to keep parcels crisp.
    from nibabel.processing import resample_from_to
    target = (reference_img.shape, reference_img.affine)
    if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine):
        img = resample_from_to(img, target, order=0 if is_atlas else 1)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    # Skip smoothing for atlas maps to avoid value bleeding.
    # Compute slice-valid AFTER any repair of empty slices.
    if not is_atlas:
        try:
            zoom_src = np.array(img.header.get_zooms()[:3], dtype=float)
            zoom_ref = np.array(reference_img.header.get_zooms()[:3], dtype=float)
            ratio = max(zoom_src[0] / zoom_ref[0], zoom_src[1] / zoom_ref[1])
            if ratio > 1.4:
                data = gaussian_filter(data, sigma=(0.6, 0.6, 0.0), mode="nearest")
        except Exception:
            pass
    else:
        data = _fill_empty_slices_nearest(data)
    # Diffusion voxelwise often carries NaN islands after the fit
    if is_diffusion and not is_atlas:
        # Prefer ICV as the inpaint domain. If missing, use a soft proxy from finite voxels.
        inpaint_domain = None
        if brain_mask is not None and brain_mask.shape == data.shape:
            inpaint_domain = brain_mask
        else:
            finite = np.isfinite(data)
            # Grow a little to bridge one-voxel gaps without bleeding into air.
            try:
                from scipy.ndimage import binary_closing

                inpaint_domain = binary_closing(finite, structure=np.ones((3, 3, 3), bool))
            except Exception:
                inpaint_domain = finite
        data = _inpaint_nans_nearest(data, inside_mask=inpaint_domain)
    slice_valid_initial = (
        np.isfinite(data).any(axis=(0, 1)) if data.ndim == 3 else np.array([], dtype=bool)
    )
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    # Fence the domain by ICV/head when available (voxelwise only)
    if brain_mask is not None and not is_atlas:
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
        # Work inside ICV and keep every finite voxel
        valmask3d = np.isfinite(data) & mask_data
    else:
        if is_atlas:
            # Build atlas display support from segmentation labels when available.
            atlas_support = None
            if segmentation_img is not None:
                from nibabel.processing import resample_from_to

                target = (data.shape, img.affine)
                seg_img = segmentation_img
                if (
                    segmentation_img.shape != data.shape
                    or not np.allclose(segmentation_img.affine, img.affine)
                ):
                    seg_img = resample_from_to(segmentation_img, target, order=0)
                seg = np.asarray(seg_img.get_fdata(), dtype=np.float32)
                atlas_support = np.isfinite(seg) & (seg > 0.5)
            if atlas_support is None:
                finite = np.isfinite(data)
                try:
                    from scipy.ndimage import binary_closing, binary_fill_holes

                    atlas_support = binary_fill_holes(
                        binary_closing(finite, structure=np.ones((3,3,3), bool))
                    )
                except Exception:
                    atlas_support = finite
            valmask3d = atlas_support.astype(bool)
        else:
            # Non-atlas, no explicit brain mask – accept all finite voxels.
            valmask3d = np.isfinite(data)
    # Use head/ICV to frame the crop whenever we have it
    if brain_mask is not None and not is_atlas:
        union_xy = np.any(brain_mask, axis=2)
    else:
        # For atlas maps, frame by the atlas support, not by "all ones".
        union_xy = np.any(valmask3d, axis=2)
    union_xy_r = np.rot90(union_xy, ref_info["rotate"])

    r0, r1, c0, c1 = _map_bbox_from_ref(ref_info["bbox_fracs"], union_xy_r.shape)

    slice_valid = np.any(valmask3d, axis=(0, 1)) if valmask3d.ndim == 3 else None
    # Parcels should span the full slab so first and last slice always appear
    if is_atlas:
        zmin, zmax = 0, data.shape[2] - 1
    else:
        zmin, zmax = _slice_valid_bounds(slice_valid_initial, slice_valid)
    z_indices = _map_z_from_ref(ref_info["z_fracs"], data.shape[2], zmin=zmin, zmax=zmax)
    if z_indices.size < rows * cols:
        pad_value = z_indices[-1] if z_indices.size else 0
        z_indices = np.pad(z_indices, (0, rows * cols - z_indices.size), constant_values=pad_value)
    else:
        z_indices = z_indices[: rows * cols]

    cmap = _get_cmap(job.cmap_name)
    # Build normalizer on the display domain. Prefer T1 HEAD when available.
    head_support_3d = None
    if overlay is not None and isinstance(overlay.get("mask_head"), np.ndarray):
        head_support_3d = overlay["mask_head"].astype(bool)
        if head_support_3d.shape != data.shape:
            from nibabel.processing import resample_from_to
            head_img = nib.Nifti1Image(head_support_3d.astype(np.float32), reference_img.affine)
            head_img = resample_from_to(head_img, (data.shape, img.affine), order=0)
            head_support_3d = np.asarray(head_img.get_fdata(), dtype=np.float32) > 0.5
    norm_data = data if brain_mask is None else np.where(brain_mask, data, np.nan)
    if is_atlas and head_support_3d is not None:
        norm_data = np.where(head_support_3d, norm_data, np.nan)
    focus_data = None
    metric_tag = (getattr(job, "metric", "") or "").lower()
    if metric_tag == "ki":
        core_mask = None
        if brain_mask is not None and brain_mask.shape == data.shape:
            core_mask = _mask_erode_mm(brain_mask, img, mm=2.0)
        elif head_support_3d is not None and head_support_3d.shape == data.shape:
            core_mask = _mask_erode_mm(head_support_3d, img, mm=2.0)
        if core_mask is not None and core_mask.any():
            focus_data = np.where(core_mask, data, np.nan)
        else:
            focus_data = norm_data

    norm, tick_values = _build_normalizer(
        norm_data,
        job,
        mask_zero_override=False if brain_mask is not None else None,
        focus_data=focus_data,
    )
    if (getattr(job, "metric", None) or "").lower() == "fa":
        vmax = float(getattr(norm, "vmax", 1.0))
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=False)
    # NaNs are transparent. Sub-vmin gets the lowest real colour, no fake black pits.
    base_under = list(cmap(0.0))
    base_under[-1] = 1.0
    cmap = cmap.with_extremes(bad=(0, 0, 0, 0), under=tuple(base_under))
    # Make fully opaque variants for image and colorbar
    cmap_img = _opaque_for_image(cmap)
    cmap_cb = _opaque_colormap_for_colorbar(cmap)

    axis_facecolor = (0, 0, 0, 0) if transparent_background else "#e0e0e0"
    fig_facecolor = (0, 0, 0, 0) if transparent_background else axis_facecolor
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), facecolor=fig_facecolor)
    fig.patch.set_alpha(0.0 if transparent_background else 1.0)
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
    overlay_slice_valid_initial: np.ndarray | None = None
    overlay_slice_valid_masked: np.ndarray | None = None
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
            if overlay_data.ndim == 3:
                overlay_slice_valid_initial = np.isfinite(overlay_data).any(axis=(0, 1))
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
            if overlay_mask_data is not None:
                overlay_mask_data = np.asarray(overlay_mask_data, dtype=bool)
                if overlay_mask_data.ndim == 3 and overlay_mask_data.shape[2]:
                    footprint2d = np.ones((3, 3), dtype=bool)
                    for idx in range(overlay_mask_data.shape[2]):
                        slice_mask = overlay_mask_data[:, :, idx]
                        try:
                            slice_mask = binary_fill_holes(slice_mask)
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            slice_mask = binary_closing(slice_mask, footprint=footprint2d)
                        except Exception:  # noqa: BLE001
                            pass
                        overlay_mask_data[:, :, idx] = slice_mask
                if overlay_mask_data.ndim == 3:
                    overlay_slice_valid_masked = overlay_mask_data.any(axis=(0, 1))
            if overlay_data is not None:
                ovl_zmin, ovl_zmax = _slice_valid_bounds(
                    overlay_slice_valid_initial, overlay_slice_valid_masked
                )
                overlay_z_indices = _map_z_from_ref(
                    ref_info["z_fracs"], overlay_data.shape[2], zmin=ovl_zmin, zmax=ovl_zmax
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

    try:
        zoom_xy = np.array(reference_img.header.get_zooms()[:2], dtype=float)
    except Exception:
        zoom_xy = np.array([1.0, 1.0], dtype=float)
    if not np.all(np.isfinite(zoom_xy)) or zoom_xy.size < 2:
        zoom_xy = np.array([1.0, 1.0], dtype=float)
    if ref_info["rotate"] % 2:
        zoom_xy = zoom_xy[::-1]

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
        ax.set_facecolor(axis_facecolor)
        for spine in ax.spines.values():
            spine.set_visible(False)

        sl = data[:, :, zi]
        slr = np.rot90(sl, ref_info["rotate"])
        slc = slr[r0:r1, c0:c1]
        if np.isfinite(slc).sum() == 0:
            print(f"[montage] z={zi}: all NaN after mask/crop")

        # Display support from T1 HEAD if present; fallback to union of valid voxels.
        union_crop = union_xy_r[r0:r1, c0:c1]
        extent = (0.0, 1.0, 1.0, 0.0)
        # Predeclare to avoid UnboundLocalError on branches
        overlay_union_crop = None
        overlay_alpha_crop = None
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
            if overlay_mask_data is not None:
                mask_slice_raw = overlay_mask_data[:, :, index_for_overlay]
                mask_rot = np.rot90(mask_slice_raw, ref_info["rotate"])
                overlay_union = mask_rot.astype(bool)
            else:
                overlay_union = union_xy_r
            overlay_union_crop = overlay_union[overlay_r0:overlay_r1, overlay_c0:overlay_c1]
            if overlay_alpha_data is not None:
                alpha_slice = overlay_alpha_data[:, :, index_for_overlay]
                alpha_rot = np.rot90(alpha_slice, ref_info["rotate"])
                overlay_alpha_crop = alpha_rot[
                    overlay_r0:overlay_r1, overlay_c0:overlay_c1
                ]
                # Safe mask for alpha ramp even if union is degenerate
                if overlay_union_crop is None:
                    overlay_union_crop = union_crop
                overlay_alpha_crop = overlay_alpha_crop * overlay_union_crop.astype(np.float32)
                overlay_alpha_crop = np.clip(overlay_alpha_crop, 0.0, 1.0)
            overlay_mask_slice = None
            if overlay_mask_data is not None:
                mask_slice_raw = overlay_mask_data[:, :, index_for_overlay]
                mask_rot = np.rot90(mask_slice_raw, ref_info["rotate"])
                overlay_mask_slice = mask_rot[overlay_r0:overlay_r1, overlay_c0:overlay_c1]
            # If overlay_union_crop was not set for any reason, fall back to T1 union crop
            if overlay_union_crop is None:
                overlay_union_crop = union_crop
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
                alpha=(overlay_alpha * overlay_alpha_crop if overlay_alpha_crop is not None else overlay_alpha),
            )
            render_shape = overlay_crop.shape

        mask_slice = valmask3d[:, :, zi]
        mask_slice_rot = np.rot90(mask_slice, ref_info["rotate"])
        mask_slice_crop = mask_slice_rot[r0:r1, c0:c1]

        # Value clipping only for explicit mask_zero jobs, never for atlas/diffusion support logic.
        if (brain_mask is None) and job.mask_zero and not is_atlas:
            finite_vals = slc[np.isfinite(slc) & (slc > 0)]
            if finite_vals.size:
                cutoff = np.percentile(finite_vals, 0.1)
                eps_dyn = max(cutoff, 1e-6)
            else:
                eps_dyn = 1e-6
            mask_slice_crop &= slc > eps_dyn

        # Atlas: treat NaNs as zeros inside support so parcels render continuous.
        if is_atlas:
            slc_filled = np.where(np.isfinite(slc), slc, 0.0)
        else:
            slc_filled = np.array(slc, copy=True)
            slc_filled[~mask_slice_crop] = 0.0
        slc_render = slc_filled.astype(np.float32)
        mask_render = mask_slice_crop
        union_render = union_crop
        if render_shape != slc.shape:
            slc_render = resize(
                slc_render,
                render_shape,
                order=0 if is_atlas else 1,
                preserve_range=True,
                anti_aliasing=False if is_atlas else True,
            ).astype(np.float32)
            # Keep support-consistent masks after raster change.
            mask_render = _resize_mask(mask_slice_crop, render_shape)
            union_render = _resize_mask(union_crop, render_shape)
        # Final visible domain = T1 HEAD support ∩ map support
        valid_mask = union_render & mask_render
        try:
            from scipy.ndimage import binary_closing, binary_fill_holes
            valid_mask = binary_fill_holes(binary_closing(valid_mask, structure=np.ones((3,3), bool)))
        except Exception:
            pass
        # Atlas: do NOT mask on NaN after we replaced them with zeros; just respect the support.
        if is_atlas:
            arr = np.ma.array(slc_render, mask=(~valid_mask))
        else:
            arr = np.ma.array(slc_render, mask=(~valid_mask) | (~np.isfinite(slc_render)))
        # If the slice is entirely below global vmin (e.g., FA in cortex),
        # use a local min/max so low-but-real values remain visible.
        scale_y = render_shape[0] / max(1, slc.shape[0])
        scale_x = render_shape[1] / max(1, slc.shape[1])
        # Hard mask for the parametric layer
        alpha_values = valid_mask.astype(np.float32)
        if alpha_values.shape != slc_render.shape:
            alpha_values = alpha_values.astype(np.float32, copy=False)
            alpha_values = _resize_mask(alpha_values > 0.5, slc_render.shape).astype(np.float32)
        finite_vals = arr.compressed() if hasattr(arr, "compressed") else np.asarray([], dtype=float)
        # One global scale for the whole figure
        ax.imshow(
            arr,
            cmap=cmap_img,
            norm=norm,
            interpolation="nearest",
            origin="upper",
            alpha=alpha_values,
            extent=extent,
        )

    # Hide any unused axes when there are fewer slices than tiles
    for ax in axes[len(z_indices) :]:
        ax.axis("off")

    cax = fig.add_axes([0.93, 0.12, 0.015, 0.3])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cb)
    finite_norm_values = norm_data[np.isfinite(norm_data)]
    if finite_norm_values.size == 0:
        finite_norm_values = np.array([float(norm.vmin), float(norm.vmax)], dtype=float)
    sm.set_array(finite_norm_values)
    extend_flag = _colorbar_extend_flag(finite_norm_values, norm)
    boundaries = np.linspace(
        float(norm.vmin),
        float(norm.vmax),
        getattr(cmap_cb, "N", 256) + 1,
    )
    cb = fig.colorbar(
        sm,
        cax=cax,
        extend=extend_flag,
        boundaries=boundaries,
    )
    # Match panel background and enforce full opacity; support both old and new Colorbar APIs.
    try:
        cb_facecolor = axis_facecolor
        cb.ax.set_facecolor(cb_facecolor)
        target_alpha = 0.0 if transparent_background else 1.0
        if hasattr(cb.ax, "patch"):
            cb.ax.patch.set_alpha(target_alpha)
        elif hasattr(cb, "patch"):
            # Some matplotlib builds use cb.patch instead of cb.ax.patch
            cb.patch.set_alpha(target_alpha)
    except Exception:
        pass

    # Guard solids: may be None on some backends
    solids = getattr(cb, "solids", None)
    if solids is not None:
        try:
            solids.set_edgecolor("face")
            solids.set_alpha(1.0)
        except Exception:
            pass
    if tick_values:
        _apply_colorbar_ticks(cb, tick_values)
    cb.ax.tick_params(labelsize=8, colors="black")
    for spine in cb.ax.spines.values():
        spine.set_edgecolor("black")

    plt.subplots_adjust(left=0.02, right=0.9, top=0.96, bottom=0.02, wspace=0.08, hspace=0.08)
    save_facecolor = "none" if transparent_background else axis_facecolor
    plt.savefig(
        out_path,
        dpi=dpi,
        edgecolor="none",
        facecolor=save_facecolor,
        transparent=bool(transparent_background),
    )
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
    reference_img: nib.Nifti1Image,
    transparent_background: bool = False,
    fixed_colorbar_width: int | None = None,
) -> None:
    # Prefer deterministic Pillow rendering (no Matplotlib figure/colorbar).
    # Fall back to the Matplotlib renderer only when Pillow is unavailable.
    if Image is not None:
        _render_projection_montage_pillow(
            data,
            ref_info,
            job,
            out_path,
            rows=rows,
            cols=cols,
            dpi=dpi,
            reference_img=reference_img,
            transparent_background=transparent_background,
            fixed_colorbar_width=fixed_colorbar_width,
        )
        return

    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    data = _fill_empty_slices_nearest(data)
    slice_valid_initial = (
        np.isfinite(data).any(axis=(0, 1)) if data.ndim == 3 else np.array([], dtype=bool)
    )

    finite_mask = np.isfinite(data)
    if job.mask_zero:
        finite_mask &= np.abs(data) > EPS
    if not finite_mask.any():
        raise ValueError("Projection map contains no finite values")

    union_xy = np.any(finite_mask, axis=2)
    union_xy_r = np.rot90(union_xy, ref_info["rotate"])

    r0, r1, c0, c1 = _map_bbox_from_ref(ref_info["bbox_fracs"], union_xy_r.shape)
    slice_valid = np.any(finite_mask, axis=(0, 1)) if finite_mask.ndim == 3 else None
    zmin, zmax = _slice_valid_bounds(slice_valid_initial, slice_valid)
    # Projection jobs are parcel displays keyed by *_map_atlas.
    # Force the range to the full slab so the montage includes both extremes.
    try:
        if (job.base or "").endswith("_map_atlas"):
            zmin, zmax = 0, data.shape[2] - 1
    except Exception:
        pass
    z_indices = _map_z_from_ref(
        ref_info["z_fracs"], data.shape[2], zmin=zmin, zmax=zmax
    )
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
    under_rgba = _dim_rgba(tuple(cmap(0.0)), 0.55)
    under_rgba = (under_rgba[0], under_rgba[1], under_rgba[2], 1.0)
    cmap = cmap.with_extremes(bad=(0, 0, 0, 0), under=under_rgba)
    cmap_img = _opaque_for_image(cmap)
    cmap_cb = _opaque_colormap_for_colorbar(cmap)

    axis_facecolor = (0, 0, 0, 0) if transparent_background else "#e0e0e0"
    fig_facecolor = (0, 0, 0, 0) if transparent_background else axis_facecolor
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), facecolor=fig_facecolor)
    fig.patch.set_alpha(0.0 if transparent_background else 1.0)
    axes = axes.ravel()

    nz = data.shape[2]
    if nz == 0:
        raise ValueError("volume has zero z-extent")
    if z_indices.size and z_indices.max() >= nz:
        raise RuntimeError(f"z mapping produced {int(z_indices.max())} with nz={nz}")

    try:
        zoom_xy = np.array(reference_img.header.get_zooms()[:2], dtype=float)
    except Exception:
        zoom_xy = np.array([1.0, 1.0], dtype=float)
    if not np.all(np.isfinite(zoom_xy)) or zoom_xy.size < 2:
        zoom_xy = np.array([1.0, 1.0], dtype=float)
    if ref_info["rotate"] % 2:
        zoom_xy = zoom_xy[::-1]

    for ax, z in zip(axes, z_indices):
        zi = int(np.clip(z, 0, data.shape[2] - 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_facecolor(axis_facecolor)
        for spine in ax.spines.values():
            spine.set_visible(False)

        sl = data[:, :, zi]
        slr = np.rot90(sl, ref_info["rotate"])
        slc = slr[r0:r1, c0:c1]
        if np.isfinite(slc).sum() == 0:
            print(f"[montage] z={zi}: all NaN after mask/crop")

        union_crop = union_xy_r[r0:r1, c0:c1]
        if job.mask_zero:
            # Only mask true zeros / numerical near-zeros. Using a data-driven
            # cutoff (e.g., a low percentile) can incorrectly hide legitimate
            # low values, making the low end of the colormap appear blank.
            mask_slice = np.isfinite(slc) & (np.abs(slc) > EPS)
        else:
            mask_slice = np.isfinite(slc)

        valid_mask = union_crop & mask_slice
        arr = np.ma.array(slc, mask=(~valid_mask))
        alpha_values = valid_mask.astype(np.float32)
        ax.imshow(
            arr,
            cmap=cmap_img,
            norm=norm,
            interpolation="nearest",
            origin="upper",
            alpha=alpha_values,
        )

    for ax in axes[len(z_indices) :]:
        ax.axis("off")

    cax = fig.add_axes([0.93, 0.12, 0.015, 0.3])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cb)
    finite_projection_values = data[finite_mask]
    if finite_projection_values.size == 0:
        finite_projection_values = np.array([float(norm.vmin), float(norm.vmax)], dtype=float)
    sm.set_array(finite_projection_values)
    extend_flag = _colorbar_extend_flag(finite_projection_values, norm)
    boundaries = np.linspace(
        float(norm.vmin),
        float(norm.vmax),
        getattr(cmap_cb, "N", 256) + 1,
    )
    cb = fig.colorbar(
        sm,
        cax=cax,
        extend=extend_flag,
        boundaries=boundaries,
    )
    # Match panel background and enforce full opacity; support both old and new Colorbar APIs.
    try:
        cb_facecolor = axis_facecolor
        cb.ax.set_facecolor(cb_facecolor)
        target_alpha = 0.0 if transparent_background else 1.0
        if hasattr(cb.ax, "patch") and cb.ax.patch is not None:
            cb.ax.patch.set_alpha(target_alpha)
        elif hasattr(cb, "patch"):
            cb.patch.set_alpha(target_alpha)
    except Exception:
        pass
    try:
        cb.solids.set_edgecolor("face")
        cb.solids.set_alpha(1.0)
    except Exception:
        pass
    if tick_values:
        _apply_colorbar_ticks(cb, tick_values)
    cb.ax.tick_params(labelsize=8, colors="black")
    for spine in cb.ax.spines.values():
        spine.set_edgecolor("black")

    plt.subplots_adjust(left=0.02, right=0.9, top=0.96, bottom=0.02, wspace=0.08, hspace=0.08)
    save_facecolor = "none" if transparent_background else axis_facecolor
    plt.savefig(
        out_path,
        dpi=dpi,
        edgecolor="none",
        facecolor=save_facecolor,
        transparent=bool(transparent_background),
    )
    plt.close(fig)


def _render_projection_montage_pillow(
    data: np.ndarray,
    ref_info: Dict[str, np.ndarray],
    job: MapJob,
    out_path: str,
    *,
    rows: int,
    cols: int,
    dpi: int,
    reference_img: nib.Nifti1Image,
    transparent_background: bool = False,
    fixed_colorbar_width: int | None = None,
) -> None:
    """Render projection montage without Matplotlib (Pillow backend).

    Goal: match the Ki projection appearance reliably across platforms/viewers.
    - opaque tile backgrounds by default
    - NaNs masked out to the background
    - colorbar always shows the full [vmin, vmax] ramp with no transparency
    """

    if Image is None:
        raise RuntimeError("Pillow is required for non-matplotlib montage rendering")

    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    data = _fill_empty_slices_nearest(data)
    finite_mask = np.isfinite(data)
    if job.mask_zero:
        finite_mask &= np.abs(data) > EPS
    if not finite_mask.any():
        raise ValueError("Projection map contains no finite values")

    union_xy = np.any(finite_mask, axis=2)
    union_xy_r = np.rot90(union_xy, ref_info["rotate"])
    r0, r1, c0, c1 = _map_bbox_from_ref(ref_info["bbox_fracs"], union_xy_r.shape)

    # Use the same slice mapping as the matplotlib version.
    slice_valid_initial = np.isfinite(data).any(axis=(0, 1))
    slice_valid = np.any(finite_mask, axis=(0, 1))
    zmin, zmax = _slice_valid_bounds(slice_valid_initial, slice_valid)
    try:
        if (job.base or "").endswith("_map_atlas"):
            zmin, zmax = 0, data.shape[2] - 1
    except Exception:
        pass
    z_indices = _map_z_from_ref(ref_info["z_fracs"], data.shape[2], zmin=zmin, zmax=zmax)
    if z_indices.size < rows * cols:
        pad_value = z_indices[-1] if z_indices.size else 0
        z_indices = np.pad(z_indices, (0, rows * cols - z_indices.size), constant_values=pad_value)
    else:
        z_indices = z_indices[: rows * cols]

    # Colormap + normalizer
    cmap = _get_cmap(job.cmap_name)
    norm, tick_values = _build_projection_normalizer(data, job)
    if (getattr(job, "metric", None) or "").lower() == "fa":
        vmax = float(getattr(norm, "vmax", 1.0))
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=False)

    units = _units_for_job(job)
    cb_w = (
        int(fixed_colorbar_width)
        if fixed_colorbar_width is not None and int(fixed_colorbar_width) > 0
        else _pillow_colorbar_required_width(
            norm=norm,
            tick_values=tick_values,
            units=units,
            units_position=PILLOW_COLORBAR_UNITS_POSITION,
            units_gap_px=PILLOW_COLORBAR_UNITS_GAP_PX,
            tick_font_size=PILLOW_COLORBAR_TICK_FONT_SIZE,
            units_font_size=PILLOW_COLORBAR_UNITS_FONT_SIZE,
        )
    )

    # Prepare a simple RGB LUT for speed.
    lut = cmap(np.linspace(0, 1, 256)).astype(np.float32)
    lut[:, 3] = 1.0

    tile_h = 260
    tile_w = 260
    gap = int(PILLOW_TILE_GAP_PX)
    pad = int(max(0, PILLOW_OUTER_MARGIN_PX))
    inner_margin = int(max(0, PILLOW_TILE_INNER_MARGIN_PX))
    bg_rgb = np.array([224, 224, 224], dtype=np.uint8)
    tile_bg_rgba = (0, 0, 0, 0) if transparent_background else (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 255)

    canvas_w = pad * 2 + cols * tile_w + (cols - 1) * gap + cb_w + gap
    canvas_h = pad * 2 + rows * tile_h + (rows - 1) * gap
    # Start fully transparent; for opaque montages, optionally paint only the tile grid background.
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if not transparent_background:
        if PILLOW_TRANSPARENT_COLORBAR_BG:
            _pillow_paint_grid_background(
                canvas,
                x0=pad,
                y0=pad,
                rows=rows,
                cols=cols,
                tile_w=tile_w,
                tile_h=tile_h,
                gap=gap,
                color=tile_bg_rgba,
            )
        else:
            ImageDraw.Draw(canvas).rectangle([0, 0, canvas_w - 1, canvas_h - 1], fill=tile_bg_rgba)

    def to_rgba(arr2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
        # arr2d expected float32
        out = np.zeros((arr2d.shape[0], arr2d.shape[1], 4), dtype=np.uint8)
        out[..., 0:3] = bg_rgb
        out[..., 3] = 255 if not transparent_background else 0

        m = mask2d & np.isfinite(arr2d)
        if job.mask_zero:
            m &= np.abs(arr2d) > EPS
        if not np.any(m):
            return out
        vals = arr2d[m]
        t = (vals - float(norm.vmin)) / max(1e-12, float(norm.vmax) - float(norm.vmin))
        t = np.clip(t, 0.0, 1.0)
        idx = (t * 255.0 + 0.5).astype(np.int32)
        colors = (lut[idx, :3] * 255.0 + 0.5).astype(np.uint8)
        out[m, 0:3] = colors
        out[m, 3] = 255
        return out

    # Render tiles
    for tile_i, z in enumerate(z_indices):
        rr = tile_i // cols
        cc = tile_i % cols
        x0 = pad + cc * (tile_w + gap)
        y0 = pad + rr * (tile_h + gap)

        zi = int(np.clip(int(z), 0, data.shape[2] - 1))
        sl = data[:, :, zi]
        slr = np.rot90(sl, ref_info["rotate"])
        slc = slr[r0:r1, c0:c1]
        union_crop = union_xy_r[r0:r1, c0:c1]
        mask_slice = np.isfinite(slc)
        if job.mask_zero:
            mask_slice &= np.abs(slc) > EPS
        valid = union_crop & mask_slice

        rgba = to_rgba(slc.astype(np.float32), valid)
        tile_img = Image.fromarray(rgba, mode="RGBA")
        if inner_margin > 0:
            inner_w = max(1, tile_w - 2 * inner_margin)
            inner_h = max(1, tile_h - 2 * inner_margin)
            tile_img = tile_img.resize((inner_w, inner_h), resample=Image.NEAREST)
            tile_base = Image.new("RGBA", (tile_w, tile_h), tile_bg_rgba)
            tile_base.alpha_composite(tile_img, (inner_margin, inner_margin))
            tile_img = tile_base
        else:
            tile_img = tile_img.resize((tile_w, tile_h), resample=Image.NEAREST)
        canvas.alpha_composite(tile_img, (x0, y0))

    # Colorbar
    cb_h = int(tile_h * 1.05)
    cb_x0 = pad + cols * tile_w + (cols - 1) * gap + gap
    cb_y0 = max(pad, pad + (canvas_h - 2 * pad - cb_h) // 2)
    _draw_colorbar_pillow(
        canvas,
        norm=norm,
        cmap=cmap,
        tick_values=tick_values,
        transparent_background=transparent_background,
        units=units,
        units_position=PILLOW_COLORBAR_UNITS_POSITION,
        units_gap_px=PILLOW_COLORBAR_UNITS_GAP_PX,
        x0=cb_x0,
        y0=cb_y0,
        w=cb_w,
        h=cb_h,
        tick_font_size=PILLOW_COLORBAR_TICK_FONT_SIZE,
        units_font_size=PILLOW_COLORBAR_UNITS_FONT_SIZE,
    )

    if PILLOW_TRANSPARENT_GUTTERS and not transparent_background:
        _pillow_punch_transparent_gutters(
            canvas,
            grid_x0=pad,
            grid_y0=pad,
            rows=rows,
            cols=cols,
            tile_w=tile_w,
            tile_h=tile_h,
            gap=gap,
            include_grid_to_colorbar_gap=True,
        )

    # (When PILLOW_TRANSPARENT_COLORBAR_BG is enabled, the right side remains transparent because
    # we never painted a background there.)

    # Save
    canvas.save(out_path, dpi=(PILLOW_OUTPUT_DPI, PILLOW_OUTPUT_DPI))


def _render_montage_pillow(
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
    transparent_background: bool = False,
    fixed_colorbar_width: int | None = None,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for non-matplotlib montage rendering")

    is_atlas = job.base.endswith("_map_atlas")
    is_diffusion = _is_diffusion_job(job)

    img = nib.load(map_path)
    target = (reference_img.shape, reference_img.affine)
    if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine):
        img = resample_from_to(img, target, order=0 if is_atlas else 1)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a 3D parametric map")

    if not is_atlas:
        try:
            zoom_src = np.array(img.header.get_zooms()[:3], dtype=float)
            zoom_ref = np.array(reference_img.header.get_zooms()[:3], dtype=float)
            ratio = max(zoom_src[0] / zoom_ref[0], zoom_src[1] / zoom_ref[1])
            if ratio > 1.4:
                data = gaussian_filter(data, sigma=(0.6, 0.6, 0.0), mode="nearest")
        except Exception:
            pass
    else:
        data = _fill_empty_slices_nearest(data)

    if is_diffusion and not is_atlas:
        inpaint_domain = None
        if brain_mask is not None and brain_mask.shape == data.shape:
            inpaint_domain = np.asarray(brain_mask, dtype=bool)
        else:
            finite = np.isfinite(data)
            try:
                from scipy.ndimage import binary_closing

                inpaint_domain = binary_closing(finite, structure=np.ones((3, 3, 3), bool))
            except Exception:
                inpaint_domain = finite
        data = _inpaint_nans_nearest(data, inside_mask=inpaint_domain)

    slice_valid_initial = np.isfinite(data).any(axis=(0, 1))

    # Display-domain mask
    if brain_mask is not None and not is_atlas:
        mask_data = np.asarray(brain_mask, dtype=bool)
        if mask_data.shape != data.shape:
            if segmentation_img is None:
                raise ValueError("Brain mask shape does not match parametric map")
            target2 = (data.shape, img.affine)
            seg_img = segmentation_img
            if (
                segmentation_img.shape != data.shape
                or not np.allclose(segmentation_img.affine, img.affine)
            ):
                seg_img = resample_from_to(segmentation_img, target2, order=0)
            seg_data = np.asarray(seg_img.get_fdata(), dtype=np.float32)
            mask_data = np.isfinite(seg_data) & (seg_data > 0.5)
        brain_mask = mask_data
        valmask3d = np.isfinite(data) & mask_data
    else:
        if is_atlas:
            atlas_support = None
            if segmentation_img is not None:
                target2 = (data.shape, img.affine)
                seg_img = segmentation_img
                if (
                    segmentation_img.shape != data.shape
                    or not np.allclose(segmentation_img.affine, img.affine)
                ):
                    seg_img = resample_from_to(segmentation_img, target2, order=0)
                seg = np.asarray(seg_img.get_fdata(), dtype=np.float32)
                atlas_support = np.isfinite(seg) & (seg > 0.5)
            if atlas_support is None:
                finite = np.isfinite(data)
                try:
                    from scipy.ndimage import binary_closing, binary_fill_holes

                    atlas_support = binary_fill_holes(
                        binary_closing(finite, structure=np.ones((3, 3, 3), bool))
                    )
                except Exception:
                    atlas_support = finite
            valmask3d = atlas_support.astype(bool)
        else:
            valmask3d = np.isfinite(data)

    # Frame crop
    union_xy = (
        np.any(brain_mask, axis=2) if (brain_mask is not None and not is_atlas) else np.any(valmask3d, axis=2)
    )
    union_xy_r = np.rot90(union_xy, ref_info["rotate"])
    r0, r1, c0, c1 = _map_bbox_from_ref(ref_info["bbox_fracs"], union_xy_r.shape)

    slice_valid = np.any(valmask3d, axis=(0, 1))
    if is_atlas:
        zmin, zmax = 0, data.shape[2] - 1
    else:
        zmin, zmax = _slice_valid_bounds(slice_valid_initial, slice_valid)
    z_indices = _map_z_from_ref(ref_info["z_fracs"], data.shape[2], zmin=zmin, zmax=zmax)
    if z_indices.size < rows * cols:
        pad_value = z_indices[-1] if z_indices.size else 0
        z_indices = np.pad(z_indices, (0, rows * cols - z_indices.size), constant_values=pad_value)
    else:
        z_indices = z_indices[: rows * cols]

    cmap = _get_cmap(job.cmap_name)
    base_under = list(cmap(0.0))
    base_under[-1] = 1.0
    cmap = cmap.with_extremes(bad=(0, 0, 0, 0), under=tuple(base_under))

    # Normalizer uses the same logic as the Matplotlib path.
    head_support_3d = None
    if overlay is not None and isinstance(overlay.get("mask_head"), np.ndarray):
        head_support_3d = overlay["mask_head"].astype(bool)
        if head_support_3d.shape != data.shape:
            head_img = nib.Nifti1Image(head_support_3d.astype(np.float32), reference_img.affine)
            head_img = resample_from_to(head_img, (data.shape, img.affine), order=0)
            head_support_3d = np.asarray(head_img.get_fdata(), dtype=np.float32) > 0.5
    norm_data = data if brain_mask is None else np.where(brain_mask, data, np.nan)
    if is_atlas and head_support_3d is not None:
        norm_data = np.where(head_support_3d, norm_data, np.nan)
    focus_data = None
    if (getattr(job, "metric", "") or "").lower() == "ki":
        core_mask = None
        if brain_mask is not None and brain_mask.shape == data.shape:
            core_mask = _mask_erode_mm(brain_mask, img, mm=2.0)
        elif head_support_3d is not None and head_support_3d.shape == data.shape:
            core_mask = _mask_erode_mm(head_support_3d, img, mm=2.0)
        if core_mask is not None and core_mask.any():
            focus_data = np.where(core_mask, data, np.nan)
        else:
            focus_data = norm_data

    norm, tick_values = _build_normalizer(
        norm_data,
        job,
        mask_zero_override=False if brain_mask is not None else None,
        focus_data=focus_data,
    )
    if (getattr(job, "metric", None) or "").lower() == "fa":
        vmax = float(getattr(norm, "vmax", 1.0))
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=False)

    units = _units_for_job(job)
    cb_w = (
        int(fixed_colorbar_width)
        if fixed_colorbar_width is not None and int(fixed_colorbar_width) > 0
        else _pillow_colorbar_required_width(
            norm=norm,
            tick_values=tick_values,
            units=units,
            units_position=PILLOW_COLORBAR_UNITS_POSITION,
            units_gap_px=PILLOW_COLORBAR_UNITS_GAP_PX,
            tick_font_size=PILLOW_COLORBAR_TICK_FONT_SIZE,
            units_font_size=PILLOW_COLORBAR_UNITS_FONT_SIZE,
        )
    )

    lut = cmap(np.linspace(0, 1, 256)).astype(np.float32)
    lut[:, 3] = 1.0

    tile_h = 260
    tile_w = 260
    gap = int(PILLOW_TILE_GAP_PX)
    pad = int(max(0, PILLOW_OUTER_MARGIN_PX))
    inner_margin = int(max(0, PILLOW_TILE_INNER_MARGIN_PX))
    bg_rgb = np.array([224, 224, 224], dtype=np.uint8)
    tile_bg_rgba = (0, 0, 0, 0) if transparent_background else (224, 224, 224, 255)

    canvas_w = pad * 2 + cols * tile_w + (cols - 1) * gap + cb_w + gap
    canvas_h = pad * 2 + rows * tile_h + (rows - 1) * gap
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if not transparent_background:
        if PILLOW_TRANSPARENT_COLORBAR_BG:
            _pillow_paint_grid_background(
                canvas,
                x0=pad,
                y0=pad,
                rows=rows,
                cols=cols,
                tile_w=tile_w,
                tile_h=tile_h,
                gap=gap,
                color=tile_bg_rgba,
            )
        else:
            ImageDraw.Draw(canvas).rectangle([0, 0, canvas_w - 1, canvas_h - 1], fill=tile_bg_rgba)

    # Overlay prep (grayscale underlay)
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
    overlay_slice_valid_initial = None
    overlay_slice_valid_masked = None
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
            target2 = (data.shape[:3], img.affine)
            if ovl_img.shape[:3] != data.shape[:3] or not np.allclose(ovl_img.affine, img.affine):
                ovl_img = resample_from_to(ovl_img, target2, order=1)
            overlay_data = np.asarray(ovl_img.get_fdata(), dtype=np.float32)
            if overlay_data.ndim == 3:
                overlay_slice_valid_initial = np.isfinite(overlay_data).any(axis=(0, 1))
            if overlay_alpha_map is not None:
                alpha_img = nib.Nifti1Image(
                    np.asarray(overlay_alpha_map, dtype=np.float32),
                    overlay_affine if overlay_affine is not None else ovl_img.affine,
                )
                if alpha_img.shape[:3] != data.shape[:3] or not np.allclose(alpha_img.affine, img.affine):
                    alpha_img = resample_from_to(alpha_img, target2, order=1)
                overlay_alpha_data = np.asarray(alpha_img.get_fdata(), dtype=np.float32)
            if overlay_mask_volume is not None:
                mask_img = None
                if isinstance(overlay_mask_volume, str):
                    mask_img = nib.load(overlay_mask_volume)
                elif hasattr(overlay_mask_volume, "shape") and hasattr(overlay_mask_volume, "affine"):
                    mask_img = overlay_mask_volume
                elif isinstance(overlay_mask_volume, np.ndarray):
                    mask_affine = overlay_affine if overlay_affine is not None else ovl_img.affine
                    mask_img = nib.Nifti1Image(overlay_mask_volume.astype(np.float32), mask_affine)
                if mask_img is not None:
                    if mask_img.shape[:3] != data.shape[:3] or not np.allclose(mask_img.affine, img.affine):
                        mask_img = resample_from_to(mask_img, target2, order=0)
                    overlay_mask_data = np.asarray(mask_img.get_fdata(), dtype=np.float32) > 0.5
            if overlay_mask_data is None and overlay_data is not None:
                overlay_mask_data = np.isfinite(overlay_data)
            if overlay_mask_data is not None:
                overlay_mask_data = np.asarray(overlay_mask_data, dtype=bool)
                if overlay_mask_data.ndim == 3 and overlay_mask_data.shape[2]:
                    footprint2d = np.ones((3, 3), dtype=bool)
                    for idx in range(overlay_mask_data.shape[2]):
                        slice_mask = overlay_mask_data[:, :, idx]
                        try:
                            slice_mask = binary_fill_holes(slice_mask)
                        except Exception:
                            pass
                        try:
                            slice_mask = binary_closing(slice_mask, footprint=footprint2d)
                        except Exception:
                            pass
                        overlay_mask_data[:, :, idx] = slice_mask
                overlay_slice_valid_masked = overlay_mask_data.any(axis=(0, 1))
            if overlay_data is not None and overlay_slice_valid_initial is not None and overlay_slice_valid_masked is not None:
                ovl_zmin, ovl_zmax = _slice_valid_bounds(overlay_slice_valid_initial, overlay_slice_valid_masked)
                overlay_z_indices = _map_z_from_ref(ref_info["z_fracs"], overlay_data.shape[2], zmin=ovl_zmin, zmax=ovl_zmax)
                if overlay_z_indices.size < rows * cols:
                    pad_value = overlay_z_indices[-1] if overlay_z_indices.size else 0
                    overlay_z_indices = np.pad(overlay_z_indices, (0, rows * cols - overlay_z_indices.size), constant_values=pad_value)
                else:
                    overlay_z_indices = overlay_z_indices[: rows * cols]

    def normalize_overlay(arr2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
        m = mask2d & np.isfinite(arr2d)
        if not np.any(m):
            return np.zeros(arr2d.shape, dtype=np.uint8)
        vmin = overlay_vmin
        vmax = overlay_vmax
        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
            vals = arr2d[m]
            lo, hi = np.percentile(vals, [2.0, 98.0]).astype(np.float32)
            if not np.isfinite(lo) or not np.isfinite(hi) or float(hi) <= float(lo):
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            vmin, vmax = float(lo), float(hi)
        t = (arr2d - float(vmin)) / max(1e-12, float(vmax) - float(vmin))
        t = np.clip(t, 0.0, 1.0)
        return (t * 255.0 + 0.5).astype(np.uint8)

    def map_to_rgb(arr2d: np.ndarray, valid2d: np.ndarray) -> np.ndarray:
        out = np.zeros((arr2d.shape[0], arr2d.shape[1], 4), dtype=np.uint8)
        out[..., 0:3] = bg_rgb
        out[..., 3] = 255 if not transparent_background else 0
        m = valid2d & np.isfinite(arr2d)
        if job.mask_zero:
            m &= np.abs(arr2d) > EPS
        if not np.any(m):
            return out
        vals = arr2d[m]
        t = (vals - float(norm.vmin)) / max(1e-12, float(norm.vmax) - float(norm.vmin))
        t = np.clip(t, 0.0, 1.0)
        idx = (t * 255.0 + 0.5).astype(np.int32)
        colors = (lut[idx, :3] * 255.0 + 0.5).astype(np.uint8)
        out[m, 0:3] = colors
        out[m, 3] = 255
        return out

    # Render tiles
    for tile_i, z in enumerate(z_indices):
        rr = tile_i // cols
        cc = tile_i % cols
        x0 = pad + cc * (tile_w + gap)
        y0 = pad + rr * (tile_h + gap)
        zi = int(np.clip(int(z), 0, data.shape[2] - 1))

        sl = data[:, :, zi]
        slr = np.rot90(sl, ref_info["rotate"])
        slc = slr[r0:r1, c0:c1]

        mask_slice = valmask3d[:, :, zi]
        mask_slice_rot = np.rot90(mask_slice, ref_info["rotate"])
        mask_slice_crop = mask_slice_rot[r0:r1, c0:c1]
        union_crop = union_xy_r[r0:r1, c0:c1]
        valid = union_crop & mask_slice_crop

        render_shape = slc.shape
        tile_rgba = None

        if overlay_data is not None:
            overlay_zi = zi
            if overlay_z_indices is not None and tile_i < overlay_z_indices.size:
                overlay_zi = int(np.clip(int(overlay_z_indices[tile_i]), 0, overlay_data.shape[2] - 1))
            overlay_slice = overlay_data[:, :, overlay_zi]
            overlay_rot = np.rot90(overlay_slice, ref_info["rotate"])
            or0, or1, oc0, oc1 = _map_bbox_from_ref(ref_info["bbox_fracs"], overlay_rot.shape)
            overlay_crop = overlay_rot[or0:or1, oc0:oc1]
            render_shape = overlay_crop.shape

            overlay_union = union_xy_r
            if overlay_mask_data is not None:
                ms = overlay_mask_data[:, :, overlay_zi]
                msr = np.rot90(ms, ref_info["rotate"])
                overlay_union = msr.astype(bool)
            overlay_union_crop = overlay_union[or0:or1, oc0:oc1]
            overlay_gray = normalize_overlay(overlay_crop, overlay_union_crop)

            base = np.zeros((render_shape[0], render_shape[1], 4), dtype=np.uint8)
            base[..., 0:3] = bg_rgb
            base[..., 3] = 255 if not transparent_background else 0
            m = overlay_union_crop & np.isfinite(overlay_crop)
            base[m, 0] = overlay_gray[m]
            base[m, 1] = overlay_gray[m]
            base[m, 2] = overlay_gray[m]
            base[m, 3] = 255
            tile_rgba = base

            # Resize parametric layer to overlay crop shape if needed.
            if slc.shape != render_shape:
                slc = resize(
                    slc.astype(np.float32),
                    render_shape,
                    order=0 if is_atlas else 1,
                    preserve_range=True,
                    anti_aliasing=False if is_atlas else True,
                ).astype(np.float32)
                valid = _resize_mask(valid, render_shape)

        layer = map_to_rgb(slc.astype(np.float32), valid)
        if tile_rgba is None:
            tile_rgba = layer
        else:
            m = layer[..., 3] > 0
            tile_rgba[m] = layer[m]

        resample = Image.NEAREST
        tile_img = Image.fromarray(tile_rgba, mode="RGBA")
        if inner_margin > 0:
            inner_w = max(1, tile_w - 2 * inner_margin)
            inner_h = max(1, tile_h - 2 * inner_margin)
            tile_img = tile_img.resize((inner_w, inner_h), resample=resample)
            tile_base = Image.new("RGBA", (tile_w, tile_h), tile_bg_rgba)
            tile_base.alpha_composite(tile_img, (inner_margin, inner_margin))
            tile_img = tile_base
        else:
            tile_img = tile_img.resize((tile_w, tile_h), resample=resample)
        canvas.alpha_composite(tile_img, (x0, y0))

    # Colorbar
    cb_h = int(tile_h * 1.05)
    cb_x0 = pad + cols * tile_w + (cols - 1) * gap + gap
    cb_y0 = max(pad, pad + (canvas_h - 2 * pad - cb_h) // 2)
    _draw_colorbar_pillow(
        canvas,
        norm=norm,
        cmap=cmap,
        tick_values=tick_values,
        transparent_background=transparent_background,
        units=units,
        units_position=PILLOW_COLORBAR_UNITS_POSITION,
        units_gap_px=PILLOW_COLORBAR_UNITS_GAP_PX,
        x0=cb_x0,
        y0=cb_y0,
        w=cb_w,
        h=cb_h,
        tick_font_size=PILLOW_COLORBAR_TICK_FONT_SIZE,
        units_font_size=PILLOW_COLORBAR_UNITS_FONT_SIZE,
    )

    if PILLOW_TRANSPARENT_GUTTERS and not transparent_background:
        _pillow_punch_transparent_gutters(
            canvas,
            grid_x0=pad,
            grid_y0=pad,
            rows=rows,
            cols=cols,
            tile_w=tile_w,
            tile_h=tile_h,
            gap=gap,
            include_grid_to_colorbar_gap=True,
        )

    canvas.save(out_path, dpi=(PILLOW_OUTPUT_DPI, PILLOW_OUTPUT_DPI))


def _build_normalizer(
    data: np.ndarray,
    job: MapJob,
    *,
    mask_zero_override: bool | None = None,
    focus_data: np.ndarray | None = None,
) -> tuple[mpl.colors.Normalize, list[float]]:
    vmin = float(job.vmin) if job.vmin is not None else np.nan
    vmax = float(job.vmax) if job.vmax is not None else np.nan
    vmin_given = np.isfinite(vmin)
    vmax_given = np.isfinite(vmax)
    metric = (getattr(job, "metric", "") or "").lower()

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        mask = np.isfinite(data)
        mask_zero = job.mask_zero if mask_zero_override is None else mask_zero_override
        if mask_zero:
            mask &= data > EPS
        finite_vals = data[mask]
        if finite_vals.size:
            lo, hi = _robust_bounds(finite_vals)
        else:
            lo, hi = 0.0, 1.0
        # Respect explicit endpoints when only one side was provided.
        if vmin_given and not vmax_given:
            vmin = float(vmin)
            vmax = float(hi)
        elif vmax_given and not vmin_given:
            vmin = float(lo)
            vmax = float(vmax)
        else:
            vmin, vmax = float(lo), float(hi)

    if vmax <= vmin:
        padding = abs(vmin) if vmin != 0 else 1.0
        vmax = vmin + padding

    span = max(1e-6, float(vmax - vmin))
    if metric == "ki":
        roi_source = focus_data if focus_data is not None else data
        central_vmax = _central_roi_percentile(
            roi_source,
            fraction=0.65,
            percentile=97.5,
        )
        if central_vmax is not None and np.isfinite(central_vmax):
            guard = float(vmin) + 0.1 * span
            candidate = max(guard, float(central_vmax))
            vmax = max(candidate, float(vmin) + 0.07 * span)

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


def _central_roi_percentile(
    data: np.ndarray,
    *,
    fraction: float = 0.45,
    percentile: float = 98.5,
) -> float | None:
    """Return a high-percentile value from the central in-plane ROI."""

    if data is None:
        return None

    arr = np.asarray(data)
    if arr.ndim < 2 or arr.size == 0:
        return None

    rows = arr.shape[0]
    cols = arr.shape[1] if arr.ndim > 1 else 0
    if rows == 0 or cols == 0:
        return None

    frac = float(np.clip(fraction, 0.1, 1.0))
    roi_rows = max(1, int(round(rows * frac)))
    roi_cols = max(1, int(round(cols * frac)))
    r0 = max(0, (rows - roi_rows) // 2)
    c0 = max(0, (cols - roi_cols) // 2)
    r1 = min(rows, r0 + roi_rows)
    c1 = min(cols, c0 + roi_cols)

    roi = arr[r0:r1, c0:c1, ...]
    finite = roi[np.isfinite(roi)]
    if finite.size == 0:
        return None

    try:
        return float(np.nanpercentile(finite, percentile))
    except Exception:
        return None


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


def _colorbar_extend_flag(values: np.ndarray, norm: mpl.colors.Normalize) -> str:
    """
    Decide whether the colourbar should show 'min', 'max', 'both' or 'neither'
    extensions by comparing the finite data to the normaliser limits.
    """
    try:
        vmin = float(norm.vmin)
        vmax = float(norm.vmax)
    except Exception:
        return "neither"
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "neither"
    has_under = np.nanmin(finite) < vmin
    has_over  = np.nanmax(finite) > vmax
    if has_under and has_over:
        return "both"
    if has_under:
        return "min"
    if has_over:
        return "max"
    return "neither"


def _format_tick_labels(values: Sequence[float]) -> list[str]:
    ticks = [float(v) for v in values]
    if not ticks:
        return []

    def _format_sci(val: float, digits: int) -> str:
        # digits = digits after decimal in scientific notation.
        s = format(val, f".{max(0, digits)}e")
        if "e" not in s:
            return s
        mant, exp = s.split("e", 1)
        mant = mant.rstrip("0").rstrip(".")
        try:
            exp_i = int(exp)
        except Exception:
            exp_i = 0
        return f"{mant}e{exp_i}"

    def _format_fixed(val: float, decimals: int) -> str:
        s = format(val, f".{max(0, decimals)}f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    def _fmt(val: float, decimals: int) -> str:
        if not np.isfinite(val):
            return ""
        aval = abs(val)
        # No scientific notation for values below 1000 (e.g., hundreds).
        # Use scientific notation for >= 1000 and for very small non-zero values.
        if aval >= 1000.0 or (aval > 0.0 and aval < 1e-3):
            return _format_sci(val, digits=min(6, max(2, decimals + 1)))
        return _format_fixed(val, decimals)

    for decimals in range(0, 8):
        labels = [_fmt(val, decimals) for val in ticks]
        if len(set(labels)) != len(labels):
            continue
        # Avoid collapsing small ranges into all-zero labels.
        if any((abs(v) > 0 and lbl in {"0", "0.0", "0.00", "-0", "-0.0", "-0.00"}) for v, lbl in zip(ticks, labels)):
            continue
        return labels

    return [_fmt(val, 7) for val in ticks]


def _apply_colorbar_ticks(cb: mpl.colorbar.Colorbar, tick_values: Sequence[float]) -> None:
    ticks = [float(v) for v in tick_values if np.isfinite(v)]
    if not ticks:
        cb.ax.set_yticks([])
        return

    fig = cb.ax.figure

    def _nearest_gap(idx: int) -> float:
        left = abs(ticks[idx] - ticks[idx - 1]) if idx > 0 else np.inf
        right = (
            abs(ticks[idx + 1] - ticks[idx])
            if idx + 1 < len(ticks)
            else np.inf
        )
        return min(left, right)

    while True:
        cb.set_ticks(ticks)
        cb.set_ticklabels(_format_tick_labels(ticks))
        try:
            fig.canvas.draw()
        except Exception:
            break

        texts = cb.ax.get_yticklabels()
        if len(texts) != len(ticks):
            break

        renderer = fig.canvas.get_renderer()
        boxes = [txt.get_window_extent(renderer) for txt in texts]
        drop_idx: int | None = None

        for i in range(1, len(boxes)):
            if not boxes[i].overlaps(boxes[i - 1]):
                continue

            candidates: list[int] = []
            if 0 < i < len(boxes) - 1:
                candidates.append(i)
            if 0 < i - 1 < len(boxes) - 1:
                candidates.append(i - 1)

            if not candidates:
                drop_idx = None
                break

            if len(candidates) == 1:
                drop_idx = candidates[0]
            else:
                drop_idx = min(candidates, key=_nearest_gap)
            break

        if drop_idx is None or drop_idx <= 0 or drop_idx >= len(ticks) - 1:
            break

        ticks.pop(drop_idx)
        if len(ticks) <= 2:
            break

    cb.set_ticks(ticks)
    cb.set_ticklabels(_format_tick_labels(ticks))
