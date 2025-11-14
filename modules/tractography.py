"""Utilities to compute and visualise diffusion tractography streamlines.

Heavy debug mode added. Set P_BRAIN_DEBUG_TRACKS=1 to enable extra prints.
Also writes a JSON snapshot alongside outputs for post-mortem inspection.
"""

from __future__ import annotations

import platform
from functools import wraps

import inspect
import multiprocessing as mp
import os
import queue
import traceback
import json
import datetime
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np

try:  # ``fury`` is optional – fall back to matplotlib when unavailable.
    from dipy.viz import actor as dipy_actor
    from dipy.viz import colormap as dipy_colormap
    from dipy.viz import window as dipy_window

    _FURY_AVAILABLE = True
except Exception:  # pragma: no cover - fury is optional
    _FURY_AVAILABLE = False

import nibabel as nib
from nibabel import streamlines as nib_streamlines
try:  # ``StatefulTractogram`` helps preserve coordinate frames when saving.
    from nibabel.streamlines.stateful_tractogram import StatefulTractogram, Space
except ImportError:  # pragma: no cover - older nibabel without stateful helper
    StatefulTractogram = None  # type: ignore[assignment]
    Space = None  # type: ignore[assignment]

try:  # ``resample_from_to`` is only needed for anatomical overlays.
    from nibabel.processing import resample_from_to
except ImportError:  # pragma: no cover - helper ships with supported nibabel
    resample_from_to = None

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.dti import TensorModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import seeds_from_mask

from modules.opt08_fa import find_dwi_files

_DBG = os.environ.get("P_BRAIN_DEBUG_TRACKS", "1").strip().lower() in {"1","true","yes","on"}

def _dbg_print(msg: str) -> None:
    print(msg, flush=True)


def _dbg(msg: str) -> None:
    if _DBG:
        _dbg_print(f"[tracks][dbg] {msg}")


def _safe_summary_array(a, name: str, max_elems: int = 6) -> dict:
    try:
        shp = tuple(int(x) for x in np.shape(a))
    except Exception:
        shp = "<unknown>"
    preview = None
    try:
        flat = np.ravel(a)
        if flat.size > 0:
            preview = [float(flat[i]) for i in range(min(flat.size, max_elems))]
    except Exception:
        preview = "<unavailable>"
    return {"name": name, "shape": shp, "preview": preview}


def _dump_debug_json(path: str, payload: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=lambda o: str(o))
    except Exception as _:
        pass


def _print_affine(label: str, A: np.ndarray) -> None:
    if not _DBG:
        return
    _dbg_print(f"[tracks][dbg] {label} shape={getattr(A,'shape',None)}")
    _dbg_print(f"[tracks][dbg] {label}=\n{np.array(A)}")

def _canonical_affine(affine: np.ndarray) -> np.ndarray:
    """Return a 4x4 voxel-to-world affine derived from ``affine``."""

    arr = np.asarray(affine, dtype=np.float64)
    if arr.shape == (4, 4):
        out = np.array(arr, copy=True)
        _print_affine("canonical affine 4x4 ok", out)
        return out

    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 4:
        raise ValueError("Expected affine with at least 4x4 elements")

    canonical = np.eye(4, dtype=arr.dtype)
    canonical[:3, :3] = arr[:3, :3]
    canonical[:3, 3] = arr[:3, 3]
    _print_affine("canonical affine coerced", canonical)
    return canonical

@dataclass
class TractographyOutputs:
    """Container describing generated tractography artefacts."""

    tract_path: str
    render_path: Optional[str] = None
    montage_path: Optional[str] = None


# ---------------------------------------------------------------------------
# DWI normalisation helpers
# ---------------------------------------------------------------------------


def _as_4d_dwi(data: np.ndarray, bvals: np.ndarray) -> np.ndarray:
    """
    Return a 4-D array shaped (X, Y, Z, N) from potentially 5-D inputs.
    We pick the gradient dimension by matching the axis whose length equals
    len(bvals). Any remaining singleton axes beyond Z are squeezed.
    """

    if _DBG:
        _dbg_print(
            f"[tracks][dbg] entering _as_4d_dwi: data.ndim={np.ndim(data)} "
            f"raw_shape={getattr(data,'shape',None)} bvals.size={int(bvals.size)}"
        )

    arr = np.asarray(data)
    if arr.ndim == 4:
        _dbg(f"DWI already 4-D: shape={arr.shape}")
        return arr
    if arr.ndim < 4:
        raise ValueError(f"DWI must be >=4-D, got {arr.ndim}D")

    grad_len = int(bvals.size)
    grad_axis = None
    for ax in range(3, arr.ndim):
        if arr.shape[ax] == grad_len:
            grad_axis = ax
            break
    if grad_axis is None:
        grad_axis = arr.ndim - 1

    arr = np.moveaxis(arr, grad_axis, -1)
    _dbg(f"DWI grad axis chosen: axis={grad_axis}, shape after move={arr.shape}")

    if arr.ndim > 4:
        to_squeeze: list[int] = []
        for ax in range(3, arr.ndim - 1):
            if arr.shape[ax] == 1:
                to_squeeze.append(ax)
        if to_squeeze:
            arr = np.squeeze(arr, axis=tuple(to_squeeze))
            _dbg(f"Squeezed singleton axes {to_squeeze}; new shape={arr.shape}")

    if arr.ndim != 4:
        raise ValueError(
            f"Failed to coerce DWI to 4-D. Shape after coercion: {arr.shape}"
        )
    _dbg(f"DWI final 4-D shape={arr.shape}")
    return arr


def _shape_str(a: np.ndarray) -> str:
    try:
        return "x".join(map(str, a.shape))
    except Exception:
        return "<unknown>"


# ---------------------------------------------------------------------------
# Streamline sanitisation helpers
# ---------------------------------------------------------------------------


def _to_xyz(points: np.ndarray) -> np.ndarray:
    """
    Return Nx3 float32 coordinates from points with possible extra columns.
    If a 4th column exists, treat it as homogeneous w and dehomogenise.
    """

    if _DBG:
        _dbg_print(
            f"[tracks][dbg] _to_xyz: incoming shape={getattr(points,'shape',None)} "
            f"dtype={getattr(points,'dtype',None)}"
        )

    P = np.asarray(points)
    if P.ndim != 2:
        _dbg(f"streamline points ndim={P.ndim}, pass through")
        return P
    if P.shape[1] == 3:
        _dbg("streamline already Nx3")
        return P.astype(np.float32)
    if P.shape[1] >= 4:
        w = P[:, 3]
        w = np.where((w == 0) | (~np.isfinite(w)), 1.0, w)
        out = (P[:, :3] / w[:, None]).astype(np.float32)
        _dbg("streamline had >=4 cols, dehomogenised to Nx3")
        return out
    # fewer than 3 columns, pass through
    if _DBG:
        _dbg_print(f"[tracks][dbg] _to_xyz: pass-through shape={P.shape}")
    return P.astype(np.float32)


def _coerce_streamlines_xyz(sls: Streamlines) -> Streamlines:
    """Ensure every streamline is Nx3 and has at least two points."""

    out = Streamlines()
    idx = 0
    for sl in sls:
        xyz = _to_xyz(sl)
        if xyz.shape[0] >= 2 and xyz.shape[1] == 3:
            out.append(xyz)
            if _DBG and idx < 3:
                _dbg(f"kept streamline[{idx}] shape={xyz.shape}, "
                     f"head={np.array2string(xyz[:2], precision=3)}")
                idx += 1
        else:
            _dbg(f"dropped streamline shape={xyz.shape}")
    return out


def _streamline_fraction_inside(
    streamlines: Streamlines,
    affine: Optional[np.ndarray],
    shape: Optional[Sequence[int]],
) -> float:
    """Return the fraction of points lying within the provided ``shape``."""

    if affine is None or shape is None:
        return 0.0

    try:
        inv_affine = np.linalg.inv(_canonical_affine(affine))
    except Exception:
        return 0.0

    if len(shape) > 3:
        shape = shape[:3]
    bounds = np.array(shape, dtype=np.float64) - 0.5

    inside = 0
    total = 0
    for sl in streamlines:
        if sl.shape[0] == 0:
            continue
        pts = nib.affines.apply_affine(inv_affine, sl)
        total += pts.shape[0]
        mask = (
            (pts[:, 0] >= -0.5)
            & (pts[:, 0] <= bounds[0])
            & (pts[:, 1] >= -0.5)
            & (pts[:, 1] <= bounds[1])
            & (pts[:, 2] >= -0.5)
            & (pts[:, 2] <= bounds[2])
        )
        inside += int(np.count_nonzero(mask))

    if total == 0:
        return 0.0

    return inside / float(total)


def _load_streamlines_world(
    path: str,
    *,
    expected_affine: Optional[np.ndarray] = None,
    expected_shape: Optional[Sequence[int]] = None,
) -> Optional[Streamlines]:
    """Load streamlines ensuring they are expressed in world space.

    Older releases wrote world-space streamlines while also recording the
    diffusion affine in the tractogram header. ``tractogram.to_world`` would
    therefore apply the affine twice, pushing fibres far outside the field of
    view. To remain backward compatible we evaluate both the raw stored
    coordinates and the transformed-to-world version, selecting whichever
    overlaps the diffusion volume best.
    """

    try:
        tractogram_file = nib_streamlines.load(path)
    except Exception:
        return None

    tractogram = tractogram_file.tractogram

    raw_streamlines = _coerce_streamlines_xyz(Streamlines(tractogram.streamlines))
    transformed_streamlines: Optional[Streamlines] = None

    try:
        transformed = tractogram.to_world(lazy=False)
    except Exception:
        if _DBG:
            _dbg_print(
                "[tracks][dbg] tractogram.to_world() failed; proceeding without transform"
            )
    else:
        transformed_streamlines = _coerce_streamlines_xyz(
            Streamlines(transformed.streamlines)
        )

    if expected_affine is not None and expected_shape is not None:
        candidates: list[tuple[str, Streamlines]] = [("raw", raw_streamlines)]
        if transformed_streamlines is not None:
            candidates.append(("world", transformed_streamlines))

        best_label = "raw"
        best_streamlines = raw_streamlines
        best_fraction = _streamline_fraction_inside(
            raw_streamlines, expected_affine, expected_shape
        )

        for label, candidate in candidates[1:]:
            fraction = _streamline_fraction_inside(
                candidate, expected_affine, expected_shape
            )
            if fraction > best_fraction + 1e-6:
                best_fraction = fraction
                best_streamlines = candidate
                best_label = label

        if _DBG:
            _dbg(
                f"streamline load candidate={best_label} inside={best_fraction:.3f}"
            )
        return best_streamlines

    if transformed_streamlines is not None:
        return transformed_streamlines

    return raw_streamlines


def _load_background_volume(
    nifti_path: str,
    reference_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Return a background volume resampled into ``reference_img`` space."""

    img = nib.load(nifti_path)
    if img.shape[:3] == reference_img.shape[:3]:
        return img

    if resample_from_to is None:
        raise RuntimeError(
            "nibabel.processing.resample_from_to is required for anatomical overlays"
        )

    spatial_shape = reference_img.shape
    if len(spatial_shape) > 3:
        spatial_shape = spatial_shape[:3]
    target = (tuple(int(dim) for dim in spatial_shape), _canonical_affine(reference_img.affine))
    resampled = resample_from_to(img, target, order=1)
    return resampled


def _seed_points_from_mask(
    mask: np.ndarray,
    voxel_to_world: np.ndarray,
    *,
    density: Union[int, Sequence[int]] = 1,
) -> np.ndarray:
    """Return evenly distributed seed points in voxel space mapped to world coordinates."""

    affine = _canonical_affine(voxel_to_world)
    return seeds_from_mask(mask, affine, density=density)


def _ensure_image_directory(
    image_directory: Optional[str], analysis_directory: str
) -> Optional[str]:
    """Return a usable image directory, falling back to ``analysis_directory``."""

    if image_directory:
        os.makedirs(image_directory, exist_ok=True)
        return image_directory

    parent = os.path.dirname(os.path.abspath(analysis_directory))
    fallback = os.path.join(parent, "Images")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _render_with_fury(
    streamlines: Streamlines,
    output_path: str,
    *,
    background_color: tuple[float, float, float] = (0.02, 0.02, 0.05),
) -> None:
    """Render colourful streamlines using ``fury`` for rich shading and lighting."""

    scene = dipy_window.Scene()
    if _DBG:
        _dbg_print(
            f"[tracks][dbg] FURY scene created. streamlines_count="
            f"{sum(1 for _ in streamlines)}"
        )
        streamlines = Streamlines(streamlines)
    scene.background(background_color)

    try:
        colours = dipy_colormap.line_colors(streamlines)
        stream_actor = dipy_actor.line(
            streamlines,
            colors=colours,
            linewidth=1.0,
        )
        scene.add(stream_actor)
    except Exception:
        import traceback as _tb
        raise RuntimeError("FURY line actor build failed:\n" + _tb.format_exc())

    # Gently rotate the scene so fibres are rendered with depth cues.
    scene.reset_camera()
    scene.pitch(-15)
    scene.yaw(20)
    scene.zoom(1.2)

    snapshot_kwargs = {"fname": output_path, "size": (1600, 1600)}
    try:
        signature = inspect.signature(dipy_window.snapshot)
    except (TypeError, ValueError):  # pragma: no cover - rare introspection issues
        signature = None
    if signature is not None and "offscreen" in signature.parameters:
        snapshot_kwargs["offscreen"] = True

    try:
        dipy_window.snapshot(scene, **snapshot_kwargs)
    except Exception:
        import traceback as _tb
        raise RuntimeError("FURY snapshot failed:\n" + _tb.format_exc())
    finally:
        clear = getattr(scene, "clear", None)
        if callable(clear):  # pragma: no branch - safety guard for older fury
            clear()


def _fury_render_worker(
    tract_path: str,
    output_path: str,
    background_color: tuple[float, float, float],
    result_queue,
) -> None:
    """Helper executed in a subprocess to isolate FURY crashes."""

    try:
        tractogram = nib_streamlines.load(tract_path).tractogram
        streamlines = _coerce_streamlines_xyz(Streamlines(tractogram.streamlines))
        if _DBG:
            print(f"[tracks][dbg] FURY worker loaded {len(streamlines)} streamlines", flush=True)
        _render_with_fury(
            streamlines,
            output_path,
            background_color=background_color,
        )
    except KeyboardInterrupt:
        raise
    except Exception:
        import traceback as _tb
        tb = _tb.format_exc()
        # include a tiny bit of state; affine shapes etc are printed earlier already
        msg = "[worker traceback follows]\n" + tb
        result_queue.put((False, msg))
    else:
        result_queue.put((True, None))


def _env_flag_disabled(value: str) -> bool:
    """Return ``True`` when ``value`` represents a disabled boolean flag."""

    return value.strip().lower() in {"0", "false", "no", "off"}


def _should_use_fury() -> bool:
    """Determine whether FURY rendering should be attempted.

    Default OFF on Apple Silicon macOS due to VTK/Metal offscreen quirks
    that yield transform shape mismatches. Allow explicit opt-in via
    P_BRAIN_ENABLE_FURY=1. Explicit disable via P_BRAIN_DISABLE_FURY=1.
    """

    if not _FURY_AVAILABLE:
        return False

    flag = os.environ.get("P_BRAIN_ENABLE_FURY", "")
    if flag:
        return not _env_flag_disabled(flag)

    disable_flag = os.environ.get("P_BRAIN_DISABLE_FURY", "")
    if disable_flag:
        return not disable_flag.strip().lower() in {"1", "true", "yes", "on"}

    # Platform default: avoid FURY on Apple Silicon macOS
    try:
        if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
            if _DBG:
                _dbg_print("[tracks][dbg] FURY default OFF on macOS arm64; use P_BRAIN_ENABLE_FURY=1 to force ON")
            return False
    except Exception:
        pass
    return True


def _render_with_fury_isolated(
    tract_path: str,
    output_path: str,
    *,
    background_color: tuple[float, float, float] = (0.02, 0.02, 0.05),
) -> bool:
    """Render streamlines via FURY inside a subprocess, returning success."""

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_fury_render_worker,
        args=(tract_path, output_path, background_color, result_queue),
    )
    process.start()
    process.join()

    success = False
    message: Optional[str] = None

    if process.exitcode == 0:
        try:
            success, message = result_queue.get_nowait()
        except queue.Empty:  # pragma: no cover - unexpected but tolerable
            success = True
    else:
        success = False
        if process.exitcode is not None:
            if process.exitcode < 0:
                message = f"terminated by signal {-process.exitcode}"
            elif process.exitcode > 0:
                message = f"exited with status {process.exitcode}"

    if not success:
        prefix = "[tracks] FURY rendering failed"
        if message:
            # message already includes traceback from worker
            print(f"{prefix}:\n{message}")
        else:
            print(f"{prefix} – falling back to matplotlib")

    return success


def _render_with_matplotlib(
    streamlines: Streamlines,
    output_path: str,
) -> None:
    """Fallback renderer relying on matplotlib when ``fury`` is unavailable."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - side effect enables 3D plots

    fig = plt.figure(figsize=(10, 10))
    if _DBG:
        _dbg_print(
            f"[tracks][dbg] matplotlib fallback renderer. "
            f"streamlines_count={sum(1 for _ in streamlines)}"
        )
        # reset iterator by re-wrapping (Streamlines is re-iterable)
        streamlines = Streamlines(streamlines)
    ax = fig.add_subplot(111, projection="3d")

    for sl in streamlines:
        sl = _to_xyz(sl)
        if len(sl) < 2:
            continue
        segments = np.diff(sl, axis=0)
        norms = np.linalg.norm(segments, axis=1, keepdims=True)
        directions = np.zeros_like(segments)
        valid = norms[:, 0] > 0
        directions[valid] = segments[valid] / norms[valid]
        colour = np.abs(directions.mean(axis=0)) if valid.any() else np.array([0.5, 0.5, 0.5])
        colour = np.clip(colour, 0.1, 1.0)
        ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], color=colour, linewidth=0.6, alpha=0.8)

    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.15)

    ax.view_init(elev=75, azim=90)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor="black")
    plt.close(fig)


def _aggregate_streamline_colours(
    streamlines: Iterable[np.ndarray],
    affine: np.ndarray,
    volume_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-voxel colour averages and streamline densities."""

    aff4 = _canonical_affine(affine)
    inv_affine = np.linalg.inv(aff4)
    _dbg(f"overlay inv_affine shape={inv_affine.shape}")
    counts = np.zeros(volume_shape, dtype=np.float32)
    colour_sum = np.zeros(volume_shape + (3,), dtype=np.float32)

    for streamline in streamlines:
        if streamline.shape[0] < 2:
            continue
        streamline = _to_xyz(streamline)

        world_a = streamline[:-1]
        world_b = streamline[1:]
        midpoints_world = 0.5 * (world_a + world_b)

        directions = world_b - world_a
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        valid = norms[:, 0] > 0
        if not np.any(valid):
            continue

        directions = directions[valid] / norms[valid]
        midpoints_world = midpoints_world[valid]

        colours = np.abs(directions)

        midpoints_vox = nib.affines.apply_affine(inv_affine, midpoints_world)
        indices = np.round(midpoints_vox).astype(int)

        valid_mask = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < volume_shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < volume_shape[1])
            & (indices[:, 2] >= 0)
            & (indices[:, 2] < volume_shape[2])
        )

        if not np.any(valid_mask):
            continue

        indices = indices[valid_mask]
        colours = colours[valid_mask]

        counts[indices[:, 0], indices[:, 1], indices[:, 2]] += 1.0
        colour_sum[indices[:, 0], indices[:, 1], indices[:, 2]] += colours

    with np.errstate(invalid="ignore", divide="ignore"):
        colour_avg = np.zeros_like(colour_sum)
        mask = counts > 0
        colour_avg[mask] = colour_sum[mask] / counts[mask][..., None]

    return colour_avg, counts


def _render_montage(
    streamlines: Streamlines,
    reference_img: nib.Nifti1Image,
    background_img: nib.Nifti1Image,
    output_path: str,
    *,
    title: str,
) -> None:
    """Create a multi-slice axial montage showing streamline overlays."""

    import matplotlib.pyplot as plt

    background = np.asarray(background_img.get_fdata(), dtype=np.float32)
    if background.ndim > 3:
        background = background[..., 0]

    background = np.nan_to_num(background)
    bg_min, bg_max = np.percentile(background, (1, 99))
    if bg_max <= bg_min:
        bg_min, bg_max = float(background.min()), float(background.max())

    colours, counts = _aggregate_streamline_colours(
        streamlines, _canonical_affine(reference_img.affine), background.shape
    )

    z_slices = np.linspace(0, background.shape[2] - 1, 6, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    for ax, z_index in zip(axes, z_slices):
        slice_bg = background[:, :, z_index]
        slice_colours = colours[:, :, z_index]
        slice_counts = counts[:, :, z_index]

        display_bg = np.rot90(slice_bg)
        ax.imshow(
            display_bg,
            cmap="gray",
            vmin=bg_min,
            vmax=bg_max,
            interpolation="bicubic",
        )

        if np.any(slice_counts):
            display_colours = np.rot90(slice_colours)
            display_alpha = np.rot90(slice_counts)
            display_alpha = display_alpha / float(display_alpha.max())
            display_alpha = np.clip(display_alpha, 0.05, 1.0)

            ax.imshow(
                display_colours,
                interpolation="bilinear",
                alpha=display_alpha,
            )

        ax.set_title(f"z = {z_index}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _header_debug(img: nib.Nifti1Image) -> dict:
    hdr = img.header
    out = {}
    try:
        out = {
            "dim": tuple(int(x) for x in hdr.get("dim", ())),
            "qform_code": int(hdr["qform_code"]), "sform_code": int(hdr["sform_code"]),
        }
    except Exception:
        pass
    return out


def generate_tractography(
    nifti_directory: str,
    analysis_directory: str,
    image_directory: Optional[str] = None,
    diffusion_filename: Optional[str] = None,
    *,
    anatomical_overlay: Optional[str] = None,
    create_montage: bool = False,
    montage_title: str = "Tractography overlay",
) -> TractographyOutputs:
    """Compute deterministic tractography and associated visualisations."""

    diffusion_dir = os.path.join(analysis_directory, "diffusion")
    os.makedirs(diffusion_dir, exist_ok=True)
    debug_json_path = os.path.join(diffusion_dir, "tractography_debug.json")
    debug_blob = {
        "ts": datetime.datetime.now().isoformat(),
        "numpy_version": getattr(np, "__version__", None),
        "dipy_version": None,
        "env": {k: v for k, v in os.environ.items() if k.startswith("P_BRAIN") or k.startswith("OMP_")},
    }
    try:
        import dipy

        debug_blob["dipy_version"] = getattr(dipy, "__version__", None)
    except Exception:
        pass

    tract_path = os.path.join(diffusion_dir, "tractography.trk")
    preferred = (diffusion_filename,) if diffusion_filename else None
    found = find_dwi_files(nifti_directory, preferred_filenames=preferred)
    if not found:
        raise FileNotFoundError(
            "No diffusion dataset found – configure utils/parameters.py for DWI filenames."
        )

    dwi_path, bval_path, bvec_path = found

    img = nib.load(dwi_path)
    # Force a 4x4 voxel->world affine even if header carries higher-dim transforms.
    voxel_to_world = _canonical_affine(getattr(img, "affine", np.eye(4)))
    if _DBG:
        _dbg_print(f"[tracks][dbg] DWI file: {dwi_path}")
        _dbg_print(f"[tracks][dbg] Header dim/qform/sform: {_header_debug(img)}")
        _print_affine("img.affine (voxel->mm)", voxel_to_world)
        try:
            xyzt = img.header.get_xyzt_units()
            _dbg_print(f"[tracks][dbg] xyzt units={xyzt}")
        except Exception:
            pass

    streamlines: Optional[Streamlines] = None
    if os.path.exists(tract_path):
        if _DBG:
            _dbg_print(f"[tracks][dbg] pre-existing tract file found: {tract_path}")
        streamlines = _load_streamlines_world(
            tract_path,
            expected_affine=voxel_to_world,
            expected_shape=img.shape,
        )

    fa_volume: Optional[np.ndarray] = None

    if streamlines is None:
        raw = img.get_fdata(dtype=np.float32)
        bvals = np.loadtxt(bval_path)
        bvecs = np.loadtxt(bvec_path)
        if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        data = _as_4d_dwi(raw, bvals)

        if data.shape[-1] != bvals.size or bvecs.shape != (bvals.size, 3):
            raise ValueError(
                f"bvals/bvecs mismatch. volumes={data.shape[-1]}, "
                f"bvals={bvals.size}, bvecs={bvecs.shape}"
            )
        if _DBG:
            _dbg_print(f"[tracks][dbg] DWI shape={data.shape}")
            _dbg_print(
                f"[tracks][dbg] bvals size={bvals.size} unique="
                f"{sorted(set(int(x) for x in bvals.tolist()))}"
            )
            _dbg_print(
                f"[tracks][dbg] bvecs shape={bvecs.shape} norms≈"
                f"{[float(f'{n:.3f}') for n in np.linalg.norm(bvecs,axis=1)[:6]]}"
            )
            _dbg_print(
                f"[tracks][dbg] affine(vox->mm) shape={voxel_to_world.shape}"
            )
            debug_blob["dwi_shape"] = tuple(int(x) for x in data.shape)
            debug_blob["bvals_len"] = int(bvals.size)
            debug_blob["bvecs_shape"] = tuple(int(x) for x in bvecs.shape)

        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        tensor_model = TensorModel(gtab)
        try:
            tensor_fit = tensor_model.fit(data)
        except Exception:
            if _DBG:
                _dbg_print("[tracks][err] TensorModel.fit failed")
                _dbg_print(traceback.format_exc())
            raise

        fa_volume = tensor_fit.fa.astype(np.float32)
        fa_volume = np.nan_to_num(fa_volume, nan=0.0, posinf=0.0, neginf=0.0)
        if _DBG:
            _dbg_print(
                f"[tracks][dbg] FA stats: min={float(np.min(fa_volume)):.4f} "
                f"max={float(np.max(fa_volume)):.4f} "
                f"mean={float(np.mean(fa_volume)):.4f}"
            )

        wm_mask = fa_volume > 0.2
        if not np.any(wm_mask):
            raise RuntimeError(
                "Diffusion data does not contain voxels above FA threshold (0.2)"
            )
        if _DBG:
            _dbg_print(
                f"[tracks][dbg] WM mask voxels={int(np.count_nonzero(wm_mask))}"
            )

        stopping_criterion = ThresholdStoppingCriterion(fa_volume, 0.15)

        try:
            peaks = peaks_from_model(
                tensor_model,
                data,
                default_sphere,
                relative_peak_threshold=0.5,
                min_separation_angle=25,
                mask=wm_mask,
                return_sh=False,
            )
        except Exception:
            if _DBG:
                _dbg_print("[tracks][err] peaks_from_model failed")
                _dbg_print(traceback.format_exc())
            raise
        if _DBG:
            try:
                _dbg_print(
                    f"[tracks][dbg] peaks.peak_dirs shape={getattr(peaks,'peak_dirs',None).shape}"
                )
                if hasattr(peaks, "affine"):
                    _print_affine("peaks.affine", peaks.affine)
            except Exception:
                pass

        # Keep seeds aligned with the diffusion affine to avoid affine shape mismatches.
        seeds = _seed_points_from_mask(wm_mask, voxel_to_world, density=1)
        if _DBG:
            _dbg_print(
                f"[tracks][dbg] seeds shape={getattr(seeds,'shape',None)} dtype={getattr(seeds,'dtype',None)} "
                f"sample={seeds[:3].tolist() if hasattr(seeds,'__array_interface__') and seeds.size>0 else '<empty>'}"
            )
            debug_blob["seeds_shape"] = tuple(int(x) for x in np.shape(seeds))
        try:
            streamline_generator = LocalTracking(
                peaks,
                stopping_criterion,
                seeds,
                affine=voxel_to_world,
                step_size=0.5,
                return_all=False,
            )
        except Exception:
            if _DBG:
                _dbg_print("[tracks][err] LocalTracking construction failed")
                _dbg_print(
                    f"[tracks][dbg] voxel_to_world shape={voxel_to_world.shape}"
                )
                try:
                    _dbg_print(
                        f"[tracks][dbg] seeds dtype/shape={seeds.dtype}/{seeds.shape}"
                    )
                except Exception:
                    pass
                try:
                    _dbg_print(
                        f"[tracks][dbg] stopping_criterion data shape="
                        f"{getattr(stopping_criterion,'data',None).shape}"
                    )
                except Exception:
                    pass
                _dbg_print(traceback.format_exc())
            raise
        streamlines = _coerce_streamlines_xyz(Streamlines(streamline_generator))
        _dbg(f"streamlines generated: {len(streamlines)}")

        if len(streamlines) == 0:
            raise RuntimeError(
                "Tractography produced no streamlines – check diffusion quality"
            )

        try:
            if StatefulTractogram is not None and Space is not None:
                sft = StatefulTractogram(streamlines, img, Space.RASMM)
                nib_streamlines.save(sft, tract_path)
            else:  # pragma: no cover - legacy nibabel fallback
                tractogram = nib_streamlines.Tractogram(
                    streamlines,
                    affine_to_rasmm=np.eye(4),
                )
                nib_streamlines.save(tractogram, tract_path)
            _dbg(f"saved tractogram: {tract_path}")
        except Exception:
            if _DBG:
                _dbg_print("[tracks][err] Saving tractogram failed")
                _dbg_print(traceback.format_exc())
            raise
    else:
        fa_candidates = (
            os.path.join(diffusion_dir, "fa_map_native_debug.nii.gz"),
            os.path.join(diffusion_dir, "fa_map.nii.gz"),
        )
        for candidate in fa_candidates:
            if not os.path.isfile(candidate):
                continue
            try:
                fa_img = nib.load(candidate)
            except Exception:
                continue
            fa_data = np.asarray(fa_img.get_fdata(), dtype=np.float32)
            if fa_data.shape[:3] == img.shape[:3]:
                fa_volume = fa_data
                break
        # Streamlines loaded from file can carry 4 columns in rare cases
        loaded_streamlines = _load_streamlines_world(
            tract_path,
            expected_affine=voxel_to_world,
            expected_shape=img.shape,
        )
        if loaded_streamlines is not None:
            streamlines = loaded_streamlines

    image_directory = _ensure_image_directory(image_directory, analysis_directory)
    tract_image_dir = os.path.join(image_directory, "tractography")
    os.makedirs(tract_image_dir, exist_ok=True)

    if _DBG:
        try:
            _dbg_print(f"[tracks][dbg] platform={platform.system()} arch={platform.machine()}")
        except Exception: pass
    # Persist debug snapshot
    if _DBG:
        try:
            debug_blob["voxel_to_world"] = np.array(voxel_to_world).tolist()
            debug_blob["tract_exists"] = os.path.exists(tract_path)
            _dump_debug_json(debug_json_path, debug_blob)
            _dbg_print(f"[tracks][dbg] wrote debug snapshot: {debug_json_path}")
        except Exception:
            pass

    render_path = os.path.join(tract_image_dir, "tractography_render.png")
    if _should_use_fury():
        if not _render_with_fury_isolated(tract_path, render_path):
            _render_with_matplotlib(streamlines, render_path)
    else:  # pragma: no cover - fallback depends on runtime environment
        _render_with_matplotlib(streamlines, render_path)

    montage_path: Optional[str] = None
    if create_montage:
        if anatomical_overlay:
            background_img = _load_background_volume(anatomical_overlay, img)
        else:
            if fa_volume is None:
                # When tractography was precomputed we may not have FA in-memory.
                raw = img.get_fdata(dtype=np.float32)
                bvals = np.loadtxt(bval_path)
                bvecs = np.loadtxt(bvec_path)
                if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
                    bvecs = bvecs.T
                data = _as_4d_dwi(raw, bvals)
                gtab = gradient_table(bvals=bvals, bvecs=bvecs)
                tensor_model = TensorModel(gtab)
                tensor_fit = tensor_model.fit(data)
                fa_volume = tensor_fit.fa.astype(np.float32)
                fa_volume = np.nan_to_num(fa_volume, nan=0.0, posinf=0.0, neginf=0.0)
            background_img = nib.Nifti1Image(
                fa_volume, voxel_to_world, img.header
            )
        montage_path = os.path.join(tract_image_dir, "tractography_montage.png")
        try:
            _render_montage(streamlines, img, background_img, montage_path, title=montage_title)
        except Exception as e:
            _dbg(f"*** FATAL in section: render montage | {type(e).__name__}: {e}")
            raise

    return TractographyOutputs(
        tract_path=tract_path,
        render_path=render_path,
        montage_path=montage_path,
    )

