"""Manual AIF — interactive GUI ROI delineation (paper §4.4).

A clinician-in-the-loop input-function provider. On the time-averaged DCE
volume the operator draws an arterial (and optionally a venous) ROI on a
chosen slice with a matplotlib polygon/lasso widget; the enclosed pixels
become a binary mask and the input-function curve is read out of that ROI
through the **same** concentration-curve machinery the CNN extractor uses
(``cnn._curve_from_masks``) — by default the single highest-peak voxel
within the ROI (``curve_method="max_voxel"``), so a manually drawn ROI and
a CNN-segmented ROI extract their curves identically.

The plug-in is structured so the *curve extraction* never needs a display:

* :func:`input_function_from_mask` — headless core. Given a 4-D DCE volume
  and a 3-D boolean ROI mask, it returns the standard
  :class:`InputFunction`. This is what the unit tests exercise and what the
  interactive path ultimately calls.
* The interactive matplotlib loop is entered **only** when a usable GUI
  backend is available *and* no mask/polygon was supplied programmatically.
  Callers can therefore drive the plug-in non-interactively by passing
  ``mask=`` (a precomputed 3-D boolean array) or ``polygon=`` + ``slice_z=``
  (vertices to rasterise on one slice), which is also how it is tested.

Importing and registering this module never opens a display.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import numpy as np

from .base import AIFExtractor, InputFunction
from .cnn import _curve_from_masks


# ────────────────────────────────────────────────────────────────────────
# Headless core — mask → InputFunction (no display required)
# ────────────────────────────────────────────────────────────────────────


def _masks_by_slice(mask3d: np.ndarray) -> dict[int, np.ndarray]:
    """Split a 3-D ``(X, Y, Z)`` boolean mask into the per-slice ``{z: (X, Y)}``
    dict that :func:`cnn._curve_from_masks` consumes."""
    out: dict[int, np.ndarray] = {}
    for z in range(mask3d.shape[2]):
        sl = mask3d[:, :, z]
        if sl.any():
            out[int(z)] = np.ascontiguousarray(sl.astype(bool))
    return out


def input_function_from_mask(
    dce_data: np.ndarray,
    t_s: np.ndarray,
    mask3d: np.ndarray,
    *,
    source: str = "user",
    curve_method: str = "max_voxel",
    baseline_frames: int = 5,
    concentration_data: np.ndarray | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> InputFunction:
    """Build an :class:`InputFunction` from a manually drawn 3-D ROI mask.

    This is the headless core: it needs no display and is the single point
    where curve logic lives. The curve is extracted with the *shared*
    :func:`cnn._curve_from_masks` path, so the default ``max_voxel`` method
    reproduces the CNN extractor's "single highest-peak ROI voxel" behaviour.

    When ``concentration_data`` (an mM volume) is supplied the curve is read
    from it (units carry each voxel's own T1/M0); otherwise the raw DCE
    signal is used.
    """
    d = np.asarray(dce_data, dtype=np.float32)
    if d.ndim != 4:
        raise ValueError(f"dce_data must be 4-D; got shape {d.shape}")
    mask3d = np.asarray(mask3d, dtype=bool)
    if mask3d.shape != d.shape[:3]:
        raise ValueError(
            f"mask shape {mask3d.shape} != DCE spatial shape {d.shape[:3]}"
        )
    if not mask3d.any():
        raise ValueError("manual AIF: ROI mask is empty (no voxels selected)")

    # Curve is read from the mM volume when provided, else from the signal.
    source_vol = d if concentration_data is None else np.asarray(
        concentration_data, dtype=np.float32
    )
    if source_vol.shape != d.shape:
        raise ValueError(
            f"concentration_data shape {source_vol.shape} must match "
            f"dce_data shape {d.shape}"
        )

    masks = _masks_by_slice(mask3d)
    curve, mask3d_full, mask3d_curve = _curve_from_masks(
        source_vol,
        masks,
        curve_method=curve_method,
        baseline_frames=int(baseline_frames),
        # Rank the max-voxel on the signal (brightest blood), exactly as cnn.py.
        select_on="signal",
        signal_volume=d,
    )

    meta: dict[str, Any] = {
        "algorithm": "manual_roi",
        "curve_method": curve_method,
        "n_voxels": int(mask3d_curve.sum()),
        "n_voxels_full_roi": int(mask3d_full.sum()),
        "roi_slices": sorted(masks.keys()),
    }
    if meta_extra:
        meta.update(meta_extra)

    return InputFunction(
        c_a=np.asarray(curve, dtype=np.float32),
        t_s=np.asarray(t_s, dtype=float),
        mask=mask3d_curve,            # tight: only the voxel(s) the curve came from
        source=source,
        meta=meta,
    )


# ────────────────────────────────────────────────────────────────────────
# Polygon rasterisation (headless) — vertices on one slice → 3-D mask
# ────────────────────────────────────────────────────────────────────────


def _rasterise_polygon(
    polygon: Sequence[Sequence[float]],
    spatial_shape: tuple[int, int, int],
    slice_z: int,
) -> np.ndarray:
    """Fill ``polygon`` (``[(x, y), …]`` in voxel coords) on slice ``z`` into a
    3-D boolean mask of ``spatial_shape``. Uses matplotlib's ``Path`` (no GUI
    backend, no display)."""
    from matplotlib.path import Path as MplPath

    X, Y, Z = spatial_shape
    z = int(slice_z)
    if not (0 <= z < Z):
        raise ValueError(f"slice_z {z} out of range [0, {Z})")
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("polygon must be >=3 (x, y) vertices")

    xs, ys = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    inside = MplPath(poly).contains_points(pts).reshape(X, Y)

    mask3d = np.zeros(spatial_shape, dtype=bool)
    mask3d[:, :, z] = inside
    return mask3d


# ────────────────────────────────────────────────────────────────────────
# Display detection
# ────────────────────────────────────────────────────────────────────────


def _display_available() -> bool:
    """True only if an interactive matplotlib backend can plausibly open a
    window. Conservative: any doubt → False (so headless/CI never blocks)."""
    if os.environ.get("MPLBACKEND", "").lower() in {"agg", "pdf", "ps", "svg", "template"}:
        return False
    if os.environ.get("PBRAIN_NO_GUI"):
        return False
    if os.name == "posix" and os.sys.platform.startswith("linux"):
        # Linux GUI needs an X / Wayland display.
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False
    try:
        import matplotlib

        backend = matplotlib.get_backend().lower()
    except Exception:
        return False
    # Non-interactive backends can't host the selector widgets.
    return backend not in {"agg", "pdf", "ps", "svg", "template"}


# ────────────────────────────────────────────────────────────────────────
# Interactive ROI drawing (display required) — thin wrapper over the core
# ────────────────────────────────────────────────────────────────────────


def _draw_roi_interactive(
    dce_data: np.ndarray,
    *,
    slice_z: int | None,
    title: str,
) -> tuple[np.ndarray, int]:
    """Open a matplotlib window showing the time-averaged DCE slice and let the
    user draw one ROI polygon. Returns ``(mask3d, slice_z)``. Blocks until the
    window is closed. Only called when :func:`_display_available` is True."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector, Slider

    d = np.asarray(dce_data, dtype=np.float32)
    X, Y, Z, _ = d.shape
    mean_vol = d.mean(axis=-1)                       # time-averaged (X, Y, Z)
    z0 = int(Z // 2 if slice_z is None else slice_z)

    state: dict[str, Any] = {"z": z0, "verts": None}

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.18)
    img = ax.imshow(mean_vol[:, :, state["z"]].T, cmap="gray", origin="lower")
    ax.set_title(f"{title}\nDraw ROI, then close the window")

    def _on_select(verts: list[tuple[float, float]]) -> None:
        state["verts"] = list(verts)

    selector = PolygonSelector(ax, _on_select)

    slider_ax = fig.add_axes((0.15, 0.05, 0.7, 0.04))
    z_slider = Slider(slider_ax, "slice", 0, max(Z - 1, 0), valinit=z0, valstep=1)

    def _on_slice(val: float) -> None:
        state["z"] = int(val)
        img.set_data(mean_vol[:, :, state["z"]].T)
        fig.canvas.draw_idle()

    z_slider.on_changed(_on_slice)

    plt.show(block=True)

    if not state["verts"] or len(state["verts"]) < 3:
        raise RuntimeError("manual AIF: no ROI was drawn (need >=3 vertices)")

    mask3d = _rasterise_polygon(state["verts"], (X, Y, Z), state["z"])
    return mask3d, int(state["z"])


# ────────────────────────────────────────────────────────────────────────
# Plugin
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _ManualAIF:
    key: ClassVar[str] = "manual"
    name: ClassVar[str] = "Manual GUI ROI drawing"
    description: ClassVar[str] = (
        "Interactive arterial/venous ROI delineation on the time-averaged "
        "DCE volume via a matplotlib PolygonSelector; the curve is extracted "
        "from the drawn ROI through the shared max-voxel path (same as the "
        "CNN extractor). Headless callers may pass a precomputed `mask=` or "
        "`polygon=`+`slice_z=` instead of drawing."
    )
    accepts: ClassVar[dict[str, type]] = {"dce_data": np.ndarray, "t_s": np.ndarray}
    produces: ClassVar[dict[str, type]] = {"input_function": InputFunction}

    def extract(
        self,
        dce_data: np.ndarray,
        t_s: np.ndarray,
        dce_affine: np.ndarray | None = None,
        *,
        t1_map: np.ndarray | None = None,
        m0_map: np.ndarray | None = None,
        t1w_volume: np.ndarray | None = None,
        t1w_affine: np.ndarray | None = None,
        concentration_data: np.ndarray | None = None,
        source: str = "user",
        mask: np.ndarray | None = None,
        polygon: Sequence[Sequence[float]] | None = None,
        slice_z: int | None = None,
        curve_method: str = "max_voxel",
        baseline_frames: int = 5,
        interactive: bool | None = None,
        title: str = "Manual AIF ROI",
        **_: Any,
    ) -> InputFunction:
        """Extract the input function from a manually delineated ROI.

        Resolution order for the ROI:

        1. ``mask`` — a precomputed 3-D boolean ROI (fully headless).
        2. ``polygon`` (+ ``slice_z``) — vertices rasterised on one slice
           (headless).
        3. Interactive drawing — only if a display is available (or
           ``interactive=True`` forces it); raises otherwise.

        The curve itself is always extracted by the shared headless core
        :func:`input_function_from_mask`, so interactive and programmatic
        ROIs behave identically downstream.
        """
        d = np.asarray(dce_data, dtype=np.float32)
        if d.ndim != 4:
            raise ValueError(f"dce_data must be 4-D; got shape {d.shape}")
        X, Y, Z, _ = d.shape

        meta_extra: dict[str, Any] = {}
        if mask is not None:
            mask3d = np.asarray(mask, dtype=bool)
            meta_extra["roi_origin"] = "mask"
        elif polygon is not None:
            if slice_z is None:
                raise ValueError("manual AIF: `polygon` requires `slice_z`")
            mask3d = _rasterise_polygon(polygon, (X, Y, Z), int(slice_z))
            meta_extra["roi_origin"] = "polygon"
            meta_extra["slice_z"] = int(slice_z)
        else:
            want_gui = _display_available() if interactive is None else bool(interactive)
            if not want_gui:
                raise RuntimeError(
                    "manual AIF: no ROI supplied and no interactive display "
                    "available. Pass `mask=` (3-D bool array) or "
                    "`polygon=`+`slice_z=`, draw with the legacy GUI and use "
                    "'from_file', or run where a matplotlib GUI backend is "
                    "usable (and set interactive=True)."
                )
            mask3d, drawn_z = _draw_roi_interactive(d, slice_z=slice_z, title=title)
            meta_extra["roi_origin"] = "interactive"
            meta_extra["slice_z"] = int(drawn_z)

        return input_function_from_mask(
            d,
            t_s,
            mask3d,
            source=source,
            curve_method=curve_method,
            baseline_frames=int(baseline_frames),
            concentration_data=concentration_data,
            meta_extra=meta_extra,
        )


PLUGIN = _ManualAIF()
