"""Render tractography streamlines to a PNG.

Produces the familiar direction-encoded ("anatomical RGB") streamline view:
red = left-right, green = anterior-posterior, blue = inferior-superior. Uses
FURY / VTK offscreen rendering when available; falls back to a Matplotlib 3-D
line projection when FURY/VTK is not installed, so the pipeline always yields
*some* streamline figure.

By default it renders a three-panel montage at successive rotations about the
vertical (superior-inferior) axis -- front, three-quarter and lateral -- so the
3-D structure reads on a flat page.
"""

from __future__ import annotations

import numpy as np


def _subsample(streamlines, n):
    sl = [np.asarray(s, dtype=float) for s in streamlines if len(s) >= 2]
    if n and len(sl) > n:
        idx = np.linspace(0, len(sl) - 1, int(n)).astype(int)
        sl = [sl[i] for i in idx]
    return sl


def _camera(sl, deg):
    """(position, focal_point, view_up) looking at the streamlines from an
    azimuth ``deg`` about the vertical (S-I / Z) axis. deg=0 is the coronal
    front view (camera on +Y, anterior); deg=90 is the left lateral view."""
    pts = np.concatenate(sl)
    c = pts.mean(axis=0)
    d = 2.6 * float(np.ptp(pts, axis=0).max())
    th = np.deg2rad(deg)
    direction = np.array([np.sin(th), np.cos(th), 0.0])   # +Y at 0, +X at 90
    return c + d * direction, c, (0.0, 0.0, 1.0)


def render_streamlines(streamlines, out_png, *, size=(1700, 1700),
                       max_streamlines=15000, background=(1.0, 1.0, 1.0),
                       angles=(0, 55)):
    """Render ``streamlines`` (world/RASMM coords) to ``out_png``.

    ``angles`` (degrees about the vertical axis) controls the panels: a tuple
    renders that many views side by side (default front / three-quarter /
    lateral); pass a single-element tuple for one view. Returns the path, or
    ``None`` if there is nothing to draw.
    """
    sl = _subsample(streamlines, max_streamlines)
    if not sl:
        return None
    angles = tuple(angles) if np.iterable(angles) else (float(angles),)
    try:
        import tempfile
        import pathlib
        with tempfile.TemporaryDirectory() as td:
            panels = []
            for i, a in enumerate(angles):
                p = pathlib.Path(td) / f"view_{i}.png"
                _render_fury_view(sl, a, p, size=size, background=background)
                panels.append(p)
            _montage(panels, out_png, background=background)
        return out_png
    except Exception:                       # noqa: BLE001 - never crash the run
        return _render_mpl(sl, out_png, angles=angles, size=size)


def _render_fury_view(sl, deg, tmp_path, *, size, background):
    """Render one view to ``tmp_path`` via FURY offscreen (window.record)."""
    from fury import actor, window
    from dipy.viz import colormap

    colors = colormap.line_colors(sl)
    scene = window.Scene()
    scene.background(background)
    # thin, semi-transparent lines: dense bundles stay translucent so the major
    # white-matter tracts read through instead of pancaking into a solid mass.
    scene.add(actor.line(sl, colors=colors, linewidth=0.8, opacity=0.55))
    pos, foc, up = _camera(sl, deg)
    window.record(scene=scene, out_path=str(tmp_path), size=size, reset_camera=False,
                  cam_pos=tuple(pos), cam_focal=tuple(foc), cam_view=tuple(up))
    return tmp_path


def _montage(paths, out_png, *, background):
    """Crop each panel to the streamlines (a small margin), then abut them at
    native resolution — no resize, so nothing blurs. Cropping removes the large
    empty border fury leaves, so the brain fills the panel and reads sharply."""
    from PIL import Image, ImageChops
    bg = tuple(int(255 * c) for c in background)
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        bbox = ImageChops.difference(im, Image.new("RGB", im.size, bg)).getbbox()
        if bbox:
            m = int(0.03 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            bbox = (max(0, bbox[0] - m), max(0, bbox[1] - m),
                    min(im.width, bbox[2] + m), min(im.height, bbox[3] + m))
            im = im.crop(bbox)
        ims.append(im)
    h = max(im.height for im in ims)                       # pad to tallest, do NOT resize
    pad = int(0.02 * h)
    canvas = Image.new("RGB", (sum(im.width for im in ims) + pad * (len(ims) - 1), h), bg)
    x = 0
    for im in ims:
        canvas.paste(im, (x, (h - im.height) // 2)); x += im.width + pad
    canvas.save(out_png)
    return out_png


def _render_mpl(sl, out_png, *, angles, size):
    """Matplotlib fallback: one 3-D panel per angle (per-streamline plot so
    ragged lengths never hit array stacking)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = sl if len(sl) <= 4000 else [sl[i] for i in np.linspace(0, len(sl) - 1, 4000).astype(int)]
    allp = np.concatenate(sub)
    cols = [np.abs(s[-1] - s[0]) / (np.linalg.norm(s[-1] - s[0]) or 1.0) for s in sub]
    fig = plt.figure(figsize=(size[0] * len(angles) / 150.0, size[1] / 150.0))
    for k, deg in enumerate(angles):
        ax = fig.add_subplot(1, len(angles), k + 1, projection="3d")
        for s, c in zip(sub, cols):
            ax.plot(s[:, 0], s[:, 1], s[:, 2], color=c, lw=0.3, alpha=0.6)
        ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
        ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
        ax.set_zlim(allp[:, 2].min(), allp[:, 2].max())
        ax.set_box_aspect(np.ptp(allp, axis=0))
        ax.set_axis_off()
        ax.view_init(elev=0, azim=-90 + float(deg))
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png
