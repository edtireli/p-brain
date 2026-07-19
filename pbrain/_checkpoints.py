"""Interactive decision checkpoints for ``--mode verify`` / ``--mode manual``.

A stage calls the relevant checkpoint (e.g. :func:`aif_checkpoint`). In
verify/manual mode — and only on an interactive run where a browser can open —
it builds a payload, serves the review page via :mod:`pbrain._webreview`, blocks
until the user confirms, and returns the (possibly adjusted) result. ``auto``
mode and non-interactive runs are a no-op. Rejecting a review raises
:class:`~pbrain.core.pipeline.CheckpointAbort`, which the orchestrator catches to
stop the run cleanly.

The maths never runs here — a checkpoint only reads what the stage already
computed and hands the user's choice back. Whatever the user picks is recorded in
the stage's manifest (``meta['review']``), so a re-run reproduces it.

Image ↔ curve coordinate convention: a slice is ``dce[:, :, z]`` of shape
``(X, Y)`` rendered with ``imshow(aspect='auto')``, so voxel ``(xi, yi)`` maps to
the page's normalised ``(u, v) = (yi / Y, xi / X)`` and back.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from dataclasses import replace

import numpy as np

from pbrain.core.logs import get_logger
from pbrain.core.pipeline import CheckpointAbort

_log = get_logger("checkpoint")

# If the browser is closed without answering, don't hang the run forever — fall
# back to the suggested choice after this long.
_REVIEW_TIMEOUT_S = 1800   # 30 minutes


def active(mode: str) -> bool:
    """A checkpoint fires only in ``verify``/``manual``, on an interactive TTY run
    where we can actually open a browser (else it would just hang)."""
    if mode not in ("verify", "manual"):
        return False
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


def _finite_list(a) -> list[float]:
    return [float(v) if np.isfinite(v) else None for v in np.asarray(a, dtype=float)]


def png_data_uri(slice2d: np.ndarray) -> str:
    """Render a 2-D array as a base64 grayscale PNG data URI (1–99.5 % window)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    a = np.asarray(slice2d, dtype=float)
    finite = a[np.isfinite(a)]
    lo, hi = (np.percentile(finite, 1), np.percentile(finite, 99.5)) if finite.size else (0.0, 1.0)
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    fig = plt.figure(figsize=(3, 3), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(np.nan_to_num(a), cmap="gray", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _orient_flips(affine) -> tuple[bool, bool]:
    """From a NIfTI affine, the ``(flip_lr, flip_ud)`` to apply AFTER a 90° CCW
    rotation so an axial slice displays **neurological** (patient-left on the
    viewer's left) and **anterior-up** — derived from the in-plane axis directions
    (``aff2axcodes``): axis 0 → L needs a post-rotation L-R flip; axis 1 → P needs
    a vertical flip. No affine ⇒ assume LAS (what dcm2niix emits for PAR/REC)."""
    if affine is None:
        return True, False
    try:
        import nibabel as nib
        c0, c1, _ = nib.aff2axcodes(np.asarray(affine))
        return (c0 == "L"), (c1 == "P")
    except Exception:
        return True, False


def _disp(a, flip_lr: bool = True, flip_ud: bool = False):
    """Orient a slice/mask for display: 90° CCW then the affine-derived flips."""
    r = np.rot90(np.asarray(a), 1)
    if flip_lr:
        r = np.fliplr(r)
    if flip_ud:
        r = np.flipud(r)
    return r


def _vox_to_disp(xi, yi, X, Y, flip_lr: bool = True, flip_ud: bool = False) -> list[float]:
    """Voxel ``(xi, yi)`` → normalised display ``(u, v)`` under :func:`_disp`."""
    u = (X - 1 - xi) / X if flip_lr else xi / X
    v = yi / Y if flip_ud else (Y - 1 - yi) / Y
    return [u, v]


def _disp_to_vox(u, v, X, Y, flip_lr: bool = True, flip_ud: bool = False) -> tuple[int, int]:
    """Inverse of :func:`_vox_to_disp`."""
    xi = int(round((X - 1) - float(u) * X)) if flip_lr else int(round(float(u) * X))
    yi = int(round(float(v) * Y)) if flip_ud else int(round((Y - 1) - float(v) * Y))
    return max(0, min(xi, X - 1)), max(0, min(yi, Y - 1))


def _polygon_of(m) -> list[list[float]]:
    """Convex-hull outline of a 2-D mask already in display orientation, as
    normalised ``[[u, v], …]`` = [col/W, row/H]. Empty if too few points."""
    pts = np.argwhere(np.asarray(m, dtype=bool))       # (n, 2) = (row, col)
    if pts.shape[0] < 3:
        return []
    try:
        from scipy.spatial import ConvexHull
        H_, W_ = m.shape
        verts = pts[ConvexHull(pts).vertices]
        return [[float(c) / W_, float(r) / H_] for r, c in verts]
    except Exception:
        return []


def _polygon_to_mask(poly_uv, X, Y, flip_lr=True, flip_ud=False) -> np.ndarray:
    """Rasterise a display polygon ``[[u, v], …]`` into an ``(X, Y)`` voxel mask."""
    verts = [_disp_to_vox(u, v, X, Y, flip_lr, flip_ud) for u, v in poly_uv]
    if len(verts) < 3:
        return np.zeros((X, Y), dtype=bool)
    from matplotlib.path import Path as MplPath
    gx, gy = np.mgrid[0:X, 0:Y]
    inside = MplPath([(float(a), float(b)) for a, b in verts]).contains_points(
        np.column_stack([gx.ravel(), gy.ravel()]))
    return inside.reshape(X, Y)


# ── AIF checkpoint ───────────────────────────────────────────────────────────

def _detect_boluses(ca, t_s, n: int = 2) -> list[dict]:
    """The main bolus peak(s) of the AIF curve — dual-bolus protocols show two.
    Marked on the review plot so you can see the boluses landed where expected."""
    ca = np.nan_to_num(np.asarray(ca, dtype=float))
    t_s = np.asarray(t_s, dtype=float)
    if ca.size < 3 or not np.isfinite(ca).any() or ca.max() <= 0:
        return []
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(ca, height=0.2 * ca.max(), distance=max(2, ca.size // 20))
    except Exception:
        peaks = np.array([int(np.argmax(ca))])
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(ca))])
    top = np.sort(peaks[np.argsort(ca[peaks])[::-1][:n]])
    return [{"t": float(t_s[i]), "c": float(ca[i]), "i": int(i)} for i in top
            if i < t_s.size]


def _peak_voxel(mask: np.ndarray, ct: np.ndarray) -> tuple[int, int, int]:
    idx = np.argwhere(mask)
    if not idx.size:
        X, Y, Z = mask.shape
        return X // 2, Y // 2, Z // 2
    peaks = np.nanmax(ct[mask], axis=1)
    xi, yi, zi = idx[int(np.nanargmax(peaks))]
    return int(xi), int(yi), int(zi)


def _otsu(vals) -> float | None:
    """Otsu's threshold on a 1-D value array (numpy-only, no skimage dep) — the
    intensity that best separates two classes (here: vessel vs surrounding tissue)."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 8 or v.max() <= v.min():
        return None
    hist, edges = np.histogram(v, bins=64)
    p = hist / hist.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(p)                                   # class-1 weight
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = w * (1.0 - w)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w - mu) ** 2 / np.where(denom > 0, denom, np.nan)
    if not np.isfinite(between).any():
        return None
    return float(centres[int(np.nanargmax(between))])


def _grow_vessel(ct_data, seed, roi_coords, *, lo_frac: float = 0.20):
    """Segment the vessel from the max voxel by **hysteresis on the peak-signal**,
    anchored by Otsu — so the whole bright structure is captured (bright core *and*
    the dimmer margins the CNN blob misses) without leaking into background tissue.

    Within a generous window around the CNN ROI: ``hi`` marks definite vessel, ``lo``
    the permissive membrane; the connected component that contains the seed *and*
    touches a ``hi`` voxel is the vessel. A size guard re-thresholds if the window
    is mostly bright (Otsu degenerate). Smoothing is done later, at render time."""
    from scipy import ndimage
    c = np.asarray(roi_coords, dtype=int)
    X, Y, Z = ct_data.shape[:3]
    # window: CNN bbox + a margin that grows with the ROI, so a small CNN blob still
    # gets room to recover a vessel that extends past it.
    ext = int(np.max(c.max(0) - c.min(0))) if c.shape[0] else 0
    m = max(6, ext // 2 + 2)
    a = np.maximum(c.min(0) - m, 0)
    b = np.minimum(c.max(0) + m + 1, [X, Y, Z])
    sub = np.nan_to_num(np.nanmax(ct_data[a[0]:b[0], a[1]:b[1], a[2]:b[2], :], axis=-1))
    s = tuple(int(v) for v in (np.asarray(seed, dtype=int) - a))
    peak = float(sub[s])
    if not np.isfinite(peak) or peak <= 0:
        return c

    otsu = _otsu(sub) or (0.30 * peak)
    lo = max(0.80 * otsu, lo_frac * peak)             # membrane (captures dim margins)
    hi = max(1.30 * otsu, 0.50 * peak)                # definite-vessel anchor

    def _hysteresis(lo_t, hi_t):
        low = sub >= lo_t
        lbl, _ = ndimage.label(low)
        seed_lab = lbl[s]
        if seed_lab == 0:                             # seed below lo → just the seed
            g = np.zeros_like(low); g[s] = True; return g
        comp = lbl == seed_lab
        return comp if (sub[comp] >= hi_t).any() else np.zeros_like(low)

    grown = _hysteresis(lo, hi)
    # size guard: only a near-total flood (degenerate Otsu / a bright artefact bridged
    # in at the low threshold) trips this — a real vessel can't fill the padded window.
    for bump in (0.35, 0.50, 0.65):
        if grown.sum() <= 0.80 * grown.size:
            break
        grown = _hysteresis(max(lo, bump * peak), hi)

    for z in range(grown.shape[2]):                   # close pinholes + fill per slice
        grown[:, :, z] = ndimage.binary_fill_holes(
            ndimage.binary_closing(grown[:, :, z], iterations=2))
    out = np.argwhere(grown) + a
    return out if out.shape[0] else c


def _mask_overlay_png(dce_slice, mask2d, flip_lr=True, flip_ud=False) -> str:
    """Anatomy slice with the vessel mask baked in as a theme-colour overlay (both
    display-oriented) → base64 PNG. The mask is **upsampled and smoothed** so the
    outline reads as a clean sub-voxel curve tracing the vessel, not a blocky staircase
    of the acquisition voxels — a translucent fill plus a darker theme-deep contour."""
    from scipy import ndimage

    from pbrain._palette import palette
    a = _disp(dce_slice, flip_lr, flip_ud).astype(float)
    m = _disp(mask2d, flip_lr, flip_ud).astype(float)
    fin = a[np.isfinite(a)]
    lo, hi = (np.percentile(fin, 1), np.percentile(fin, 99.5)) if fin.size else (0.0, 1.0)
    g = np.clip((np.nan_to_num(a) - lo) / (hi - lo + 1e-9), 0, 1)

    up = max(2, min(6, round(768 / max(a.shape))))     # ~768px long side, capped
    g = ndimage.zoom(g, up, order=1)                   # bilinear anatomy (no staircase)
    mm = ndimage.zoom(m, up, order=1)
    mm = ndimage.gaussian_filter(mm, sigma=up * 0.55)  # round the voxel corners…
    M = mm > 0.5                                        # …then re-threshold → smooth edge
    edge = M & ~ndimage.binary_erosion(M, iterations=max(2, up // 3))

    base_hex = palette()[0].lstrip("#")
    deep_hex = palette()[1].lstrip("#")
    br, bg, bb = (int(base_hex[i:i + 2], 16) / 255 for i in (0, 2, 4))
    dr, dg, db = (int(deep_hex[i:i + 2], 16) / 255 for i in (0, 2, 4))
    rgba = np.stack([g, g, g, np.ones_like(g)], axis=-1)
    al = 0.34
    for k, col in enumerate((br, bg, bb)):
        rgba[M, k] = (1 - al) * g[M] + al * col        # translucent theme-base fill
    for k, col in enumerate((dr, dg, db)):
        rgba[edge, k] = col                            # darker theme-deep contour
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(rgba, aspect="auto", interpolation="bilinear")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _vessel_roi(coords, ct_data, dce_signal, pf, X, Y, flip_lr=True, flip_ud=False):
    """Per-vessel ROI: grow the vessel from the max voxel, bake the exact shape
    into each slice it spans, and give the max voxel + its slice."""
    c = np.asarray(coords, dtype=int)
    if c.ndim != 2 or c.shape[0] == 0:
        return None
    pk = np.nanmax(ct_data[c[:, 0], c[:, 1], c[:, 2], :], axis=1)
    mx, my, mz = c[int(np.nanargmax(pk))] if np.isfinite(pk).any() else c[0]
    grown = _grow_vessel(ct_data, (mx, my, mz), c)
    slices = {}
    for z in np.unique(grown[:, 2]):
        m2 = np.zeros((X, Y), dtype=bool)
        zc = grown[grown[:, 2] == z]
        m2[zc[:, 0], zc[:, 1]] = True
        slices[str(int(z))] = {"png": _mask_overlay_png(dce_signal[:, :, z, pf], m2, flip_lr, flip_ud)}
    # motion-corrected curve: the per-frame brightest voxel WITHIN the vessel (so the
    # curve follows the bolus peak as the vessel moves, vs a single fixed voxel).
    vol = ct_data[grown[:, 0], grown[:, 1], grown[:, 2], :]
    adaptive = np.nanmax(vol, axis=0) if vol.size else np.full(ct_data.shape[-1], np.nan)
    return {"slices": slices, "max": _vox_to_disp(int(mx), int(my), X, Y, flip_lr, flip_ud),
            "max_slice": int(mz), "n": int(grown.shape[0]), "adaptive": _finite_list(adaptive)}


def build_aif_payload(config, ifn, dce_signal, ct_data, t_s, affine=None) -> dict:
    mask = np.asarray(ifn.mask, dtype=bool)
    ca = np.asarray(ifn.c_a, dtype=float)
    X, Y, Z = mask.shape
    flr, fud = _orient_flips(affine)             # neurological display from the affine
    src = ifn.source or "aif"
    cand_curves = (ifn.meta or {}).get("candidate_curves") or {}
    cand_masks = (ifn.meta or {}).get("candidate_masks") or {}
    vessels = [src] + [v for v in cand_curves if v != src]
    curves = {f"{v}|max": _finite_list(cv) for v, cv in cand_curves.items()}
    curves[f"{src}|max"] = _finite_list(ca)      # the curve actually fed downstream

    pf = int(np.nanargmax(ca)) if np.isfinite(ca).any() else 0
    pf = min(pf, dce_signal.shape[-1] - 1)
    # per-vessel ROI: grow the vessel from its max voxel + bake the shape per slice
    rois = {}
    for v in vessels:
        coords = cand_masks.get(v) or (np.argwhere(mask).tolist() if v == src else None)
        r = _vessel_roi(coords, ct_data, dce_signal, pf, X, Y, flr, fud) if coords else None
        if r is not None:
            rois[v] = r

    for v, r in rois.items():                    # motion-corrected curve per vessel
        if r.get("adaptive"):
            curves[f"{v}|adaptive"] = r["adaptive"]
    stats = ["max", "adaptive"] if any("adaptive" in r for r in rois.values()) else ["max"]
    zi = int(rois.get(src, {}).get("max_slice", Z // 2))
    pngs = {str(z): png_data_uri(_disp(dce_signal[:, :, z, pf], flr, fud)) for z in range(Z)}

    sl_mask = mask[:, :, zi]
    roi_c = ct_data[:, :, zi, :][sl_mask] if sl_mask.any() else np.empty((0, ca.size))
    if roi_c.shape[0] > 40:
        roi_c = roi_c[np.linspace(0, roi_c.shape[0] - 1, 40).astype(int)]
    return {
        "subject": getattr(config, "subject_id", ""), "checkpoint": "aif",
        "mode": config.mode, "t_s": _finite_list(t_s),
        "vessels": vessels, "stats": stats,
        "selected": {"vessel": src, "stat": "max"},
        "boluses": _detect_boluses(ca, t_s),
        "curves": curves,
        "roi_curves": {src: [_finite_list(c) for c in roi_c]},
        "slice": {"n_slices": int(Z), "idx": zi, "pngs": pngs},
        "rois": rois,
    }


def apply_aif_result(config, ifn, ct_data, result, affine=None, curves=None) -> "object":
    """Fold the review result back into the InputFunction, or abort. Handles three
    outcomes: rejected → abort; a drawn polygon (manual) → curve from that ROI; a
    moved/placed max voxel → curve from that voxel; plain confirm → unchanged."""
    if not result or not result.get("accepted", False):
        raise CheckpointAbort("AIF review rejected")
    mask = np.asarray(ifn.mask, dtype=bool)
    X, Y, Z = mask.shape
    flr, fud = _orient_flips(affine)                  # match the display orientation
    zi = max(0, min(int(result.get("slice", 0)), Z - 1))
    meta = dict(ifn.meta)
    meta["review"] = {"mode": config.mode, "vessel": result.get("vessel"),
                      "stat": result.get("stat")}

    poly = result.get("polygon")
    if poly and len(poly) >= 3:                       # manual: user drew the ROI
        m2 = _polygon_to_mask(poly, X, Y, flr, fud)
        m3 = np.zeros((X, Y, Z), dtype=bool)
        m3[:, :, zi] = m2
        sub = ct_data[m3]
        if not sub.size:
            raise CheckpointAbort("drawn AIF ROI is empty")
        ca = np.asarray(sub[int(np.nanargmax(np.nanmax(sub, axis=1)))], dtype=float)
        meta["review"]["roi_voxels"] = int(m3.sum())
        meta["review"]["vessel"] = "custom"
        return replace(ifn, c_a=ca, mask=m3, source="custom", meta=meta)

    mv = result.get("max")
    if mv and len(mv) == 2:                           # a moved/placed max voxel
        xi, yi = _disp_to_vox(mv[0], mv[1], X, Y, flr, fud)
        ca = np.asarray(ct_data[xi, yi, zi, :], dtype=float)
        m3 = np.zeros((X, Y, Z), dtype=bool)
        m3[xi, yi, zi] = True
        meta["review"]["max_voxel"] = [xi, yi, zi]
        return replace(ifn, c_a=ca, mask=m3,
                       source=(result.get("vessel") or ifn.source), meta=meta)

    # vessel / motion-correction selection → the chosen curve from the payload
    vessel, stat = result.get("vessel"), result.get("stat", "max")
    if curves and vessel and not (vessel == ifn.source and stat == "max"):
        cv = curves.get(f"{vessel}|{stat}")
        if cv is not None:
            meta["review"]["curve"] = f"{vessel}|{stat}"
            arr = np.asarray([np.nan if x is None else x for x in cv], dtype=float)
            return replace(ifn, c_a=arr, source=vessel, meta=meta)
    return replace(ifn, meta=meta)                    # plain confirm — unchanged


# ── Generic plug-in review (models, and any plug-in with a review()) ─────────

def _figure_to_uri(fig) -> str:
    """A matplotlib Figure → base64 PNG data URI (dark, to match the page)."""
    import base64
    import io
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0b0c0e")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _sanitize_panels(panels) -> list[dict]:
    """Make a declarative spec's panels JSON-safe: arrays → finite lists, an image
    panel's raw ``array`` → a PNG data URI (rotated like the DCE display)."""
    out = []
    for p in panels or []:
        p = dict(p)
        p["series"] = [dict(s, x=_finite_list(s.get("x", [])), y=_finite_list(s.get("y", [])))
                       for s in p.get("series", [])]
        if p.get("kind") == "image" and "array" in p:
            p["png"] = png_data_uri(_disp(p.pop("array")))
        out.append(p)
    return out


def spec_to_payload(spec, *, checkpoint: str, title: str, config) -> dict:
    """A plug-in's :meth:`review` return → a review payload. Accepts a matplotlib
    Figure (shown as an image) or a declarative spec dict (rendered as interactive
    panels) — the hybrid contract."""
    payload = {"subject": getattr(config, "subject_id", ""), "checkpoint": checkpoint,
               "mode": config.mode, "title": title}
    if hasattr(spec, "savefig"):                      # a matplotlib Figure
        payload["figure"] = _figure_to_uri(spec)
    elif isinstance(spec, dict):
        payload["title"] = spec.get("title", title)
        payload["panels"] = _sanitize_panels(spec.get("panels", []))
        if spec.get("controls"):                      # manual-mode editable fit params
            payload["controls"] = _sanitize_controls(spec["controls"])
        if "figure" in spec:                          # a spec may also embed a figure
            payload["figure"] = (_figure_to_uri(spec["figure"])
                                 if hasattr(spec["figure"], "savefig") else spec["figure"])
    return payload


def _sanitize_controls(controls) -> list[dict]:
    """JSON-safe editable-control declarations (numeric bounds → float)."""
    out = []
    for c in controls or []:
        c = dict(c)
        v = c.get("value")
        if isinstance(v, bool):
            pass
        elif isinstance(v, (int, float)):
            c["value"] = float(v)
        for k in ("min", "max", "step"):
            if c.get(k) is not None:
                c[k] = float(c[k])
        out.append(c)
    return out


def _coerce_params(params, controls=None) -> dict:
    """Browser control values arrive as strings — coerce back to number/bool/str,
    guided by the declarations (a control with min/max/step is numeric)."""
    numeric = {c.get("key") for c in (controls or [])
               if any(k in c for k in ("min", "max", "step"))}
    out = {}
    for k, v in (params or {}).items():
        if not isinstance(v, str):
            out[k] = v
            continue
        s = v.strip()
        if k in numeric:
            try:
                out[k] = float(s)
                continue
            except ValueError:
                pass
        if s.lower() in ("true", "false"):
            out[k] = (s.lower() == "true")
        else:
            out[k] = s
    return out


def review_checkpoint(config, plugin, *, checkpoint: str, title: str, args=(), kwargs=None):
    """Generic checkpoint for any plug-in that declares ``review(...) -> spec|fig|None``.
    No-op in auto/headless, or when the plug-in has no review() or returns None.
    The user confirms (proceed) or rejects (→ :class:`CheckpointAbort`)."""
    if not active(config.mode):
        return
    review = getattr(plugin, "review", None)
    if review is None:
        return
    try:
        spec = review(*args, **(kwargs or {}))
    except Exception as exc:                           # a broken review() must not kill a run
        _log.info("%s review() errored (%s) — skipping", title, str(exc)[:100])
        return
    if spec is None:                                   # the plug-in opted out
        return
    from pbrain import _webreview
    _log.info("%s: %s review — opening browser…", checkpoint, config.mode)
    payload = spec_to_payload(spec, checkpoint=checkpoint, title=title, config=config)
    result = _webreview.review(payload, timeout=_REVIEW_TIMEOUT_S)
    if result is None:
        _log.info("%s: review timed out — proceeding", checkpoint)
        return
    if not result.get("accepted", False):
        raise CheckpointAbort(f"{title} review rejected")


def model_checkpoint(config, model_key, model, inputs, result):
    """Per-model verification in the kinetic stage. Returns the ``ModelResult`` to use
    downstream: the **original** on confirm / auto / no-review / timeout, or a
    **re-fitted** one when the user edits parameters in ``--mode manual`` (the review's
    declared ``controls``). Reject → :class:`CheckpointAbort`. Only models that ran and
    define a ``review()`` are shown; the model decides what to plot."""
    if not active(config.mode):
        return result
    review = getattr(model, "review", None)
    if review is None:
        return result
    try:
        spec = review(inputs, result)
    except Exception as exc:                           # a broken review() must not kill a run
        _log.info("%s review() errored (%s) — skipping", model_key, str(exc)[:100])
        return result
    if spec is None:                                   # the model opted out
        return result
    from pbrain import _webreview
    _log.info("model: %s review — opening browser…", config.mode)
    payload = spec_to_payload(spec, checkpoint="model", title=model_key, config=config)
    out = _webreview.review(payload, timeout=_REVIEW_TIMEOUT_S)
    if out is None:
        _log.info("model: review timed out — proceeding")
        return result
    if not out.get("accepted", False):
        raise CheckpointAbort(f"{model_key} review rejected")

    params = out.get("params") or {}
    if params and config.mode == "manual":
        controls = spec.get("controls") if isinstance(spec, dict) else None
        kw = _coerce_params(params, controls)
        try:
            refit = model.fit(inputs, **kw)
        except Exception as exc:
            _log.warning("kinetic: %s manual re-fit failed (%s) — keeping original fit",
                         model_key, str(exc)[:100])
            return result
        from pbrain.models.base import ModelResult
        aux = dict(getattr(refit, "aux", {}) or {})
        aux["manual_params"] = dict(kw)                # recorded so the run stays reproducible
        opt_hint = " ".join(f"--opt models.{model_key}.{k}={v}" for k, v in kw.items())
        _log.info("kinetic: %s re-fitted with manual params %s · reproduce with %s",
                  model_key, kw, opt_hint)
        return ModelResult(maps=refit.maps,
                           units=getattr(refit, "units", result.units), aux=aux)
    return result


# ── Baseline checkpoint (signal_to_conc) ─────────────────────────────────────

def build_baseline_payload(config, signal, *, method="auto", fallback=10, skip=3,
                           height_frac=0.5, t_s=None) -> dict | None:
    """Focused payload for the baseline review: the mean high-signal curve, the
    detected baseline point + bolus onset, and a **zoom window** around the first
    peak (no point showing the flat 10-minute tail). The page draws a draggable
    circle ON the curve at the baseline point."""
    from pbrain.signal_to_conc.baseline import _mean_high_signal_curve, resolve_baseline_frames
    S = np.asarray(signal, dtype=float)
    curve = _mean_high_signal_curve(S, skip=skip)
    if curve is None:
        return None
    curve = np.asarray(curve, dtype=float)
    T = curve.size
    # the interactive default is the gradient/walk-back finder (Butterworth + np.gradient):
    # the last baseline frame *before* the bolus rise, which is what the reviewer wants
    # to confirm. `auto`/unset upgrades to it; an explicit method is honoured.
    method = "gradient" if method in (None, "auto") else method
    nbf = min(int(resolve_baseline_frames(S, method=method, fallback=fallback)), T - 1)
    t = [float(v) for v in (np.asarray(t_s, dtype=float)[:T] if t_s is not None else np.arange(T))]
    mx = float(np.nanmax(curve))
    base0 = float(np.nanmean(curve[:max(2, skip)]))
    thr = base0 + height_frac * (mx - base0)
    above = np.flatnonzero(curve >= thr)
    onset = int(above[0]) if above.size else nbf
    zmax = min(int(onset * 2.2) + 4, T - 1)          # show pre-bolus + first peak only
    return {
        "subject": getattr(config, "subject_id", ""), "checkpoint": "baseline",
        "mode": config.mode, "title": "baseline",
        "t": t, "curve": _finite_list(curve),
        "baseline_frame": int(nbf), "onset_frame": int(onset), "n_frames": int(T),
        "xlim": [t[0], t[zmax]], "method": method,
    }


def baseline_checkpoint(config, signal, *, method="auto", fallback=10, skip=3,
                        height_frac=0.5, t_s=None):
    """Interactive baseline review (--mode verify/manual): confirm or drag the
    baseline point. Returns the chosen frame count (or None = keep the detected /
    auto value). No-op in auto/headless; reject → :class:`CheckpointAbort`."""
    if not active(config.mode):
        return None
    payload = build_baseline_payload(config, signal, method=method, fallback=fallback,
                                     skip=skip, height_frac=height_frac, t_s=t_s)
    if payload is None:
        return None
    from pbrain import _webreview
    _log.info("signal_to_conc: %s baseline review — opening browser…", config.mode)
    result = _webreview.review(payload, timeout=_REVIEW_TIMEOUT_S)
    if result is None:
        return None
    if not result.get("accepted", False):
        raise CheckpointAbort("baseline review rejected")
    bf = result.get("baseline_frame")
    return int(bf) if bf is not None else None


def aif_checkpoint(config, ifn, dce_signal, ct_data, t_s, affine=None):
    """Entry point the AIF stage calls. No-op unless a checkpoint is active. The
    affine orients the display neurologically (patient-left on the viewer's left)."""
    if not active(config.mode):
        return ifn
    from pbrain import _webreview
    _log.info("aif: %s review — opening browser…", config.mode)
    payload = build_aif_payload(config, ifn, dce_signal, ct_data, t_s, affine)
    result = _webreview.review(payload, timeout=_REVIEW_TIMEOUT_S)
    if result is None:                                # browser closed / never answered
        _log.info("aif: review timed out — keeping the suggested AIF")
        return ifn
    return apply_aif_result(config, ifn, ct_data, result, affine, payload.get("curves"))


# ── Tissue-ROI checkpoint (tissue_roi) ───────────────────────────────────────

def _hex_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _label_colors(label_ids) -> dict[int, str]:
    """A distinct-but-cohesive colour per region: shades spread in a **narrow hue band
    around the theme accent** (with varied saturation/brightness) so the parcellation
    reads as the user's theme rather than a rainbow — hover gives the exact region."""
    import colorsys

    from pbrain._palette import palette
    ah, _as, _av = colorsys.rgb_to_hsv(*_hex_rgb01(palette()[0]))
    labs = sorted(int(l) for l in set(int(x) for x in label_ids) if int(l) != 0)
    g = 0.61803398875
    out = {}
    for i, l in enumerate(labs):
        f = (i * g) % 1.0
        hue = (ah + (f - 0.5) * 0.17) % 1.0            # ±~30° around the accent
        sat = 0.40 + 0.42 * ((i * 0.37) % 1.0)
        val = 0.58 + 0.38 * (((i + 1) * 0.29) % 1.0)
        r, gr, b = colorsys.hsv_to_rgb(hue, sat, val)
        out[l] = "#%02x%02x%02x" % (int(r * 255), int(gr * 255), int(b * 255))
    return out


def _label_overlay_png(dce_slice, label_slice, colors, flip_lr=True, flip_ud=False) -> str:
    """Anatomy slice with each parcellation region painted its own theme shade
    (translucent) + thin darker boundaries between regions → base64 PNG. Labels are
    nearest-upsampled so region edges stay crisp (no colour bleeding across regions)."""
    from scipy import ndimage

    from pbrain._palette import palette
    a = _disp(dce_slice, flip_lr, flip_ud).astype(float)
    L = _disp(label_slice, flip_lr, flip_ud).astype(int)
    fin = a[np.isfinite(a)]
    lo, hi = (np.percentile(fin, 1), np.percentile(fin, 99.5)) if fin.size else (0.0, 1.0)
    g = np.clip((np.nan_to_num(a) - lo) / (hi - lo + 1e-9), 0, 1)
    up = max(2, min(5, round(640 / max(a.shape))))
    gU = ndimage.zoom(g, up, order=1)                  # smooth anatomy
    LU = ndimage.zoom(L, up, order=0)                  # crisp labels (nearest)
    rgba = np.stack([gU, gU, gU, np.ones_like(gU)], axis=-1)
    al = 0.5
    for l, hexc in colors.items():
        m = LU == l
        if not m.any():
            continue
        for k, c in enumerate(_hex_rgb01(hexc)):
            rgba[m, k] = (1 - al) * gU[m] + al * c
    dr, dg, db = _hex_rgb01(palette()[1])              # region boundaries
    edge = (LU != 0) & ((LU != np.roll(LU, 1, 0)) | (LU != np.roll(LU, 1, 1)))
    for k, c in enumerate((dr, dg, db)):
        rgba[edge, k] = c
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(rgba, aspect="auto", interpolation="bilinear")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_tissue_payload(config, parc, dce_baseline, labels, affine=None) -> dict | None:
    """Bake the tissue segmentation for review: **every region in its own theme shade**
    on the DCE baseline, one PNG per slice so the slider scrolls the volume, plus a
    per-slice **label grid** + region names so the page can name the region under the
    cursor on hover. Reports the region + voxel tally so a bad segmentation (skull
    leak, whole-brain fallback) is obvious."""
    from scipy import ndimage
    parc = np.asarray(parc)
    if parc.ndim != 3:
        return None
    X, Y, Z = parc.shape
    flr, fud = _orient_flips(affine)
    tissue = parc > 0
    per_slice = tissue.sum(axis=(0, 1))
    base = np.asarray(dce_baseline, dtype=float)
    if base.shape[:3] != (X, Y, Z):                   # baseline off-grid → grey backdrop
        base = np.zeros((X, Y, Z), dtype=float)
    uniq = [int(l) for l in np.unique(parc) if int(l) != 0]
    colors = _label_colors(uniq)

    grid_up = min(1.0, 110.0 / max(X, Y))             # cap the hover grid (~110 px/side)
    slices, grids = {}, {}
    for z in range(Z):
        slices[str(z)] = (_label_overlay_png(base[:, :, z], parc[:, :, z], colors, flr, fud)
                          if tissue[:, :, z].any()
                          else png_data_uri(_disp(base[:, :, z], flr, fud)))
        Ld = _disp(parc[:, :, z], flr, fud).astype(np.int32)   # display-oriented (matches PNG)
        if grid_up < 1.0:
            Ld = ndimage.zoom(Ld, grid_up, order=0)
        grids[str(z)] = Ld.astype(int).tolist()
    names = {int(l): str((labels or {}).get(int(l), f"label {int(l)}")) for l in uniq}
    return {
        "subject": getattr(config, "subject_id", ""), "checkpoint": "tissue",
        "mode": config.mode, "title": "tissue segmentation",
        "n_slices": int(Z), "idx": int(np.argmax(per_slice)) if per_slice.any() else Z // 2,
        "slices": slices, "label_grid": grids, "label_names": names,
        "colors": {str(k): v for k, v in colors.items()},
        "n_voxels": int(tissue.sum()), "n_regions": len(uniq),
        "regions": list(names.values())[:40],
    }


def apply_tissue_result(config, parc, result, affine=None):
    """Fold the tissue review back in. Reject → abort. In manual mode, each drawn
    **exclusion** polygon removes its voxels (sets label 0) on that slice — a quick
    way to cut an artefact or a mis-segmented region out of the tissue mask."""
    if not result or not result.get("accepted", False):
        raise CheckpointAbort("tissue segmentation rejected")
    parc = np.asarray(parc).copy()
    X, Y, Z = parc.shape
    flr, fud = _orient_flips(affine)
    removed = 0
    for ex in (result.get("exclusions") or []):
        z = int(ex.get("slice", -1))
        poly = ex.get("polygon")
        if not (0 <= z < Z) or not poly or len(poly) < 3:
            continue
        m2 = _polygon_to_mask(poly, X, Y, flr, fud)
        sl = parc[:, :, z]
        removed += int((sl[m2] != 0).sum())
        sl[m2] = 0
        parc[:, :, z] = sl
    if removed:
        _log.info("tissue_roi: manual review excluded %d voxels from the tissue mask", removed)
    return parc


def tissue_checkpoint(config, parc, dce_baseline, labels, affine=None):
    """Entry point the tissue_roi stage calls. Returns the parcellation to use — the
    original on confirm / auto / timeout, or one with manual exclusions applied. The
    affine orients the display neurologically; reject → :class:`CheckpointAbort`."""
    if not active(config.mode):
        return parc
    payload = build_tissue_payload(config, parc, dce_baseline, labels, affine)
    if payload is None:
        return parc
    from pbrain import _webreview
    _log.info("tissue_roi: %s review — opening browser…", config.mode)
    result = _webreview.review(payload, timeout=_REVIEW_TIMEOUT_S)
    if result is None:
        _log.info("tissue_roi: review timed out — keeping the segmentation")
        return parc
    return apply_tissue_result(config, parc, result, affine)


# ── Diffusion checkpoint (diffusion stage) ───────────────────────────────────

def _diffusion_summary_spec(model, result, brain_mask=None) -> dict | None:
    """A generic map summary for any diffusion model: the median scalar values (over
    the brain) + the primary map's most-populated central slice. Covers dti/dki/noddi/…
    with no per-model code; a model may still define its own ``review()`` to override."""
    maps = getattr(result, "maps", None) or {}
    if not maps:
        return None
    units = getattr(model, "units", {}) or {}
    bm = (np.asarray(brain_mask, dtype=bool) if brain_mask is not None else None)

    def _med(a):
        a = np.asarray(a, dtype=float)
        m = np.isfinite(a)
        if bm is not None and bm.shape == a.shape:
            m &= bm
        v = a[m]
        v = v[v != 0]
        return float(np.median(v)) if v.size else None

    keys = getattr(model, "outputs", None) or tuple(maps.keys())
    items = {}
    for k in keys:
        if k in maps:
            mv = _med(maps[k])
            items[f"{k} (median)"] = "—" if mv is None else f"{mv:.4g} {units.get(k, '')}".strip()

    panels = [{"kind": "values", "title": "scalar medians (brain)", "items": items}]
    prim = getattr(model, "primary_map", None) or (keys[0] if keys else None)
    arr = maps.get(prim) if prim else None
    if arr is not None:
        a = np.asarray(arr, dtype=float)
        if a.ndim == 3 and a.shape[2] >= 1:
            fin = np.isfinite(a) & (a != 0)
            counts = fin.reshape(-1, a.shape[2]).sum(0)
            z = int(np.argmax(counts)) if counts.any() else a.shape[2] // 2
            panels.insert(0, {"kind": "image", "title": f"{prim} · slice {z}", "array": a[:, :, z]})
    return {"title": f"{getattr(model, 'name', 'diffusion')}", "panels": panels}


def diffusion_checkpoint(config, model_key, model, inputs, result, *,
                         brain_mask=None, affine=None):
    """Per-model diffusion verification (--mode verify/manual): the model's own
    ``review()`` if it defines one, else a generic map summary (median scalars + the
    primary map's central slice). View + confirm/reject — diffusion maps aren't
    hand-edited. No-op in auto/headless; reject → :class:`CheckpointAbort`."""
    if not active(config.mode):
        return
    review = getattr(model, "review", None)
    spec = None
    if review is not None:
        try:
            spec = review(inputs, result)
        except Exception as exc:
            _log.info("%s review() errored (%s) — using the map summary", model_key, str(exc)[:80])
            spec = None
    if spec is None:
        spec = _diffusion_summary_spec(model, result, brain_mask)
    if spec is None:
        return
    from pbrain import _webreview
    _log.info("diffusion: %s review — opening browser…", config.mode)
    payload = spec_to_payload(spec, checkpoint="model", title=model_key, config=config)
    out = _webreview.review(payload, timeout=_REVIEW_TIMEOUT_S)
    if out is None:
        _log.info("diffusion: review timed out — proceeding")
        return
    if not out.get("accepted", False):
        raise CheckpointAbort(f"{model_key} diffusion review rejected")
