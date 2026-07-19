"""Vision-assisted AIF/VIF localisation — optional, coarse, CNN-cross-checked.

The idea (validated on real data): a DCE series' *temporal max-projection* is an
angiographic-like map where the AIF vessels are the brightest structures. A
capable vision-language model, given that correctly-oriented map, can propose
which slice and coarse region hold the **superior sagittal sinus** and the
**right / left internal carotid** — and *only* those, never the transverse /
sigmoid sinus. Classic image processing then refines each proposal to the peak
voxel (``auto_vessel``-style). The model never touches the numbers; it points, the
deterministic code measures.

Vision backend is HuggingFace (mlx-vlm on Apple Silicon, transformers elsewhere),
loaded lazily. If it isn't installed, ``find_aif`` returns None and the caller
falls back to p-Brain's existing extractors — this module is purely additive.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

TARGETS = ("sss", "r_ica", "l_ica")
_LABEL = {"sss": "superior sagittal sinus", "r_ica": "right internal carotid",
          "l_ica": "left internal carotid"}


# ---------------------------------------------------------------- projections
def load_canonical(conc_path: str | Path):
    """Load a 4-D concentration (or signal) volume in canonical RAS+ orientation,
    so 'anterior', 'posterior', 'left', 'right' mean what they say."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(str(conc_path)))
    return np.asarray(img.dataobj, dtype=np.float32), np.asarray(img.affine, dtype=float)


def max_projection(vol4d: np.ndarray) -> np.ndarray:
    """Peak enhancement per voxel over time (X,Y,Z) — vessels are the brightest.
    NaN-hardened: all-NaN voxels (masked background, division artefacts) collapse to
    0 rather than propagating NaN and warning from ``np.nanmax``."""
    finite = np.where(np.isfinite(vol4d), vol4d, -np.inf)
    mx = np.asarray(finite.max(axis=3), dtype=np.float32)
    mx[~np.isfinite(mx)] = 0.0          # voxels that were all-NaN
    return mx


def canonical_glob(root: str | Path, name: str) -> str | None:
    """First ``name`` under ``root``, skipping backup/hidden derivative trees.

    A naive ``**`` glob ranks ``derivatives.jun24_bak/…`` *before* ``derivatives/…``
    ('.' 0x2E < '/' 0x2F in ASCII), silently picking a stale backup. Drop any path
    whose components look like a backup (``*bak*``), a dotfile, or a sibling of
    ``derivatives`` (``derivatives.*``), then choose the shortest path deterministically."""
    import glob as _glob
    hits = _glob.glob(str(Path(root) / "**" / name), recursive=True)

    def clean(p: str) -> bool:
        return not any(s.startswith(".") or "bak" in s.lower()
                       or (s.startswith("derivatives") and s != "derivatives")
                       for s in Path(p).parts)

    good = [p for p in hits if clean(p)] or hits
    return sorted(good, key=lambda p: (len(Path(p).parts), p))[0] if good else None


def _axial_anterior_up(sl: np.ndarray) -> np.ndarray:
    """RAS+ slice [i=L→R, j=P→A] → display image with anterior at TOP, patient
    left on the left (neurological)."""
    return np.flipud(sl.T)


def montage_png(mx: np.ndarray, out: str | Path, *, cols: int = 5, tile: int = 200) -> Path:
    """A labelled contact sheet of the anterior-up max-projection — one image for
    the VLM, slice indices burned in so it can name the slice.

    ``tile`` is the per-slice side in pixels; keep the whole sheet's long side near
    ~1000 px. Vision transformers (notably the 4-bit Qwen-VL) exceed their patch
    budget on oversized inputs and emit garbage instead of downscaling, so the
    contact sheet is rendered small on purpose — the VLM only needs a coarse region
    and slice number; ``refine_peak`` does the precise, full-resolution measurement."""
    from PIL import Image, ImageDraw
    X, Y, Z = mx.shape
    lo, hi = np.nanpercentile(mx, 1), np.nanpercentile(mx, 99.5)
    n = np.nan_to_num(np.clip((mx - lo) / (hi - lo + 1e-6), 0, 1), nan=0.0)
    rows = (Z + cols - 1) // cols
    pad = 6
    sheet = Image.new("RGB", (cols * tile + (cols + 1) * pad, rows * tile + (rows + 1) * pad), (10, 10, 12))
    draw = ImageDraw.Draw(sheet)
    for z in range(Z):
        g = (_axial_anterior_up(n[:, :, z]) * 255).astype(np.uint8)
        im = Image.fromarray(np.stack([g, g, g], -1)).resize((tile, tile), Image.LANCZOS)
        cx = (z % cols) * (tile + pad) + pad
        cy = (z // cols) * (tile + pad) + pad
        sheet.paste(im, (cx, cy))
        draw.text((cx + 4, cy + 3), str(z), fill=(255, 180, 90))
    out = Path(out)
    sheet.save(out)
    return out


# ---------------------------------------------------------------- vision backend
_PROMPT = (
    "This is a contact sheet of axial slices from a temporal MAX-projection of a "
    "brain DCE-MRI (the brightest structures are blood vessels filled with contrast). "
    "Each tile is one slice, numbered in its top-left corner. In every tile ANTERIOR "
    "is at the TOP, POSTERIOR at the bottom, patient-LEFT on the left.\n"
    "The slices are ordered INFERIOR to SUPERIOR: the LOWEST-numbered slices are at the "
    "skull base — you can see the dark EYE ORBITS at the front (top) — while the "
    "HIGHEST-numbered slices are near the top of the head (vertex), rounder, no orbits.\n"
    "Locate ONLY these three vessels, each on the single slice where it is clearest:\n"
    "- sss: the superior sagittal sinus — a bright dot on the POSTERIOR MIDLINE, present "
    "on almost every slice; pick one where it is bright and well-separated.\n"
    "- r_ica / l_ica: the right and left internal carotid arteries — bright round spots "
    "ANTERIOR and to each side of the midline, on the LOWER-NUMBERED skull-base slices "
    "WHERE THE EYE ORBITS ARE VISIBLE (not on the upper vertex slices).\n"
    "Do NOT report the transverse sinus, sigmoid sinus, confluence, veins, or the skull. "
    "If a vessel is not visible, mark found=false.\n"
    'Return ONLY JSON: {"sss":{"slice":<int>,"ap":"anterior|posterior|central",'
    '"lr":"left|right|midline","found":<bool>}, "r_ica":{...}, "l_ica":{...}}'
)

_model_cache: dict = {}


def vlm_available() -> bool:
    try:
        import mlx_vlm  # noqa: F401
        return True
    except Exception:
        try:
            import transformers  # noqa: F401
            return True
        except Exception:
            return False


def _vlm_locate(png: Path, repo: str, prompt: str | None = None) -> dict | None:
    """Send the montage to a HF vision model, parse the JSON localisation."""
    try:
        import mlx_vlm
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        if repo not in _model_cache:
            model, processor = load(repo)
            _model_cache[repo] = (model, processor, load_config(repo))
        model, processor, cfg = _model_cache[repo]
        # cfg must come from load_config, NOT model.config: the processor uses it to
        # expand the image placeholder into the right number of patch tokens; the
        # wrong config silently detaches the image and the model free-associates.
        formatted = apply_chat_template(processor, cfg, prompt or _PROMPT, num_images=1)
        # temperature=0 → greedy, deterministic decoding. p-Brain's hard rule is
        # byte-identical re-runs; pin it explicitly rather than trust the default.
        out = generate(model, processor, formatted, [str(png)], max_tokens=400,
                       temperature=0.0, verbose=False)
        txt = out if isinstance(out, str) else getattr(out, "text", str(out))
    except Exception:
        return None
    try:
        a, b = txt.find("{"), txt.rfind("}")
        return json.loads(txt[a:b + 1])
    except Exception:
        return None


# ---------------------------------------------------------------- refine + validate
def _box(ap: str, lr: str, X: int, Y: int, mid_x: float | None = None) -> tuple[slice, slice]:
    """Coarse region → an (i,j) search window in RAS+ indices (i=L→R, j=P→A).

    ``mid_x`` is the sagittal midline in i-pixels (from the SSS, which runs down the
    midline). The head is often not FOV-centred, so splitting left/right at a fixed
    half-width mislabels the carotids; splitting at the true midline fixes it."""
    m = X / 2.0 if mid_x is None else float(mid_x)
    half = 0.18 * X
    xr = {"left": (0.0, m), "right": (m, float(X)), "midline": (m - half, m + half)}.get(lr, (0.0, float(X)))
    fy = {"posterior": (0.0, 0.5), "anterior": (0.5, 1.0), "central": (0.3, 0.7)}.get(ap, (0.0, 1.0))
    x0, x1 = int(max(0.0, xr[0])), int(min(float(X), xr[1]))
    if x1 <= x0:
        x1 = x0 + 1
    return (slice(x0, x1),
            slice(int(fy[0] * Y), max(int(fy[1] * Y), int(fy[0] * Y) + 1)))


def refine_peak(mx: np.ndarray, z: int, ap: str, lr: str, mid_x: float | None = None) -> dict | None:
    """Within the proposed region on slice z, the peak-enhancement voxel and a
    small high-intensity cluster around it — the deterministic measurement."""
    X, Y, Z = mx.shape
    if not (0 <= z < Z):
        return None
    sx, sy = _box(ap, lr, X, Y, mid_x)
    region = mx[sx, sy, z]
    if region.size == 0 or not np.isfinite(region).any():
        return None
    li, lj = np.unravel_index(int(np.nanargmax(region)), region.shape)
    i, j = sx.start + li, sy.start + lj
    peak = float(mx[i, j, z])
    if not np.isfinite(peak) or peak <= 0.0:
        return None                        # no enhancement in this region → not a vessel
    thr = 0.7 * peak
    cluster = int(((mx[sx, sy, z] >= thr) & np.isfinite(mx[sx, sy, z])).sum())
    return {"voxel": [int(i), int(j), int(z)], "peak": round(peak, 3), "cluster": cluster}


def find_aif(conc_path: str | Path, repo: str, work_dir: str | Path,
             cnn_mask: str | Path | None = None) -> dict | None:
    """Full pass: projections → VLM region proposal → peak refinement per vessel.
    Returns {sss, r_ica, l_ica: {voxel, peak, cluster, region, dist_cnn}} plus
    ``_montage`` and ``_cnn`` (the CNN peak voxel), or None if no VLM.

    ``cnn_mask`` (the CNN's AIF mask) is the cross-check: its peak-enhancement voxel
    is compared to each proposal. The SSS distance is the meaningful one — the CNN
    marks the sinus — so the ICAs are *expected* to sit far from it."""
    if not vlm_available():
        return None
    vol, _ = load_canonical(conc_path)
    mx = max_projection(vol)
    png = montage_png(mx, Path(work_dir) / "dce_maxproj.png")
    loc = _vlm_locate(png, repo)
    if not loc:
        return None
    cnn = cnn_peak_voxel(cnn_mask, mx) if cnn_mask else None
    out: dict = {"_montage": str(png), "_cnn": cnn}
    mid_x: float | None = None                       # sagittal midline, set from the SSS
    for t in TARGETS:                                # TARGETS[0] == "sss", so midline is known before the carotids
        d = loc.get(t) or {}
        if not d.get("found", True) or "slice" not in d:
            out[t] = None
            continue
        r = refine_peak(mx, int(d["slice"]), d.get("ap", ""), d.get("lr", ""),
                        None if t == "sss" else mid_x)
        if r:
            r["region"] = f"{d.get('ap','')}-{d.get('lr','')}"
            r["dist_cnn"] = distance(r["voxel"], cnn)
            if t == "sss":
                mid_x = float(r["voxel"][0])
        out[t] = r
    return out


def cnn_peak_voxel(aif_mask_path: str | Path, mx: np.ndarray | None = None) -> list[int] | None:
    """Reduce the CNN AIF mask to one comparison point in canonical RAS+.

    With the max-projection ``mx``, this is the true peak-enhancement voxel *inside*
    the mask — what 'peak' means, and where an AIF is actually sampled. Without it,
    the mask centroid: a robust single-point summary when the CNN marks a blob
    rather than one voxel (the old code returned an arbitrary corner voxel — fine
    only for the degenerate 1-voxel case, wrong for a region)."""
    import nibabel as nib
    m = np.asarray(nib.as_closest_canonical(nib.load(str(aif_mask_path))).dataobj) > 0
    if not m.any():
        return None
    xs, ys, zs = np.where(m)
    if mx is not None and mx.shape == m.shape:
        vals = np.asarray(mx)[xs, ys, zs]
        if np.isfinite(vals).any():
            k = int(np.nanargmax(vals))
            return [int(xs[k]), int(ys[k]), int(zs[k])]
    return [int(round(float(xs.mean()))), int(round(float(ys.mean()))), int(round(float(zs.mean())))]


def distance(a: list[int] | None, b: list[int] | None) -> float | None:
    if not a or not b:
        return None
    return round(float(np.linalg.norm(np.array(a, float) - np.array(b, float))), 1)


# ---------------------------------------------------------------- temporal gating
# A static max-projection cannot tell a vessel from mucosa: arteries, the venous
# sinus, and nasal/glandular mucosa can all be equally bright. The discriminator is
# *temporal shape* — a vessel shows a bolus (fast wash-in, then wash-out, so it is
# "early-dominated"); mucosa/tissue accumulates contrast slowly and monotonically
# ("late-dominated"). These functions read a candidate region's time-course and
# classify it, so the finder rejects a confidently-bright-but-wrong structure.
_EARLY_LATE_MIN = 1.5    # early-phase peak / late-phase peak below this ⇒ late-dominated ⇒ not a vessel
_TTP_LATE_FRAC = 0.55    # a peak past this fraction of the run is late-enhancing (mucosa-like)


def region_curve(vol4d: np.ndarray, mask3d: np.ndarray, method: str = "max") -> np.ndarray | None:
    """Representative time-course of a region: the brightest-peak voxel (``"max"``,
    default — matches the CNN's ``max_voxel``) or the ROI ``"mean"`` / ``"median"``.
    NaN voxels (failed T1/M0) are dropped so one bad voxel can't poison the curve."""
    tc = np.asarray(vol4d)[np.asarray(mask3d, bool)]
    if tc.size == 0:
        return None
    tc = tc[np.isfinite(tc).all(axis=1)]
    if tc.size == 0:
        return None
    m = method.lower()
    if m == "mean":
        return np.asarray(np.nanmean(tc, axis=0), dtype=np.float32)
    if m == "median":
        return np.asarray(np.median(tc, axis=0), dtype=np.float32)
    return np.asarray(tc[int(np.argmax(np.nanmax(tc, axis=1)))], dtype=np.float32)


def temporal_features(curve: np.ndarray, baseline_frames: int = 5) -> dict | None:
    """Bolus-shape descriptors of a concentration-time curve. ``early_late`` (early
    vs late peak amplitude) and ``ttp_frac`` (peak position) separate a vascular
    bolus from monotonic mucosal accumulation; ``arrival`` is the 10 %-rise frame."""
    c = np.asarray(curve, dtype=float)
    T = c.size
    if T < 6:
        return None
    b = float(np.nanmean(c[:max(1, baseline_frames)]))
    c0 = c - b
    peak = float(np.nanmax(c0))
    if not np.isfinite(peak) or peak <= 0.0:
        return None
    late = float(np.nanmean(c0[int(0.8 * T):]))
    early_max = float(np.nanmax(c0[:max(1, T // 3)]))
    late_max = float(np.nanmax(c0[int(0.6 * T):]))
    denom = max(late_max, 0.05 * peak)          # floor: full wash-out (late≤0) ⇒ high ratio, not a sign flip
    return {
        "ttp": int(np.nanargmax(c0)),
        "ttp_frac": round(float(np.nanargmax(c0)) / T, 3),
        "arrival": int(np.argmax(c0 > 0.1 * peak)),
        "peak": round(peak, 3),
        "washout": round((peak - late) / peak, 3),
        "early_late": round(early_max / (denom + 1e-6), 3),
    }


def is_vessel(features: dict | None) -> bool:
    """True if the curve looks like a vascular bolus (early-dominated, washes out)
    rather than late-accumulating mucosa/tissue — what brightness alone cannot tell."""
    if not features:
        return False
    return features["early_late"] >= _EARLY_LATE_MIN and features["ttp_frac"] <= _TTP_LATE_FRAC


def classify_curve(features: dict | None) -> str:
    """``"vessel"`` (arterial or venous bolus), ``"mucosa/tissue"`` (late-accumulating),
    or ``"none"`` (no enhancement)."""
    if not features:
        return "none"
    return "vessel" if is_vessel(features) else "mucosa/tissue"


def vesselness_map(vol4d: np.ndarray, baseline_frames: int = 5, enhance_pct: float = 85.0) -> np.ndarray:
    """Per-voxel 'is this a vascular bolus?' map from temporal *shape*, not brightness:
    early-phase peak / late-phase peak. High for a vessel (early-dominated bolus), low
    for mucosa/tissue (late accumulation). Feeding THIS to the localiser makes it point
    at true vessels — on a plain max-projection the brightest blob is often late-
    enhancing mucosa, and the model grounds on brightness. Non-enhancing voxels zeroed.

    Validated: a VLM grounding the carotid slice went from 0/203 voxel overlap with the
    CNN ROI on a max-projection to 203/203 on this map."""
    v = np.asarray(vol4d, dtype=np.float32)
    T = v.shape[3]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN voxels are expected
        b = np.nanmean(v[..., :max(1, baseline_frames)], axis=3)
        c0 = v - b[..., None]
        peak = np.nanmax(c0, axis=3)
        early = np.nanmax(c0[..., :max(1, T // 3)], axis=3)
        late = np.nanmax(c0[..., int(0.6 * T):], axis=3)
    thr = np.nanpercentile(peak, enhance_pct) if enhance_pct > 0 else -np.inf
    # Floor the late amplitude at 5 % of the voxel peak: a vessel that washes out to
    # (or below) baseline has late≈0 or negative — that is *most* vessel-like, so the
    # ratio must stay high, not divide by a negative and collapse to 0.
    denom = np.maximum(late, 0.05 * np.maximum(peak, 1e-6))
    vk = np.where(np.isfinite(peak) & (peak > thr), early / (denom + 1e-6), 0.0)
    return np.nan_to_num(np.clip(vk, 0.0, None), nan=0.0).astype(np.float32)


# ---------------------------------------------------------------- region finding (VLM)
# Two-stage, mirroring the CNN: (A) pick the slice from a vesselness contact sheet
# (slice-level is all a tiled montage supports reliably), then (B) ground a bounding
# box on that single slice's vesselness image (single-slice grounding works; contact-
# sheet bbox grounding does not). Grow the region inside the box, then temporal-gate it.
_SLICE_PROMPT = (
    "Contact sheet of axial brain slices; bright = blood vessels (contrast). Tiles are "
    "numbered top-left and ordered INFERIOR (low numbers, eye orbits visible) to SUPERIOR "
    "(high numbers, vertex); ANTERIOR is at the top of each tile. Pick the single tile that "
    "most clearly shows each:\n"
    "- artery: a compact bright blob near the CENTRE of a low-numbered (skull-base) slice.\n"
    "- vein: the bright POSTERIOR-MIDLINE spot, clearest on a higher-numbered slice.\n"
    'Return ONLY JSON: {"artery":{"slice":<int>},"vein":{"slice":<int>}}'
)
_GROUND_PROMPT = (
    "Locate the single brightest compact blob in this brain image. Return ONLY JSON: "
    "{\"bbox_2d\":[x1,y1,x2,y2]} in pixel coordinates of this image."
)
_GROUND_SIZE = 512


def _slice_png(arr2d: np.ndarray, out: str | Path, size: int = _GROUND_SIZE) -> Path:
    """Render one anterior-up slice, per-slice normalised, at ``size`` px for grounding."""
    from PIL import Image
    a = _axial_anterior_up(np.asarray(arr2d, dtype=float))
    a = a / (np.nanmax(a) + 1e-6)
    g = (np.clip(np.nan_to_num(a), 0.0, 1.0) * 255).astype(np.uint8)
    out = Path(out)
    Image.fromarray(np.stack([g, g, g], -1)).resize((size, size), Image.LANCZOS).save(out)
    return out


def _bbox_to_voxel(bbox: list, X: int, Y: int, size: int = _GROUND_SIZE):
    """A grounding ``bbox_2d`` (pixels in a ``size``×``size`` anterior-up slice image) →
    voxel (i,j) index ranges in RAS+ (i=L→R, j=P→A). The display flips j, so the box's
    top/bottom map to high/low j."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    i0, i1 = int(min(x1, x2) / size * X), int(max(x1, x2) / size * X)
    r0, r1 = int(min(y1, y2) / size * Y), int(max(y1, y2) / size * Y)
    j0, j1 = Y - 1 - r1, Y - 1 - r0
    return ((max(0, i0), min(X, max(i0 + 1, i1))), (max(0, j0), min(Y, max(j0 + 1, j1))))


def grow_in_bbox(field: np.ndarray, z: int, irange, jrange, frac: float = 0.5) -> np.ndarray | None:
    """Region inside the box on slice ``z``: voxels ≥ ``frac``×(box max), connected to
    the peak (50 %-of-max + connected-component — the CNN's own U-Net threshold rule)."""
    (i0, i1), (j0, j1) = irange, jrange
    sub = np.asarray(field)[i0:i1, j0:j1, z]
    if sub.size == 0 or not np.isfinite(sub).any() or np.nanmax(sub) <= 0:
        return None
    binm = sub >= frac * float(np.nanmax(sub))
    try:
        from scipy import ndimage
        lab, _ = ndimage.label(binm)
        pk = np.unravel_index(int(np.nanargmax(sub)), sub.shape)
        keep = lab == lab[pk]
    except Exception:
        keep = binm
    full = np.zeros(field.shape, dtype=bool)
    full[i0:i1, j0:j1, z] = keep
    return full


def region_metrics(a: np.ndarray | None, b: np.ndarray | None) -> dict | None:
    """Region-vs-region agreement: voxel counts, centroid distance, overlap flag, IoU."""
    if a is None or b is None:
        return None
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    ca, cb = np.argwhere(a), np.argwhere(b)
    if not len(ca) or not len(cb):
        return None
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return {"vlm_vox": int(a.sum()), "cnn_vox": int(b.sum()),
            "centroid_dist": round(float(np.linalg.norm(ca.mean(0) - cb.mean(0))), 1),
            "overlap": inter > 0, "iou": round(inter / union, 3) if union else 0.0}


def _head_frame(mx: np.ndarray) -> tuple[float, int, int]:
    """Data-derived anatomical reference from the tissue silhouette: sagittal midline
    (i) and anterior–posterior extent (j_min, j_max). Robust to off-centre heads and
    partial FOV — nothing is a fixed fraction of the image."""
    X, Y, Z = mx.shape
    tissue = mx > np.nanpercentile(mx, 60)
    proj = tissue.any(axis=2)
    ii, jj = np.where(proj)
    if ii.size == 0:
        return X / 2.0, 0, Y - 1
    return float(np.median(ii)), int(jj.min()), int(jj.max())


def detect_candidates(vol4d: np.ndarray, vesselness: np.ndarray | None = None,
                      min_voxels: int = 8, vk_pct: float = 96.0) -> list[dict]:
    """Connected-component vessel candidates from the vesselness map. Each carries its
    mask, size, centroid, slice, temporal features and vessel/mucosa class — a small,
    deterministic shortlist (no VLM, no training) to label artery vs vein from."""
    vk = vesselness if vesselness is not None else vesselness_map(vol4d)
    pos = vk[vk > 0]
    if pos.size == 0:
        return []
    thr = max(float(np.nanpercentile(pos, vk_pct)), 1e-6)
    binm = vk >= thr
    try:
        from scipy import ndimage
        lab, n = ndimage.label(binm)
    except Exception:
        lab, n = (binm.astype(int), 1)
    out: list[dict] = []
    for k in range(1, int(n) + 1):
        mask = lab == k
        size = int(mask.sum())
        if size < min_voxels:
            continue
        ci, cj, cz = np.argwhere(mask).mean(axis=0)
        feat = temporal_features(region_curve(vol4d, mask))
        out.append({"mask": mask, "size": size, "centroid": [round(float(ci), 1), round(float(cj), 1), round(float(cz), 1)],
                    "slice": int(round(float(cz))), "features": feat, "klass": classify_curve(feat)})
    return out


def label_vessels(cands: list[dict], mx: np.ndarray) -> dict:
    """Assign artery / vein from vessel candidates by geometry + bolus timing.

    The SSS is uniquely posterior + near-midline + large → found first. The artery is
    the earliest-arriving of the remaining vessels (arteries fill before the sinus).
    Geometry references (midline, A–P extent) are data-derived, not fixed fractions."""
    X, Y, Z = mx.shape
    mid_i, j_min, j_max = _head_frame(mx)
    span = max(1, j_max - j_min)
    vessels = [c for c in cands if c["klass"] == "vessel" and c["features"]]
    if not vessels:
        return {"artery": None, "vein": None}
    big = max(c["size"] for c in vessels)
    T_arr = max((c["features"]["arrival"] for c in vessels), default=1) or 1

    def sss_score(c):
        ci, cj, cz = c["centroid"]
        posteriority = (j_max - cj) / span                      # 1 = most posterior
        midline_prox = 1.0 - min(1.0, abs(ci - mid_i) / (0.5 * span))
        return 0.5 * posteriority + 0.3 * midline_prox + 0.2 * (c["size"] / big)

    vein = max(vessels, key=sss_score)
    rest = [c for c in vessels if c is not vein] or vessels

    def ica_score(c):
        # carotid siphon: skull-base (low z), central (near midline, not peripheral),
        # and early-arriving. Geometry anchors it — 'earliest arrival' alone picks any
        # of the many arteries that fill early.
        ci, cj, cz = c["centroid"]
        inferiority = 1.0 - cz / max(1, Z - 1)                  # 1 = skull base
        centrality = 1.0 - min(1.0, abs(ci - mid_i) / (0.5 * span))
        earliness = 1.0 - c["features"]["arrival"] / T_arr
        return 0.4 * inferiority + 0.35 * centrality + 0.25 * earliness

    artery = max(rest, key=ica_score)
    return {"artery": artery, "vein": vein}


def find_regions_temporal(conc_path: str | Path, cnn_rois: dict | None = None) -> dict:
    """Deterministic temporal AIF/VIF region finder — no VLM, no training, ~1 s.

    vesselness (time-shape, not brightness) → connected-component candidates →
    temporal gating (reject mucosa) → label artery/vein by arrival + geometry. This is
    the independent second opinion the CNN can't give: the CNN trains on time-*averaged*
    volumes, so it is blind to the arrival timing this uses. ``cnn_rois`` adds region
    metrics. The 'math zoom' to the single AIF voxel is ``region_curve(..., 'max')``."""
    vol, _ = load_canonical(conc_path)
    mx = max_projection(vol)
    vk = vesselness_map(vol)
    cands = detect_candidates(vol, vk)
    picks = label_vessels(cands, mx)
    out: dict = {"n_candidates": len(cands)}
    for vessel in ("artery", "vein"):
        c = picks.get(vessel)
        if not c:
            out[vessel] = None
            continue
        rec = {"slice": c["slice"], "centroid": c["centroid"], "voxels": c["size"],
               "features": c["features"], "klass": c["klass"], "mask": c["mask"]}
        if cnn_rois is not None:
            rec["vs_cnn"] = region_metrics(c["mask"], cnn_rois.get(vessel))
        out[vessel] = rec
    return out


def find_regions(conc_path: str | Path, repo: str, work_dir: str | Path,
                 cnn_rois: dict | None = None) -> dict | None:
    """VLM AIF/VIF region finder: vesselness → slice-pick → single-slice grounding →
    region-grow → temporal gate. Returns per-vessel {slice, bbox, voxels, features,
    klass, mask, vs_cnn?}. ``cnn_rois={'artery':mask3d,'vein':mask3d}`` adds region
    metrics. None if no VLM backend."""
    if not vlm_available():
        return None
    vol, _ = load_canonical(conc_path)
    mx = max_projection(vol)
    vk = vesselness_map(vol)
    X, Y, Z, _ = vol.shape
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    picks = _vlm_locate(montage_png(vk, wd / "vesselness_sheet.png"), repo, prompt=_SLICE_PROMPT) or {}
    out: dict = {"_picks": picks}
    for vessel in ("artery", "vein"):
        d = picks.get(vessel) or {}
        z = d.get("slice")
        if z is None or not (0 <= int(z) < Z):
            out[vessel] = None
            continue
        z = int(z)
        spng = _slice_png(vk[:, :, z], wd / f"vesselness_{vessel}_z{z}.png")
        gr = _vlm_locate(spng, repo, prompt=_GROUND_PROMPT) or {}
        bbox = gr.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            out[vessel] = {"slice": z, "bbox": None, "voxels": 0, "klass": "none"}
            continue
        ir, jr = _bbox_to_voxel(bbox, X, Y)
        mask = grow_in_bbox(vk, z, ir, jr)
        feat = temporal_features(region_curve(vol, mask)) if mask is not None else None
        rec = {"slice": z, "bbox": bbox, "voxels": int(mask.sum()) if mask is not None else 0,
               "features": feat, "klass": classify_curve(feat), "mask": mask}
        if cnn_rois is not None:
            rec["vs_cnn"] = region_metrics(mask, cnn_rois.get(vessel))
        out[vessel] = rec
    return out
