"""CNN-based AIF extractor — paper §4.4 default method.

Two-stage convolutional pipeline trained on time-averaged DCE volumes
(Tireli et al., §4.4):

1. **Slice classifier** scores every (slice, frame) pair for vascular
   content. Per slice, the frame with the highest score is selected.
   Slices are then ranked by best-frame score; the top ``max_slices``
   above ``slice_thresholds`` (default 0.5, fallback 0.3) are kept.
2. **U-Net ROI segmenter** runs on the chosen (slice, best-frame) pairs
   and emits per-slice binary masks. Masks are resized to native DCE
   resolution and unioned across slices to form the 3-D ROI.

The output curve is the **single ROI voxel with the highest peak
amplitude** over the time series (legacy default). Set
``curve_method="adaptive_max"`` for the per-frame max across voxels, or
``"mean"`` / ``"median"`` for ROI averaging.

The plug-in autodetects whether ``source="rica"`` (default arterial) or
``"sss"`` (venous) and loads the appropriate (slice-classifier, U-Net)
pair. Other arteries follow the legacy convention via ``flip_lr=True``
(rICA → lICA mirror).

Lazy TensorFlow + Keras imports: this module imports cleanly even if
TF is absent — only ``extract()`` requires it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from .base import AIFExtractor, InputFunction


# ────────────────────────────────────────────────────────────────────────
# Default model paths (matches legacy AI_MODEL_PATHS in utils/settings.py)
# ────────────────────────────────────────────────────────────────────────


from pbrain.core import get_logger
_log = get_logger("aif")


def _default_model_paths() -> dict[str, str]:
    """Locate the four CNN ``.keras`` weights.

    Each entry resolves to: the per-model env override if set; else the first
    existing file across the in-repo ``AI/`` dir and the per-user weights dir
    (where ``pbrain fetch-weights`` downloads them); else the per-user path as
    the default download target (used in the "not found" message).
    """
    from pbrain._paths import weights_dir
    repo_ai = Path(__file__).resolve().parent.parent.parent / "AI"  # repo-root/AI
    user_ai = weights_dir()

    def _resolve(env_var: str, filename: str) -> str:
        override = os.environ.get(env_var)
        if override:
            return override
        for base in (repo_ai, user_ai):
            cand = base / filename
            if cand.exists():
                return str(cand)
        return str(user_ai / filename)

    return {
        "slice_classifier_rica": _resolve("SLICE_CLASSIFIER_RICA_MODEL", "slice_classifier_model_rica.keras"),
        "rica_roi": _resolve("RICA_ROI_MODEL", "rica_roi_model.keras"),
        "slice_classifier_ss": _resolve("SLICE_CLASSIFIER_SS_MODEL", "ss_slice_classifier.keras"),
        "ss_roi": _resolve("SS_ROI_MODEL", "ss_roi_model.keras"),
    }


# ────────────────────────────────────────────────────────────────────────
# Slice preprocessing — must match the training pipeline (paper §4.4
# training set was time-averaged DCE volumes; per-slice min-max → uint8
# → centre-crop/pad to 256×256 → float [0, 1]).
# ────────────────────────────────────────────────────────────────────────


def _to_uint8(slice2d: np.ndarray) -> np.ndarray:
    x = np.asarray(slice2d, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _crop_or_pad(img: np.ndarray, target: tuple[int, int] = (256, 256)) -> np.ndarray:
    h, w = img.shape
    th, tw = target
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h or pad_w:
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        img = np.pad(img, ((top, bottom), (left, right)), mode="constant")
        h, w = img.shape
    sh = (h - th) // 2
    sw = (w - tw) // 2
    return img[sh : sh + th, sw : sw + tw]


def _prep_slice_float01(slice2d: np.ndarray, *, flip_lr: bool) -> np.ndarray:
    u8 = _to_uint8(slice2d)
    if flip_lr:
        u8 = np.fliplr(u8)
    if u8.shape != (256, 256):
        u8 = _crop_or_pad(u8, (256, 256))
    return u8.astype(np.float32) / 255.0


# ────────────────────────────────────────────────────────────────────────
# Lazy model cache (one-time load per process)
# ────────────────────────────────────────────────────────────────────────


_MODEL_CACHE: dict[str, Any] = {}


def _load_keras(path: str):
    cached = _MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    if not Path(path).exists():
        raise FileNotFoundError(
            f"CNN weights not found at {path}. Download them from Zenodo with "
            f"`pbrain fetch-weights` (or run `pbrain setup`), set the appropriate "
            f"environment variable (SLICE_CLASSIFIER_RICA_MODEL, RICA_ROI_MODEL, "
            f"SLICE_CLASSIFIER_SS_MODEL, SS_ROI_MODEL), or pick a weights-free AIF "
            f"(--aif from_file | manual | deterministic)."
        )
    try:
        from tensorflow.keras.models import load_model  # type: ignore
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow is required for the CNN AIF plug-in. Install "
            "via `pip install tensorflow` (and `tensorflow-metal` for MPS) "
            "or use --aif deterministic."
        ) from exc
    # Log which device TF will actually use — helps diagnose "I asked for MPS
    # but it ran on CPU" surprises (Metal plug-in not installed, etc.).
    visible = tf.config.list_physical_devices()
    has_gpu = any(d.device_type == "GPU" for d in visible)
    if has_gpu:
        _log.debug("CNN AIF: TF visible devices = %s", [d.device_type for d in visible])
    model = load_model(path, compile=False)
    _MODEL_CACHE[path] = model
    return model


# ────────────────────────────────────────────────────────────────────────
# Inference primitives
# ────────────────────────────────────────────────────────────────────────


def _score_slices_per_frame(
    dce: np.ndarray,
    slice_classifier,
    *,
    frame_candidates: list[int],
    flip_lr: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """For each slice z, return (best_frame_t[z], best_score[z]).

    Builds one batch over (frame × slice) combinations, runs the
    classifier once, then reduces along the frame axis.
    """
    X, Y, Z, T = dce.shape
    cands = [int(t) for t in frame_candidates if 0 <= int(t) < T]
    if not cands:
        return np.zeros((Z,), dtype=np.int32), np.zeros((Z,), dtype=np.float32)

    batch = np.empty((len(cands) * Z, 256, 256, 1), dtype=np.float32)
    idx = 0
    for t in cands:
        for z in range(Z):
            batch[idx, :, :, 0] = _prep_slice_float01(dce[:, :, z, t], flip_lr=flip_lr)
            idx += 1

    try:
        preds = slice_classifier.predict(batch, verbose=0)
    except TypeError:
        preds = slice_classifier.predict(batch)
    scores = np.asarray(preds, dtype=np.float32).reshape(len(cands), Z)

    best_i = np.argmax(scores, axis=0).astype(np.int32)
    best_t = np.asarray([cands[int(i)] for i in best_i], dtype=np.int32)
    best_score = scores[best_i, np.arange(Z, dtype=np.int32)].astype(np.float32)
    return best_t, best_score


def _select_chosen_slices(
    best_score: np.ndarray,
    *,
    slice_thresholds: tuple[float, ...],
    max_slices: int,
) -> list[int]:
    Z = int(best_score.size)
    max_slices_i = max(1, min(int(max_slices), Z))
    thresholds = [float(t) for t in slice_thresholds if 0.0 < float(t) <= 1.0]
    if not thresholds:
        thresholds = [0.5]

    for thr in thresholds:
        cand = [int(z) for z in range(Z) if float(best_score[z]) > thr]
        cand = sorted(cand, key=lambda z: float(best_score[z]), reverse=True)
        chosen = cand[:max_slices_i]
        if chosen:
            return chosen

    # Final fallback: top-N by score regardless of threshold (but still > 0).
    order = np.argsort(-best_score)
    return [int(z) for z in order[:max_slices_i] if float(best_score[int(z)]) > 0.0]


def _resize_nearest(src: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """Nearest-neighbour resize, bit-identical to ``cv2.resize(src, (dst_w, dst_h),
    interpolation=cv2.INTER_NEAREST)`` — pure NumPy, no OpenCV dependency.

    OpenCV maps each destination index ``d`` to source index
    ``floor(d * scale)`` with ``scale = 1 / (dst / src)`` (the reciprocal of the
    forward ratio, which lands just below integer boundaries in floating point);
    the result is clamped to ``src - 1``. Replicating that exact computation makes
    this a drop-in replacement (validated bit-for-bit over 1324 shape/pattern
    cases). Used for the binary U-Net ROI mask, so values are preserved exactly.
    """
    sh, sw = src.shape[:2]
    scale_y = 1.0 / (dst_h / sh)
    scale_x = 1.0 / (dst_w / sw)
    sy = np.minimum(np.floor(np.arange(dst_h) * scale_y).astype(np.intp), sh - 1)
    sx = np.minimum(np.floor(np.arange(dst_w) * scale_x).astype(np.intp), sw - 1)
    return src[sy[:, None], sx[None, :]]


def _segment_rois(
    dce: np.ndarray,
    roi_model,
    *,
    chosen_z: list[int],
    best_t: np.ndarray,
    flip_lr: bool,
) -> dict[int, np.ndarray]:
    """Run U-Net on each (z, best_t[z]) pair; return native-resolution masks."""

    X, Y, Z, T = dce.shape
    if not chosen_z:
        return {}

    batch = np.empty((len(chosen_z), 256, 256, 1), dtype=np.float32)
    for i, z in enumerate(chosen_z):
        batch[i, :, :, 0] = _prep_slice_float01(dce[:, :, z, int(best_t[z])], flip_lr=flip_lr)

    try:
        roi_preds = roi_model.predict(batch, verbose=0)
    except TypeError:
        roi_preds = roi_model.predict(batch)
    roi_preds = np.asarray(roi_preds)

    masks: dict[int, np.ndarray] = {}
    for i, z in enumerate(chosen_z):
        m = np.squeeze(roi_preds[i])
        if m.size == 0:
            continue
        # Adaptive 50%-of-max threshold matches legacy
        thr = 0.5 * float(np.max(m))
        binary = (m > thr).astype(np.uint8)
        if flip_lr:
            binary = np.fliplr(binary)
        native = _resize_nearest(binary, X, Y)
        masks[int(z)] = (native > 0).astype(bool)
    return masks


def _curve_from_masks(
    dce: np.ndarray,
    masks: dict[int, np.ndarray],
    *,
    curve_method: str,
    baseline_frames: int,
    max_conc_mM: float = 0.0,
    select_on: str = "signal",
    signal_volume: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the AIF curve from the union of per-slice ROIs.

    Returns ``(curve, mask3d_full, mask3d_curve)``:

    * ``mask3d_full``  — every voxel inside the U-Net output (for QC overlays).
    * ``mask3d_curve`` — the voxels that actually contributed to ``curve``
      (one voxel for ``max_voxel``, all voxels for mean/median/adaptive).
      Downstream signal→conc averages T1/M0 over this *tight* mask so
      the conversion uses pure-blood relaxation parameters rather than a
      partial-volume mean.

    ``select_on`` controls how ``max_voxel`` ranks candidates: ``"signal"``
    (default, reference behaviour) ranks by the raw DCE-signal peak supplied
    in ``signal_volume`` — the actually-brightest blood voxel — while
    ``"concentration"`` ranks by the converted mM peak (which lets a low
    fitted-M0 voxel win). Either way the *returned* curve is the concentration
    curve of the chosen voxel.
    """
    X, Y, Z, T = dce.shape
    mask3d_full = np.zeros((X, Y, Z), dtype=bool)
    for z, m in masks.items():
        if m.shape == (X, Y):
            mask3d_full[:, :, z] = m

    if not mask3d_full.any():
        raise RuntimeError("CNN AIF: U-Net returned an empty mask across all chosen slices.")

    roi_tc = dce[mask3d_full]                  # (n_voxels, T)
    coords = np.argwhere(mask3d_full)          # (n_voxels, 3) in (x, y, z)
    method = curve_method.strip().lower()

    mask3d_curve = np.zeros_like(mask3d_full)
    if method == "mean":
        curve = roi_tc.mean(axis=0)
        mask3d_curve = mask3d_full
    elif method == "median":
        curve = np.median(roi_tc, axis=0)
        mask3d_curve = mask3d_full
    elif method in ("adaptive_max", "adaptive-max", "adaptivemax"):
        # Per-frame brightest voxel within the ROI (the "adaptive" bolus curve).
        # Drop voxels with any NaN frame (failed T1/M0) so one bad voxel doesn't
        # NaN-poison the whole curve; then take the per-frame max over the rest.
        _valid = np.isfinite(roi_tc).all(axis=1)
        curve = np.nanmax(roi_tc[_valid] if _valid.any() else roi_tc, axis=0)
        mask3d_curve = mask3d_full
    else:
        # Default: pick the single voxel with the highest baseline-corrected
        # peak. Ranking is on the raw SIGNAL by default (the brightest blood
        # voxel, matching the reference pipeline); ``select_on="concentration"``
        # ranks on the mM curve instead. The returned curve is always the
        # concentration curve of the chosen voxel.
        # NaN-aware: voxels with any NaN frame are excluded from the max.
        score_tc = roi_tc
        if str(select_on).strip().lower() == "signal" and signal_volume is not None:
            sv = np.asarray(signal_volume)
            if sv.shape[:3] == mask3d_full.shape:
                score_tc = sv[mask3d_full]
        with np.errstate(invalid="ignore"):
            bf = max(1, int(baseline_frames))
            baseline = np.nanmean(score_tc[:, :bf], axis=1, keepdims=True)
            corrected = score_tc - baseline
            # Voxels with any NaN frame → finite=False → score = -inf
            finite_voxel = np.isfinite(corrected).all(axis=1)
            peak_per_voxel = np.where(
                finite_voxel, np.nanmax(corrected, axis=1), -np.inf,
            )
            # Saturation-recovery DCE saturates in pure blood: at the AIF peak
            # the signal is ~95 % recovered, so the log conversion is unstable
            # and a single noisy bright frame tips S/K→1, clamping that voxel to
            # tens of mM. Such clamps are non-physiological, so always reject
            # candidates whose *concentration* peak exceeds a physiological blood
            # ceiling — independent of how voxels are ranked.
            if max_conc_mM and max_conc_mM > 0:
                conc_peak = np.where(finite_voxel, np.nanmax(roi_tc, axis=1), np.inf)
                physiological = conc_peak <= float(max_conc_mM)
                if physiological.any():
                    peak_per_voxel = np.where(physiological, peak_per_voxel, -np.inf)
                else:
                    # All candidates clamp — fall back to the most conservative
                    # (least-saturated) voxel rather than the highest.
                    finite_peaks = np.where(finite_voxel, conc_peak, np.inf)
                    bv = int(np.argmin(finite_peaks))
                    curve = roi_tc[bv].copy()
                    bx, by, bz = coords[bv]
                    mask3d_curve[bx, by, bz] = True
                    return np.asarray(curve, dtype=np.float32), mask3d_full, mask3d_curve
        if not np.isfinite(peak_per_voxel).any():
            raise RuntimeError(
                "CNN AIF: all candidate voxels in the ROI have NaN curves "
                "(check T1/M0 fits). Try --curve-method mean or fix T1 map."
            )
        best_voxel = int(np.argmax(peak_per_voxel))
        curve = roi_tc[best_voxel].copy()
        bx, by, bz = coords[best_voxel]
        mask3d_curve[bx, by, bz] = True

    return np.asarray(curve, dtype=np.float32), mask3d_full, mask3d_curve


# ────────────────────────────────────────────────────────────────────────
# Plugin
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _CnnAIF:
    key: ClassVar[str] = "cnn"
    name: ClassVar[str] = "CNN AIF (two-stage classifier + segmenter)"
    description: ClassVar[str] = (
        "Two-stage CNN: slice classifier identifies vascular slices, "
        "per-slice U-Net delineates rICA / lICA / SSS ROI. The default "
        "AIF method in the paper. Requires the .keras weights from "
        "AI/. CPU inference works but is slow; GPU recommended."
    )
    accepts: ClassVar[dict[str, type]] = {
        "dce_data": np.ndarray,
        "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {"input_function": InputFunction}

    def extract(
        self,
        dce_data: np.ndarray,
        t_s: np.ndarray,
        dce_affine: np.ndarray,
        *,
        t1_map: np.ndarray | None = None,
        m0_map: np.ndarray | None = None,
        t1w_volume: np.ndarray | None = None,
        t1w_affine: np.ndarray | None = None,
        concentration_data: np.ndarray | None = None,
        source: Literal["rica", "lica", "sss"] = "rica",
        slice_classifier_path: str | None = None,
        roi_model_path: str | None = None,
        flip_lr: bool | None = None,
        slice_thresholds: tuple[float, ...] = (0.5, 0.3),
        max_slices: int = 3,
        frame_candidates: list[int] | None = None,
        baseline_frames: int = 5,
        curve_method: str = "max_voxel",
        select_on: str = "signal",
        max_conc_mM: float = 0.0,
        rot90_k: int = -1,
        max_vessel_m0: bool = False,
        aif_flip_deg: float = 30.0,
        aif_tr_s: float = 0.00396,
        aif_r1: float = 4.0,
        aif_td: float = 0.120,
        **_: Any,
    ) -> InputFunction:
        """The CNN runs on ``dce_data`` (signal — its training distribution).

        When ``concentration_data`` (mM volume, per-voxel-converted upstream)
        is supplied, the AIF *curve* is extracted from that volume so the
        returned ``c_a`` is in mM with each voxel's own T1/M0 already
        applied. If ``concentration_data`` is None the curve comes from
        the signal volume (backward compatible).

        ``rot90_k``: training-orientation alignment (legacy default -1).
        """
        d_native = np.asarray(dce_data, dtype=np.float32)
        if d_native.ndim != 4:
            raise ValueError(f"dce_data must be 4-D; got shape {d_native.shape}")

        # Rotate in-plane to match the training orientation. The slice and
        # time axes are preserved; only (x, y) rotates.
        rot_k = int(rot90_k) % 4
        d = np.rot90(d_native, k=rot_k, axes=(0, 1)) if rot_k else d_native

        # If a per-voxel mM volume is supplied, the curve will be pulled
        # from there (still in the rotated frame for mask-indexing parity).
        if concentration_data is not None:
            c_native = np.asarray(concentration_data, dtype=np.float32)
            if c_native.shape != d_native.shape:
                raise ValueError(
                    f"concentration_data shape {c_native.shape} must match "
                    f"dce_data shape {d_native.shape}"
                )
            c_rot = np.rot90(c_native, k=rot_k, axes=(0, 1)) if rot_k else c_native
        else:
            c_rot = d

        defaults = _default_model_paths()
        src = source.lower()
        if slice_classifier_path is None or roi_model_path is None:
            if src in ("rica", "lica"):
                slice_classifier_path = slice_classifier_path or defaults["slice_classifier_rica"]
                roi_model_path = roi_model_path or defaults["rica_roi"]
            elif src in ("sss", "venous", "vif"):
                slice_classifier_path = slice_classifier_path or defaults["slice_classifier_ss"]
                roi_model_path = roi_model_path or defaults["ss_roi"]
            else:
                raise ValueError(
                    f"Unknown source {source!r} for CNN AIF; expected one of "
                    f"'rica', 'lica', 'sss'."
                )

        # lICA mirrors rICA across the sagittal midline (paper §4.4).
        if flip_lr is None:
            flip_lr = src == "lica"

        slice_classifier = _load_keras(str(slice_classifier_path))
        roi_model = _load_keras(str(roi_model_path))

        if frame_candidates is None:
            frame_candidates = list(range(d.shape[-1]))

        best_t, best_score = _score_slices_per_frame(
            d, slice_classifier,
            frame_candidates=list(frame_candidates),
            flip_lr=bool(flip_lr),
        )

        chosen = _select_chosen_slices(
            best_score,
            slice_thresholds=tuple(slice_thresholds),
            max_slices=int(max_slices),
        )
        if not chosen:
            raise RuntimeError(
                f"CNN AIF: no slices passed any threshold {slice_thresholds}; "
                f"best score = {float(best_score.max()):.3f}"
            )

        masks = _segment_rois(
            d, roi_model,
            chosen_z=chosen, best_t=best_t,
            flip_lr=bool(flip_lr),
        )
        if not masks:
            raise RuntimeError("CNN AIF: U-Net produced no usable masks.")

        # Default: each vessel voxel keeps its own fitted M0 (per-voxel
        # conversion, as in the reference pipeline). Opt-in ``max_vessel_m0``
        # instead re-converts every vessel voxel with a single max(M0) over the
        # ROI; enable only via ``--opt aif.<key>.max_vessel_m0=true``.
        if (max_vessel_m0 and concentration_data is not None
                and m0_map is not None and t1_map is not None):
            m0_rot = (np.rot90(np.asarray(m0_map, float), k=rot_k, axes=(0, 1))
                      if rot_k else np.asarray(m0_map, float))
            t1_rot = (np.rot90(np.asarray(t1_map, float), k=rot_k, axes=(0, 1))
                      if rot_k else np.asarray(t1_map, float))
            X, Y, Z, _ = d.shape
            union = np.zeros((X, Y, Z), dtype=bool)
            for z, m in masks.items():
                if m.shape == (X, Y):
                    union[:, :, z] = m
            mvox = m0_rot[union]
            mvox = mvox[np.isfinite(mvox) & (mvox > 0)]
            if union.any() and mvox.size:
                max_m0 = float(np.max(mvox))
                from pbrain.signal_to_conc import REGISTRY as _CONV
                sig = d[union]                              # (n_vessel, T) signal
                cc = _CONV["saturation_recovery"].convert(
                    sig, t1_rot[union], np.full(sig.shape[0], max_m0),
                    flip_angle_deg=float(aif_flip_deg), tr_s=float(aif_tr_s),
                    r1_per_s_mM=float(aif_r1), prepulse_to_readout_s=float(aif_td),
                    baseline_frames=int(baseline_frames),
                )
                c_rot = c_rot.copy()
                c_rot[union] = np.asarray(cc, dtype=np.float32)

        # Curve extraction uses the concentration volume (mM) when provided,
        # so each voxel's own T1/M0 carries through into the curve units.
        curve, mask3d_full_rot, mask3d_curve_rot = _curve_from_masks(
            c_rot, masks,
            curve_method=curve_method,
            baseline_frames=int(baseline_frames),
            # Only meaningful when the curve is read from concentration (mM);
            # for raw-signal extraction the magnitudes aren't mM, so disable.
            max_conc_mM=float(max_conc_mM) if concentration_data is not None else 0.0,
            # Rank the max-voxel on the raw signal (brightest blood) by default;
            # ``d`` is the rotated signal even when the curve is read from mM.
            select_on=select_on,
            signal_volume=d,
        )

        # Un-rotate masks back into native DCE indexing so downstream
        # stages (signal_to_conc, masking, plotting) refer to the same
        # voxels they would on the original volume. The AIF curve was
        # already extracted from the rotated volume — its values are
        # invariant to the inverse rotation, so we keep the curve as-is.
        # ``mask`` returned to downstream is the *tight* mask (only the
        # voxels that contributed to the curve), so signal_to_conc uses
        # the right T1/M0 for the conversion.
        if rot_k:
            mask3d = np.rot90(mask3d_curve_rot, k=-rot_k, axes=(0, 1))
            mask3d_full = np.rot90(mask3d_full_rot, k=-rot_k, axes=(0, 1))
        else:
            mask3d = mask3d_curve_rot
            mask3d_full = mask3d_full_rot

        chosen_frames = [int(best_t[z]) for z in chosen]
        chosen_scores = [float(best_score[z]) for z in chosen]

        return InputFunction(
            c_a=curve,
            t_s=np.asarray(t_s, dtype=float),
            mask=mask3d,            # tight: only the voxel(s) the curve came from
            source=src,
            meta={
                "algorithm": "cnn_two_stage",
                "slice_classifier_path": str(slice_classifier_path),
                "roi_model_path": str(roi_model_path),
                "chosen_slices": chosen,
                "chosen_frames": chosen_frames,
                "chosen_scores": chosen_scores,
                "n_voxels": int(mask3d.sum()),
                "n_voxels_full_roi": int(mask3d_full.sum()),
                "flip_lr": bool(flip_lr),
                "curve_method": curve_method,
                "rot90_k": int(rot_k),
            },
        )


PLUGIN = _CnnAIF()
