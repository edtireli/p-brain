#!/usr/bin/env python3
"""Debug script: run the RICA slice-classifier + ROI model on DCE.

Goal: quickly answer "does the classifier ever light up for ICA" when fed
per-frame slices, and which rotation (np.rot90 k) produces the strongest signal.

It prints, per rotation:
- global max probability
- top N (slice, frame) hits

It can also generate plots comparing ROI model behavior on:
- best single frame (by slice-classifier)
- time-mean projection
- time-max projection
- many frames (union + a few example frames)

Usage examples:
  python tools/debug_rica_raw_dce.py --subject /Volumes/T5_EVO_EDT/hemisure/20240618x2_flot
  python tools/debug_rica_raw_dce.py --nifti /path/to/WIPhperf120long.nii --top 20

Notes:
- This uses the same input resize as `modules/AI_input_functions.py` (256x256, 1 channel).
- Normalization defaults to "global" min-max across the whole 4D volume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


def _add_repo_root_to_syspath() -> Path:
    # tools/debug_rica_raw_dce.py -> repo root is ../
    this = Path(__file__).resolve()
    repo_root = this.parent.parent
    sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _add_repo_root_to_syspath()


def _resolve_nifti(subject_dir: Path) -> Path:
    nifti_dir = subject_dir / "NIfTI"
    candidates = [
        nifti_dir / "WIPhperf120long.nii",
        nifti_dir / "WIPDelRec-hperf120long.nii",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No known DCE NIfTI found under {nifti_dir}. Tried: {', '.join(str(c) for c in candidates)}"
    )


def _unique_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


NormMode = Literal["global", "frame", "slice"]


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    vmin = float(np.nanmin(x))
    vmax = float(np.nanmax(x))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - vmin) / (vmax - vmin)
    return y.astype(np.float32, copy=False)


@dataclass(frozen=True)
class Hit:
    prob: float
    slice_index: int
    frame_index: int


def _iter_hits_for_rotation(
    *,
    vol4d: np.ndarray,
    rot_k: int,
    norm: NormMode,
    flip_lr: bool,
    model,
    top: int,
    frame_stride: int,
) -> tuple[float, list[Hit]]:
    import cv2

    k_mod = rot_k % 4
    vol = np.rot90(vol4d, k=k_mod, axes=(0, 1))

    # Choose normalization strategy.
    if norm == "global":
        vol_norm = _minmax01(vol)
    elif norm == "frame":
        vol_norm = vol.astype(np.float32, copy=False)
        # normalize each 3D frame
        for t in range(vol.shape[3]):
            vol_norm[:, :, :, t] = _minmax01(vol[:, :, :, t])
    else:  # slice
        vol_norm = vol.astype(np.float32, copy=False)
        for z in range(vol.shape[2]):
            for t in range(vol.shape[3]):
                vol_norm[:, :, z, t] = _minmax01(vol[:, :, z, t])

    hits: list[Hit] = []
    max_prob = 0.0

    # Iterate slices and frames.
    z_dim = int(vol_norm.shape[2])
    t_dim = int(vol_norm.shape[3])

    for z in range(z_dim):
        for t in range(0, t_dim, max(1, int(frame_stride))):
            sl = vol_norm[:, :, z, t]
            if flip_lr:
                sl = np.fliplr(sl)

            x = cv2.resize(sl, (256, 256), interpolation=cv2.INTER_LINEAR)
            x = np.expand_dims(x, axis=-1)
            x = np.expand_dims(x, axis=0)

            try:
                p = float(model.predict(x, verbose=0)[0][0])
            except TypeError:
                p = float(model.predict(x)[0][0])

            if p > max_prob:
                max_prob = p

            if len(hits) < top:
                hits.append(Hit(prob=p, slice_index=z, frame_index=t))
                hits.sort(key=lambda h: h.prob, reverse=True)
            else:
                if p > hits[-1].prob:
                    hits[-1] = Hit(prob=p, slice_index=z, frame_index=t)
                    hits.sort(key=lambda h: h.prob, reverse=True)

    return float(max_prob), hits


def _predict_roi_mask_256(img01_2d: np.ndarray, roi_model) -> np.ndarray:
    """Return a 256x256 uint8 mask (0/1)."""
    img01_2d = np.asarray(img01_2d, dtype=np.float32)
    x = np.expand_dims(img01_2d, axis=-1)
    x = np.expand_dims(x, axis=0)
    try:
        pred = roi_model.predict(x, verbose=0).squeeze()
    except TypeError:
        pred = roi_model.predict(x).squeeze()
    thr = 0.5 * float(np.max(pred)) if np.size(pred) else 0.0
    mask = (pred > thr).astype(np.uint8)
    return mask


def _predict_roi_pred_256(img01_2d: np.ndarray, roi_model) -> np.ndarray:
    """Return a 256x256 float32 prediction map."""
    img01_2d = np.asarray(img01_2d, dtype=np.float32)
    x = np.expand_dims(img01_2d, axis=-1)
    x = np.expand_dims(x, axis=0)
    try:
        pred = roi_model.predict(x, verbose=0).squeeze()
    except TypeError:
        pred = roi_model.predict(x).squeeze()
    return np.asarray(pred, dtype=np.float32)


def _resize01_to_256(img01_2d: np.ndarray) -> np.ndarray:
    import cv2

    img01_2d = np.asarray(img01_2d, dtype=np.float32)
    return cv2.resize(img01_2d, (256, 256), interpolation=cv2.INTER_LINEAR)


def _overlay_contours(ax, img01_256: np.ndarray, mask01_256: np.ndarray, title: str) -> None:
    import cv2
    import matplotlib.pyplot as plt

    base = (np.clip(img01_256, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask01_256.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 0, 0), 2)
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")


def _softmax(x: np.ndarray, *, temperature: float = 0.10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        t = 0.10
    z = x / t
    z = z - np.max(z)
    ez = np.exp(z)
    s = float(np.sum(ez))
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(x, dtype=np.float32) / float(len(x) or 1)
    return (ez / s).astype(np.float32)


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_roi_methods(
    *,
    out_dir: Path,
    rot_k: int,
    slice_index: int,
    best_frame: int,
    mri_rot: np.ndarray,
    mri_rot_norm: np.ndarray,
    roi_model,
    flip_lr: bool,
    frame_stride: int,
    max_examples: int = 6,
    slice_classifier=None,
    weight_temperature: float = 0.10,
) -> Path:
    """Generate a single comparison figure and return its path."""
    import matplotlib.pyplot as plt

    out_dir = _ensure_out_dir(out_dir)

    # Build method inputs in [0,1] (native resolution), then resize to 256.
    sl_best = mri_rot_norm[:, :, slice_index, best_frame]
    sl_mean = _minmax01(np.mean(mri_rot[:, :, slice_index, :], axis=-1))
    sl_max = _minmax01(np.max(mri_rot[:, :, slice_index, :], axis=-1))

    if flip_lr:
        sl_best = np.fliplr(sl_best)
        sl_mean = np.fliplr(sl_mean)
        sl_max = np.fliplr(sl_max)

    best_256 = _resize01_to_256(sl_best)
    mean_256 = _resize01_to_256(sl_mean)
    max_256 = _resize01_to_256(sl_max)

    mask_best = _predict_roi_mask_256(best_256, roi_model)
    mask_mean = _predict_roi_mask_256(mean_256, roi_model)
    mask_max = _predict_roi_mask_256(max_256, roi_model)

    # Many-frames: compute weighted-union mask and pick a few example frames.
    stride = max(1, int(frame_stride))
    frame_rows: list[tuple[int, float, np.ndarray, np.ndarray, int]] = []
    # (t, p_cls, roi_pred, roi_mask, area)
    for t in range(0, mri_rot_norm.shape[3], stride):
        sl = mri_rot_norm[:, :, slice_index, t]
        if flip_lr:
            sl = np.fliplr(sl)
        sl_256 = _resize01_to_256(sl)

        p_cls = 0.0
        if slice_classifier is not None:
            x_cls = np.expand_dims(sl_256, axis=-1)
            x_cls = np.expand_dims(x_cls, axis=0)
            try:
                p_cls = float(slice_classifier.predict(x_cls, verbose=0)[0][0])
            except TypeError:
                p_cls = float(slice_classifier.predict(x_cls)[0][0])

        roi_pred = _predict_roi_pred_256(sl_256, roi_model)
        thr = 0.5 * float(np.max(roi_pred)) if np.size(roi_pred) else 0.0
        roi_mask = (roi_pred > thr).astype(np.uint8)
        area = int(np.count_nonzero(roi_mask))
        frame_rows.append((int(t), float(p_cls), roi_pred, roi_mask, area))

    # Weighted union: softmax weights over classifier confidence.
    if frame_rows:
        p_vec = np.array([r[1] for r in frame_rows], dtype=np.float32)
        # If no classifier provided, fall back to uniform weights.
        if slice_classifier is None:
            w = np.ones_like(p_vec, dtype=np.float32) / float(len(p_vec))
        else:
            w = _softmax(p_vec, temperature=float(weight_temperature))
        pred_stack = np.stack([r[2] for r in frame_rows], axis=0)  # (T,256,256)
        weighted_pred = np.tensordot(w, pred_stack, axes=(0, 0))  # (256,256)
        thr_u = 0.5 * float(np.max(weighted_pred)) if np.size(weighted_pred) else 0.0
        weighted_union = (weighted_pred > thr_u).astype(np.uint8)
    else:
        w = np.array([], dtype=np.float32)
        weighted_union = np.zeros((256, 256), dtype=np.uint8)

    # Examples: largest mask areas.
    frame_rows_sorted = sorted(frame_rows, key=lambda r: r[4], reverse=True)
    examples = frame_rows_sorted[: max(1, int(max_examples))] if frame_rows_sorted else []

    # Plot layout: top row overlays for (best, mean, max, weighted-union), then timeline,
    # then example overlays.
    n_cols = 4
    n_rows = 2 + int(np.ceil(len(examples) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    # Row 0
    _overlay_contours(axes[0, 0], best_256, mask_best, f"best frame t={best_frame}")
    _overlay_contours(axes[0, 1], mean_256, mask_mean, "time-mean")
    _overlay_contours(axes[0, 2], max_256, mask_max, "time-max")
    # For weighted union overlay, use best-frame image as base for visibility.
    union_title = f"weighted-union (stride={stride})"
    if slice_classifier is not None:
        union_title += f"\nsoftmax temp={float(weight_temperature):.2f}"
    _overlay_contours(axes[0, 3], best_256, weighted_union, union_title)

    # Row 1: weights + area timeline (span all columns).
    try:
        import matplotlib.pyplot as plt

        ax_t = axes[1, 0]
        ax_t.axis("off")
        ax = fig.add_subplot(n_rows, 1, 2)  # big axis across full width
        if frame_rows:
            t_vec = np.array([r[0] for r in frame_rows], dtype=int)
            p_vec = np.array([r[1] for r in frame_rows], dtype=np.float32)
            area_vec = np.array([r[4] for r in frame_rows], dtype=np.float32)
            if slice_classifier is None:
                w_vec = np.ones_like(p_vec) / float(len(p_vec))
            else:
                w_vec = w
            ax.plot(t_vec, p_vec, label="slice-cls p", linewidth=1)
            ax.plot(t_vec, area_vec / (256.0 * 256.0), label="mask area (frac)", linewidth=1)
            ax.plot(t_vec, w_vec, label="weight", linewidth=1)
            ax.set_xlabel("frame")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No frames evaluated", ha="center", va="center")
    except Exception:
        # If the timeline subplot fails, don't block plot generation.
        pass

    # Remaining rows: example frames
    idx = 0
    for r in range(2, n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if idx >= len(examples):
                ax.axis("off")
                continue
            t, p_cls, roi_pred, m, area = examples[idx]
            sl = mri_rot_norm[:, :, slice_index, t]
            if flip_lr:
                sl = np.fliplr(sl)
            sl_256 = _resize01_to_256(sl)
            _overlay_contours(ax, sl_256, m, f"t={t} p={p_cls:.2f} area={area}")
            idx += 1

    fig.suptitle(f"RICA ROI methods | rot90 k={rot_k} (k%4={rot_k%4}) | slice={slice_index}")
    fig.tight_layout()

    out_path = out_dir / f"rica_roi_methods_k{rot_k}_slice{slice_index}_t{best_frame}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RICA slice-classifier on raw DCE frames.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--subject", type=Path, help="Subject folder containing NIfTI/WIPhperf120long.nii")
    src.add_argument("--nifti", type=Path, help="Path to a DCE NIfTI (4D)")

    parser.add_argument(
        "--rotations",
        type=str,
        default="1,0,2,3,-1",
        help=(
            "Comma-separated rot90 k values to try (order preserved). "
            "Default: 1,0,2,3,-1 (rotate up / CCW first)"
        ),
    )
    parser.add_argument(
        "--norm",
        choices=["global", "frame", "slice"],
        default="global",
        help="Normalization mode: global minmax (default), per-frame, or per-slice-per-frame.",
    )
    parser.add_argument(
        "--artery",
        choices=["rica", "lica", "both"],
        default="rica",
        help="Which side to evaluate. LICA uses left-right flip with the RICA-trained model.",
    )
    parser.add_argument(
        "--flip-lr",
        action="store_true",
        help="(Deprecated) Flip left-right before inference; prefer --artery lica/both.",
    )
    parser.add_argument("--top", type=int, default=15, help="Top N (slice,frame) hits to keep per rotation")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Evaluate every Nth frame (1=all frames). Increase for speed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--plot-roi",
        action="store_true",
        help="Generate ROI-method comparison plots using the RICA ROI model.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write plots (defaults to <subject>/Images/AI/debug_rica_raw_dce).",
    )

    args = parser.parse_args()

    rotations_raw = [int(s.strip()) for s in str(args.rotations).split(",") if s.strip()]
    rotations = _unique_preserve_order(rotations_raw)

    # Resolve input NIfTI path.
    if args.subject is not None:
        nifti_path = _resolve_nifti(args.subject)
        subject_dir = args.subject
    else:
        nifti_path = args.nifti
        subject_dir = args.nifti.parent

    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(str(nifti_path))

    # Import TF lazily so argparse is responsive.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow.keras.models import load_model  # pylint: disable=import-error

    try:
        from utils.settings import AI_MODEL_PATHS
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed importing utils.settings.AI_MODEL_PATHS; run this from the p-brain repo."
        ) from e

    model_path = Path(AI_MODEL_PATHS["slice_classifier_rica"])
    if not model_path.exists():
        raise FileNotFoundError(f"RICA slice-classifier not found: {model_path}")

    roi_model_path = Path(AI_MODEL_PATHS["rica_roi"])
    if args.plot_roi and not roi_model_path.exists():
        raise FileNotFoundError(f"RICA ROI model not found: {roi_model_path}")

    print(f"NIfTI: {nifti_path}")
    print(f"Model: {model_path}")
    # Determine which artery modes to run.
    artery = str(args.artery).strip().lower()
    artery_runs: list[tuple[str, bool]]
    if artery == "both":
        artery_runs = [("rica", False), ("lica", True)]
    elif artery == "lica":
        artery_runs = [("lica", True)]
    else:
        artery_runs = [("rica", False)]

    # Back-compat: explicit --flip-lr forces flip for all runs.
    if bool(args.flip_lr):
        artery_runs = [(name, True) for (name, _) in artery_runs]

    print(
        f"norm={args.norm} artery={artery} frame_stride={int(args.frame_stride)} top={int(args.top)} rotations={rotations}"
    )

    model = load_model(str(model_path), compile=False)
    roi_model = load_model(str(roi_model_path), compile=False) if args.plot_roi else None

    # Load NIfTI.
    import nibabel as nib

    vol4d = np.asarray(nib.load(str(nifti_path)).get_fdata())
    if vol4d.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got shape={vol4d.shape}")

    results: dict[str, object] = {
        "nifti": str(nifti_path),
        "shape": list(vol4d.shape),
        "rotations": rotations,
        "norm": args.norm,
        "artery": artery,
        "frame_stride": int(args.frame_stride),
        "top": int(args.top),
        "runs": {},
    }

    for run_name, flip_lr in artery_runs:
        print("=")
        print(f"RUN: {run_name.upper()} (flip_lr={flip_lr})")

        per_rotation: dict[str, object] = {}
        best = {"rot_k": None, "max_prob": -1.0}
        best_hit: Hit | None = None

        for k in rotations:
            max_prob, hits = _iter_hits_for_rotation(
                vol4d=vol4d,
                rot_k=int(k),
                norm=args.norm,
                flip_lr=bool(flip_lr),
                model=model,
                top=int(args.top),
                frame_stride=int(args.frame_stride),
            )

            if max_prob > float(best["max_prob"]):
                best = {"rot_k": int(k), "max_prob": float(max_prob)}
                best_hit = hits[0] if hits else None

            print("-")
            print(f"rot90 k={int(k)} (k%4={int(k)%4})  max_prob={max_prob:.4f}")
            for h in hits[: int(args.top)]:
                print(f"  p={h.prob:.4f}  slice={h.slice_index}  frame={h.frame_index}")

            per_rotation[str(int(k))] = {
                "k": int(k),
                "k_mod": int(k) % 4,
                "max_prob": float(max_prob),
                "hits": [
                    {"prob": float(h.prob), "slice": int(h.slice_index), "frame": int(h.frame_index)}
                    for h in hits
                ],
            }

        print("=")
        print(f"BEST {run_name.upper()}: rot90 k={best['rot_k']}  max_prob={float(best['max_prob']):.4f}")

        run_result: dict[str, object] = {
            "flip_lr": bool(flip_lr),
            "best": best,
            "per_rotation": per_rotation,
        }

        if args.plot_roi and best_hit is not None:
            # Reconstruct the rotated raw volume for the best rotation.
            rot_k = int(best["rot_k"]) if best["rot_k"] is not None else int(rotations[0])
            rot_k_mod = rot_k % 4
            mri_rot = np.rot90(vol4d, k=rot_k_mod, axes=(0, 1))

            # Normalize raw frames.
            if args.norm == "global":
                mri_rot_norm = _minmax01(mri_rot)
            elif args.norm == "frame":
                mri_rot_norm = mri_rot.astype(np.float32, copy=False)
                for t in range(mri_rot.shape[3]):
                    mri_rot_norm[:, :, :, t] = _minmax01(mri_rot[:, :, :, t])
            else:
                mri_rot_norm = mri_rot.astype(np.float32, copy=False)
                for z in range(mri_rot.shape[2]):
                    for t in range(mri_rot.shape[3]):
                        mri_rot_norm[:, :, z, t] = _minmax01(mri_rot[:, :, z, t])

            if args.out_dir is not None:
                out_dir = Path(args.out_dir)
            else:
                out_dir = Path(subject_dir) / "Images" / "AI" / "debug_rica_raw_dce"

            # Write separate files for RICA/LICA.
            plot_path = _plot_roi_methods(
                out_dir=out_dir,
                rot_k=rot_k,
                slice_index=int(best_hit.slice_index),
                best_frame=int(best_hit.frame_index),
                mri_rot=mri_rot,
                mri_rot_norm=mri_rot_norm,
                roi_model=roi_model,
                flip_lr=bool(flip_lr),
                frame_stride=int(args.frame_stride),
                slice_classifier=model,
                weight_temperature=float(os.getenv("P_BRAIN_AI_ROI_WEIGHT_TEMP", "0.10")),
            )
            # Rename to include side for clarity.
            side_plot = plot_path.with_name(plot_path.stem + f"_{run_name}" + plot_path.suffix)
            try:
                plot_path.replace(side_plot)
                plot_path = side_plot
            except Exception:
                plot_path = side_plot

            print(f"ROI plot ({run_name.upper()}): {plot_path}")
            run_result["roi_plot"] = str(plot_path)

        results["runs"][run_name] = run_result

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
