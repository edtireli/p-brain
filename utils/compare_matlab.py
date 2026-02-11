"""Best-effort comparisons against MATLAB reference outputs.

Currently supports T1/M0 comparisons vs a MATLAB .mat file such as
`T1_M0_plusError_maps_.mat`.

Outputs:
- JSON metrics under `Analysis/Fitting/compare_matlab_t1m0.json`
- Montage PNG under `Images/Fit/Compare_MATLAB_T1_M0.png`

This is intended for lightweight QA, not for strict scientific validation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class CompareSummary:
    mat_path: str
    t1_units_matlab: str
    t1_units_nifti: str
    t1_scale_applied_to_nifti: float
    m0_scale_applied_to_nifti: float
    metrics: Dict[str, Any]
    paths: Dict[str, str]


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_default_mat_path(subject_root: Path) -> Optional[Path]:
    # Common p-Brain MATLAB export filename.
    candidates = [
        subject_root / "T1_M0_plusError_maps_.mat",
        subject_root / "T1_M0_plusError_maps.mat",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _maybe_convert_t1_units(*, nifti_t1: "Any", matlab_t1: "Any") -> Tuple[float, str, str]:
    """Return (scale_to_apply_to_nifti, nifti_units, matlab_units).

    Heuristic: MATLAB exports are often in seconds (0..~8), while the NIfTI
    maps are often in milliseconds (0..~12000). If so, scale NIfTI by 1/1000.
    """

    import numpy as np  # type: ignore

    a = np.asarray(nifti_t1, dtype=float)
    b = np.asarray(matlab_t1, dtype=float)

    def _robust_max(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        return float(np.percentile(x, 99.9))

    amax = _robust_max(a)
    bmax = _robust_max(b)

    # If MATLAB looks like seconds and NIfTI looks like ms.
    if bmax > 0 and bmax < 30 and amax > 200:
        return 1.0 / 1000.0, "ms", "s"

    return 1.0, "unknown", "unknown"


def _fit_scale(a, b, mask) -> float:
    """Least-squares scale s such that (a*s) ~ b over mask."""

    import numpy as np  # type: ignore

    aa = a[mask].astype(float, copy=False)
    bb = b[mask].astype(float, copy=False)
    denom = float((aa * aa).sum())
    if denom <= 0:
        return 1.0
    return float((aa * bb).sum() / denom)


def _metrics(a, b, mask) -> Dict[str, Any]:
    import numpy as np  # type: ignore

    aa = a[mask].astype(float, copy=False)
    bb = b[mask].astype(float, copy=False)

    if aa.size == 0:
        return {"n": 0}

    diff = aa - bb
    out: Dict[str, Any] = {
        "n": int(aa.size),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "bias": float(np.mean(diff)),
    }

    # Correlation can fail when arrays are constant.
    try:
        if float(np.std(aa)) > 0 and float(np.std(bb)) > 0:
            out["corr"] = float(np.corrcoef(aa, bb)[0, 1])
        else:
            out["corr"] = None
    except Exception:
        out["corr"] = None

    for name, arr in (("a", aa), ("b", bb)):
        out[f"{name}_p1"] = float(np.percentile(arr, 1))
        out[f"{name}_p50"] = float(np.percentile(arr, 50))
        out[f"{name}_p99"] = float(np.percentile(arr, 99))

    return out


def _robust_vmin_vmax(a, b, *, mask, pmin: float = 1.0, pmax: float = 99.5) -> tuple[float, float]:
    import numpy as np  # type: ignore

    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mm = np.asarray(mask, dtype=bool)
    vals = np.concatenate([aa[mm & np.isfinite(aa)].ravel(), bb[mm & np.isfinite(bb)].ravel()])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(vals, pmin))
    vmax = float(np.percentile(vals, pmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if vmin == vmax:
            vmax = vmin + 1.0
    return vmin, vmax


def _robust_sym_limit(diff, *, mask, p: float = 99.9) -> float:
    import numpy as np  # type: ignore

    dd = np.asarray(diff, dtype=float)
    mm = np.asarray(mask, dtype=bool)
    vals = np.abs(dd[mm & np.isfinite(dd)].ravel())
    if vals.size == 0:
        return 1.0
    lim = float(np.percentile(vals, p))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(vals)) if vals.size else 1.0
    return lim if lim > 0 else 1.0


def _apply_inplane_ops(arr, ops: Tuple[str, ...]):
    """Apply in-plane operations to a 3D volume.

    Supported ops:
    - r0/r1/r2/r3: rot90(k) around (x,y)
    - t: transpose x/y
    - ud: flip x axis
    - lr: flip y axis

    All operations preserve shape.
    """

    import numpy as np  # type: ignore

    out = np.asarray(arr)
    for op in ops:
        if op == "t":
            out = np.transpose(out, (1, 0, 2))
        elif op == "ud":
            out = out[::-1, :, :]
        elif op == "lr":
            out = out[:, ::-1, :]
        elif op in {"r0", "r1", "r2", "r3"}:
            k = int(op[1:])
            if k:
                out = np.rot90(out, k=k, axes=(0, 1))
        else:
            raise ValueError(f"Unknown op: {op}")
    return out


def _getenv_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in {"0", "false", "no", "off", ""}:
        return False
    if val in {"1", "true", "yes", "on"}:
        return True
    return default


def _apply_validator_nifti_orientation(vol3d):
    """Apply the validator's in-plane NIfTI orientation to a 3D volume.

    The Hemisure validator uses np.rot90(k=+1, axes=(0,1)) on NIfTI inputs.
    This keeps the slice axis intact and preserves shape.

    Override via env vars:
    - P_BRAIN_VALIDATOR_NIFTI_ROT90_K (default: 1)
    - P_BRAIN_VALIDATOR_NIFTI_FLIP_LR (default: 0)
    - P_BRAIN_VALIDATOR_NIFTI_FLIP_UD (default: 0)
    """

    import numpy as np  # type: ignore

    out = np.asarray(vol3d)
    try:
        k = int((os.environ.get("P_BRAIN_VALIDATOR_NIFTI_ROT90_K") or "1").strip()) % 4
    except Exception:
        k = 1
    if k:
        out = np.rot90(out, k=k, axes=(0, 1))

    if _getenv_bool("P_BRAIN_VALIDATOR_NIFTI_FLIP_LR", default=False):
        out = out[:, ::-1, :]
    if _getenv_bool("P_BRAIN_VALIDATOR_NIFTI_FLIP_UD", default=False):
        out = out[::-1, :, :]

    return out


def _iter_inplane_transforms() -> Tuple[Tuple[str, ...], ...]:
    """Enumerate a small, deterministic set of candidate transforms."""

    transforms = []
    for rot in ("r0", "r1", "r2", "r3"):
        for t in (False, True):
            for ud in (False, True):
                for lr in (False, True):
                    ops = [rot]
                    if t:
                        ops.append("t")
                    if ud:
                        ops.append("ud")
                    if lr:
                        ops.append("lr")
                    transforms.append(tuple(ops))
    return tuple(transforms)


def _choose_best_matlab_transform(*, nifti_t1_s, nifti_m0, matlab_t1_s, matlab_m0):
    """Pick the MATLAB transform that best aligns with NIfTI.

    Objective: maximize T1 correlation (primary), then M0 correlation (secondary)
    while discouraging huge T1 RMSE.
    """

    import numpy as np  # type: ignore

    A_T1 = np.asarray(nifti_t1_s, dtype=float)
    A_M0 = np.asarray(nifti_m0, dtype=float)
    B_T1 = np.asarray(matlab_t1_s, dtype=float)
    B_M0 = np.asarray(matlab_m0, dtype=float)

    def _corr(a, b, mask) -> float:
        aa = a[mask]
        bb = b[mask]
        if aa.size < 10:
            return float("nan")
        sa = float(np.std(aa))
        sb = float(np.std(bb))
        if sa <= 0 or sb <= 0:
            return float("nan")
        return float(np.corrcoef(aa, bb)[0, 1])

    def _rmse(a, b, mask) -> float:
        d = a[mask] - b[mask]
        if d.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(d * d)))

    def _fit_scale(a, b, mask) -> float:
        aa = a[mask].astype(float, copy=False)
        bb = b[mask].astype(float, copy=False)
        denom = float((aa * aa).sum())
        if denom <= 0:
            return 1.0
        return float((aa * bb).sum() / denom)

    best = None
    best_ops: Tuple[str, ...] = ("r0",)

    # Base mask independent of transform.
    base = np.isfinite(A_T1) & np.isfinite(A_M0)

    for ops in _iter_inplane_transforms():
        bt1 = _apply_inplane_ops(B_T1, ops)
        bm0 = _apply_inplane_ops(B_M0, ops)
        if bt1.shape != A_T1.shape or bm0.shape != A_M0.shape:
            continue

        # Conservative tissue mask in seconds from MATLAB side.
        mask_t1 = base & np.isfinite(bt1) & np.isfinite(bm0) & (bt1 > 0.2) & (bt1 < 5.0)
        mask_m0 = base & np.isfinite(bt1) & np.isfinite(bm0) & (bm0 > 0)

        if int(np.count_nonzero(mask_t1)) < 1000:
            continue

        m0_scale = _fit_scale(A_M0, bm0, mask_m0)
        ct1 = _corr(A_T1, bt1, mask_t1)
        cm0 = _corr(A_M0 * float(m0_scale), bm0, mask_m0)
        rt1 = _rmse(A_T1, bt1, mask_t1)

        # Score: prioritize T1 corr, then M0 corr, penalize high T1 RMSE.
        score = (ct1 if np.isfinite(ct1) else -1.0) + 0.25 * (cm0 if np.isfinite(cm0) else -1.0)
        if np.isfinite(rt1):
            score -= 0.05 * rt1

        cand = (score, ct1, cm0, rt1, float(m0_scale))
        if best is None or cand[0] > best[0]:
            best = cand
            best_ops = ops

    return best_ops


def compare_t1m0_to_matlab(
    *,
    subject_root: str | os.PathLike[str],
    analysis_directory: str | os.PathLike[str],
    image_directory: str | os.PathLike[str],
    mat_path: Optional[str | os.PathLike[str]] = None,
) -> Optional[CompareSummary]:
    """Compare `Analysis/Fitting/{t1_map,m0_map}.nii.gz` to MATLAB reference.

    Returns CompareSummary if a reference .mat exists and outputs are present,
    else returns None.
    """

    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore

    try:
        import scipy.io as sio  # type: ignore
    except Exception:
        return None

    subject_root_p = Path(subject_root)
    analysis_p = Path(analysis_directory)
    images_p = Path(image_directory)

    t1_nii = analysis_p / "Fitting" / "t1_map.nii.gz"
    m0_nii = analysis_p / "Fitting" / "m0_map.nii.gz"

    if not t1_nii.exists() or not m0_nii.exists():
        return None

    mat_p: Optional[Path]
    if mat_path:
        mat_p = Path(mat_path)
        if not mat_p.exists():
            return None
    else:
        mat_p = _find_default_mat_path(subject_root_p)
        if not mat_p:
            return None

    md = sio.loadmat(mat_p)
    if "t1_map" not in md or "m0_map" not in md:
        return None

    nifti_t1 = np.asanyarray(nib.load(t1_nii).dataobj)
    nifti_m0 = np.asanyarray(nib.load(m0_nii).dataobj)
    matlab_t1 = np.asarray(md["t1_map"])
    matlab_m0 = np.asarray(md["m0_map"])

    if nifti_t1.shape != matlab_t1.shape or nifti_m0.shape != matlab_m0.shape:
        return None

    t1_scale, t1_units_n, t1_units_m = _maybe_convert_t1_units(nifti_t1=nifti_t1, matlab_t1=matlab_t1)
    a_t1_raw = np.asarray(nifti_t1, dtype=float)
    a_m0_raw = np.asarray(nifti_m0, dtype=float)
    b_t1_raw = np.asarray(matlab_t1, dtype=float)
    b_m0_raw = np.asarray(matlab_m0, dtype=float)

    chosen_ops: Tuple[str, ...] | None = None
    orient_info: Dict[str, Any] | None = None

    # Prefer a deterministic, validator-style NIfTI orientation when enabled.
    # This makes the comparison unambiguous and matches validate_hemisure_20240618x2.py.
    if _getenv_bool("P_BRAIN_COMPARE_MATLAB_USE_VALIDATOR_ORIENT", default=True):
        a_t1_o = _apply_validator_nifti_orientation(a_t1_raw) * float(t1_scale)
        a_m0_o = _apply_validator_nifti_orientation(a_m0_raw)

        # Quick sanity check: if correlation is healthy, keep this path; otherwise fall back.
        base_mask = np.isfinite(a_t1_o) & np.isfinite(b_t1_raw) & (a_t1_o > 0) & (b_t1_raw > 0)
        corr_ok = False
        try:
            if int(np.count_nonzero(base_mask)) > 1000:
                corr = float(np.corrcoef(a_t1_o[base_mask].ravel(), b_t1_raw[base_mask].ravel())[0, 1])
                corr_ok = bool(np.isfinite(corr) and corr > 0.80)
        except Exception:
            corr_ok = False

        if corr_ok:
            a_t1 = a_t1_o
            a_m0 = a_m0_o
            b_t1 = b_t1_raw
            b_m0_raw = b_m0_raw
            orient_info = {
                "nifti_rot90_k": int((os.environ.get("P_BRAIN_VALIDATOR_NIFTI_ROT90_K") or "1").strip() or 1),
                "nifti_flip_lr": _getenv_bool("P_BRAIN_VALIDATOR_NIFTI_FLIP_LR", default=False),
                "nifti_flip_ud": _getenv_bool("P_BRAIN_VALIDATOR_NIFTI_FLIP_UD", default=False),
            }
        else:
            a_t1 = np.asarray(a_t1_raw, dtype=float) * float(t1_scale)
            a_m0 = np.asarray(a_m0_raw, dtype=float)

            chosen_ops = _choose_best_matlab_transform(
                nifti_t1_s=a_t1,
                nifti_m0=a_m0,
                matlab_t1_s=b_t1_raw,
                matlab_m0=b_m0_raw,
            )
            b_t1 = _apply_inplane_ops(b_t1_raw, chosen_ops)
            b_m0_raw = _apply_inplane_ops(b_m0_raw, chosen_ops)
    else:
        a_t1 = np.asarray(a_t1_raw, dtype=float) * float(t1_scale)
        a_m0 = np.asarray(a_m0_raw, dtype=float)

        chosen_ops = _choose_best_matlab_transform(
            nifti_t1_s=a_t1,
            nifti_m0=a_m0,
            matlab_t1_s=b_t1_raw,
            matlab_m0=b_m0_raw,
        )
        b_t1 = _apply_inplane_ops(b_t1_raw, chosen_ops)
        b_m0_raw = _apply_inplane_ops(b_m0_raw, chosen_ops)

    # Base mask: finite + positive in both.
    base = np.isfinite(a_t1) & np.isfinite(b_t1) & np.isfinite(nifti_m0) & np.isfinite(b_m0_raw)

    # Typical tissue range in seconds: keep conservative bounds.
    # This avoids background/forced-min voxels dominating the comparison.
    t1_mask = base & (b_t1 > 0.2) & (b_t1 < 5.0)

    # M0 can be very scanner-dependent; we still compute after scaling.
    b_m0 = b_m0_raw
    m0_mask = base & (b_m0 > 0)

    m0_scale = _fit_scale(a_m0, b_m0, m0_mask)
    a_m0s = a_m0 * float(m0_scale)

    out_metrics = {
        "t1": _metrics(a_t1, b_t1, t1_mask),
        "m0": _metrics(a_m0s, b_m0, m0_mask),
        "t1_mask_voxels": int(np.count_nonzero(t1_mask)),
        "m0_mask_voxels": int(np.count_nonzero(m0_mask)),
    }

    # Validator-style per-slice metrics (used by validate_hemisure_20240618x2.py
    # for the T1/M0 stage): max_abs and mean_abs of (arr - ref) on a display mask.
    try:
        z1_env = (os.environ.get("P_BRAIN_T1_VALIDATE_SLICE") or "").strip()
        z1 = int(z1_env) if z1_env else 0
    except Exception:
        z1 = 0
    if z1 <= 0:
        z0 = int(a_t1.shape[2] // 2)
        z1 = int(z0 + 1)
    z0 = int(max(0, min(int(a_t1.shape[2] - 1), z1 - 1)))

    try:
        import numpy as np  # type: ignore

        t1_pb_sl = np.asarray(a_t1[:, :, z0], dtype=float)
        t1_ref_sl = np.asarray(b_t1[:, :, z0], dtype=float)
        m0_pb_sl = np.asarray(a_m0s[:, :, z0], dtype=float)
        m0_ref_sl = np.asarray(b_m0[:, :, z0], dtype=float)

        # Validator display mask: reference T1 finite and > 0.
        mask_show = np.isfinite(t1_ref_sl) & (t1_ref_sl > 0)
        # Require finiteness in both for diff stats.
        t1_ok = mask_show & np.isfinite(t1_pb_sl)
        m0_ok = mask_show & np.isfinite(m0_pb_sl) & np.isfinite(m0_ref_sl)

        def _diff_stats(a, b, ok):
            if int(np.count_nonzero(ok)) == 0:
                return {"n": 0, "max_abs": None, "mean_abs": None}
            d = (a - b)[ok]
            return {
                "n": int(d.size),
                "max_abs": float(np.max(np.abs(d))),
                "mean_abs": float(np.mean(np.abs(d))),
            }

        out_metrics["slice"] = {
            "z1": int(z1),
            "t1": _diff_stats(t1_pb_sl, t1_ref_sl, t1_ok),
            "m0": _diff_stats(m0_pb_sl, m0_ref_sl, m0_ok),
        }
    except Exception:
        out_metrics["slice"] = {"z1": int(z1)}

    # Write JSON metrics.
    fitting_dir = analysis_p / "Fitting"
    _safe_mkdir(fitting_dir)
    metrics_path = fitting_dir / "compare_matlab_t1m0.json"
    payload = {
        "mat_path": str(mat_p),
        "comparison_mode": "validator_nifti_orient" if orient_info is not None else "brute_matlab_transform",
        "nifti_orientation": orient_info,
        "matlab_transform_ops": list(chosen_ops or ()),
        "t1_units": {"nifti": t1_units_n, "matlab": t1_units_m},
        "t1_scale_applied_to_nifti": float(t1_scale),
        "m0_scale_applied_to_nifti": float(m0_scale),
        "metrics": out_metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write montage PNG.
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        from matplotlib.ticker import ScalarFormatter  # type: ignore

        out_dir = images_p / "Fit"
        _safe_mkdir(out_dir)
        out_png = out_dir / "Compare_MATLAB_T1_M0.png"

        # Show middle slice by default.
        z = int(a_t1.shape[2] // 2)

        def _prep2d(x):
            x2 = np.asarray(x[:, :, z], dtype=float)
            x2 = np.rot90(x2)
            return x2

        t1_a2 = _prep2d(a_t1)
        t1_b2 = _prep2d(b_t1)
        t1_d2 = t1_a2 - t1_b2
        m0_a2 = _prep2d(a_m0s)
        m0_b2 = _prep2d(b_m0)
        m0_d2 = m0_a2 - m0_b2

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax in axes.ravel():
            ax.axis("off")

        im = axes[0, 0].imshow(t1_a2, cmap="viridis")
        axes[0, 0].set_title(f"T1 NIfTI ({t1_units_m})")
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im = axes[0, 1].imshow(t1_b2, cmap="viridis")
        axes[0, 1].set_title("T1 MATLAB")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im = axes[0, 2].imshow(t1_d2, cmap="coolwarm")
        axes[0, 2].set_title("T1 diff")
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im = axes[1, 0].imshow(m0_a2, cmap="magma")
        axes[1, 0].set_title("M0 NIfTI (scaled)")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im = axes[1, 1].imshow(m0_b2, cmap="magma")
        axes[1, 1].set_title("M0 MATLAB")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im = axes[1, 2].imshow(m0_d2, cmap="coolwarm")
        axes[1, 2].set_title("M0 diff")
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Compare vs MATLAB (slice z={z})")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        png_path = str(out_png)
    except Exception:
        png_path = ""

    # Write validator-like slice comparison figure(s) (2x3, robust scaling).
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        from matplotlib.ticker import ScalarFormatter  # type: ignore

        out_dir = images_p / "Fit"
        _safe_mkdir(out_dir)

        def _write_slice_png(z1: int) -> None:
            z0 = int(max(0, min(int(a_t1.shape[2] - 1), int(z1) - 1)))

            # Use MATLAB/reference T1 positivity as display mask (mirrors validator).
            t1_ref = np.asarray(b_t1[:, :, z0], dtype=float)
            mask_show = np.isfinite(t1_ref) & (t1_ref > 0)

            t1_pb = np.asarray(a_t1[:, :, z0], dtype=float)
            m0_pb = np.asarray(a_m0s[:, :, z0], dtype=float)
            m0_ref = np.asarray(b_m0[:, :, z0], dtype=float)

            t1_vmin, t1_vmax = _robust_vmin_vmax(t1_pb, t1_ref, mask=mask_show, pmin=1.0, pmax=99.5)
            m0_vmin, m0_vmax = _robust_vmin_vmax(m0_pb, m0_ref, mask=mask_show, pmin=1.0, pmax=99.5)

            t1_diff = t1_pb - t1_ref
            m0_diff = m0_pb - m0_ref
            t1_lim = _robust_sym_limit(t1_diff, mask=mask_show, p=99.9)
            m0_lim = _robust_sym_limit(m0_diff, mask=mask_show, p=99.9)

            def _masked(x):
                return np.ma.array(np.asarray(x, dtype=float), mask=~mask_show)

            cmap_t1 = plt.get_cmap("viridis").copy()
            cmap_m0 = plt.get_cmap("magma").copy()
            cmap_diff = plt.get_cmap("coolwarm").copy()
            for cm in (cmap_t1, cmap_m0, cmap_diff):
                try:
                    cm.set_bad("white")
                except Exception:
                    pass

            fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300, constrained_layout=True)
            for ax in axes.ravel():
                ax.axis("off")

            im = axes[0, 0].imshow(_masked(t1_pb), cmap=cmap_t1, vmin=t1_vmin, vmax=t1_vmax, interpolation="nearest")
            axes[0, 0].set_title(f"T1 p-brain ({t1_units_m})")
            plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

            im = axes[0, 1].imshow(_masked(t1_ref), cmap=cmap_t1, vmin=t1_vmin, vmax=t1_vmax, interpolation="nearest")
            axes[0, 1].set_title("T1 MATLAB/reference")
            plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

            im = axes[0, 2].imshow(
                _masked(t1_diff),
                cmap=cmap_diff,
                vmin=-t1_lim,
                vmax=t1_lim,
                interpolation="nearest",
            )
            axes[0, 2].set_title("T1 diff")
            cb = plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
            cb.formatter = ScalarFormatter(useMathText=True)
            cb.formatter.set_powerlimits((-3, 3))
            cb.update_ticks()

            im = axes[1, 0].imshow(_masked(m0_pb), cmap=cmap_m0, vmin=m0_vmin, vmax=m0_vmax, interpolation="nearest")
            axes[1, 0].set_title("M0 p-brain (scaled)")
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

            im = axes[1, 1].imshow(_masked(m0_ref), cmap=cmap_m0, vmin=m0_vmin, vmax=m0_vmax, interpolation="nearest")
            axes[1, 1].set_title("M0 MATLAB/reference")
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

            im = axes[1, 2].imshow(
                _masked(m0_diff),
                cmap=cmap_diff,
                vmin=-m0_lim,
                vmax=m0_lim,
                interpolation="nearest",
            )
            axes[1, 2].set_title("M0 diff")
            cb = plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
            cb.formatter = ScalarFormatter(useMathText=True)
            cb.formatter.set_powerlimits((-3, 3))
            cb.update_ticks()

            fig.suptitle(f"Compare vs MATLAB/reference (slice z={int(z1)})")
            out_path = out_dir / f"t1_m0_compare_slice{int(z1)}.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)

        write_all = (os.environ.get("P_BRAIN_T1M0_WRITE_ALL_SLICES") or "").strip().lower() in {"1", "true", "yes", "on"}
        if write_all:
            for z1 in range(1, int(a_t1.shape[2]) + 1):
                _write_slice_png(z1)
        else:
            # 1-based slice index for display; default: middle slice.
            try:
                z1 = int((os.environ.get("P_BRAIN_T1_VALIDATE_SLICE") or "0").strip())
            except Exception:
                z1 = 0
            if z1 <= 0:
                z0 = int(a_t1.shape[2] // 2)
                z1 = int(z0 + 1)
            _write_slice_png(z1)
    except Exception:
        pass

    return CompareSummary(
        mat_path=str(mat_p),
        t1_units_matlab=t1_units_m,
        t1_units_nifti=t1_units_n,
        t1_scale_applied_to_nifti=float(t1_scale),
        m0_scale_applied_to_nifti=float(m0_scale),
        metrics=out_metrics,
        paths={
            "metrics": str(metrics_path),
            "png": png_path,
        },
    )
