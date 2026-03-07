"""Quality control (QC) utilities for p-Brain.

Writes structured QC reports into `Analysis/qc/`.

Design goals:
- Deterministic and machine-readable (JSON)
- Stage-scoped (t1_fit, input_functions, time_shift, segmentation, tissue_ctc, modelling, diffusion)
- Lightweight (no heavy deps beyond what p-Brain already uses)
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _top_peaks_by_height(curve, *, num_peaks=2, separation=5):
    """Return up to `num_peaks` peak indices sorted by height."""
    import numpy as np  # type: ignore

    curve = np.asarray(curve, dtype=float)
    if curve.size < 3:
        return []

    try:
        from scipy.signal import find_peaks  # type: ignore

        peaks, _ = find_peaks(curve)
    except Exception:
        peaks = []

    if len(peaks) == 0:
        return []

    peaks = [int(p) for p in peaks if np.isfinite(curve[int(p)])]
    peaks.sort(key=lambda p: float(curve[p]), reverse=True)

    chosen = []
    for p in peaks:
        if float(curve[p]) <= 0:
            continue
        if all(abs(p - c) > separation for c in chosen):
            chosen.append(p)
            if len(chosen) >= int(num_peaks):
                break
    return chosen


def detect_truncated_bolus(
    curve,
    *,
    num_peaks=2,
    peak_fraction=0.99,
    plateau_min_points=3,
    plateau_slope_fraction=0.02,
    edge_window=2,
):
    """Heuristically detect truncated/clipped bolus peaks.

    Returns
    -------
    (is_truncated, details)

    Heuristics (robust, low-assumption):
    - Flat/plateau region around a dominant peak (many points near max with near-zero slope)
    - Dominant peak occurring at the very beginning/end of the series
    """
    import numpy as np  # type: ignore

    arr = np.asarray(curve, dtype=float)
    details = {
        "reason": None,
        "peak_indices": [],
        "plateau_width": None,
        "peak_value": None,
    }

    if arr.size < 8:
        return False, details

    finite = np.isfinite(arr)
    if not np.any(finite):
        return False, details
    arr = arr.copy()
    arr[~finite] = np.nan

    # Keep it light: no filtering beyond what caller provides.
    smoothed = arr

    finite_s = np.isfinite(smoothed)
    if np.sum(finite_s) < 8:
        return False, details

    peaks = _top_peaks_by_height(smoothed, num_peaks=num_peaks, separation=5)
    if not peaks:
        peak_idx = int(np.nanargmax(smoothed))
        peaks = [peak_idx]

    details["peak_indices"] = peaks

    n = int(smoothed.size)
    for peak_idx in peaks:
        peak_val = float(smoothed[peak_idx])
        if not np.isfinite(peak_val) or peak_val <= 0:
            continue

        if peak_idx <= int(edge_window) or peak_idx >= (n - 1 - int(edge_window)):
            details.update({"reason": "edge_peak", "peak_value": peak_val, "plateau_width": 1})
            return True, details

        thr = peak_val * float(peak_fraction)
        left = peak_idx
        while left - 1 >= 0 and np.isfinite(smoothed[left - 1]) and float(smoothed[left - 1]) >= thr:
            left -= 1
        right = peak_idx
        while right + 1 < n and np.isfinite(smoothed[right + 1]) and float(smoothed[right + 1]) >= thr:
            right += 1

        width = int(right - left + 1)
        if width >= int(plateau_min_points):
            plateau = smoothed[left : right + 1]
            diffs = np.diff(plateau)
            max_step = float(np.nanmax(np.abs(diffs))) if diffs.size else 0.0
            if max_step <= float(plateau_slope_fraction) * max(1e-9, abs(peak_val)):
                details.update({"reason": "plateau_near_peak", "peak_value": peak_val, "plateau_width": width})
                return True, details

            ptp = float(np.nanmax(plateau) - np.nanmin(plateau))
            if ptp <= (1e-6 + 1e-3 * abs(peak_val)):
                details.update({"reason": "flat_top_quantized", "peak_value": peak_val, "plateau_width": width})
                return True, details

    return False, details


def _now_iso() -> str:
    # Keep it simple; no timezone dependency.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class QcCheck:
    id: str
    status: str  # pass|warn|fail
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QcReport:
    schemaVersion: str
    createdAt: str
    stage: str
    overallStatus: str  # pass|warn|fail
    checks: List[QcCheck]
    metrics: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)


def _qc_dir(analysis_directory: str) -> str:
    return os.path.join(analysis_directory, "qc")


def _safe_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _summarize(checks: List[QcCheck]) -> str:
    # fail beats warn beats pass
    st = "pass"
    for c in checks:
        if c.status == "fail":
            return "fail"
        if c.status == "warn":
            st = "warn"
    return st


def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _glob(pattern: str) -> List[str]:
    try:
        return sorted(glob.glob(pattern, recursive=True))
    except Exception:
        return []


def _try_load_json(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def _nifti_nonzero_voxels(path: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore

        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        nnz = int(np.count_nonzero(data > 0))
        return nnz, None
    except Exception as e:
        return None, str(e)


def _add(checks: List[QcCheck], *, id: str, status: str, message: str, **details: Any) -> None:
    checks.append(QcCheck(id=id, status=status, message=message, details=details or {}))


def run_stage_qc(
    *,
    stage: str,
    analysis_directory: str,
    nifti_directory: Optional[str] = None,
    image_directory: Optional[str] = None,
    settings_snapshot: Optional[Dict[str, Any]] = None,
) -> QcReport:
    stage = (stage or "").strip().lower()
    checks: List[QcCheck] = []

    analysis_directory = os.path.abspath(analysis_directory)
    if nifti_directory:
        nifti_directory = os.path.abspath(nifti_directory)
    if image_directory:
        image_directory = os.path.abspath(image_directory)

    # --- Common sanity -----------------------------------------------------
    if not os.path.isdir(analysis_directory):
        _add(checks, id="analysis_dir", status="fail", message="Analysis directory missing", path=analysis_directory)
        return QcReport(
            schemaVersion="1",
            createdAt=_now_iso(),
            stage=stage,
            overallStatus=_summarize(checks),
            checks=checks,
            paths={"analysis": analysis_directory},
            environment=settings_snapshot or {},
        )

    # --- Stage-specific checks --------------------------------------------
    if stage in {"t1_fit", "t1", "t1_m0"}:
        t1_pkl = os.path.join(analysis_directory, "Fitting", "voxel_T1_matrix.pkl")
        m0_pkl = os.path.join(analysis_directory, "Fitting", "voxel_M0_matrix.pkl")
        if _exists(t1_pkl) and _exists(m0_pkl):
            _add(checks, id="t1m0_pickles", status="pass", message="T1/M0 fit outputs present", t1=t1_pkl, m0=m0_pkl)
        else:
            _add(
                checks,
                id="t1m0_pickles",
                status="fail",
                message="Missing T1/M0 fit outputs",
                t1_exists=_exists(t1_pkl),
                m0_exists=_exists(m0_pkl),
            )

    if stage in {"input_functions", "aif", "vif"}:
        # Determine ROI method from settings snapshot to tailor checks.
        _roi_method = ""
        if settings_snapshot:
            _roi_method = str(settings_snapshot.get("ROI_METHOD") or "").strip().lower()

        roi_dir = os.path.join(analysis_directory, "ROI NIfTI")
        masks = _glob(os.path.join(roi_dir, "*.nii*"))

        # AI pipeline does not produce ROI NIfTI masks; it produces ROI Data
        # voxel arrays, CTC Data, and ITC Data instead.  Check those when the
        # ROI method is "ai" (or when ROI NIfTI is absent but AI outputs exist).
        ai_roi_dir = os.path.join(analysis_directory, "ROI Data")
        ai_ctc_dir = os.path.join(analysis_directory, "CTC Data")
        ai_itc_dir = os.path.join(analysis_directory, "ITC Data")
        ai_roi_files = _glob(os.path.join(ai_roi_dir, "**", "*.npy"))
        ai_ctc_files = _glob(os.path.join(ai_ctc_dir, "**", "*.npy"))
        ai_itc_files = _glob(os.path.join(ai_itc_dir, "**", "*.npy"))
        has_ai_outputs = bool(ai_roi_files or ai_ctc_files or ai_itc_files)

        if masks:
            # Geometry / deterministic path — check mask quality.
            nonempty = 0
            failures: List[Dict[str, Any]] = []
            for p in masks:
                nnz, err = _nifti_nonzero_voxels(p)
                if err:
                    failures.append({"path": p, "error": err})
                    continue
                if nnz and nnz > 0:
                    nonempty += 1
            if failures:
                _add(checks, id="roi_masks_load", status="warn", message="Some ROI masks could not be read", failures=failures)
            if nonempty >= 3:
                _add(checks, id="roi_masks_nonempty", status="pass", message="ROI masks look non-empty", count=nonempty, total=len(masks))
            elif nonempty > 0:
                _add(checks, id="roi_masks_nonempty", status="warn", message="Few non-empty ROI masks", count=nonempty, total=len(masks))
            else:
                _add(checks, id="roi_masks_nonempty", status="fail", message="All ROI masks appear empty", total=len(masks))
        elif has_ai_outputs:
            # AI pipeline — validate that ROI, CTC, and ITC outputs exist.
            _add(
                checks,
                id="roi_masks_present",
                status="pass",
                message="AI ROI outputs present (ROI Data / CTC Data / ITC Data)",
                roi_files=len(ai_roi_files),
                ctc_files=len(ai_ctc_files),
                itc_files=len(ai_itc_files),
            )
        else:
            _add(checks, id="roi_masks_present", status="fail", message="No ROI masks found", roi_dir=roi_dir)

    if stage in {"time_shift", "timeshift", "tscc"}:
        max_info = os.path.join(analysis_directory, "max_info.json")
        tscc_sel = os.path.join(analysis_directory, "tscc_selection.json")

        if _exists(max_info):
            blob, err = _try_load_json(max_info)
            if err:
                _add(checks, id="max_info", status="warn", message="max_info.json unreadable", error=err)
            else:
                _add(checks, id="max_info", status="pass", message="max_info.json present", keys=sorted(list(blob.keys())) if isinstance(blob, dict) else [])
        else:
            _add(checks, id="max_info", status="fail", message="Missing max_info.json", path=max_info)

        use_sss = None
        if settings_snapshot is not None:
            use_sss = settings_snapshot.get("INPUT_FUNCTION_USE_SSS")
        if _exists(tscc_sel):
            blob, err = _try_load_json(tscc_sel)
            if err:
                _add(checks, id="tscc_selection", status="warn", message="tscc_selection.json unreadable", error=err)
            else:
                _add(checks, id="tscc_selection", status="pass", message="tscc_selection.json present", blob=blob)
        else:
            # If the run is configured to use pure artery, missing TSCC selection is acceptable.
            if use_sss is False:
                _add(checks, id="tscc_selection", status="warn", message="tscc_selection.json missing (pure artery configured)")
            else:
                _add(checks, id="tscc_selection", status="fail", message="Missing tscc_selection.json", path=tscc_sel)

    if stage in {"segmentation"}:
        if nifti_directory:
            seg_mgz = os.path.join(nifti_directory, "segmentation", "segmentation", "mri", "aparc.DKTatlas+aseg.deep.mgz")
            if _exists(seg_mgz):
                _add(checks, id="seg_mgz", status="pass", message="Segmentation output present", path=seg_mgz)
            else:
                _add(checks, id="seg_mgz", status="fail", message="Segmentation output missing", path=seg_mgz)
        else:
            _add(checks, id="seg_mgz", status="warn", message="No nifti_directory provided; cannot verify segmentation output")

    if stage in {"tissue_ctc"}:
        tissue_dir = os.path.join(analysis_directory, "CTC Data", "Tissue")
        candidates = _glob(os.path.join(tissue_dir, "**", "*"))
        files = [p for p in candidates if os.path.isfile(p)]
        if files:
            _add(checks, id="tissue_ctc_files", status="pass", message="Tissue CTC outputs present", count=len(files), dir=tissue_dir)
        else:
            _add(checks, id="tissue_ctc_files", status="fail", message="No tissue CTC outputs found", dir=tissue_dir)

    if stage in {"modelling", "model"}:
        # Model outputs may be renamed with suffixes (_patlak/_tikhonov/_two_compartment).
        patterns = [
            os.path.join(analysis_directory, "AI_values_median*.json"),
            os.path.join(analysis_directory, "AI_values_median_total*.json"),
            os.path.join(analysis_directory, "Ki_wm*.nii.gz"),
            os.path.join(analysis_directory, "Ki_per_voxel*.nii.gz"),
            os.path.join(analysis_directory, "vp_per_voxel*.nii.gz"),
            os.path.join(analysis_directory, "CBF_per_voxel*.nii.gz"),
        ]
        found: Dict[str, List[str]] = {p: _glob(p) for p in patterns}
        have_any_json = bool(found[patterns[0]]) and bool(found[patterns[1]])
        have_any_maps = any(bool(found[p]) for p in patterns[2:])

        if have_any_json:
            _add(checks, id="model_json", status="pass", message="Model summary JSON outputs present", median=found[patterns[0]], total=found[patterns[1]])
        else:
            # Segmentation-free (voxelwise-only) runs may not produce atlas/tissue summary JSON.
            # Accept map-only outputs as a successful modelling run.
            status = "warn" if have_any_maps else "fail"
            _add(checks, id="model_json", status=status, message="Model summary JSON outputs missing (maps-only run is acceptable)", found=found)

        if have_any_maps:
            preview: List[str] = []
            for p in patterns[2:]:
                if found.get(p):
                    preview.extend(found[p][:2])
                if len(preview) >= 6:
                    break
            _add(checks, id="model_maps", status="pass", message="Model map outputs present", maps=preview)
        else:
            _add(checks, id="model_maps", status="warn", message="No model map outputs found", patterns=patterns[2:])

    if stage in {"diffusion", "dti"}:
        diffusion_dir = os.path.join(analysis_directory, "diffusion")
        if not os.path.isdir(diffusion_dir):
            _add(checks, id="diffusion_dir", status="fail", message="Diffusion output directory missing", dir=diffusion_dir)
        else:
            patterns = [
                os.path.join(diffusion_dir, "diffusion_values_median_total.json"),
                os.path.join(diffusion_dir, "diffusion_values_atlas.json"),
                os.path.join(diffusion_dir, "*_map.nii.gz"),
                os.path.join(diffusion_dir, "fa_mean.txt"),
            ]
            found_json_total = _exists(patterns[0])
            found_maps = _glob(patterns[2])
            found_any_txt = _exists(patterns[3])

            if found_json_total or found_maps:
                _add(
                    checks,
                    id="diffusion_outputs",
                    status="pass",
                    message="Diffusion outputs present",
                    diffusion_dir=diffusion_dir,
                    diffusion_values_total_exists=found_json_total,
                    maps_count=len(found_maps),
                )
            elif found_any_txt:
                _add(
                    checks,
                    id="diffusion_outputs",
                    status="warn",
                    message="Only minimal diffusion summary present",
                    diffusion_dir=diffusion_dir,
                    fa_mean_txt=patterns[3],
                )
            else:
                _add(
                    checks,
                    id="diffusion_outputs",
                    status="fail",
                    message="No diffusion outputs found",
                    diffusion_dir=diffusion_dir,
                )

    overall = _summarize(checks)
    return QcReport(
        schemaVersion="1",
        createdAt=_now_iso(),
        stage=stage,
        overallStatus=overall,
        checks=checks,
        paths={
            "analysis": analysis_directory,
            **({"nifti": nifti_directory} if nifti_directory else {}),
            **({"images": image_directory} if image_directory else {}),
        },
        environment=settings_snapshot or {},
    )


def persist_report(*, report: QcReport, analysis_directory: str) -> Dict[str, str]:
    qdir = _qc_dir(analysis_directory)
    os.makedirs(qdir, exist_ok=True)

    stage = (report.stage or "unknown").strip().lower()
    stamp = report.createdAt.replace(":", "").replace("-", "")
    out_path = os.path.join(qdir, f"qc_{stage}_{stamp}.json")
    latest_stage = os.path.join(qdir, f"qc_{stage}_latest.json")
    latest_any = os.path.join(qdir, "qc_latest.json")

    payload = asdict(report)
    _safe_write_json(out_path, payload)
    _safe_write_json(latest_stage, payload)
    _safe_write_json(latest_any, payload)

    return {"report": out_path, "latestStage": latest_stage, "latest": latest_any}


def run_and_persist(
    *,
    stage: str,
    analysis_directory: str,
    nifti_directory: Optional[str] = None,
    image_directory: Optional[str] = None,
    settings_snapshot: Optional[Dict[str, Any]] = None,
) -> Tuple[QcReport, Dict[str, str]]:
    report = run_stage_qc(
        stage=stage,
        analysis_directory=analysis_directory,
        nifti_directory=nifti_directory,
        image_directory=image_directory,
        settings_snapshot=settings_snapshot,
    )
    paths = persist_report(report=report, analysis_directory=analysis_directory)
    return report, paths
