"""p-brain-web Defaults support.

p-brain-web (Tauri) drives the Python runner via explicit user-facing Defaults.
This module provides a stable, minimal contract: a JSON file can be passed to
`main.py --defaults-json <path>` and will be translated into the corresponding
`P_BRAIN_*` environment variables and (where needed) `utils.settings` values.

Design goals:
- Keep p-brain's validated math as the single source of truth.
- Make selection deterministic for p-brain-web runs.
- Avoid adding new modelling implementations or alternative code paths.

The JSON schema is intentionally permissive. Supported keys (case-insensitive):
- t1RecoveryModel / t1_recovery_model: inversion | saturation | turboflash
- pkModel / model / kineticModel / p_brain_model: patlak | tikhonov | both
- ctcModel / ctc_model: MUST be turboflash (only supported model)
- turboFlashBaselineFrames / turboflash_baseline_frames: int
- vascularRoiCurveMethod / vascular_roi_curve_method: max | mean | median
- writeSliceDiagnostics / write_slice_diagnostics: bool
- skipForkedMaxCtcPeaks / skip_forked_max_ctc_peaks: bool (default: true)
- writeCtcMaps / write_ctc_maps: bool (default: true)
- writeCtc4d / write_ctc_4d: bool (default: true)
- ctcMapSlice / ctc_map_slice: int (1-based, default: 5)

The env vars written are:
- P_BRAIN_T1_RECOVERY_MODEL
- P_BRAIN_MODEL
- P_BRAIN_TURBOFLASH_BASELINE_FRAMES
- P_BRAIN_VASCULAR_ROI_CURVE_METHOD
- P_BRAIN_WRITE_SLICE_DIAGNOSTICS
- P_BRAIN_CTC_MODEL (forced to turboflash when provided)
- P_BRAIN_TSCC_SKIP_FORKED_PEAKS
- P_BRAIN_CTC_MAPS
- P_BRAIN_CTC_4D
- P_BRAIN_CTC_MAP_SLICE
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import utils.settings as settings


def _norm_key(key: str) -> str:
    return "".join(ch.lower() for ch in str(key) if ch.isalnum() or ch in {"_"})


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    raw = str(value).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _set_env(name: str, value: Any) -> None:
    if value is None:
        return
    os.environ[str(name)] = str(value)


def _apply_t1_recovery_model(raw: Any) -> None:
    if raw is None:
        return
    model = str(raw).strip().lower()
    if model not in {"inversion", "saturation", "turboflash"}:
        raise ValueError(
            "t1RecoveryModel must be one of inversion|saturation|turboflash; "
            f"got {raw!r}"
        )
    _set_env("P_BRAIN_T1_RECOVERY_MODEL", model)
    settings.T1_RECOVERY_MODEL = model


def _apply_pk_model(raw: Any) -> None:
    if raw is None:
        return
    val = str(raw).strip().lower()
    # Accept a few legacy UI spellings.
    if val in {"two_compartment", "2comp", "two-comp", "two-compartment"}:
        val = "tikhonov"
    if val not in {"patlak", "tikhonov", "both"}:
        raise ValueError("pkModel must be patlak|tikhonov|both; got %r" % (raw,))

    # Keep the external contract stable for p-brain-web.
    _set_env("P_BRAIN_MODEL", val)

    # Update settings to match the same mapping used in utils/settings.py.
    if val == "patlak":
        settings.KINETIC_MODEL = "patlak"
    elif val == "tikhonov":
        settings.KINETIC_MODEL = "tikhonov"
    else:
        settings.KINETIC_MODEL = "both"


def _apply_ctc_model(raw: Any) -> None:
    if raw is None:
        return
    model = str(raw).strip().lower()
    if model != "turboflash":
        raise ValueError(
            "ctcModel must be 'turboflash' (validator-parity only); got %r" % (raw,)
        )
    # settings.py enforces this, but keep env explicit for runtime_metadata.
    _set_env("P_BRAIN_CTC_MODEL", "turboflash")
    settings.CTC_MODEL = "turboflash"


def _apply_vascular_roi_curve_method(raw: Any) -> None:
    if raw is None:
        return
    method = str(raw).strip().lower()
    if method not in {"max", "mean", "median"}:
        raise ValueError(
            "vascularRoiCurveMethod must be max|mean|median; got %r" % (raw,)
        )
    _set_env("P_BRAIN_VASCULAR_ROI_CURVE_METHOD", method)
    settings.VASCULAR_ROI_CURVE_METHOD = method


def apply_defaults_json(path: str, *, args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Apply Defaults JSON to env vars + settings.

    Returns the parsed JSON dict.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Defaults JSON must be an object")

    # Normalize supported keys.
    norm = {_norm_key(k): v for k, v in data.items()}

    # Apply models first (they affect downstream defaults).
    _apply_ctc_model(norm.get("ctcmodel") or norm.get("ctc_model"))
    _apply_t1_recovery_model(
        norm.get("t1recoverymodel")
        or norm.get("t1_recovery_model")
        or norm.get("pbrain_t1_recovery_model")
    )
    _apply_pk_model(
        norm.get("pkmodel")
        or norm.get("model")
        or norm.get("kineticmodel")
        or norm.get("p_brain_model")
        or norm.get("pbrain_model")
    )

    _apply_vascular_roi_curve_method(
        norm.get("vascularroicurvemethod")
        or norm.get("vascular_roi_curve_method")
        or norm.get("vascularcurvemethod")
        or norm.get("vascular_curve_method")
        or norm.get("roicurvemethod")
        or norm.get("roi_curve_method")
    )

    # Baseline frames (TurboFLASH conversion).
    baseline = (
        norm.get("turboflashbaselineframes")
        or norm.get("turboflash_baseline_frames")
        or norm.get("pbrain_turboflash_baseline_frames")
    )
    if baseline is not None:
        try:
            baseline_i = int(baseline)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("turboflashBaselineFrames must be an int") from exc
        if baseline_i < 0:
            raise ValueError("turboflashBaselineFrames must be >= 0")
        _set_env("P_BRAIN_TURBOFLASH_BASELINE_FRAMES", str(baseline_i))
        settings.TURBOFLASH_BASELINE_FRAMES = baseline_i

    write_diag = (
        norm.get("writeslicediagnostics")
        or norm.get("write_slice_diagnostics")
        or norm.get("pbrain_write_slice_diagnostics")
    )
    if write_diag is not None:
        _set_env("P_BRAIN_WRITE_SLICE_DIAGNOSTICS", "1" if _as_bool(write_diag) else "0")

    skip_forked = (
        norm.get("skipforkedmaxctcpeaks")
        or norm.get("skip_forked_max_ctc_peaks")
        or norm.get("tsccskipforkedpeaks")
        or norm.get("tscc_skip_forked_peaks")
    )
    if skip_forked is not None:
        val = _as_bool(skip_forked)
        _set_env("P_BRAIN_TSCC_SKIP_FORKED_PEAKS", "1" if val else "0")
        try:
            settings.TSCC_SKIP_FORKED_PEAKS = val
        except Exception:
            pass

    write_maps = norm.get("writectcmaps") or norm.get("write_ctc_maps")
    if write_maps is not None:
        val = _as_bool(write_maps)
        _set_env("P_BRAIN_CTC_MAPS", "1" if val else "0")
        try:
            settings.CTC_WRITE_MAPS = val
        except Exception:
            pass

    write_4d = norm.get("writectc4d") or norm.get("write_ctc_4d")
    if write_4d is not None:
        val = _as_bool(write_4d)
        _set_env("P_BRAIN_CTC_4D", "1" if val else "0")
        try:
            settings.CTC_WRITE_4D = val
        except Exception:
            pass

    map_slice = norm.get("ctcmapslice") or norm.get("ctc_map_slice")
    if map_slice is not None:
        try:
            slice_1b = int(float(map_slice))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("ctcMapSlice must be an int") from exc
        if slice_1b < 1:
            raise ValueError("ctcMapSlice must be >= 1")
        _set_env("P_BRAIN_CTC_MAP_SLICE", str(slice_1b))
        try:
            settings.CTC_MAP_SLICE = slice_1b
        except Exception:
            pass

    # Record a stable marker for p-brain-web log parsers / runtime metadata.
    _set_env("P_BRAIN_DEFAULTS_JSON", os.path.abspath(path))

    return data
