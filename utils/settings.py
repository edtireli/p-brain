import os
import sys
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import matplotlib

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback for very old Python versions
    import importlib_metadata  # type: ignore
_MPL_ENV_BACKEND = os.environ.get("P_BRAIN_MPL_BACKEND") or os.environ.get("MPLBACKEND")
if _MPL_ENV_BACKEND:
    matplotlib.use(_MPL_ENV_BACKEND)
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")

# Global toggle for analysing control data. The flag can also be
# overridden by setting the environment variable ``PBRAIN_CONTROLS`` to
# ``1``/``true``.
CONTROLS = 0#os.environ.get("PBRAIN_CONTROLS", "False").lower() in ("1", "true", "yes")
CONTROL_FLAG_FILENAME = "control.json"

# Toggle multiprocessing for compute-heavy routines.
# Default on, but allow disabling via env when debugging or when the Python
# runtime has issues with multiprocessing (e.g. some macOS setups).
MULTIPROCESSING = (os.environ.get("P_BRAIN_MULTIPROCESSING", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _default_core_count() -> int:
    ncpu = os.cpu_count() or 1
    # Default to 80% of available cores, clamped.
    return max(1, int(ncpu * 0.8))


def _parse_core_setting(raw: str | None) -> int:
    """Parse P_BRAIN_CORES.

    Supported values:
    - unset/empty/"auto": 80% of available CPUs
    - integer (e.g. 4)
    - fraction in (0,1] (e.g. 0.8 => 80% of CPUs)
    """

    raw = (raw or "").strip().lower()
    if not raw or raw == "auto":
        return _default_core_count()

    try:
        if any(ch in raw for ch in (".", "e")):
            frac = float(raw)
            if 0 < frac <= 1:
                return max(1, int((os.cpu_count() or 1) * frac))
            return max(1, int(frac))
        return max(1, int(raw))
    except Exception:
        return _default_core_count()


# Default number of CPU cores used when multiprocessing is enabled.
_cores = _parse_core_setting(os.environ.get("P_BRAIN_CORES"))
if not MULTIPROCESSING:
    _cores = 1
else:
    ncpu = os.cpu_count() or 1
    _cores = max(1, min(int(_cores), int(ncpu)))
NUMBER_OF_CORES = int(_cores)

# Input-function ROI extraction method.
#
# - ai (default): TensorFlow-backed AI ROI extraction.
# - deterministic: deterministic (non-AI) ROI extraction using PCA/connected-components.
# - file: load ROI masks/curves from files (e.g. user-drawn ROI exported by p-brain-web).
# - geometry: legacy alias for deterministic.
ROI_METHOD = (os.environ.get("P_BRAIN_ROI_METHOD") or os.environ.get("ROI_METHOD") or "ai").strip().lower()
if ROI_METHOD == "geometry":
    ROI_METHOD = "deterministic"
if ROI_METHOD not in {"ai", "deterministic", "file", "manual"}:
    ROI_METHOD = "ai"

# Metadata strictness / overrides.
# If STRICT_METADATA is enabled, missing required acquisition metadata should be treated as an error.
STRICT_METADATA = (os.environ.get("P_BRAIN_STRICT_METADATA", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Optional flip-angle override (degrees).
# NOTE: The primary flip-angle configuration is defined further below as
# FLIP_ANGLE_SETTING/FLIP_ANGLE_DEG. This early parsing exists for historical
# reasons; keep it robust to non-numeric values like "auto".
_fa_env = os.environ.get("P_BRAIN_FLIP_ANGLE") or os.environ.get("P_BRAIN_FLIP_ANGLE_DEG")
try:
    _fa_raw = (str(_fa_env).strip() if _fa_env not in {None, ""} else "")
    if not _fa_raw or _fa_raw.lower() == "auto":
        FLIP_ANGLE_DEG = None
    else:
        FLIP_ANGLE_DEG = float(_fa_raw)
except Exception:
    FLIP_ANGLE_DEG = None

# Signal->concentration conversion model.
# Validator parity requirement: only the validated TurboFLASH conversion is supported.
CTC_MODEL = "turboflash"

# Baseline frames for TurboFLASH conversion (validator default = 10 frames).
TURBOFLASH_BASELINE_FRAMES = int(os.environ.get("P_BRAIN_TURBOFLASH_BASELINE_FRAMES", "10"))

# Control exporting voxelwise CTC outputs.
# Defaults enabled to match the validated pipeline expectations.
CTC_WRITE_MAPS = (os.environ.get("P_BRAIN_CTC_MAPS", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CTC_WRITE_4D = (os.environ.get("P_BRAIN_CTC_4D", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
try:
    CTC_MAP_SLICE = int(float(os.environ.get("P_BRAIN_CTC_MAP_SLICE", "5") or "5"))
except Exception:
    CTC_MAP_SLICE = 5

# When selecting the global "Max" TSCC curve, optionally skip curves whose main
# bolus apex appears as a narrow split/fork (two near-maximum local maxima with
# a dip in-between). Enabled by default to avoid unstable max selection.
TSCC_SKIP_FORKED_PEAKS = (os.environ.get("P_BRAIN_TSCC_SKIP_FORKED_PEAKS", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Conservative defaults for forked-peak detection (see modules/time_shifting.py).
TSCC_FORKED_PEAK_MIN_PEAK_MM = float(os.environ.get("P_BRAIN_TSCC_FORKED_PEAK_MIN_PEAK_MM", 1.0))
TSCC_FORKED_PEAK_NEAR_FRAC = float(os.environ.get("P_BRAIN_TSCC_FORKED_PEAK_NEAR_FRAC", 0.95))
TSCC_FORKED_PEAK_WINDOW_FRAMES = int(os.environ.get("P_BRAIN_TSCC_FORKED_PEAK_WINDOW_FRAMES", 3))
TSCC_FORKED_PEAK_MAX_SEPARATION_FRAMES = int(
    os.environ.get("P_BRAIN_TSCC_FORKED_PEAK_MAX_SEPARATION_FRAMES", 3)
)
TSCC_FORKED_PEAK_MIN_DIP_FRAC = float(os.environ.get("P_BRAIN_TSCC_FORKED_PEAK_MIN_DIP_FRAC", 0.06))

# Select the recovery equation to use when fitting T1/M0 from inversion or
# saturation recovery acquisitions.
# - inversion:  A - B*exp(-TI/T1)
# - saturation: M0*(1-exp(-TI/T1))
# - turboflash: TurboFLASH readout-train model (matches legacy MATLAB turbof_* fit)
T1_RECOVERY_MODEL = os.environ.get("P_BRAIN_T1_RECOVERY_MODEL", "inversion").strip().lower()
if T1_RECOVERY_MODEL not in {"inversion", "saturation", "turboflash"}:
    T1_RECOVERY_MODEL = "inversion"

# Select kinetic modelling strategy.
# Consolidated options:
# - patlak: permeability only
# - tikhonov: deconvolution/perfusion only
# - both: run patlak then tikhonov
# - extended_tofts: Extended Tofts model only
# - all: run patlak + tikhonov + extended tofts
# - "+"-delimited combos: e.g. patlak+etofts, etofts+patlak+tikhonov
_KINETIC_MODEL_RAW = (os.environ.get("P_BRAIN_MODEL", "both") or "both").strip().lower()
try:
    from models import normalise_pk_model as _normalise_pk
    KINETIC_MODEL = _normalise_pk(_KINETIC_MODEL_RAW)
except Exception:
    # Fallback when models package is not importable (early bootstrap).
    if _KINETIC_MODEL_RAW == "all":
        KINETIC_MODEL = "all"
    elif _KINETIC_MODEL_RAW in {
        "both",
        "patlak_then_tikhonov",
        "patlak-then-tikhonov",
        "patlak_tikhonov",
        "patlak_tikhonov_fast",
        "patlak-then-tikhonov-fast",
    }:
        KINETIC_MODEL = "both"
    elif _KINETIC_MODEL_RAW in {
        "tikhonov",
        "tikhonov_only",
        "tikhonov_fast",
        "tikhonov-only-fast",
        "two_compartment",
        "2comp",
        "two-comp",
        "two-compartment",
    }:
        KINETIC_MODEL = "tikhonov"
    elif _KINETIC_MODEL_RAW in {"patlak"}:
        KINETIC_MODEL = "patlak"
    elif _KINETIC_MODEL_RAW in {
        "extended_tofts",
        "etofts",
        "etm",
        "tofts",
        "extended-tofts",
    }:
        KINETIC_MODEL = "extended_tofts"
    else:
        KINETIC_MODEL = "both"

# Optionally use the Patlak permeability (Ki) as the initial guess for
# Ktrans in the two-compartment optimisation.
TWO_COMPARTMENT_INIT_FROM_PATLAK = (
    os.environ.get("P_BRAIN_TWO_COMPARTMENT_INIT_FROM_PATLAK", "0") or "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Control writing of additional voxelwise parametric maps derived from the
# deconvolution residue.
WRITE_MTT = True
WRITE_CTH = True
CTH_MTT_METHOD = os.environ.get("P_BRAIN_CTH_MTT_METHOD", "tikhonov")
CTH_MTT_GAMMA_VOXELWISE = os.environ.get(
    "P_BRAIN_CTH_MTT_GAMMA_VOXELWISE", "0"
).lower() in {"1", "true", "yes"}

# Regularisation defaults for Tikhonov deconvolution (CBF/MTT/CTH).
# Use L-curve by default across a MATLAB-matched grid (0.05:0.3329:40).
_lambda_env = os.environ.get("P_BRAIN_LAMBDA")
TIKHONOV_LAMBDA = float(_lambda_env) if _lambda_env not in {None, ""} else 0.5
TIKHONOV_LAMBDA_MIN = float(os.environ.get("P_BRAIN_LAMBDA_MIN", 0.05))
TIKHONOV_LAMBDA_MAX = float(os.environ.get("P_BRAIN_LAMBDA_MAX", 40.0))
TIKHONOV_LAMBDA_STEPS = int(os.environ.get("P_BRAIN_LAMBDA_STEPS", 121))
# If true, prefer the L-curve selection over any fixed lambda.
AUTO_LAMBDA = (os.environ.get("P_BRAIN_AUTO_LAMBDA", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}

# Tikhonov penalty operator used for deconvolution/residue-based metrics.
# Supported values: "identity" (0th-order) and "derivative" (1st-order finite difference).
TIKHONOV_PENALTY = (os.environ.get("P_BRAIN_TIKHONOV_PENALTY", "derivative") or "derivative").strip().lower()
if TIKHONOV_PENALTY not in {"identity", "derivative"}:
    TIKHONOV_PENALTY = "derivative"

# Stabilisation applied when computing MTT/CTH from the residue.
RESIDUE_ENFORCE_NONNEG = (os.environ.get("P_BRAIN_RESIDUE_ENFORCE_NONNEG", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RESIDUE_ENFORCE_MONOTONE = (os.environ.get("P_BRAIN_RESIDUE_ENFORCE_MONOTONE", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Physiological constants for residue-based perfusion metrics
TISSUE_DENSITY = float(os.environ.get("P_BRAIN_TISSUE_DENSITY", 1.04))
HEMATOCRIT = float(os.environ.get("P_BRAIN_HEMATOCRIT", 0.42))
PLASMA_DERIVED_AIF = os.environ.get("P_BRAIN_PLASMA_AIF", "0").lower() in {"1", "true", "yes"}

# Optional voxelwise delay alignment prior to deconvolution
ALIGN_AIF_BY_XCORR = os.environ.get("P_BRAIN_ALIGN_AIF", "0").lower() in {"1", "true", "yes"}
ALIGN_AIF_MAX_SHIFT_S = float(os.environ.get("P_BRAIN_ALIGN_AIF_MAX_SHIFT", 4.0))

# MATLAB-style voxelwise offset correction (arrival-time estimation) prior to
# deconvolution. When enabled, each voxel's tissue curve is shifted in time
# using MATLAB-equivalent `curve2InitEnhanc_2` + `inputshift3` (pchip).
MATLAB_OFFSET_CORRECTION = os.environ.get("P_BRAIN_MATLAB_OFFSET_CORRECTION", "0").lower() in {
    "1",
    "true",
    "yes",
}
# Write the estimated offset (seconds) as a NIfTI map in voxelwise-only mode.
WRITE_OFFSET_MAP = os.environ.get("P_BRAIN_WRITE_OFFSET_MAP", "1").lower() in {"1", "true", "yes"}

# Arterial input function configuration.
#
# There are two concepts:
# 1) Which *arterial* ROI to use as the reference artery (RICA/LICA).
# 2) Whether the *output* AIF used for modelling should be the time-shifted
#    and rescaled SSS-derived curve (TSCC) or a pure arterial curve.
#
# Defaults requested:
# - reference artery: RICA
# - use SSS time shifting: enabled
#
# Backwards compatibility:
# - P_BRAIN_INPUT_FUNCTION=SSS forces SSS/TSCC output (reference artery defaults to RICA)
# - P_BRAIN_INPUT_FUNCTION=RICA or LICA forces pure arterial output

_AIF_ARTERY = (os.environ.get("P_BRAIN_AIF_ARTERY", "RICA") or "RICA").strip().upper()
if _AIF_ARTERY not in {"RICA", "LICA"}:
    _AIF_ARTERY = "RICA"

_AIF_USE_SSS = (os.environ.get("P_BRAIN_AIF_USE_SSS", "1") or "1").strip().lower() in {"1", "true", "yes"}

_legacy_if = (os.environ.get("P_BRAIN_INPUT_FUNCTION", "") or "").strip().upper()
if _legacy_if:
    if _legacy_if == "SSS":
        _AIF_USE_SSS = True
    elif _legacy_if in {"RICA", "LICA"}:
        _AIF_USE_SSS = False
        _AIF_ARTERY = _legacy_if

# Optional explicit modelling input-function selector.
# - tscc (default): time-shifted+rescaled SSS-derived curve
# - aif: pure arterial curve
# - vif: pure venous curve (SSS)
_model_if_raw = (os.environ.get("P_BRAIN_MODELLING_INPUT_FUNCTION", "") or "").strip().lower()
if _model_if_raw in {"tscc", "aif", "vif"}:
    MODELLING_INPUT_FUNCTION = _model_if_raw
else:
    # Backwards compatible default based on legacy toggle.
    MODELLING_INPUT_FUNCTION = "tscc" if _AIF_USE_SSS else "aif"

# Align legacy toggle with the modelling selector.
if MODELLING_INPUT_FUNCTION == "tscc":
    _AIF_USE_SSS = True
elif MODELLING_INPUT_FUNCTION in {"aif", "vif"}:
    _AIF_USE_SSS = False

# Exported settings.
INPUT_FUNCTION_ARTERY = _AIF_ARTERY
INPUT_FUNCTION_USE_SSS = bool(_AIF_USE_SSS)

# Legacy-style source string (used by some older callsites).
INPUT_FUNCTION_SOURCE = "SSS" if INPUT_FUNCTION_USE_SSS else INPUT_FUNCTION_ARTERY


# TSCC (time-shifted and rescaled concentration curve) amplitude handling.
# - P_BRAIN_TSCC_RESCALE=1 enables amplitude rescaling (default).
# - P_BRAIN_TSCC_RESCALE=0 disables amplitude rescaling (time-shift only).
# - P_BRAIN_TSCC_RESCALE_METHOD selects how to compute the scale factor:
#     - peak: match peak heights (legacy behaviour: only upscale when vein peak < artery peak).
#     - auc:  match areas under the curve ("total volume"; legacy behaviour: only upscale when vein AUC < artery AUC).
TSCC_RESCALE = (os.environ.get("P_BRAIN_TSCC_RESCALE", "1") or "1").strip().lower() in {"1", "true", "yes"}
TSCC_RESCALE_METHOD = (os.environ.get("P_BRAIN_TSCC_RESCALE_METHOD", "peak") or "peak").strip().lower()
if TSCC_RESCALE_METHOD not in {"peak", "auc"}:
    TSCC_RESCALE_METHOD = "peak"

# TSCC alignment strategy.
# - peaks (default): peak/cross-correlation heuristic (fast, integer-frame shift).
# - pchip_fit: continuous-time pchip shifting + shift/scale fitting.
_tscc_method_raw = (os.environ.get("P_BRAIN_TSCC_METHOD", "") or "").strip().lower()
# Alignment implementation selector.
# - peaks: fast discrete peak-based alignment (legacy)
# - pchip_fit: continuous-time pchip shifting + shift/scale fitting
if not _tscc_method_raw:
    TSCC_METHOD = "peaks"
elif _tscc_method_raw in {"peaks", "pchip_fit"}:
    TSCC_METHOD = _tscc_method_raw
else:
    # Back-compat: older configs may contain previous labels.
    # Treat any non-empty unknown as the pchip+fit implementation.
    TSCC_METHOD = "pchip_fit"

# Optional demo plot showing raw SSS vs time-shifted and rescaled TSCC overlayed
# with available arterial curves.
TSCC_WRITE_DEMO_PLOT = (os.environ.get("P_BRAIN_TSCC_WRITE_DEMO_PLOT", "1") or "1").strip().lower() in {"1", "true", "yes"}

# How to derive a representative vascular curve from an ROI mask.
# - max (default): pick the single brightest voxel within the ROI (over time).
# - mean: average the signal across all ROI voxels.
VASCULAR_ROI_CURVE_METHOD = (
    os.environ.get("P_BRAIN_VASCULAR_ROI_CURVE_METHOD", "max") or "max"
).strip().lower()
if VASCULAR_ROI_CURVE_METHOD not in {"max", "mean", "median"}:
    VASCULAR_ROI_CURVE_METHOD = "max"

# Optional adaptive-max tracking for vascular ROI curves.
# When enabled (and VASCULAR_ROI_CURVE_METHOD=max), the brightest voxel is
# re-selected independently at each time frame within the ROI.
VASCULAR_ROI_ADAPTIVE_MAX = (
    os.environ.get("P_BRAIN_VASCULAR_ROI_ADAPTIVE_MAX", "1") or "1"
).strip().lower() in {"1", "true", "yes"}

# How to aggregate tissue/atlas ROI voxels into representative curves/values.
# - median (default): robust to outliers
# - mean: historical behaviour in some tissue paths
TISSUE_ROI_AGGREGATION = (os.environ.get("P_BRAIN_TISSUE_ROI_AGGREGATION", "median") or "median").strip().lower()
if TISSUE_ROI_AGGREGATION not in {"mean", "median"}:
    TISSUE_ROI_AGGREGATION = "median"

# Tissue ROI method — how tissue regions are defined for CTC extraction.
# - automatic (default): uses segmentation (FastSurfer/SynthSeg) to create
#   tissue masks automatically.
# - manual: user draws ROIs interactively on DCE/T1 slices; skips
#   brain segmentation entirely.
TISSUE_ROI_METHOD = (os.environ.get("P_BRAIN_TISSUE_ROI_METHOD", "automatic") or "automatic").strip().lower()
if TISSUE_ROI_METHOD not in {"automatic", "manual"}:
    TISSUE_ROI_METHOD = "automatic"

# Background volume for manual ROI drawing.
# - auto (default): picks the best available structural (T1 > T2 > FLAIR),
#   falling back to DCE if none exist.
# - dce / t1 / t2 / flair: force a specific volume type.
MANUAL_ROI_OVERLAY = (
    os.environ.get("P_BRAIN_MANUAL_ROI_OVERLAY", "auto") or "auto"
).strip().lower()
if MANUAL_ROI_OVERLAY not in {"auto", "dce", "t1", "t2", "flair"}:
    MANUAL_ROI_OVERLAY = "auto"

# Flip angle (degrees) used by signal->concentration conversion.
# Set `P_BRAIN_FLIP_ANGLE=auto` (default) to rely on metadata.
# Set `P_BRAIN_FLIP_ANGLE=<number>` to override (e.g. 30).
FLIP_ANGLE_SETTING = (os.environ.get("P_BRAIN_FLIP_ANGLE", "auto") or "auto").strip()
_flip_angle_lower = FLIP_ANGLE_SETTING.lower()
FLIP_ANGLE_DEG = None
if _flip_angle_lower and _flip_angle_lower != "auto":
    try:
        FLIP_ANGLE_DEG = float(FLIP_ANGLE_SETTING)
    except ValueError:
        FLIP_ANGLE_SETTING = "auto"
        FLIP_ANGLE_DEG = None

# When enabled, missing critical acquisition metadata becomes an error rather
# than silently falling back to heuristics/defaults.
#
# Examples of metadata that may be required when STRICT_METADATA is enabled:
# - FlipAngle for signal->concentration conversion
# - RepetitionTimeExcitation / InversionTime for TurboFLASH models
STRICT_METADATA = (os.environ.get("P_BRAIN_STRICT_METADATA", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Signal-to-concentration conversion model.
# Validator parity requirement: there is exactly ONE supported conversion path.
#
# TurboFLASH (Hemisure validator / MATLAB `menu_15_5.m`, method 4):
#   c = (-1/(beta*TI_dyn))*( log(1 - s/(M0*sin(alpha))) + TI_dyn*R1_pre )
#   baseline: c = c - mean(c(1:baseline)); c(1:baseline)=0; nan->0
CTC_MODEL = (os.environ.get("P_BRAIN_CTC_MODEL", "turboflash") or "turboflash").strip().lower()
if CTC_MODEL in {"advanced", "validated", "method4", "validated_method4"}:
    CTC_MODEL = "turboflash"
if CTC_MODEL != "turboflash":
    CTC_MODEL = "turboflash"

# TurboFLASH baseline length (frames) for the validated conversion.
try:
    TURBOFLASH_BASELINE_FRAMES = int(float(os.environ.get("P_BRAIN_TURBOFLASH_BASELINE_FRAMES", "10")))
except Exception:
    TURBOFLASH_BASELINE_FRAMES = 10
TURBOFLASH_BASELINE_FRAMES = max(1, int(TURBOFLASH_BASELINE_FRAMES))

# Optional override for TurboFLASH excitation TR (seconds).
#
# IMPORTANT: DCE JSON sidecars often store `RepetitionTime` as the *volume*
# sampling interval (seconds per dynamic), not the excitation TR (ms).
# For TurboFLASH we need the excitation TR. Prefer `RepetitionTimeExcitation`
# in the JSON sidecar, or set this override explicitly.
TURBOFLASH_TR_S = None
_tf_tr_s = (os.environ.get("P_BRAIN_TURBOFLASH_TR_S") or "").strip()
_tf_tr_ms = (os.environ.get("P_BRAIN_TURBOFLASH_TR_MS") or "").strip()
try:
    if _tf_tr_s:
        v = float(_tf_tr_s)
        if v > 0:
            TURBOFLASH_TR_S = v
    elif _tf_tr_ms:
        v = float(_tf_tr_ms)
        if v > 0:
            TURBOFLASH_TR_S = v * 1e-3
except Exception:
    TURBOFLASH_TR_S = None

# Optional override for TurboFLASH nph (turbo factor / ky=0 line index).
TURBOFLASH_NPH = None
_tf_nph = (os.environ.get("P_BRAIN_TURBOFLASH_NPH") or os.environ.get("P_BRAIN_TURBO_NPH") or "").strip()
if _tf_nph:
    try:
        v = int(float(_tf_nph))
        if v > 0:
            TURBOFLASH_NPH = v
    except Exception:
        TURBOFLASH_NPH = None

# TurboFLASH CTC conversion method.
#
# p-brain uses a single validated closed-form TurboFLASH conversion (MATLAB menu_5 case12 method1).
# Exposed as:
#   TURBOFLASH_CTC_METHOD = "turboflash_advanced"
TURBOFLASH_CTC_METHOD = (
    os.environ.get("P_BRAIN_TURBOFLASH_CTC_METHOD", "turboflash_advanced") or "turboflash_advanced"
).strip().lower()

_TF_ADV_ALIASES = {
    "turboflash_advanced",
    "case12",
    "case12_method1",
    "matlab_case12",
    "menu5_case12",
    "closed_form",
}
if TURBOFLASH_CTC_METHOD not in _TF_ADV_ALIASES:
    TURBOFLASH_CTC_METHOD = "turboflash_advanced"

# TurboFLASH TI during bolus passage (seconds) for the closed-form conversion.
# MATLAB default is 0.12 s.
TURBOFLASH_TI_DYN_S = None
_tf_ti_s = (os.environ.get("P_BRAIN_TURBOFLASH_TI_DYN_S") or os.environ.get("P_BRAIN_TURBOFLASH_TI_S") or "").strip()
_tf_ti_ms = (os.environ.get("P_BRAIN_TURBOFLASH_TI_DYN_MS") or os.environ.get("P_BRAIN_TURBOFLASH_TI_MS") or "").strip()
try:
    if _tf_ti_s:
        v = float(_tf_ti_s)
        if v > 0:
            TURBOFLASH_TI_DYN_S = v
    elif _tf_ti_ms:
        v = float(_tf_ti_ms)
        if v > 0:
            TURBOFLASH_TI_DYN_S = v * 1e-3
except Exception:
    TURBOFLASH_TI_DYN_S = None

# When enabled, use the strict saturation inversion *without* the
# additional stability clamps / M0 re-estimation.
CTC_SATURATION_STRICT = (os.environ.get("P_BRAIN_CTC_SATURATION_STRICT", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

# DCE temporal sampling (seconds per volume).
#
# Defaults to "auto" (None) and will be resolved from metadata (NIfTI header
# pixdim[4]/zooms[3], then JSON sidecar) when needed.
#
# Optional overrides:
# - P_BRAIN_DCE_TIME_STEP_S=<float>
# - P_BRAIN_DCE_DT_S=<float>
_dce_dt_raw = None
for _key in (
    "P_BRAIN_DCE_TIME_STEP_S",
    "P_BRAIN_DCE_DT_S",
    "P_BRAIN_DCE_TEMPORAL_RESOLUTION_S",
    "P_BRAIN_DCE_TEMPORAL_RESOLUTION",
):
    _val = os.environ.get(_key)
    if _val is not None and str(_val).strip() != "":
        _dce_dt_raw = str(_val).strip()
        break

DCE_TIME_STEP_S = None
if _dce_dt_raw is not None and _dce_dt_raw.lower() != "auto":
    try:
        _dt = float(_dce_dt_raw)
        if _dt > 0:
            DCE_TIME_STEP_S = _dt
    except (TypeError, ValueError):
        DCE_TIME_STEP_S = None

# TurboFLASH model parameters (only used when CTC_MODEL=turboflash).
# NOTE: Do not clobber TURBOFLASH_NPH here; it is resolved above from
# overrides/metadata with better fallbacks.

# T1/M0 fitting input source.
# - auto: prefer inversion-recovery if present, otherwise try VFA.
# - ir: require inversion-recovery series.
# - vfa: require VFA spoiled GRE series.
# - none: skip T1/M0 fitting (outputs NaNs).
T1_FIT_MODE = (os.environ.get("P_BRAIN_T1_FIT", "auto") or "auto").strip().lower()
if T1_FIT_MODE not in {"auto", "ir", "vfa", "none"}:
    T1_FIT_MODE = "auto"

# VFA discovery glob(s) used when T1_FIT_MODE is vfa/auto.
# Comma-separated patterns relative to the NIfTI directory.
VFA_FILE_GLOB = (os.environ.get("P_BRAIN_VFA_GLOB", "*VFA*.nii*") or "*VFA*.nii*").strip()

# When enabled, pick the regularisation weight automatically using an L-curve
# search across ``AUTO_LAMBDA_CANDIDATES``.
AUTO_LAMBDA = (os.environ.get("P_BRAIN_AUTO_LAMBDA", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_auto_lambda_candidates_raw = (os.environ.get("P_BRAIN_AUTO_LAMBDA_CANDIDATES") or "").strip()
_auto_lambda_space = (os.environ.get("P_BRAIN_AUTO_LAMBDA_SPACE", "linear") or "linear").strip().lower()
try:
    _auto_lambda_min = float(os.environ.get("P_BRAIN_AUTO_LAMBDA_MIN", 0.05))
except Exception:
    _auto_lambda_min = 0.05
try:
    _auto_lambda_max = float(os.environ.get("P_BRAIN_AUTO_LAMBDA_MAX", 40.0))
except Exception:
    _auto_lambda_max = 40.0
try:
    _auto_lambda_steps = int(os.environ.get("P_BRAIN_AUTO_LAMBDA_STEPS", 120))
except Exception:
    _auto_lambda_steps = 120

if _auto_lambda_candidates_raw:
    try:
        AUTO_LAMBDA_CANDIDATES = np.array(
            [float(x) for x in re.split(r"[ ,;]+", _auto_lambda_candidates_raw) if x.strip() != ""],
            dtype=float,
        )
    except Exception:
        AUTO_LAMBDA_CANDIDATES = np.array([], dtype=float)
else:
    # Legacy default: linear sweep (uniform spacing) across [0.05, 40] with 120 intervals.
    # This yields 121 candidates, matching MATLAB's `L_min:(L_max-L_min)/L_no:L_max`.
    if _auto_lambda_steps < 1:
        _auto_lambda_steps = 120
    if not np.isfinite(_auto_lambda_min) or _auto_lambda_min <= 0:
        _auto_lambda_min = 0.05
    if not np.isfinite(_auto_lambda_max) or _auto_lambda_max <= _auto_lambda_min:
        _auto_lambda_max = 40.0

    if _auto_lambda_space == "log":
        AUTO_LAMBDA_CANDIDATES = np.logspace(
            np.log10(_auto_lambda_min),
            np.log10(_auto_lambda_max),
            int(_auto_lambda_steps) + 1,
        )
    else:
        AUTO_LAMBDA_CANDIDATES = np.linspace(
            _auto_lambda_min,
            _auto_lambda_max,
            int(_auto_lambda_steps) + 1,
        )

if AUTO_LAMBDA_CANDIDATES.size < 3:
    # Safety fallback for curvature estimation.
    AUTO_LAMBDA_CANDIDATES = np.linspace(0.05, 40.0, 121)
# Holds the most recently chosen value when ``AUTO_LAMBDA`` is True
AUTO_LAMBDA_VALUE = None

# Number of leading DCE volumes (time frames) to discard before analysis.
# Useful when the first frame(s) are blurry or contain artefacts.
# Default: 0 (keep all).  Override via ``P_BRAIN_DCE_SKIP_VOLUMES``.
try:
    DCE_SKIP_VOLUMES = max(0, int(os.environ.get("P_BRAIN_DCE_SKIP_VOLUMES", 0)))
except (TypeError, ValueError):
    DCE_SKIP_VOLUMES = 0

# Number of bolus peaks expected in the acquisition.  Defaults to two but can
# be overridden via the ``P_BRAIN_NUMBER_OF_PEAKS`` environment variable.
# Use ``--single-bolus`` CLI flag (sets this to 1) for single-injection protocols.
NUMBER_OF_PEAKS = int(os.environ.get("P_BRAIN_NUMBER_OF_PEAKS", 2))

# Lower bound for the Patlak inclusion window expressed as a fraction of the
# maximum Patlak x-coordinate.  Higher values restrict the fit to the tail.
PATLAK_WINDOW_START_FRACTION = float(
    os.environ.get("P_BRAIN_PATLAK_WINDOW_START_FRACTION", 1 / 3)
)

# Minimum coefficient of determination required when choosing the Patlak
# fitting segment.  The solver progressively discards early points until this
# target is reached (or the best achievable R² is used).
PATLAK_MIN_R2 = float(os.environ.get("P_BRAIN_PATLAK_MIN_R2", 0.985))

# Number of slices to omit from the inferior (bottom) and superior (top)
# ends when computing global Ki for white matter, cortical grey matter
# and boundary tissues.  These can be overridden via the environment
# variables ``P_BRAIN_GLOBAL_KI_SKIP_BOTTOM`` and
# ``P_BRAIN_GLOBAL_KI_SKIP_TOP``.
GLOBAL_KI_SKIP_BOTTOM = int(os.environ.get("P_BRAIN_GLOBAL_KI_SKIP_BOTTOM", 2))
GLOBAL_KI_SKIP_TOP = int(os.environ.get("P_BRAIN_GLOBAL_KI_SKIP_TOP", 2))

# Paths to the neural network models used for artery and vein ROI extraction.
# These can be overridden by environment variables to use custom models.
AI_MODEL_PATHS = {
    'slice_classifier_rica': os.environ.get(
        'SLICE_CLASSIFIER_RICA_MODEL', 'AI/slice_classifier_model_rica.keras'
    ),
    'rica_roi': os.environ.get(
        'RICA_ROI_MODEL', 'AI/rica_roi_model.keras'
    ),
    'slice_classifier_ss': os.environ.get(
        'SLICE_CLASSIFIER_SS_MODEL', 'AI/ss_slice_classifier.keras'
    ),
    'ss_roi': os.environ.get(
        'SS_ROI_MODEL', 'AI/ss_roi_model.keras'
    ),
}

# ROI extraction method.
# (Defined and validated earlier; supports ROI_METHOD=file for user-supplied ROIs.)

# Deterministic ROI parameters (used when ROI_METHOD=deterministic).
ROI_RICA_RADIUS = int(os.environ.get("P_BRAIN_ROI_RICA_RADIUS", 10))
ROI_LICA_RADIUS = int(os.environ.get("P_BRAIN_ROI_LICA_RADIUS", 10))
ROI_SSS_RADIUS = int(os.environ.get("P_BRAIN_ROI_SSS_RADIUS", 12))

ROI_RICA_SLICES = int(os.environ.get("P_BRAIN_ROI_RICA_SLICES", 3))
ROI_LICA_SLICES = int(os.environ.get("P_BRAIN_ROI_LICA_SLICES", 3))
ROI_SSS_SLICES = int(os.environ.get("P_BRAIN_ROI_SSS_SLICES", 4))

# Optional explicit z-ranges (inclusive) as "start:end" (e.g. "0:2").
ROI_RICA_Z_RANGE = (os.environ.get("P_BRAIN_ROI_RICA_Z_RANGE", "") or "").strip()
ROI_LICA_Z_RANGE = (os.environ.get("P_BRAIN_ROI_LICA_Z_RANGE", "") or "").strip()
ROI_SSS_Z_RANGE = (os.environ.get("P_BRAIN_ROI_SSS_Z_RANGE", "") or "").strip()

# Midline band (pixels) used to constrain SSS search to the sagittal midline.
ROI_SSS_MIDLINE_BAND = int(os.environ.get("P_BRAIN_ROI_SSS_MIDLINE_BAND", 24))

# Baseline frames used to compute peak contrast on DCE.
ROI_DCE_BASELINE_FRAMES = int(os.environ.get("P_BRAIN_ROI_DCE_BASELINE_FRAMES", 5))

# If enabled (default), normalize voxelwise curves for PCA/diagnostic plots by
# subtracting a baseline mean, clamping the baseline window to zero, and
# scaling by a robust percentile of the deviation. Disable for raw-curve PCA.
ROI_NORMALIZE_CURVES = (os.environ.get("P_BRAIN_ROI_NORMALIZE_CURVES", "1") or "1").strip().lower() in {"1", "true", "yes"}


def _default_data_root():
    """Return the default root directory for datasets.

    Prefers the ``P_BRAIN_DATA_DIR`` environment variable and falls back to a
    ``Data`` folder next to the executable.
    """

    env_root = os.environ.get("P_BRAIN_DATA_DIR")
    if env_root:
        return os.path.abspath(env_root)
    return os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "Data")

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_directories(log_number, data_root=None):
    if data_root is None:
        data_root = _default_data_root()
    else:
        data_root = os.path.abspath(data_root)

    data_directory = os.path.join(data_root, log_number)

    # Look for control data if enabled
    if CONTROLS:
        control_directory = os.path.join(data_root, 'controls', log_number)
        if os.path.isdir(control_directory):
            data_directory = control_directory
            flag_path = os.path.join(control_directory, CONTROL_FLAG_FILENAME)
            if not os.path.exists(flag_path):
                with open(flag_path, 'w') as f:
                    json.dump({"control": True, "id": log_number}, f, indent=4)
    analysis_directory = os.path.join(data_directory, 'Analysis')
    nifti_directory = os.path.join(data_directory, 'NIfTI')
    image_directory = os.path.join(data_directory, 'Images')
    
    # Directories to create
    dirs_to_create = [
        analysis_directory,
        os.path.join(analysis_directory, 'TSCC Data'),
        os.path.join(analysis_directory, 'CTC Data'),
        os.path.join(analysis_directory, 'CTC Data', 'Artery'),
        os.path.join(analysis_directory, 'CTC Data', 'Vein'),
        os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'Grey Matter'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'White Matter'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'Boundary'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'AI'),
        os.path.join(analysis_directory, 'ITC Data'),
        os.path.join(analysis_directory, 'TSCC Data', 'Max'),
        os.path.join(analysis_directory, 'ITC Data', 'Artery'),
        os.path.join(analysis_directory, 'ITC Data', 'Vein'),
        os.path.join(analysis_directory, 'ITC Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'ROI Data'),
        os.path.join(analysis_directory, 'ROI Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'Frame Data'),
        os.path.join(analysis_directory, 'Frame Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'Fitting'),
        image_directory,
        os.path.join(image_directory, 'Intensity Time Curves'),
        os.path.join(image_directory, 'AI'),
        os.path.join(image_directory, 'AI', 'Input functions'),
        os.path.join(image_directory, 'AI', 'Tissue functions'),
        os.path.join(image_directory, 'AI', 'Segmentation'),
        os.path.join(image_directory, 'Fit'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Artery'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Vein'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Vein', 'Sinus Sagittalis'),
        os.path.join(image_directory, 'Concentration Time Curves'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Artery'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Vein'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Vein', 'Sinus Sagittalis'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'White Matter'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'Grey Matter'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'Boundary'),
        os.path.join(image_directory, 'Time Shifted Concentration Curves'),
        os.path.join(image_directory, 'Time Shifted Concentration Curves', 'Max'),
        nifti_directory
    ]
    
    artery_names = ["Left Interior Carotid", "Right Interior Carotid", "Basilar", "Left Middle Cerebral", "Right Middle Cerebral"]
    
    # Append directories involving artery names
    for artery in artery_names:
        dirs_to_create.extend([
            os.path.join(analysis_directory, 'CTC Data', 'Artery', artery),
            os.path.join(analysis_directory, 'ITC Data', 'Artery', artery),
            os.path.join(analysis_directory, 'TSCC Data', artery),
            os.path.join(analysis_directory, 'ROI Data', 'Artery', artery),
            os.path.join(analysis_directory, 'Frame Data', 'Artery', artery),
            os.path.join(image_directory, 'Concentration Time Curves', 'Artery', artery),
            os.path.join(image_directory, 'Intensity Time Curves', 'Artery', artery),
            os.path.join(image_directory, 'Time Shifted Concentration Curves', artery)
        ])
    
    # Create all directories
    for dir_path in dirs_to_create:
        create_directory(dir_path)

    return data_directory, analysis_directory, nifti_directory, image_directory


def save_run_settings(analysis_directory, parameters):
    """Save the settings used for an analysis run.

    Parameters are provided as the tuple returned by ``global_parameters``.
    The summary is written to ``run_settings.json`` in ``analysis_directory``.
    """
    names = [
        "IsVFA",
        "IsIR",
        "apple_metal",
        "boundary",
        "RERUN_SEGMENTATION",
        "SEGMENTATION_METHOD",
        "COMPUTE_FA",
    ]
    settings = dict(zip(names, parameters))
    settings.update({
        "MULTIPROCESSING": MULTIPROCESSING,
        "NUMBER_OF_CORES": NUMBER_OF_CORES,
        "KINETIC_MODEL": KINETIC_MODEL,
        "AI_MODEL_PATHS": AI_MODEL_PATHS,
        "CONTROLS": CONTROLS,
        "TIKHONOV_LAMBDA": TIKHONOV_LAMBDA,
        "GLOBAL_KI_SKIP_BOTTOM": GLOBAL_KI_SKIP_BOTTOM,
        "GLOBAL_KI_SKIP_TOP": GLOBAL_KI_SKIP_TOP,
        "TWO_COMPARTMENT_INIT_FROM_PATLAK": TWO_COMPARTMENT_INIT_FROM_PATLAK,
        "WRITE_MTT": WRITE_MTT,
        "WRITE_CTH": WRITE_CTH,
        "CTH_MTT_METHOD": CTH_MTT_METHOD,
        "CTH_MTT_GAMMA_VOXELWISE": CTH_MTT_GAMMA_VOXELWISE,
        "ALIGN_AIF_BY_XCORR": ALIGN_AIF_BY_XCORR,
        "ALIGN_AIF_MAX_SHIFT_S": ALIGN_AIF_MAX_SHIFT_S,
    })
    if AUTO_LAMBDA:
        settings["AUTO_LAMBDA_VALUE"] = AUTO_LAMBDA_VALUE
    settings_path = os.path.join(analysis_directory, "run_settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)


def _git_metadata():
    """Collect basic Git metadata if the repository is available."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

    metadata = {"commit": commit}

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        metadata["is_dirty"] = bool(status.strip())
    except Exception:
        pass

    return metadata


def save_runtime_metadata(analysis_directory, *, extra_environment_keys=None):
    """Record runtime metadata for the executed analysis pipeline.

    The metadata is written to ``runtime_metadata.json`` within the analysis
    directory.  The recorded information focuses on reproducibility and avoids
    dumping the entire environment to reduce the risk of leaking secrets.
    """

    env_keys = [
        "P_BRAIN_DATA_DIR",
        "P_BRAIN_MODEL",
        "P_BRAIN_CORES",
        "P_BRAIN_LAMBDA",
        "P_BRAIN_GLOBAL_KI_SKIP_BOTTOM",
        "P_BRAIN_GLOBAL_KI_SKIP_TOP",
        "PBRAIN_CONTROLS",
        "PBRAIN_TURBO",
        "P_BRAIN_CTH_MTT_METHOD",
        "P_BRAIN_CTH_MTT_GAMMA_VOXELWISE",
        "P_BRAIN_ALIGN_AIF",
        "P_BRAIN_ALIGN_AIF_MAX_SHIFT",
    ]
    if extra_environment_keys:
        env_keys.extend(k for k in extra_environment_keys if k not in env_keys)

    environment = {
        key: os.environ.get(key)
        for key in env_keys
        if os.environ.get(key) is not None
    }

    packages = {}
    for package in ("numpy", "scipy", "matplotlib", "nibabel", "torch"):
        try:
            packages[package] = importlib_metadata.version(package)
        except Exception:
            continue

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "argv": sys.argv,
    }

    if environment:
        metadata["environment"] = environment
    if packages:
        metadata["packages"] = packages

    git_info = _git_metadata()
    if git_info:
        metadata["git"] = git_info

    os.makedirs(analysis_directory, exist_ok=True)
    metadata_path = os.path.join(analysis_directory, "runtime_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_path
