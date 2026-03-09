turbo_mode = False  # Set to True to suppress all plots
force_recreate_masks = False  # If True: recreate all masks regardless of existence
# If True, drop detection is performed on tissue CTCs and the ignored
# regions are excluded from the Patlak fit.  When False, every sample
# is used for the Patlak analysis.
correct_signal_jumps = False


# When True, a two-step FLIRT registration is used when aligning
# segmentation masks to DCE and T2 images.  When False (default), the
# previous one-step "-applyxfm -usesqform" approach is retained.
use_flirt_registration = False

import json
import functools
import logging
import multiprocessing
import os
import pickle
import shutil
import subprocess
import warnings
from typing import Any
from types import SimpleNamespace

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

from utils import settings
from utils.cli_logging import auto_logging_suppressed
from utils.loading import (
    get_input_function_curve,
    load_dce_4d,
    resolve_flip_angle_deg,
    resolve_turboflash_ti_s,
)
from utils.plotting import (
    butter_lowpass_filter,
    close_plot_after_delay,
    compute_CTC,
    custom_shifter,
    find_major_peaks,
)

from modules.kinetic_models import (
    build_spline_lcurve_deconvolution_solver,
    construct_convolution_matrix as km_construct_convolution_matrix,
    tikhonov_regularization as km_tikhonov_regularization,
    solve_single_voxel_diagnostic,
)

# Patlak model implementation (new modular location). Some code paths still
# expect `model_patlak_with_exclusions` to exist at module scope.
from models.patlak import fit_patlak_tuple as model_patlak_with_exclusions
from models.patlak import fit_patlak as _fit_patlak_diagnostic

# Extended Tofts model (modular implementation).
from models.extended_tofts import fit_voxel as _etofts_fit_voxel


def load_from_pickle(file_path: str) -> Any:
    """Load and return a Python object from a pickle file.

    Compatibility shim: some runners import this module as `AIT` and expect
    `AIT.load_from_pickle(...)` to exist.
    """

    path = os.fspath(file_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as file:
        return pickle.load(file)


def _tqdm(*args, **kwargs):
    kwargs.setdefault("disable", False)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    return tqdm(*args, **kwargs)


def _resolve_lambda_candidates(
    *,
    lambda_candidates,
    auto_lambda: bool,
    auto_lambda_value,
    lambd_default,
):
    """Resolve lambda candidates for Tikhonov/L-curve selection.

    Notes:
    - `build_spline_lcurve_deconvolution_solver` requires *uniformly spaced* lambda
      candidates when performing curvature-based selection.
    - If a non-uniform grid is provided, we fall back to a single `lambd_default`.
    """

    if lambda_candidates is not None:
        cand = np.asarray(lambda_candidates, dtype=float).reshape(-1)
    elif auto_lambda and auto_lambda_value is not None and np.isfinite(auto_lambda_value):
        cand = np.asarray([float(auto_lambda_value)], dtype=float)
    elif auto_lambda:
        cand = np.asarray(getattr(settings, "AUTO_LAMBDA_CANDIDATES", np.array([], dtype=float)), dtype=float).reshape(-1)
    else:
        cand = np.asarray([float(lambd_default)], dtype=float)

    cand = cand[np.isfinite(cand) & (cand > 0)]
    if cand.size == 0:
        return np.asarray([float(lambd_default)], dtype=float)
    if cand.size < 3:
        return cand

    diffs = np.diff(cand)
    if diffs.size and not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-10):
        return np.asarray([float(lambd_default)], dtype=float)
    return cand


def build_tikhonov_validated_slow_solver(
    time_s,
    ca,
    lambda_candidates=None,
    *,
    auto_lambda=settings.AUTO_LAMBDA,
    auto_lambda_value=settings.AUTO_LAMBDA_VALUE,
    lambd_default=settings.TIKHONOV_LAMBDA,
    tissue_density=None,
    hematocrit=None,
    plasma_derived_aif=None,
    **_ignored,
):
    """Return a validated (L-curve) spline+Tikhonov solver with a legacy API.

    Returns a callable `solver(Ct_mat, offsets_s=None)` that produces an object
    with `.cbf_ml_per_100g_min`, `.mtt_s`, `.cth_s`, `.lambda_opt`, `.cbv_vd`.

    This matches the attribute-based access patterns in older code paths.
    """

    lam = _resolve_lambda_candidates(
        lambda_candidates=lambda_candidates,
        auto_lambda=bool(auto_lambda),
        auto_lambda_value=auto_lambda_value,
        lambd_default=lambd_default,
    )
    core = build_spline_lcurve_deconvolution_solver(np.asarray(time_s, dtype=float), np.asarray(ca, dtype=float), lam)

    def _solve(Ct_mat, offsets_s=None):
        # offsets_s is accepted for API compatibility; validated spline solver
        # does not currently apply per-curve sub-frame shifting in this wrapper.
        _ = offsets_s
        sol = core(Ct_mat)
        cbf = np.asarray(sol.get("cbf_ml_per_100g_min"), dtype=float)
        mtt = np.asarray(sol.get("mtt_s"), dtype=float)
        cth = np.asarray(sol.get("cth_s"), dtype=float)
        lam_opt = np.asarray(sol.get("lambda_opt"), dtype=float)
        cbv_vd = cbf * mtt / 60.0

        return SimpleNamespace(
            cbf_ml_per_100g_min=cbf,
            mtt_s=mtt,
            cth_s=cth,
            lambda_opt=lam_opt,
            cbv_vd=cbv_vd,
            residue=sol.get("residue"),
            f_internal=sol.get("f_internal"),
        )

    return _solve


def model_tikhonov_validated_tuple(
    C_a,
    C_t,
    t,
    *,
    offsets_s=0.0,
    return_residue: bool = False,
    auto_lambda=settings.AUTO_LAMBDA,
    auto_lambda_value=settings.AUTO_LAMBDA_VALUE,
    lambd_default=settings.TIKHONOV_LAMBDA,
):
    """Legacy tuple API used by plotting/QA paths.

    Returns:
    - when `return_residue=False`: `(Ki, vp, SD_Ki, fit_curve)`
    - when `return_residue=True`: `(Ki, vp, SD_Ki, fit_curve, impulse, cbf)`

    The validated deconvolution path primarily produces perfusion metrics.
    Ki/vp are not estimated here and are returned as NaN.
    """

    t_use = np.asarray(t, dtype=float).reshape(-1)
    ca_use = np.asarray(C_a, dtype=float).reshape(-1)
    ct_use = np.asarray(C_t, dtype=float).reshape(-1)
    n = int(min(t_use.size, ca_use.size, ct_use.size))
    if n < 2:
        if return_residue:
            return (float("nan"), float("nan"), float("nan"), None, None, float("nan"))
        return (float("nan"), float("nan"), float("nan"), None)

    t_use = t_use[:n]
    ca_use = ca_use[:n]
    ct_use = ct_use[:n]

    solver = build_tikhonov_validated_slow_solver(
        t_use,
        ca_use,
        lambda_candidates=None,
        auto_lambda=bool(auto_lambda),
        auto_lambda_value=auto_lambda_value,
        lambd_default=lambd_default,
        tissue_density=float(getattr(settings, "TISSUE_DENSITY", 1.04)),
        hematocrit=float(getattr(settings, "HEMATOCRIT", 0.42)),
        plasma_derived_aif=bool(getattr(settings, "PLASMA_DERIVED_AIF", False)),
    )
    sol = solver(ct_use.reshape(-1, 1), offsets_s=offsets_s)

    cbf = float(np.asarray(sol.cbf_ml_per_100g_min, dtype=float).reshape(-1)[0])

    impulse = None
    try:
        residue = sol.residue
        f_internal = sol.f_internal
        if residue is not None and f_internal is not None:
            residue0 = np.asarray(residue, dtype=float)
            fin0 = float(np.asarray(f_internal, dtype=float).reshape(-1)[0])
            if residue0.ndim == 2:
                residue0 = residue0[:, 0]
            if residue0.size and np.isfinite(fin0):
                impulse = residue0 * fin0
    except Exception:
        impulse = None

    Ki = float("nan")
    vp = float("nan")
    SD_Ki = float("nan")
    fit_curve = None

    if return_residue:
        return (Ki, vp, SD_Ki, fit_curve, impulse, cbf)
    return (Ki, vp, SD_Ki, fit_curve)


# Back-compat alias used throughout this module.
_pbrain_tqdm = _tqdm


def export_brain_concentration_4d(
    *,
    data_4d: np.ndarray,
    T1_matrix: np.ndarray,
    M0_matrix: np.ndarray,
    brain_mask: np.ndarray,
    dce_path: str,
    analysis_directory: str,
    ref_affine,
    ref_header,
    flip_angle_deg=None,
    output_path: str | None = None,
) -> str:
    """Write a brain-masked voxelwise concentration 4D NIfTI.

    Intended to run *before* PK fitting so downstream PK stages can reuse Ct.
    Uses TurboFLASH conversion (validator parity) in batched form.
    """

    import math

    from utils.loading import resolve_turboflash_ti_s
    from utils.plotting import turboflash

    if output_path is None:
        output_path = os.path.join(
            analysis_directory,
            "CTC Data",
            "Tissue",
            "brain_concentration_4d.nii.gz",
        )

    # Hardcoded behaviour: always (re)generate the Ct map.
    # This is intentionally unconditional for p-brain-web stage runners.

    brain_mask = np.asarray(brain_mask).astype(bool)
    idx = np.argwhere(brain_mask)
    if idx.size == 0:
        raise ValueError("Brain mask is empty; cannot export concentration map.")

    batch_voxels = int(os.environ.get("P_BRAIN_CONCENTRATION_BATCH_VOXELS") or 20000)
    batch_voxels = max(256, batch_voxels)
    baseline_frames = os.environ.get("P_BRAIN_CONCENTRATION_BASELINE_FRAMES")
    if baseline_frames is None or str(baseline_frames).strip() == "":
        baseline_frames = int(getattr(settings, "TURBOFLASH_BASELINE_FRAMES", 10) or 10)
    else:
        baseline_frames = int(baseline_frames)
    baseline_frames = max(1, baseline_frames)

    ti_s = resolve_turboflash_ti_s(dce_path, default=0.12)
    td_ms = float(ti_s) * 1e3

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mm_path = output_path + ".memmap"
    ctc4d = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=data_4d.shape)

    total_chunks = int(math.ceil(idx.shape[0] / float(batch_voxels)))
    for chunk_i in _pbrain_tqdm(range(total_chunks), desc="Loading brain Ct (voxelwise)"):
        start = chunk_i * batch_voxels
        chunk = idx[start : start + batch_voxels]
        if chunk.size == 0:
            continue
        xs, ys, zs = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        S = np.asarray(data_4d[xs, ys, zs, :], dtype=np.float32)
        T1 = np.asarray(T1_matrix[xs, ys, zs], dtype=np.float32)
        M0 = np.asarray(M0_matrix[xs, ys, zs], dtype=np.float32)[:, None]

        Ct = turboflash(
            S,
            T1,
            TD=td_ms,
            m0=M0,
            prints=False,
            flip_angle_deg=flip_angle_deg,
            ctc_model="turboflash",
            baseline_frames=baseline_frames,
        ).astype(np.float32)

        ctc4d[xs, ys, zs, :] = Ct

    try:
        ctc4d.flush()
    except Exception:
        pass

    header = ref_header.copy() if ref_header is not None else None
    if header is not None:
        try:
            header.set_data_dtype(np.float32)
        except Exception:
            pass
    out_img = nib.Nifti1Image(ctc4d, affine=ref_affine, header=header)
    nib.save(out_img, output_path)

    try:
        del ctc4d
    except Exception:
        pass
    try:
        if os.path.exists(mm_path):
            os.remove(mm_path)
    except Exception:
        pass

    return output_path


logger = logging.getLogger(__name__)


def _aggregate_roi_curves(curves, *, axis=0):
    """Aggregate a stack of ROI curves using the configured statistic."""

    method = (getattr(settings, "TISSUE_ROI_AGGREGATION", "median") or "median").strip().lower()
    arr = np.asarray(curves, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if method == "mean":
        return np.nanmean(arr, axis=axis)
    return np.nanmedian(arr, axis=axis)

# Allow overriding the default mask regeneration behaviour via an
# environment variable.  By default, existing masks are re-used to avoid
# unnecessary recomputation.  Setting ``FORCE_RECREATE_MASKS`` to ``1``
# (or ``true``/``yes``) restores the previous eager regeneration.
if os.getenv("FORCE_RECREATE_MASKS", "0").lower() in {"1", "true", "yes"}:
    force_recreate_masks = True

def _estimate_aif_shift(tissue_curve, arterial_curve, max_shift_samples):
    """Return the sample shift (applied to the AIF) that maximises alignment."""

    if max_shift_samples <= 0:
        return 0, 0

    tissue = np.asarray(tissue_curve, dtype=float)
    arterial = np.asarray(arterial_curve, dtype=float)
    n = min(tissue.size, arterial.size)
    if n < 2:
        return 0, 0

    tissue = tissue[:n]
    arterial = arterial[:n]
    if not (np.all(np.isfinite(tissue)) and np.all(np.isfinite(arterial))):
        return 0, 0

    tissue_zero = tissue - tissue.mean()
    arterial_zero = arterial - arterial.mean()
    if np.allclose(tissue_zero, 0.0) or np.allclose(arterial_zero, 0.0):
        return 0, 0

    corr = np.correlate(tissue_zero, arterial_zero, mode="full")
    lags = np.arange(-n + 1, n, dtype=int)
    limit = int(max_shift_samples)
    if limit <= 0:
        return 0, 0
    mask = (lags >= -limit) & (lags <= limit)
    if not np.any(mask):
        return 0, 0

    masked_corr = corr[mask]
    masked_lags = lags[mask]
    best_idx = np.argmax(masked_corr)
    lag = int(masked_lags[best_idx])
    # ``lag`` reports how much the tissue needs to be shifted relative to the
    # arterial curve.  We return the opposite so that the caller can shift the
    # AIF instead of re-sampling every tissue voxel.
    shift = -lag
    return shift, lag


def _shift_with_zeros(arr, shift):
    """Shift ``arr`` by ``shift`` samples, padding with zeros rather than wrapping."""

    arr = np.asarray(arr, dtype=float)
    if shift == 0 or arr.size == 0:
        return arr.copy()

    result = np.zeros_like(arr)
    if shift > 0:
        result[shift:] = arr[:-shift]
    else:
        shift = -shift
        result[: arr.size - shift] = arr[shift:]
    return result


def _get_first_npy(folder):
    npy_files = [f for f in os.listdir(folder) if f.endswith('.npy') and not f.startswith('.')]
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {folder}.")
    if len(npy_files) > 1:
        print(f"[!] Warning: multiple .npy files found in {folder}; using {npy_files[0]}.")
    return npy_files[0]

def patlak_total(C_t, C_a, t):
    """Optional drop correction then Patlak fit."""
    if C_t.size == 0:
        return (np.nan, np.nan, np.nan)          # Ki, λ, SD_Ki
    if correct_signal_jumps:
        _, bad, _ = mask_problematic(C_t)        # <- same length as C_t
    else:
        bad = None
    Ki, lam, SD, *_ = patlak_with_exclusions(C_t, C_a, t, bad_mask=bad)
    return Ki, lam, SD

def two_compartment_tikhonov(aif, tissue_curve, *, time_array,
                             lambd=settings.TIKHONOV_LAMBDA,
                             penalty="identity",
                             return_residue=False):
    """Two-compartment fit using Tikhonov regularisation.

    Validator parity requirement: uses the validated slow Tikhonov solver
    (L-curve lambda selection, AIF shift inside operator, residue-derived CTH).
    """

    # Back-compat: some callers/tests expect that when
    # `TWO_COMPARTMENT_INIT_FROM_PATLAK` is enabled we seed an initial
    # guess for the legacy Extended-Tofts Tikhonov fitter.
    # The validated slow solver does not use `x0`, but we still make the call
    # so downstream hooks can observe the chosen seed.
    if bool(getattr(settings, "TWO_COMPARTMENT_INIT_FROM_PATLAK", False)):
        try:
            ki_patlak, _lam, _sd = patlak_total(np.asarray(tissue_curve), np.asarray(aif), np.asarray(time_array))
            if np.isfinite(ki_patlak):
                x0 = (float(ki_patlak) / 6000.0, 0.2, 0.05)
            else:
                x0 = (0.001, 0.2, 0.05)
            # Call through the symbol imported into this module so tests can monkeypatch it.
            extended_tofts_tikhonov(aif, tissue_curve, time_array, lambd=lambd, x0=x0)
        except Exception:
            pass

    # Keep args for API compatibility; validated solver always selects lambda via L-curve.
    if return_residue:
        return model_tikhonov_validated_tuple(
            aif,
            tissue_curve,
            time_array,
            return_residue=True,
        )
    return model_tikhonov_validated_tuple(
        aif,
        tissue_curve,
        time_array,
        return_residue=False,
    )


def _existing_tikhonov_metrics(C_t, C_a, t, *, auto_lambda=settings.AUTO_LAMBDA,
                               auto_lambda_value=settings.AUTO_LAMBDA_VALUE,
                               lambd_default=settings.TIKHONOV_LAMBDA,
                               _solver=None):
    """Replicate the legacy Tikhonov-based MTT/CTH computation.

    Parameters
    ----------
    _solver : callable, optional
        Pre-built Tikhonov solver.  When provided the expensive solver
        construction (121 Cholesky factorisations) is skipped.
    """

    if C_t.size == 0:
        return {
            "cbf": float("nan"),
            "mtt": float("nan"),
            "cth": float("nan"),
            "residue": None,
            "delta_t": None,
            "h": None,
            "impulse_response": None,
            "xcorr_shift_samples": 0,
            "xcorr_lag_samples": 0,
            "xcorr_shift_seconds": 0.0,
        }

    n = min(len(C_t), len(C_a), len(t))
    if n < 2:
        return {
            "cbf": float("nan"),
            "mtt": float("nan"),
            "cth": float("nan"),
            "residue": None,
            "delta_t": None,
            "h": None,
            "impulse_response": None,
            "xcorr_shift_samples": 0,
            "xcorr_lag_samples": 0,
            "xcorr_shift_seconds": 0.0,
        }

    C_t_use = np.asarray(C_t[:n], dtype=float)
    C_a_use = np.asarray(C_a[:n], dtype=float)
    t_use = np.asarray(t[:n], dtype=float)

    if (not np.all(np.isfinite(C_t_use)) or
            not np.all(np.isfinite(C_a_use)) or
            not np.all(np.isfinite(t_use))):
        return {
            "cbf": float("nan"),
            "mtt": float("nan"),
            "cth": float("nan"),
            "residue": None,
            "delta_t": None,
            "h": None,
            "impulse_response": None,
            "xcorr_shift_samples": 0,
            "xcorr_lag_samples": 0,
            "xcorr_shift_seconds": 0.0,
        }

    deltas = np.diff(t_use)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return {
            "cbf": float("nan"),
            "mtt": float("nan"),
            "cth": float("nan"),
            "residue": None,
            "delta_t": None,
            "h": None,
            "impulse_response": None,
            "xcorr_shift_samples": 0,
            "xcorr_lag_samples": 0,
            "xcorr_shift_seconds": 0.0,
        }

    delta_t = float(deltas[0])

    # Optional MATLAB-style voxelwise offset correction.
    # MATLAB applies the offset by shifting the AIF; equivalently, we can shift
    # the tissue curve by -offset and keep the AIF constant.
    if bool(getattr(settings, "MATLAB_OFFSET_CORRECTION", False)):
        try:
            offset_s, _ = estimate_bolus_arrival_shift_seconds(C_t_use, t_use)
            if np.isfinite(offset_s) and float(offset_s) != 0.0:
                C_t_use = shift_curve_pchip(t_use, C_t_use, -float(offset_s))
        except Exception:
            pass

    shift_applied = 0
    xcorr_lag = 0
    if settings.ALIGN_AIF_BY_XCORR:
        max_shift_samples = int(np.floor(settings.ALIGN_AIF_MAX_SHIFT_S / delta_t))
        shift_applied, xcorr_lag = _estimate_aif_shift(C_t_use, C_a_use, max_shift_samples)
        if shift_applied:
            C_a_use = _shift_with_zeros(C_a_use, shift_applied)

    try:
        if _solver is not None and not shift_applied:
            solver = _solver
        else:
            solver = build_tikhonov_validated_slow_solver(
                t_use,
                C_a_use,
                auto_lambda=bool(auto_lambda),
                auto_lambda_value=auto_lambda_value,
                lambd_default=lambd_default,
                tissue_density=float(getattr(settings, "TISSUE_DENSITY", 1.04)),
                hematocrit=float(getattr(settings, "HEMATOCRIT", 0.42)),
                plasma_derived_aif=bool(getattr(settings, "PLASMA_DERIVED_AIF", False)),
            )
        sol = solver(C_t_use.reshape(-1, 1))
    except Exception:
        return {
            "cbf": float("nan"),
            "mtt": float("nan"),
            "cth": float("nan"),
            "residue": None,
            "delta_t": None,
            "h": None,
            "impulse_response": None,
            "lambda_opt": float("nan"),
            "vd": float("nan"),
            "ss": float("nan"),
            "xcorr_shift_samples": shift_applied,
            "xcorr_lag_samples": xcorr_lag,
            "xcorr_shift_seconds": shift_applied * delta_t,
        }

    cbf = float(np.asarray(sol.cbf_ml_per_100g_min, dtype=float).reshape(-1)[0])
    mtt = float(np.asarray(sol.mtt_s, dtype=float).reshape(-1)[0])
    cth = float(np.asarray(sol.cth_s, dtype=float).reshape(-1)[0])
    lambda_opt = float(np.asarray(sol.lambda_opt, dtype=float).reshape(-1)[0])
    vd = float(np.asarray(sol.cbv_vd, dtype=float).reshape(-1)[0])
    ss = float("nan")

    impulse = None
    residue = None
    h = None
    try:
        # The validated spline solver provides residue(t) directly.
        # Recover a non-normalized impulse response as: impulse = residue * f_internal.
        _res = sol.residue
        _fin = sol.f_internal
        if _res is not None and _fin is not None:
            _res = np.asarray(_res, dtype=float)
            if _res.ndim == 2:
                _res = _res[:, 0]
            fin0 = float(np.asarray(_fin, dtype=float).reshape(-1)[0])
            if _res.size and np.isfinite(fin0):
                residue = _res
                impulse = _res * fin0
    except Exception:
        impulse = None
        residue = None

    return {
        "cbf": cbf,
        "mtt": mtt,
        "cth": cth,
        "residue": residue,
        "delta_t": delta_t,
        "h": h,
        "impulse_response": impulse,
        "lambda_opt": lambda_opt,
        "vd": vd,
        "ss": ss,
        "xcorr_shift_samples": shift_applied,
        "xcorr_lag_samples": xcorr_lag,
        "xcorr_shift_seconds": shift_applied * delta_t,
    }


def compute_mtt_cth(method, C_t, C_a, t, *, Ki=None, allow_gamma=True,
                    logger=None,
                    auto_lambda=settings.AUTO_LAMBDA,
                    auto_lambda_value=settings.AUTO_LAMBDA_VALUE,
                    lambd_default=settings.TIKHONOV_LAMBDA,
                    _solver=None):
    """Dispatch between Tikhonov, gamma, and hybrid MTT/CTH computations.

    Parameters
    ----------
    _solver : callable, optional
        Pre-built Tikhonov solver.  Forwarded to _existing_tikhonov_metrics.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    method = (method or "tikhonov").lower()
    if method not in {"tikhonov", "gamma", "hybrid"}:
        logger.warning("Unknown CTH/MTT method '%s'; defaulting to Tikhonov", method)
        method = "tikhonov"

    legacy = _existing_tikhonov_metrics(
        C_t, C_a, t,
        auto_lambda=auto_lambda,
        auto_lambda_value=auto_lambda_value,
        lambd_default=lambd_default,
        _solver=_solver,
    )

    extras = {
        "method": method,
        "tikhonov": legacy,
    }

    gamma_result = None
    if method in {"gamma", "hybrid"} and allow_gamma:
        gamma_result = gamma_fit_metrics(
            C_t, C_a, t,
            cbf_seed=legacy["cbf"],
            Ki=Ki,
        )
        if not gamma_result.get("success"):
            logger.warning("Gamma fit unavailable: %s", gamma_result.get("message", ""))
            gamma_result = {
                "success": False,
                "MTT_gamma": float("nan"),
                "CTH_gamma": float("nan"),
                "a": float("nan"),
                "b": float("nan"),
                "t0": float("nan"),
                "F_ml_per_100g_min": float("nan"),
                "E": float("nan"),
                "shape_ratio": float("nan"),
                "residual_norm": float("nan"),
            }
        extras["gamma"] = gamma_result
    elif method in {"gamma", "hybrid"} and not allow_gamma:
        extras["gamma"] = {
            "success": False,
            "MTT_gamma": float("nan"),
            "CTH_gamma": float("nan"),
            "a": float("nan"),
            "b": float("nan"),
            "t0": float("nan"),
            "F_ml_per_100g_min": float("nan"),
            "E": float("nan"),
            "shape_ratio": float("nan"),
            "residual_norm": float("nan"),
        }

    return legacy["cbf"], legacy["mtt"], legacy["cth"], extras


def extract_cth_mtt_sidecar_fields(extras):
    """Return metadata fields for JSON sidecars given compute_mtt_cth extras."""

    fields = {}
    if not isinstance(extras, dict):
        return fields

    method = extras.get("method")
    if method:
        fields["cth_mtt_method"] = method

    tikh = extras.get("tikhonov") or {}
    if tikh:
        fields["MTT_tikh_s"] = float(tikh.get("mtt", float("nan")))
        fields["CTH_tikh_s"] = float(tikh.get("cth", float("nan")))

    gamma = extras.get("gamma") or {}
    if gamma:
        iterations = gamma.get("iterations", float("nan"))
        try:
            iterations_val = int(iterations)
        except (TypeError, ValueError):
            iterations_val = 0
        fields.update({
            "MTT_gamma_s": float(gamma.get("MTT_gamma", float("nan"))),
            "CTH_gamma_s": float(gamma.get("CTH_gamma", float("nan"))),
            "gamma_a": float(gamma.get("a", float("nan"))),
            "gamma_b": float(gamma.get("b", float("nan"))),
            "gamma_t0_s": float(gamma.get("t0", float("nan"))),
            "gamma_F_ml_per_100g_min": float(gamma.get("F_ml_per_100g_min", float("nan"))),
            "gamma_E": float(gamma.get("E", float("nan"))),
            "gamma_shape_ratio": float(gamma.get("shape_ratio", float("nan"))),
            "gamma_residual_norm": float(gamma.get("residual_norm", float("nan"))),
            "gamma_iterations": iterations_val,
            "gamma_success": bool(gamma.get("success", False)),
        })

    return fields


def annotate_cth_mtt_header(img):
    """Annotate NIfTI headers with the selected CTH/MTT method."""

    method = (settings.CTH_MTT_METHOD or "tikhonov").lower()
    if method in {"gamma", "hybrid"}:
        description = f"cth_mtt_method={settings.CTH_MTT_METHOD}"[:79]
        try:
            img.header["descrip"] = description.encode("ascii", errors="ignore")
        except Exception:  # pragma: no cover - header assignment safety
            pass
    return img

def mask_problematic(ctc, *, tail_start: int = 100, thresh_factor: float = 0.5):
    """
    Replace “bad” (post-tail drop) samples in *ctc* with NaN and return
    (ctc_masked, bad_mask, drop_idxs).

    ── NEW: now **always** returns 3 values ──
    """
    ctc = np.asarray(ctc, dtype=float)

    # ── EARLY EXIT ────────────────────────────────────────────────
    if ctc.size <= tail_start + 1:
        # too short to analyse — nothing is “bad”
        return (
            ctc.astype(float),                       # ctc_masked
            np.zeros_like(ctc, dtype=bool),          # bad_mask
            np.array([], dtype=int)                  # drop_idxs
        )

    # identify the drop points
    drop_idxs, *_ = identify_drop_points(ctc, tail_start, thresh_factor)

    # boolean mask of the dropped samples
    bad_mask = np.zeros_like(ctc, dtype=bool)
    if drop_idxs.size:
        bad_mask[drop_idxs] = True

    # masked copy of the curve
    ctc_masked = ctc.copy()
    ctc_masked[bad_mask] = np.nan

    return ctc_masked, bad_mask, drop_idxs


# -------------- patlak_analysis.py (new version) --------------
def patlak_with_exclusions(C_t, C_a, t, bad_mask=None):
    """Patlak fit aligned with ``patlak_analysis_plotting`` semantics.

    The mathematical model implementation lives in `models/patlak.py`.
    """

    window_start = float(getattr(settings, "PATLAK_WINDOW_START_FRACTION", 1 / 3))
    _single_bolus = int(getattr(settings, "NUMBER_OF_PEAKS", 2)) == 1
    return model_patlak_with_exclusions(
        C_t,
        C_a,
        t,
        bad_mask,
        window_start_fraction=window_start,
        single_bolus=_single_bolus,
    )


def identify_drop_points(signal, tail_start: int = 100, threshold_factor: float = 0.5):
    """
    Detect “drop” samples occurring at/after *tail_start* where the curve
    falls more than *threshold_factor*·σ below a fitted linear tail trend.

    Parameters
    ----------
    signal : 1-D array-like
        Concentration-time curve.
    tail_start : int, default 100
        Index that marks the beginning of the tail region.
    threshold_factor : float, default 0.5
        Multiplier for the residual standard deviation that sets the
        drop threshold.

    Returns
    -------
    drop_idxs : np.ndarray (int)
        Indices judged to be ‘bad’ (empty if none).
    trend : np.ndarray | None
        The fitted linear trend across the *entire* signal,
        or *None* when the curve is too short to fit.
    thresh : float | None
        The absolute drop threshold, or *None* when no fit is done.
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size

    # ── EARLY EXIT ────────────────────────────────────────────────
    # need ≥2 points after tail_start to fit a line
    if n <= tail_start + 1:
        return np.array([], dtype=int), None, None
    # also bail if everything is NaN
    if np.all(np.isnan(signal)):
        return np.array([], dtype=int), None, None

    # standard path
    x = np.arange(tail_start, n)
    y = signal[tail_start:]

    # handle NaNs in the tail region
    good_tail = ~np.isnan(y)
    if good_tail.sum() < 2:                     # not enough data to fit
        return np.array([], dtype=int), None, None

    # fit linear trend to *clean* tail samples
    m, b = np.polyfit(x[good_tail], y[good_tail], 1)
    trend = m * np.arange(n) + b

    # residuals & threshold
    resid = signal - trend
    mu, sigma = np.nanmean(resid), np.nanstd(resid)
    thresh = mu - threshold_factor * sigma

    # flag all drops beyond threshold, only in tail
    drop_idxs = np.where((resid < thresh) & (np.arange(n) >= tail_start))[0]
    return drop_idxs.astype(int), trend, thresh

# -- Helper functions for compute_Ki_from_atlas -----------------------------

# These globals are populated in ``_init_compute_Ki`` and used by
# ``_process_label``.  They allow child processes spawned by
# ``multiprocessing`` to access the large numpy arrays without needing to
# pickle and send them with every task.
_atlas_data = None
_data_4d = None
_T1_matrix = None
_M0_matrix = None
_time_points_s = None
_C_a_full = None
_compute_CTC = None
_find_baseline_point_advanced = None
_custom_shifter = None
_patlak_analysis_plotting = None
_kinetic_model = None
_two_compartment_tikhonov = None


def _load_label_lookup(lut_path=None):
    """Return a dict mapping segmentation indices to region names."""
    if lut_path is None:
        fs_home = os.environ.get("FREESURFER_HOME")
        if fs_home:
            lut_path = os.path.join(fs_home, "FreeSurferColorLUT.txt")
    lookup = {}
    if lut_path and os.path.exists(lut_path):
        try:
            with open(lut_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = re.split(r"\s+", line)
                    if len(parts) >= 2 and parts[0].isdigit():
                        lookup[int(parts[0])] = parts[1]
        except Exception:
            pass
    return lookup


def _init_compute_Ki(atlas_data, data_4d, T1_matrix, M0_matrix, time_points_s,
                     C_a_full, compute_CTC, find_baseline_point_advanced,
                     custom_shifter, patlak_analysis_plotting,
                     kinetic_model=None, two_compartment_tikhonov=None):
    """Initialise global read-only data for compute_Ki_from_atlas workers."""

    global _atlas_data, _data_4d, _T1_matrix, _M0_matrix, _time_points_s
    global _C_a_full, _compute_CTC, _find_baseline_point_advanced
    global _custom_shifter, _patlak_analysis_plotting
    global _kinetic_model, _two_compartment_tikhonov
    global _atlas_solver

    _atlas_data = atlas_data
    _data_4d = data_4d
    _T1_matrix = T1_matrix
    _M0_matrix = M0_matrix
    _time_points_s = time_points_s
    _C_a_full = C_a_full
    _compute_CTC = compute_CTC
    _find_baseline_point_advanced = find_baseline_point_advanced
    _custom_shifter = custom_shifter
    _patlak_analysis_plotting = patlak_analysis_plotting
    _kinetic_model = (kinetic_model or settings.KINETIC_MODEL or "both").strip().lower()
    _two_compartment_tikhonov = two_compartment_tikhonov

    # Pre-build the Tikhonov solver once so all atlas labels share it.
    _atlas_solver = None
    if not settings.ALIGN_AIF_BY_XCORR and len(time_points_s) >= 2:
        try:
            _atlas_solver = build_tikhonov_validated_slow_solver(
                time_points_s,
                C_a_full,
                tissue_density=float(getattr(settings, "TISSUE_DENSITY", 1.04)),
                hematocrit=float(getattr(settings, "HEMATOCRIT", 0.42)),
                plasma_derived_aif=bool(getattr(settings, "PLASMA_DERIVED_AIF", False)),
            )
        except Exception:
            _atlas_solver = None


def _process_label(lbl):
    """Worker for ``compute_Ki_from_atlas`` to process a single label."""

    mask = (_atlas_data == lbl)
    indices = np.argwhere(mask)
    default_extras = {
        "method": settings.CTH_MTT_METHOD,
        "tikhonov": {
            "cbf": float("nan"),
            "mtt": float("nan"),
            "cth": float("nan"),
        },
    }

    if len(indices) < 1:
        return (lbl, np.nan, np.nan, np.nan, float("nan"), float("nan"), float("nan"), 0, default_extras)

    curves_for_label = []
    for (x, y, z) in indices:
        voxel_time_course = _data_4d[x, y, z, :]
        if np.isnan(voxel_time_course).any():
            continue
        T1 = _T1_matrix[x, y, z]
        M0 = _M0_matrix[x, y, z]
        c_t_0 = _compute_CTC(voxel_time_course, T1, m0=M0)
        if np.isnan(c_t_0).any():
            continue
        baseline_point = _find_baseline_point_advanced(c_t_0)
        c_t = _custom_shifter(c_t_0, baseline_point)
        if np.isnan(c_t).any():
            continue
        if np.all(c_t == 0.0):
            continue
        curves_for_label.append(c_t)

    if len(curves_for_label) == 0:
        return (lbl, np.nan, np.nan, np.nan, float("nan"), float("nan"), float("nan"), 0, default_extras)

    curves_for_label = np.array(curves_for_label)
    label_ct = _aggregate_roi_curves(curves_for_label, axis=0)

    min_len = min(len(label_ct), len(_C_a_full))
    C_t_label = label_ct[:min_len]
    C_a_label = _C_a_full[:min_len]
    t_label = _time_points_s[:min_len]

    # Canonical behavior:
    # - Patlak computes Ki/vp.
    # - Tikhonov computes perfusion metrics.
    # Avoid computing Ki via deconvolution fits.
    Ki = float('nan')
    lam = float('nan')
    SD_Ki = float('nan')
    if bool(getattr(settings, 'COMPUTE_ATLAS_KI', True)):
        try:
            Ki, lam, SD_Ki, _, _, _ = _patlak_analysis_plotting(C_t_label, C_a_label, t_label)
        except Exception:
            Ki, lam, SD_Ki = np.nan, np.nan, np.nan

    cbf = float('nan')
    mtt = float('nan')
    cth = float('nan')
    extras = {
        "method": settings.CTH_MTT_METHOD,
        "tikhonov": {
            "cbf": cbf,
            "mtt": mtt,
            "cth": cth,
        },
    }

    if len(t_label) >= 2 and bool(getattr(settings, 'COMPUTE_ATLAS_CBF', True)):
        cbf, mtt, cth, extras = compute_mtt_cth(
            settings.CTH_MTT_METHOD,
            C_t_label,
            C_a_label,
            t_label,
            Ki=(Ki if np.isfinite(Ki) else None),
            allow_gamma=True,
            logger=logger,
            _solver=_atlas_solver,
        )

    return (lbl, Ki, SD_Ki, lam, cbf, mtt, cth, len(indices), extras)


def compute_Ki_from_atlas(
    atlas_path,
    data_4d,
    T1_matrix,
    M0_matrix,
    time_points_s,
    C_a_full,
    affine,
    output_directory,
    compute_CTC,
    find_baseline_point_advanced,
    custom_shifter,
    patlak_analysis_plotting,
    *,
    compute_ki: bool = True,
    compute_cbf: bool = True,
):

    # Load the atlas and find unique labels
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)

    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels != 0]  # exclude background label=0 if present

    label_lookup = _load_label_lookup()

    # Exclude ventricular/CSF regions from parcel-based pharmacokinetic statistics.
    unique_labels = np.asarray(
        [
            lbl
            for lbl in unique_labels
            if not _pk_should_exclude_atlas_label(int(lbl), label_lookup)
        ],
        dtype=unique_labels.dtype,
    )

    # Prepare empty 3D volumes for Ki, SD(Ki), vp and perfusion metrics
    Ki_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    SD_Ki_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    vp_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    CBF_map = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    MTT_map = (np.full(atlas_data.shape, np.nan, dtype=np.float32)
               if settings.WRITE_MTT else None)
    CTH_map = (np.full(atlas_data.shape, np.nan, dtype=np.float32)
               if settings.WRITE_CTH else None)

    # Dictionary to keep numerical results per label for JSON output
    atlas_results = {}

    # Configure which atlas metrics are computed in the worker.
    settings.COMPUTE_ATLAS_KI = bool(compute_ki)
    settings.COMPUTE_ATLAS_CBF = bool(compute_cbf)

    # Initialise worker globals for multiprocessing or direct execution
    _init_compute_Ki(
        atlas_data,
        data_4d,
        T1_matrix,
        M0_matrix,
        time_points_s,
        C_a_full,
        compute_CTC,
        find_baseline_point_advanced,
        custom_shifter,
        patlak_analysis_plotting,
        kinetic_model=settings.KINETIC_MODEL,
        two_compartment_tikhonov=two_compartment_tikhonov,
    )

    total_labels = int(getattr(unique_labels, "size", len(unique_labels)))

    if total_labels == 0:
        results = []
    elif settings.MULTIPROCESSING:
        # Use imap_unordered to enable a tqdm progress bar even when running
        # in multiprocessing mode.
        chunksize = int(getattr(settings, "ATLAS_LABEL_CHUNKSIZE", 1))
        chunksize = max(1, chunksize)
        with multiprocessing.Pool(
            settings.NUMBER_OF_CORES,
            initializer=_init_compute_Ki,
            initargs=(
                atlas_data,
                data_4d,
                T1_matrix,
                M0_matrix,
                time_points_s,
                C_a_full,
                compute_CTC,
                find_baseline_point_advanced,
                custom_shifter,
                patlak_analysis_plotting,
                settings.KINETIC_MODEL,
                two_compartment_tikhonov,
            ),
        ) as pool:
            with auto_logging_suppressed():
                iterator = pool.imap_unordered(_process_label, unique_labels, chunksize=chunksize)
                results = list(
                    _pbrain_tqdm(
                        iterator,
                        total=total_labels,
                        desc="Pharmacokinetic modelling (atlas labels)",
                    )
                )
    else:
        with auto_logging_suppressed():
            results = [
                _process_label(lbl)
                for lbl in _pbrain_tqdm(
                    unique_labels,
                    total=total_labels,
                    desc="Pharmacokinetic modelling (atlas labels)",
                )
            ]

    for lbl, Ki, SD_Ki, lam, cbf, mtt, cth, voxel_count, extras in results:
        mask = (atlas_data == lbl)

        if compute_ki and np.isfinite(Ki):
            if np.isfinite(lam):
                lam = float(max(lam, 0.0))
            Ki_map[mask] = float(Ki)
            SD_Ki_map[mask] = float(SD_Ki) if np.isfinite(SD_Ki) else float('nan')
            vp_map[mask] = float(lam) if np.isfinite(lam) else float('nan')

        if compute_cbf:
            CBF_map[mask] = float(cbf) if np.isfinite(cbf) else float('nan')
            if MTT_map is not None:
                MTT_map[mask] = float(mtt) if np.isfinite(mtt) else float('nan')
            if CTH_map is not None:
                CTH_map[mask] = float(cth) if np.isfinite(cth) else float('nan')

        label_key = label_lookup.get(int(lbl), str(lbl))
        atlas_entry = {
            "voxel_count": int(voxel_count),
        }
        if compute_ki:
            atlas_entry.update(
                {
                    "Ki": float(Ki) if np.isfinite(Ki) else float('nan'),
                    "SD_Ki": float(SD_Ki) if np.isfinite(SD_Ki) else float('nan'),
                    "vp": float(lam) if np.isfinite(lam) else float('nan'),
                }
            )
        if compute_cbf:
            atlas_entry.update(
                {
                    "CBF_tikhonov": float(cbf) if np.isfinite(cbf) else float('nan'),
                    "MTT_tikhonov": float(mtt) if np.isfinite(mtt) else float('nan'),
                    "CTH_tikhonov": float(cth) if np.isfinite(cth) else float('nan'),
                }
            )
            atlas_entry.update(extract_cth_mtt_sidecar_fields(extras))
        atlas_results[label_key] = atlas_entry

    # Save results as NIfTI
    os.makedirs(output_directory, exist_ok=True)

    if compute_ki:
        Ki_nii = nib.Nifti1Image(Ki_map, affine)
        SD_Ki_nii = nib.Nifti1Image(SD_Ki_map, affine)
        vp_nii = nib.Nifti1Image(vp_map, affine)
        nib.save(Ki_nii, os.path.join(output_directory, 'Ki_map_atlas.nii.gz'))
        nib.save(SD_Ki_nii, os.path.join(output_directory, 'SD_Ki_map_atlas.nii.gz'))
        nib.save(vp_nii, os.path.join(output_directory, 'vp_map_atlas.nii.gz'))

    if compute_cbf:
        nib.save(
            nib.Nifti1Image(CBF_map, affine),
            os.path.join(output_directory, 'CBF_tikhonov_map_atlas.nii.gz'),
        )
        if MTT_map is not None:
            mtt_img = annotate_cth_mtt_header(nib.Nifti1Image(MTT_map, affine))
            nib.save(mtt_img, os.path.join(output_directory, 'MTT_tikhonov_map_atlas.nii.gz'))
        if CTH_map is not None:
            cth_img = annotate_cth_mtt_header(nib.Nifti1Image(CTH_map, affine))
            nib.save(cth_img, os.path.join(output_directory, 'CTH_tikhonov_map_atlas.nii.gz'))

    # Save numerical results to JSON.
    # p-brain-web expects model-specific filenames; we write both while keeping
    # the legacy name for compatibility.
    json_paths = [
        os.path.join(output_directory, 'Ki_values_atlas.json'),
        os.path.join(output_directory, 'Ki_values_atlas_patlak.json'),
        os.path.join(output_directory, 'Ki_values_atlas_tikhonov.json'),
    ]
    for json_path in json_paths:
        with open(json_path, 'w') as jf:
            json.dump(atlas_results, jf, indent=4)

    print("Done. Wrote:")
    print(f"  cth_mtt_method={settings.CTH_MTT_METHOD}")
    if compute_ki:
        print("  Ki_map_atlas.nii.gz")
        print("  SD_Ki_map_atlas.nii.gz")
        print("  vp_map_atlas.nii.gz")
    if compute_cbf:
        print("  CBF_tikhonov_map_atlas.nii.gz")
        if MTT_map is not None:
            print("  MTT_tikhonov_map_atlas.nii.gz")
        if CTH_map is not None:
            print("  CTH_tikhonov_map_atlas.nii.gz")


def construct_convolution_matrix(C_a, delta_t):
    return km_construct_convolution_matrix(C_a, delta_t)


def _json_finite_to_none(obj):
    """Convert NaN/Inf floats to None for strict JSON output."""
    try:
        import numpy as _np
    except Exception:  # pragma: no cover
        _np = None

    if obj is None:
        return None

    # Numpy scalars
    if _np is not None and isinstance(obj, _np.generic):
        try:
            obj = obj.item()
        except Exception:
            obj = float(obj)

    if isinstance(obj, float):
        try:
            return obj if np.isfinite(obj) else None
        except Exception:
            return obj

    if isinstance(obj, dict):
        return {k: _json_finite_to_none(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_finite_to_none(v) for v in obj]

    # Avoid leaking arrays into JSON; convert to list if encountered.
    if _np is not None and isinstance(obj, _np.ndarray):
        return _json_finite_to_none(obj.tolist())

    return obj


_PK_EXCLUDED_ATLAS_LABEL_IDS = {
    # FreeSurfer aseg common CSF/ventricle-related structures.
    4,   # Left-Lateral-Ventricle
    5,   # Left-Inf-Lat-Vent
    14,  # 3rd-Ventricle
    15,  # 4th-Ventricle
    24,  # CSF
    43,  # Right-Lateral-Ventricle
    44,  # Right-Inf-Lat-Vent
    72,  # 5th-Ventricle
}


def _pk_should_exclude_atlas_label(label: int, label_lookup: dict) -> bool:
    if int(label) in _PK_EXCLUDED_ATLAS_LABEL_IDS:
        return True

    name = label_lookup.get(int(label), "") if isinstance(label_lookup, dict) else ""
    lower = str(name).lower()
    if not lower:
        return False

    return (
        "ventricle" in lower
        or lower.endswith("vent")
        or "csf" in lower
    )


def tikhonov_regularization(A, C_t, lambd, *, penalty="identity"):
    return km_tikhonov_regularization(A, C_t, lambd, penalty=penalty)

def plot_predictions_with_masks(image, wm_mask, cortical_gm_mask, subcortical_gm_mask, gm_brainstem_mask, gm_cerebellum_mask, wm_cerebellum_mask, wm_cc_mask, image_directory):
    n_slices = image.shape[2]
    n_cols = 5
    n_rows = (n_slices + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

    for i in range(n_slices):
        row = i // n_cols
        col = i % n_cols

        image_slice = np.rot90(image[:, :, i])
        wm_slice = np.rot90(wm_mask[:, :, i])
        cortical_gm_slice = np.rot90(cortical_gm_mask[:, :, i])
        subcortical_gm_slice = np.rot90(subcortical_gm_mask[:, :, i])
        gm_brainstem_slice = np.rot90(gm_brainstem_mask[:, :, i])
        gm_cerebellum_slice = np.rot90(gm_cerebellum_mask[:, :, i])
        wm_cerebellum_slice = np.rot90(wm_cerebellum_mask[:, :, i])
        wm_cc_slice = np.rot90(wm_cc_mask[:, :, i])

        color_overlay = np.zeros((*image_slice.shape, 3))
        color_overlay[:, :, 2][wm_slice == 1] = 1.0  # Blue channel for white matter

        # Assign bright red to cortical gray matter
        color_overlay[:, :, 0][cortical_gm_slice == 1] = 1.0  # Bright red

        # Assign dark red to subcortical gray matter
        color_overlay[:, :, 0][subcortical_gm_slice == 1] = 0.5  # Darker red

        # Assign orange to brainstem (red + green)
        color_overlay[:, :, 0][gm_brainstem_slice == 1] = 1.0  # Red channel
        color_overlay[:, :, 1][gm_brainstem_slice == 1] = 0.5  # Green channel

        # Assign yellow to cerebellum GM (red + green)
        color_overlay[:, :, 0][gm_cerebellum_slice == 1] = 1.0  # Red channel
        color_overlay[:, :, 1][gm_cerebellum_slice == 1] = 1.0  # Green channel

        # Assign cyan to cerebellum WM (green + blue)
        color_overlay[:, :, 1][wm_cerebellum_slice == 1] = 1.0  # Green channel
        color_overlay[:, :, 2][wm_cerebellum_slice == 1] = 1.0  # Blue channel

        # Assign magenta to corpus callosum WM (red + blue)
        color_overlay[:, :, 0][wm_cc_slice == 1] = 1.0  # Red channel
        color_overlay[:, :, 2][wm_cc_slice == 1] = 1.0  # Blue channel

        ax = axes[row, col]
        ax.imshow(image_slice, cmap='gray')
        ax.imshow(color_overlay, alpha=0.5)
        ax.set_title(f'Slice {i+1}')

        ax.grid(False)
        ax.axis("off")

    # Remove empty subplots
    for j in range(n_slices, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        plt.tight_layout()
    os.makedirs(os.path.join(image_directory, 'AI', 'Segmentation'), exist_ok=True)
    plt.savefig(os.path.join(image_directory, 'AI', 'Segmentation', 'T2_WM_GM_masks.png'))
    if not turbo_mode:
        def on_esc(event):
            if getattr(event, "key", None) == "escape":
                plt.close(getattr(event, "canvas", None).figure if getattr(event, "canvas", None) else fig)

        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay(3, fig)
        plt.show()
    else:
        plt.close(fig)


def _clean_dot_underscore(root_dir: str) -> None:
    """Remove macOS AppleDouble ``._*`` resource-fork files under *root_dir*.

    On external drives (ExFAT / HFS+) macOS silently creates ``._`` companion
    files.  FreeSurfer's internal ``rm`` calls then prompt for confirmation on
    these hidden files, which stalls unattended pipeline runs.  Calling this
    before **and** after every subprocess that touches the segmentation tree
    prevents the prompt from ever appearing.
    """
    if not os.path.isdir(root_dir):
        return
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.startswith("._"):
                try:
                    os.remove(os.path.join(dirpath, fn))
                except OSError:
                    pass


def _fs_shell_preamble() -> str:
    """Return a shell preamble string for FreeSurfer / FastSurfer subprocesses.

    Includes:
    - ``COPYFILE_DISABLE=1`` – prevents macOS ``cp`` from creating ``._`` files.
    - A shell function ``rm() { command rm -f "$@"; }`` – so that any ``rm``
      invoked by FreeSurfer scripts silently removes files without prompting
      for confirmation on macOS resource-fork ``._`` artefacts.
    - Re-exports of ``FSLDIR``, ``FSLOUTPUTTYPE``, and ``FREESURFER_HOME`` so
      they survive a ``shell=True`` subprocess even when launched from a GUI.
    """
    parts = [
        'export COPYFILE_DISABLE=1',
        # Override rm for the whole subprocess tree so FreeSurfer's internal
        # cleanup never stalls on ._* files with restrictive permissions.
        'rm() { command rm -f "$@"; }',
    ]
    _fsldir = os.environ.get("FSLDIR", "").strip()
    _fslout = os.environ.get("FSLOUTPUTTYPE", "NIFTI_GZ")
    if _fsldir:
        parts.append(f"export FSLDIR={_fsldir} FSLOUTPUTTYPE={_fslout}")
    _fs_home = os.environ.get("FREESURFER_HOME", "").strip()
    if _fs_home:
        parts.append(f"export FREESURFER_HOME={_fs_home}")
    return " && ".join(parts) + " && "


def segmentation(
    fastsurfer_path,
    seg_mgz_path,
    t1_path,
    output_dir,
    sid,
    apple_metal=True,
    rerun=False,
    method="fastsurfer",
):
    method = (method or "fastsurfer").lower()

    # Resolve FreeSurfer license path once for all methods that need it.
    _fs_license = os.environ.get("FS_LICENSE", "").strip()
    _fs_home = os.environ.get("FREESURFER_HOME", "").strip()
    if not _fs_license and _fs_home:
        for _lic_name in ("license.txt", ".license", "license.dat"):
            _lic_candidate = os.path.join(_fs_home, _lic_name)
            if os.path.isfile(_lic_candidate):
                _fs_license = _lic_candidate
                break

    if method == "fastsurfer":
        # Check if FastSurfer is installed
        if not os.path.exists(fastsurfer_path):
            raise Exception(
                "FastSurfer not found, ensure correct installation and configuration of path."
            )

        # FastSurfer changes the naming of the generated stats file depending on
        # the version.  Older releases used ``asegdkt.stats`` while newer
        # versions ship ``aseg+DKT.stats``.  To support both we consider the
        # possible names when checking for existing results.
        seg_stats_candidates = [
            os.path.join(output_dir, sid, "stats", "aseg+DKT.stats"),
            os.path.join(output_dir, sid, "stats", "asegdkt.stats"),
        ]

        seg_stats_path = next(
            (p for p in seg_stats_candidates if os.path.exists(p)),
            seg_stats_candidates[0],
        )

        def seg_stats_exists():
            return any(os.path.exists(p) for p in seg_stats_candidates)

        # Run FastSurfer if the segmentation or stats files don't exist or rerun is forced
        if rerun or not (os.path.exists(seg_mgz_path) and seg_stats_exists()):
            if os.path.exists(seg_mgz_path) and seg_stats_exists():
                print("Rerunning FastSurfer segmentation...")
            else:
                print("Segmentation output not found, running FastSurfer...")
            # Shell preamble: exports env vars, disables ._ creation,
            # and overrides rm with rm -f so FreeSurfer never stalls.
            _env_exports = _fs_shell_preamble()
            _lic_flag = f"--fs_license {_fs_license} " if _fs_license else ""
            _clean_dot_underscore(output_dir)  # pre-clean before launch
            if apple_metal:
                command = (
                    f"{_env_exports}"
                    f"export PYTORCH_ENABLE_MPS_FALLBACK=1 && "
                    f"{fastsurfer_path} --device mps "
                    f"{_lic_flag}"
                    f"--t1 {t1_path} "
                    f"--sid {sid} "
                    f"--sd {output_dir}"
                )
            else:
                command = (
                    f"{_env_exports}"
                    f"{fastsurfer_path} "
                    f"{_lic_flag}"
                    f"--t1 {t1_path} "
                    f"--sid {sid} "
                    f"--sd {output_dir}"
                )
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            _clean_dot_underscore(output_dir)
            if result.returncode != 0 or not (os.path.exists(seg_mgz_path) and seg_stats_exists()):
                print("FastSurfer segmentation failed, attempting run with vox_size 1 ...")
                _clean_dot_underscore(output_dir)  # pre-clean before retry
                if apple_metal:
                    command = (
                        f"{_env_exports}"
                        f"export PYTORCH_ENABLE_MPS_FALLBACK=1 && "
                        f"{fastsurfer_path} --device mps "
                        f"{_lic_flag}"
                        f"--t1 {t1_path} "
                        f"--vox_size 1 "
                        f"--sid {sid} "
                        f"--sd {output_dir} "
                        f"--no_cereb"
                    )
                else:
                    command = (
                        f"{_env_exports}"
                        f"{fastsurfer_path} "
                        f"{_lic_flag}"
                        f"--t1 {t1_path} "
                        f"--vox_size 1 "
                        f"--sid {sid} "
                        f"--sd {output_dir} "
                        f"--no_cereb"
                    )
                subprocess.run(command, shell=True)
                _clean_dot_underscore(output_dir)
                if not (os.path.exists(seg_mgz_path) and seg_stats_exists()):
                    raise RuntimeError("FastSurfer segmentation failed even with vox_size 1")
        else:
            print("Segmentation file already exists, skipping FastSurfer segmentation.")

    elif method == "synthseg":
        # ------------------------------------------------------------------ #
        #  SynthSeg  (FreeSurfer >= 8.0)
        #  ``mri_synthseg`` produces a whole-brain parcellation from a single
        #  T1w input without atlas registration.  The output is a standard
        #  FreeSurfer-compatible aseg volume.
        # ------------------------------------------------------------------ #
        import shutil as _shutil
        synthseg_bin = _shutil.which("mri_synthseg")
        if not synthseg_bin:
            fs_home = os.environ.get("FREESURFER_HOME", "")
            candidate = os.path.join(fs_home, "bin", "mri_synthseg") if fs_home else ""
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                synthseg_bin = candidate
        if not synthseg_bin:
            raise FileNotFoundError(
                "mri_synthseg not found on PATH or under FREESURFER_HOME. "
                "SynthSeg requires FreeSurfer 8.0 or later."
            )

        # SynthSeg writes directly to a NIfTI / mgz output file.
        # We place it so it matches the path the rest of the pipeline expects.
        mri_dir = os.path.dirname(seg_mgz_path)
        os.makedirs(mri_dir, exist_ok=True)

        if rerun or not os.path.exists(seg_mgz_path):
            if os.path.exists(seg_mgz_path):
                print("Rerunning SynthSeg segmentation...")
            else:
                print("Segmentation output not found, running SynthSeg...")

            _preamble = _fs_shell_preamble()
            command = (
                f"{_preamble} "
                f"{synthseg_bin} "
                f"--i {t1_path} "
                f"--o {seg_mgz_path} "
                f"--robust "
                f"--vol {os.path.join(mri_dir, 'synthseg_volumes.csv')} "
                f"--qc {os.path.join(mri_dir, 'synthseg_qc.csv')}"
            )
            _clean_dot_underscore(output_dir)  # pre-clean before launch
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            _clean_dot_underscore(output_dir)
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(
                    f"SynthSeg segmentation failed (exit {result.returncode}): "
                    f"{err[-800:]}"
                )
            if not os.path.exists(seg_mgz_path):
                raise RuntimeError(
                    "SynthSeg completed but the output file was not created: "
                    f"{seg_mgz_path}"
                )
            print("SynthSeg segmentation completed.")
        else:
            print("Segmentation file already exists, skipping SynthSeg.")

    elif method == "recon-all":
        # ------------------------------------------------------------------ #
        #  recon-all  (FreeSurfer < 8.0)
        #  Classic full cortical reconstruction.  We only run the
        #  ``-autorecon1`` stage (skull-strip + intensity norm) and
        #  ``-autorecon2`` (segmentation / parcellation) to get the
        #  aparc+aseg volume that the rest of the pipeline needs.
        # ------------------------------------------------------------------ #
        import shutil as _shutil
        recon_bin = _shutil.which("recon-all")
        if not recon_bin:
            raise FileNotFoundError(
                "recon-all not found on PATH.  Ensure FreeSurfer is "
                "installed and FREESURFER_HOME is sourced."
            )

        # recon-all outputs to $SUBJECTS_DIR/sid.  The aseg file we need is
        # aparc.DKTatlas+aseg.mgz (or aparc+aseg.mgz on older versions).
        subjects_dir = output_dir
        aseg_candidate_names = [
            os.path.join(subjects_dir, sid, "mri", "aparc.DKTatlas+aseg.mgz"),
            os.path.join(subjects_dir, sid, "mri", "aparc+aseg.mgz"),
        ]

        def _recon_output_exists():
            return any(os.path.exists(p) for p in aseg_candidate_names)

        if rerun or not _recon_output_exists():
            if _recon_output_exists():
                print("Rerunning recon-all segmentation...")
            else:
                print("Segmentation output not found, running recon-all...")

            _preamble = _fs_shell_preamble()
            command = (
                f"{_preamble} "
                f"{recon_bin} "
                f"-i {t1_path} "
                f"-s {sid} "
                f"-sd {subjects_dir} "
                f"-all"
            )
            _clean_dot_underscore(output_dir)  # pre-clean before launch
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            _clean_dot_underscore(output_dir)
            if result.returncode != 0 and not _recon_output_exists():
                err = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(
                    f"recon-all segmentation failed (exit {result.returncode}): "
                    f"{err[-800:]}"
                )

            # Symlink or copy the recon-all output to the path the pipeline
            # expects (aparc.DKTatlas+aseg.deep.mgz).
            if not os.path.exists(seg_mgz_path):
                source = next(
                    (p for p in aseg_candidate_names if os.path.exists(p)), None
                )
                if source:
                    os.makedirs(os.path.dirname(seg_mgz_path), exist_ok=True)
                    try:
                        os.symlink(source, seg_mgz_path)
                    except OSError:
                        import shutil as _shutil2
                        _shutil2.copy2(source, seg_mgz_path)
                else:
                    raise RuntimeError(
                        "recon-all completed but no aparc+aseg volume found."
                    )
            print("recon-all segmentation completed.")
        else:
            print("Segmentation file already exists, skipping recon-all.")

    else:
        print(f"Segmentation method '{method}' selected. Skipping automated execution.")
        if not os.path.exists(seg_mgz_path):
            raise FileNotFoundError(
                f"Segmentation file not found: {seg_mgz_path}. Provide your own segmentation before running."
            )

    aseg_mgz_path = seg_mgz_path

    # Convert aseg.mgz to aseg.nii if needed
    aseg_nii_path = aseg_mgz_path.replace('.mgz', '.nii.gz')
    if not os.path.exists(aseg_nii_path):
        print(f"Converting {aseg_mgz_path} to {aseg_nii_path}...")
        subprocess.run(['mri_convert', aseg_mgz_path, aseg_nii_path])
        _clean_dot_underscore(output_dir)
    else:
        print(f"{aseg_nii_path} already exists, skipping conversion.")

    # Paths for the masks
    cortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'cortical_gm.nii.gz')
    subcortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'subcortical_gm.nii.gz')
    wm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'wm.nii.gz')

    # Create masks using predefined flags
    # White Matter Mask
    temp_wm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'temp_wm.nii.gz')
    if force_recreate_masks or not os.path.exists(temp_wm_mask_path):
        wm_command = f"mri_binarize --i {aseg_nii_path} --all-wm --o {temp_wm_mask_path}"
        subprocess.run(wm_command, shell=True)
    else:
        print("Temporary WM mask already exists, skipping mri_binarize for temp WM.")

    # Subcortical Gray Matter Mask
    temp_subcortical_gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'temp_subcortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(temp_subcortical_gm_mask_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_nii_path} --subcort-gm --o {temp_subcortical_gm_mask_path}"
        subprocess.run(subcortical_gm_command, shell=True)
    else:
        print("Temporary subcortical GM mask already exists, skipping mri_binarize for temp subcortical GM.")

    # Cortical Gray Matter Mask
    # Create overall gray matter mask
    gm_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'gm.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_mask_path):
        gm_command = f"mri_binarize --i {aseg_nii_path} --gm --o {gm_mask_path}"
        subprocess.run(gm_command, shell=True)
    else:
        print("GM mask already exists, skipping mri_binarize for GM.")

    # Create gm_brainstem_mask
    gm_brainstem_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'gm_brainstem.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_brainstem_mask_path):
        gm_brainstem_command = f"mri_binarize --i {aseg_nii_path} --match 16 --o {gm_brainstem_mask_path}"
        subprocess.run(gm_brainstem_command, shell=True)
    else:
        print("Brainstem GM mask already exists, skipping mri_binarize for brainstem GM.")

    # Create gm_cerebellum_mask
    gm_cerebellum_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'gm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_cerebellum_mask_path):
        gm_cerebellum_command = f"mri_binarize --i {aseg_nii_path} --match 8 47 --o {gm_cerebellum_mask_path}"
        subprocess.run(gm_cerebellum_command, shell=True)
    else:
        print("Cerebellum GM mask already exists, skipping mri_binarize for cerebellum GM.")

    # Create wm_cerebellum_mask
    wm_cerebellum_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'wm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cerebellum_mask_path):
        wm_cerebellum_command = f"mri_binarize --i {aseg_nii_path} --match 7 46 --o {wm_cerebellum_mask_path}"
        subprocess.run(wm_cerebellum_command, shell=True)
    else:
        print("Cerebellum WM mask already exists, skipping mri_binarize for cerebellum WM.")

    # Create wm_cc_mask
    wm_cc_mask_path = os.path.join(os.path.dirname(aseg_mgz_path), 'wm_cc.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cc_mask_path):
        wm_cc_command = f"mri_binarize --i {aseg_nii_path} --match 251 252 253 254 255 --o {wm_cc_mask_path}"
        subprocess.run(wm_cc_command, shell=True)
    else:
        print("Corpus Callosum WM mask already exists, skipping mri_binarize for corpus callosum WM.")

    # Create cortical gray matter mask by subtracting subcortical gray matter, brainstem, and cerebellum from total gray matter
    if force_recreate_masks or not os.path.exists(cortical_gm_mask_path):
        cortical_gm_command = f"fslmaths {gm_mask_path} -sub {temp_subcortical_gm_mask_path} -sub {gm_brainstem_mask_path} -sub {gm_cerebellum_mask_path} -thr 0.5 -bin {cortical_gm_mask_path}"
        subprocess.run(cortical_gm_command, shell=True)
    else:
        print("Cortical GM mask already exists, skipping creation.")

    # Create subcortical gray matter mask by subtracting brainstem and cerebellum from the temp subcortical GM mask
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_path):
        subcortical_gm_command = f"fslmaths {temp_subcortical_gm_mask_path} -sub {gm_brainstem_mask_path} -sub {gm_cerebellum_mask_path} -thr 0.5 -bin {subcortical_gm_mask_path}"
        subprocess.run(subcortical_gm_command, shell=True)
    else:
        print("Subcortical GM mask already exists, skipping creation.")

    # Create white matter mask by subtracting cerebellar WM and corpus callosum from the temp WM mask
    if force_recreate_masks or not os.path.exists(wm_mask_path):
        wm_command = f"fslmaths {temp_wm_mask_path} -sub {wm_cerebellum_mask_path} -sub {wm_cc_mask_path} -thr 0.5 -bin {wm_mask_path}"
        subprocess.run(wm_command, shell=True)
    else:
        print("WM mask already exists, skipping creation.")

    # Optionally, remove temporary files
    if os.path.exists(temp_wm_mask_path):
        os.remove(temp_wm_mask_path)
    if os.path.exists(temp_subcortical_gm_mask_path):
        os.remove(temp_subcortical_gm_mask_path)
    if os.path.exists(gm_mask_path):
        os.remove(gm_mask_path)

    # Final sweep: remove any ._ files left by macOS on external drives.
    _clean_dot_underscore(output_dir)

def plot_total_ct_and_patlak(time_points, C_t_total, C_a,
                             Ki, lam, SD_Ki, tissue_name,
                             fit_curve=None, save_path=None):
    """
    Re-written to drop the black cross markers on the Patlak panel.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    if C_t_total.size == 0 or C_a.size == 0:
        return

    if correct_signal_jumps:
        drop_idxs, trend, thresh = identify_drop_points(C_t_total)
    else:
        drop_idxs = np.array([], dtype=int)

    # Patlak co-ordinates
    dt = np.diff(time_points)
    x_patlak = np.zeros_like(C_a)
    y_patlak = np.zeros_like(C_a)
    for i in range(1, len(C_a)):
        if C_a[i] == 0:
            continue
        x_patlak[i] = np.sum(C_a[:i]*dt[:i]) / C_a[i]
        y_patlak[i] = C_t_total[i] / C_a[i]
    valid = (C_a!=0) & (x_patlak!=0) & (y_patlak!=0)
    x_pat, y_pat = x_patlak[valid], y_patlak[valid]
    idx_valid = np.where(valid)[0]
    bad_mask_pat = np.isin(idx_valid, drop_idxs)

    # ------------- figure ----------------
    fig = plt.figure(figsize=(12,5))
    gs  = plt.GridSpec(1,2,width_ratios=[2,1], wspace=0.35)

    # ---- CTC panel
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_points, C_t_total, color='blue', lw=2, label=f'{tissue_name} C(t)')
    if drop_idxs.size:
        t0, t1 = time_points[drop_idxs[0]], time_points[drop_idxs[-1]]
        ax1.axvspan(t0, t1, color='grey', alpha=0.3)
        ax1.scatter(time_points[drop_idxs], C_t_total[drop_idxs],
                    facecolors='none', edgecolors='black', s=50,
                    label='Ignored')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tissue C(t)')
    ax1.grid(True)
    ax1.legend(loc='upper right')

    ax1_a = ax1.twinx()
    ax1_a.plot(time_points, C_a, color='red', ls='--', lw=2, label='AIF')
    ax1_a.set_ylabel('C_a(t)')
    ax1_a.tick_params(axis='y', labelcolor='red')
    # merge legends
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax1_a.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')

    # ---- Patlak or two-compartment panel
    ax2 = fig.add_subplot(gs[1])

    if settings.KINETIC_MODEL.lower() in {'patlak', 'both'}:
        keep = ~bad_mask_pat
        ax2.scatter(x_pat[keep], y_pat[keep],
                    color='blue', s=25, marker='o',
                    label='Used in fit')

        # now explicitly plot *all* the dropped points hollow:
        ax2.scatter(x_pat[bad_mask_pat], y_pat[bad_mask_pat],
                    facecolors='none', edgecolors='blue', s=40,
                    label='Ignored')

        if not np.isnan(Ki):
            ax2.plot(x_pat, lam/100 + (Ki/6000)*x_pat,
                     color='green', ls='--', label='Patlak fit')

        ax2.set_xlabel('∫C_a dt / C_a')
        ax2.set_ylabel('C_t / C_a')
        ax2.set_title(f"{tissue_name} | Ki={Ki:.4f}, λ={lam:.4f}")
        ax2.grid(True)
        ax2.legend(loc='best')
    elif settings.KINETIC_MODEL.lower() == 'two_compartment':
        ax2.plot(time_points, C_t_total, 'o', label='Data')
        if fit_curve is not None:
            ax2.plot(time_points, fit_curve, '-', label='Tikhonov fit')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('C_t')
        ax2.set_title('Two-Compartment (Tikhonov) Fit')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No fit', ha='center', va='center')


    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)




def coregistration(seg_mgz_path, dce_path, t2_path):
    import subprocess
    import nibabel as nib
    import numpy as np
    import os

    # Step 1: Convert segmentation file from .mgz to .nii.gz format
    aseg_nii_path = seg_mgz_path.replace('.mgz', '.nii.gz')
    if not os.path.exists(aseg_nii_path):
        print(f"Converting {seg_mgz_path} to {aseg_nii_path}...")
        result = subprocess.run(['mri_convert', seg_mgz_path, aseg_nii_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_convert failed with error:\n{result.stderr}")
            raise RuntimeError("mri_convert command failed.")
    else:
        print(f"{aseg_nii_path} already exists, skipping conversion.")

    # Ensure the converted file exists
    if not os.path.exists(aseg_nii_path):
        raise FileNotFoundError(f"Converted segmentation file not found: {aseg_nii_path}")

    # Step 2: Align the segmentation image to the DCE space
    aseg_in_dce_path = aseg_nii_path.replace('.nii.gz', '_in_DCE.nii.gz')
    if not os.path.exists(aseg_in_dce_path):
        if use_flirt_registration:
            mat_dce = aseg_nii_path.replace('.nii.gz', '_to_DCE.mat')
            flirt_reg_dce = [
                'flirt', '-in', aseg_nii_path, '-ref', dce_path,
                '-omat', mat_dce, '-dof', '6'
            ]
            print(f"Running FLIRT registration for DCE: {' '.join(flirt_reg_dce)}")
            result = subprocess.run(flirt_reg_dce, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT registration failed for DCE with error:\n{result.stderr}")
                raise RuntimeError("FLIRT registration for DCE failed.")

            flirt_apply_dce = [
                'flirt', '-in', aseg_nii_path, '-ref', dce_path,
                '-applyxfm', '-init', mat_dce, '-interp', 'nearestneighbour',
                '-out', aseg_in_dce_path
            ]
            print(f"Applying transform for DCE: {' '.join(flirt_apply_dce)}")
            result = subprocess.run(flirt_apply_dce, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT applyxfm failed for DCE with error:\n{result.stderr}")
                raise RuntimeError("FLIRT applyxfm for DCE failed.")
        else:
            flirt_cmd_dce = [
                'flirt', '-in', aseg_nii_path, '-ref', dce_path,
                '-applyxfm', '-usesqform', '-interp', 'nearestneighbour',
                '-out', aseg_in_dce_path
            ]
            print(f"Running FLIRT command for DCE: {' '.join(flirt_cmd_dce)}")
            result = subprocess.run(flirt_cmd_dce, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT failed for DCE with error:\n{result.stderr}")
                raise RuntimeError("FLIRT command for DCE failed.")
    else:
        print(f"Aligned segmentation to DCE already exists at {aseg_in_dce_path}.")

    # Ensure the output file was created
    if not os.path.exists(aseg_in_dce_path):
        raise FileNotFoundError(f"Expected output not found: {aseg_in_dce_path}")

    # Step 3: Align the segmentation image to the T2 space
    aseg_in_t2_path = aseg_nii_path.replace('.nii.gz', '_in_T2.nii.gz')
    if not os.path.exists(aseg_in_t2_path):
        if use_flirt_registration:
            mat_t2 = aseg_nii_path.replace('.nii.gz', '_to_T2.mat')
            flirt_reg_t2 = [
                'flirt', '-in', aseg_nii_path, '-ref', t2_path,
                '-omat', mat_t2, '-dof', '6'
            ]
            print(f"Running FLIRT registration for T2: {' '.join(flirt_reg_t2)}")
            result = subprocess.run(flirt_reg_t2, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT registration failed for T2 with error:\n{result.stderr}")
                raise RuntimeError("FLIRT registration for T2 failed.")

            flirt_apply_t2 = [
                'flirt', '-in', aseg_nii_path, '-ref', t2_path,
                '-applyxfm', '-init', mat_t2, '-interp', 'nearestneighbour',
                '-out', aseg_in_t2_path
            ]
            print(f"Applying transform for T2: {' '.join(flirt_apply_t2)}")
            result = subprocess.run(flirt_apply_t2, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT applyxfm failed for T2 with error:\n{result.stderr}")
                raise RuntimeError("FLIRT applyxfm for T2 failed.")
        else:
            flirt_cmd_t2 = [
                'flirt', '-in', aseg_nii_path, '-ref', t2_path,
                '-applyxfm', '-usesqform', '-interp', 'nearestneighbour',
                '-out', aseg_in_t2_path
            ]
            print(f"Running FLIRT command for T2: {' '.join(flirt_cmd_t2)}")
            result = subprocess.run(flirt_cmd_t2, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FLIRT failed for T2 with error:\n{result.stderr}")
                raise RuntimeError("FLIRT command for T2 failed.")
    else:
        print(f"Aligned segmentation to T2 already exists at {aseg_in_t2_path}.")

    # Ensure the output file was created
    if not os.path.exists(aseg_in_t2_path):
        raise FileNotFoundError(f"Expected output not found: {aseg_in_t2_path}")

    # Step 4: Create masks from the aligned segmentation images
    # For DCE space
    wm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_wm.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_mask_dce_path):
        wm_command = f"mri_binarize --i {aseg_in_dce_path} --all-wm --o {wm_mask_dce_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for WM in DCE space failed.")
    else:
        print("WM mask in DCE space already exists, skipping mri_binarize for WM.")

    subcortical_gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_subcortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_dce_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_in_dce_path} --subcort-gm --o {subcortical_gm_mask_dce_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for subcortical GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for subcortical GM in DCE space failed.")
    else:
        print("Subcortical GM mask in DCE space already exists, skipping mri_binarize.")

    gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_mask_dce_path):
        gm_command = f"mri_binarize --i {aseg_in_dce_path} --gm --o {gm_mask_dce_path}"
        print(f"Running command: {gm_command}")
        result = subprocess.run(gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for GM in DCE space failed.")
    else:
        print("GM mask in DCE space already exists, skipping mri_binarize.")

    # Create gm_brainstem_mask in DCE space
    gm_brainstem_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_gm_brainstem.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_brainstem_mask_dce_path):
        gm_brainstem_command = f"mri_binarize --i {aseg_in_dce_path} --match 16 --o {gm_brainstem_mask_dce_path}"
        print(f"Running command: {gm_brainstem_command}")
        result = subprocess.run(gm_brainstem_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for brainstem GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for brainstem GM in DCE space failed.")
    else:
        print("Brainstem GM mask in DCE space already exists, skipping mri_binarize.")

    # Create gm_cerebellum_mask in DCE space
    gm_cerebellum_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_gm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_cerebellum_mask_dce_path):
        gm_cerebellum_command = f"mri_binarize --i {aseg_in_dce_path} --match 8 47 --o {gm_cerebellum_mask_dce_path}"
        print(f"Running command: {gm_cerebellum_command}")
        result = subprocess.run(gm_cerebellum_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for cerebellum GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for cerebellum GM in DCE space failed.")
    else:
        print("Cerebellum GM mask in DCE space already exists, skipping mri_binarize.")

    # Create wm_cerebellum_mask in DCE space
    wm_cerebellum_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_wm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cerebellum_mask_dce_path):
        wm_cerebellum_command = f"mri_binarize --i {aseg_in_dce_path} --match 7 46 --o {wm_cerebellum_mask_dce_path}"
        print(f"Running command: {wm_cerebellum_command}")
        result = subprocess.run(wm_cerebellum_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for cerebellum WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for cerebellum WM in DCE space failed.")
    else:
        print("Cerebellum WM mask in DCE space already exists, skipping mri_binarize.")

    # Create wm_cc_mask in DCE space
    wm_cc_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_wm_cc.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cc_mask_dce_path):
        wm_cc_command = f"mri_binarize --i {aseg_in_dce_path} --match 251 252 253 254 255 --o {wm_cc_mask_dce_path}"
        print(f"Running command: {wm_cc_command}")
        result = subprocess.run(wm_cc_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for corpus callosum WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for corpus callosum WM in DCE space failed.")
    else:
        print("Corpus Callosum WM mask in DCE space already exists, skipping mri_binarize.")

    # Create cortical gray matter mask by subtracting subcortical GM, brainstem, and cerebellum from total GM
    cortical_gm_mask_dce_path = aseg_in_dce_path.replace('.nii.gz', '_cortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(cortical_gm_mask_dce_path):
        cortical_gm_command = f"fslmaths {gm_mask_dce_path} -sub {subcortical_gm_mask_dce_path} -sub {gm_brainstem_mask_dce_path} -sub {gm_cerebellum_mask_dce_path} -thr 0.5 -bin {cortical_gm_mask_dce_path}"
        print(f"Running command: {cortical_gm_command}")
        result = subprocess.run(cortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for cortical GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for cortical GM in DCE space failed.")
    else:
        print("Cortical GM mask in DCE space already exists, skipping creation.")

    # Create subcortical GM mask by subtracting brainstem and cerebellum from subcortical GM
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_dce_path):
        subcortical_gm_command = f"fslmaths {subcortical_gm_mask_dce_path} -sub {gm_brainstem_mask_dce_path} -sub {gm_cerebellum_mask_dce_path} -thr 0.5 -bin {subcortical_gm_mask_dce_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for subcortical GM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for subcortical GM in DCE space failed.")
    else:
        print("Subcortical GM mask in DCE space already exists, skipping creation.")

    # Create WM mask by subtracting cerebellar WM and corpus callosum from total WM
    if force_recreate_masks or not os.path.exists(wm_mask_dce_path):
        wm_command = f"fslmaths {wm_mask_dce_path} -sub {wm_cerebellum_mask_dce_path} -sub {wm_cc_mask_dce_path} -thr 0.5 -bin {wm_mask_dce_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for WM in DCE space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for WM in DCE space failed.")
    else:
        print("WM mask in DCE space already exists, skipping creation.")

    # Similarly for T2 space
    wm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_wm.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_mask_t2_path):
        wm_command = f"mri_binarize --i {aseg_in_t2_path} --all-wm --o {wm_mask_t2_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for WM in T2 space failed.")
    else:
        print("WM mask in T2 space already exists, skipping mri_binarize for WM.")

    subcortical_gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_subcortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_t2_path):
        subcortical_gm_command = f"mri_binarize --i {aseg_in_t2_path} --subcort-gm --o {subcortical_gm_mask_t2_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for subcortical GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for subcortical GM in T2 space failed.")
    else:
        print("Subcortical GM mask in T2 space already exists, skipping mri_binarize.")

    gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_mask_t2_path):
        gm_command = f"mri_binarize --i {aseg_in_t2_path} --gm --o {gm_mask_t2_path}"
        print(f"Running command: {gm_command}")
        result = subprocess.run(gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for GM in T2 space failed.")
    else:
        print("GM mask in T2 space already exists, skipping mri_binarize.")

    # Create gm_brainstem_mask in T2 space
    gm_brainstem_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_gm_brainstem.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_brainstem_mask_t2_path):
        gm_brainstem_command = f"mri_binarize --i {aseg_in_t2_path} --match 16 --o {gm_brainstem_mask_t2_path}"
        print(f"Running command: {gm_brainstem_command}")
        result = subprocess.run(gm_brainstem_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for brainstem GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for brainstem GM in T2 space failed.")
    else:
        print("Brainstem GM mask in T2 space already exists, skipping mri_binarize.")

    # Create gm_cerebellum_mask in T2 space
    gm_cerebellum_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_gm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(gm_cerebellum_mask_t2_path):
        gm_cerebellum_command = f"mri_binarize --i {aseg_in_t2_path} --match 8 47 --o {gm_cerebellum_mask_t2_path}"
        print(f"Running command: {gm_cerebellum_command}")
        result = subprocess.run(gm_cerebellum_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for cerebellum GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for cerebellum GM in T2 space failed.")
    else:
        print("Cerebellum GM mask in T2 space already exists, skipping mri_binarize.")

    # Create wm_cerebellum_mask in T2 space
    wm_cerebellum_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_wm_cerebellum.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cerebellum_mask_t2_path):
        wm_cerebellum_command = f"mri_binarize --i {aseg_in_t2_path} --match 7 46 --o {wm_cerebellum_mask_t2_path}"
        print(f"Running command: {wm_cerebellum_command}")
        result = subprocess.run(wm_cerebellum_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for cerebellum WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for cerebellum WM in T2 space failed.")
    else:
        print("Cerebellum WM mask in T2 space already exists, skipping mri_binarize.")

    # Create wm_cc_mask in T2 space
    wm_cc_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_wm_cc.nii.gz')
    if force_recreate_masks or not os.path.exists(wm_cc_mask_t2_path):
        wm_cc_command = f"mri_binarize --i {aseg_in_t2_path} --match 251 252 253 254 255 --o {wm_cc_mask_t2_path}"
        print(f"Running command: {wm_cc_command}")
        result = subprocess.run(wm_cc_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"mri_binarize failed for corpus callosum WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("mri_binarize command for corpus callosum WM in T2 space failed.")
    else:
        print("Corpus Callosum WM mask in T2 space already exists, skipping mri_binarize.")

    # Create cortical gray matter mask by subtracting subcortical GM, brainstem, and cerebellum from total GM
    cortical_gm_mask_t2_path = aseg_in_t2_path.replace('.nii.gz', '_cortical_gm.nii.gz')
    if force_recreate_masks or not os.path.exists(cortical_gm_mask_t2_path):
        cortical_gm_command = f"fslmaths {gm_mask_t2_path} -sub {subcortical_gm_mask_t2_path} -sub {gm_brainstem_mask_t2_path} -sub {gm_cerebellum_mask_t2_path} -thr 0.5 -bin {cortical_gm_mask_t2_path}"
        print(f"Running command: {cortical_gm_command}")
        result = subprocess.run(cortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for cortical GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for cortical GM in T2 space failed.")
    else:
        print("Cortical GM mask in T2 space already exists, skipping creation.")

    # Create subcortical GM mask by subtracting brainstem and cerebellum from subcortical GM
    if force_recreate_masks or not os.path.exists(subcortical_gm_mask_t2_path):
        subcortical_gm_command = f"fslmaths {subcortical_gm_mask_t2_path} -sub {gm_brainstem_mask_t2_path} -sub {gm_cerebellum_mask_t2_path} -thr 0.5 -bin {subcortical_gm_mask_t2_path}"
        print(f"Running command: {subcortical_gm_command}")
        result = subprocess.run(subcortical_gm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for subcortical GM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for subcortical GM in T2 space failed.")
    else:
        print("Subcortical GM mask in T2 space already exists, skipping creation.")

    # Create WM mask by subtracting cerebellar WM and corpus callosum from total WM
    if force_recreate_masks or not os.path.exists(wm_mask_t2_path):
        wm_command = f"fslmaths {wm_mask_t2_path} -sub {wm_cerebellum_mask_t2_path} -sub {wm_cc_mask_t2_path} -thr 0.5 -bin {wm_mask_t2_path}"
        print(f"Running command: {wm_command}")
        result = subprocess.run(wm_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"fslmaths failed for WM in T2 space with error:\n{result.stderr}")
            raise RuntimeError("fslmaths command for WM in T2 space failed.")
    else:
        print("WM mask in T2 space already exists, skipping creation.")

    # Ensure all mask files exist before loading
    required_files = [
        wm_mask_dce_path, cortical_gm_mask_dce_path, subcortical_gm_mask_dce_path,
        gm_brainstem_mask_dce_path, gm_cerebellum_mask_dce_path, wm_cerebellum_mask_dce_path, wm_cc_mask_dce_path,
        wm_mask_t2_path, cortical_gm_mask_t2_path, subcortical_gm_mask_t2_path,
        gm_brainstem_mask_t2_path, gm_cerebellum_mask_t2_path, wm_cerebellum_mask_t2_path, wm_cc_mask_t2_path
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required mask file not found: {file_path}")

    # Load the masks
    wm_mask_dce = nib.load(wm_mask_dce_path).get_fdata().astype(bool)
    cortical_gm_mask_dce = nib.load(cortical_gm_mask_dce_path).get_fdata().astype(bool)
    subcortical_gm_mask_dce = nib.load(subcortical_gm_mask_dce_path).get_fdata().astype(bool)
    gm_brainstem_mask_dce = nib.load(gm_brainstem_mask_dce_path).get_fdata().astype(bool)
    gm_cerebellum_mask_dce = nib.load(gm_cerebellum_mask_dce_path).get_fdata().astype(bool)
    wm_cerebellum_mask_dce = nib.load(wm_cerebellum_mask_dce_path).get_fdata().astype(bool)
    wm_cc_mask_dce = nib.load(wm_cc_mask_dce_path).get_fdata().astype(bool)

    wm_mask_t2 = nib.load(wm_mask_t2_path).get_fdata().astype(bool)
    cortical_gm_mask_t2 = nib.load(cortical_gm_mask_t2_path).get_fdata().astype(bool)
    subcortical_gm_mask_t2 = nib.load(subcortical_gm_mask_t2_path).get_fdata().astype(bool)
    gm_brainstem_mask_t2 = nib.load(gm_brainstem_mask_t2_path).get_fdata().astype(bool)
    gm_cerebellum_mask_t2 = nib.load(gm_cerebellum_mask_t2_path).get_fdata().astype(bool)
    wm_cerebellum_mask_t2 = nib.load(wm_cerebellum_mask_t2_path).get_fdata().astype(bool)
    wm_cc_mask_t2 = nib.load(wm_cc_mask_t2_path).get_fdata().astype(bool)

    return (wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce,
            subcortical_gm_mask_t2, subcortical_gm_mask_dce, gm_brainstem_mask_t2,
            gm_brainstem_mask_dce, gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
            wm_cerebellum_mask_t2, wm_cerebellum_mask_dce, wm_cc_mask_t2, wm_cc_mask_dce)


def patlak_analysis_plotting(c_tissue, c_input, time):
    """
    Patlak fit that *ignores* any x or y that is NaN.
    All maths identical otherwise.
    Returns: Ki, lam, SD_Ki, x_full, y_full, included_mask
    where included_mask is True for points used in the fit.
    """
    if len(time) < 2:
        return (np.nan,)*3 + (np.array([]),)*3

    _single_bolus = int(getattr(settings, "NUMBER_OF_PEAKS", 2)) == 1

    delta_t = np.diff(time)

    # ── Single-bolus: AIF threshold masking ──
    bad = np.zeros(len(time), dtype=bool)
    if _single_bolus:
        ca = np.asarray(c_input, dtype=float)
        ca_peak = float(np.nanmax(ca))
        if ca_peak > 0:
            bad |= ca < 0.05 * ca_peak
        peak_idx = int(np.argmax(ca))
        t_start = time[peak_idx] + 60.0
        bad |= np.asarray(time, dtype=float) < t_start

    y = c_tissue / c_input
    x = np.concatenate(([0], np.cumsum(c_input[:-1]*delta_t))) / c_input
    good = (~np.isnan(x)) & (~np.isnan(y)) & (c_input != 0) & (~bad)

    if not good.any():
        return (np.nan,)*3 + (x, y, good)

    x_max = np.nanmax(x[good]) if good.any() else np.nan

    if _single_bolus:
        # Time-based windowing already applied via bad mask.
        w = np.ones(len(time), dtype=bool)
    else:
        # 1/3–2/3 Patlak window
        window_start = settings.PATLAK_WINDOW_START_FRACTION
        if not (0 < window_start < 1):
            window_start = 1/3
        w = (x >= window_start * x_max) & (x <= x_max)
    good &= w

    if good.sum() < 2:
        return (np.nan,)*3 + (x, y, good)

    xm, ym = x[good].mean(), y[good].mean()
    Ki_raw = ((x[good]-xm)*(y[good]-ym)).sum() / ((x[good]-xm)**2).sum()
    lam_raw = ym - Ki_raw*xm

    resid = y[good] - (lam_raw + Ki_raw*x[good])
    SD_raw = np.sqrt((resid**2).sum() / ((x[good]-xm)**2).sum() / (good.sum()-2))

    return Ki_raw*6000, lam_raw*100, SD_raw*6000, x, y, good



def find_baseline_point_advanced(y_data, fs=15, cutoff=4.0, order=3, radius=10):
    """
    Finds the baseline point in the given 1D array of y-values based on advanced filtering and gradient analysis.
    """
    # Ignore the first point
    y_data = y_data[1:]
    
    # Apply the low-pass filter
    y_filtered = butter_lowpass_filter(y_data, cutoff, fs, order)
    
    # Compute the gradient of the filtered data
    gradient_filtered = np.gradient(y_filtered)
    
    # Find the major peaks in the gradient
    major_peaks_gradient = find_major_peaks(gradient_filtered, radius)
    
    # Find the baseline points as the points right before the major peaks in the gradient
    baseline_points_gradient = [peak - 1 for peak in major_peaks_gradient]
    
    # Select the baseline point with the smaller index
    baseline_point = min(baseline_points_gradient) if baseline_points_gradient else None
    
    # Adjust the index due to ignoring the first point
    if baseline_point is not None:
        baseline_point += 1
    
    return baseline_point

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.transform import resize

def plot_ctcs_and_patlak(
    t2_img_slice, dce_img_slice,
    wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
    wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
    avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
    time_points, C_a,
    x_patlak_wm, y_patlak_wm, Ki_wm, lambda_wm,
    x_patlak_cortical_gm, y_patlak_cortical_gm, Ki_cortical_gm, lambda_cortical_gm,
    x_patlak_subcortical_gm, y_patlak_subcortical_gm, Ki_subcortical_gm, lambda_subcortical_gm,
    slice_idx, save_path=None, boundary_mask=None, boundary_ctc=None,
    x_patlak_boundary=None, y_patlak_boundary=None, Ki_boundary=None, lambda_boundary=None,
    included_wm=None, included_cortical_gm=None, included_subcortical_gm=None, included_boundary=None,
    gm_brainstem_ctc=None, x_patlak_gm_brainstem=None, y_patlak_gm_brainstem=None, Ki_gm_brainstem=None, lambda_gm_brainstem=None, included_gm_brainstem=None,
    gm_cerebellum_ctc=None, x_patlak_gm_cerebellum=None, y_patlak_gm_cerebellum=None, Ki_gm_cerebellum=None, lambda_gm_cerebellum=None, included_gm_cerebellum=None,
    wm_cerebellum_ctc=None, x_patlak_wm_cerebellum=None, y_patlak_wm_cerebellum=None, Ki_wm_cerebellum=None, lambda_wm_cerebellum=None, included_wm_cerebellum=None,
    wm_cc_ctc=None, x_patlak_wm_cc=None, y_patlak_wm_cc=None, Ki_wm_cc=None, lambda_wm_cc=None, included_wm_cc=None,
    gm_brainstem_mask_t2=None, gm_brainstem_mask_dce=None,
    gm_cerebellum_mask_t2=None, gm_cerebellum_mask_dce=None,
    wm_cerebellum_mask_t2=None, wm_cerebellum_mask_dce=None,
    wm_cc_mask_t2=None, wm_cc_mask_dce=None,
    bad_wm=None, bad_cortical_gm=None, bad_subcortical_gm=None,
    bad_gm_brainstem=None, bad_gm_cerebellum=None, bad_wm_cerebellum=None,
    bad_wm_cc=None, bad_boundary=None,
    model_fits=None
):
    """
    Re-written version that draws **one** grey band for each continuous
    union-segment of bad samples.  Black ‘×’ on Patlak removed.
    All args identical to the original signature.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from skimage.transform import resize

    # --------------- helper(s) -----------------
    def resize_and_binarize(mask, target_shape):
        from skimage.transform import resize
        m = resize(mask.astype(float), target_shape, order=0,
                   preserve_range=True, anti_aliasing=False)
        return (m > 0.5).astype(float)

    def overlay_mask(ax, mask, rgba):
        if mask.any():
            rgba_img = np.zeros((*mask.shape, 4))
            rgba_img[..., :3] = rgba[:3]
            rgba_img[..., 3] = rgba[3] * mask
            ax.imshow(np.rot90(rgba_img), interpolation='none')

    # --------------- figure set-up --------------
    fig  = plt.figure(figsize=(14, 22))
    gs   = GridSpec(4, 2, figure=fig,
                    height_ratios=[1, 1, 1, 1], width_ratios=[1, 1])
    gs.update(hspace=0.4)

    ax_t2  = fig.add_subplot(gs[0, 0])
    ax_dce = fig.add_subplot(gs[0, 1])
    ax_aif = fig.add_subplot(gs[1, :])
    ax_ctc = fig.add_subplot(gs[2, :])
    ax_pat = fig.add_subplot(gs[3, :])

    # ---------------- T2 / DCE panels ----------------
    t2_vmin, t2_vmax   = np.percentile(t2_img_slice, (1, 99))
    dce_vmin, dce_vmax = np.percentile(dce_img_slice, (1, 99))
    t2_norm  = (np.clip(t2_img_slice,  t2_vmin,  t2_vmax)-t2_vmin )/(t2_vmax -t2_vmin)
    dce_norm = (np.clip(dce_img_slice, dce_vmin, dce_vmax)-dce_vmin)/(dce_vmax-dce_vmin)

    ax_t2.imshow(np.rot90(t2_norm),  cmap='gray', vmin=0, vmax=1)
    ax_dce.imshow(np.rot90(dce_norm), cmap='gray', vmin=0, vmax=1)

    # colour scheme
    col = dict(
        white_matter  =[0,0,1,0.5],
        cortical_gm   =[1,0,0,0.5],
        subcortical_gm=[0.5,0,0,0.5],
        gm_brainstem  =[1,0.5,0,0.5],
        gm_cerebellum =[1,1,0,0.5],
        wm_cerebellum =[0,1,1,0.5],
        wm_cc         =[1,0,1,0.5],
        boundary      =[0,1,0,0.5]
    )

    # --- resize masks for display
    wm_mask_t2_r          = resize_and_binarize(wm_mask_t2,          t2_img_slice.shape)
    cortical_gm_mask_t2_r = resize_and_binarize(cortical_gm_mask_t2, t2_img_slice.shape)
    subcortical_gm_mask_t2_r = resize_and_binarize(subcortical_gm_mask_t2, t2_img_slice.shape)
    wm_mask_dce_r         = resize_and_binarize(wm_mask_dce,         dce_img_slice.shape)
    cortical_gm_mask_dce_r   = resize_and_binarize(cortical_gm_mask_dce, dce_img_slice.shape)
    subcortical_gm_mask_dce_r = resize_and_binarize(subcortical_gm_mask_dce, dce_img_slice.shape)
    # optional masks
    gm_brainstem_mask_t2_r   = resize_and_binarize(gm_brainstem_mask_t2,   t2_img_slice.shape) if gm_brainstem_mask_t2 is not None else np.zeros_like(t2_img_slice)
    gm_brainstem_mask_dce_r  = resize_and_binarize(gm_brainstem_mask_dce,  dce_img_slice.shape) if gm_brainstem_mask_dce is not None else np.zeros_like(dce_img_slice)
    gm_cerebellum_mask_t2_r  = resize_and_binarize(gm_cerebellum_mask_t2,  t2_img_slice.shape) if gm_cerebellum_mask_t2 is not None else np.zeros_like(t2_img_slice)
    gm_cerebellum_mask_dce_r = resize_and_binarize(gm_cerebellum_mask_dce, dce_img_slice.shape) if gm_cerebellum_mask_dce is not None else np.zeros_like(dce_img_slice)
    wm_cerebellum_mask_t2_r  = resize_and_binarize(wm_cerebellum_mask_t2,  t2_img_slice.shape) if wm_cerebellum_mask_t2 is not None else np.zeros_like(t2_img_slice)
    wm_cerebellum_mask_dce_r = resize_and_binarize(wm_cerebellum_mask_dce, dce_img_slice.shape) if wm_cerebellum_mask_dce is not None else np.zeros_like(dce_img_slice)
    wm_cc_mask_t2_r          = resize_and_binarize(wm_cc_mask_t2,          t2_img_slice.shape) if wm_cc_mask_t2 is not None else np.zeros_like(t2_img_slice)
    wm_cc_mask_dce_r         = resize_and_binarize(wm_cc_mask_dce,         dce_img_slice.shape) if wm_cc_mask_dce is not None else np.zeros_like(dce_img_slice)
    boundary_t2_r  = resize_and_binarize(boundary_mask,  t2_img_slice.shape)  if boundary_mask  is not None else None
    boundary_dce_r = resize_and_binarize(boundary_mask,  dce_img_slice.shape) if boundary_mask  is not None else None

    # apply overlays
    overlay_pairs = [
        (wm_mask_t2_r,          col['white_matter']),
        (cortical_gm_mask_t2_r, col['cortical_gm']),
        (subcortical_gm_mask_t2_r, col['subcortical_gm']),
        (gm_brainstem_mask_t2_r, col['gm_brainstem']),
        (gm_cerebellum_mask_t2_r,col['gm_cerebellum']),
        (wm_cerebellum_mask_t2_r,col['wm_cerebellum']),
        (wm_cc_mask_t2_r,       col['wm_cc']),
    ]
    for m,c in overlay_pairs:
        overlay_mask(ax_t2, m, c)
    if boundary_t2_r is not None:
        overlay_mask(ax_t2, boundary_t2_r, col['boundary'])
    ax_t2.set_title(f'T2 Slice {slice_idx} with masks'); ax_t2.axis('off')

    overlay_pairs_dce = [
        (wm_mask_dce_r,          col['white_matter']),
        (cortical_gm_mask_dce_r, col['cortical_gm']),
        (subcortical_gm_mask_dce_r,col['subcortical_gm']),
        (gm_brainstem_mask_dce_r, col['gm_brainstem']),
        (gm_cerebellum_mask_dce_r,col['gm_cerebellum']),
        (wm_cerebellum_mask_dce_r,col['wm_cerebellum']),
        (wm_cc_mask_dce_r,       col['wm_cc']),
    ]
    for m,c in overlay_pairs_dce:
        overlay_mask(ax_dce, m, c)
    if boundary_dce_r is not None:
        overlay_mask(ax_dce, boundary_dce_r, col['boundary'])
    ax_dce.set_title(f'DCE Slice {slice_idx} with masks'); ax_dce.axis('off')

    # ---------------- Input function panel -----------------
    ax_aif.set_facecolor('#f7f7f7')
    if C_a is not None and np.asarray(C_a).size and time_points is not None and np.asarray(time_points).size:
        t_arr = np.asarray(time_points)
        C_a_arr = np.asarray(C_a)
        n = min(t_arr.size, C_a_arr.size)
        t_arr = t_arr[:n]
        C_a_arr = C_a_arr[:n]
        ax_aif.plot(t_arr, C_a_arr, color='purple', lw=2, label='Input function')
        ax_aif.set_title('Maximum time-shifted input function')
        ax_aif.set_xlabel('Time (s)')
        ax_aif.set_ylabel('C_a(t) (mmol)')
        ax_aif.grid(True)
        ax_aif.legend(loc='upper right')
    else:
        ax_aif.set_visible(False)

    # ---------------- CTC panel -----------------
    ax_ctc.set_facecolor('#f7f7f7')

    # helper to add curve, points & build bad-mask list
    bad_masks = []
    t_arr = np.asarray(time_points) if time_points is not None else None

    def add_curve(ctc, label, colour, bad_mask):
        if ctc is None or not ctc.size:
            return
        if t_arr is not None and t_arr.size >= ctc.size:
            x_vals = t_arr[:ctc.size]
        else:
            x_vals = np.arange(ctc.size)
        ax_ctc.plot(x_vals, ctc, label=label, color=colour)
        if bad_mask is not None and bad_mask.any():
            bad_mask = bad_mask[:ctc.size]
            if t_arr is not None and t_arr.size >= ctc.size:
                bad_times = t_arr[:ctc.size][bad_mask]
            else:
                bad_times = np.where(bad_mask)[0]
            ax_ctc.scatter(bad_times, ctc[bad_mask],
                           facecolors='none', edgecolors='black', s=50)
            bad_masks.append(bad_mask.copy())

    add_curve(avg_wm_ctc,              'Cortical WM',   'blue',      bad_wm)
    add_curve(avg_cortical_gm_ctc,     'Cortical GM',   'red',       bad_cortical_gm)
    add_curve(avg_subcortical_gm_ctc,  'Subcortical GM','darkred',   bad_subcortical_gm)
    add_curve(gm_brainstem_ctc,        'Brainstem',     'orange',    bad_gm_brainstem)
    add_curve(gm_cerebellum_ctc,       'Cerebellar GM', 'yellow',    bad_gm_cerebellum)
    add_curve(wm_cerebellum_ctc,       'Cerebellar WM', 'cyan',      bad_wm_cerebellum)
    add_curve(wm_cc_ctc,               'WM Corpus Callosum', 'magenta', bad_wm_cc)
    add_curve(boundary_ctc,            'Boundary',      'green',     bad_boundary)

    # ----------- compute & shade UNION of bad regions -----------
    if bad_masks:
        if t_arr is not None and t_arr.size:
            union_len = min(t_arr.size, max(m.size for m in bad_masks))
            unified = np.zeros(union_len, dtype=bool)
            for m in bad_masks:
                unified |= m[:union_len]

            idx = np.where(unified)[0]
            if idx.size:
                boundaries = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                for block in boundaries:
                    ax_ctc.axvspan(t_arr[block[0]], t_arr[block[-1]], color='grey', alpha=0.3)
        else:
            union_len = max(m.size for m in bad_masks)
            unified = np.zeros(union_len, dtype=bool)
            for m in bad_masks:
                pad = union_len - m.size
                if pad > 0:
                    m = np.pad(m, (0, pad), constant_values=False)
                unified |= m

            idx = np.where(unified)[0]
            if idx.size:
                boundaries = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                for block in boundaries:
                    ax_ctc.axvspan(block[0], block[-1], color='grey', alpha=0.3)

    ax_ctc.set_title('Concentration functions')
    ax_ctc.legend(loc='upper right')
    ax_ctc.grid(True)
    if t_arr is not None and t_arr.size:
        ax_ctc.set_xlabel('Time (s)')
    else:
        ax_ctc.set_xlabel('Frames')
    ax_ctc.set_ylabel('C_tissue(t) (mmol)')


    # ---------------- Patlak panel (bottom) ------------------
    ax_pat.set_facecolor('#f7f7f7')

    if settings.KINETIC_MODEL.lower() == 'two_compartment' and model_fits is not None:
        def add_fit(ctc, fit, colour, label):
            if ctc is None or fit is None:
                return
            ax_pat.plot(ctc, color=colour, label=f"{label} data")
            ax_pat.plot(fit, '--', color=colour, label=f"{label} fit")

        add_fit(avg_wm_ctc,             model_fits.get('wm'),            'blue',    'Cortical WM')
        add_fit(avg_cortical_gm_ctc,    model_fits.get('cortical_gm'),   'red',     'Cortical GM')
        add_fit(avg_subcortical_gm_ctc, model_fits.get('subcortical_gm'), 'darkred','Subcortical GM')
        add_fit(gm_brainstem_ctc,       model_fits.get('gm_brainstem'),  'orange',  'Brainstem')
        add_fit(gm_cerebellum_ctc,      model_fits.get('gm_cerebellum'), 'gold',    'Cerebellar GM')
        add_fit(wm_cerebellum_ctc,      model_fits.get('wm_cerebellum'), 'cyan',    'Cerebellar WM')
        add_fit(wm_cc_ctc,              model_fits.get('wm_cc'),         'magenta', 'WM CC')
        add_fit(boundary_ctc,           model_fits.get('boundary'),      'green',   'Boundary')

        ax_pat.set_title('Two-compartment fit')
        ax_pat.set_xlabel('Time (frames)')
        ax_pat.set_ylabel('Concentration (mM)')
        ax_pat.grid(True)
        ax_pat.legend(loc='upper right')
    else:
        included_y_values = []

        def add_patlak(xp, yp, inc_mask, Ki, lam, colour, label):
            if xp.size == 0 or np.isnan(Ki):
                return
            ax_pat.scatter(xp[inc_mask], yp[inc_mask],
                           color=colour, marker='o', s=25, label=label)
            included_y_values.extend(yp[inc_mask].tolist())
            excl = ~inc_mask & np.isfinite(xp) & np.isfinite(yp)
            ax_pat.scatter(xp[excl], yp[excl],
                           facecolors='none', edgecolors=colour, s=40)
            ax_pat.plot(xp, lam/100 + (Ki/6000)*xp,
                        color=colour, linestyle='--')

        add_patlak(x_patlak_wm,             y_patlak_wm,             included_wm,
                  Ki_wm,             lambda_wm,             'blue',    'Cortical WM')
        add_patlak(x_patlak_cortical_gm,    y_patlak_cortical_gm,    included_cortical_gm,    Ki_cortical_gm,    lambda_cortical_gm,    'red',     'Cortical GM')
        add_patlak(x_patlak_subcortical_gm, y_patlak_subcortical_gm, included_subcortical_gm, Ki_subcortical_gm, lambda_subcortical_gm, 'darkred', 'Subcortical GM')
        add_patlak(x_patlak_gm_brainstem,   y_patlak_gm_brainstem,   included_gm_brainstem,   Ki_gm_brainstem,   lambda_gm_brainstem,   'orange',  'Brainstem')
        add_patlak(x_patlak_gm_cerebellum,  y_patlak_gm_cerebellum,  included_gm_cerebellum,  Ki_gm_cerebellum,  lambda_gm_cerebellum,  'gold',    'Cerebellar GM')
        add_patlak(x_patlak_wm_cerebellum,  y_patlak_wm_cerebellum,  included_wm_cerebellum,  Ki_wm_cerebellum,  lambda_wm_cerebellum,  'cyan',    'Cerebellar WM')
        add_patlak(x_patlak_wm_cc,          y_patlak_wm_cc,          included_wm_cc,          Ki_wm_cc,          lambda_wm_cc,          'magenta', 'WM CC')
        add_patlak(x_patlak_boundary,       y_patlak_boundary,       included_boundary,       Ki_boundary,       lambda_boundary,       'green',   'Boundary')

        if included_y_values:
            ymin, ymax = min(included_y_values), max(included_y_values)
            ax_pat.set_ylim(ymin, ymax)

        ax_pat.set_title('Patlak fit')
        ax_pat.set_xlim(0, 800)
        ax_pat.set_xlabel('∫C_a dt / C_a')
        ax_pat.set_ylabel('C_t / C_a')
        ax_pat.grid(True)
        ax_pat.legend(loc='lower left')
    

    plt.suptitle(f"Slice {slice_idx}", y=0.98)
    fit_text = ""
    if not np.isnan(Ki_wm):
        fit_text += f"Cortical WM:    Ki = {Ki_wm:.5f} ml/100g/min, λ = {lambda_wm:.5f} ml/100g\n"
    if not np.isnan(Ki_cortical_gm):
        fit_text += f"Cortical GM:    Ki = {Ki_cortical_gm:.5f} ml/100g/min, λ = {lambda_cortical_gm:.5f} ml/100g\n"
    if not np.isnan(Ki_subcortical_gm):
        fit_text += f"Subcortical GM: Ki = {Ki_subcortical_gm:.5f} ml/100g/min, λ = {lambda_subcortical_gm:.5f} ml/100g\n"
    if gm_brainstem_ctc is not None and not np.isnan(Ki_gm_brainstem):
        fit_text += f"Brainstem:      Ki = {Ki_gm_brainstem:.5f} ml/100g/min, λ = {lambda_gm_brainstem:.5f} ml/100g\n"
    if gm_cerebellum_ctc is not None and not np.isnan(Ki_gm_cerebellum):
        fit_text += f"Cerebellar GM:  Ki = {Ki_gm_cerebellum:.5f} ml/100g/min, λ = {lambda_gm_cerebellum:.5f} ml/100g\n"
    if wm_cerebellum_ctc is not None and not np.isnan(Ki_wm_cerebellum):
        fit_text += f"Cerebellar WM:  Ki = {Ki_wm_cerebellum:.5f} ml/100g/min, λ = {lambda_wm_cerebellum:.5f} ml/100g\n"
    if boundary_ctc is not None and not np.isnan(Ki_boundary):
        fit_text += f"Boundary:       Ki = {Ki_boundary:.5f} ml/100g/min, λ = {lambda_boundary:.5f} ml/100g"

    ax_pat.text(0.5, -0.23, fit_text.strip(),
                transform=ax_pat.transAxes, fontsize=10,
                ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.75))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        plt.tight_layout()
    if save_path:
        # Ensure the destination directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



def compute_and_plot_ctcs_median(
    data_4d, t2_img,
    wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
    wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
    T1_matrix, M0_matrix, analysis_directory, time_points_s, image_directory,
    dce_path, ref_affine=None, ref_header=None, boundary=False, compute_per_voxel_Ki=False, compute_per_voxel_CBF=False,
    gm_brainstem_mask_t2=None, gm_brainstem_mask_dce=None,
    gm_cerebellum_mask_t2=None, gm_cerebellum_mask_dce=None,
    wm_cerebellum_mask_t2=None, wm_cerebellum_mask_dce=None,
    wm_cc_mask_t2=None, wm_cc_mask_dce=None,
    flip_angle_deg=None,
    voxelwise_only: bool = False,
    compute_per_voxel_ETofts: bool = False,
):
    """
    Computes median CTCs for different tissue types across slices, performs Patlak analysis,
    saves the results, and generates plots. Also computes the total median for the entire tissue volume.
    Optionally computes K_i and CBF per voxel and generates overlay images and NIfTI files.
    CBF values are scaled to millilitres per 100 grams of tissue per minute (ml/100g/min).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from scipy.ndimage import binary_dilation

    def _maybe_write_reference_voxelwise_compare() -> None:
        """Best-effort QA: compare external reference voxelwise outputs against p-brain maps.

        Enabled only when `P_BRAIN_COMPARE_REFERENCE=1`.

        Looks for a *_CBF_maps_*.mat in the subject root (parent of Analysis), or an
        explicit `P_BRAIN_VOXELWISE_COMPARE_REF_PATH`.

        Writes montage PNGs under:
            Images/Fit/<NAME>_ref_pbrain_diff.png
        """

        compare_env = (os.environ.get('P_BRAIN_COMPARE_REFERENCE') or '').strip().lower()
        if compare_env not in {'1', 'true', 'yes', 'on'}:
            return

        try:
            from scipy.io import loadmat
        except Exception:
            return

        # Resolve reference file.
        ref_path = (os.environ.get('P_BRAIN_VOXELWISE_COMPARE_REF_PATH') or '').strip()
        subject_dir = os.path.abspath(os.path.join(analysis_directory, os.pardir))
        if not ref_path:
            try:
                import glob

                # Prefer the method4+tikhonov file (user-provided naming).
                patterns = [
                    '*CBF_maps*method4*tik*.mat',
                    '*CBF_maps*.mat',
                ]
                matches = []
                for pat in patterns:
                    matches.extend(glob.glob(os.path.join(subject_dir, pat)))
                ref_path = matches[0] if matches else ''
            except Exception:
                ref_path = ''

        if not ref_path or not os.path.isfile(ref_path):
            return

        # Resolve p-brain outputs (written earlier in this function).
        # Check model subfolders first, then fallback to root analysis dir.
        def _find_nifti(name, *subdirs):
            for sd in subdirs:
                p = os.path.join(analysis_directory, sd, name) if sd else os.path.join(analysis_directory, name)
                if os.path.isfile(p):
                    return p
            return os.path.join(analysis_directory, name)

        pb_paths = {
            'CBF': _find_nifti('CBF_per_voxel_tikhonov.nii.gz', 'tikhonov', ''),
            'MTT': _find_nifti('mtt_map.nii.gz', 'tikhonov', ''),
            'CTH': _find_nifti('cth_map.nii.gz', 'tikhonov', ''),
            'Ki': _find_nifti('Ki_per_voxel.nii.gz', 'patlak', ''),
            'vp': _find_nifti('vp_per_voxel.nii.gz', 'patlak', ''),
        }

        # Reference variable -> p-brain key
        want = {
            'CBF': 'CBF',
            'MTT': 'MTT',
            'CBKi': 'Ki',
            # Closest match to vp in provided file naming.
            'CBV_p': 'vp',
        }

        # Load reference maps.
        try:
            md = loadmat(ref_path)
        except Exception:
            return

        def _to_float3(name: str):
            v = md.get(name)
            if v is None:
                return None
            a = np.asarray(v)
            if a.dtype.kind in {'U', 'S'}:
                return None
            if a.dtype == object:
                return None
            try:
                a = np.asarray(a, dtype=float)
            except Exception:
                return None
            a = np.squeeze(a)
            if a.ndim != 3:
                return None
            return a

        ref_maps = {k: _to_float3(k) for k in want.keys()}
        ref_maps = {k: v for k, v in ref_maps.items() if v is not None}
        if not ref_maps:
            return

        # Load p-brain maps.
        def _load_pb(path: str):
            if not os.path.isfile(path):
                return None
            try:
                a = np.asarray(nib.load(path).get_fdata(), dtype=float)
                a = np.squeeze(a)
                if a.ndim != 3:
                    return None
                return a
            except Exception:
                return None

        pb_maps = {name: _load_pb(path) for name, path in pb_paths.items()}
        pb_maps = {k: v for k, v in pb_maps.items() if v is not None}
        if not pb_maps:
            return

        # Display convention: rotate p-brain 90° CCW for QA.
        def _rot_ccw(vol3d: np.ndarray) -> np.ndarray:
            return np.stack([np.rot90(vol3d[:, :, i], 1) for i in range(vol3d.shape[2])], axis=2)

        def _robust_range(a, b):
            x = np.concatenate([np.ravel(np.asarray(a, dtype=float)), np.ravel(np.asarray(b, dtype=float))])
            x = x[np.isfinite(x)]
            x = x[x != 0]
            if x.size == 0:
                return 0.0, 1.0
            lo = float(np.percentile(x, 1))
            hi = float(np.percentile(x, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.min(x))
                hi = float(np.max(x))
            if hi <= lo:
                hi = lo + 1.0
            return lo, hi

        def _diff_range(d):
            x = np.ravel(np.asarray(d, dtype=float))
            x = x[np.isfinite(x)]
            if x.size == 0:
                return -1.0, 1.0
            m = float(np.percentile(np.abs(x), 99))
            if not np.isfinite(m) or m <= 0:
                m = float(np.max(np.abs(x)))
            if not np.isfinite(m) or m <= 0:
                m = 1.0
            return -m, m

        def _render(out_prefix: str, label: str, mat_vol: np.ndarray, pb_vol: np.ndarray):
            z = int(min(mat_vol.shape[2], pb_vol.shape[2]))
            if z <= 0:
                return
            mat_vol = mat_vol[:, :, :z]
            pb_vol = pb_vol[:, :, :z]

            vmin, vmax = _robust_range(mat_vol, pb_vol)
            diff = pb_vol - mat_vol
            dvmin, dvmax = _diff_range(diff)
            diff_cmap = plt.get_cmap('coolwarm', 13)

            base = f'{label} comparison ({os.path.basename(ref_path)})'

            for k in range(z):
                a = mat_vol[:, :, k]
                b = pb_vol[:, :, k]
                d = diff[:, :, k]

                fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), dpi=150)
                axs[0].imshow(a, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
                axs[1].imshow(b, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
                axs[2].imshow(d, cmap=diff_cmap, vmin=dvmin, vmax=dvmax, origin='lower')

                axs[0].set_title('Reference', fontsize=10)
                axs[1].set_title('p-brain', fontsize=10)
                axs[2].set_title('diff (p-brain - ref)', fontsize=10)

                for j in range(3):
                    h, w = a.shape
                    axs[j].set_xticks([0, int(w // 2), int(w - 1)])
                    axs[j].set_yticks([0, int(h // 2), int(h - 1)])
                    axs[j].tick_params(labelsize=6)

                fig.suptitle(f'{base} – slice {k+1}', fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.9])

                out_path = f'{out_prefix}_slice_{k+1:02d}.png'
                plt.savefig(out_path)
                plt.close(fig)

        out_dir = os.path.join(image_directory, 'Fit')
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            return

        for mat_name, pb_name in want.items():
            mat_vol = ref_maps.get(mat_name)
            pb_vol = pb_maps.get(pb_name)
            if mat_vol is None or pb_vol is None:
                continue
            try:
                pb_vol = _rot_ccw(pb_vol)
                out_prefix = os.path.join(out_dir, f'{mat_name}_ref_pbrain_diff')
                _render(out_prefix, mat_name, mat_vol, pb_vol)
            except Exception:
                continue

    # Configure CTC conversion model (validator parity: TurboFLASH only).
    ctc_model = (getattr(settings, "CTC_MODEL", "turboflash") or "turboflash").strip().lower()
    if ctc_model in {"advanced", "method4", "validated_method4"}:
        ctc_model = "turboflash"
    if ctc_model != "turboflash":
        raise ValueError(
            f"Unsupported CTC_MODEL={ctc_model!r}. Validator parity requires 'turboflash'."
        )

    tr_s = None
    nph = None
    ti_s = resolve_turboflash_ti_s(dce_path, default=0.12)
    td_ms = float(ti_s) * 1e3
    # Validated closed-form conversion does not require excitation TR or nph.

    compute_CTC_meta = functools.partial(
        compute_CTC,
        TD=td_ms,
        flip_angle_deg=flip_angle_deg,
        ctc_model=ctc_model,
        tr_s=tr_s,
        nph=nph,
    )

    # Prefer DCE volume depth for slice count (t2_img may be absent in voxelwise-only mode).
    try:
        n_slices = int(data_4d.shape[2])
    except Exception:
        n_slices = t2_img.shape[2]
    skip_bottom = settings.GLOBAL_KI_SKIP_BOTTOM
    skip_top = settings.GLOBAL_KI_SKIP_TOP

    # Load C_a once according to the configured input function source
    try:
        C_a_full, input_metadata = get_input_function_curve(analysis_directory)
    except (FileNotFoundError, ValueError) as exc:
        print(f'[!] No valid input function available — skipping PK modelling: {exc}')
        C_a_full = None
        input_metadata = {}

    if ref_affine is None or ref_header is None:
        ref_img = nib.load(dce_path)
        if ref_affine is None:
            ref_affine = ref_img.affine
        if ref_header is None:
            ref_header = ref_img.header.copy()

    # ------------------------------------------------------------------
    # Pre-PK output: voxelwise brain concentration map (segmentation-masked).
    # ------------------------------------------------------------------
    try:
        brain_mask_dce = None
        for m in (
            wm_mask_dce,
            cortical_gm_mask_dce,
            subcortical_gm_mask_dce,
            gm_brainstem_mask_dce,
            gm_cerebellum_mask_dce,
            wm_cerebellum_mask_dce,
            wm_cc_mask_dce,
        ):
            if m is None:
                continue
            mm = np.asarray(m).astype(bool)
            brain_mask_dce = mm if brain_mask_dce is None else (brain_mask_dce | mm)

        if brain_mask_dce is None or not np.any(brain_mask_dce):
            # Fallback when segmentation isn't available: treat finite non-zero
            # baseline signal as "brain".
            baseline_3d = np.nanmean(np.asarray(data_4d, dtype=float), axis=3)
            brain_mask_dce = np.isfinite(baseline_3d) & (baseline_3d != 0)

        export_brain_concentration_4d(
            data_4d=np.asarray(data_4d),
            T1_matrix=np.asarray(T1_matrix),
            M0_matrix=np.asarray(M0_matrix),
            brain_mask=brain_mask_dce,
            dce_path=dce_path,
            analysis_directory=analysis_directory,
            ref_affine=ref_affine,
            ref_header=ref_header,
            flip_angle_deg=flip_angle_deg,
        )
    except Exception as exc:
        logger.warning("Failed to export voxelwise brain concentration map: %s", exc)

    # ------------------------------------------------------------------
    # Voxelwise-only mode: skip segmentation/tissue masks entirely.
    # ------------------------------------------------------------------
    if voxelwise_only:
        # In voxelwise-only mode, boundary and tissue medians are not computed.
        boundary = False

        if C_a_full is None:
            # No valid AIF → cannot do PK fitting; concentration map already
            # exported above so just return.
            return

        # If caller forgot to request voxelwise outputs, default to Ki maps.
        if not compute_per_voxel_Ki and not compute_per_voxel_CBF:
            compute_per_voxel_Ki = True

        if compute_per_voxel_Ki:
            Ki_per_voxel = np.full(data_4d.shape[:3], np.nan)
            lambda_per_voxel = np.full(data_4d.shape[:3], np.nan)
            SD_per_voxel = np.full(data_4d.shape[:3], np.nan)
        if compute_per_voxel_CBF:
            CBF_per_voxel = np.full(data_4d.shape[:3], np.nan)
            MTT_per_voxel = np.full(data_4d.shape[:3], np.nan) if settings.WRITE_MTT else None
            CTH_per_voxel = np.full(data_4d.shape[:3], np.nan) if settings.WRITE_CTH else None
            offset_per_voxel = (
                np.full(data_4d.shape[:3], np.nan)
                if bool(getattr(settings, "MATLAB_OFFSET_CORRECTION", False))
                and bool(getattr(settings, "WRITE_OFFSET_MAP", True))
                else None
            )
        if compute_per_voxel_ETofts:
            Ktrans_per_voxel = np.full(data_4d.shape[:3], np.nan)
            ve_per_voxel = np.full(data_4d.shape[:3], np.nan)
            vp_etofts_per_voxel = np.full(data_4d.shape[:3], np.nan)
            kep_per_voxel = np.full(data_4d.shape[:3], np.nan)

        # Brain mask heuristic (DCE space): finite and non-zero mean signal.
        try:
            baseline_3d = np.nanmean(np.asarray(data_4d, dtype=float), axis=3)
        except Exception:
            baseline_3d = np.nanmean(data_4d, axis=3)
        brain_mask_full = np.isfinite(baseline_3d) & (baseline_3d != 0)

        # Ensure we have an analysis/image output directory.
        try:
            os.makedirs(analysis_directory, exist_ok=True)
        except Exception:
            pass

        # Local bindings.
        _ctc = compute_CTC_meta
        _baseline = find_baseline_point_advanced
        _shift = custom_shifter
        _patlak = patlak_analysis_plotting

        with auto_logging_suppressed():
            for i in _pbrain_tqdm(range(n_slices), desc="Processing slices (voxelwise-only)"):
                brain_mask_slice = brain_mask_full[:, :, i]
                brain_indices = np.argwhere(brain_mask_slice)
                if brain_indices.size == 0:
                    continue

                # Precompute common time-axis and (optional) cached deconvolution solver.
                common_len = min(int(data_4d.shape[3]), len(C_a_full), len(time_points_s))
                C_a_voxel_common = C_a_full[:common_len]
                time_points_voxel_common = time_points_s[:common_len]

                fast_tikh_solver = None
                fast_delta_t = None
                use_fast_tikh = (
                    compute_per_voxel_CBF
                    and (not settings.ALIGN_AIF_BY_XCORR)
                    and not (
                        settings.CTH_MTT_METHOD.lower() in {"gamma", "hybrid"}
                        and settings.CTH_MTT_GAMMA_VOXELWISE
                    )
                )
                if use_fast_tikh and common_len >= 2:
                    deltas = np.diff(time_points_voxel_common)
                    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
                    if deltas.size:
                        fast_delta_t = float(deltas[0])
                        try:
                            # Validator-parity slow solver: lambda grid is SVD-derived + L-curve curvature.
                            fast_tikh_solver = build_tikhonov_validated_slow_solver(
                                time_points_voxel_common,
                                C_a_voxel_common,
                                lambda_candidates=None,
                                offset_grouping_s=0.05,
                                f_win=50,
                                tissue_density=float(getattr(settings, "TISSUE_DENSITY", 1.04)),
                                hematocrit=float(getattr(settings, "HEMATOCRIT", 0.42)),
                                plasma_derived_aif=bool(getattr(settings, "PLASMA_DERIVED_AIF", False)),
                            )
                        except Exception:
                            fast_tikh_solver = None
                            fast_delta_t = None

                # Initialize per-voxel slice arrays.
                if compute_per_voxel_Ki:
                    Ki_slice = np.full(brain_mask_slice.shape, np.nan)
                    lam_slice = np.full(brain_mask_slice.shape, np.nan)
                    SD_slice = np.full(brain_mask_slice.shape, np.nan)
                if compute_per_voxel_CBF:
                    CBF_slice = np.full(brain_mask_slice.shape, np.nan)
                    MTT_slice = np.full(brain_mask_slice.shape, np.nan) if settings.WRITE_MTT else None
                    CTH_slice = np.full(brain_mask_slice.shape, np.nan) if settings.WRITE_CTH else None
                    offset_slice = (
                        np.full(brain_mask_slice.shape, np.nan)
                        if offset_per_voxel is not None
                        else None
                    )
                if compute_per_voxel_ETofts:
                    Ktrans_slice = np.full(brain_mask_slice.shape, np.nan)
                    ve_slice = np.full(brain_mask_slice.shape, np.nan)
                    vp_etofts_slice = np.full(brain_mask_slice.shape, np.nan)
                    kep_slice = np.full(brain_mask_slice.shape, np.nan)

                fast_coords = []
                fast_curves = []
                fast_offsets = []

                for (x, y) in brain_indices:
                    voxel_time_course = data_4d[x, y, i, :]
                    T1 = T1_matrix[x, y, i]
                    M0 = M0_matrix[x, y, i]
                    C_t_0 = _ctc(voxel_time_course, T1, m0=M0)
                    baseline_point = _baseline(C_t_0)
                    C_t = _shift(C_t_0, baseline_point)

                    if np.isnan(C_t).any() or np.all(C_t == 0):
                        continue

                    min_length_voxel = common_len
                    C_t_voxel = C_t[:min_length_voxel]
                    C_a_voxel = C_a_voxel_common
                    time_points_voxel = time_points_voxel_common
                    if min_length_voxel < 2:
                        continue

                    Ki_value = None
                    if compute_per_voxel_Ki:
                        Ki_voxel, lam_voxel, SD_voxel, _, _, _ = _patlak(
                            C_t_voxel, C_a_voxel, time_points_voxel
                        )
                        Ki_slice[x, y] = Ki_voxel
                        lam_slice[x, y] = lam_voxel
                        SD_slice[x, y] = SD_voxel
                        Ki_value = Ki_voxel

                    if compute_per_voxel_ETofts:
                        # Optionally initialise from Patlak Ki / vp estimates.
                        _init_kt = None
                        _init_vp = None
                        if Ki_value is not None and np.isfinite(Ki_value):
                            _init_kt = max(float(Ki_value) / 6000.0, 1e-8)
                        if compute_per_voxel_Ki and np.isfinite(lam_slice[x, y]):
                            _init_vp = np.clip(float(lam_slice[x, y]) / 100.0, 0.0, 0.99)
                        try:
                            ef = _etofts_fit_voxel(
                                C_a_voxel, C_t_voxel, time_points_voxel,
                                init_ktrans=_init_kt,
                                init_vp=_init_vp,
                            )
                            if ef.success:
                                Ktrans_slice[x, y] = ef.ktrans_ml_100g
                                ve_slice[x, y] = ef.ve
                                vp_etofts_slice[x, y] = ef.vp
                                kep_slice[x, y] = ef.kep_per_min
                        except Exception:
                            pass

                    if compute_per_voxel_CBF:
                        # Optional MATLAB-style arrival-time (offset) correction.
                        # Validator parity: apply offset by shifting the AIF inside the operator,
                        # not by time-warping the tissue curve.
                        offset_s = 0.0
                        if bool(getattr(settings, "MATLAB_OFFSET_CORRECTION", False)):
                            try:
                                offset_s, _ = estimate_bolus_arrival_shift_seconds(C_t_voxel, time_points_voxel)
                            except Exception:
                                offset_s = 0.0
                            if offset_slice is not None and np.isfinite(offset_s):
                                offset_slice[x, y] = float(offset_s)

                        if fast_tikh_solver is not None and fast_delta_t is not None:
                            fast_coords.append((int(x), int(y)))
                            fast_curves.append(np.asarray(C_t_voxel, dtype=float))
                            fast_offsets.append(float(offset_s) if np.isfinite(offset_s) else 0.0)
                            continue
                        cbf_voxel, mtt_voxel, cth_voxel, _ = compute_mtt_cth(
                            settings.CTH_MTT_METHOD,
                            C_t_voxel,
                            C_a_voxel,
                            time_points_voxel,
                            Ki=Ki_value,
                            allow_gamma=settings.CTH_MTT_GAMMA_VOXELWISE,
                            logger=logger,
                        )
                        if np.isfinite(cbf_voxel):
                            CBF_slice[x, y] = cbf_voxel
                        if settings.WRITE_MTT and MTT_slice is not None and np.isfinite(mtt_voxel):
                            MTT_slice[x, y] = mtt_voxel
                        if settings.WRITE_CTH and CTH_slice is not None and np.isfinite(cth_voxel):
                            CTH_slice[x, y] = cth_voxel

                # Batched fast Tikhonov path.
                if compute_per_voxel_CBF and fast_tikh_solver is not None and fast_delta_t is not None and fast_curves:
                    n_vox = len(fast_curves)
                    n_time = int(common_len)
                    chunk = int(getattr(settings, "TIKHONOV_BATCH_SIZE", 4096))

                    for start in range(0, n_vox, chunk):
                        end = min(start + chunk, n_vox)
                        Ct_mat = np.stack(fast_curves[start:end], axis=1)
                        if Ct_mat.shape[0] != n_time:
                            Ct_mat = Ct_mat[:n_time, :]

                        off_chunk = np.asarray(fast_offsets[start:end], dtype=float) if fast_offsets else None
                        sol = fast_tikh_solver(Ct_mat, offsets_s=off_chunk)
                        cbf_vals = np.asarray(sol.cbf_ml_per_100g_min, dtype=float).reshape(-1)
                        mtt_vals = np.asarray(sol.mtt_s, dtype=float).reshape(-1)
                        cth_vals = np.asarray(sol.cth_s, dtype=float).reshape(-1)

                        for k in range(end - start):
                            x, y = fast_coords[start + k]
                            cbf_k = float(cbf_vals[k]) if k < cbf_vals.size else float("nan")
                            if np.isfinite(cbf_k):
                                CBF_slice[x, y] = cbf_k
                            if settings.WRITE_MTT and MTT_slice is not None:
                                mtt_k = float(mtt_vals[k]) if k < mtt_vals.size else float("nan")
                                if np.isfinite(mtt_k):
                                    MTT_slice[x, y] = mtt_k
                            if settings.WRITE_CTH and CTH_slice is not None:
                                cth_k = float(cth_vals[k]) if k < cth_vals.size else float("nan")
                                if np.isfinite(cth_k):
                                    CTH_slice[x, y] = cth_k

                if compute_per_voxel_Ki:
                    Ki_per_voxel[:, :, i] = Ki_slice
                    lambda_per_voxel[:, :, i] = lam_slice
                    SD_per_voxel[:, :, i] = SD_slice
                if compute_per_voxel_CBF:
                    CBF_per_voxel[:, :, i] = CBF_slice
                    if settings.WRITE_MTT and MTT_per_voxel is not None and MTT_slice is not None:
                        MTT_per_voxel[:, :, i] = MTT_slice
                    if settings.WRITE_CTH and CTH_per_voxel is not None and CTH_slice is not None:
                        CTH_per_voxel[:, :, i] = CTH_slice
                    if offset_per_voxel is not None and offset_slice is not None:
                        offset_per_voxel[:, :, i] = offset_slice
                if compute_per_voxel_ETofts:
                    Ktrans_per_voxel[:, :, i] = Ktrans_slice
                    ve_per_voxel[:, :, i] = ve_slice
                    vp_etofts_per_voxel[:, :, i] = vp_etofts_slice
                    kep_per_voxel[:, :, i] = kep_slice

        affine = ref_affine

        # ── Model-specific output subdirectories ──────────────────
        patlak_analysis_dir = os.path.join(analysis_directory, 'patlak')
        tikhonov_analysis_dir = os.path.join(analysis_directory, 'tikhonov')
        etofts_analysis_dir = os.path.join(analysis_directory, 'etofts')
        patlak_image_dir = os.path.join(image_directory, 'AI', 'patlak')
        tikhonov_image_dir = os.path.join(image_directory, 'AI', 'tikhonov')
        etofts_image_dir = os.path.join(image_directory, 'AI', 'etofts')

        if compute_per_voxel_Ki:
            os.makedirs(patlak_analysis_dir, exist_ok=True)
            Ki_per_voxel_nii = nib.Nifti1Image(np.asarray(Ki_per_voxel, dtype=np.float32), affine=affine, header=ref_header.copy())
            Ki_per_voxel_path = os.path.join(patlak_analysis_dir, 'Ki_per_voxel.nii.gz')
            nib.save(Ki_per_voxel_nii, Ki_per_voxel_path)
            print(f"K_i per voxel saved to {Ki_per_voxel_path}")

            vp_data = np.asarray(lambda_per_voxel, dtype=np.float32)
            vp_data = np.where(np.isfinite(vp_data), np.maximum(vp_data, 0.0), vp_data).astype(np.float32)
            vp_per_voxel_nii = nib.Nifti1Image(vp_data, affine=affine, header=ref_header.copy())
            vp_per_voxel_path = os.path.join(patlak_analysis_dir, 'vp_per_voxel.nii.gz')
            nib.save(vp_per_voxel_nii, vp_per_voxel_path)
            print(f"v_p per voxel saved to {vp_per_voxel_path}")

            # Optional overlays
            try:
                global_Ki_min = np.nanmin(Ki_per_voxel)
                global_Ki_max = np.nanmax(Ki_per_voxel)
                for i in range(n_slices):
                    Ki_slice = Ki_per_voxel[:, :, i]
                    if np.isnan(Ki_slice).all():
                        continue
                    save_dir_overlay = os.path.join(patlak_image_dir, 'Ki Overlays')
                    os.makedirs(save_dir_overlay, exist_ok=True)
                    t_idx = min(20, int(data_4d.shape[3]) - 1)
                    save_path_overlay = os.path.join(save_dir_overlay, f"Ki_overlay_slice_{i+1}.png")
                    plot_Ki_overlay(
                        data_4d[:, :, i, t_idx], Ki_slice, slice_idx=i+1, save_path=save_path_overlay,
                        vmin=global_Ki_min, vmax=global_Ki_max
                    )
            except Exception:
                pass

        if compute_per_voxel_CBF:
            os.makedirs(tikhonov_analysis_dir, exist_ok=True)
            CBF_per_voxel_nii = nib.Nifti1Image(np.asarray(CBF_per_voxel, dtype=np.float32),
                                                affine=affine,
                                                header=ref_header.copy())
            CBF_per_voxel_path = os.path.join(tikhonov_analysis_dir, 'CBF_per_voxel_tikhonov.nii.gz')
            nib.save(CBF_per_voxel_nii, CBF_per_voxel_path)
            print(f"CBF per voxel saved to {CBF_per_voxel_path}")

            if settings.WRITE_MTT and MTT_per_voxel is not None:
                mtt_img = nib.Nifti1Image(np.asarray(MTT_per_voxel, dtype=np.float32),
                                          affine=affine,
                                          header=ref_header.copy())
                mtt_img = annotate_cth_mtt_header(mtt_img)
                mtt_path = os.path.join(tikhonov_analysis_dir, 'mtt_map.nii.gz')
                nib.save(mtt_img, mtt_path)
                print(f"MTT map saved to {mtt_path} (method={settings.CTH_MTT_METHOD})")

            if settings.WRITE_CTH and CTH_per_voxel is not None:
                cth_img = nib.Nifti1Image(np.asarray(CTH_per_voxel, dtype=np.float32),
                                          affine=affine,
                                          header=ref_header.copy())
                cth_img = annotate_cth_mtt_header(cth_img)
                cth_path = os.path.join(tikhonov_analysis_dir, 'cth_map.nii.gz')
                nib.save(cth_img, cth_path)
                print(f"CTH map saved to {cth_path} (method={settings.CTH_MTT_METHOD})")

            # CBV = CBF * MTT / 60  (ml/100g)
            if (
                settings.WRITE_MTT
                and MTT_per_voxel is not None
                and CBF_per_voxel is not None
            ):
                cbv_data = np.asarray(CBF_per_voxel, dtype=np.float32) * np.asarray(MTT_per_voxel, dtype=np.float32) / 60.0
                cbv_img = nib.Nifti1Image(cbv_data, affine=affine, header=ref_header.copy())
                cbv_path = os.path.join(tikhonov_analysis_dir, 'cbv_map.nii.gz')
                nib.save(cbv_img, cbv_path)
                print(f"CBV map saved to {cbv_path}")

            if offset_per_voxel is not None:
                offset_img = nib.Nifti1Image(
                    np.asarray(offset_per_voxel, dtype=np.float32),
                    affine=affine,
                    header=ref_header.copy(),
                )
                offset_path = os.path.join(tikhonov_analysis_dir, 'offset_map.nii.gz')
                nib.save(offset_img, offset_path)
                print(f"Offset map saved to {offset_path}")

            # Optional overlays
            try:
                global_CBF_min = np.nanmin(CBF_per_voxel)
                global_CBF_max = np.nanmax(CBF_per_voxel)
                for i in range(n_slices):
                    CBF_slice = CBF_per_voxel[:, :, i]
                    if np.isnan(CBF_slice).all():
                        continue
                    save_dir_overlay = os.path.join(tikhonov_image_dir, 'CBF Overlays')
                    os.makedirs(save_dir_overlay, exist_ok=True)
                    t_idx = min(20, int(data_4d.shape[3]) - 1)
                    save_path_overlay = os.path.join(save_dir_overlay, f"CBF_overlay_slice_{i+1}.png")
                    plot_CBF_overlay(
                        data_4d[:, :, i, t_idx], CBF_slice, slice_idx=i+1, save_path=save_path_overlay,
                        vmin=global_CBF_min, vmax=global_CBF_max
                    )
            except Exception:
                pass

        # ── Extended Tofts NIfTI outputs ────────────────────────────
        if compute_per_voxel_ETofts:
            os.makedirs(etofts_analysis_dir, exist_ok=True)
            for _name, _arr in [
                ('Ktrans_per_voxel',  Ktrans_per_voxel),
                ('ve_per_voxel',      ve_per_voxel),
                ('vp_etofts_per_voxel', vp_etofts_per_voxel),
                ('kep_per_voxel',     kep_per_voxel),
            ]:
                _nii = nib.Nifti1Image(np.asarray(_arr, dtype=np.float32),
                                       affine=affine, header=ref_header.copy())
                _p = os.path.join(etofts_analysis_dir, f'{_name}.nii.gz')
                nib.save(_nii, _p)
                print(f"{_name} saved to {_p}")

            # Ktrans overlays (same style as Ki overlays)
            try:
                global_Kt_min = np.nanmin(Ktrans_per_voxel)
                global_Kt_max = np.nanmax(Ktrans_per_voxel)
                for i in range(n_slices):
                    Kt_sl = Ktrans_per_voxel[:, :, i]
                    if np.isnan(Kt_sl).all():
                        continue
                    save_dir_overlay = os.path.join(etofts_image_dir, 'Ktrans Overlays')
                    os.makedirs(save_dir_overlay, exist_ok=True)
                    t_idx = min(20, int(data_4d.shape[3]) - 1)
                    save_path_overlay = os.path.join(save_dir_overlay, f"Ktrans_overlay_slice_{i+1}.png")
                    plot_Ki_overlay(
                        data_4d[:, :, i, t_idx], Kt_sl, slice_idx=i+1, save_path=save_path_overlay,
                        vmin=global_Kt_min, vmax=global_Kt_max
                    )
            except Exception:
                pass

        # Optional reference comparison of voxelwise maps (best-effort, opt-in).
        try:
            _maybe_write_reference_voxelwise_compare()
        except Exception:
            pass

        # ── PK diagnostic grid (voxelwise-only path) ──────────────
        try:
            _generate_pk_diagnostic_grid(
                CBF_per_voxel=CBF_per_voxel if compute_per_voxel_CBF else None,
                MTT_per_voxel=MTT_per_voxel if compute_per_voxel_CBF else None,
                CTH_per_voxel=CTH_per_voxel if compute_per_voxel_CBF else None,
                Ki_per_voxel=Ki_per_voxel if compute_per_voxel_Ki else None,
                C_a_full=C_a_full,
                time_points_s=time_points_s,
                data_4d=data_4d,
                T1_matrix=T1_matrix,
                M0_matrix=M0_matrix,
                brain_mask_full=brain_mask_full,
                analysis_directory=tikhonov_analysis_dir if compute_per_voxel_CBF else analysis_directory,
                compute_CTC_func=_ctc,
                baseline_func=_baseline,
                shift_func=_shift,
            )
        except Exception as _pk_err:
            print(f"[pk_diag] Failed to generate PK diagnostic grid: {_pk_err}")

        # ── Extended Tofts diagnostic grid (voxelwise-only path) ──
        if compute_per_voxel_ETofts:
            try:
                _generate_etofts_diagnostic_grid(
                    Ktrans_per_voxel=Ktrans_per_voxel,
                    ve_per_voxel=ve_per_voxel,
                    vp_etofts_per_voxel=vp_etofts_per_voxel,
                    kep_per_voxel=kep_per_voxel,
                    Ki_per_voxel=Ki_per_voxel if compute_per_voxel_Ki else None,
                    C_a_full=C_a_full,
                    time_points_s=time_points_s,
                    data_4d=data_4d,
                    T1_matrix=T1_matrix,
                    M0_matrix=M0_matrix,
                    brain_mask_full=brain_mask_full,
                    analysis_directory=etofts_analysis_dir,
                    compute_CTC_func=_ctc,
                    baseline_func=_baseline,
                    shift_func=_shift,
                )
            except Exception as _et_err:
                print(f"[etofts_diag] Failed to generate ETofts diagnostic grid: {_et_err}")

        # ── Global average Patlak plot (voxelwise-only) ───────────
        try:
            _generate_global_average_patlak_plot(
                data_4d=data_4d,
                T1_matrix=T1_matrix,
                M0_matrix=M0_matrix,
                brain_mask_full=brain_mask_full,
                C_a_full=C_a_full,
                time_points_s=time_points_s,
                analysis_directory=patlak_analysis_dir if compute_per_voxel_Ki else analysis_directory,
                compute_CTC_func=_ctc,
                baseline_func=_baseline,
                shift_func=_shift,
            )
        except Exception as _gp_err:
            print(f"[global_patlak] Failed to generate global average Patlak plot: {_gp_err}")

        # Nothing else to do in voxelwise-only mode.
        return

    all_patlak_data = []
    Ki_wm_list = []
    Ki_cortical_gm_list = []
    Ki_subcortical_gm_list = []
    Ki_gm_brainstem_list = []
    Ki_gm_cerebellum_list = []
    Ki_wm_cerebellum_list = []
    Ki_wm_cc_list = []
    Ki_boundary_list = []

    # Lists to collect T1 and M0 values for each tissue across slices
    T1_wm_vals,           M0_wm_vals           = [], []
    T1_cortical_gm_vals,  M0_cortical_gm_vals  = [], []
    T1_subcortical_gm_vals, M0_subcortical_gm_vals = [], []
    T1_gm_brainstem_vals, M0_gm_brainstem_vals = [], []
    T1_gm_cerebellum_vals, M0_gm_cerebellum_vals = [], []
    T1_wm_cerebellum_vals, M0_wm_cerebellum_vals = [], []
    T1_wm_cc_vals,        M0_wm_cc_vals        = [], []
    if boundary:
        T1_boundary_vals, M0_boundary_vals = [], []

    # Initialize lists to collect all valid CTCs across slices
    wm_ctcs_total = []
    cortical_gm_ctcs_total = []
    subcortical_gm_ctcs_total = []
    gm_brainstem_ctcs_total = []
    gm_cerebellum_ctcs_total = []
    wm_cerebellum_ctcs_total = []
    wm_cc_ctcs_total = []
    boundary_ctcs_total = []

    

    # Initialize empty 3D arrays to store K_i values per voxel
    Ki_wm_image = np.full(data_4d.shape[:3], np.nan)
    Ki_cortical_gm_image = np.full(data_4d.shape[:3], np.nan)
    Ki_subcortical_gm_image = np.full(data_4d.shape[:3], np.nan)
    Ki_gm_brainstem_image = np.full(data_4d.shape[:3], np.nan)
    Ki_gm_cerebellum_image = np.full(data_4d.shape[:3], np.nan)
    Ki_wm_cerebellum_image = np.full(data_4d.shape[:3], np.nan)
    Ki_wm_cc_image = np.full(data_4d.shape[:3], np.nan)
    if boundary:
        Ki_boundary_image = np.full(data_4d.shape[:3], np.nan)

    # Initialize per-voxel Ki-related arrays if needed
    if compute_per_voxel_Ki:
        Ki_per_voxel = np.full(data_4d.shape[:3], np.nan)
        lambda_per_voxel = np.full(data_4d.shape[:3], np.nan)
        SD_per_voxel = np.full(data_4d.shape[:3], np.nan)
    if compute_per_voxel_CBF:
        CBF_per_voxel = np.full(data_4d.shape[:3], np.nan)
        MTT_per_voxel = np.full(data_4d.shape[:3], np.nan) if settings.WRITE_MTT else None
        CTH_per_voxel = np.full(data_4d.shape[:3], np.nan) if settings.WRITE_CTH else None

    gm_mask_dce_full = np.logical_or.reduce(
        (
            cortical_gm_mask_dce,
            subcortical_gm_mask_dce,
            gm_brainstem_mask_dce,
            gm_cerebellum_mask_dce,
        )
    )

    if boundary:
        boundary_mask_full = np.zeros(data_4d.shape[:3], dtype=bool)

    # Add tqdm progress bar to the loop
    with auto_logging_suppressed():
        for i in _pbrain_tqdm(range(n_slices), desc="Processing slices"):
            # Extract relevant masks for the current slice
            wm_slice_t2 = wm_mask_t2[:, :, i]
            cortical_gm_slice_t2 = cortical_gm_mask_t2[:, :, i]
            subcortical_gm_slice_t2 = subcortical_gm_mask_t2[:, :, i]
            gm_brainstem_slice_t2 = gm_brainstem_mask_t2[:, :, i]
            gm_cerebellum_slice_t2 = gm_cerebellum_mask_t2[:, :, i]
            wm_cerebellum_slice_t2 = wm_cerebellum_mask_t2[:, :, i]
            wm_cc_slice_t2 = wm_cc_mask_t2[:, :, i]

            wm_slice_dce = wm_mask_dce[:, :, i]
            cortical_gm_slice_dce = cortical_gm_mask_dce[:, :, i]
            subcortical_gm_slice_dce = subcortical_gm_mask_dce[:, :, i]
            gm_brainstem_slice_dce = gm_brainstem_mask_dce[:, :, i]
            gm_cerebellum_slice_dce = gm_cerebellum_mask_dce[:, :, i]
            wm_cerebellum_slice_dce = wm_cerebellum_mask_dce[:, :, i]
            wm_cc_slice_dce = wm_cc_mask_dce[:, :, i]

            # Combine cortical and subcortical GM masks for boundary calculation
            gm_slice_dce = np.logical_or(cortical_gm_slice_dce, subcortical_gm_slice_dce)

            # Compute the boundary mask if required
            if boundary:
                wm_dilated = binary_dilation(wm_slice_dce, iterations=1)
                gm_dilated = binary_dilation(gm_slice_dce, iterations=1)
                boundary_mask = np.logical_and(wm_dilated, gm_dilated)
                boundary_indices = np.argwhere(boundary_mask)
            else:
                boundary_mask = None
                boundary_indices = []

            # Find voxel indices for each tissue type in the slice
            wm_indices = np.argwhere(wm_slice_dce)
            cortical_gm_indices = np.argwhere(cortical_gm_slice_dce)
            subcortical_gm_indices = np.argwhere(subcortical_gm_slice_dce)
            gm_brainstem_indices = np.argwhere(gm_brainstem_slice_dce)
            gm_cerebellum_indices = np.argwhere(gm_cerebellum_slice_dce)
            wm_cerebellum_indices = np.argwhere(wm_cerebellum_slice_dce)
            wm_cc_indices = np.argwhere(wm_cc_slice_dce)

            # Collect T1 and M0 values for each tissue type
            T1_wm_vals.extend(T1_matrix[:, :, i][wm_slice_dce].ravel())
            M0_wm_vals.extend(M0_matrix[:, :, i][wm_slice_dce].ravel())
            T1_cortical_gm_vals.extend(T1_matrix[:, :, i][cortical_gm_slice_dce].ravel())
            M0_cortical_gm_vals.extend(M0_matrix[:, :, i][cortical_gm_slice_dce].ravel())
            T1_subcortical_gm_vals.extend(T1_matrix[:, :, i][subcortical_gm_slice_dce].ravel())
            M0_subcortical_gm_vals.extend(M0_matrix[:, :, i][subcortical_gm_slice_dce].ravel())
            T1_gm_brainstem_vals.extend(T1_matrix[:, :, i][gm_brainstem_slice_dce].ravel())
            M0_gm_brainstem_vals.extend(M0_matrix[:, :, i][gm_brainstem_slice_dce].ravel())
            T1_gm_cerebellum_vals.extend(T1_matrix[:, :, i][gm_cerebellum_slice_dce].ravel())
            M0_gm_cerebellum_vals.extend(M0_matrix[:, :, i][gm_cerebellum_slice_dce].ravel())
            T1_wm_cerebellum_vals.extend(T1_matrix[:, :, i][wm_cerebellum_slice_dce].ravel())
            M0_wm_cerebellum_vals.extend(M0_matrix[:, :, i][wm_cerebellum_slice_dce].ravel())
            T1_wm_cc_vals.extend(T1_matrix[:, :, i][wm_cc_slice_dce].ravel())
            M0_wm_cc_vals.extend(M0_matrix[:, :, i][wm_cc_slice_dce].ravel())
            if boundary_mask is not None:
                T1_boundary_vals.extend(T1_matrix[:, :, i][boundary_mask].ravel())
                M0_boundary_vals.extend(M0_matrix[:, :, i][boundary_mask].ravel())

            # Initialize lists to store valid CTCs
            wm_ctcs = []
            cortical_gm_ctcs = []
            subcortical_gm_ctcs = []
            gm_brainstem_ctcs = []
            gm_cerebellum_ctcs = []
            wm_cerebellum_ctcs = []
            wm_cc_ctcs = []
            boundary_ctcs = []

            # Function to process CTCs for a given set of indices
            def process_ctcs(indices):
                ctcs = []
                for (x, y) in indices:
                    voxel_time_course = data_4d[x, y, i, :]
                    T1 = T1_matrix[x, y, i]
                    M0 = M0_matrix[x, y, i]
                    C_t_0 = compute_CTC_meta(voxel_time_course, T1, m0=M0)
                    baseline_point = find_baseline_point_advanced(C_t_0)
                    C_t = custom_shifter(C_t_0, baseline_point)

                    # Exclude CTCs with NaNs or zeros
                    if np.isnan(C_t).any() or np.all(C_t == 0):
                        continue
                    ctcs.append(C_t)
                return ctcs

            # Process CTCs for each tissue type
            wm_ctcs = process_ctcs(wm_indices)
            cortical_gm_ctcs = process_ctcs(cortical_gm_indices)
            subcortical_gm_ctcs = process_ctcs(subcortical_gm_indices)
            gm_brainstem_ctcs = process_ctcs(gm_brainstem_indices)
            gm_cerebellum_ctcs = process_ctcs(gm_cerebellum_indices)
            wm_cerebellum_ctcs = process_ctcs(wm_cerebellum_indices)
            wm_cc_ctcs = process_ctcs(wm_cc_indices)

            # Process CTCs for boundary if required
            if boundary and len(boundary_indices) > 0:
                boundary_ctcs = process_ctcs(boundary_indices)

            # Add the valid CTCs from this slice to the total lists used for
            # global Ki calculations.  White matter, cortical grey matter and
            # boundary contributions can optionally exclude slices from the
            # inferior and superior ends of the volume.
            include_global = (i >= skip_bottom) and (i < n_slices - skip_top)
            if include_global:
                wm_ctcs_total.extend(wm_ctcs)
                cortical_gm_ctcs_total.extend(cortical_gm_ctcs)
                if boundary and boundary_ctcs:
                    boundary_ctcs_total.extend(boundary_ctcs)
            else:
                # Even if excluded from global totals we still compute per-slice
                # and per-voxel values.
                pass
            subcortical_gm_ctcs_total.extend(subcortical_gm_ctcs)
            gm_brainstem_ctcs_total.extend(gm_brainstem_ctcs)
            gm_cerebellum_ctcs_total.extend(gm_cerebellum_ctcs)
            wm_cerebellum_ctcs_total.extend(wm_cerebellum_ctcs)
            wm_cc_ctcs_total.extend(wm_cc_ctcs)

            # Compute aggregated CTCs if valid CTCs are available
            avg_wm_ctc = _aggregate_roi_curves(wm_ctcs, axis=0) if wm_ctcs else np.array([])
            avg_cortical_gm_ctc = _aggregate_roi_curves(cortical_gm_ctcs, axis=0) if cortical_gm_ctcs else np.array([])
            avg_subcortical_gm_ctc = _aggregate_roi_curves(subcortical_gm_ctcs, axis=0) if subcortical_gm_ctcs else np.array([])
            avg_gm_brainstem_ctc = _aggregate_roi_curves(gm_brainstem_ctcs, axis=0) if gm_brainstem_ctcs else np.array([])
            avg_gm_cerebellum_ctc = _aggregate_roi_curves(gm_cerebellum_ctcs, axis=0) if gm_cerebellum_ctcs else np.array([])
            avg_wm_cerebellum_ctc = _aggregate_roi_curves(wm_cerebellum_ctcs, axis=0) if wm_cerebellum_ctcs else np.array([])
            avg_wm_cc_ctc = _aggregate_roi_curves(wm_cc_ctcs, axis=0) if wm_cc_ctcs else np.array([])
            if boundary and boundary_ctcs:
                avg_boundary_ctc = _aggregate_roi_curves(boundary_ctcs, axis=0)
            else:
                avg_boundary_ctc = np.array([])

            if correct_signal_jumps:
                avg_wm_ctc,               bad_wm,_               = mask_problematic(avg_wm_ctc)
                avg_cortical_gm_ctc,      bad_cortical_gm,_      = mask_problematic(avg_cortical_gm_ctc)
                avg_subcortical_gm_ctc,   bad_subcortical_gm,_   = mask_problematic(avg_subcortical_gm_ctc)

                avg_gm_brainstem_ctc,  bad_gm_brainstem,  _ = mask_problematic(avg_gm_brainstem_ctc)
                avg_gm_cerebellum_ctc, bad_gm_cerebellum, _ = mask_problematic(avg_gm_cerebellum_ctc)
                avg_wm_cerebellum_ctc, bad_wm_cerebellum, _ = mask_problematic(avg_wm_cerebellum_ctc)
                avg_wm_cc_ctc,         bad_wm_cc,         _ = mask_problematic(avg_wm_cc_ctc)
                if boundary and avg_boundary_ctc.size:
                    avg_boundary_ctc, bad_boundary,_ = mask_problematic(avg_boundary_ctc)
                else:
                    bad_boundary = None
            else:
                bad_wm = bad_cortical_gm = bad_subcortical_gm = None
                bad_gm_brainstem = bad_gm_cerebellum = bad_wm_cerebellum = None
                bad_wm_cc = bad_boundary = None
            # Save the tissue concentration curves as .npy files
            save_dir_ctc = os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'AI')
            os.makedirs(save_dir_ctc, exist_ok=True)

            np.save(os.path.join(save_dir_ctc, f'wm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_wm_ctc)
            np.save(os.path.join(save_dir_ctc, f'cortical_gm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_cortical_gm_ctc)
            np.save(os.path.join(save_dir_ctc, f'subcortical_gm_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_subcortical_gm_ctc)
            np.save(os.path.join(save_dir_ctc, f'gm_brainstem_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_gm_brainstem_ctc)
            np.save(os.path.join(save_dir_ctc, f'gm_cerebellum_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_gm_cerebellum_ctc)
            np.save(os.path.join(save_dir_ctc, f'wm_cerebellum_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_wm_cerebellum_ctc)
            np.save(os.path.join(save_dir_ctc, f'wm_cc_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_wm_cc_ctc)
            if boundary and avg_boundary_ctc.size > 0:
                np.save(os.path.join(save_dir_ctc, f'boundary_AI_Tissue_slice_{i+1}_segmented_median.npy'), avg_boundary_ctc)

            # Ensure the CTCs and C_a_full have the same length
            if C_a_full is None:
                # No valid AIF → skip per-slice PK fitting but CTCs are saved.
                continue
            min_length = len(C_a_full)
            ctc_list = [
                avg_wm_ctc, avg_cortical_gm_ctc, avg_subcortical_gm_ctc,
                avg_gm_brainstem_ctc, avg_gm_cerebellum_ctc, avg_wm_cerebellum_ctc, avg_wm_cc_ctc,
                avg_boundary_ctc
            ]
            for ctc in ctc_list:
                if ctc.size > 0:
                    min_length = min(min_length, ctc.size)

            C_a_slice = C_a_full[:min_length]
            time_points = time_points_s[:min_length]

            # Truncate CTCs to match length
            C_t_wm = avg_wm_ctc[:min_length] if avg_wm_ctc.size > 0 else np.array([])
            C_t_cortical_gm = avg_cortical_gm_ctc[:min_length] if avg_cortical_gm_ctc.size > 0 else np.array([])
            C_t_subcortical_gm = avg_subcortical_gm_ctc[:min_length] if avg_subcortical_gm_ctc.size > 0 else np.array([])
            C_t_gm_brainstem = avg_gm_brainstem_ctc[:min_length] if avg_gm_brainstem_ctc.size > 0 else np.array([])
            C_t_gm_cerebellum = avg_gm_cerebellum_ctc[:min_length] if avg_gm_cerebellum_ctc.size > 0 else np.array([])
            C_t_wm_cerebellum = avg_wm_cerebellum_ctc[:min_length] if avg_wm_cerebellum_ctc.size > 0 else np.array([])
            C_t_wm_cc = avg_wm_cc_ctc[:min_length] if avg_wm_cc_ctc.size > 0 else np.array([])
            if boundary and avg_boundary_ctc.size > 0:
                C_t_boundary = avg_boundary_ctc[:min_length]
            else:
                C_t_boundary = np.array([])

            def trim_mask(mask):
                if mask is None or not getattr(mask, 'size', 0):
                    return mask
                return mask[:min_length]

            bad_wm = trim_mask(bad_wm)
            bad_cortical_gm = trim_mask(bad_cortical_gm)
            bad_subcortical_gm = trim_mask(bad_subcortical_gm)
            bad_gm_brainstem = trim_mask(bad_gm_brainstem)
            bad_gm_cerebellum = trim_mask(bad_gm_cerebellum)
            bad_wm_cerebellum = trim_mask(bad_wm_cerebellum)
            bad_wm_cc = trim_mask(bad_wm_cc)
            bad_boundary = trim_mask(bad_boundary)

            # Perform kinetic model fit for each tissue type
            def perform_model_fit(C_t):
                if C_t.size == 0:
                    return (np.nan, np.nan, np.nan, None, np.array([], dtype=bool))

                if settings.KINETIC_MODEL.lower() == 'two_compartment':
                    Ki_raw, lam_raw, SD_Ki, fit_curve = two_compartment_tikhonov(
                        C_a_slice, C_t, time_array=time_points
                    )
                    Ki = Ki_raw * 6000
                    lam = lam_raw * 100
                    return Ki, lam, SD_Ki, fit_curve, np.array([], dtype=bool)

                Ki, lam, SD_Ki, x_patlak, y_patlak, included = patlak_analysis_plotting(
                    C_t, C_a_slice, time_points
                )
                return Ki, lam, SD_Ki, (x_patlak, y_patlak), included

            Ki_wm, lambda_wm, SD_Ki_wm, curve_wm, included_wm = perform_model_fit(C_t_wm)
            Ki_cortical_gm, lambda_cortical_gm, SD_Ki_cortical_gm, curve_cortical_gm, included_cortical_gm = perform_model_fit(C_t_cortical_gm)
            Ki_subcortical_gm, lambda_subcortical_gm, SD_Ki_subcortical_gm, curve_subcortical_gm, included_subcortical_gm = perform_model_fit(C_t_subcortical_gm)
            Ki_gm_brainstem, lambda_gm_brainstem, SD_Ki_gm_brainstem, curve_gm_brainstem, included_gm_brainstem = perform_model_fit(C_t_gm_brainstem)
            Ki_gm_cerebellum, lambda_gm_cerebellum, SD_Ki_gm_cerebellum, curve_gm_cerebellum, included_gm_cerebellum = perform_model_fit(C_t_gm_cerebellum)
            Ki_wm_cerebellum, lambda_wm_cerebellum, SD_Ki_wm_cerebellum, curve_wm_cerebellum, included_wm_cerebellum = perform_model_fit(C_t_wm_cerebellum)
            Ki_wm_cc, lambda_wm_cc, SD_Ki_wm_cc, curve_wm_cc, included_wm_cc = perform_model_fit(C_t_wm_cc)
            if boundary and C_t_boundary.size > 0:
                Ki_boundary, lambda_boundary, SD_Ki_boundary, curve_boundary, included_boundary = perform_model_fit(C_t_boundary)
            else:
                Ki_boundary = np.nan
                lambda_boundary = np.nan
                SD_Ki_boundary = np.nan
                curve_boundary = None
                included_boundary = np.array([], dtype=bool)

            # ── Build Tikhonov solver ONCE per slice ──────────────────
            # The solver only depends on (time_points, C_a_slice), identical
            # for all 8 tissue types.  Building once avoids 7 redundant
            # sets of 121 Cholesky factorisations (~8x speed-up).
            _slice_solver = None
            if not settings.ALIGN_AIF_BY_XCORR and len(time_points) >= 2:
                try:
                    _slice_solver = build_tikhonov_validated_slow_solver(
                        time_points,
                        C_a_slice,
                        tissue_density=float(getattr(settings, "TISSUE_DENSITY", 1.04)),
                        hematocrit=float(getattr(settings, "HEMATOCRIT", 0.42)),
                        plasma_derived_aif=bool(getattr(settings, "PLASMA_DERIVED_AIF", False)),
                    )
                except Exception:
                    _slice_solver = None

            def run_mtt_cth(C_t, Ki_value):
                return compute_mtt_cth(
                    settings.CTH_MTT_METHOD,
                    C_t,
                    C_a_slice,
                    time_points,
                    Ki=Ki_value,
                    allow_gamma=True,
                    logger=logger,
                    _solver=_slice_solver,
                )

            CBF_wm, MTT_wm, CTH_wm, extras_wm = run_mtt_cth(C_t_wm, Ki_wm)
            (CBF_cortical_gm, MTT_cortical_gm, CTH_cortical_gm,
             extras_cortical_gm) = run_mtt_cth(C_t_cortical_gm, Ki_cortical_gm)
            (CBF_subcortical_gm, MTT_subcortical_gm, CTH_subcortical_gm,
             extras_subcortical_gm) = run_mtt_cth(C_t_subcortical_gm, Ki_subcortical_gm)
            (CBF_gm_brainstem, MTT_gm_brainstem, CTH_gm_brainstem,
             extras_gm_brainstem) = run_mtt_cth(C_t_gm_brainstem, Ki_gm_brainstem)
            (CBF_gm_cerebellum, MTT_gm_cerebellum, CTH_gm_cerebellum,
             extras_gm_cerebellum) = run_mtt_cth(C_t_gm_cerebellum, Ki_gm_cerebellum)
            (CBF_wm_cerebellum, MTT_wm_cerebellum, CTH_wm_cerebellum,
             extras_wm_cerebellum) = run_mtt_cth(C_t_wm_cerebellum, Ki_wm_cerebellum)
            CBF_wm_cc, MTT_wm_cc, CTH_wm_cc, extras_wm_cc = run_mtt_cth(C_t_wm_cc, Ki_wm_cc)
            if boundary and C_t_boundary.size > 0:
                (CBF_boundary, MTT_boundary, CTH_boundary,
                 extras_boundary) = run_mtt_cth(C_t_boundary, Ki_boundary)
            else:
                CBF_boundary = float('nan')
                MTT_boundary = float('nan')
                CTH_boundary = float('nan')
                extras_boundary = {
                    "method": settings.CTH_MTT_METHOD,
                    "tikhonov": {
                        "cbf": CBF_boundary,
                        "mtt": MTT_boundary,
                        "cth": CTH_boundary,
                    },
                }

            # Collect Ki values for plotting
            Ki_wm_list.append(Ki_wm)
            Ki_cortical_gm_list.append(Ki_cortical_gm)
            Ki_subcortical_gm_list.append(Ki_subcortical_gm)
            Ki_gm_brainstem_list.append(Ki_gm_brainstem)
            Ki_gm_cerebellum_list.append(Ki_gm_cerebellum)
            Ki_wm_cerebellum_list.append(Ki_wm_cerebellum)
            Ki_wm_cc_list.append(Ki_wm_cc)
            if boundary:
                Ki_boundary_list.append(Ki_boundary)

            # Assign Ki values to the masks in the current slice
            Ki_wm_image[:, :, i][wm_slice_dce] = Ki_wm
            Ki_cortical_gm_image[:, :, i][cortical_gm_slice_dce] = Ki_cortical_gm
            Ki_subcortical_gm_image[:, :, i][subcortical_gm_slice_dce] = Ki_subcortical_gm
            Ki_gm_brainstem_image[:, :, i][gm_brainstem_slice_dce] = Ki_gm_brainstem
            Ki_gm_cerebellum_image[:, :, i][gm_cerebellum_slice_dce] = Ki_gm_cerebellum
            Ki_wm_cerebellum_image[:, :, i][wm_cerebellum_slice_dce] = Ki_wm_cerebellum
            Ki_wm_cc_image[:, :, i][wm_cc_slice_dce] = Ki_wm_cc
            if boundary:
                Ki_boundary_image[:, :, i][boundary_mask] = Ki_boundary

            # Plot the results for the current slice. Images are always written
            # under ``AI/Tissue functions`` so that ``_rename_model_outputs`` can
            # move the entire directory to ``AI_patlak`` or ``AI_tikhonov`` after
            # the model run completes.
            fit_curves = {
                'wm': curve_wm,
                'cortical_gm': curve_cortical_gm,
                'subcortical_gm': curve_subcortical_gm,
                'gm_brainstem': curve_gm_brainstem,
                'gm_cerebellum': curve_gm_cerebellum,
                'wm_cerebellum': curve_wm_cerebellum,
                'wm_cc': curve_wm_cc,
                'boundary': curve_boundary
            }

            if settings.KINETIC_MODEL.lower() in {'patlak', 'both'}:
                def unpack(curve):
                    if curve is None:
                        return np.array([]), np.array([])
                    return curve

                x_wm, y_wm = unpack(curve_wm)
                x_cgm, y_cgm = unpack(curve_cortical_gm)
                x_sgm, y_sgm = unpack(curve_subcortical_gm)
                x_bs, y_bs = unpack(curve_gm_brainstem)
                x_gc, y_gc = unpack(curve_gm_cerebellum)
                x_wc, y_wc = unpack(curve_wm_cerebellum)
                x_cc, y_cc = unpack(curve_wm_cc)
                x_bnd, y_bnd = unpack(curve_boundary)
            else:
                x_wm = y_wm = np.array([])
                x_cgm = y_cgm = np.array([])
                x_sgm = y_sgm = np.array([])
                x_bs = y_bs = np.array([])
                x_gc = y_gc = np.array([])
                x_wc = y_wc = np.array([])
                x_cc = y_cc = np.array([])
                x_bnd = y_bnd = np.array([])

            plot_ctcs_and_patlak(
                t2_img[:, :, i], data_4d[:, :, i, 20],
                wm_slice_t2, cortical_gm_slice_t2, subcortical_gm_slice_t2,
                wm_slice_dce, cortical_gm_slice_dce, subcortical_gm_slice_dce,
                C_t_wm, C_t_cortical_gm, C_t_subcortical_gm,
                time_points, C_a_slice,
                x_wm, y_wm, Ki_wm, lambda_wm,
                x_cgm, y_cgm, Ki_cortical_gm, lambda_cortical_gm,
                x_sgm, y_sgm, Ki_subcortical_gm, lambda_subcortical_gm,
                slice_idx=i+1,
                save_path=os.path.join(image_directory, 'AI', 'Tissue functions', f"AI_Tissue_slice_{i+1}_segmented_median.png"),
                boundary_mask=boundary_mask,
                boundary_ctc=C_t_boundary,
                x_patlak_boundary=x_bnd, y_patlak_boundary=y_bnd,
                Ki_boundary=Ki_boundary, lambda_boundary=lambda_boundary,
                included_wm=included_wm,
                included_cortical_gm=included_cortical_gm,
                included_subcortical_gm=included_subcortical_gm,
                included_boundary=included_boundary,
                gm_brainstem_ctc=C_t_gm_brainstem,
                x_patlak_gm_brainstem=x_bs,
                y_patlak_gm_brainstem=y_bs,
                Ki_gm_brainstem=Ki_gm_brainstem,
                lambda_gm_brainstem=lambda_gm_brainstem,
                included_gm_brainstem=included_gm_brainstem,
                gm_cerebellum_ctc=C_t_gm_cerebellum,
                x_patlak_gm_cerebellum=x_gc,
                y_patlak_gm_cerebellum=y_gc,
                Ki_gm_cerebellum=Ki_gm_cerebellum,
                lambda_gm_cerebellum=lambda_gm_cerebellum,
                included_gm_cerebellum=included_gm_cerebellum,
                wm_cerebellum_ctc=C_t_wm_cerebellum,
                x_patlak_wm_cerebellum=x_wc,
                y_patlak_wm_cerebellum=y_wc,
                Ki_wm_cerebellum=Ki_wm_cerebellum,
                lambda_wm_cerebellum=lambda_wm_cerebellum,
                included_wm_cerebellum=included_wm_cerebellum,
                wm_cc_ctc=C_t_wm_cc,
                x_patlak_wm_cc=x_cc,
                y_patlak_wm_cc=y_cc,
                Ki_wm_cc=Ki_wm_cc,
                lambda_wm_cc=lambda_wm_cc,
                included_wm_cc=included_wm_cc,
                model_fits=fit_curves,
                gm_brainstem_mask_t2=gm_brainstem_slice_t2,
                gm_brainstem_mask_dce=gm_brainstem_slice_dce,
                gm_cerebellum_mask_t2=gm_cerebellum_slice_t2,
                gm_cerebellum_mask_dce=gm_cerebellum_slice_dce,
                wm_cerebellum_mask_t2=wm_cerebellum_slice_t2,          
                wm_cerebellum_mask_dce=wm_cerebellum_slice_dce,        
                wm_cc_mask_t2=wm_cc_slice_t2,                          
                wm_cc_mask_dce=wm_cc_slice_dce,
                bad_wm=bad_wm,
                bad_cortical_gm=bad_cortical_gm,
                bad_subcortical_gm=bad_subcortical_gm,
                bad_gm_brainstem=bad_gm_brainstem,
                bad_gm_cerebellum=bad_gm_cerebellum,
                bad_wm_cerebellum=bad_wm_cerebellum,
                bad_wm_cc=bad_wm_cc,
                bad_boundary=bad_boundary                      
    )

            # Collect data for JSON output
            patlak_data = {
                'slice': i + 1,
                'cth_mtt_method': settings.CTH_MTT_METHOD,
                'white_matter_median': {
                    'Ki': Ki_wm,
                    'SD_Ki': SD_Ki_wm,
                    'lambda': lambda_wm,
                    'CBF_tikhonov': CBF_wm,
                    'MTT_tikhonov': MTT_wm,
                    'CTH_tikhonov': CTH_wm,
                    'voxel_count': int(np.sum(wm_slice_dce))
                },
                'cortical_gray_matter_median': {
                    'Ki': Ki_cortical_gm,
                    'SD_Ki': SD_Ki_cortical_gm,
                    'lambda': lambda_cortical_gm,
                    'CBF_tikhonov': CBF_cortical_gm,
                    'MTT_tikhonov': MTT_cortical_gm,
                    'CTH_tikhonov': CTH_cortical_gm,
                    'voxel_count': int(np.sum(cortical_gm_slice_dce))
                },
                'subcortical_gray_matter_median': {
                    'Ki': Ki_subcortical_gm,
                    'SD_Ki': SD_Ki_subcortical_gm,
                    'lambda': lambda_subcortical_gm,
                    'CBF_tikhonov': CBF_subcortical_gm,
                    'MTT_tikhonov': MTT_subcortical_gm,
                    'CTH_tikhonov': CTH_subcortical_gm,
                    'voxel_count': int(np.sum(subcortical_gm_slice_dce))
                },
                'gm_brainstem_median': {
                    'Ki': Ki_gm_brainstem,
                    'SD_Ki': SD_Ki_gm_brainstem,
                    'lambda': lambda_gm_brainstem,
                    'CBF_tikhonov': CBF_gm_brainstem,
                    'MTT_tikhonov': MTT_gm_brainstem,
                    'CTH_tikhonov': CTH_gm_brainstem,
                    'voxel_count': int(np.sum(gm_brainstem_slice_dce))
                },
                'gm_cerebellum_median': {
                    'Ki': Ki_gm_cerebellum,
                    'SD_Ki': SD_Ki_gm_cerebellum,
                    'lambda': lambda_gm_cerebellum,
                    'CBF_tikhonov': CBF_gm_cerebellum,
                    'MTT_tikhonov': MTT_gm_cerebellum,
                    'CTH_tikhonov': CTH_gm_cerebellum,
                    'voxel_count': int(np.sum(gm_cerebellum_slice_dce))
                },
                'wm_cerebellum_median': {
                    'Ki': Ki_wm_cerebellum,
                    'SD_Ki': SD_Ki_wm_cerebellum,
                    'lambda': lambda_wm_cerebellum,
                    'CBF_tikhonov': CBF_wm_cerebellum,
                    'MTT_tikhonov': MTT_wm_cerebellum,
                    'CTH_tikhonov': CTH_wm_cerebellum,
                    'voxel_count': int(np.sum(wm_cerebellum_slice_dce))
                },
                'wm_cc_median': {
                    'Ki': Ki_wm_cc,
                    'SD_Ki': SD_Ki_wm_cc,
                    'lambda': lambda_wm_cc,
                    'CBF_tikhonov': CBF_wm_cc,
                    'MTT_tikhonov': MTT_wm_cc,
                    'CTH_tikhonov': CTH_wm_cc,
                    'voxel_count': int(np.sum(wm_cc_slice_dce))
                }
            }

            patlak_data['white_matter_median'].update(extract_cth_mtt_sidecar_fields(extras_wm))
            patlak_data['cortical_gray_matter_median'].update(extract_cth_mtt_sidecar_fields(extras_cortical_gm))
            patlak_data['subcortical_gray_matter_median'].update(extract_cth_mtt_sidecar_fields(extras_subcortical_gm))
            patlak_data['gm_brainstem_median'].update(extract_cth_mtt_sidecar_fields(extras_gm_brainstem))
            patlak_data['gm_cerebellum_median'].update(extract_cth_mtt_sidecar_fields(extras_gm_cerebellum))
            patlak_data['wm_cerebellum_median'].update(extract_cth_mtt_sidecar_fields(extras_wm_cerebellum))
            patlak_data['wm_cc_median'].update(extract_cth_mtt_sidecar_fields(extras_wm_cc))

            if boundary and avg_boundary_ctc.size > 0:
                patlak_data['boundary_median'] = {
                'Ki': Ki_boundary,
                'SD_Ki': SD_Ki_boundary,
                'lambda': lambda_boundary,
                'CBF_tikhonov': CBF_boundary,
                'MTT_tikhonov': MTT_boundary,
                'CTH_tikhonov': CTH_boundary,
                'voxel_count': int(np.sum(boundary_mask))
            }
                patlak_data['boundary_median'].update(extract_cth_mtt_sidecar_fields(extras_boundary))

            all_patlak_data.append(patlak_data)

            # Compute K_i and/or CBF per voxel if enabled
            if compute_per_voxel_Ki or compute_per_voxel_CBF:
                if (compute_per_voxel_CBF and
                        settings.CTH_MTT_METHOD.lower() in {"gamma", "hybrid"} and
                        not settings.CTH_MTT_GAMMA_VOXELWISE and
                        not getattr(logger, "_gamma_voxel_warning_emitted", False)):
                    logger.warning(
                        "CTH/MTT method '%s' requested but voxelwise gamma fitting is disabled;"
                        " using Tikhonov values for per-voxel maps.",
                        settings.CTH_MTT_METHOD,
                    )
                    logger._gamma_voxel_warning_emitted = True
                # Combine WM and GM masks for the current slice
                gm_slice_dce = np.logical_or.reduce(
                    (
                        cortical_gm_slice_dce,
                        subcortical_gm_slice_dce,
                        gm_brainstem_slice_dce,
                        gm_cerebellum_slice_dce,
                    )
                )
                wm_slice_with_cerebellum = np.logical_or(wm_slice_dce, wm_cerebellum_slice_dce)
                brain_mask_slice = np.logical_or(wm_slice_with_cerebellum, gm_slice_dce)
                brain_indices = np.argwhere(brain_mask_slice)

                # Precompute common time-axis and (optional) cached deconvolution solver.
                common_len = min(int(data_4d.shape[3]), len(C_a_full), len(time_points_s))
                C_a_voxel_common = C_a_full[:common_len]
                time_points_voxel_common = time_points_s[:common_len]

                fast_tikh_solver = None
                fast_delta_t = None
                use_fast_tikh = (
                    compute_per_voxel_CBF
                    and (not settings.ALIGN_AIF_BY_XCORR)
                    and not (
                        settings.CTH_MTT_METHOD.lower() in {"gamma", "hybrid"}
                        and settings.CTH_MTT_GAMMA_VOXELWISE
                    )
                )
                if use_fast_tikh and common_len >= 2:
                    deltas = np.diff(time_points_voxel_common)
                    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
                    if deltas.size:
                        fast_delta_t = float(deltas[0])
                        try:
                            fast_tikh_solver = build_tikhonov_validated_slow_solver(
                                time_points_voxel_common,
                                C_a_voxel_common,
                                lambda_candidates=None,
                                offset_grouping_s=0.05,
                                f_win=50,
                                tissue_density=float(getattr(settings, "TISSUE_DENSITY", 1.04)),
                                hematocrit=float(getattr(settings, "HEMATOCRIT", 0.42)),
                                plasma_derived_aif=bool(getattr(settings, "PLASMA_DERIVED_AIF", False)),
                            )
                        except Exception:
                            fast_tikh_solver = None
                            fast_delta_t = None

                # Initialize per-voxel slice arrays
                if compute_per_voxel_Ki:
                    Ki_slice = np.full(brain_mask_slice.shape, np.nan)
                    lam_slice = np.full(brain_mask_slice.shape, np.nan)
                    SD_slice = np.full(brain_mask_slice.shape, np.nan)
                if compute_per_voxel_CBF:
                    CBF_slice = np.full(brain_mask_slice.shape, np.nan)
                    MTT_slice = np.full(brain_mask_slice.shape, np.nan) if settings.WRITE_MTT else None
                    CTH_slice = np.full(brain_mask_slice.shape, np.nan) if settings.WRITE_CTH else None

                # For each voxel in the brain mask, compute K_i and/or CBF
                # Bind hot functions locally (Python loop micro-optimisation).
                _ctc = compute_CTC_meta
                _baseline = find_baseline_point_advanced
                _shift = custom_shifter
                _patlak = patlak_analysis_plotting

                # When we can use the cached Tikhonov solver, batch solves across voxels.
                fast_coords = []
                fast_curves = []
                fast_offsets = []

                for (x, y) in brain_indices:
                    voxel_time_course = data_4d[x, y, i, :]
                    T1 = T1_matrix[x, y, i]
                    M0 = M0_matrix[x, y, i]
                    C_t_0 = _ctc(voxel_time_course, T1, m0=M0)
                    baseline_point = _baseline(C_t_0)
                    C_t = _shift(C_t_0, baseline_point)

                    # Exclude CTCs with NaNs or zeros
                    if np.isnan(C_t).any() or np.all(C_t == 0):
                        continue

                    # Ensure C_t and C_a_full have the same length
                    min_length_voxel = common_len
                    C_t_voxel = C_t[:min_length_voxel]
                    C_a_voxel = C_a_voxel_common
                    time_points_voxel = time_points_voxel_common

                    if min_length_voxel < 2:
                        continue

                    Ki_value = None
                    if compute_per_voxel_Ki:
                        # Perform Patlak analysis
                        Ki_voxel, lam_voxel, SD_voxel, _, _, _ = _patlak(
                            C_t_voxel, C_a_voxel, time_points_voxel
                        )
                        Ki_slice[x, y] = Ki_voxel
                        lam_slice[x, y] = lam_voxel
                        SD_slice[x, y] = SD_voxel
                        Ki_value = Ki_voxel

                    if compute_per_voxel_CBF:
                        if fast_tikh_solver is not None and fast_delta_t is not None:
                            offset_s = 0.0
                            if bool(getattr(settings, "MATLAB_OFFSET_CORRECTION", False)):
                                try:
                                    offset_s, _ = estimate_bolus_arrival_shift_seconds(C_t_voxel, time_points_voxel)
                                except Exception:
                                    offset_s = 0.0
                            fast_coords.append((int(x), int(y)))
                            fast_curves.append(np.asarray(C_t_voxel, dtype=float))
                            fast_offsets.append(float(offset_s) if np.isfinite(offset_s) else 0.0)
                            continue
                        else:
                            cbf_voxel, mtt_voxel, cth_voxel, _ = compute_mtt_cth(
                                settings.CTH_MTT_METHOD,
                                C_t_voxel,
                                C_a_voxel,
                                time_points_voxel,
                                Ki=Ki_value,
                                allow_gamma=settings.CTH_MTT_GAMMA_VOXELWISE,
                                logger=logger,
                            )
                        if np.isfinite(cbf_voxel):
                            CBF_slice[x, y] = cbf_voxel
                        if settings.WRITE_MTT and MTT_slice is not None and np.isfinite(mtt_voxel):
                            MTT_slice[x, y] = mtt_voxel
                        if settings.WRITE_CTH and CTH_slice is not None and np.isfinite(cth_voxel):
                            CTH_slice[x, y] = cth_voxel

                # Batched fast Tikhonov path (multiple RHS) for voxelwise maps.
                if compute_per_voxel_CBF and fast_tikh_solver is not None and fast_delta_t is not None and fast_curves:
                    n_vox = len(fast_curves)
                    n_time = int(common_len)
                    # Chunk to avoid huge temporary allocations.
                    chunk = int(getattr(settings, "TIKHONOV_BATCH_SIZE", 4096))

                    for start in range(0, n_vox, chunk):
                        end = min(start + chunk, n_vox)
                        Ct_mat = np.stack(fast_curves[start:end], axis=1)
                        if Ct_mat.shape[0] != n_time:
                            Ct_mat = Ct_mat[:n_time, :]

                        off_chunk = np.asarray(fast_offsets[start:end], dtype=float) if fast_offsets else None
                        sol = fast_tikh_solver(Ct_mat, offsets_s=off_chunk)
                        cbf_vals = np.asarray(sol.cbf_ml_per_100g_min, dtype=float).reshape(-1)
                        mtt_vals = np.asarray(sol.mtt_s, dtype=float).reshape(-1)
                        cth_vals = np.asarray(sol.cth_s, dtype=float).reshape(-1)

                        for k in range(end - start):
                            x, y = fast_coords[start + k]
                            cbf_k = float(cbf_vals[k]) if k < cbf_vals.size else float("nan")
                            if np.isfinite(cbf_k):
                                CBF_slice[x, y] = cbf_k
                            if settings.WRITE_MTT and MTT_slice is not None:
                                mtt_k = float(mtt_vals[k]) if k < mtt_vals.size else float("nan")
                                if np.isfinite(mtt_k):
                                    MTT_slice[x, y] = mtt_k
                            if settings.WRITE_CTH and CTH_slice is not None:
                                cth_k = float(cth_vals[k]) if k < cth_vals.size else float("nan")
                                if np.isfinite(cth_k):
                                    CTH_slice[x, y] = cth_k

                # Store the K_i and/or CBF slice in the 3D arrays
                if compute_per_voxel_Ki:
                    Ki_per_voxel[:, :, i] = Ki_slice
                    lambda_per_voxel[:, :, i] = lam_slice
                    SD_per_voxel[:, :, i] = SD_slice
                if compute_per_voxel_CBF:
                    CBF_per_voxel[:, :, i] = CBF_slice
                    if settings.WRITE_MTT and MTT_per_voxel is not None and MTT_slice is not None:
                        MTT_per_voxel[:, :, i] = MTT_slice
                    if settings.WRITE_CTH and CTH_per_voxel is not None and CTH_slice is not None:
                        CTH_per_voxel[:, :, i] = CTH_slice

            if boundary:
                boundary_mask_full[:, :, i] = boundary_mask if boundary_mask is not None else False

    affine = ref_affine

    # Save Patlak tissue Ki images only when Ki computation is enabled.
    if compute_per_voxel_Ki:
        Ki_wm_nii = nib.Nifti1Image(Ki_wm_image, affine)
        Ki_wm_path = os.path.join(analysis_directory, 'Ki_wm.nii.gz')
        nib.save(Ki_wm_nii, Ki_wm_path)
        print(f"K_i WM saved to {Ki_wm_path}")

        Ki_cortical_gm_nii = nib.Nifti1Image(Ki_cortical_gm_image, affine)
        Ki_cortical_gm_path = os.path.join(analysis_directory, 'Ki_cortical_gm.nii.gz')
        nib.save(Ki_cortical_gm_nii, Ki_cortical_gm_path)
        print(f"K_i Cortical GM saved to {Ki_cortical_gm_path}")

        Ki_subcortical_gm_nii = nib.Nifti1Image(Ki_subcortical_gm_image, affine)
        Ki_subcortical_gm_path = os.path.join(analysis_directory, 'Ki_subcortical_gm.nii.gz')
        nib.save(Ki_subcortical_gm_nii, Ki_subcortical_gm_path)
        print(f"K_i Subcortical GM saved to {Ki_subcortical_gm_path}")

        if boundary:
            Ki_boundary_nii = nib.Nifti1Image(Ki_boundary_image, affine)
            Ki_boundary_path = os.path.join(analysis_directory, 'Ki_boundary.nii.gz')
            nib.save(Ki_boundary_nii, Ki_boundary_path)
            print(f"K_i Boundary saved to {Ki_boundary_path}")

    # Compute global min and max for K_i
    if compute_per_voxel_Ki:
        global_Ki_min = np.nanmin(Ki_per_voxel)
        global_Ki_max = np.nanmax(Ki_per_voxel)
        print(f"Global K_i min: {global_Ki_min}, max: {global_Ki_max}")

        # Generate overlay images for K_i
        for i in range(n_slices):
            Ki_slice = Ki_per_voxel[:, :, i]
            if np.isnan(Ki_slice).all():
                continue  # Skip slices without valid K_i values

            save_dir_overlay = os.path.join(image_directory, 'AI', 'Ki Overlays')
            os.makedirs(save_dir_overlay, exist_ok=True)
            save_path_overlay = os.path.join(save_dir_overlay, f"Ki_overlay_slice_{i+1}.png")
            plot_Ki_overlay(
                data_4d[:, :, i, 20], Ki_slice, slice_idx=i+1, save_path=save_path_overlay,
                vmin=global_Ki_min, vmax=global_Ki_max
            )

        # Save Ki_per_voxel as a .nii file
        Ki_per_voxel_nii = nib.Nifti1Image(Ki_per_voxel, affine=ref_affine)
        Ki_per_voxel_path = os.path.join(analysis_directory, 'Ki_per_voxel.nii.gz')
        nib.save(Ki_per_voxel_nii, Ki_per_voxel_path)
        print(f"K_i per voxel saved to {Ki_per_voxel_path}")

        # Save Patlak vp (lambda) per voxel as a .nii file
        vp_data = np.asarray(lambda_per_voxel, dtype=np.float32)
        vp_data = np.where(np.isfinite(vp_data), np.maximum(vp_data, 0.0), vp_data).astype(np.float32)
        if ref_header is not None:
            vp_per_voxel_nii = nib.Nifti1Image(vp_data, affine=ref_affine, header=ref_header.copy())
        else:
            vp_per_voxel_nii = nib.Nifti1Image(vp_data, affine=ref_affine)
        vp_per_voxel_path = os.path.join(analysis_directory, 'vp_per_voxel.nii.gz')
        nib.save(vp_per_voxel_nii, vp_per_voxel_path)
        print(f"v_p per voxel saved to {vp_per_voxel_path}")

    # Compute global min and max for CBF
    if compute_per_voxel_CBF:
        global_CBF_min = np.nanmin(CBF_per_voxel)
        global_CBF_max = np.nanmax(CBF_per_voxel)
        print(f"Global CBF min: {global_CBF_min}, max: {global_CBF_max}")

        # Generate overlay images for CBF
        for i in range(n_slices):
            CBF_slice = CBF_per_voxel[:, :, i]
            if np.isnan(CBF_slice).all():
                continue  # Skip slices without valid CBF values

            save_dir_overlay = os.path.join(image_directory, 'AI', 'CBF Overlays')
            os.makedirs(save_dir_overlay, exist_ok=True)
            save_path_overlay = os.path.join(save_dir_overlay, f"CBF_overlay_slice_{i+1}.png")
            plot_CBF_overlay(
                data_4d[:, :, i, 20], CBF_slice, slice_idx=i+1, save_path=save_path_overlay,
                vmin=global_CBF_min, vmax=global_CBF_max
            )

        # Save CBF_per_voxel as a .nii file
        CBF_per_voxel_nii = nib.Nifti1Image(np.asarray(CBF_per_voxel, dtype=np.float32),
                                            affine=ref_affine,
                                            header=ref_header.copy())
        CBF_per_voxel_path = os.path.join(analysis_directory, 'CBF_per_voxel_tikhonov.nii.gz')
        nib.save(CBF_per_voxel_nii, CBF_per_voxel_path)
        print(f"CBF per voxel saved to {CBF_per_voxel_path}")

        def finite_median(data):
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return np.nan
            return float(np.median(finite))

        def log_median(metric_name, array):
            gm_median = finite_median(array[gm_mask_dce_full])
            wm_median = finite_median(array[wm_mask_dce])
            if np.isfinite(gm_median) or np.isfinite(wm_median):
                print(f"{metric_name} median GM: {gm_median:.3f} s, WM: {wm_median:.3f} s")
            else:
                print(f"{metric_name} median GM/WM: no finite values")

        if settings.WRITE_MTT and MTT_per_voxel is not None:
            mtt_img = nib.Nifti1Image(np.asarray(MTT_per_voxel, dtype=np.float32),
                                      affine=ref_affine,
                                      header=ref_header.copy())
            mtt_img = annotate_cth_mtt_header(mtt_img)
            mtt_path = os.path.join(analysis_directory, 'mtt_map.nii.gz')
            nib.save(mtt_img, mtt_path)
            print(f"MTT map saved to {mtt_path} (method={settings.CTH_MTT_METHOD})")
            log_median("MTT", MTT_per_voxel)

        if settings.WRITE_CTH and CTH_per_voxel is not None:
            cth_img = nib.Nifti1Image(np.asarray(CTH_per_voxel, dtype=np.float32),
                                      affine=ref_affine,
                                      header=ref_header.copy())
            cth_img = annotate_cth_mtt_header(cth_img)
            cth_path = os.path.join(analysis_directory, 'cth_map.nii.gz')
            nib.save(cth_img, cth_path)
            print(f"CTH map saved to {cth_path} (method={settings.CTH_MTT_METHOD})")
            log_median("CTH", CTH_per_voxel)

        # CBV = CBF * MTT / 60  (ml/100g)
        if (
            settings.WRITE_MTT
            and MTT_per_voxel is not None
            and CBF_per_voxel is not None
        ):
            cbv_data = np.asarray(CBF_per_voxel, dtype=np.float32) * np.asarray(MTT_per_voxel, dtype=np.float32) / 60.0
            cbv_img = nib.Nifti1Image(cbv_data, affine=ref_affine, header=ref_header.copy())
            cbv_path = os.path.join(analysis_directory, 'cbv_map.nii.gz')
            nib.save(cbv_img, cbv_path)
            print(f"CBV map saved to {cbv_path}")

    # ── PK diagnostic grid (full tissue path) ──────────────────────
    try:
        _ctc_diag = compute_CTC_meta if "compute_CTC_meta" in dir() else compute_CTC
        _bl_diag = find_baseline_point_advanced
        _sh_diag = custom_shifter
        _generate_pk_diagnostic_grid(
            CBF_per_voxel=CBF_per_voxel if compute_per_voxel_CBF else None,
            MTT_per_voxel=MTT_per_voxel if compute_per_voxel_CBF else None,
            CTH_per_voxel=CTH_per_voxel if compute_per_voxel_CBF else None,
            Ki_per_voxel=Ki_per_voxel if compute_per_voxel_Ki else None,
            C_a_full=C_a_full,
            time_points_s=time_points_s,
            data_4d=data_4d,
            T1_matrix=T1_matrix,
            M0_matrix=M0_matrix,
            brain_mask_full=brain_mask_full if "brain_mask_full" in dir() else (
                np.isfinite(np.nanmean(data_4d, axis=3)) & (np.nanmean(data_4d, axis=3) != 0)
            ),
            analysis_directory=analysis_directory,
            compute_CTC_func=_ctc_diag,
            baseline_func=_bl_diag,
            shift_func=_sh_diag,
        )
    except Exception as _pk_err:
        print(f"[pk_diag] Failed to generate PK diagnostic grid: {_pk_err}")

    # ------------------------------------------------------------------
    # Per-voxel Ki statistics and JSON output
    # ------------------------------------------------------------------
    if compute_per_voxel_Ki:
        voxelwise_slice_data = []
        for i in range(n_slices):
            def median_for(mask, skip=False):
                m = mask.copy()
                if skip:
                    if i < skip_bottom or i >= n_slices - skip_top:
                        m[...] = False
                vals_Ki = Ki_per_voxel[:, :, i][m]
                vals_lam = lambda_per_voxel[:, :, i][m]
                vals_SD = SD_per_voxel[:, :, i][m]
                return (
                    float(np.nanmedian(vals_Ki)) if vals_Ki.size else float('nan'),
                    float(np.nanmedian(vals_SD)) if vals_SD.size else float('nan'),
                    float(np.nanmedian(vals_lam)) if vals_lam.size else float('nan'),
                    int(np.sum(m))
                )

            wm_Ki, wm_SD, wm_lam, wm_vox = median_for(wm_mask_dce[:, :, i], skip=True)
            cgm_Ki, cgm_SD, cgm_lam, cgm_vox = median_for(cortical_gm_mask_dce[:, :, i], skip=True)
            sgm_Ki, sgm_SD, sgm_lam, sgm_vox = median_for(subcortical_gm_mask_dce[:, :, i])
            bs_Ki, bs_SD, bs_lam, bs_vox = median_for(gm_brainstem_mask_dce[:, :, i])
            gc_Ki, gc_SD, gc_lam, gc_vox = median_for(gm_cerebellum_mask_dce[:, :, i])
            wc_Ki, wc_SD, wc_lam, wc_vox = median_for(wm_cerebellum_mask_dce[:, :, i])
            cc_Ki, cc_SD, cc_lam, cc_vox = median_for(wm_cc_mask_dce[:, :, i])
            slice_entry = {
                'slice': i + 1,
                'cth_mtt_method': settings.CTH_MTT_METHOD,
                'white_matter_voxelwise': {'Ki': wm_Ki, 'SD_Ki': wm_SD, 'lambda': wm_lam, 'voxel_count': wm_vox},
                'cortical_gray_matter_voxelwise': {'Ki': cgm_Ki, 'SD_Ki': cgm_SD, 'lambda': cgm_lam, 'voxel_count': cgm_vox},
                'subcortical_gray_matter_voxelwise': {'Ki': sgm_Ki, 'SD_Ki': sgm_SD, 'lambda': sgm_lam, 'voxel_count': sgm_vox},
                'gm_brainstem_voxelwise': {'Ki': bs_Ki, 'SD_Ki': bs_SD, 'lambda': bs_lam, 'voxel_count': bs_vox},
                'gm_cerebellum_voxelwise': {'Ki': gc_Ki, 'SD_Ki': gc_SD, 'lambda': gc_lam, 'voxel_count': gc_vox},
                'wm_cerebellum_voxelwise': {'Ki': wc_Ki, 'SD_Ki': wc_SD, 'lambda': wc_lam, 'voxel_count': wc_vox},
                'wm_cc_voxelwise': {'Ki': cc_Ki, 'SD_Ki': cc_SD, 'lambda': cc_lam, 'voxel_count': cc_vox}
            }
            if boundary:
                b_Ki, b_SD, b_lam, b_vox = median_for(boundary_mask_full[:, :, i], skip=True)
                slice_entry['boundary_voxelwise'] = {'Ki': b_Ki, 'SD_Ki': b_SD, 'lambda': b_lam, 'voxel_count': b_vox}
            voxelwise_slice_data.append(slice_entry)

        json_voxel_slice = os.path.join(analysis_directory, 'AI_values_voxelwise.json')
        with open(json_voxel_slice, 'w') as jf:
            json.dump(voxelwise_slice_data, jf, indent=4)

        # Global medians across slices
        slice_mask = np.ones(n_slices, dtype=bool)
        slice_mask[:skip_bottom] = False
        slice_mask[n_slices - skip_top:] = False

        def global_median(mask, skip=False):
            m = mask.copy()
            if skip:
                m[:, :, ~slice_mask] = False
            vals_Ki = Ki_per_voxel[m]
            vals_lam = lambda_per_voxel[m]
            vals_SD = SD_per_voxel[m]
            return {
                'Ki': float(np.nanmedian(vals_Ki)) if vals_Ki.size else float('nan'),
                'SD_Ki': float(np.nanmedian(vals_SD)) if vals_SD.size else float('nan'),
                'lambda': float(np.nanmedian(vals_lam)) if vals_lam.size else float('nan'),
                'voxel_count': int(np.sum(m))
            }

        voxelwise_global = {
            'white_matter_voxelwise_total': global_median(wm_mask_dce, skip=True),
            'cortical_gm_voxelwise_total': global_median(cortical_gm_mask_dce, skip=True),
            'subcortical_gm_voxelwise_total': global_median(subcortical_gm_mask_dce),
            'gm_brainstem_voxelwise_total': global_median(gm_brainstem_mask_dce),
            'gm_cerebellum_voxelwise_total': global_median(gm_cerebellum_mask_dce),
            'wm_cerebellum_voxelwise_total': global_median(wm_cerebellum_mask_dce),
            'wm_cc_voxelwise_total': global_median(wm_cc_mask_dce)
        }
        if boundary:
            voxelwise_global['boundary_voxelwise_total'] = global_median(boundary_mask_full, skip=True)

        json_voxel_global = os.path.join(analysis_directory, 'AI_values_voxelwise_total.json')
        with open(json_voxel_global, 'w') as jf:
            json.dump(voxelwise_global, jf, indent=4)

    if compute_per_voxel_Ki:
        # Save all Patlak data to JSON file after processing all slices
        json_file_path = os.path.join(analysis_directory, "AI_values_median.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(all_patlak_data, json_file, indent=4)

        # Plot Ki values as a function of slice number
        if Ki_wm_list:
            num_processed_slices = len(Ki_wm_list)
            slice_numbers = range(1, num_processed_slices + 1)

            plt.figure(figsize=(10, 6))
            plt.plot(slice_numbers, Ki_wm_list, label='White Matter Ki', marker='o')
            plt.plot(slice_numbers, Ki_cortical_gm_list, label='Cortical Gray Matter Ki', marker='o')
            plt.plot(slice_numbers, Ki_subcortical_gm_list, label='Subcortical Gray Matter Ki', marker='o')
            if boundary and Ki_boundary_list:
                plt.plot(slice_numbers, Ki_boundary_list, label='Boundary Ki', marker='o')
            plt.xlabel('Slice Number')
            plt.ylabel('K_i')
            plt.title('K_i values across Slices')
            plt.legend()
            plt.grid(True)

            # Ensure the directory exists
            save_dir = os.path.join(image_directory, 'AI', 'Tissue functions')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'Ki_vs_slice_median.png'))
            plt.close()
        else:
            print("No Ki values were computed; skipping Ki plot.")

    # ----------------------------------------------------------------------------- 
    # 1) Build overall-median CTCs (already trimmed to the common min_length)
    # -----------------------------------------------------------------------------
    avg_wm_ctc_total             = _aggregate_roi_curves(wm_ctcs_total, axis=0) if wm_ctcs_total else np.array([])
    avg_cortical_gm_ctc_total    = _aggregate_roi_curves(cortical_gm_ctcs_total, axis=0) if cortical_gm_ctcs_total else np.array([])
    avg_subcortical_gm_ctc_total = _aggregate_roi_curves(subcortical_gm_ctcs_total, axis=0) if subcortical_gm_ctcs_total else np.array([])
    avg_boundary_ctc_total       = _aggregate_roi_curves(boundary_ctcs_total, axis=0) if boundary_ctcs_total else np.array([])
    avg_gm_brainstem_ctc_total   = _aggregate_roi_curves(gm_brainstem_ctcs_total, axis=0) if gm_brainstem_ctcs_total else np.array([])
    avg_gm_cerebellum_ctc_total  = _aggregate_roi_curves(gm_cerebellum_ctcs_total, axis=0) if gm_cerebellum_ctcs_total else np.array([])
    avg_wm_cerebellum_ctc_total  = _aggregate_roi_curves(wm_cerebellum_ctcs_total, axis=0) if wm_cerebellum_ctcs_total else np.array([])
    avg_wm_cc_ctc_total          = _aggregate_roi_curves(wm_cc_ctcs_total, axis=0) if wm_cc_ctcs_total else np.array([])

    # find common length
    min_length = len(C_a_full)
    for ctc in (avg_wm_ctc_total, avg_cortical_gm_ctc_total, avg_subcortical_gm_ctc_total,
                avg_boundary_ctc_total, avg_gm_brainstem_ctc_total, avg_gm_cerebellum_ctc_total,
                avg_wm_cerebellum_ctc_total, avg_wm_cc_ctc_total):
        if ctc.size:
            min_length = min(min_length, ctc.size)

    C_a_total         = C_a_full[:min_length]
    time_points_total = time_points_s[:min_length]

    C_t_wm_total             = avg_wm_ctc_total[:min_length]
    C_t_cortical_gm_total    = avg_cortical_gm_ctc_total[:min_length]
    C_t_subcortical_gm_total = avg_subcortical_gm_ctc_total[:min_length]
    C_t_boundary_total       = avg_boundary_ctc_total[:min_length]
    C_t_gm_brainstem_total   = avg_gm_brainstem_ctc_total[:min_length]
    C_t_gm_cerebellum_total  = avg_gm_cerebellum_ctc_total[:min_length]
    C_t_wm_cerebellum_total  = avg_wm_cerebellum_ctc_total[:min_length]
    C_t_wm_cc_total          = avg_wm_cc_ctc_total[:min_length]

    if settings.AUTO_LAMBDA and settings.AUTO_LAMBDA_VALUE is None:
        global_ctcs = [
            C_t_wm_total,
            C_t_cortical_gm_total,
            C_t_subcortical_gm_total,
            C_t_gm_brainstem_total,
            C_t_gm_cerebellum_total,
            C_t_wm_cerebellum_total,
            C_t_wm_cc_total,
        ]
        if boundary and C_t_boundary_total.size:
            global_ctcs.append(C_t_boundary_total)
        stacked = np.vstack([ct for ct in global_ctcs if ct.size])
        median_ct = _aggregate_roi_curves(stacked, axis=0)
        lambd = pick_lambda_via_l_curve(
            C_a_total,
            median_ct,
            time_points_total,
            settings.AUTO_LAMBDA_CANDIDATES,
            penalty=getattr(settings, "TIKHONOV_PENALTY", "identity"),
        )
        settings.AUTO_LAMBDA_VALUE = lambd
        plot_l_curve(
            C_a_total,
            median_ct,
            time_points_total,
            settings.AUTO_LAMBDA_CANDIDATES,
            best=lambd,
            penalty=getattr(settings, "TIKHONOV_PENALTY", "identity"),
        )
        save_dir = os.path.join(image_directory, 'AI', 'Tissue functions')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'lcurve_global.png'), dpi=300)
        plt.close()

    # ----------------------------------------------------------------------------- 
    # 2) Helper: run mask_problematic on the *trimmed* curve, then Patlak
    # -----------------------------------------------------------------------------
    def patlak_total(C_t):
        if not C_t.size:
            return np.nan, np.nan, np.nan, None
        if settings.KINETIC_MODEL.lower() == 'two_compartment':
            Ki, lam, SD_Ki, fit_curve = two_compartment_tikhonov(
                C_a_total, C_t, time_array=time_points_total
            )
            return Ki, lam, SD_Ki, fit_curve
        if correct_signal_jumps:
            _, bad, _ = mask_problematic(C_t)
        else:
            bad = None
        Ki, lam, SD, *_ = patlak_with_exclusions(C_t, C_a_total, time_points_total, bad_mask=bad)
        return Ki, lam, SD, None

    # ── Build Tikhonov solver ONCE for global-total tissue curves ────
    _total_solver = None
    if not settings.ALIGN_AIF_BY_XCORR and len(time_points_total) >= 2:
        try:
            _total_solver = build_tikhonov_validated_slow_solver(
                time_points_total,
                C_a_total,
                tissue_density=float(getattr(settings, "TISSUE_DENSITY", 1.04)),
                hematocrit=float(getattr(settings, "HEMATOCRIT", 0.42)),
                plasma_derived_aif=bool(getattr(settings, "PLASMA_DERIVED_AIF", False)),
            )
        except Exception:
            _total_solver = None

    def tikhonov_total(C_t, Ki_value):
        cbf, mtt, cth, extras = compute_mtt_cth(
            settings.CTH_MTT_METHOD,
            C_t,
            C_a_total,
            time_points_total,
            Ki=Ki_value,
            allow_gamma=True,
            logger=logger,
            _solver=_total_solver,
        )
        return cbf, mtt, cth, extras

    # ----------------------------------------------------------------------------- 
    # 3) Patlak for every tissue
    # -----------------------------------------------------------------------------
    Ki_wm_total,           lambda_wm_total,           SD_Ki_wm_total,           fit_wm_total = patlak_total(C_t_wm_total)
    Ki_cortical_gm_total,  lambda_cortical_gm_total,  SD_Ki_cortical_gm_total,  fit_cortical_gm_total = patlak_total(C_t_cortical_gm_total)
    Ki_subcortical_gm_total,lambda_subcortical_gm_total,SD_Ki_subcortical_gm_total, fit_subcortical_gm_total = patlak_total(C_t_subcortical_gm_total)
    Ki_boundary_total,     lambda_boundary_total,     SD_Ki_boundary_total,     fit_boundary_total = patlak_total(C_t_boundary_total)
    Ki_gm_brainstem_total, lambda_gm_brainstem_total, SD_Ki_gm_brainstem_total, fit_gm_brainstem_total = patlak_total(C_t_gm_brainstem_total)
    Ki_gm_cerebellum_total,lambda_gm_cerebellum_total,SD_Ki_gm_cerebellum_total, fit_gm_cerebellum_total = patlak_total(C_t_gm_cerebellum_total)
    Ki_wm_cerebellum_total,lambda_wm_cerebellum_total,SD_Ki_wm_cerebellum_total, fit_wm_cerebellum_total = patlak_total(C_t_wm_cerebellum_total)
    Ki_wm_cc_total,        lambda_wm_cc_total,        SD_Ki_wm_cc_total,        fit_wm_cc_total = patlak_total(C_t_wm_cc_total)

    (CBF_wm_total,          MTT_wm_total,          CTH_wm_total,
     extras_wm_total) = tikhonov_total(C_t_wm_total, Ki_wm_total)
    (CBF_cortical_gm_total, MTT_cortical_gm_total, CTH_cortical_gm_total,
     extras_cortical_gm_total) = tikhonov_total(C_t_cortical_gm_total, Ki_cortical_gm_total)
    (CBF_subcortical_gm_total, MTT_subcortical_gm_total, CTH_subcortical_gm_total,
     extras_subcortical_gm_total) = tikhonov_total(C_t_subcortical_gm_total, Ki_subcortical_gm_total)
    (CBF_boundary_total,    MTT_boundary_total,    CTH_boundary_total,
     extras_boundary_total) = tikhonov_total(C_t_boundary_total, Ki_boundary_total)
    (CBF_gm_brainstem_total, MTT_gm_brainstem_total, CTH_gm_brainstem_total,
     extras_gm_brainstem_total) = tikhonov_total(C_t_gm_brainstem_total, Ki_gm_brainstem_total)
    (CBF_gm_cerebellum_total, MTT_gm_cerebellum_total, CTH_gm_cerebellum_total,
     extras_gm_cerebellum_total) = tikhonov_total(C_t_gm_cerebellum_total, Ki_gm_cerebellum_total)
    (CBF_wm_cerebellum_total, MTT_wm_cerebellum_total, CTH_wm_cerebellum_total,
     extras_wm_cerebellum_total) = tikhonov_total(C_t_wm_cerebellum_total, Ki_wm_cerebellum_total)
    (CBF_wm_cc_total,       MTT_wm_cc_total,       CTH_wm_cc_total,
     extras_wm_cc_total) = tikhonov_total(C_t_wm_cc_total, Ki_wm_cc_total)

    # ----------------------------------------------------------------------------- 
    # 4) Collect everything for JSON and plotting
    # -----------------------------------------------------------------------------
    tissue_results = {
        "cth_mtt_method": settings.CTH_MTT_METHOD,
        "white_matter": {
            "C_t": C_t_wm_total,
            "Ki": Ki_wm_total,
            "lam": lambda_wm_total,
            "SD_Ki": SD_Ki_wm_total,
            "fit_curve": fit_wm_total,
            "vox": len(wm_ctcs_total),
            "CBF_tikhonov": CBF_wm_total,
            "MTT_tikhonov": MTT_wm_total,
            "CTH_tikhonov": CTH_wm_total,
        },
        "cortical_gm": {
            "C_t": C_t_cortical_gm_total,
            "Ki": Ki_cortical_gm_total,
            "lam": lambda_cortical_gm_total,
            "SD_Ki": SD_Ki_cortical_gm_total,
            "fit_curve": fit_cortical_gm_total,
            "vox": len(cortical_gm_ctcs_total),
            "CBF_tikhonov": CBF_cortical_gm_total,
            "MTT_tikhonov": MTT_cortical_gm_total,
            "CTH_tikhonov": CTH_cortical_gm_total,
        },
        "subcortical_gm": {
            "C_t": C_t_subcortical_gm_total,
            "Ki": Ki_subcortical_gm_total,
            "lam": lambda_subcortical_gm_total,
            "SD_Ki": SD_Ki_subcortical_gm_total,
            "fit_curve": fit_subcortical_gm_total,
            "vox": len(subcortical_gm_ctcs_total),
            "CBF_tikhonov": CBF_subcortical_gm_total,
            "MTT_tikhonov": MTT_subcortical_gm_total,
            "CTH_tikhonov": CTH_subcortical_gm_total,
        },
        "gm_brainstem": {
            "C_t": C_t_gm_brainstem_total,
            "Ki": Ki_gm_brainstem_total,
            "lam": lambda_gm_brainstem_total,
            "SD_Ki": SD_Ki_gm_brainstem_total,
            "fit_curve": fit_gm_brainstem_total,
            "vox": len(gm_brainstem_ctcs_total),
            "CBF_tikhonov": CBF_gm_brainstem_total,
            "MTT_tikhonov": MTT_gm_brainstem_total,
            "CTH_tikhonov": CTH_gm_brainstem_total,
        },
        "gm_cerebellum": {
            "C_t": C_t_gm_cerebellum_total,
            "Ki": Ki_gm_cerebellum_total,
            "lam": lambda_gm_cerebellum_total,
            "SD_Ki": SD_Ki_gm_cerebellum_total,
            "fit_curve": fit_gm_cerebellum_total,
            "vox": len(gm_cerebellum_ctcs_total),
            "CBF_tikhonov": CBF_gm_cerebellum_total,
            "MTT_tikhonov": MTT_gm_cerebellum_total,
            "CTH_tikhonov": CTH_gm_cerebellum_total,
        },
        "wm_cerebellum": {
            "C_t": C_t_wm_cerebellum_total,
            "Ki": Ki_wm_cerebellum_total,
            "lam": lambda_wm_cerebellum_total,
            "SD_Ki": SD_Ki_wm_cerebellum_total,
            "fit_curve": fit_wm_cerebellum_total,
            "vox": len(wm_cerebellum_ctcs_total),
            "CBF_tikhonov": CBF_wm_cerebellum_total,
            "MTT_tikhonov": MTT_wm_cerebellum_total,
            "CTH_tikhonov": CTH_wm_cerebellum_total,
        },
        "wm_cc": {
            "C_t": C_t_wm_cc_total,
            "Ki": Ki_wm_cc_total,
            "lam": lambda_wm_cc_total,
            "SD_Ki": SD_Ki_wm_cc_total,
            "fit_curve": fit_wm_cc_total,
            "vox": len(wm_cc_ctcs_total),
            "CBF_tikhonov": CBF_wm_cc_total,
            "MTT_tikhonov": MTT_wm_cc_total,
            "CTH_tikhonov": CTH_wm_cc_total,
        },
    }

    tissue_results["white_matter"].update(extract_cth_mtt_sidecar_fields(extras_wm_total))
    tissue_results["cortical_gm"].update(extract_cth_mtt_sidecar_fields(extras_cortical_gm_total))
    tissue_results["subcortical_gm"].update(extract_cth_mtt_sidecar_fields(extras_subcortical_gm_total))
    tissue_results["gm_brainstem"].update(extract_cth_mtt_sidecar_fields(extras_gm_brainstem_total))
    tissue_results["gm_cerebellum"].update(extract_cth_mtt_sidecar_fields(extras_gm_cerebellum_total))
    tissue_results["wm_cerebellum"].update(extract_cth_mtt_sidecar_fields(extras_wm_cerebellum_total))
    tissue_results["wm_cc"].update(extract_cth_mtt_sidecar_fields(extras_wm_cc_total))

    if boundary and C_t_boundary_total.size:
        tissue_results["boundary"] = {
            "C_t": C_t_boundary_total,
            "Ki": Ki_boundary_total,
            "lam": lambda_boundary_total,
            "SD_Ki": SD_Ki_boundary_total,
            "fit_curve": fit_boundary_total,
            "vox": len(boundary_ctcs_total),
            "CBF_tikhonov": CBF_boundary_total,
            "MTT_tikhonov": MTT_boundary_total,
            "CTH_tikhonov": CTH_boundary_total,
        }
        tissue_results["boundary"].update(extract_cth_mtt_sidecar_fields(extras_boundary_total))
    elif boundary:
        tissue_results["boundary"] = extract_cth_mtt_sidecar_fields(extras_boundary_total)

    # ----------------------------------------------------------------------
    # Compute global median T1 and M0 values for each tissue
    # ----------------------------------------------------------------------
    def median_or_nan(vals):
        return float(np.median(vals)) if vals else float('nan')

    t1_m0_results = {
        "white_matter_median_total": {
            "T1": median_or_nan(T1_wm_vals),
            "M0": median_or_nan(M0_wm_vals),
            "voxel_count": len(T1_wm_vals)
        },
        "cortical_gm_median_total": {
            "T1": median_or_nan(T1_cortical_gm_vals),
            "M0": median_or_nan(M0_cortical_gm_vals),
            "voxel_count": len(T1_cortical_gm_vals)
        },
        "subcortical_gm_median_total": {
            "T1": median_or_nan(T1_subcortical_gm_vals),
            "M0": median_or_nan(M0_subcortical_gm_vals),
            "voxel_count": len(T1_subcortical_gm_vals)
        },
        "gm_brainstem_median_total": {
            "T1": median_or_nan(T1_gm_brainstem_vals),
            "M0": median_or_nan(M0_gm_brainstem_vals),
            "voxel_count": len(T1_gm_brainstem_vals)
        },
        "gm_cerebellum_median_total": {
            "T1": median_or_nan(T1_gm_cerebellum_vals),
            "M0": median_or_nan(M0_gm_cerebellum_vals),
            "voxel_count": len(T1_gm_cerebellum_vals)
        },
        "wm_cerebellum_median_total": {
            "T1": median_or_nan(T1_wm_cerebellum_vals),
            "M0": median_or_nan(M0_wm_cerebellum_vals),
            "voxel_count": len(T1_wm_cerebellum_vals)
        },
        "wm_cc_median_total": {
            "T1": median_or_nan(T1_wm_cc_vals),
            "M0": median_or_nan(M0_wm_cc_vals),
            "voxel_count": len(T1_wm_cc_vals)
        },
    }

    if boundary and T1_boundary_vals:
        t1_m0_results["boundary_median_total"] = {
            "T1": median_or_nan(T1_boundary_vals),
            "M0": median_or_nan(M0_boundary_vals),
            "voxel_count": len(T1_boundary_vals)
        }

    json_file_path_t1m0 = os.path.join(analysis_directory, "T1_M0_values_median_total.json")
    with open(json_file_path_t1m0, "w") as jf:
        safe = _json_finite_to_none(t1_m0_results)
        try:
            json.dump(safe, jf, indent=4, allow_nan=False)
        except ValueError:
            json.dump(safe, jf, indent=4)

    # Write JSON (legacy + model-specific names expected by p-brain-web).
    extra_keys = [
        "cth_mtt_method", "MTT_tikh_s", "CTH_tikh_s", "MTT_gamma_s", "CTH_gamma_s",
        "gamma_a", "gamma_b", "gamma_t0_s", "gamma_F_ml_per_100g_min", "gamma_E",
        "gamma_shape_ratio", "gamma_residual_norm", "gamma_iterations", "gamma_success",
    ]
    payload = {
        t + "_median_total": {
            "Ki":            d["Ki"],
            "SD_Ki":         d["SD_Ki"],
            "lambda":        d["lam"],
            "CBF_tikhonov":  d["CBF_tikhonov"],
            "MTT_tikhonov":  d["MTT_tikhonov"],
            "CTH_tikhonov":  d["CTH_tikhonov"],
            "voxel_count":   d["vox"],
            **{k: d[k] for k in extra_keys if k in d},
        }
        for t, d in tissue_results.items()
        if isinstance(d, dict) and {"Ki", "SD_Ki", "lam", "CBF_tikhonov", "MTT_tikhonov", "CTH_tikhonov", "vox"} <= d.keys()
    }
    payload["cth_mtt_method"] = settings.CTH_MTT_METHOD

    json_paths = [
        os.path.join(analysis_directory, "AI_values_median_total.json"),
        os.path.join(analysis_directory, "AI_values_median_total_tikhonov.json"),
        os.path.join(analysis_directory, "AI_values_median_total_patlak.json"),
    ]
    for json_file_path_total in json_paths:
        with open(json_file_path_total, "w") as jf:
            safe = _json_finite_to_none(payload)
            try:
                json.dump(safe, jf, indent=4, allow_nan=False)
            except ValueError:
                json.dump(safe, jf, indent=4)

    # ----------------------------------------------------------------------------- 
    # 5) Create one PNG per tissue
    # -----------------------------------------------------------------------------
    for tissue_name, vals in tissue_results.items():
        if not isinstance(vals, dict) or "C_t" not in vals:
            continue
        save_path = os.path.join(image_directory, "AI", "Tissue functions",
                                f"{tissue_name}_total_CT_and_patlak.png")
        plot_total_ct_and_patlak(
            time_points=time_points_total,
            C_t_total  = vals["C_t"],
            C_a        = C_a_total,
            Ki         = vals["Ki"],
            lam        = vals["lam"],
            SD_Ki      = vals["SD_Ki"],
            fit_curve  = vals.get("fit_curve"),
            tissue_name= tissue_name.replace('_', ' ').title(),
            save_path  = save_path
        )


# ── Global-average Patlak diagnostic plot ────────────────────────────────
def _generate_global_average_patlak_plot(
    *,
    data_4d,
    T1_matrix,
    M0_matrix,
    brain_mask_full,
    C_a_full,
    time_points_s,
    analysis_directory,
    compute_CTC_func,
    baseline_func,
    shift_func,
):
    """Compute the brain-averaged CTC and run Patlak analysis on it.

    Produces a 2-panel figure:
        Top:   brain-average CTC vs. AIF (concentration curves)
        Bottom: Patlak plot with fitted regression line

    Saved as ``<analysis_directory>/global_average_patlak.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    spatial = data_4d.shape[:3]
    n_time = int(data_4d.shape[3])
    common_len = min(n_time, len(C_a_full), len(time_points_s))
    C_a = np.asarray(C_a_full[:common_len], dtype=float)
    t = np.asarray(time_points_s[:common_len], dtype=float)

    # Accumulate brain-averaged CTC across all brain voxels.
    accum = np.zeros(common_len, dtype=float)
    count = 0
    for k in range(spatial[2]):
        mask_slice = brain_mask_full[:, :, k]
        indices = np.argwhere(mask_slice)
        if indices.size == 0:
            continue
        for (x, y) in indices:
            voxel = data_4d[x, y, k, :]
            T1 = T1_matrix[x, y, k]
            M0 = M0_matrix[x, y, k]
            C_t_0 = compute_CTC_func(voxel, T1, m0=M0)
            bp = baseline_func(C_t_0)
            C_t = shift_func(C_t_0, bp)
            if np.isnan(C_t).any() or np.all(C_t == 0):
                continue
            c_t_clip = np.asarray(C_t[:common_len], dtype=float)
            if c_t_clip.size < common_len:
                continue
            accum += c_t_clip
            count += 1

    if count < 10:
        print(f"[global_patlak] Only {count} brain voxels — skipping.")
        return

    avg_ct = accum / count

    # Patlak analysis on the averaged curve.
    Ki, lam, SD, x_p, y_p, good = patlak_analysis_plotting(avg_ct, C_a, t)

    # ── Plot ──
    fig = plt.figure(figsize=(10, 9))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
    gs.update(hspace=0.35)

    ax_ctc = fig.add_subplot(gs[0])
    ax_pat = fig.add_subplot(gs[1])

    # Panel 1: CTC + AIF
    ax_ctc.set_facecolor("#f7f7f7")
    ax_ctc.plot(t, C_a, color="purple", lw=2, alpha=0.6, label="AIF  (input function)")
    ax_ctc.plot(t, avg_ct, color="steelblue", lw=2, label=f"Brain avg CTC  (n={count})")
    ax_ctc.set_xlabel("Time (s)")
    ax_ctc.set_ylabel("Concentration (mmol)")
    ax_ctc.set_title("Global Brain-Averaged CTC")
    ax_ctc.legend(loc="upper right")
    ax_ctc.grid(True, alpha=0.4)

    # Panel 2: Patlak plot
    ax_pat.set_facecolor("#f7f7f7")
    if x_p is not None and y_p is not None and x_p.size:
        ax_pat.scatter(x_p[~good], y_p[~good], c="lightgray", s=12, alpha=0.5, label="excluded")
        ax_pat.scatter(x_p[good], y_p[good], c="steelblue", s=18, alpha=0.8, label="included")
        if np.isfinite(Ki) and np.isfinite(lam):
            xg = x_p[good]
            ax_pat.plot(xg, Ki / 6000.0 * xg + lam / 100.0, color="red", lw=2,
                        label=f"Ki={Ki:.2f} ml/100g/min\n$v_p$={lam:.2f}%")
    ax_pat.set_xlabel("∫ Ca(τ)dτ / Ca(t)  (s)")
    ax_pat.set_ylabel("Ct(t) / Ca(t)")
    ax_pat.set_title("Global Average Patlak Plot")
    ax_pat.legend(loc="upper left")
    ax_pat.grid(True, alpha=0.4)

    save_path = os.path.join(analysis_directory, "global_average_patlak.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[global_patlak] Saved → {save_path}")
    print(f"[global_patlak] Ki = {Ki:.4f} ml/100g/min, vp = {lam:.4f}%, SD = {SD:.4f}, n_voxels = {count}")


def _generate_etofts_diagnostic_grid(
    *,
    Ktrans_per_voxel,
    ve_per_voxel,
    vp_etofts_per_voxel,
    kep_per_voxel,
    Ki_per_voxel,
    C_a_full,
    time_points_s,
    data_4d,
    T1_matrix,
    M0_matrix,
    brain_mask_full,
    analysis_directory,
    compute_CTC_func,
    baseline_func,
    shift_func,
):
    """Generate a diagnostic grid for the Extended Tofts model.

    3×3 layout:
      Row 1: Ktrans map slice | ve map slice | vp map slice
      Row 2: Measured vs fitted CTC | Residual plot | kep map slice
      Row 3: Ktrans histogram | ve histogram | vp histogram
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from models.extended_tofts import forward as etofts_forward, fit_voxel as _etofts_fit

    Ktrans_map = np.asarray(Ktrans_per_voxel, dtype=float)
    if Ktrans_map.ndim != 3 or np.all(~np.isfinite(Ktrans_map)):
        print("[etofts_diag] Ktrans map empty — skipping.")
        return

    # Pick a representative voxel: highest Ktrans in the middle slices.
    n_sl = Ktrans_map.shape[2]
    mid_start = max(0, n_sl // 4)
    mid_end = min(n_sl, n_sl - n_sl // 4)
    sub = Ktrans_map[:, :, mid_start:mid_end].copy()
    sub[~np.isfinite(sub)] = -np.inf
    flat_idx = np.argmax(sub)
    vx, vy, vz_rel = np.unravel_index(flat_idx, sub.shape)
    vz = vz_rel + mid_start

    print(f"[etofts_diag] Diagnostic voxel: x={vx}, y={vy}, z={vz}  "
          f"Ktrans={Ktrans_map[vx, vy, vz]:.2f} ml/100g/min")

    # Extract and fit that voxel.
    voxel_signal = np.asarray(data_4d[vx, vy, vz, :], dtype=float)
    T1_vox = float(T1_matrix[vx, vy, vz])
    M0_vox = float(M0_matrix[vx, vy, vz])
    C_t_raw = compute_CTC_func(voxel_signal, T1_vox, m0=M0_vox)
    bp = baseline_func(C_t_raw)
    C_t = shift_func(C_t_raw, bp)

    common_len = min(int(data_4d.shape[3]), len(C_a_full), len(time_points_s))
    C_t_v = np.asarray(C_t[:common_len], dtype=float)
    C_a_v = np.asarray(C_a_full[:common_len], dtype=float)
    t_v = np.asarray(time_points_s[:common_len], dtype=float)

    if C_t_v.size < 10:
        print("[etofts_diag] Tissue curve too short — skipping.")
        return

    # Patlak init if available.
    _init_kt = None
    _init_vp = None
    if Ki_per_voxel is not None:
        ki_val = float(Ki_per_voxel[vx, vy, vz])
        if np.isfinite(ki_val):
            _init_kt = max(ki_val / 6000.0, 1e-8)

    ef = _etofts_fit(C_a_v, C_t_v, t_v, init_ktrans=_init_kt, init_vp=_init_vp)
    if not ef.success:
        print("[etofts_diag] ETofts fit failed at diagnostic voxel — skipping.")
        return

    # Forward prediction.
    C_t_pred = etofts_forward(t_v, ef.ktrans_raw, ef.ve, ef.vp, C_a_v)
    residual = C_t_v - C_t_pred

    # ── Figure ──
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    gs.update(hspace=0.4, wspace=0.35)

    # Row 1: Map slices
    maps_info = [
        (Ktrans_per_voxel, "Ktrans [ml/100g/min]", "hot"),
        (ve_per_voxel, "ve [fraction]", "viridis"),
        (vp_etofts_per_voxel, "vp [fraction]", "plasma"),
    ]
    for col, (arr, title, cmap) in enumerate(maps_info):
        ax = fig.add_subplot(gs[0, col])
        sl = np.asarray(arr[:, :, vz], dtype=float)
        sl[~np.isfinite(sl)] = np.nan
        im = ax.imshow(sl.T, origin="lower", cmap=cmap, interpolation="nearest")
        ax.plot(vx, vy, "r+", ms=12, mew=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)

    # Row 2, col 0: Measured vs Fitted CTC
    ax_ctc = fig.add_subplot(gs[1, 0])
    ax_ctc.set_facecolor("#f7f7f7")
    ax_ctc.plot(t_v, C_t_v, "k.", ms=3, alpha=0.5, label="Measured")
    ax_ctc.plot(t_v, C_t_pred, "r-", lw=1.5, label="ETofts fit")
    ax_ctc.set_xlabel("Time [s]")
    ax_ctc.set_ylabel("Ct")
    ax_ctc.set_title("Measured vs ETofts Fit", fontsize=10)
    ax_ctc.legend(fontsize=8)
    ax_ctc.grid(True, alpha=0.3)

    # Row 2, col 1: Residual
    ax_res = fig.add_subplot(gs[1, 1])
    ax_res.set_facecolor("#f7f7f7")
    ax_res.plot(t_v, residual, "b-", lw=0.8, alpha=0.7)
    ax_res.axhline(0, color="gray", ls="--", lw=0.5)
    ax_res.set_xlabel("Time [s]")
    ax_res.set_ylabel("Residual")
    ax_res.set_title("Fit Residual", fontsize=10)
    ax_res.grid(True, alpha=0.3)

    # Row 2, col 2: kep map slice
    ax_kep = fig.add_subplot(gs[1, 2])
    kep_sl = np.asarray(kep_per_voxel[:, :, vz], dtype=float)
    kep_sl[~np.isfinite(kep_sl)] = np.nan
    im_kep = ax_kep.imshow(kep_sl.T, origin="lower", cmap="inferno", interpolation="nearest")
    ax_kep.plot(vx, vy, "r+", ms=12, mew=2)
    ax_kep.set_title("kep [1/min]", fontsize=10)
    ax_kep.set_xticks([])
    ax_kep.set_yticks([])
    fig.colorbar(im_kep, ax=ax_kep, shrink=0.7)

    # Row 3: Histograms
    hist_info = [
        (Ktrans_per_voxel, "Ktrans [ml/100g/min]", "orangered"),
        (ve_per_voxel, "ve [fraction]", "teal"),
        (vp_etofts_per_voxel, "vp [fraction]", "purple"),
    ]
    for col, (arr, xlabel, color) in enumerate(hist_info):
        ax_h = fig.add_subplot(gs[2, col])
        vals = np.asarray(arr, dtype=float).ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size > 0:
            p99 = np.percentile(vals, 99)
            vals_clip = vals[vals <= p99]
            ax_h.hist(vals_clip, bins=80, color=color, alpha=0.7, edgecolor="none")
        ax_h.set_xlabel(xlabel, fontsize=9)
        ax_h.set_ylabel("Count", fontsize=9)
        ax_h.set_title(f"Distribution (n={vals.size})", fontsize=9)
        ax_h.grid(True, alpha=0.3)

    # Suptitle with fit parameters.
    fig.suptitle(
        f"Extended Tofts Diagnostic  —  voxel ({vx},{vy},{vz})\n"
        f"Ktrans={ef.ktrans_ml_100g:.2f} ml/100g/min   ve={ef.ve:.3f}   vp={ef.vp:.4f}   "
        f"kep={ef.kep_per_min:.2f} /min   R²={1 - np.sum(residual**2)/np.sum((C_t_v - np.mean(C_t_v))**2):.4f}",
        fontsize=11, y=0.99,
    )

    os.makedirs(analysis_directory, exist_ok=True)
    save_path = os.path.join(analysis_directory, "etofts_diagnostic_grid.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[etofts_diag] Saved → {save_path}")


def _generate_pk_diagnostic_grid(
    *,
    CBF_per_voxel,
    MTT_per_voxel,
    CTH_per_voxel,
    Ki_per_voxel,
    C_a_full,
    time_points_s,
    data_4d,
    T1_matrix,
    M0_matrix,
    brain_mask_full,
    analysis_directory,
    compute_CTC_func,
    baseline_func,
    shift_func,
):
    """Generate a per-voxel PK diagnostic grid plot.

    Picks a representative voxel (highest CBF in middle slices), re-solves
    the Tikhonov deconvolution for that single voxel to capture all
    intermediate L-curve data, runs a Patlak fit, and renders a 4×3
    diagnostic panel saved as ``pk_diagnostic_grid.png``.
    """

    from utils.pk_diagnostics import write_pk_diagnostic_grid, pick_diagnostic_voxel

    # Need at least CBF to pick a voxel.
    if CBF_per_voxel is None:
        print("[pk_diag] CBF map unavailable — skipping diagnostic grid.")
        return

    cbf_map = np.asarray(CBF_per_voxel, dtype=float)
    if cbf_map.ndim != 3 or np.all(~np.isfinite(cbf_map)):
        print("[pk_diag] CBF map empty or wrong shape — skipping.")
        return

    # Pick representative voxel.
    vx, vy, vz = pick_diagnostic_voxel(
        cbf_map,
        mtt_map=np.asarray(MTT_per_voxel, dtype=float) if MTT_per_voxel is not None else None,
    )
    print(f"[pk_diag] Diagnostic voxel: x={vx}, y={vy}, z={vz}  "
          f"CBF={cbf_map[vx, vy, vz]:.2f} ml/100g/min")

    # Extract tissue curve for that voxel.
    voxel_signal = np.asarray(data_4d[vx, vy, vz, :], dtype=float)
    T1_vox = float(T1_matrix[vx, vy, vz])
    M0_vox = float(M0_matrix[vx, vy, vz])
    C_t_raw = compute_CTC_func(voxel_signal, T1_vox, m0=M0_vox)
    bp = baseline_func(C_t_raw)
    C_t = shift_func(C_t_raw, bp)

    common_len = min(int(data_4d.shape[3]), len(C_a_full), len(time_points_s))
    C_t_voxel = np.asarray(C_t[:common_len], dtype=float)
    C_a_voxel = np.asarray(C_a_full[:common_len], dtype=float)
    t_voxel = np.asarray(time_points_s[:common_len], dtype=float)

    if C_t_voxel.size < 10:
        print("[pk_diag] Tissue curve too short — skipping.")
        return

    # ── Tikhonov diagnostic solve ──
    diag = solve_single_voxel_diagnostic(t_voxel, C_a_voxel, C_t_voxel)

    # ── Patlak diagnostic fit ──
    patlak_data = None
    _single_bolus = int(getattr(settings, "NUMBER_OF_PEAKS", 2)) == 1
    try:
        pfit = _fit_patlak_diagnostic(
            C_t_voxel, C_a_voxel, t_voxel,
            single_bolus=_single_bolus,
        )
        patlak_data = {
            "ki": float(pfit.ki_ml_per_100g_min),
            "vp": float(pfit.vp_ml_per_100g),
            "sd_ki": float(pfit.sd_ki_ml_per_100g_min),
            "x_patlak": np.asarray(pfit.x_patlak, dtype=float),
            "y_patlak": np.asarray(pfit.y_patlak, dtype=float),
            "good_mask": np.asarray(pfit.good_mask, dtype=bool),
        }
    except Exception as _pe:
        print(f"[pk_diag] Patlak fit failed for diagnostic voxel: {_pe}")

    # ── 2-D map slice for the top-left panel ──
    map_2d = None
    if CTH_per_voxel is not None:
        map_slice = np.asarray(CTH_per_voxel[:, :, vz], dtype=float)
        if np.any(np.isfinite(map_slice)):
            map_2d = map_slice

    # ── Write the plot ──
    save_path = os.path.join(analysis_directory, "pk_diagnostic_grid.png")
    write_pk_diagnostic_grid(
        diag,
        patlak=patlak_data,
        map_2d=map_2d,
        voxel_xy=(vx, vy),
        map_label="CTH [s]",
        save_path=save_path,
    )

        
def plot_Ki_overlay(dce_slice, Ki_slice, slice_idx, save_path, vmin, vmax):
    """
    Plots the DCE image slice with an overlay of K_i values.

    Parameters:
    - dce_slice: 2D numpy array of the DCE image at a specific time point.
    - Ki_slice: 2D numpy array of K_i values for the slice.
    - slice_idx: Integer indicating the slice number.
    - save_path: Path to save the overlay image.
    - vmin: Global minimum K_i value for consistent color scaling.
    - vmax: Global maximum K_i value for consistent color scaling.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy.ma as ma

    # Mask out NaN values in K_i
    Ki_masked = ma.masked_invalid(Ki_slice)

    # Set up the figure and axis
    plt.figure(figsize=(8, 8))
    plt.imshow(np.rot90(dce_slice), cmap='gray', interpolation='nearest')

    # Overlay K_i values using a colormap
    plt.imshow(np.rot90(Ki_masked), cmap='jet', interpolation='nearest', alpha=0.6,
               norm=Normalize(vmin=vmin, vmax=vmax))

    plt.colorbar(label='K$_i$ (ml/100g/min)')
    plt.title(f'Blood Brain Barrier permeability (K$_i$) per voxel - Slice {slice_idx}')
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_CBF_overlay(dce_slice, CBF_slice, slice_idx, save_path, vmin, vmax):
    """
    Plots the DCE image slice with an overlay of CBF values.

    Parameters:
    - dce_slice: 2D numpy array of the DCE image at a specific time point.
    - CBF_slice: 2D numpy array of CBF values for the slice.
    - slice_idx: Integer indicating the slice number.
    - save_path: Path to save the overlay image.
    - vmin: Global minimum CBF value for consistent color scaling.
    - vmax: Global maximum CBF value for consistent color scaling.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy.ma as ma

    # Mask out NaN values in CBF
    CBF_masked = ma.masked_invalid(CBF_slice)

    # Set up the figure and axis
    plt.figure(figsize=(8, 8))
    plt.imshow(np.rot90(dce_slice), cmap='gray', interpolation='nearest')

    # Overlay CBF values using a colormap
    plt.imshow(np.rot90(CBF_masked), cmap='viridis', interpolation='nearest', alpha=0.6,
               norm=Normalize(vmin=vmin, vmax=vmax))

    plt.colorbar(label='CBF (ml/100g/min)')
    plt.title(f'Cerebral Blood Flow (CBF) per voxel - Slice {slice_idx}')
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()


def _rename_model_outputs(analysis_directory, image_directory, suffix, boundary=False):
    """Deprecated.

    Model-specific suffixing created invalid cross-model names (e.g. Ki_tikhonov,
    cbf_patlak). Outputs are now written using canonical filenames:
    - Patlak: Ki / vp
    - Tikhonov: CBF / MTT / CTH
    """
    return



def _tissue_function_AI(model, analysis_directory, nifti_directory, image_directory, filenames, parameters):
    """Run tissue function analysis for a single kinetic model."""
    settings.KINETIC_MODEL = model
    (
        t1_3D_filename,
        axial_t1_3D_filename,
        t2_3D_filename,
        axial_t2_3D_filename,
        flair_3D_filename,
        axial_flair_3D_filename,
        axial_t2_2D_filename,
        diffusion_filename,
        dce_filename,
    ) = filenames

    IsVFA, IsIR, apple_metal, boundary, RERUN_SEGMENTATION, SEGMENTATION_METHOD, _ = parameters

    # Automatically enable jump correction when requested via a JSON file
    global correct_signal_jumps
    jumpfix_file = os.path.join(os.path.dirname(analysis_directory), 'apply_jumpfix.json')
    if os.path.exists(jumpfix_file):
        print('[!] apply_jumpfix.json detected – enabling signal jump correction')
        correct_signal_jumps = True

    # Allow optional FLIRT-based coregistration via an environment variable
    global use_flirt_registration
    if os.getenv('USE_FLIRT_REGISTRATION') == '1':
        print('[!] USE_FLIRT_REGISTRATION=1 – enabling FLIRT-based coregistration')
        use_flirt_registration = True

    fastsurfer_path = '/Users/edt/FastSurfer/run_fastsurfer.sh'
    t1_path = os.path.join(nifti_directory, t1_3D_filename) if t1_3D_filename else None
    seg_dir = os.path.join(nifti_directory, 'segmentation')
    sid = 'segmentation'  # Define the subject ID
    seg_mgz_path = os.path.join(seg_dir, sid, 'mri', 'aparc.DKTatlas+aseg.deep.mgz')
    t2_path = os.path.join(nifti_directory, axial_t2_2D_filename) if axial_t2_2D_filename else None
    dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else None
    if not dce_path or not os.path.exists(dce_path):
        raise RuntimeError('Missing DCE NIfTI. Ensure DCE is present and imported.')
    flip_angle_deg = resolve_flip_angle_deg(dce_path, default=None)

    def _looks_like_structural_t1(path: str) -> bool:
        try:
            name = os.path.basename(path).lower()
            parts = [p.lower() for p in os.path.normpath(path).split(os.sep) if p]
        except Exception:
            return True

        deny_tokens = (
            't1_map', 't1map', 'voxel_t1', 't1_matrix', 't1matrix', 't1_fit', 't1fit',
            'm0', 'ki_', 'vp_', 'cbf_', 'mtt_', 'cth_',
            'in_dce',
        )
        if any(tok in name for tok in deny_tokens):
            return False
        if 'dce' in name:
            return False
        if any(p in {'analysis', 'fitting'} for p in parts):
            return False
        if any(tok in name for tok in ('t1w', 'mprage', 'spgr', 'bravo', 'tfl')):
            return True
        if any(p in {'anat', 'anatomy'} for p in parts):
            return True
        return True

    force_voxelwise_only = (os.environ.get('PBRAIN_VOXELWISE_ONLY') or '').strip().lower() in {'1', 'true', 'yes'}
    has_struct_t1 = bool(t1_path and os.path.exists(t1_path) and _looks_like_structural_t1(t1_path))
    has_t2 = bool(t2_path and os.path.exists(t2_path))
    voxelwise_only = force_voxelwise_only or (not has_struct_t1)

    # Ensure segmentation directory exists
    os.makedirs(seg_dir, exist_ok=True)

    if not voxelwise_only and SEGMENTATION_METHOD == 'pbrain':
        # ---- pbrain lightweight tissue model path ----
        from modules.pbrain_segmentation import run_pbrain_segmentation
        print('[!] Running pbrain tissue segmentation (no FreeSurfer needed)')
        (wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce,
         subcortical_gm_mask_t2, subcortical_gm_mask_dce, gm_brainstem_mask_t2,
         gm_brainstem_mask_dce, gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
         wm_cerebellum_mask_t2, wm_cerebellum_mask_dce, wm_cc_mask_t2, wm_cc_mask_dce) = run_pbrain_segmentation(
            t1_path=t1_path,
            dce_path=dce_path,
            t2_path=t2_path,
            seg_dir=seg_dir,
        )
        # pbrain handles registration internally — skip segmentation() + coregistration()

        if has_t2:
            t2_img = nib.load(t2_path).get_fdata()
            plot_predictions_with_masks(t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
                                        gm_brainstem_mask_t2, gm_cerebellum_mask_t2, wm_cerebellum_mask_t2,
                                        wm_cc_mask_t2, image_directory)
    elif not voxelwise_only:
        # ---- FastSurfer / SynthSeg / recon-all path ----
        segmentation(
            fastsurfer_path,
            seg_mgz_path,
            t1_path,
            seg_dir,
            sid,
            apple_metal,
            RERUN_SEGMENTATION,
            SEGMENTATION_METHOD,
        )

    # Paths to masks in the same directory as aparc.DKTatlas+aseg.deep.mgz
    mask_dir = os.path.dirname(seg_mgz_path)
    cortical_gm_mask_path = os.path.join(mask_dir, 'cortical_gm.nii.gz')
    subcortical_gm_mask_path = os.path.join(mask_dir, 'subcortical_gm.nii.gz')
    wm_mask_path = os.path.join(mask_dir, 'wm.nii.gz')

    if not voxelwise_only and SEGMENTATION_METHOD != 'pbrain' and has_t2:
        print('[!] Coregistering GM/WM masks onto T2 and DCE space')
        (wm_mask_t2, wm_mask_dce, cortical_gm_mask_t2, cortical_gm_mask_dce,
         subcortical_gm_mask_t2, subcortical_gm_mask_dce, gm_brainstem_mask_t2,
         gm_brainstem_mask_dce, gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
         wm_cerebellum_mask_t2, wm_cerebellum_mask_dce, wm_cc_mask_t2, wm_cc_mask_dce) = coregistration(
            seg_mgz_path=seg_mgz_path,
            dce_path=dce_path,
            t2_path=t2_path
        )

        # Load the T2 image for visualization
        t2_img = nib.load(t2_path).get_fdata()

        # Plot the predictions with gray and white matter masks on T2
        plot_predictions_with_masks(t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
                                    gm_brainstem_mask_t2, gm_cerebellum_mask_t2, wm_cerebellum_mask_t2,
                                    wm_cc_mask_t2, image_directory)

    # Continue with the rest of your processing, ensuring to include the new masks in your analysis and plotting

    # Load the DCE 4D data (consistent scaling; prefer complex magnitude when available)
    ref_img, data_4d = load_dce_4d(dce_path, prefer_complex_mag=True, dtype=np.float32)
    data_4d = np.asarray(data_4d)
    ref_affine = ref_img.affine
    ref_header = ref_img.header.copy()

    # Load T1 and M0 matrices
    T1_matrix = load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl'))
    M0_matrix = load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl'))

    # Resolve time axis (prefer previously generated time_points_s.npy).
    time_path = os.path.join(analysis_directory, 'Fitting', 'time_points_s.npy')
    time_points_s = None
    if os.path.isfile(time_path):
        try:
            time_points_s = np.load(time_path)
        except Exception:
            time_points_s = None

    if time_points_s is None:
        num_volumes = data_4d.shape[-1]
        dt_s = resolve_dce_time_step_s(dce_path, default=None)
        time_points_s = build_time_points_s(num_volumes, dt_s)
        try:
            os.makedirs(os.path.dirname(time_path), exist_ok=True)
            np.save(time_path, time_points_s)
        except Exception:
            pass

    # Decide which outputs to compute for this run.
    model_norm = (model or '').strip().lower()
    # Support "+" combos (e.g. "patlak+etofts", "extended_tofts+tikhonov")
    _model_parts = set(model_norm.split("+")) if "+" in model_norm else {model_norm}
    compute_ki = bool({"patlak", "both", "all"} & _model_parts)
    compute_cbf = bool({"tikhonov", "both", "all"} & _model_parts)
    compute_etofts = bool({"extended_tofts", "etofts", "etm", "tofts", "all"} & _model_parts)

    # Run CTC + modelling.
    if voxelwise_only or not has_t2:
        compute_and_plot_ctcs_median(
            data_4d,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            T1_matrix,
            M0_matrix,
            analysis_directory,
            time_points_s,
            image_directory,
            dce_path=dce_path,
            ref_affine=ref_affine,
            ref_header=ref_header,
            boundary=False,
            compute_per_voxel_Ki=compute_ki,
            compute_per_voxel_CBF=compute_cbf,
            compute_per_voxel_ETofts=compute_etofts,
            flip_angle_deg=flip_angle_deg,
            voxelwise_only=True,
        )
    else:
        compute_and_plot_ctcs_median(
            data_4d, t2_img, wm_mask_t2, cortical_gm_mask_t2, subcortical_gm_mask_t2,
            wm_mask_dce, cortical_gm_mask_dce, subcortical_gm_mask_dce,
            T1_matrix, M0_matrix, analysis_directory, time_points_s, image_directory,
            dce_path=dce_path, ref_affine=ref_affine, ref_header=ref_header,
            boundary=boundary,
            compute_per_voxel_Ki=compute_ki,
            compute_per_voxel_CBF=compute_cbf,
            compute_per_voxel_ETofts=compute_etofts,
            gm_brainstem_mask_t2=gm_brainstem_mask_t2, gm_brainstem_mask_dce=gm_brainstem_mask_dce,
            gm_cerebellum_mask_t2=gm_cerebellum_mask_t2, gm_cerebellum_mask_dce=gm_cerebellum_mask_dce,
            wm_cerebellum_mask_t2=wm_cerebellum_mask_t2, wm_cerebellum_mask_dce=wm_cerebellum_mask_dce,
            wm_cc_mask_t2=wm_cc_mask_t2, wm_cc_mask_dce=wm_cc_mask_dce,
            flip_angle_deg=flip_angle_deg,
        )

    if not voxelwise_only and (compute_ki or compute_cbf) and SEGMENTATION_METHOD != 'pbrain':
        # The atlas is the segmentation in DCE space
        # (pbrain produces 7 tissue classes, not a full parcellation atlas,
        #  so per-parcel Ki/CBF is skipped for the pbrain method.)
        atlas_path = os.path.join(
            nifti_directory,
            'segmentation',
            'segmentation',
            'mri',
            'aparc.DKTatlas+aseg.deep_in_DCE.nii.gz'
        )

        try:
            C_a_full, input_metadata = get_input_function_curve(analysis_directory)
        except (FileNotFoundError, ValueError) as exc:
            print(f'[!] No valid input function for atlas Ki — skipping: {exc}')
            C_a_full = None

        if C_a_full is None:
            return

        output_dir = analysis_directory

        ctc_model = (getattr(settings, "CTC_MODEL", "saturation") or "saturation").strip().lower()
        tr_s = None
        nph = None
        td_ms = 120
        if ctc_model in {"turboflash", "advanced"}:
            ti_s = resolve_turboflash_ti_s(dce_path, default=0.12)
            td_ms = float(ti_s) * 1e3
            # Platform standard conversion is `turboflash_advanced` (MATLAB menu_5 case12 method1).
            # It does not require the excitation TR or nph.

        compute_CTC_meta = functools.partial(
            compute_CTC,
            TD=td_ms,
            flip_angle_deg=flip_angle_deg,
            ctc_model=ctc_model,
            tr_s=tr_s,
            nph=nph,
        )

        compute_Ki_from_atlas(
            atlas_path=atlas_path,
            data_4d=data_4d,
            T1_matrix=T1_matrix,
            M0_matrix=M0_matrix,
            time_points_s=time_points_s,
            C_a_full=C_a_full,
            affine=nib.load(dce_path).affine,
            output_directory=output_dir,
            compute_CTC=compute_CTC_meta,
            find_baseline_point_advanced=find_baseline_point_advanced,
            custom_shifter=custom_shifter,
            patlak_analysis_plotting=patlak_analysis_plotting,
            compute_ki=compute_ki,
            compute_cbf=compute_cbf,
        )

    # Do not rename outputs by model; filenames are canonical.


def tissue_function_AI(
    analysis_directory,
    nifti_directory,
    image_directory,
    filenames,
    parameters,
    *,
    compute_diffusion: bool = False,
):
    """Run tissue function analysis using the configured kinetic model."""
    model_setting = (settings.KINETIC_MODEL or 'patlak').strip().lower()
    print(f'[!] Running model: {model_setting}')
    _tissue_function_AI(
        model_setting,
        analysis_directory,
        nifti_directory,
        image_directory,
        filenames,
        parameters,
    )

    if compute_diffusion:
        diffusion_filename = filenames[-2] if filenames else None
        if diffusion_filename:
            try:
                from . import opt08_fa
            except ImportError as exc:  # pragma: no cover - import error is user-facing
                print(f"[diffusion] Unable to import diffusion workflow: {exc}")
            else:
                dce_filename = filenames[-1] if filenames else None
                dce_path = (
                    os.path.join(nifti_directory, dce_filename)
                    if dce_filename
                    else None
                )
                try:
                    print("[diffusion] Computing diffusion metrics")
                    opt08_fa.compute_fa(
                        nifti_directory,
                        analysis_directory,
                        image_directory,
                        diffusion_filename=diffusion_filename,
                        dce_path=dce_path,
                    )
                except Exception as exc:  # noqa: BLE001 - expose runtime issues
                    print(f"[diffusion] Failed to compute diffusion metrics: {exc}")
        else:
            print("[diffusion] No diffusion filename configured; skipping diffusion metrics.")

    dce_filename = filenames[-1] if filenames else None
    if dce_filename:
        dce_path = os.path.join(nifti_directory, dce_filename)
        if os.path.exists(dce_path):
            try:
                segmentation_path = os.path.join(
                    nifti_directory,
                    "segmentation",
                    "segmentation",
                    "mri",
                    "aparc.DKTatlas+aseg.deep_in_DCE.nii.gz",
                )
                if not os.path.isfile(segmentation_path):
                    segmentation_path = None
                from utils.montage import generate_parametric_montages
                generate_parametric_montages(
                    analysis_directory,
                    image_directory,
                    dce_path,
                    segmentation_path=segmentation_path,
                )
            except Exception as exc:
                print(f"[montage] Unexpected error during montage rendering: {exc}")
        else:
            print(f"[montage] DCE file missing; skipping montage rendering: {dce_path}")
    else:
        print('[montage] No DCE filename available; skipping montage rendering.')

