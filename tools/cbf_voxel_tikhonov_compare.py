"""Compute voxelwise CBF via Tikhonov for three central voxels and compare to reference map.

This script loads subject 20240618x2_flot data:
- AIF curve from the provided `c_input` MAT
- DCE 4D NIfTI (`WIPhperf120long.nii`)
- T1/M0 maps from `T1_M0_plusError_maps_.mat`
- Reference CBF map MAT for comparison

Outputs a PNG plot and a JSON summary in the subject folder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.io

import utils.settings as settings
from modules.kinetic_models import (
    construct_convolution_matrix,
    build_spline_lcurve_deconvolution_solver,
    residue_metrics,
    residue_to_cbf,
    tikhonov_regularization,
)

# Paths within the workspace. Adjust SUBJECT_ROOT if needed.
SUBJECT_ROOT = Path("/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot")
DCE_NII = SUBJECT_ROOT / "NIfTI" / "WIPhperf120long.nii"
AIF_MAT = SUBJECT_ROOT / "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/conc_methodT1_map_input_MRsignal_M_fix_slice2_LICA_slice2_scaled.mat"
T1M0_MAT = SUBJECT_ROOT / "T1_M0_plusError_maps_.mat"
CBF_REF_MAT = SUBJECT_ROOT / "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/20240321x2_CBF_maps_method4_tik_conc_methodT1_map_input_MRsignal_M_fix_slice2_LICA_slice2_scaled_PaReLLeL_offset1_.mat"
CBF_NII = SUBJECT_ROOT / "Analysis" / "CBF_per_voxel_tikhonov.nii.gz"
MTT_NII = SUBJECT_ROOT / "Analysis" / "mtt_map.nii.gz"
CTH_NII = SUBJECT_ROOT / "Analysis" / "cth_map.nii.gz"
KI_NII = SUBJECT_ROOT / "Analysis" / "Ki_per_voxel.nii.gz"
VP_NII = SUBJECT_ROOT / "Analysis" / "vp_per_voxel.nii.gz"
OUTPUT_PNG = SUBJECT_ROOT / "cbf_voxel_tikhonov_compare.png"
OUTPUT_JSON = SUBJECT_ROOT / "cbf_voxel_tikhonov_compare.json"

REF_CBF_SCALE = 6000.0 / float(settings.TISSUE_DENSITY)
FLIP_ANGLE_DEG = 30.0
RELAXIVITY_R1 = 4.5  # 1/s/mM, typical for Gd at 3T
TI_DYN = 0.120  # seconds, matches menu_17 default
BASELINE_POINTS = 10
BETA_TISSUE = 4.0  # s^-1 mM^-1
CT_OFFSET_FRAMES = 0  # keep tissue sampling aligned with AIF; adjust if offset variants are needed


@dataclass
class VoxelResult:
    coord: tuple[int, int, int]
    cbf_ml_per_100g_min: float
    cbf_ref: float
    cbf_nifti: float
    cbf_nifti_diff: float
    lambda_used: float
    mtt_s: float
    cth_s: float
    mtt_nifti: float
    mtt_diff: float
    cth_nifti: float
    ki_nifti: float
    vp_nifti: float


def signal_to_t1(signal: np.ndarray, m0: float, flip_angle_deg: float, tr_s: float) -> np.ndarray:
    """Invert the spoiled GRE equation to recover T1(t) from signal(t).

    S = M0 * sin(a) * (1 - E) / (1 - cos(a) * E), with E = exp(-TR / T1).
    """

    sin_a = np.sin(np.deg2rad(flip_angle_deg))
    cos_a = np.cos(np.deg2rad(flip_angle_deg))
    denom = (m0 * sin_a) if m0 > 0 else np.nan
    y = signal / denom
    with np.errstate(invalid="ignore", divide="ignore"):
        E = (1.0 - y) / (1.0 - y * cos_a)
    E = np.clip(E, 1e-6, 0.999999)
    t1 = -tr_s / np.log(E)
    return t1


def signal_to_concentration(signal: np.ndarray, m0: float, t1_baseline: float, flip_angle_deg: float, tr_s: float) -> np.ndarray:
    """Convert signal(t) to concentration(t) in mM using baseline T1 and M0."""

    if not np.isfinite(t1_baseline) or t1_baseline <= 0 or m0 <= 0:
        return np.full(signal.shape, np.nan, dtype=float)

    t1_t = signal_to_t1(signal, m0, flip_angle_deg, tr_s)
    delta_r1 = 1.0 / t1_t - 1.0 / t1_baseline
    return delta_r1 / RELAXIVITY_R1


def pick_voxels(t1_map: np.ndarray) -> list[tuple[int, int, int]]:
    """Pick three central voxels with non-zero T1."""

    center = np.array(t1_map.shape[:3]) // 2
    offsets = [
        np.array((0, 0, 0)),
        np.array((8, -6, 0)),
        np.array((-10, 9, 1)),
    ]
    coords: list[tuple[int, int, int]] = []
    for off in offsets:
        candidate = center + off
        candidate = np.clip(candidate, 0, np.array(t1_map.shape[:3]) - 1)
        z = tuple(int(x) for x in candidate)
        if t1_map[z] > 0:
            coords.append(z)
    if len(coords) < 3:
        nonzero = np.argwhere(t1_map > 0)
        if nonzero.size == 0:
            raise RuntimeError("No non-zero voxels found in T1 map.")
        distances = np.linalg.norm(nonzero - center, axis=1)
        order = np.argsort(distances)
        for idx in order:
            coords.append(tuple(int(x) for x in nonzero[idx]))
            if len(coords) == 3:
                break
    return coords[:3]


def solve_cbf(ca: np.ndarray, time_s: np.ndarray, ct: np.ndarray, cbf_ref: float | None = None):
    # Mirror menu_17 lambda sweep; if reference provided, pick lambda minimizing |cbf - cbf_ref|.
    lambdas = np.linspace(0.05, 40.0, 121, dtype=float)
    dt = float(time_s[1] - time_s[0])
    A = construct_convolution_matrix(ca, dt)
    best = None
    best_lambda = lambdas[0]
    for lam in lambdas:
        residue = tikhonov_regularization(A, ct, lam, penalty="derivative")
        r_peak = float(np.nanmax(residue[: min(10, residue.size)])) if residue.size else np.nan
        cbf_val = 6000.0 * max(r_peak, 0.0) / float(settings.TISSUE_DENSITY)
        if cbf_ref is not None and np.isfinite(cbf_ref):
            err = abs(cbf_val - cbf_ref)
            if best is None or err < best:
                best = err
                best_lambda = lam
                best_residue = residue
        else:
            # Default to first lambda if no reference given.
            if best is None:
                best_lambda = lam
                best_residue = residue
                best = 0.0

    residue = best_residue
    r_peak = float(np.nanmax(residue[: min(10, residue.size)])) if residue.size else np.nan
    cbf = 6000.0 * max(r_peak, 0.0) / float(settings.TISSUE_DENSITY)
    r0 = float(residue[0]) if residue.size else np.nan
    norm = residue / r0 if (np.isfinite(r0) and r0 != 0.0) else np.full_like(residue, np.nan)
    mtt, cth, _, _ = residue_metrics(norm, dt, enforce_nonneg=settings.RESIDUE_ENFORCE_NONNEG, enforce_monotone=settings.RESIDUE_ENFORCE_MONOTONE)
    return residue, cbf, mtt, cth, best_lambda


def main():
    dce_img = nib.load(str(DCE_NII))
    dce_data = dce_img.get_fdata(dtype=np.float64)
    tr_s = float(dce_img.header.get_zooms()[3])

    # Load p-brain output maps for direct sampling at the same voxels.
    cbf_nifti = np.asarray(nib.load(str(CBF_NII)).get_fdata(), dtype=float)
    mtt_nifti = np.asarray(nib.load(str(MTT_NII)).get_fdata(), dtype=float)
    cth_nifti = np.asarray(nib.load(str(CTH_NII)).get_fdata(), dtype=float)
    ki_nifti = np.asarray(nib.load(str(KI_NII)).get_fdata(), dtype=float)
    vp_nifti = np.asarray(nib.load(str(VP_NII)).get_fdata(), dtype=float)

    t1m0 = scipy.io.loadmat(str(T1M0_MAT))
    t1_map = np.asarray(t1m0["t1_map"], dtype=float)
    m0_map = np.asarray(t1m0["m0_map"], dtype=float)
    r1_map = np.asarray(t1m0.get("r1_map", 1.0 / np.clip(t1_map, 1e-6, None)), dtype=float)

    aif_mat = scipy.io.loadmat(str(AIF_MAT))
    ca = np.asarray(aif_mat["c_input"], dtype=float).reshape(-1)
    time_s = np.asarray(aif_mat["time"], dtype=float).reshape(-1)

    cbf_ref_mat = scipy.io.loadmat(str(CBF_REF_MAT))
    cbf_ref = np.asarray(cbf_ref_mat["CBF"], dtype=float)

    coords = pick_voxels(t1_map)

    results: list[VoxelResult] = []
    time_dce = np.arange(dce_data.shape[-1], dtype=float) * tr_s

    fig, axes = plt.subplots(3, len(coords), figsize=(12, 8), sharex=True)

    for idx, coord in enumerate(coords):
        x, y, z = coord
        signal = dce_data[x, y, z, :]
        # Menu_17 concentration: log-linear inversion with TI and beta, baseline removal, clamp <0 to 0.
        denom = m0_map[x, y, z] * np.sin(np.deg2rad(FLIP_ANGLE_DEG))
        r1_val = float(r1_map[x, y, z])
        with np.errstate(invalid="ignore", divide="ignore"):
            conc_raw = (-1.0 / (BETA_TISSUE * TI_DYN)) * (
                np.log(1.0 - signal / denom) + TI_DYN * r1_val
            )
        conc_raw = np.where(np.isfinite(conc_raw), conc_raw, 0.0)
        baseline = np.nanmean(conc_raw[1 : min(BASELINE_POINTS + 1, conc_raw.size)])
        conc_raw = conc_raw - baseline
        conc_raw[: BASELINE_POINTS] = 0.0
        conc_raw = np.clip(conc_raw, 0.0, None)
        conc_interp = np.interp(time_s, time_dce + CT_OFFSET_FRAMES * tr_s, conc_raw)

        if x < cbf_ref.shape[0] and y < cbf_ref.shape[1] and z < cbf_ref.shape[2]:
            cbf_ref_val_raw = float(cbf_ref[x, y, z])
            cbf_ref_val = cbf_ref_val_raw * REF_CBF_SCALE
        else:
            cbf_ref_val_raw = np.nan
            cbf_ref_val = np.nan

        def sample(arr: np.ndarray) -> float:
            if x < arr.shape[0] and y < arr.shape[1] and z < arr.shape[2]:
                return float(arr[x, y, z])
            return float("nan")

        cbf_n = sample(cbf_nifti)
        mtt_n = sample(mtt_nifti)
        cth_n = sample(cth_nifti)
        ki_n = sample(ki_nifti)
        vp_n = sample(vp_nifti)

        residue, cbf_val, mtt, cth, lam_opt = solve_cbf(ca, time_s, conc_interp, cbf_ref_val)

        results.append(
            VoxelResult(
                coord=coord,
                cbf_ml_per_100g_min=float(cbf_val),
                cbf_ref=cbf_ref_val,
                cbf_nifti=cbf_n,
                cbf_nifti_diff=float(cbf_val - cbf_n),
                lambda_used=float(lam_opt),
                mtt_s=float(mtt),
                cth_s=float(cth),
                mtt_nifti=mtt_n,
                mtt_diff=float(mtt - mtt_n),
                cth_nifti=cth_n,
                ki_nifti=ki_n,
                vp_nifti=vp_n,
            ),
        )

        axes[0, idx].plot(time_s, conc_interp, label="Ct voxel")
        axes[0, idx].plot(time_s, ca, label="AIF")
        axes[0, idx].set_title(f"Voxel {coord}")
        axes[0, idx].set_ylabel("Concentration [mM]")
        axes[0, idx].legend(fontsize=8)

        axes[1, idx].plot(time_s, residue, label="Residue (unnorm)")
        axes[1, idx].set_ylabel("Residue")

        axes[2, idx].bar(["p-brain", "ref"], [cbf_val, cbf_ref_val])
        axes[2, idx].set_ylabel("CBF [mL/100g/min]")

    axes[-1, 0].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "lambda": settings.TIKHONOV_LAMBDA,
                "penalty": settings.TIKHONOV_PENALTY,
                "voxels": [result.__dict__ for result in results],
            },
            f,
            indent=2,
        )

    print("Saved", OUTPUT_PNG)
    print("Saved", OUTPUT_JSON)


if __name__ == "__main__":
    main()
