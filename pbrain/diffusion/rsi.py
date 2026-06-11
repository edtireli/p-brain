"""RSI — Restriction Spectrum Imaging (3-compartment).

White 2013, Reynaud 2017. Decomposes the multi-shell DWI signal into
isotropic compartments with fixed apparent diffusion coefficients:

* **restricted**  D ≈ 0          mm²/s — intracellular / membrane-bound water
* **hindered**    D ≈ 1.0 × 10⁻³ mm²/s — extracellular tissue water
* **free**        D ≈ 3.0 × 10⁻³ mm²/s — CSF / free water

Spherical-mean variant: the per-direction signal is averaged across
gradient directions within each shell to remove orientation dependence
(SMT-style). Per voxel the three fractions are solved by vectorised
constrained least-squares with the sum-to-1 constraint enforced as a
penalty row, then clipped to [0, 1] and renormalised.

Requires multi-shell DWI with at least one high-b shell (b ≥ 1500 s/mm²).

Outputs:

* ``restricted_fraction``  — sensitive to dense cellular tissue
* ``hindered_fraction``    — tissue extracellular space
* ``free_fraction``        — CSF, oedema, vasogenic free water

The restricted fraction is sensitive to subtle white-matter changes that
DTI FA can miss (Wu et al. 2018; Bartnik-Olson et al. 2014).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import DWIInputs, DiffusionResult


@dataclass(frozen=True, slots=True)
class _RSI:
    key:         ClassVar[str] = "rsi"
    name:        ClassVar[str] = "Restriction Spectrum Imaging (3-comp)"
    description: ClassVar[str] = (
        "Multi-shell spherical-mean decomposition into restricted / "
        "hindered / free isotropic compartments (White 2013). Needs "
        "multi-shell DWI with a high-b shell (b ≥ 1500 s/mm²); e.g. "
        "1×b=0 + 6×b=500 + 6×b=1500 + 15×b=4000. The restricted "
        "fraction is sensitive to subtle white-matter changes that "
        "DTI FA can miss."
    )
    accepts:  ClassVar[dict[str, type]] = {
        "signal": np.ndarray, "bvals": np.ndarray, "bvecs": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "restricted_fraction": np.ndarray,
        "hindered_fraction": np.ndarray,
        "free_fraction": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = (
        "restricted_fraction", "hindered_fraction", "free_fraction",
    )
    units: ClassVar[dict[str, str]] = {
        "restricted_fraction": "(fraction)",
        "hindered_fraction":   "(fraction)",
        "free_fraction":       "(fraction)",
    }
    primary_map: ClassVar[str] = "restricted_fraction"

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        D_R = float(opts.get("D_restricted", 0.1e-3))   # mm²/s (small but not 0 for cond.)
        D_H = float(opts.get("D_hindered",   1.0e-3))
        D_F = float(opts.get("D_free",       3.0e-3))
        b0_thr = float(opts.get("b0_threshold", 50.0))
        sum_lam = float(opts.get("sum_constraint_weight", 100.0))

        signal = np.asarray(inputs.signal, dtype=np.float32)
        bvals = np.asarray(inputs.bvals, dtype=float)
        if signal.shape[-1] != bvals.size:
            raise ValueError(
                f"signal directions {signal.shape[-1]} ≠ bvals {bvals.size}"
            )

        # Group into shells (round to nearest 50 s/mm²)
        b_rounded = (np.round(bvals / 50.0) * 50.0).astype(int)
        shells_all = np.unique(b_rounded)
        b0_mask = b_rounded <= b0_thr
        nonzero_shells = [int(b) for b in shells_all if b > b0_thr]
        if len(nonzero_shells) < 2:
            raise RuntimeError(
                f"RSI needs ≥2 non-zero shells; got {nonzero_shells}. "
                f"Single-shell data is incompatible with the multi-exponential decay."
            )

        X, Y, Z, N = signal.shape

        # Spherical mean per shell
        S0 = signal[..., b0_mask].mean(axis=-1, keepdims=False)             # (X,Y,Z)
        S_sm = np.zeros((X, Y, Z, len(nonzero_shells)), dtype=np.float32)
        for i, b in enumerate(nonzero_shells):
            sel = b_rounded == b
            S_sm[..., i] = signal[..., sel].mean(axis=-1)

        # Normalise by S0 → ratios r_i = E[S(b_i)] / E[S(0)]
        with np.errstate(divide="ignore", invalid="ignore"):
            R = S_sm / np.where(S0[..., None] > 0, S0[..., None], 1.0)

        # Design matrix: A_ij = exp(-b_i · D_j) for j ∈ {restricted, hindered, free}
        b_arr = np.array(nonzero_shells, dtype=float)
        A = np.column_stack([
            np.exp(-b_arr * D_R),
            np.exp(-b_arr * D_H),
            np.exp(-b_arr * D_F),
        ])

        # Append a row enforcing sum = 1 with weight ``sum_lam``
        A_aug = np.vstack([A, sum_lam * np.ones(3)])
        # Equivalent ridge: (AᵀA + λ²·11ᵀ)⁻¹ Aᵀ
        ATA_inv_AT = np.linalg.solve(
            A_aug.T @ A_aug + 1e-9 * np.eye(3),
            A_aug.T,
        )                                                                    # (3, n_shells+1)

        # Per-voxel solve: append sum_lam to each voxel's RHS
        R_aug = np.concatenate(
            [R, np.full((X, Y, Z, 1), sum_lam, dtype=np.float32)], axis=-1
        )                                                                    # (X,Y,Z, n_shells+1)
        f = np.einsum("ij,xyzj->xyzi", ATA_inv_AT, R_aug)                    # (X,Y,Z,3)

        # Clip + renormalise to enforce the simplex constraint exactly
        f = np.clip(f, 0.0, None)
        f_sum = f.sum(axis=-1, keepdims=True)
        f = np.where(f_sum > 0, f / f_sum, 0.0)

        f_R = f[..., 0].astype(np.float32)
        f_H = f[..., 1].astype(np.float32)
        f_F = f[..., 2].astype(np.float32)

        if inputs.mask is not None:
            m = np.asarray(inputs.mask, dtype=bool)
            for arr in (f_R, f_H, f_F):
                arr[~m] = np.nan

        return DiffusionResult(
            maps={
                "restricted_fraction": f_R,
                "hindered_fraction":   f_H,
                "free_fraction":       f_F,
            },
            units=dict(self.units),
            aux={
                "D_restricted_mm2_per_s": D_R,
                "D_hindered_mm2_per_s":   D_H,
                "D_free_mm2_per_s":       D_F,
                "shells_used":            nonzero_shells,
                "sum_constraint_weight":  sum_lam,
            },
        )


PLUGIN = _RSI()
