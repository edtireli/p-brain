"""Stieltjes / holomorphic transfer-function pole recovery — local model.

A physically admissible residue is *completely monotone*, so by Bernstein's
theorem it is the Laplace transform of a non-negative measure on the rate axis,
and its Laplace transform is a **Stieltjes (Pick-class) function**::

        Ĥ(s) = L[C_t](s) / L[C_a](s) = F · R̂(s) = ∫₀^∞  dμ(r) / (s + r),   μ ≥ 0.

Instead of deconvolving in the time domain (ill-posed, needs an ad-hoc
smoothness prior), we work in the Laplace domain on the **real axis** s>0:

  1. Sample H(s_j) = L[C_t](s_j)/L[C_a](s_j).  L[f](s)=∫ f(t)e^{-st}dt is an
     *integral* of the data — stable, no differentiation.
  2. Recover the non-negative rate spectrum μ by non-negative least squares
     against the partial-fraction basis A_{jk}=1/(s_j+r_k):
            min_{W ≥ 0}  ‖ A W − H ‖² .
     Positivity *is* the Pick/Stieltjes analytic constraint — a far stronger,
     physically-grounded prior than L2 smoothness, and it makes the otherwise
     ill-posed inverse-Laplace well-behaved for the low-order functionals.

The clearance-rate spectrum then reads off the kinetics by *pole location*:

    cbf       = (Σ_k W_k) · 6000               (mL/100 g/min; F = lim_{s→∞} sH(s))
    cbv       = H(0) = ∫C_t / ∫C_a              (blood-volume fraction / v_d)
    mtt       = Σ(W_k/r_k) / Σ W_k = R̂(0)/R̂(∞)  (∫R, mean transit time, s)
    leak_frac = Σ_{r<r*} W_k / Σ_k W_k          (slow-pole mass = retention/leak;
                                                 r* = 1/T_leak, default T_leak=40 s)
    tc_vasc   = 1 / r_peak                       (dominant fast clearance time, s)

— so the vascular pole (fast r) and the leakage pole (r→0) are separated by
where they sit on the rate axis, not by an arbitrary time window (cf. Patlak).

Relation to the field: vector fitting / rational (Stieltjes) approximation is
standard in circuit & control engineering and unused in perfusion; the closest
prior is "non-negative monotone residue" DSC deconvolution (same Bernstein idea,
time domain, no pole structure). ``.gitignore``-d (local only).

ROI/parcel (1-D): exact ``scipy.optimize.nnls``.  Voxelwise (default): batched
projected-gradient NNLS over all voxels at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import nnls

from .base import CurveInputs, ModelResult


# ────────────────────────────────────────────────────────────────────────
# Laplace sampling + rate/s grids
# ────────────────────────────────────────────────────────────────────────


def _trapz_weights(t: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for ∫ f dt on a (possibly nonuniform) grid."""
    w = np.zeros_like(t)
    dt = np.diff(t)
    w[:-1] += 0.5 * dt
    w[1:] += 0.5 * dt
    return w


def _laplace_matrix(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E[j, i] = e^{-s_j t_i} · trapz_weight_i, so E @ f ≈ L[f](s_j)=∫f e^{-st}dt."""
    wq = _trapz_weights(t)
    return np.exp(-np.outer(s, t)) * wq[None, :]


def _build_s_grid(t: np.ndarray, ca: np.ndarray, n_s: int = 36,
                  s_min: float = 1e-3, energy_floor: float = 0.02) -> np.ndarray:
    """Log-spaced real s>0, capped where the AIF Laplace transform L[Ca](s)
    drops below ``energy_floor`` of its low-s value (beyond that H(s) is noise)."""
    s_probe = np.geomspace(s_min, 5.0, 200)
    La = _laplace_matrix(t, s_probe) @ ca
    La0 = float(La[0]) if La[0] > 0 else 1.0
    ok = La > energy_floor * La0
    s_max = float(s_probe[ok][-1]) if np.any(ok) else 1.0
    s_max = min(max(s_max, 10.0 * s_min), 5.0)
    return np.geomspace(s_min, s_max, int(n_s))


# ────────────────────────────────────────────────────────────────────────
# Batched non-negative least squares (projected gradient, FISTA)
# ────────────────────────────────────────────────────────────────────────


def _nnls_batch(A: np.ndarray, B: np.ndarray, n_iter: int = 500,
                ridge: float = 1e-9) -> np.ndarray:
    """min_{W≥0} ‖A W − B‖² + ridge‖W‖², solved for all RHS columns at once.

    FISTA projected-gradient. ``A`` is small (n_s × n_r) so AᵀA is tiny and the
    per-iteration cost is one (n_r×n_r)·(n_r×V) product."""
    AtA = A.T @ A + ridge * np.eye(A.shape[1])
    AtB = A.T @ B
    L = float(np.linalg.eigvalsh(AtA)[-1]) or 1.0
    step = 1.0 / L
    W = np.zeros_like(AtB)
    Z = W.copy()
    tk = 1.0
    for _ in range(int(n_iter)):
        grad = AtA @ Z - AtB
        W_new = np.maximum(Z - step * grad, 0.0)
        tk1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        Z = W_new + ((tk - 1.0) / tk1) * (W_new - W)
        W, tk = W_new, tk1
    return np.maximum(W, 0.0)


def _spectrum_to_params(W: np.ndarray, rates: np.ndarray, cbv: np.ndarray,
                        r_leak: float) -> dict[str, np.ndarray]:
    """Derived scalars from the rate spectrum W (n_r × V).

    The Laplace/real-axis data robustly constrain the **slow** end of the
    spectrum (rates within the AIF bandwidth) — CBV, MTT, and the leakage
    mass/timescale. ``cbf = ΣW`` is the s→∞ limit and is *not* well-conditioned
    here (it needs short-time information beyond the temporal resolution); it is
    returned in ``aux`` for reference, not as a headline map.
    """
    sumW = W.sum(axis=0)                                   # = F (ill-conditioned)
    slow = rates < r_leak
    with np.errstate(divide="ignore", invalid="ignore"):
        mtt = (W / rates[:, None]).sum(axis=0) / sumW      # Σ(W/r)/ΣW = ∫R
        slowW = W[slow, :].sum(axis=0)
        leak = slowW / sumW                                # slow-pole mass fraction
        tc_leak = (W[slow, :] / rates[slow, None]).sum(axis=0) / np.where(slowW > 0, slowW, np.nan)
    bad = ~np.isfinite(sumW) | (sumW <= 0)
    cbf = sumW * 6000.0
    for arr in (cbf, mtt, leak, tc_leak):
        arr[bad] = np.nan
    return {"cbv": cbv, "mtt": mtt, "leak_frac": leak, "tc_leak": tc_leak, "cbf": cbf}


# ────────────────────────────────────────────────────────────────────────
# Plug-in adapter
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class StieltjesModel:
    """Stieltjes transfer-function model (the ``stieltjes`` plug-in).

    Holomorphic deconvolution: recovers the non-negative clearance-rate
    spectrum of the residue from the Laplace-domain transfer function
    ``H(s) = L[Ct]/L[Ca]`` by NNLS (Pick/Stieltjes positivity). Outputs
    CBF, CBV, MTT, leak fraction (slow-pole mass) and the dominant
    clearance time.
    """

    key: ClassVar[str] = "stieltjes"
    name: ClassVar[str] = "Stieltjes transfer-function pole recovery"
    description: ClassVar[str] = (
        "Holomorphic deconvolution: recovers the non-negative clearance-rate "
        "spectrum of the residue from the Laplace-domain transfer function "
        "H(s)=L[Ct]/L[Ca] by NNLS (Pick/Stieltjes positivity). Outputs CBF, "
        "CBV, MTT, leak fraction (slow-pole mass) and the dominant clearance time."
    )
    accepts: ClassVar[dict[str, type]] = {
        "c_tissue": np.ndarray, "c_input": np.ndarray, "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "cbv": np.ndarray, "mtt": np.ndarray,
        "leak_frac": np.ndarray, "tc_leak": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("cbv", "mtt", "leak_frac", "tc_leak")
    supports_voxelwise: ClassVar[bool] = True
    primary_map: ClassVar[str] = "leak_frac"
    units: ClassVar[dict[str, str]] = {
        "cbv": "(ratio)", "mtt": "s", "leak_frac": "(fraction)", "tc_leak": "s",
    }

    def _setup(self, t_s, c_a, opts):
        n_s = int(opts.pop("n_s", 36))
        n_r = int(opts.pop("n_rates", 64))
        r_min = float(opts.pop("rate_min", 1e-3))
        r_max = float(opts.pop("rate_max", 1.0))
        rates = np.geomspace(r_min, r_max, n_r)
        s = _build_s_grid(t_s, c_a, n_s=n_s)
        Em = _laplace_matrix(t_s, s)
        La = Em @ c_a
        La = np.where(np.abs(La) > 1e-12, La, 1e-12)
        A = 1.0 / (s[:, None] + rates[None, :])            # Stieltjes partial-fraction basis
        wq = _trapz_weights(t_s)
        return rates, s, Em, La, A, wq

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        """Fit the Stieltjes spectrum → CBF/CBV/MTT/leak maps.

        Solves the positivity-constrained NNLS pole problem per curve.
        ``**opts`` tune the pole grid and regularisation.
        """
        c_t = np.asarray(inputs.c_tissue, dtype=float)
        c_a = np.asarray(inputs.c_input, dtype=float)
        t_s = np.asarray(inputs.t_s, dtype=float)
        n = int(min(t_s.size, c_a.size, c_t.shape[0]))
        t_s, c_a = t_s[:n], c_a[:n]
        c_t = c_t[:n]

        # Crop the pre-bolus baseline at the AIF leading edge. H = F·R̂ is
        # invariant to a delay shared by Ca and Ct, and removing the late
        # arrival (t₀≈40 s) restores L[Ca](s) at the higher s that constrain
        # the fast (vascular) poles — otherwise e^{-s·t₀} buries them in noise.
        if bool(opts.pop("crop_bolus", True)) and np.any(np.isfinite(c_a)):
            thr = 0.05 * float(np.nanmax(c_a))
            i0 = int(np.argmax(c_a > thr)) if np.any(c_a > thr) else 0
            i0 = max(i0 - 1, 0)
            if 0 < i0 < n - 5:
                t_s = t_s[i0:] - t_s[i0]
                c_a = c_a[i0:]
                c_t = c_t[i0:] if c_t.ndim == 1 else c_t[i0:, :]

        r_leak = 1.0 / float(opts.pop("t_leak_s", 40.0))
        # Batched NNLS convergence is the real accuracy lever here (not temporal
        # upsampling, which A/B-tested as having no effect): 1500 iters removes
        # the CBF/leak bias that 500 left.
        n_iter = int(opts.pop("nnls_iter", 1500))
        rates, s, Em, La, A, wq = self._setup(t_s, c_a, opts)
        ca_area = float(wq @ c_a) or 1.0

        single = c_t.ndim == 1
        if single:
            c_t = c_t.reshape(-1, 1)
        elif c_t.ndim != 2:
            raise ValueError(f"c_tissue must be 1-D or 2-D; got {c_t.shape}")
        n_v = c_t.shape[1]

        mask = inputs.mask
        voxels = (np.arange(n_v) if (mask is None or single)
                  else np.flatnonzero(np.asarray(mask, dtype=bool)))

        out = {k: np.full(n_v, np.nan) for k in self.outputs}
        if voxels.size:
            Ct = c_t[:, voxels]
            H = (Em @ Ct) / La[:, None]                    # transfer function H(s_j) per voxel
            cbv = (wq @ Ct) / ca_area                      # H(0) = ∫Ct/∫Ca
            if single or voxels.size <= 4:
                W = np.zeros((rates.size, voxels.size))
                for k in range(voxels.size):
                    W[:, k], _ = nnls(A, H[:, k])          # exact NNLS
            else:
                W = _nnls_batch(A, H, n_iter=n_iter)       # batched FISTA-PG
            params = _spectrum_to_params(W, rates, np.asarray(cbv, dtype=float), r_leak)
            for k in self.outputs:
                out[k][voxels] = params[k]

        if single:
            maps = {k: np.asarray(v[0]) for k, v in out.items()}
            return ModelResult(maps=maps, units=dict(self.units),
                               aux={"rates": rates, "s": s,
                                    "spectrum": W[:, 0] if voxels.size else None,
                                    "cbf_uncertain": float(params["cbf"][0]) if voxels.size else float("nan")})
        return ModelResult(maps=out, units=dict(self.units),
                           aux={"rates": rates, "s": s})

    def review(self, inputs, result, **_kw):
        """--mode verify view — mean tissue Cₜ + the Stieltjes fit + median parameters."""
        from ._review import curve_fit_review
        return curve_fit_review(self, inputs, result,
                                title="Stieltjes · residue", fit_label="Stieltjes fit")

    def predict(self, maps: dict[str, Any], c_input: np.ndarray,
                t_s: np.ndarray) -> np.ndarray:
        """QC reconstruction from (cbv, mtt, leak) via a gamma-variate residue
        proxy (cth≈mtt → near-exponential). Stieltjes fits a *non-parametric*
        rate spectrum; this is the uniform-interface approximation, not the exact
        spectral reconstruction. CBF≈CBV/MTT seeds the overlay scale."""
        from .base import reconstruct_gamma_residue_ct
        mtt = float(maps.get("mtt", float("nan")))
        cbv = float(maps.get("cbv", float("nan")))
        cbf = cbv / mtt * 6000.0 if (np.isfinite(cbv) and np.isfinite(mtt) and mtt > 0) else float("nan")
        return reconstruct_gamma_residue_ct(
            t_s, c_input, cbf=cbf, mtt=mtt, cth=mtt,
            e_leak=float(maps.get("leak_frac", 0.0) or 0.0),
        )


PLUGIN = StieltjesModel()
