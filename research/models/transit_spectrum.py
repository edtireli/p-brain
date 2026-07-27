"""Transit-time spectrum — a de novo, shape-free microvascular model.

Motivation
----------
Every parametric perfusion model silently commits to a *shape* for the
capillary transit-time distribution ``h(tau)`` and then can only report its
first two moments (mean transit MTT, heterogeneity CTH):

* the gamma-variate (Larsson) model locks the standardised skewness to
  ``2 * RD``  (a gamma identity, RD = CTH/MTT),
* the inverse-Gaussian / Wald model locks it to ``3 * RD``,
* Tikhonov deconvolution recovers a residue but reads it out only to CTH and
  throws the 3rd moment away.

So *no* existing model can answer a basic question: **what shape is the transit
distribution, really?** This model recovers ``h(tau)`` non-parametrically as a
non-negative distribution on a transit-time grid, with a separate irreversible
"leak" column, and then reads off the moments the others cannot:

    cbf, mtt, cth        (validation anchors — should track Tikhonov)
    skew                 the standardised 3rd cumulant of h(tau)          [NEW]
    shape_idx = skew/RD  =2 iff gamma-shaped, =3 iff Wald, <2 peaked/plug  [NEW]
    fast_frac            transit mass with tau < MTT/2 (short-transit shunt) [NEW]

``shape_idx`` is the headline: a per-voxel test of the shape assumption that
gamma and Wald *impose*. A map that sits near 2 everywhere vindicates the
gamma-variate; structured departures falsify it.

Method
------
Forward model is the standard indicator-dilution convolution

    C_t(t) = (CBF/6000) * (C_a  conv  R)(t),   R(t) = 1 - CDF_h(t) = residue.

We discretise ``h`` on a log-spaced transit grid ``{tau_k}`` (delta atoms, so the
moments ``E[tau^n] = sum_k p_k tau_k^n`` are exact) plus one *leak* atom at
``tau -> infinity`` (``R = 1``, the Patlak cumulative-AIF regressor). The design
column for atom k is ``B[:,k] = (C_a conv R_k)`` with ``R_k(t) = 1{t < tau_k}``.
Amplitudes are recovered by non-negative least squares

    min_{a >= 0}  || B a - C_t ||^2 + ridge ||a||^2 ,

positivity being the only prior (it *is* the statement that h is a probability
density). ``a_leak / sum(a)`` separates retention from transit by grid location,
not by an arbitrary time window. Transit moments are computed from the finite
atoms after renormalisation.

Unlike the ``stieltjes`` model (which recovers a *rate* spectrum and requires a
completely-monotone residue, hence ``RD >= 1``, excluding peaked transit), this
recovers the transit-time distribution directly and admits any shape.

Relation to the field: non-parametric transport-function deconvolution exists in
DSC-MRI (Ostergaard control-point, Mouridsen); reporting the *3rd-moment shape*
and an explicit gamma-vs-Wald discriminant per voxel is the new element.
Local / experimental model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import CurveInputs, ModelResult


# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────


def _residue_design(t: np.ndarray, ca: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Design matrix ``B`` (T x (K+1)); column k = C_a convolved with the residue
    ``R_k(t)=1{t<tau_k}`` of a delta transit at ``tau_k``. The final column is the
    leak term ``R=1`` (cumulative AIF)."""
    n = t.size
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    K = taus.size
    B = np.empty((n, K + 1), dtype=float)
    for k, tau in enumerate(taus):
        Rk = (t < tau).astype(float)                 # step residue
        B[:, k] = np.convolve(ca, Rk)[:n] * dt
    B[:, K] = np.convolve(ca, np.ones(n))[:n] * dt   # leak: R = 1
    return B


def _nnls_batch(A: np.ndarray, Bmat: np.ndarray, n_iter: int = 800,
                ridge: float = 1e-6) -> np.ndarray:
    """min_{W>=0} ||A W - Bmat||^2 + ridge||W||^2 for all RHS columns (FISTA-PG)."""
    AtA = A.T @ A + ridge * np.eye(A.shape[1])
    AtB = A.T @ Bmat
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


def _spectrum_to_params(A: np.ndarray, taus: np.ndarray) -> dict[str, np.ndarray]:
    """Turn recovered amplitudes ``A`` ((K+1) x V) into the output maps."""
    K = taus.size
    a_fin = A[:K, :]                                  # finite transit atoms
    a_leak = A[K, :]
    tot = A.sum(axis=0)
    massf = a_fin.sum(axis=0)                          # finite (transit) mass
    with np.errstate(divide="ignore", invalid="ignore"):
        p = a_fin / massf[None, :]                     # transit distribution (renormalised)
        m1 = (p * taus[:, None]).sum(axis=0)           # MTT
        m2 = (p * (taus ** 2)[:, None]).sum(axis=0)
        m3 = (p * (taus ** 3)[:, None]).sum(axis=0)
        var = m2 - m1 ** 2
        cth = np.sqrt(np.clip(var, 0.0, None))
        mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3       # central 3rd moment
        skew = mu3 / np.clip(cth ** 3, 1e-12, None)
        rd = cth / np.clip(m1, 1e-12, None)
        shape_idx = skew / np.clip(rd, 1e-6, None)     # gamma=2, Wald=3, plug->0
        fast = (p * (taus[:, None] < 0.5 * m1[None, :])).sum(axis=0)  # tau < MTT/2 mass
        cbf = tot * 6000.0
        leak = a_leak / np.clip(tot, 1e-12, None)
    bad = ~np.isfinite(massf) | (massf <= 0)
    for arr in (m1, cth, skew, shape_idx, fast, cbf, leak):
        arr[bad] = np.nan
    return {"cbf": cbf, "mtt": m1, "cth": cth,
            "skew": skew, "shape_idx": shape_idx, "fast_frac": fast, "leak_frac": leak}


# ────────────────────────────────────────────────────────────────────────
# plug-in
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TransitSpectrumModel:
    """De novo transit-time-spectrum model (the ``transit_spectrum`` plug-in)."""

    key: ClassVar[str] = "transit_spectrum"
    name: ClassVar[str] = "Transit-time spectrum (shape-free CTH + skew)"
    description: ClassVar[str] = (
        "Non-parametric non-negative transit-time distribution h(tau) by NNLS "
        "deconvolution on a transit grid plus a leak atom. Reports cbf/mtt/cth "
        "(validation) and the shape the parametric models cannot: skew (3rd "
        "cumulant), shape_idx = skew/RD (=2 gamma, =3 Wald, <2 plug-like) and "
        "fast_frac (short-transit shunt mass)."
    )
    accepts: ClassVar[dict[str, type]] = {
        "c_tissue": np.ndarray, "c_input": np.ndarray, "t_s": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "cbf": np.ndarray, "mtt": np.ndarray, "cth": np.ndarray,
        "skew": np.ndarray, "shape_idx": np.ndarray, "fast_frac": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = (
        "cbf", "mtt", "cth", "skew", "shape_idx", "fast_frac",
    )
    supports_voxelwise: ClassVar[bool] = True
    primary_map: ClassVar[str] = "shape_idx"
    units: ClassVar[dict[str, str]] = {
        "cbf": "mL/100g/min", "mtt": "s", "cth": "s",
        "skew": "(dimensionless)", "shape_idx": "(gamma=2, Wald=3)",
        "fast_frac": "(fraction)",
    }

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        c_t = np.asarray(inputs.c_tissue, dtype=float)
        c_a = np.asarray(inputs.c_input, dtype=float)
        t_s = np.asarray(inputs.t_s, dtype=float)
        n = int(min(t_s.size, c_a.size, c_t.shape[0]))
        t_s, c_a = t_s[:n], c_a[:n]
        c_t = c_t[:n]

        # Crop pre-bolus baseline at the AIF leading edge (H = F·R̂ is invariant to
        # a delay shared by Ca and Ct; removing it conditions the fast atoms).
        if bool(opts.pop("crop_bolus", True)) and np.any(np.isfinite(c_a)):
            thr = 0.05 * float(np.nanmax(c_a))
            i0 = int(np.argmax(c_a > thr)) if np.any(c_a > thr) else 0
            i0 = max(i0 - 1, 0)
            if 0 < i0 < n - 5:
                t_s = t_s[i0:] - t_s[i0]
                c_a = c_a[i0:]
                c_t = c_t[i0:] if c_t.ndim == 1 else c_t[i0:, :]

        tau_min = float(opts.pop("tau_min_s", 0.5))
        tau_max = float(opts.pop("tau_max_s", 30.0))
        n_tau = int(opts.pop("n_tau", 32))
        ridge = float(opts.pop("ridge", 1e-6))
        n_iter = int(opts.pop("nnls_iter", 800))
        taus = np.geomspace(tau_min, tau_max, n_tau)

        B = _residue_design(t_s, c_a, taus)

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
        A = None
        if voxels.size:
            Y = c_t[:, voxels]
            A = _nnls_batch(B, Y, n_iter=n_iter, ridge=ridge)     # (K+1) x Vsel
            params = _spectrum_to_params(A, taus)
            for k in self.outputs:
                out[k][voxels] = params[k]

        if single:
            maps = {k: np.asarray(v[0]) for k, v in out.items()}
            return ModelResult(maps=maps, units=dict(self.units),
                               aux={"taus": taus,
                                    "amps": A[:, 0] if A is not None else None})
        return ModelResult(maps=out, units=dict(self.units), aux={"taus": taus})

    def review(self, inputs, result, **_kw):
        """--mode verify view — mean tissue Cₜ + the spectrum fit + median parameters."""
        from ._review import curve_fit_review
        return curve_fit_review(self, inputs, result,
                                title="Transit-time spectrum", fit_label="spectrum fit")

    def predict(self, maps: dict[str, Any], c_input: np.ndarray,
                t_s: np.ndarray) -> np.ndarray:
        """QC overlay via a gamma-residue proxy at the fitted (cbf, mtt, cth)."""
        from .base import reconstruct_gamma_residue_ct
        return reconstruct_gamma_residue_ct(
            t_s, c_input,
            cbf=float(maps.get("cbf", float("nan"))),
            mtt=float(maps.get("mtt", float("nan"))),
            cth=float(maps.get("cth", float("nan"))),
        )


PLUGIN = TransitSpectrumModel()
