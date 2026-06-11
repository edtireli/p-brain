# "Tikhonov Bayes" in p-Brain — what it does, and how it relates to Larsson-style DCE-MRI deconvolution

*Based on reading `pbrain/models/tikhonov_bayes.py`, `pbrain/models/tikhonov.py`, and `pbrain/diagnostics/tikhonov_bayes.py`.*

## 1. What the code actually is

`tikhonov_bayes` is **not a separate solver**. It is a thin plugin wrapper (`pbrain/models/tikhonov_bayes.py`, ~90 lines) that delegates to the existing `tikhonov` solver (`build_tikhonov_solver` in `pbrain/models/tikhonov.py`) with three option defaults flipped:

```python
opts.setdefault("lambda_selection", "evidence")   # vs "gcv"/"lcurve"
opts.setdefault("lambda_spacing",   "log")
opts.setdefault("mtt_cth_method",   "residue_integral")
opts["compute_cbf_sd"] = True                       # unless sampling is on
```

So the "Bayesian" part is two concrete additions inside the shared solver: an **evidence-maximisation λ-selector** and a **closed-form posterior SD on CBF** (with optional posterior sampling for MTT/CTH SD). Everything else — the forward model, the basis, the residue post-processing — is identical to the classic Tikhonov path. The wrapper's docstring is candid that the motivation was a concrete failure: on smooth high-SNR DCE curves the L-curve has no corner and GCV is monotonic, so both pickers "collapse to the λ_min grid floor" (it names subject `20250217x4`).

## 2. The shared forward model (same for classic and Bayes)

The deconvolution problem is the standard one:

```
C_t(t) = F · ∫₀ᵗ C_a(τ) R(t−τ) dτ,   R(0)=1
```

Implementation details from `build_tikhonov_solver`:

- The impulse response `g(t) = F·R(t)` is represented in a **cubic (order-4) B-spline basis** `B` with knots every `5·dt` (Cox–de Boor recursion in `_bspline_basis`).
- The AIF is turned into a lower-triangular **Toeplitz convolution matrix** `Ca` (`_toeplitz_ca_matrix`); the design matrix is `D = Ca · Bᵀ`.
- The estimate solves the **Tikhonov problem with a first-difference roughness penalty** `L`:
  ```
  x̂ = argminₓ ‖D·x − C_t‖² + λ²‖L·x‖²
     = (DᵀD + λ²LᵀL)⁻¹ Dᵀ C_t
  ```
  solved via a per-λ Cholesky factorisation of `A(λ) = DᵀD + λ²LᵀL`.
- **CBF** = peak of the reconstructed `g(t)` over the early `f_win` frames, scaled by `6000·(1−Hct)/ρ` to mL/100 g/min (the `(1−Hct)` only applied for a plasma-derived AIF).
- **CBV** = `∫C_t / ∫C_a` (`vd`); **MTT** and **CTH** from the constrained residue (`_residue_metrics_batch`: clip ≥0, enforce monotone non-increase, MTT = ∫R dt, CTH = SD of `h = −dR/dt`).
- Per-voxel **AIF offset/delay** handled by PCHIP-shifting the AIF and caching one design/factorisation per quantised offset.

A λ-grid (default 121 points, log-spaced from `1e-3` to an SVD-derived `σ_max` of the AIF Toeplitz matrix) is searched per voxel; the selected λ index picks the solution.

## 3. The Bayesian recasting — precisely what is added

The MAP/posterior-mean equivalence is exact and standard: `x̂ = (DᵀD + λ²LᵀL)⁻¹Dᵀb` is the posterior mean of Bayesian linear regression with a Gaussian likelihood `b ~ N(Dx, σ²I)` and a Gaussian smoothness prior `Lx ~ N(0, (σ²/λ²)I)`. λ is the prior-to-noise precision ratio.

**(a) λ by evidence maximisation (`lambda_selection="evidence"`).** Instead of L-curve curvature or GCV, λ is chosen per voxel by maximising the log marginal likelihood with σ² integrated out under a Jeffreys prior:

```
log Z(λ) = −(N/2)·log(RSS(λ) + λ²·PEN(λ))     ← data misfit
           + rank(L)·log(λ)                      ← prior-normalisation / Occam term
           − ½·log|A(λ)|                          ← Occam complexity penalty
```

In code (`tikhonov.py` lines ~513–528): `reg_term = RSS + λ²·PEN`, then `log_ev = −0.5·N·log(reg_term) + rank_L·log(λ) − 0.5·logdetA`, and `idx_max = argmax(log_ev)`. `log|A(λ)|` is read off the diagonal of the existing Cholesky factor (free). The claim — credible from the functional form — is that the growing Occam terms vs. the shrinking misfit term give a **guaranteed interior maximum**, so it cannot collapse to `λ_min` the way the curvature/GCV pickers do on smooth curves.

This is **empirical Bayes / evidence (type-II ML) maximisation**: σ² is marginalised analytically, but λ is still a point estimate (the evidence maximiser). It is not a hierarchical prior on λ and not a full posterior over λ — except in the sampling path below.

**(b) Closed-form posterior SD on CBF (`compute_cbf_sd`, the default).** Because `F = (B[:,k]ᵀx)/dt` is a linear functional of the coefficients at the (fixed) peak index `k`, its posterior variance is closed form:

```
Var(F) = σ̂²·gᵀ A(λ)⁻¹ g,   g = B[:,k]/dt,   σ̂² = (RSS + λ²·PEN)/N
```

producing a `cbf_sd` map (lines ~592–604). The code is explicit that this fixes `k` (first-order; it under-estimates uncertainty at very low SNR where the peak location jitters).

**(c) Optional full posterior sampling (`uncertainty_samples=N`, opt-in).** When turned on it does the more complete thing: it **marginalises λ** by drawing `λ_k ~ p(λ|b) ∝ exp(log Z(λ))` over the grid, then `x_k ~ N(x̂(λ_k), σ̂²(λ_k)·A(λ_k)⁻¹)` (sampling via the Cholesky factor), and reports `cbf_sd`, `mtt_sd`, `cth_sd` as the spread. This captures both coefficient and λ-selection uncertainty and propagates through the nonlinear MTT/CTH functionals. By default this is off (only `cbf_sd` from the closed form is produced); the diagnostic plot turns it on for the single ROI curve, where cost is negligible.

### Inputs / outputs summary
- **Inputs:** tissue curve(s) `C_t` (1-D or voxel batch), AIF `C_a`, time vector, optional per-voxel offsets, optional brain mask; physiological constants (density 1.04, Hct 0.42, plasma flag).
- **Outputs:** `cbf` (mL/100 g/min), `mtt` (s), `cth` (s), `lambda_opt`, `cbf_sd`; plus `mtt_sd`/`cth_sd` when sampling is enabled; `cbv_vd` as aux.
- **Inference performed:** per-voxel evidence-maximised λ → Tikhonov/posterior-mean coefficient solve → CBF/MTT/CTH readout, with closed-form (or sampled) posterior uncertainty.

## 4. Larsson et al. and classic Tikhonov deconvolution in DCE-MRI

Larsson et al. 2009 (MRM 62:1270–1281) estimate cerebral **perfusion by model-free deconvolution using Tikhonov's method**, while **CBV and BBB permeability come from a separate Patlak analysis and a two-compartment model** — not from the deconvolution itself. Reported GM/WM/tumour perfusion: 72±16 / 30±8 / 56±45 mL/100 g/min. The architecture is the same split p-Brain uses: deconvolution → flow; Patlak/2-compartment → leakage (the repo has `models/patlak.py` and `extended_tofts.py` alongside the Tikhonov models.)

Classic Tikhonov-regularised perfusion deconvolution (the literature p-Brain's `tikhonov.py` ports — the docstring cites "L-curve curvature (Hansen)") chooses the regularization parameter by **a point-estimate, data-fit-only criterion**:
- **L-curve criterion (LCC)** — maximise curvature of the (log‖residual‖², log‖Lx‖²) trade-off curve (Hansen). This is exactly the `"lcurve"` branch in `tikhonov.py`.
- **Generalized Cross-Validation (GCV)** — minimise `‖(I−H(λ))b‖² / (n−tr H(λ))²` with `H(λ)=D A(λ)⁻¹Dᵀ`. Exactly the `"gcv"` branch.

Sourbron et al. (2004, Phys Med Biol; PubMed 15357199) is the canonical comparison of L-curve vs GCV for perfusion; standard-form Tikhonov with per-pixel LCC/GCV is established to reduce residue oscillations and improve CBF at short MTT versus (o)SVD. None of these classic methods produces a per-voxel uncertainty on CBF — they return point estimates.

Bayesian/probabilistic perfusion estimators do exist in the prior art — e.g. Mouridsen et al. 2006 (NeuroImage, "Bayesian estimation of cerebral perfusion using a physiological model of microvasculature"), and more recent neural-network posterior estimators — and these report posterior distributions and outperform oSVD at low SNR. So a Bayesian framing of perfusion deconvolution is not itself novel; what is specific here is the **empirical-Bayes evidence λ-selector grafted directly onto the standard B-spline Tikhonov solver**, motivated by a reproducible failure of LCC/GCV on this lab's smooth high-SNR curves.

## 5. Comparison: what the Bayesian formulation adds vs. classic / Larsson

**Regularization-parameter selection.** Classic (Larsson / Sourbron): point estimate of λ from a data-fit geometric criterion (L-curve corner) or predictive criterion (GCV) — both can be ill-defined (no corner / monotonic GCV) on smooth curves and degenerate to the grid floor. Bayes: point estimate of λ too, but from **evidence maximisation** with σ² marginalised, whose Occam terms guarantee an interior optimum. With `uncertainty_samples>0` it goes further and **marginalises λ over the evidence-weighted grid** rather than fixing it. Net: a more robust λ on this data, and (optionally) propagation of λ ambiguity.

**Uncertainty quantification.** Classic: none — CBF/MTT/CTH are point estimates. Bayes: a **closed-form posterior SD on CBF** per voxel by default (`cbf_sd`), and full sampled SDs on CBF/MTT/CTH when enabled. This is the most substantive practical addition: it answers whether a given CBF or perfusion deficit is resolvable above fit noise — exactly what Larsson-style maps cannot say.

**Priors / physiological constraints.** Both use the same first-difference roughness penalty `L`, which in the Bayesian reading is a Gaussian smoothness prior on the residue. The physiological constraints (R monotone non-increasing, non-negative, R(0)=1) are imposed identically in **post-processing** in both paths, not in the prior. So the Bayes version does **not** add stronger physiological priors than the classic solver — it reinterprets the existing penalty probabilistically. (Contrast with Mouridsen 2006, where the prior is a genuine vascular/physiological model.)

**Computational approach.** Both are MAP/linear-solve at heart — per-λ Cholesky of `A(λ)`, no nonlinear optimisation. The evidence selector adds only `log|A(λ)|` (free from the Cholesky diagonal); the closed-form CBF SD adds one extra solve `A⁻¹g` per voxel. So default cost ≈ classic Tikhonov. There is **no MCMC**; the "full posterior" mode is cheap grid-weighted Gaussian sampling via the existing factorisations, not a sampler over a generative model. This is much lighter than fully-Bayesian perfusion methods in the literature.

**Practical implications for estimated parameters.** For flow (CBF), the main effect is robustness: avoiding the λ_min collapse should reduce over-fit, oscillatory residues and the resulting CBF over-estimation on smooth curves, and the new `cbf_sd` flags unreliable voxels. MTT/CTH are derived from the same residue, so they inherit the more stable λ; their uncertainty is only available via the sampling path. **BBB permeability/Ktrans and CBV are untouched** — as in Larsson, those come from Patlak/two-compartment/extended-Tofts models, not from this deconvolution, so "Tikhonov Bayes" changes the perfusion estimate and its error bar, not the leakage estimates.

## 6. Honest limitations / what I could not determine from the code alone

- The "L-curve/GCV collapse to λ_min" claim is asserted from one named subject in the docstring; I read the code, not the validation runs, so I can't confirm how general the failure is or quantify the improvement.
- λ selection is empirical-Bayes (type-II ML point estimate). Only the opt-in sampling path is "full posterior" over λ, and even that is a grid-weighted Gaussian approximation, not a proper hierarchical/MCMC posterior — so calling the default mode "Bayesian" is a reasonable but loose label (it's MAP + Laplace-style closed-form SD).
- The closed-form `cbf_sd` fixes the peak index `k`; the code itself notes this under-estimates SD at low SNR. Whether that matters in practice depends on SNR regimes I haven't run.
- I did not find a written paper section in the repo deriving the evidence functional; the derivation lives only in the docstrings. The functional form is standard and looks correct, but I did not numerically verify the `log|A|`/Occam balance.
- I read the `pbrain/` package implementation; there is a parallel legacy `models/tikhonov.py` and `models/tikhonov_matlab.py`. I confirmed the `pbrain` Tikhonov claims byte-parity with the legacy one but did not diff them.

## Sources
- Larsson HBW, Courivaud F, Rostrup E, Hansen AE. *Measurement of brain perfusion, blood volume, and blood-brain barrier permeability, using dynamic contrast-enhanced T1-weighted MRI at 3 tesla.* Magn Reson Med. 2009;62:1270–1281. https://onlinelibrary.wiley.com/doi/10.1002/mrm.22136 — https://pubmed.ncbi.nlm.nih.gov/19780145/
- Sourbron S, et al. *Choice of the regularization parameter for perfusion quantification with MRI.* Phys Med Biol. 2004. https://pubmed.ncbi.nlm.nih.gov/15357199/
- Mouridsen K, et al. *Bayesian estimation of cerebral perfusion using a physiological model of microvasculature.* NeuroImage. 2006. https://pubmed.ncbi.nlm.nih.gov/16971140/
- Sourbron S, et al. *Deconvolution-Based CT and MR Brain Perfusion Measurement: Theoretical Model Revisited and Practical Implementation Details.* https://pmc.ncbi.nlm.nih.gov/articles/PMC3166726/
