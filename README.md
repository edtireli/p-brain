# _p_-Brain: Advanced Neuroimaging platform
<img width="2500" height="549" alt="pbrainplatform_banner" src="https://github.com/user-attachments/assets/e0c55c26-5c31-468f-9380-3b045e3a495c" />


_p_-Brain is an end-to-end neuroimaging analysis script that turns raw dynamic contrast-enhanced (DCE) MRI series into quantitative maps of blood–brain barrier leakage, vascular volume, and perfusion. The toolkit combines classical pharmacokinetic modeling with CNN-based region-of-interest (ROI) extraction, anatomical parcellation, and transparent quality-control outputs so that a single command can deliver voxel-wise, parcel-wise, and whole-brain readouts of:

- BBB influx constant Ki
- Plasma volume vp
- Extended Tofts parameters (Ktrans, kep, ve)
- Cerebral blood flow (CBF)
- Mean transit time (MTT)
- Capillary transit-time heterogeneity (CTH)

> **Author:** Edis Devin Tireli, M.Sc., Ph.D. student  
> **Affiliations:** Functional Imaging Unit, Copenhagen University Hospital – Rigshospitalet, Department of Neuroscience and Department of Clinical Medicine, University of Copenhagen. 

---

## Quick navigation
1. [Why p-Brain?](#why-p-brain)
2. [p-brain Platform (Desktop app)](#p-brain-platform-desktop-app)
3. [p-brain (Standalone pipeline)](#p-brain-standalone-pipeline)
4. [Data layout and repository structure](#data-layout-and-repository-structure)
5. [Installation](#installation)
6. [Running the script](#running-the-script)
7. [Workflow details](#workflow-details)
8. [Outputs and deliverables](#outputs-and-deliverables)
9. [Automation features](#automation-features)
10. [Configuration and environment variables](#configuration-and-environment-variables)
11. [Addons](#addons)
12. [Contributing & support](#contributing--support)
13. [License & acknowledgments](#license--acknowledgments)

---

## Why p-Brain?
Traditional DCE-MRI analysis requires hand-drawn ROIs for arterial/venous input functions, manual tissue masking, and bespoke scripts for each pharmacokinetic model. _p_-Brain removes these bottlenecks:

- **Single script, full pipeline** – From T1/M0 fitting to Patlak, extended Tofts, and deconvolution-based residue analysis.
- **CNN-driven automation** – Neural networks detect the right internal carotid artery (rICA) and superior sagittal sinus (SSS); FastSurfer-based anatomical segmentations define tissue ROIs.
- **Multi-scale reporting** – Every run produces voxel mosaics, parcel tables, slice-wise distributions, and whole-brain medians for Ki, vp, CBF, MTT, and CTH.
- **Reproducible QC** – Time-shifted concentration curves, Patlak fits, reference comparisons, and cohort projections are generated automatically so every decision is traceable.
- **Batch-ready** – `enumerator.py` runs the pipeline over entire cohorts with optional control handling and environment-based overrides.

---

## p-brain Platform (Desktop app)

If you want a **production-style UI** on top of the pipeline (projects/subjects, job monitoring, and a rich QC/review workspace), see:

- p-brain Platform: https://github.com/edtireli/p-brain-platform

Most users should just download the **macOS desktop app** (a `.dmg`) from GitHub Releases:

- Releases: https://github.com/edtireli/p-brain-platform/releases

The platform surfaces p-brain outputs in a review-first workflow:

- Project/subject browser and pipeline status
- QC overlays for DCE outputs (maps, curves, fit diagnostics)
- Interactive diffusion/tractography viewer
- Local desktop launcher that bundles the UI and runs a small local bridge for file access

### Platform screenshots (v1.0.0)

![p-brain Platform dashboard](src/img/platform/Screenshot%202026-01-08%20at%2017.43.51.png)

![Tractography viewer](src/img/platform/Screenshot%202026-01-08%20at%2017.45.21.png)

![DCE series with AI overlay](src/img/platform/Screenshot%202026-01-08%20at%2017.47.11.png)

![Voxelwise map example](src/img/platform/Screenshot%202026-01-08%20at%2017.45.41.png)

![Parcelwise map example](src/img/platform/Screenshot%202026-01-08%20at%2017.45.56.png)

![AIF/VIF and tissue functions](src/img/platform/Screenshot%202026-01-08%20at%2017.46.02.png)

![Function isolation / selection](src/img/platform/Screenshot%202026-01-08%20at%2017.46.20.png)

![Patlak analysis](src/img/platform/Screenshot%202026-01-08%20at%2017.46.35.png)

![Extended Tofts estimation](src/img/platform/Screenshot%202026-01-08%20at%2017.46.54.png)

![Segmentation results table](src/img/platform/Screenshot%202026-01-08%20at%2017.49.45.png)

![Additional visualization](src/img/platform/Screenshot%202026-01-08%20at%2017.47.18.png)

![Subjects overview / job status](src/img/platform/Screenshot%202026-01-08%20at%2017.49.13.png)

---

## p-brain (Standalone pipeline)

The screenshots above are from the optional **p-brain Platform** desktop app.

Everything below describes **p-brain as a standalone pipeline** in this repository: the expected `data/` layout, how to install dependencies, and how to run `main.py` / `enumerator.py` directly.

## Data layout and repository structure
### Expected dataset tree
By default the GUI scans the `data/` directory (override via `--data-dir` or `P_BRAIN_DATA_DIR`). Each exam folder should contain raw input as well as the derived analysis subfolders:

```
data/
└── subject_id/
    ├── x.PAR / x.REC   # raw Philips exports (optional if NIfTI already provided)
    ├── NIfTI/          # populated automatically when converting PAR/REC
    ├── Analysis/
    │   ├── CTC Data/
    │   ├── TSCC Data/
    │   ├── ITC Data/
    │   └── ROI Data/
    └── Images/
```

Control cohorts live under `data/controls/<id>`. Set `PBRAIN_CONTROLS=1` or pass `--controls` to `enumerator.py` so the script automatically tags outputs with a `control.json` descriptor.

### Repository overview
| Path | Description |
|------|-------------|
| `modules/` | CLI menus, modeling backends, and GUI hooks. |
| `utils/` | Configuration, plotting helpers, and shared utilities. |
| `AI/` | Default CNN weights for rICA/SSS slice detection and ROI segmentation. |
| `addons/` | Optional plugins (e.g., GM/WM boundary ROIs). |
| `src/img/` | Repository-owned images used exclusively in the README. |
| `main.py` | Interactive runner used by the GUI/CLI. |
| `enumerator.py` | Batch launcher that iterates over multiple datasets. |

Key filenames (configured in `utils/parameters.py`) include the axial 2D reference image, DCE series, inversion recovery stack (WIPTI_xxxxx.nii), and optional 3D T1/T2/FLAIR reconstructions. Dedicated `control_*` entries allow alternative names for control acquisitions.

---

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/edtireli/p-brain.git
   cd p-brain
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. *(Optional)* **Fetch addon submodules**
   ```bash
   git submodule update --init -- addons/<addon_name>
   ```

Version strings are derived automatically from `git describe --tags` inside `modules/__init__.py`, so releases always match the tag checked out locally.

---

## Running the script
### Interactive GUI/CLI
```bash
python3 main.py
```
1. A small GUI lists available dataset folders under the configured data directory. Select one and click **Accept**.
2. The terminal menu appears and offers three modes:
   - **Manual mode** – Step-by-step execution with GUI ROI drawing.
   - **Automatic mode** – Fully automated pipeline (CNN inputs, FastSurfer segmentation, Patlak/Tofts/deconvolution, reporting).
   - **Pseudo-automatic mode** – Hybrid workflow where the user can review intermediate ROIs before modeling.

### Manual menu overview
| Option | Purpose |
|--------|---------|
|0|View MRI series (axial/sagittal).|
|1|Fit T1/M0 from the inversion recovery stack.|
|2|Generate concentration time curves (CTCs) from user-drawn ROIs.|
|3|Time-shift venous curves to arterial peaks (with amplitude rescaling if necessary).|
|4|Create tissue-specific CTCs (GM, WM, cerebellum, boundary, etc.).|
|5|Estimate BBB permeability (Patlak + extended Tofts) and residue-derived perfusion metrics.|
|6|Add free-form analysis notes to the dataset.|
|7|Invoke addons (boundary ROI extraction, screenshots, ...).|
|9|Exit.|

### Batch processing
`enumerator.py` wraps `main.py` so whole cohorts can be processed unattended:
```bash
python enumerator.py 1001 1002
python enumerator.py --all
python enumerator.py --controls 01 02
python enumerator.py --controls --all
```
Use `--data-dir` or `P_BRAIN_DATA_DIR` to point to an alternate root. The script automatically toggles `PBRAIN_CONTROLS` when `--controls` is provided.

#### Diffusion/tractography overrides
- `--diffusion-file <filename>` lets you select a specific diffusion volume for both FA metrics and tractography. Pass either an absolute path or a filename relative to each dataset’s `NIfTI/` folder (e.g. `--diffusion-file WIPDWI_highres.nii.gz`).
- `--orientation {tensor,dti,csd,mt_csd,qball,gqi}` continues to override the tractography model. `csd` uses the legacy single-shell fit, whereas `mt_csd` (alias: `msmt`, `msmt_csd`) runs the multi-tissue MSMT-CSD solver when multi-shell diffusion data are available. In both modes, the pipeline inspects how many non-b0 diffusion directions are available and automatically picks the largest safe spherical-harmonic (SH) order; sparse datasets fall back to lower SH orders to avoid ill-conditioned fits.
- `--tracks_dont_recompute` skips streamline regeneration whenever `--tracks` (or `--tracks_only`) is present. This is handy for combos like `--diffusion_only --tracks_only --tracks_dont_recompute`, which recompute FA metrics but simply refresh tract renders/montages from an existing `tractography.trk`.
- `--tracks_force` ignores cached streamlines and forces a fresh tractography build, even if `tractography.trk` already exists.
- Advanced users can still pin the SH order via `P_BRAIN_TRACK_CSD_SH_ORDER`. Setting it to `auto` (default) keeps the adaptive behavior; numeric values force a specific even order.
- Tractography attempts now support parallel execution via `P_BRAIN_TRACK_WORKERS`. Set it to the number of CPUs you want to dedicate (defaults to 1 to preserve historical behavior). The default backend uses threads; set `P_BRAIN_TRACK_PARALLEL_BACKEND=process` if you prefer separate worker processes. Combine this with `OMP_NUM_THREADS=1` when running on multi-socket machines so BLAS-heavy steps do not oversubscribe the system. Progress bars remain accurate even when attempts finish out of order.
- To keep the CLI `--orientation csd` flag but still force multi-tissue MSMT-CSD, export `P_BRAIN_TRACK_FORCE_MT_CSD=1`. When set, every CSD request runs through the MSMT solver and logs the override inside `Analysis/diffusion/tractography_debug.json`.

---

## Workflow details
The automated workflow mirrors the structure shown below. Gray boxes are completely unsupervised; white boxes correspond to manual overrides when running in manual or pseudo-automatic mode.

1. **Inputs** – Minimum requirements: 3D T1-weighted structural volume, inversion recovery series (for T1/M0), and a 4D DCE time series. Optional diffusion data enables automated FA reporting.
2. **Preprocessing** – Optional PAR/REC conversion via `dcm2niix`, rigid alignment of structural volumes to DCE space, and consistency checks on slice timing.
3. **T1/M0 fitting** – Trust-region reflective solver fits the inversion recovery signal model with configurable inversion delays and relaxivity (default r1 = 4 s-1 mM-1).
4. **Input-function extraction** – CNN slice classifier + ROI segmentation detect rICA and SSS. Venous curves are cross-correlated and rescaled to the arterial peak, compensating for transit delays and dispersion.
5. **Tissue ROIs** – FastSurfer-based parcellations (with optional FSL anatomical priors) define cortical GM, subcortical GM, WM, cerebellar lobes, brainstem, and GM/WM boundary masks. Affine transforms propagate labels to DCE geometry.
6. **Signal-to-concentration conversion** – Spoiled-GRE equation transforms signal intensity into gadolinium concentration using fitted T1, M0, flip angle, and TR. Guards prevent invalid logarithms or unstable tails.
7. **Modeling**  
   - **Patlak graphical analysis** for Ki and vp with user-configurable linear windows and residual-based uncertainty estimates.
   - **Extended Tofts model** with Levenberg–Marquardt fitting for Ktrans, ve, vp, and kep.
   - **Model-free deconvolution** (Tikhonov-regularized) providing CBF, MTT, and CTH from the residue function. An experimental gamma-variate estimator is also exposed for benchmarking.
8. **Outputs** – Quantitative NIfTI maps, PNG mosaics, JSON summaries, CSV/TSV tables, cohort boxplots, atlas projections, and optional reference comparisons.
9. **Quality assurance** – Automated checks for segmentation failures, mask overlaps, motion spikes, atypical AIFs, fit residuals, and log integrity. All warnings are logged alongside the outputs.

---

## Outputs and deliverables
Every automatic run produces the following without additional scripting:

- **Voxel-wise maps** – Ki, vp, CBF, CTH, and MTT stored as NIfTI volumes plus pre-rendered mosaics.
- **Parcellated summaries** – FastSurfer atlas statistics for each parameter, exported as tables and overlay images.
- **Slice-wise distributions** – Boxplots showing superior–inferior trends for Ki, vp, and perfusion metrics; useful for QC and cohort comparisons.
- **Whole-brain medians** – GM, WM, cerebellar, and boundary medians saved in JSON for rapid reporting or EHR integration.
- **Cohort projections** – When multiple datasets exist, the script averages parcel values across subjects and projects them onto a reference segmentation to create cohort fingerprints.
- **Reference comparisons** – Optional automated figures contrasting _p_-Brain outputs with the Perffit2 implementation (GM/WM boxplots and subject-wise scatter plots).
- **Processing transparency** – Composite figures stacking segmentations, input functions, tissue curves, Patlak fits, and resulting parameter maps, ensuring every automated decision is reviewable.

All generated assets reside under the selected dataset folder inside `Analysis/`, `Images/AI_patlak`, `Images/AI_tikhonov`, and companion JSON/CSV directories.

---

## Representative results gallery
The figures below summarize what a fully automatic run produces for a technically uniform cohort of 97 DCE-MRI scans from 58 participants with mild traumatic brain injury (mTBI) but no macroscopic lesions on structural MRI. Each dataset was processed with the same automated sequence of segmentation, vascular input extraction, concentration conversion, and kinetic modeling (Patlak + extended Tofts + deconvolution). The resulting deliverables span voxelwise maps, parcellated summaries, slice-wise distributions, cohort fingerprints, and compact QC dashboards. The PNGs stored under `src/img/` are the actual exports from the pipeline.

### Voxelwise maps
Voxelwise maps quantify physiological parameters at native spatial resolution so you can examine localized BBB leakage, perfusion, and vascular volume without aggregating over parcels. These maps constitute the foundation for every downstream summary in the pipeline.

- **BBB influx (Ki)** – Patlak-derived unidirectional transfer constant that reflects blood–brain barrier permeability.

  ![Voxelwise Ki map](src/img/ki_voxel_montage_patlak.png)

- **Cerebral blood flow (CBF)** – Model-free residue deconvolution highlights expected perfusion contrast between cortical/subcortical gray matter and deep white matter and resolves major vessels such as the circle of Willis.

  ![Voxelwise CBF map](src/img/cbf_montage.png)

- **Plasma volume (vp)** – Patlak intercept emphasizes the intravascular compartment along cortical ribbons and venous structures.

  ![Voxelwise vp map](src/img/vp_per_voxel_patlak.png)

- **Capillary transit-time heterogeneity (CTH)** – Derived from the normalized outflow $h(t)=-r'(t)/\int(-r')$, revealing spatial mottling that reflects variability in capillary passage times.

  ![Voxelwise CTH map](src/img/cth_montage.png)

- **Mean transit time (MTT)** – First-moment summary of the residue function that complements CTH by capturing overall transit duration.

  ![Voxelwise MTT map](src/img/mtt_montage.png)

### Regional and parcellated organization
FastSurfer anatomical labels propagated to DCE space allow every quantitative map to be summarized into parcel medians for rapid comparisons across lobes, networks, or subject groups. These exports double as CSV/TSV tables for statistics packages, see e.g. the parcelwise CBF map. 

- ![Parcel-level CBF](src/img/cbf_parcel_montage_tikhonov.png)

### Cohort distributions and QC
Slice-wise boxplots summarize how each metric evolves along the superior–inferior axis, preserving the expected gray/white hierarchy while flagging outliers or motion-contaminated slabs.

### Cohort-level atlas projection
Aggregating parcel statistics across subjects produces cohort fingerprints that can be projected back onto a reference segmentation for quick visual baselines.

### End-to-end transparency
Composite panels document the entire automation chain—segmentation, vascular input functions, tissue curves, Patlak fits, and resulting parameter maps—so every decision remains auditable.

<img width="3560" height="5721" alt="AI_Tissue_slice_5_segmented_median (1) (6)" src="https://github.com/user-attachments/assets/f6458d56-7f39-4e2e-a3e0-9b3b16ddef67" />

### Whole-brain medians
For dashboards or EHR-style summaries, the pipeline reports tissue-specific medians that retain GM>WM ordering while condensing each scan to a few numbers.

#### Summary of findings
The automated pipeline delivers physiologically consistent voxelwise maps, regional summaries, cohort-level projections, and QC figures without user interaction. Exporting these assets alongside transparent diagnostics provides a repeatable baseline for longitudinal monitoring, multi-site harmonization, and future research extensions.

---


## Automation features
### Neural-network models
Four CNNs orchestrate input-function detection:
- Slice classifier (rICA)
- ROI segmentation (rICA)
- Slice classifier (SSS)
- ROI segmentation (SSS)

Default paths live in `utils/settings.py` under `AI_MODEL_PATHS`. Override via environment variables:
```
SLICE_CLASSIFIER_RICA_MODEL
RICA_ROI_MODEL
SLICE_CLASSIFIER_SS_MODEL
SS_ROI_MODEL
```
Pretrained weights are hosted on [Zenodo](https://doi.org/10.5281/zenodo.15655347); download them into the `AI/` directory.

### Kinetic model selection
Set `P_BRAIN_MODEL` (or edit `KINETIC_MODEL` in `utils/settings.py`) to control which models run:
- `patlak`
- `two_compartment` (extended Tofts + deconvolution)
- `both` (default)

Output files are suffixed with `_patlak` or `_tikhonov` and written to `Images/AI_patlak` and `Images/AI_tikhonov` respectively.

### T1 recovery model
Choose between inversion recovery (default) and saturation recovery by setting `P_BRAIN_T1_RECOVERY_MODEL` to `saturation`.

### Regularisation strength
Adjust the Tikhonov parameter via `--lambda`, `P_BRAIN_LAMBDA`, or the corresponding entry inside `utils/settings.py`. The default value is 5.0.

### Global Ki slice exclusion
Skip inferior/superior slices when summarizing whole-brain Ki values by setting:
```
P_BRAIN_GLOBAL_KI_SKIP_BOTTOM
P_BRAIN_GLOBAL_KI_SKIP_TOP
```
Both default to 2.

### Custom datasets and jump-fix
- Drop an `apply_jumpfix.json` next to a dataset to enable automatic correction of sudden signal jumps.  
- Provide your own neural-network weights by placing them inside `AI/` and updating `utils/settings.py`.

---

## Configuration and environment variables
Most behaviour is controlled through `utils/settings.py` and `utils/parameters.py`. You can override almost everything per-run via environment variables.

Notes:
- Booleans generally accept `1/0`, `true/false`, `yes/no` (case-insensitive).
- Paths can be absolute or relative to the repo / dataset directory, depending on the setting.

### Core paths and runtime
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_DATA_DIR` | `data/` | Root directory scanned by the GUI/CLI and by `enumerator.py`/`roi_only.py`. |
| `PBRAIN_CONTROLS` | `0` | Treat datasets as controls during batch runs (also toggled by `enumerator.py --controls`). |
| `P_BRAIN_CORES` | `4` | CPU cores used for multiprocessing where enabled. |
| `P_BRAIN_MPL_BACKEND` | *(auto)* | Matplotlib backend override (also respects `MPLBACKEND`). |
| `PBRAIN_DECOMPRESS_NIFTI_GZ` | *(unset)* | If set, prefer transparently decompressing `.nii.gz` inputs for tooling that can't read gzip. |

### AI model checkpoint overrides (input-function detection)
| Env var | Description |
|---|---|
| `SLICE_CLASSIFIER_RICA_MODEL` | Path to rICA slice classifier weights. |
| `RICA_ROI_MODEL` | Path to rICA ROI segmentation weights. |
| `SLICE_CLASSIFIER_SS_MODEL` | Path to SSS slice classifier weights. |
| `SS_ROI_MODEL` | Path to SSS ROI segmentation weights. |

### Model selection and modelling knobs
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_MODEL` | `both` | `patlak`, `two_compartment`, or `both`. |
| `P_BRAIN_LAMBDA` | `0.5` | Tikhonov regularisation strength for deconvolution/two-compartment modelling. |
| `P_BRAIN_PATLAK_WINDOW_START_FRACTION` | *(code default)* | Controls where the Patlak linear window starts (fraction of the acquisition). |
| `P_BRAIN_PATLAK_MIN_R2` | *(code default)* | Minimum $R^2$ threshold for accepting Patlak fits. |
| `P_BRAIN_NUMBER_OF_PEAKS` | *(code default)* | Peak-detection control used by some input-function utilities. |
| `P_BRAIN_CTH_MTT_METHOD` | `tikhonov` | Residue-derived CTH/MTT strategy. |
| `P_BRAIN_CTH_MTT_GAMMA_VOXELWISE` | `0` | If enabled, writes additional voxelwise gamma-derived maps. |
| `P_BRAIN_GLOBAL_KI_SKIP_BOTTOM` | `2` | Inferior slices skipped for whole-brain Ki medians. |
| `P_BRAIN_GLOBAL_KI_SKIP_TOP` | `2` | Superior slices skipped for whole-brain Ki medians. |

### Signal-to-concentration conversion (CTC)
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_CTC_MODEL` | `saturation` | `saturation` (legacy closed-form) or `turboflash` (numerical inversion). |
| `P_BRAIN_TURBO_NPH` | `1` | TurboFLASH ky=0 line index (1-based), used when `P_BRAIN_CTC_MODEL=turboflash`. |
| `P_BRAIN_FLIP_ANGLE` | `auto` | Flip angle in degrees (number) or `auto` to use metadata. |
| `P_BRAIN_T1_RECOVERY_MODEL` | `inversion` | `inversion` or `saturation` for T1/M0 fitting. |
| `P_BRAIN_T1_FIT` | `auto` | `auto`, `ir`, `vfa`, or `none` for T1/M0 fitting inputs. |
| `P_BRAIN_VFA_GLOB` | *(unset)* | Comma-separated VFA discovery glob(s) when `P_BRAIN_T1_FIT=vfa` or `auto` falls back. |
| `P_BRAIN_HEMATOCRIT` | `0.42` | Haematocrit used in plasma/tissue conversions. |
| `P_BRAIN_TISSUE_DENSITY` | `1.04` | Tissue density used for residue-derived perfusion metrics. |
| `P_BRAIN_CTC_BATCH_VOXELS` | `20000` | Batch size for voxelwise CTC exports/debug in some routines. |
| `P_BRAIN_WRITE_BRAIN_CTC_NIFTI` | `0` | If enabled, writes a whole-brain CTC 4D NIfTI (debug/inspection). |
| `P_BRAIN_BRAIN_CTC_NIFTI_PATH` | *(unset)* | Optional explicit output path for the brain CTC NIfTI. |

### Input-function choice and alignment
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_AIF_ARTERY` | `RICA` | Reference artery for arterial curves: `RICA` or `LICA`. |
| `P_BRAIN_AIF_USE_SSS` | `1` | If enabled, use SSS-derived time-shifted/rescaled output curve (TSCC) for modelling. |
| `P_BRAIN_TSCC_RESCALE` | `1` | If enabled, apply amplitude rescaling when generating TSCC (SSS aligned to artery). Set `0` for time-shift only. |
| `P_BRAIN_TSCC_RESCALE_METHOD` | `peak` | Rescaling method: `peak` (peak-height; legacy upscaling only) or `auc` (area under curve / “total volume”; legacy upscaling only). |
| `P_BRAIN_TSCC_WRITE_DEMO_PLOT` | `1` | If enabled, writes `Images/Time Shifted Concentration Curves/tscc_shift_rescale_demo.png`. |
| `P_BRAIN_INPUT_FUNCTION` | *(legacy)* | Legacy alias: `SSS` forces TSCC output; `RICA`/`LICA` forces pure arterial output. |
| `P_BRAIN_PLASMA_AIF` | `0` | If enabled, uses plasma-derived AIF conventions. |
| `P_BRAIN_ALIGN_AIF` | `0` | If enabled, performs voxelwise delay alignment prior to deconvolution. |
| `P_BRAIN_ALIGN_AIF_MAX_SHIFT` | `4.0` | Maximum allowed AIF shift (seconds) for delay alignment. |

### Vascular ROI: curve extraction
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_VASCULAR_ROI_CURVE_METHOD` | `max` | `max` uses the single brightest voxel (max over time) within the ROI mask; `mean` averages over ROI voxels (more stable, can reduce peak amplitude). |
| `P_BRAIN_VASCULAR_ROI_ADAPTIVE_MAX` | `1` | If enabled (and `P_BRAIN_VASCULAR_ROI_CURVE_METHOD=max`), re-selects the brightest ROI voxel independently at each time frame (“adaptive-max”). |

## GUI applet: environment settings

For quick, reproducible configuration without manually typing env vars, you can run a small Python UI that lists common `P_BRAIN_*` settings, lets you toggle/edit them, and applies them with one click.

```zsh
cd /Users/edt/Desktop/p-brain
python pbrain_env_gui.py
```

**Apply** writes a `.env` file (default: `.pbrain.env` next to `main.py`) and updates the GUI process environment. To apply the same settings in your current shell session:

```zsh
source .pbrain.env
```

The applet can also launch `main.py` directly using the selected settings.

### ROI extraction: method selection
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_ROI_METHOD` | `ai` | `ai` (CNN-driven) or `deterministic` (PCA/connected-components). `geometry` is accepted as a legacy alias for `deterministic`. |

### ROI extraction: shared knobs (AI + deterministic)
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_ROI_DCE_BASELINE_FRAMES` | *(code default)* | Baseline frames used for curve normalization in ROI routines. |
| `P_BRAIN_ROI_NORMALIZE_CURVES` | `1` | If enabled, normalizes curves for PCA/diagnostic plots by baseline-subtraction + baseline-to-zero + robust scaling. |
| `P_BRAIN_ROI_DILATE_PIXELS` | `2` | Final ROI dilation in pixels (applies to saved NIfTI masks). |
| `P_BRAIN_ROI_SKULLSTRIP_DILATE` | `2` | Dilation iterations used by skullstrip-related masks in some ROI utilities. |
| `P_BRAIN_ROI_SCORE_SPACE` | `signal` | Score space used by deterministic candidate scoring (`signal` or `ctc`, depending on module). |
| `P_BRAIN_ROI_MID_Z_FRAC` | `0.30` | Fractional z-slab used by some ROI sampling/debug routines (middle of the volume). |
| `P_BRAIN_ROI_AP_FLIP` | `0` | If enabled, flips anterior/posterior heuristics (useful for flipped datasets). |
| `P_BRAIN_ROI_AP_GAMMA` | `1.5` | Gamma used to emphasize anterior/posterior weighting in some heuristics. |
| `P_BRAIN_ROI_WRITE_SUMMARY_PLOTS` | `1` | If enabled, writes ROI QC montages and debug figures. |
| `P_BRAIN_ROI_SUMMARY_FONTSIZE` | `18` | Font size for ROI summary figures/montages. |

### Deterministic ROI: vessel candidate selection (PCA/peak-based)
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_ROI_VESSEL_PCT` | `99.0` | Start percentile for vascular candidate selection. |
| `P_BRAIN_ROI_VESSEL_MIN_PCT` | `95.0` | Minimum percentile allowed when relaxing thresholds. |
| `P_BRAIN_ROI_VESSEL_MIN_VOX` | `1200` | Minimum voxel count target for vascular candidates. |
| `P_BRAIN_ROI_AV_SEED_PCT` | `99.5` | Seed percentile for artery/vein seed selection. |
| `P_BRAIN_ROI_AV_SEED_MIN_PCT` | `95.0` | Minimum seed percentile when relaxing. |
| `P_BRAIN_ROI_AV_SEED_MIN_VOX` | `200` | Minimum voxel count for seed masks. |
| `P_BRAIN_ROI_AV_MIN_SEP_FRAMES` | `2` | Minimum separation (frames) between artery/vein timing peaks. |
| `P_BRAIN_ROI_AV_TTP_LOGISTIC_SCALE` | `2.0` | Logistic scale used for artery/vein timing split weighting. |

### Deterministic ROI: PCA debug and clustering
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_ROI_USE_PCA` | `1` | If enabled, uses PCA-derived curve-family logic for artery/vein separation. |
| `P_BRAIN_ROI_DEBUG_PCA` | `1` | If enabled, writes PCA debug figures/masks. |
| `P_BRAIN_ROI_DEBUG_PCA_MIDZ_ONLY` | `1` | If enabled, restricts PCA sampling to a middle z-slab. |
| `P_BRAIN_ROI_PCA_MIN_VOXELS` | `1500` | Minimum vascular voxels required to run PCA debug. |
| `P_BRAIN_ROI_PCA_MAX_VOXELS` | `20000` | Maximum voxels sampled for PCA debug. |
| `P_BRAIN_ROI_PCA_COMPONENTS` | `3` | Number of PCA components used. |
| `P_BRAIN_ROI_PCA_N_CLUSTERS` | `6` | k-means cluster count in PCA score space. |
| `P_BRAIN_ROI_PCA_SEED` | `0` | Seed for k-means initialization. |

PCA debug outputs are written under `Analysis/ROI Debug/` and include:
- `pca_curve_families.png`
- `pca_space_clusters.png` (PC1 vs PC2, dpi=300)
- `pca_family_mask_cluster_*.nii.gz`, `pca_family_mask_artery.nii.gz`, `pca_family_mask_vein.nii.gz`

### Deterministic ROI: ICA constraints and connected-components
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_ROI_ICA_COMPONENT_PCT` | `99.5` | Start percentile for ICA component candidate thresholding. |
| `P_BRAIN_ROI_ICA_COMPONENT_MIN_PCT` | `95.0` | Minimum percentile when relaxing ICA thresholding. |
| `P_BRAIN_ROI_ICA_COMPONENT_PCT_STEP` | `0.5` | Percentile decrement step when relaxing ICA thresholds. |
| `P_BRAIN_ROI_ICA_MIN_AREA` | `8` | Minimum 2D component area (pixels) to consider per slice. |
| `P_BRAIN_ROI_ICA_MAX_AREA` | `800` | Maximum 2D component area (pixels) to consider per slice. |
| `P_BRAIN_ROI_ICA_MAX_COMPONENTS_PER_SLICE` | `3` | Max candidate components evaluated per slice. |
| `P_BRAIN_ROI_ICA_MAX_BOUNDARY_CONTACT` | `0.20` | Reject components that contact the brain boundary too heavily. |
| `P_BRAIN_ROI_ICA_EDGE_MARGIN` | `4` | Ignore components too close to the image edge. |
| `P_BRAIN_ROI_ICA_MIN_LR_FROM_MIDLINE` | `0` | Optional minimum left/right distance from midline (in pixels). |
| `P_BRAIN_ROI_ICA_MIDLINE_EXCLUDE_BAND` | *(unset)* | Optional midline exclusion band (pixels) to prevent midline leakage. |
| `P_BRAIN_ROI_ICA_MIN_DIST_INNER` | `3` | Minimum distance inside the brain mask for ICA candidates. |
| `P_BRAIN_ROI_ICA_MIN_ZSPAN` | `1` | Minimum z-span (slices) for ICA 3D components when enabled. |
| `P_BRAIN_ROI_ICA_DILATE3D` | `1` | 3D dilation iterations applied to ICA mask after selection (if enabled). |
| `P_BRAIN_ROI_Z_ALIGN_MIN_ICA` | `0.0` | Minimum z-orientation score for keeping ICA components in 3D mode. |

ICA adjacent-slice augmentation:
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_ROI_ICA_ADD_ADJACENT` | `1` | If enabled, adds ICA voxels in slices above the lowest confirmed ICA slice when arterial evidence exists. |
| `P_BRAIN_ROI_ICA_ADJACENT_SLICES` | *(unset)* | Explicit number of slices to extend upward (overrides fraction). |
| `P_BRAIN_ROI_ICA_ADJACENT_FRAC` | `0.2` | Fraction of volume z-slices used when `..._SLICES` is unset (uses `ceil(frac * zdim)`). |
| `P_BRAIN_ROI_ICA_ADJ_SCORE_PCT` | `99.5` | Percentile threshold used for arterial-evidence fallback when extending. |
| `P_BRAIN_ROI_DEBUG_ICA_ADJ` | `0` | Debug-print which adjacent slices were added. |

### Deterministic ROI: SSS constraints
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_SSS_MAX_AREA` | `220` | Maximum 2D component area (pixels) allowed for SSS. |
| `P_BRAIN_SSS_MAX_DIST_INNER` | `5.0` | Max distance-from-brain-interior used in SSS scoring. |
| `P_BRAIN_SSS_MAX_BOUNDARY_CONTACT` | `0.55` | Maximum boundary contact fraction for SSS candidates. |
| `P_BRAIN_SSS_MIN_BOUNDARY_CONTACT` | `0.00` | Optional minimum boundary contact fraction for SSS candidates. |
| `P_BRAIN_SSS_MAX_LR_SPAN` | `18` | Maximum left/right span (pixels) for SSS candidates. |
| `P_BRAIN_SSS_MAX_ELONG` | `6.0` | Maximum elongation allowed for SSS candidates. |
| `P_BRAIN_SSS_ELONG_PEN` | `1.0` | Penalty scale applied when SSS candidates are overly elongated. |
| `P_BRAIN_SSS_POSTERIOR_BONUS` | `0.5` | Posterior bonus weight for SSS scoring. |
| `P_BRAIN_ROI_SSS_MIN_ZSPAN` | `2` | Minimum z-span (slices) for SSS 3D components. |
| `P_BRAIN_ROI_SSS_MIDLINE_BAND` | *(code default)* | Midline band used to constrain SSS. |
| `P_BRAIN_ROI_Z_ALIGN_MIN_SSS` | *(code default)* | Minimum z-orientation score for keeping SSS components in 3D mode. |

### Deterministic ROI: basilar artery constraints (optional)
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_ROI_BASILAR_PCT` | `99.0` | Start percentile for basilar candidate selection. |
| `P_BRAIN_ROI_BASILAR_MIN_PCT` | `97.0` | Minimum percentile when relaxing basilar thresholds. |
| `P_BRAIN_ROI_BASILAR_PCT_STEP` | `0.5` | Percentile decrement step for basilar selection. |
| `P_BRAIN_ROI_BASILAR_MIN_AREA` | `6` | Minimum basilar component area (pixels). |
| `P_BRAIN_ROI_BASILAR_MAX_AREA` | `400` | Maximum basilar component area (pixels). |
| `P_BRAIN_ROI_BASILAR_MIDLINE_BAND` | *(auto)* | Midline band used to constrain basilar candidates. |
| `P_BRAIN_ROI_BASILAR_SLICES` | `1` | Number of slices considered for basilar selection. |
| `P_BRAIN_ROI_BASILAR_MIN_ZSPAN` | `1` | Minimum z-span (slices) for basilar 3D components. |
| `P_BRAIN_ROI_BASILAR_MIN_DIST_INNER` | `2` | Minimum distance inside brain mask for basilar candidates. |
| `P_BRAIN_ROI_BASILAR_MAX_BOUNDARY_CONTACT` | `0.50` | Maximum boundary contact fraction for basilar candidates. |
| `P_BRAIN_ROI_Z_ALIGN_MIN_BASILAR` | *(inherits)* | z-orientation score threshold for basilar selection. |

### Deterministic ROI: 3D component selection and debugging
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_ROI_USE_3D_COMPONENTS` | `1` | If enabled, selects ROIs via 3D connected-components rather than slice-by-slice only. |
| `P_BRAIN_ROI_KEEP_Z_ORIENTED` | `1` | If enabled, prefers components aligned along the z-axis. |
| `P_BRAIN_ROI_Z_ALIGN_MIN` | `0.55` | Minimum z-orientation score for keeping 3D components (global default). |
| `P_BRAIN_ROI_COMP3D_MIN_VOX` | `40` | Minimum voxels for considering a 3D component. |
| `P_BRAIN_ROI_DEBUG` | `0` | If enabled, writes extra ROI debug NIfTIs/figures. |
| `P_BRAIN_ROI_DEBUG_STRICT` | `0` | If enabled, raises more aggressively on ROI debug failures. |
| `P_BRAIN_ROI_DEBUG_3D` | `0` | If enabled, writes 3D-component debug exports. |
| `P_BRAIN_ROI_DEBUG_MONTAGE_MAX_SLICES` | `12` | Max slices shown in ROI debug montages. |
| `P_BRAIN_DEBUG_VEIN_MIN_DIST_INNER` | `2.0` | Debug-only vein selection constraint used in some ROI debug plots. |

### Legacy (deprecated) geometry ROI knobs
These variables remain for backward compatibility with older geometry-based ROI behaviour and/or older debugging paths:
| Env var | Description |
|---|---|
| `P_BRAIN_ROI_RICA_SLICES`, `P_BRAIN_ROI_RICA_Z_RANGE`, `P_BRAIN_ROI_RICA_RADIUS` | Legacy geometry ROI parameters for rICA. |
| `P_BRAIN_ROI_LICA_SLICES`, `P_BRAIN_ROI_LICA_Z_RANGE`, `P_BRAIN_ROI_LICA_RADIUS` | Legacy geometry ROI parameters for lICA. |
| `P_BRAIN_ROI_SSS_SLICES`, `P_BRAIN_ROI_SSS_Z_RANGE`, `P_BRAIN_ROI_SSS_RADIUS` | Legacy geometry ROI parameters for SSS. |

### Diffusion / tractography
Diffusion file priority (which NIfTI is selected when multiple are present):
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_DIFFUSION_PRIORITY` | `dti,dwi_reg,dwi_iso,dwi` | Comma-separated file-group preference order (see `utils/parameters.py`). |

Per-group diffusion model overrides:
| Env var | Default | Description |
|---|---:|---|
| `P_BRAIN_DIFFUSION_MODEL_DTI` | `DTI` | Model for the `dti` group (`DTI` or `CSD`). |
| `P_BRAIN_DIFFUSION_MODEL_DWI` | `CSD` | Model for the `dwi` group (`DTI` or `CSD`). |
| `P_BRAIN_DIFFUSION_MODEL_DWI_REG` | `CSD` | Model for the `dwi_reg` group (`DTI` or `CSD`). |
| `P_BRAIN_DIFFUSION_MODEL_DWI_ISO` | `CSD` | Model for the `dwi_iso` group (`DTI` or `CSD`). |

Tractography tuning (advanced):
| Env var | Description |
|---|---|
| `P_BRAIN_ENABLE_FURY`, `P_BRAIN_DISABLE_FURY` | Force-enable/disable FURY rendering backend if available. |
| `P_BRAIN_DEBUG_TRACKS` | Enables additional tractography debug logs/exports. |
| `P_BRAIN_TRACK_FORCE_MT_CSD` | Forces MSMT-CSD solver even when `--orientation csd` is used. |
| `P_BRAIN_TRACK_CSD_SH_ORDER`, `P_BRAIN_TRACK_QBALL_SH_ORDER` | Spherical-harmonic order controls (often `auto`). |
| `P_BRAIN_TRACK_WORKERS`, `P_BRAIN_TRACK_PARALLEL_BACKEND` | Parallelism controls for tractography attempts. |
| `P_BRAIN_TRACK_*` | Many additional tuning knobs exist (lengths, thresholds, seeding, ACT/PFT toggles, animation export). See `modules/tractography.py` for details. |

### Figure export styling (PNG/tiles)
These mostly affect the mosaic/tile helpers under `utils/plotting.py`:
| Env var | Description |
|---|---|
| `PBRAIN_PNG_DPI` | Default dpi for PNG exports. |
| `PBRAIN_TURBO` | Use a high-contrast colormap preset where supported. |
| `PBRAIN_NO_COLOR` | Disable colored colormaps (grayscale outputs). |
| `PBRAIN_OUTER_MARGIN_PX`, `PBRAIN_TILE_INNER_MARGIN_PX`, `PBRAIN_TILE_GAP_PX` | Layout spacing for tile/mosaic figures. |
| `PBRAIN_TRANSPARENT_GUTTERS`, `PBRAIN_TRANSPARENT_COLORBAR_BG` | Transparency controls for gutters/colorbar backgrounds. |
| `PBRAIN_COLORBAR_UNITS_POSITION`, `PBRAIN_COLORBAR_UNITS_GAP_PX` | Controls where unit labels are placed on colorbars. |
| `PBRAIN_COLORBAR_TICK_FONT_SIZE`, `PBRAIN_COLORBAR_UNITS_FONT_SIZE` | Font sizes for colorbar ticks/units. |

Edit the Python files directly for permanent defaults or export environment variables for per-run overrides.

---

## Addons
Addons extend the manual workflow via menu option 7:
- **Boundary addon** – Generates GM/WM boundary ROIs (via `fsl_anat`) and associated CTCs.
- **Screenshot addon** – Navigate through axial slices and export presentation-quality PNG images.

Initialize individual addons with:
```bash
git submodule update --init -- addons/<addon_name>
```

---

## Contributing & support
- Open issues or feature requests on GitHub.
- For direct contact, reach out to Edis Tireli.
- Pull requests should follow the existing directory layout and reference the appropriate configuration flags in `utils/settings.py` and `utils/parameters.py`.

---

## License & acknowledgments
- **License:** MIT (see `LICENSE`).
- **Acknowledgments:** Henrik B. W. Larsson, Ulrich Lindberg, Stig P. Cramer, Mark Vestergaard, and Antonis Asiminas for continuous collaboration and discussions.

_p_-Brain is developed within the Functional Imaging Unit, Department of Clinical Physiology and Nuclear Medicine, Copenhagen University Hospital – Rigshospitalet, and the University of Copenhagen. The released CNN weights are available on Zenodo for reproducible deployment.
