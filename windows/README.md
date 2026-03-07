# p-brain Windows CLI

Pure-Python command-line interface for the p-brain neuroimaging analysis
pipeline.  **No FreeSurfer, FastSurfer, or FSL required.**

## Overview

The Windows CLI produces **identical outputs** to the macOS desktop version
(v1.2.8).  All neuroimaging operations that normally call FreeSurfer / FSL
command-line tools are replaced with pure-Python equivalents:

| macOS tool       | Windows replacement                 | Library   |
|-----------------|--------------------------------------|-----------|
| `mri_convert`   | `neuroimaging.mgz_to_nifti()`        | nibabel   |
| `mri_binarize`  | `neuroimaging.binarize_labels()`     | numpy     |
| `fslmaths`      | `neuroimaging.fslmaths_chain()`      | numpy     |
| `flirt`         | `neuroimaging.affine_coregister()`   | scipy     |
| FastSurfer      | `segmentation.run_synthseg()`        | SynthSeg  |

## Prerequisites

1. **Python 3.9+** (3.10 or 3.11 recommended)
2. **SynthSeg** — clone the repo and download models:
   ```bash
   git clone https://github.com/BBillot/SynthSeg.git vendor/SynthSeg
   # Download models per https://github.com/BBillot/SynthSeg#models
   ```
3. Install Python dependencies:
   ```bash
   pip install -r windows/requirements.txt
   ```

## Usage

From the `p-brain/` root directory:

```bash
# Basic run (Patlak model, auto T1 fitting)
python -m windows --id 20230403x3 --data-dir /path/to/data

# Specify SynthSeg location
python -m windows --id 20230403x3 --synthseg-home /path/to/SynthSeg

# Force CPU (no GPU)
python -m windows --id 20230403x3 --cpu

# Tikhonov model with custom lambda
python -m windows --id 20230403x3 --pk-model tikhonov --lambda 0.01

# Both Patlak and Tikhonov
python -m windows --id 20230403x3 --pk-model both

# VFA T1 fitting
python -m windows --id 20230403x3 --t1-fit vfa

# Force mask re-creation
python -m windows --id 20230403x3 --force-masks
```

## Architecture

```
windows/
├── __init__.py          # Package marker + version
├── __main__.py          # python -m windows entry point
├── cli.py               # argparse CLI (mirrors main.py flags)
├── neuroimaging.py      # Pure-Python FreeSurfer/FSL replacements
├── pipeline.py          # Orchestrator (mirrors main.py auto mode)
├── segmentation.py      # SynthSeg wrapper
├── requirements.txt     # Windows-specific dependencies
└── README.md            # This file
```

### Separation from macOS

The Windows CLI is **completely self-contained** in the `windows/` directory.
It does not modify any existing macOS code.  It *imports* (but does not change)
the shared pure-Python modules:

- `modules/opt01_T1_fit.py` — T1/M0 fitting
- `modules/opt03_time_shifting.py` — Time-shifted concentration curves
- `modules/AI_tissue_functions.py` — CTC computation, Patlak/Tikhonov modelling
  (only the pure-Python functions; the `segmentation()` and `coregistration()`
  functions that call FreeSurfer/FSL are **not** used on Windows)
- `modules/input_function_dispatch.py` — ROI extraction
- `utils/settings.py`, `utils/plotting.py`, etc.

### Output compatibility

The Windows CLI writes the same directory structure and file names as the macOS
version:

```
Data/<subject>/Analysis/
├── Fitting/
│   ├── voxel_T1_matrix.pkl
│   ├── voxel_M0_matrix.pkl
│   └── time_points_s.npy
├── InputFunction/
├── Plots/
└── ...
Data/<subject>/NIfTI/segmentation/segmentation/mri/
├── aparc.DKTatlas+aseg.deep.nii.gz   ← SynthSeg output
├── cortical_gm.nii.gz
├── subcortical_gm.nii.gz
├── wm.nii.gz
├── gm_brainstem.nii.gz
├── gm_cerebellum.nii.gz
├── wm_cerebellum.nii.gz
└── wm_cc.nii.gz
```

## Notes

- The **AI ROI method** (`--roi-method ai`) is not available on Windows because
  it requires a TensorFlow model that is only distributed with the macOS app.
  Use `--roi-method deterministic` (default) or `--roi-method file` instead.
- SynthSeg outputs at **1 mm isotropic resolution** using the same FreeSurfer
  label IDs as FastSurfer's `aparc.DKTatlas+aseg.deep.mgz`.
- On Windows, all `subprocess.run()` calls to external tools are eliminated;
  everything runs in-process via numpy/scipy/nibabel.
