# p-Brain: Advanced Neuroimaging Analysis Tool

![411586563-4f8bdcba-dbe6-41c7-b644-c0cdaf031fe9](https://github.com/user-attachments/assets/3376d87a-26af-4b73-ba57-e906aaf8e13c)

## AI powered results

![AI_input_function_ROIs](https://github.com/user-attachments/assets/17844819-7fde-4dd5-8ef1-a9a3151bb5c4)
![AI_Tissue_slice_5_segmented_median](https://github.com/user-attachments/assets/328c1a43-294d-42fa-bad5-518bd1af8439)
![kipervoxel](https://github.com/user-attachments/assets/560a42eb-1670-4b0a-a3ae-6bb3c004b359)
![kimap](https://github.com/user-attachments/assets/e80c23e6-8880-4c63-bd6e-7a10080ad9fe)
![cbfmap](https://github.com/user-attachments/assets/2fe56e0b-1d29-4f89-88f9-175105d4436e)

Author: Edis Devin Tireli, M.Sc, Ph.D. student

Affiliation: [Copenhagen University](https://www.ku.dk/english/)

# Table of Contents
1. Introduction
2. Data Requirements
3. Directory & Data Structure
4. Installation
5. How to use
6. Core Features
7. Fully Automated Mode
8. Control Data & Quality Corrections
9. Contributions
10. License
11. Acknowledgments

## 1. Introduction
p-Brain is a Python toolkit for quantitative MRI analysis focusing on dynamic contrast-enhanced (DCE) protocols. It supports both Philips PAR/REC and NIfTI files and covers the entire workflow from T1/M0 fitting, input function extraction and tissue segmentation to blood-brain barrier permeability estimation.

Core functionality lives in `modules/` while helper utilities reside in `utils/`. Neural networks for artery and vein identification are stored in the `AI/` directory. A small GUI is used to select the dataset, after which a terminal menu guides the user through each processing step. For cohort processing, `enumerator.py` launches `main.py` for multiple subjects in sequence.

## 2. Data Requirements
p-Brain expects a set of MRI sequences. When running the fully automated mode only a subset is needed. Filenames can be customised in `utils/parameters.py`.

### Minimum for fully automated analysis
- **t1_3D_filename** – 3D T1-weighted scan used for FastSurfer segmentation
- **axial_t2_2D_filename** – axial 2D T2-weighted image (same geometry as DCE)
- **dce_filename** – dynamic contrast-enhanced sequence
- **WIPTI_xxxxx.nii** – inversion recovery series for T1/M0 fitting

### Additional files for manual ROI delineation
If you plan to draw ROIs manually, p-Brain can utilise extra sequences for better visualisation:
- **flair_3D_filename** – 3D FLAIR
- **t2_3D_filename** – 3D T2
- **axial_flair_3D_filename** – axial reconstruction of FLAIR
- **axial_t2_3D_filename** – axial reconstruction of T2
- **t1_3D_filename** and **axial_t1_3D_filename** – optional when delineating ROIs yourself

Control datasets may use alternative filenames. These are defined in `control_filenames` inside `utils/parameters.py`.

## 3. Directory & Data Structure
The data directory should be organised as follows:

```text
data
└── data_1
    ├── x.PAR
    ├── x.REC
    └── Analysis
        ├── TSCC Data
        ├── CTC Data
        ├── ITC Data
        └── ROI Data
    └── Images
    └── NIfTI
└── data_2
...
```
Place your MRI data under `data/` using any folder name. p-Brain will create the `Analysis`, `Images` and `NIfTI` subdirectories automatically.

### Repository overview
- **modules/** – implementation of the menu options such as T1 fitting and permeability models
- **utils/** – helper utilities for plotting, configuration and file handling
- **AI/** – default neural network models used in the fully automated mode
- **addons/** – optional plugins, e.g. boundary ROI extraction
- **enumerator.py** – script to process multiple datasets automatically

## 4. Installation

### Required
1. **Clone the repository**
   ```bash
   git clone https://github.com/edtireli/p-brain.git
   ```
2. **Navigate to the directory**
   ```bash
   cd p-brain
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Optional addon installation
To enable addons run:
```bash
git submodule update --init -- addons/addon_name
```

The application derives its version directly from Git tags (`modules/__init__.py`).

## 5. How to use
Start p-Brain with:
```bash
python3 main.py
```
Select the desired dataset in the GUI and the following CLI menu appears:
```bash
         Welcome to p-brain - a neuroimaging & analysis tool
=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=

        / /                                                  / /
       / /    eeeee      eeeee  eeeee  eeeee e  eeeee       / /
      / /     8   8      8   8  8   8  8   8 8  8   8      / /
eeee / /      8eee8 eeee 8eee8e 8eee8e 8eee8 8e 8e  8     / /    eeee
    / /       88         88   8 88   8 88  8 88 88  8    / /
   / /        88         88eee8 88   8 88  8 88 88  8   / /


                  Developed by Edis Devin Tireli
                     University of Copenhagen
=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=
=-=-= Choose between the following options =-=-==-=-==-=-==-=-==-=-=-=
| 0 | View MRI images
| 1 | Compute M0 and T1 map from MRI data
| 2 | Generate Concentration Time Curves (CTC) based on ROI
| 3 | Generate Time-Shifted Concentration Curves (TSCC) from CTC
| 4 | Create Tissue (Grey/White matter) CTCs
| 5 | Compute BBB permeability and perfusion parameters
| 6 | Add analysis notes
| 7 | Addons
| 9 | Exit program
=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=
[!] Enter option (1-9):
```
Use the options in order; each step depends on the previous.

## 6. Core Features
- **Option 0** – View MRI images
- **Option 1** – T1/M0 fitting using the inversion recovery series
- **Option 2** – Generate concentration time curves from a manually drawn ROI
  <img width="375" alt="Figure_1_github" src="https://github.com/edtireli/p-brain/assets/129996957/370ecb97-7dae-4148-b60e-93b72bfab24c">
  <img width="375" alt="Figure_2_github" src="https://github.com/edtireli/p-brain/assets/129996957/8f464073-1c6f-4cf2-91ff-5131f36bcfd5">
- **Option 3** – Time shifting of venous and arterial curves
- **Option 4** – Tissue concentration curves via ROI selection
- **Option 5** – BBB permeability estimation (Patlak and Extended Tofts)
- **Automatic DWI processing** generates FA maps when DWI data are present
- **Addons**
  - *Boundary*: compute GM/WM boundary concentration curves
    ![CTC+ROI_slice_7](https://github.com/edtireli/p-brain/assets/129996957/32bc922a-dcce-4b9c-a31e-053d021351e4)
  - *Screenshot*: save reconstructed axial T1 slices

## 7. Fully Automated Mode
Four neural networks identify carotid artery and sinus sagittalis vein slices and ROIs. FastSurfer performs rapid brain segmentation. The pipeline automatically conducts T1/M0 fitting, ROI drawing, tissue segmentation and Patlak analysis, yielding BBB permeability and cerebral blood flow maps.

## 8. Control Data & Quality Corrections
- **Control datasets**: create a `controls` folder inside `data` and place subfolders there (for example `data/controls/log1`). Enable via the `CONTROLS` flag in `utils/settings.py` or by setting `PBRAIN_CONTROLS=1`. The `enumerator.py` script sets this automatically when invoked with `--controls`.
- **Signal jump correction**: placing `apply_jumpfix.json` next to the analysis directory enables automatic correction of signal jumps in tissue curves.
- **Motion/registration**: set `USE_FLIRT_REGISTRATION=1` to use FLIRT-based two-step registration when aligning masks to DCE and T2 images.

Example enumerator usage:
```bash
python enumerator.py 1001 1002
python enumerator.py --controls 01 02
python enumerator.py --all
python enumerator.py --controls --all
```

NIfTI files can be used directly by placing them in a folder with the same name. This bypasses conversion from PAR/REC.

### Custom AI model paths
Neural network paths can be changed in `utils/settings.py` (`AI_MODEL_PATHS`) or via the environment variables `SLICE_CLASSIFIER_RICA_MODEL`, `RICA_ROI_MODEL`, `SLICE_CLASSIFIER_SS_MODEL` and `SS_ROI_MODEL`.

## 9. Contributions
For feature requests and bug reports please contact Edis Tireli or open an issue.

## 10. License
This project is licensed under the MIT License.

## 11. Acknowledgments
Special thanks to Henrik B. W. Larsson for collaborations and discussions.
