# _p_-Brain: Advanced Neuroimaging Analysis Tool
![411586563-4f8bdcba-dbe6-41c7-b644-c0cdaf031fe9](https://github.com/user-attachments/assets/3376d87a-26af-4b73-ba57-e906aaf8e13c)

Author: Edis Devin Tireli, M.Sc, Ph.D. student

Affiliation: [Copenhagen University](https://www.ku.dk/english/)

## Motivation
Understanding blood-brain barrier permeability is essential for assessing neurological disease and treatment response. _p_-Brain integrates segmentation, neural network-based ROI detection, and advanced kinetic modeling to compute the blood-brain barrier permeability (Ki) and cerebral blood flow (CBF). The toolkit can operate fully automatically, minimizing user error while allowing manual ROI delineation when desired. Patlak analysis, Tikhonov model-free deconvolution and Fick's two-compartment model underpin the Ki estimation workflow.


# Table of Contents
1. Introduction
2. Directory & Data Structure
3. Installation
4. How to use
5. Core Features
6. Fully Automated Mode
7. Contributions
8. License
9. Acknowledgments


## 1. Introduction
_p_-Brain is a Python toolkit for quantitative analysis of MRI data with a focus on dynamic contrast-enhanced (DCE) protocols. It supports both Philips PAR/REC and NIfTI files and provides a set of modules for converting, viewing and processing images. The pipeline covers the full workflow from T1/M0 fitting, input function extraction and tissue segmentation to blood‑brain barrier permeability estimation.

All core functionality resides in the `modules/` package while helper routines are located under `utils/`. Optional neural networks for artery and vein identification are stored in the `AI/` directory. A small GUI is used to select a dataset, after which a terminal menu guides the user through each processing step. For cohort processing the script `enumerator.py` can launch `main.py` for multiple subjects in sequence, enabling unattended analysis.

## 2. Directory & Data Structure
The software expects a specific directory structure for optimal functioning. The MRI data to be analysed upon should be placed within the Data folder as follows:

```
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
Place your .PAR/.REC MRI data in the `data` directory under any folder name of your choosing. p-Brain will create the required subdirectories (e.g. `Analysis`, `Images`, `NIfTI`) automatically. NIfTI files generated from PAR/REC input or provided directly are stored under `NIfTI`, while derived figures are written to the `Images` directory.

### Repository overview
- **modules/** – menu implementations such as T1 fitting and permeability models.
- **utils/** – utilities for configuration, plotting and file handling.
- **AI/** – default neural network models used for the automated pipeline.
- **addons/** – optional plugins (e.g. boundary ROI extraction).
- **enumerator.py** – helper script that invokes `main.py` for multiple datasets.

Place control datasets in `data/controls/<id>` and enable control handling by
setting the `CONTROLS` flag in `utils/settings.py` or by exporting
`PBRAIN_CONTROLS=1`. The `enumerator.py` helper honours the `--controls` flag
and will set this environment variable automatically. When a control run is
processed, a small `control.json` file is written to the dataset directory.  The
script accepts a list of log numbers or the `--all` flag. Combining
`--controls` with `--all` processes every available control dataset.

Placing an `apply_jumpfix.json` file next to a dataset enables automatic
correction of sudden signal jumps during the analysis.

Usage examples:
```
python enumerator.py 1001 1002
python enumerator.py --all
python enumerator.py --controls 01 02
python enumerator.py --controls --all
```

NIfTI files can also be used directly by simply creating a folder of the same name and placing the .nii files therein. This will avoid the automatic conversion from .PAR/.REC to .nii/.json. 

There are several files that are important to the analysis, I will list the variables below, which can be changed in the parameters.py file to fit your naming scheme: 

- **axial_t2_2D_filename**: An axial 2D T2 weighted image that is in the same geometry as the DCE below. It can in essence also be a T1 weighted image, this is simply a naming convention.
- **dce_filename**: The data for the dynamic contrast-enhanced (DCE) sequence. In the case of the default file, there are actually two filenames that this file can take. If your file only has one filename, then simply ignore the previous lines (dce_filename_primary, dce_filename_fallback) and name the dce_filename as you would otherwise. 
- **WIPTI_xxxxx.nii**: A series of n inversion recovery sequences where the x's are times in ms (by default set to 120, 300, ..., 1e5). It is very important that your inversion sequence files are named in the same manor, as this is hardcoded into the fitting proceedure. 
- Extra: The following files are not needed for the minimal case, but p-brain has an extended behavior (e.g. in GUI or plotting) if they are available.
    - **flair_3D_filename**: A 3D FLAIR sequence.
    - **t2_3D_filename**: A 3D T2 sequence.
    - **t1_3D_filename**: A 3D T1 sequence.
    - **axial_flair_3D_filename**: An axial reconstruction of the 3D FLAIR sequence above.
    - **axial_t2_3D_filename**: An axial reconstruction of the 3D T2 sequence above.
    - **axial_t1_3D_filename**: : An axial reconstruction of the 3D T1 sequence above.

### Control filenames
Control datasets may use different filenames. These can be configured in the `control_filenames` function inside `utils/parameters.py`. Only the sequences required by the AI methods are listed:
- **control_t1_3D_filename**
- **control_axial_t1_3D_filename**
- **control_t2_3D_filename**
- **control_axial_t2_3D_filename**
- **control_flair_3D_filename**
- **control_axial_flair_3D_filename**
- **control_axial_t2_2D_filename**
- **control_dce_filename**

The above files can be renamed to suit different purposes/sequences which can be done globally in the `utils/parameters.py` file. This file also contains a `SEGMENTATION_METHOD` setting that controls which tool is used for the automated tissue segmentation (default `fastsurfer`). See below for some of the most useful, especially with the _boundary_ addon:
![correlated_slices](https://github.com/edtireli/p-brain/assets/129996957/e2c952ea-25ce-431b-bedd-a3eb24e49d67)


## 3. Installation

To get started with p-Brain, please follow the steps below to install the software on your local machine. Before you do so, make sure you have python and git installed. 

### Required Installation

1. **Clone the Repository**: Clone the p-Brain repository to your local machine using the following command:
    ```bash
    git clone https://github.com/edtireli/p-brain.git
    ```

2. **Navigate to the Directory**: Change to the directory containing the cloned repository:
    ```bash
    cd p-brain
    ```

3. **Install Dependencies**: Install the required Python packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

At this point, p-Brain is installed and you can run the program.

### Optional Addon Installation

For advanced functionalities, p-Brain supports optional addons. To install the addons:

1. **Initialize the Submodule**: While in the root directory of the p-Brain repository, run the following command to initialize and update the `addons`:
    ```bash
    git submodule update --init -- addons/addon_name
    ```

By following these steps, the `boundary` addon will be available for use within p-Brain.

### Automatic Versioning

The application derives its version directly from the Git tags. `modules/__init__.py`
will read the most recent tag via `git describe --tags` when executed, so you do
not need to manually update the version string.

## 4. How to use
To start p-Brain, navigate to the project directory and execute the following command:
```bash
python3 main.py
```
This will open up a GUI in which the subfolders of the data folder will be enumerated. Here you will need to select a subfolder (e.g. analysis ID, exam ID, log number etc.). Thereafter, you can press Accept and after the GUI will close, the following command line interface will be printed in the terminal:

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

The idea is to use the options in chronological order, as each step requires the files of each subsequent step. See below for a detailed description of the options and features. The software displays usage instructions as well.  

## 5. Core Features

- Option 0 - View MRI images: This option opens a GUI that allows for the selection and viewing of axial and saggital images of the aforementioned data types.
- Option 1 - T1/M0 Fitting: This option utilizes a standard least-squares non-linear curve fitting algorithms for precise T1/M0 fitting: it uses the WIPTI_xxxx.nii files to do so.
- Option 2 - Concentration Time Curve (CTC) Generation: Generates signal time curves from the MRI data from a ROI drawn on DCE data (GUI): using the DCE file for this step.
  
  <img width="375" alt="Figure_1_github" src="https://github.com/edtireli/p-brain/assets/129996957/370ecb97-7dae-4148-b60e-93b72bfab24c">  
  <img width="375" alt="Figure_2_github" src="https://github.com/edtireli/p-brain/assets/129996957/8f464073-1c6f-4cf2-91ff-5131f36bcfd5">

- Option 3 - Time shifting:  The venous CTC is shifted in time to the arterial CTC this is done by peak analysis so that a sufficient input function can be used. If the arterial CTC has taller peaks than the venous, then the venous curve is also rescaled to match.
- Option 4 - Tissue Concentration Time Curves: Generates Tissue CTCs via ROI selection (GUI) in the same way as in Option 2: using the DCE file and the 2D T2W image for this step.
- Option 5 - BBB Permeability Estimation: Estimates BBB permeability using both Patlak and the extended Tofts models.
- Automatic DWI processing computes Fractional Anisotropy (FA) maps per patient when DWI data are present. When a white matter mask is available the mean FA of that region is additionally reported and a WM-restricted FA map is produced.
- Addons:
    -  Boundary: Computes the concentration time function for Grey Matter/White Matter boundary (segmented with fsl_anat).
    ![CTC+ROI_slice_7](https://github.com/edtireli/p-brain/assets/129996957/32bc922a-dcce-4b9c-a31e-053d021351e4)
    -  Screenshot: A simple screenshot module that takes the reconstructed axial T1 slice, presents the user with a GUI to move through slices, and then a button to save the image to a png. 

## 6. Fully Automated Mode
From v2.0.0 onwards, a new fully automated implementation is available within which 4 neural networks were trained on carotid artery and sinus sagitalis vein identification and ROI drawing, see below: 

![AI_input_function_ROIs](https://github.com/user-attachments/assets/adda1abb-c1b7-4a49-937b-cf64ea51acef)

Further a fast AI segmentation tool, FastSurfer, is also implemented and segments the brain within few minutes. Our pipeline now integrates both AI utilities to conduct the entire analysis automatically: T1/M0 fit, vein/artery ROI drawing, tissue segmentation/ROIs and the final Patlak analysis of the determination of Ki (BBB permeability, slice by slice and voxelwise) as well as a CBF map (using a 2-compartment model) and a Ki map, as well as a whole-volume Ki calculation. The CBF values are scaled to ml/100g/min. The image below shows an example of the results of one such automated result (slice-by-slice Ki determination)

![AI_Tissue_slice_5_segmented_median](https://github.com/user-attachments/assets/328c1a43-294d-42fa-bad5-518bd1af8439)

The pipeline produces a large collection of data in the form of .nii files, e.g. the BBB permeability per voxel, per slice (as shown above) per tissue type, and per parcel from segmentation, as well as a global estimation (using whole-brain tissue segments as one large ROI per major tissue type). The following is a per-voxel BBB permeability over few slices:

![kipervoxel](https://github.com/user-attachments/assets/560a42eb-1670-4b0a-a3ae-6bb3c004b359)

Below is the BBB permeability per parcel from segmentation: 

![kimap](https://github.com/user-attachments/assets/e80c23e6-8880-4c63-bd6e-7a10080ad9fe)

And finally, a CBF estimation per voxel: 

![cbfmap](https://github.com/user-attachments/assets/2fe56e0b-1d29-4f89-88f9-175105d4436e)


### Custom AI model paths
The four neural networks used for Right Internal Carotid Artery (RICA) and sinus
 sagittalis segmentation can be replaced with your own models. The default
 locations are defined in `utils/settings.py` under `AI_MODEL_PATHS`. Set these
 paths or provide the environment variables `SLICE_CLASSIFIER_RICA_MODEL`,
 `RICA_ROI_MODEL`, `SLICE_CLASSIFIER_SS_MODEL` and `SS_ROI_MODEL` to override the
 defaults.

Pre-trained models are available at
[Zenodo](https://doi.org/10.5281/zenodo.15655347). Download and extract them
into the `AI/` directory so the default paths resolve.

## 7. Contributions
For contributions, feature requests, and bug reporting, please contact me (Edis Tireli) through here, or add an issue. 

## 8. License
This project is licensed under the MIT License. For full license information, please refer to the LICENSE.md file in the repository.

## 9. Acknowledgments
Special thanks to Henrik B. W. Larsson, Ulrich Lindberg, Stig P. Cramer, Mark Vestergaard and Antonis Asiminas for collaborations and discussions.


