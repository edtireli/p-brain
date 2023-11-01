# p-Brain: Advanced Neuroimaging Analysis Tool

Author: Edis Devin Tireli, M.Sc, Ph.D. student

Affiliation: Copenhagen University

# Table of Contents
1. Introduction
2. Directory & Data Structure
3. Installation
4. How to use
5. Core Features
6. Contributions
7. License
8. Acknowledgments


## 1. Introduction
p-Brain is a state-of-the-art neuroimaging tool developed for in-depth analysis of .PAR/.REC MRI data. The application is engineered to perform critical tasks such as T1/M0 fitting, creation of signal time curves, and plotting concentration curves for arteries and tissues. Additionally, the tool offers sophisticated methods for Grey Matter/White Matter (GM/WM) segmentation and Blood-Brain Barrier (BBB) permeability estimation, utilizing both the Patlak and the extended Tofts models. GUI modules are incorporated for high-precision drawing of Regions of Interest (ROIs) across tissue and arterial/venous structures.

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
Place your .PAR/.REC MRI data in the 'data' directory under the appropriate data folder. p-Brain will create subdirectories within the data folders (data_1, data_2 etc.) automatically. The names of the data folders are irrelevant, but listed above as data_1 and data_2 for clarity.  Further, analysis images will be placed in the Images subfolder, and the NIfTI files will be placed in the subfolder of the same name.

NIfTI files can also be used directly by simply creating a folder of the same name and placing the .nii files therein. This will avoid the automatic conversion from .PAR/.REC to .nii/.json. 

There are several files that are important to the analysis, I will list the sequence names below: 

- WIPAxT2TSEmatrix.nii: An axial 2D T2 weighted image.
- WIPhperf120long.nii: The data for the dynamic contrast-enhanced (DCE) sequence.
- WIPTI_xxxxx.nii: A series of n inversion recovery sequences where the x's are times in ms (by default set to 120, 300, ..., 1e5).
- Extra: The following files are not needed for the minimal case, but p-brain has an extended behavior (e.g. in GUI or plotting) if they are available.
    - WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii: A 3D FLAIR sequence.
    - WIPcs_3D_Brain_VIEW_T2_32chSHC.nii: A 3D T2 sequence.
    - WIPcs_T1W_3D_TFE_32channel.nii: A 3D T1 sequence.
    - axVWIPcs_3D_Brain_VIEW_FLAIR_SHC.nii: An axial reconstruction of the 3D FLAIR sequence above.
    - axVWIPcs_3D_Brain_VIEW_T2_32chSHC.nii: An axial reconstruction of the 3D T2 sequence above.
    - axVWIPcs_T1W_3D_TFE_32channel.nii: : An axial reconstruction of the 3D T1 sequence above.

The above files can be renamed to suit different purposes/sequences. See below for some of the most useful, especially with the _boundary_ addon: 
![correlated_slices](https://github.com/edtireli/p-brain/assets/129996957/e2c952ea-25ce-431b-bedd-a3eb24e49d67)


## 3. Installation

To get started with p-Brain, please follow the steps below to install the software on your local machine.

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
  <img width="1277" alt="Screenshot 2023-11-01 at 18 34 13" src="https://github.com/edtireli/p-brain/assets/129996957/c27ff5d8-1b9c-4bda-8af6-25655fe4da42">
- Option 2 - Concentration Time Curve (CTC) Generation: Generates signal time curves from the MRI data from a ROI drawn on DCE data (GUI): using the DCE file for this step.
 
  <img width="470" alt="Figure_1_github" src="https://github.com/edtireli/p-brain/assets/129996957/370ecb97-7dae-4148-b60e-93b72bfab24c">
  <img width="470" alt="Figure_2_github" src="https://github.com/edtireli/p-brain/assets/129996957/8f464073-1c6f-4cf2-91ff-5131f36bcfd5">

- Option 3 - Time shifting:  The venous CTC is shifted in time to the arterial CTC this is done by peak analysis so that a sufficient input function can be used. If the arterial CTC has taller peaks than the venous, then the venous curve is also rescaled to match.
- Option 4 - Tissue Concentration Time Curves: Generates Tissue CTCs via ROI selection (GUI) in the same way as in Option 2: using the DCE file and the 2D T2W image for this step.
- Option 5 - BBB Permeability Estimation: Estimates BBB permeability using both Patlak and the extended Tofts models.
- Addons:
    -  Boundary: Computes the concentration time function for Grey Matter/White Matter boundary (segmented with fsl_anat).
    ![CTC+ROI_slice_7](https://github.com/edtireli/p-brain/assets/129996957/32bc922a-dcce-4b9c-a31e-053d021351e4)
    -  Screenshot: A simple screenshot module that takes the reconstructed axial T1 slice, presents the user with a GUI to move through slices, and then a button to save the image to a png. 

## 6. Contributions
For contributions, feature requests, and bug reporting, please contact me (Edis Tireli) through here, or add an issue. 

## 7. License
This project is licensed under the MIT License. For full license information, please refer to the LICENSE.md file in the repository.

## 8. Acknowledgments
Special thanks to Henrik B. W. Larsson for collaborations and discussions.


