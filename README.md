# p-Brain: Advanced Neuroimaging Analysis Tool

Author: Edis Devin Tireli, M.Sc, Ph.D. student

Affiliation: Copenhagen University

# Table of Contents
1. Introduction
2. Directory Structure
3. Installation
4. Command-Line Interface Usage
5. Core Features
6. Contributions
7. License
8. Acknowledgments


## 1. Introduction
p-Brain is a state-of-the-art neuroimaging tool developed for in-depth analysis of .PAR/.REC MRI data. The application is engineered to perform critical tasks such as T1/M0 fitting, creation of signal time curves, and plotting concentration curves for arteries and tissues. Additionally, the tool offers sophisticated methods for Grey Matter/White Matter (GM/WM) segmentation and Blood-Brain Barrier (BBB) permeability estimation, utilizing both the Patlak and the extended Tofts models. GUI modules are incorporated for high-precision drawing of Regions of Interest (ROIs) across tissue and arterial/venous structures.

## 2. Directory Structure
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
Place your .PAR/.REC MRI data in the 'data' directory under the appropriate data folder. p-Brain will create subdirectories within the data folders (data_1, data_2 etc.) automatically. The names of the data folders are irrelevant, but listed above as data_1 and data_2 for clarity. 

Further, analysis images will be placed in the Images subfolder, and the NIfTI files will be placed in the subfolder of the same name.

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

For advanced functionalities, p-Brain supports optional addons. One such addon is `boundary`.

To install the `boundary` addon:

1. **Initialize the Submodule**: While in the root directory of the p-Brain repository, run the following command to initialize and update the `boundary` submodule:
    ```bash
    git submodule update --init -- addons/boundary
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
[!] Enter option (1-9): ^C% 
```

The idea is to use the options in chronological order, as each step requires the files of each subsequent step. 


## 5. Core Features

- T1/M0 Fitting: Utilizes advanced algorithms for precise T1/M0 fitting.
- Signal Time Curve Generation: Generates signal time curves from the MRI data.
- Concentration Curve Plotting: Plots the concentration curves for both arteries and tissues.
- GM/WM Segmentation: Conducts Grey Matter/White Matter segmentation.
- BBB Permeability Estimation: Estimates BBB permeability using both Patlak and ETofts models.
- ROI GUI: A dedicated GUI for drawing high-precision ROIs for tissue and arterial/venous concentration curves.

## 6. Contributions
For contributions, feature requests, and bug reporting, please contact Edis Devin Tireli at [Contact Information].

## 7. License
This project is licensed under the MIT License. For full license information, please refer to the LICENSE.md file in the repository.

## 8. Acknowledgments
Special thanks to Henrik B. W. Larsson for collaboration and discussion.


