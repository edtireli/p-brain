# p-Brain: Advanced Neuroimaging Analysis Tool

Author: Edis Devin Tireli, M.Sc, Ph.D. student

Affiliation: Copenhagen University

Contact: edis.devin.tireli@regionh.dk

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
Data
└── Data_1
    ├── x.PAR
    ├── x.REC
    └── Analysis
        ├── TSCC Data
        ├── CTC Data
        ├── ITC Data
        └── ROI Data
└── Data_2
...
```
Place your .PAR/.REC MRI data in the 'Data' directory under the appropriate data folder. p-Brain will create subdirectories within the data folders (Data_1, Data_2 etc.) automatically. The names of the data folders are irrelevant, but listed above as data_1 and data_2 for clarity. 

## 3. Installation

To install p-Brain, simply clone the repository to your local machine:

1. Clone this repository to your local machine.
    ```bash
    git clone https://github.com/edtireli/p-brain.git
    ```
2. Navigate to the cloned directory.
    ```bash
    cd p-brain
    ```
3. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```

Now you are ready to run the program.


## 4. Command-Line Interface Usage
To start p-Brain, navigate to the project directory and execute the following command:


The program will prompt you for various inputs corresponding to the task you want to perform. Follow the on-screen instructions to complete the process.


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
Special thanks to Henrik B. W. Larsson for collaboration and discussion


