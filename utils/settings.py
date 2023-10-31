import os
import sys
import matplotlib
matplotlib.use("TkAgg")

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_directories(log_number):
    base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_directory = os.path.join(base_path, 'Data', log_number)
    analysis_directory = os.path.join(data_directory, 'Analysis')
    nifti_directory = os.path.join(data_directory, 'NIfTI')
    image_directory = os.path.join(data_directory, 'Images')
    
    # Directories to create
    dirs_to_create = [
        analysis_directory,
        os.path.join(analysis_directory, 'TSCC Data'),
        os.path.join(analysis_directory, 'CTC Data'),
        os.path.join(analysis_directory, 'CTC Data', 'Artery'),
        os.path.join(analysis_directory, 'CTC Data', 'Vein'),
        os.path.join(analysis_directory, 'CTC Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'Grey Matter'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'White Matter'),
        os.path.join(analysis_directory, 'CTC Data', 'Tissue', 'Boundary'),
        os.path.join(analysis_directory, 'ITC Data'),
        os.path.join(analysis_directory, 'TSCC Data', 'Max'),
        os.path.join(analysis_directory, 'ITC Data', 'Artery'),
        os.path.join(analysis_directory, 'ITC Data', 'Vein'),
        os.path.join(analysis_directory, 'ITC Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'ROI Data'),
        os.path.join(analysis_directory, 'ROI Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'Frame Data'),
        os.path.join(analysis_directory, 'Frame Data', 'Vein', 'Sinus Sagittalis'),
        os.path.join(analysis_directory, 'Fitting'),
        image_directory,
        os.path.join(image_directory, 'Intensity Time Curves'),
        os.path.join(image_directory, 'Fit'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Artery'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Vein'),
        os.path.join(image_directory, 'Intensity Time Curves', 'Vein', 'Sinus Sagittalis'),
        os.path.join(image_directory, 'Concentration Time Curves'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Artery'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Vein'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Vein', 'Sinus Sagittalis'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'White Matter'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'Grey Matter'),
        os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', 'Boundary'),
        os.path.join(image_directory, 'Time Shifted Concentration Curves'),
        os.path.join(image_directory, 'Time Shifted Concentration Curves', 'Max'),
        nifti_directory
    ]
    
    artery_names = ["Left Interior Carotid", "Right Interior Carotid", "Basilar", "Left Middle Cerebral", "Right Middle Cerebral"]
    
    # Append directories involving artery names
    for artery in artery_names:
        dirs_to_create.extend([
            os.path.join(analysis_directory, 'CTC Data', 'Artery', artery),
            os.path.join(analysis_directory, 'ITC Data', 'Artery', artery),
            os.path.join(analysis_directory, 'TSCC Data', artery),
            os.path.join(analysis_directory, 'ROI Data', 'Artery', artery),
            os.path.join(analysis_directory, 'Frame Data', 'Artery', artery),
            os.path.join(image_directory, 'Concentration Time Curves', 'Artery', artery),
            os.path.join(image_directory, 'Intensity Time Curves', 'Artery', artery),
            os.path.join(image_directory, 'Time Shifted Concentration Curves', artery)
        ])
    
    # Create all directories
    for dir_path in dirs_to_create:
        create_directory(dir_path)

    return data_directory, analysis_directory, nifti_directory, image_directory    