from utils import *
from modules import *

def main():
    # Handle log_number choice and data directories
    log_number = select_log_number()
    data_directory, analysis_directory, nifti_directory, image_directory = setup_directories(log_number)
    filenames = global_filenames(nifti_directory)
    parameters = global_parameters()
    parrec2nifti(data_directory, nifti_directory)
   
    # Welcome screen
    while True:
        welcome_screen()
        choice = welcome_screen_choice()

        #Choices
        if choice == 0: # Show MRI images: DCE, Saggital T1/T2, Axial T1/T2
            viewer = MRIViewer(nifti_directory, filenames)
            viewer.display()

        elif choice == 1: # T1/M0 fit
            T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)

        elif choice == 2: # Input function from ROI 
            input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
        
        elif choice == 3: # Time shifting of input functions AND find maximum AIF
            time_shifting(analysis_directory, nifti_directory, image_directory)
        
        elif choice == 4: # Tissue concentration functions
            tissue_function(analysis_directory, nifti_directory, image_directory, filenames)

        elif choice == 5: # Compute BBB parameters
            BBB_parameters(analysis_directory, image_directory)

        elif choice == 6: # Analysis notes
            add_notes(analysis_directory) 
        
        elif choice == 7: # Axial FLAIR to axial T1&T2
            check_axial(nifti_directory, filenames)

        elif choice == 8: # Addons
            selected_addon = list_addons()
            load_addon(selected_addon, analysis_directory, nifti_directory, image_directory, filenames, parameters)
    

        elif choice == 9:
            break

if __name__ == '__main__':
    main()
