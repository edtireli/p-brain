import argparse
import os
from utils import *
from modules import *
import utils.plotting as plotting
import modules.opt01_T1_fit as opt01_T1_fit
import modules.AI_input_functions as AI_input_functions
import modules.AI_tissue_functions as AI_tissue_functions
import modules.opt03_time_shifting as opt03_time_shifting
import modules.opt02_input_functions as opt02_input_functions
import modules.opt04_tissue_function as opt04_tissue_function
import modules.opt05_BBB_parameters as opt05_BBB_parameters
import modules.opt00_images as opt00_images


def parse_args():
    parser = argparse.ArgumentParser(description = "Run the neuroimagining analysis tool")
    parser.add_argument('--id', type=str, help = 'Patient ID, corresponding to folder names in data/', required = False)
    parser.add_argument('--option', type=int, help = 'Analysis option (welcome screen)', required = False)
    return parser.parse_args()


def main():
    args = parse_args()

    # Show figures when running interactively unless explicitly disabled via the
    # ``PBRAIN_TURBO`` environment variable. The enumerator sets this variable to
    # keep plotting off during batch processing.
    if os.environ.get("PBRAIN_TURBO") != "1":
        plotting.turbo_mode = False
        opt01_T1_fit.turbo_mode = False
        AI_input_functions.turbo_mode = False
        AI_tissue_functions.turbo_mode = False
        opt03_time_shifting.turbo_mode = False
        opt02_input_functions.turbo_mode = False
        opt04_tissue_function.turbo_mode = False
        opt05_BBB_parameters.turbo_mode = False
        opt00_images.turbo_mode = False

    if args.id:
        log_number = args.id
    else:
        log_number = select_log_number()
    
    data_directory, analysis_directory, nifti_directory, image_directory = setup_directories(log_number)
    if CONTROLS:
        filenames = control_filenames(nifti_directory)
    else:
        filenames = global_filenames(nifti_directory)
    parameters = global_parameters()
    parrec2nifti(data_directory, nifti_directory)
    if CONTROLS:
        filenames = control_filenames(nifti_directory)
    else:
        filenames = global_filenames(nifti_directory)
    parameters = global_parameters()
    refresh_nifti_directory(nifti_directory)
    check_axial(nifti_directory, filenames)

    while True:
        if args.option:
            choice = args.option
        else:
            welcome_screen()
            choice = welcome_screen_choice()

        #Choices
        if choice == 0: # Show MRI images: DCE, Saggital T1/T2, Axial T1/T2
            viewer = MRIViewer(nifti_directory, filenames)
            viewer.display()

        elif choice == 1: # T1/M0 fit
            T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            if args.option:
                break

        elif choice == 2: # Input function from ROI 
            input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            if args.option:
                break

        elif choice == 3: # Time shifting of input functions AND find maximum AIF
            time_shifting(analysis_directory, nifti_directory, image_directory)
            if args.option:
                break

        elif choice == 4: # Tissue concentration functions
            tissue_function(analysis_directory, nifti_directory, image_directory, filenames)
            if args.option:
                break

        elif choice == 5: # Compute BBB parameters
            BBB_parameters(analysis_directory, image_directory)
            if args.option:
                break

        elif choice == 7: # Analysis notes
            add_notes(analysis_directory) 
            if args.option:
                break

        elif choice == 8: # Addons
            selected_addon = list_addons()
            load_addon(selected_addon, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            time.sleep(3)
            if args.option:
                break
            
        elif choice == 9:
            break

        elif choice == 6:
            #print('[!] Executing order 66...')
            T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            input_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            break

        elif choice == 66:
            print('[!] Executing order 66...')
            #T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            #input_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            break
if __name__ == '__main__':
    main()