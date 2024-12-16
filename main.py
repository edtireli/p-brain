import argparse 
from utils import *
from modules import *


def parse_args():
    parser = argparse.ArgumentParser(description = "Run the neuroimagining analysis tool")
    parser.add_argument('--id', type=str, help = 'Patient ID, corresponding to folder names in data/', required = False)
    parser.add_argument('--option', type=int, help = 'Analysis option (welcome screen)', required = False)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.id:
        log_number = args.id
    else:
        log_number = select_log_number()
    
    data_directory, analysis_directory, nifti_directory, image_directory = setup_directories(log_number)
    filenames = global_filenames(nifti_directory)
    parameters = global_parameters()
    parrec2nifti(data_directory, nifti_directory)
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
<<<<<<< HEAD
            if args.option:
                break

        elif choice == 2: # Input function from ROI 
            input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            if args.option:
                break

=======

        elif choice == 2: # Input function from ROI 
            input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
        
>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21
        elif choice == 3: # Time shifting of input functions AND find maximum AIF
            time_shifting(analysis_directory, nifti_directory, image_directory)
            if args.option:
                break

        elif choice == 4: # Tissue concentration functions
            tissue_function(analysis_directory, nifti_directory, image_directory, filenames)
<<<<<<< HEAD
            if args.option:
                break
=======
>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21

        elif choice == 5: # Compute BBB parameters
            BBB_parameters(analysis_directory, image_directory)
            if args.option:
                break

        elif choice == 7: # Analysis notes
            add_notes(analysis_directory) 
<<<<<<< HEAD
            if args.option:
                break
=======
        
        elif choice == 7: # Axial FLAIR to axial T1&T2
            check_axial(nifti_directory, filenames)
>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21

        elif choice == 8: # Addons
            selected_addon = list_addons()
            load_addon(selected_addon, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            time.sleep(3)
<<<<<<< HEAD
            if args.option:
                break
            
=======
    

>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21
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
