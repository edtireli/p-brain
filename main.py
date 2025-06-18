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


def mode_screen():
    os.system('clear')
    print('=-=-= Select analysis mode =-=-=')
    print('| 1 | Manual mode')
    print('| 2 | Automatic mode')
    print('| 3 | Pseudo-Automatic mode')
    print('| 9 | Exit program')
    print('=-=-=---------------------=-=-=')


def mode_choice():
    choice = input('[!] Enter mode (1-3 or 9): ')
    if not choice.isdigit():
        print('[!] Only integer input!')
        time.sleep(2)
        return mode_choice()
    return int(choice)


def parse_args():
    parser = argparse.ArgumentParser(description = "Run the neuroimagining analysis tool")
    parser.add_argument('--id', type=str, help = 'Patient ID, corresponding to folder names in data/', required = False)
    parser.add_argument('--option', type=int, help = 'Analysis option (welcome screen)', required = False)
    # Optional mode argument to skip the interactive mode selection
    parser.add_argument('--mode', type=str, choices=['manual', 'auto', 'pseudo'],
                        help='Start directly in a specific analysis mode', required=False)
    return parser.parse_args()


def manual_cli_loop(option, data_directory, analysis_directory, nifti_directory,
                    image_directory, filenames, parameters):
    """Run the classic CLI interface."""
    while True:
        if option is not None:
            choice = option
        else:
            welcome_screen()
            choice = welcome_screen_choice()

        if choice == 0:
            viewer = MRIViewer(nifti_directory, filenames)
            viewer.display()

        elif choice == 1:
            T1_fit(data_directory, analysis_directory, nifti_directory,
                   image_directory, filenames, parameters)
            if option is not None:
                break

        elif choice == 2:
            input_function(analysis_directory, nifti_directory, image_directory,
                           filenames, parameters)
            if option is not None:
                break

        elif choice == 3:
            time_shifting(analysis_directory, nifti_directory, image_directory)
            if option is not None:
                break

        elif choice == 4:
            tissue_function(analysis_directory, nifti_directory, image_directory,
                            filenames)
            if option is not None:
                break

        elif choice == 5:
            BBB_parameters(analysis_directory, image_directory)
            if option is not None:
                break

        elif choice == 7:
            add_notes(analysis_directory)
            if option is not None:
                break

        elif choice == 8:
            selected_addon = list_addons()
            load_addon(selected_addon, analysis_directory, nifti_directory,
                       image_directory, filenames, parameters)
            time.sleep(3)
            if option is not None:
                break

        elif choice == 9:
            break

        elif choice == 6:
            T1_fit(data_directory, analysis_directory, nifti_directory,
                   image_directory, filenames, parameters)
            input_function_AI(analysis_directory, nifti_directory, image_directory,
                               filenames, parameters)
            tissue_function_AI(analysis_directory, nifti_directory, image_directory,
                                filenames, parameters)
            break

        elif choice == 66:
            print('[!] Executing order 66...')
            tissue_function_AI(analysis_directory, nifti_directory, image_directory,
                                filenames, parameters)
            break


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

    mode = args.mode
    if mode is None and args.option is None:
        mode_screen()
        mode_choice_val = mode_choice()
        if mode_choice_val == 1:
            mode = 'manual'
        elif mode_choice_val == 2:
            mode = 'auto'
        elif mode_choice_val == 3:
            mode = 'pseudo'
        else:
            return

    if mode == 'manual' or args.option is not None:
        manual_cli_loop(args.option, data_directory, analysis_directory, nifti_directory,
                        image_directory, filenames, parameters)
    elif mode == 'auto':
        T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
        input_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
        tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
    elif mode == 'pseudo':
        T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
        input_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
        tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
        manual_cli_loop(None, data_directory, analysis_directory, nifti_directory,
                        image_directory, filenames, parameters)
if __name__ == '__main__':
    main()
