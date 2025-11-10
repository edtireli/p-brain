import argparse
import os
import time
from utils import *
import utils.settings as settings
from modules import *
import utils.plotting as plotting
from utils.cli_logging import (
    auto_logging_suppressed,
    install_auto_logging_hooks,
    log_auto,
    log_process_end,
    log_process_start,
    uninstall_auto_logging_hooks,
)
import modules.opt01_T1_fit as opt01_T1_fit
import modules.AI_input_functions as AI_input_functions
import modules.AI_tissue_functions as AI_tissue_functions
import modules.opt03_time_shifting as opt03_time_shifting
import modules.opt02_input_functions as opt02_input_functions
import modules.opt04_tissue_function as opt04_tissue_function
import modules.opt05_BBB_parameters as opt05_BBB_parameters
import modules.opt00_images as opt00_images
import modules.opt08_fa as opt08_fa


def mode_screen():
    print_banner()
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


def _str2bool(value):
    if isinstance(value, bool):
        return value
    val = value.lower()
    if val in {"true", "t", "1", "yes", "y"}:
        return True
    if val in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the neuroimagining analysis tool")
    parser.add_argument('--id', type=str, help='Patient ID, corresponding to folder names in data/', required=False)
    parser.add_argument('--option', type=int, help='Analysis option (welcome screen)', required=False)
    # Optional mode argument to skip the interactive mode selection
    parser.add_argument('--mode', type=str, choices=['manual', 'auto', 'pseudo'],
                        help='Start directly in a specific analysis mode', required=False)
    parser.add_argument('--lambda', dest='tikhonov_lambda', type=float,
                        help='Tikhonov regularisation weight for the two-compartment model',
                        required=False)
    parser.add_argument('--enable-lcurve', action='store_true',
                        help='Automatically pick Tikhonov \u03bb via L-curve')
    parser.add_argument('--data-dir', dest='data_dir', type=str,
                        default=os.environ.get('P_BRAIN_DATA_DIR'),
                        help='Directory containing subject folders (default: ./Data)')
    parser.add_argument('--init-ktrans-from-patlak', action='store_true',
                        help='Use the Patlak Ki as the initial Ktrans guess for the two-compartment fit')
    parser.add_argument('--write-mtt', type=_str2bool, nargs='?', const=True, default=None,
                        help='Enable or disable writing the voxelwise MTT map (default: True in auto mode)')
    parser.add_argument('--write-cth', type=_str2bool, nargs='?', const=True, default=None,
                        help='Enable or disable writing the voxelwise CTH map (default: True in auto mode)')
    parser.add_argument('--diffusion', action='store_true',
                        help='Run diffusion tensor processing after the automatic pipeline')
    return parser.parse_args()


def manual_cli_loop(option, data_directory, analysis_directory, nifti_directory,
                    image_directory, filenames, parameters, pseudo: bool = False):
    """Run the classic CLI interface.

    When ``pseudo`` is True the menu text changes to reflect that the automatic
    pipeline has already been executed.
    """
    while True:
        if option is not None:
            choice = option
        else:
            if pseudo:
                welcome_screen_pseudo()
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

        elif choice == 6:
            add_notes(analysis_directory)
            if option is not None:
                break

        elif choice == 7:
            selected_addon = list_addons()
            load_addon(selected_addon, analysis_directory, nifti_directory,
                       image_directory, filenames, parameters)
            time.sleep(3)
            if option is not None:
                break

        elif choice == 9:
            break

        elif choice == 66:
            print('[!] Executing order 66...')
            tissue_function_AI(analysis_directory, nifti_directory, image_directory,
                                filenames, parameters)
            break


def main():
    args = parse_args()

    if args.tikhonov_lambda is not None:
        settings.TIKHONOV_LAMBDA = args.tikhonov_lambda
    if args.enable_lcurve:
        settings.AUTO_LAMBDA = True
    if args.init_ktrans_from_patlak:
        settings.TWO_COMPARTMENT_INIT_FROM_PATLAK = True
    if args.write_mtt is not None:
        settings.WRITE_MTT = args.write_mtt
    if args.write_cth is not None:
        settings.WRITE_CTH = args.write_cth

    # Respect ``PBRAIN_TURBO`` to disable plotting when running in batch mode.
    # Individual modes may override this later (manual/pseudo always show plots).
    def set_turbo_mode(enabled: bool):
        modules = [plotting, opt01_T1_fit, AI_input_functions, AI_tissue_functions,
                   opt03_time_shifting, opt02_input_functions,
                   opt04_tissue_function, opt05_BBB_parameters, opt00_images]
        for m in modules:
            m.turbo_mode = enabled

    turbo_env = os.environ.get("PBRAIN_TURBO") == "1"
    set_turbo_mode(turbo_env)

    data_root = args.data_dir
    if args.id:
        log_number = args.id
    else:
        log_number = select_log_number(data_root)

    data_directory, analysis_directory, nifti_directory, image_directory = setup_directories(log_number, data_root)
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

    hooks_installed = False

    try:
        if mode == 'manual' or args.option is not None:
            set_turbo_mode(False)
            if args.diffusion:
                print("[diffusion] The --diffusion flag is only available in automatic and pseudo-automatic modes.")
            manual_cli_loop(args.option, data_directory, analysis_directory, nifti_directory,
                            image_directory, filenames, parameters)
        elif mode == 'auto':
            install_auto_logging_hooks()
            hooks_installed = True
            with auto_logging_suppressed():
                print_banner()
            log_auto("Fully automatic analysis pipeline initialised.", level="info")
            log_process_start("T1 fitting")
            T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            log_process_end("T1 fitting")
            log_process_start("AI input function extraction")
            input_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            log_process_end("AI input function extraction")
            log_process_start("Tissue kinetic modelling")
            tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            log_process_end("Tissue kinetic modelling")
            log_process_start("Segmented M0/T1 rendering")
            opt01_T1_fit.generate_segmented_m0_t1_maps(
                analysis_directory,
                image_directory,
                nifti_directory,
            )
            log_process_end("Segmented M0/T1 rendering")
            if args.diffusion:
                log_process_start("Diffusion tensor processing")
                diffusion_filename = filenames[-2] if filenames else None
                opt08_fa.compute_fa(
                    nifti_directory,
                    analysis_directory,
                    image_directory,
                    diffusion_filename=diffusion_filename,
                )
                log_process_end("Diffusion tensor processing")
            log_auto("Fully automatic analysis pipeline completed.", level="success")
        elif mode == 'pseudo':
            set_turbo_mode(False)
            print_banner()
            T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            input_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            tissue_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            if args.diffusion:
                print("[diffusion] Computing diffusion metrics (pseudo-automatic mode)")
                diffusion_filename = filenames[-2] if filenames else None
                opt08_fa.compute_fa(
                    nifti_directory,
                    analysis_directory,
                    image_directory,
                    diffusion_filename=diffusion_filename,
                )
            manual_cli_loop(None, data_directory, analysis_directory, nifti_directory,
                            image_directory, filenames, parameters, pseudo=True)

        save_run_settings(analysis_directory, parameters)
        settings.save_runtime_metadata(analysis_directory)
    finally:
        if hooks_installed:
            uninstall_auto_logging_hooks()
if __name__ == '__main__':
    main()
