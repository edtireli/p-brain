import argparse
import os
import sys

# When running in batch/headless mode, ensure matplotlib uses a non-GUI backend.
# This must be set before any modules import matplotlib.pyplot.
if os.environ.get("PBRAIN_TURBO") == "1":
    os.environ.setdefault("MPLBACKEND", "Agg")
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
from models import normalise_pk_model
import modules.opt01_T1_fit as opt01_T1_fit
import modules.AI_tissue_functions as AI_tissue_functions
import modules.opt03_time_shifting as opt03_time_shifting
import modules.opt02_input_functions as opt02_input_functions
import modules.opt04_tissue_function as opt04_tissue_function
import modules.opt05_BBB_parameters as opt05_BBB_parameters
import modules.opt00_images as opt00_images
from modules.input_function_dispatch import (
    PBRAIN_WAITING_FOR_ROI_EXIT_CODE,
    InputFunctionUserInteractionRequired,
    run_input_function,
)
from utils.loading import discover_ir_series, discover_vfa_series
from utils import qc as qc
from utils.compare_matlab import compare_t1m0_to_matlab


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
                        help='Automatically pick Tikhonov lambda via L-curve')
    parser.add_argument('--model-only', '--pk-only', dest='model_only', action='store_true',
                        help='Run only the pharmacokinetic modelling stage (skip T1/input/time-shift).')
    parser.add_argument(
        '--pk-model',
        dest='pk_model',
        type=str,
        choices=[
            'patlak',
            'tikhonov',
            'both',
        ],
        help='Explicit PK model: patlak | tikhonov | both',
    )
    parser.add_argument('--tik-fast', '--tikhonov-fast', dest='tik_fast', action='store_true',
                        help='Run only the fast Tikhonov flow (skip Patlak).')
    parser.add_argument('--patlak', '--patlak-only', dest='patlak_only', action='store_true',
                        help='Run only Patlak (skip Tikhonov/deconvolution).')
    parser.add_argument('--patlak-then-tik-fast', dest='patlak_then_tik_fast', action='store_true',
                        help='Run Patlak then fast Tikhonov (combo).')
    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        type=str,
        default=os.environ.get('P_BRAIN_DATA_DIR', './Data'),
        help='Directory containing subject folders (default: ./Data)'
    )
    parser.add_argument(
        '--defaults-json',
        dest='defaults_json',
        type=str,
        default=None,
        help='Path to a JSON file containing p-brain-web Defaults to apply as environment variables.',
    )
    parser.add_argument('--init-ktrans-from-patlak', action='store_true',
                        help='Use the Patlak Ki as the initial Ktrans guess for the two-compartment fit')
    parser.add_argument('--write-mtt', type=_str2bool, nargs='?', const=True, default=None,
                        help='Enable or disable writing the voxelwise MTT map (default: True in auto mode)')
    parser.add_argument('--write-cth', type=_str2bool, nargs='?', const=True, default=None,
                        help='Enable or disable writing the voxelwise CTH map (default: True in auto mode)')
    parser.add_argument('--diffusion', action='store_true',
                        help='Run diffusion tensor processing after the automatic pipeline')
    parser.add_argument(
        '--ctc',
        dest='ctc_only',
        action='store_true',
        help='Run only T1 fitting + input-function CTC extraction and exit (skips TSCC/tissue modelling).',
    )
    parser.add_argument('--flip-angle', dest='flip_angle', type=str, default=None,
                        help='Flip angle in degrees (number) or "auto" (default: from metadata)')
    parser.add_argument('--t1-fit', dest='t1_fit', type=str,
                        choices=['auto', 'ir', 'vfa', 'none'], default=None,
                        help='T1/M0 fitting source: auto|ir|vfa|none (default: auto)')
    parser.add_argument('--vfa-glob', dest='vfa_glob', type=str, default=None,
                        help='Comma-separated glob(s) for VFA NIfTI discovery in NIfTI dir (default: *VFA*.nii*)')
    parser.add_argument(
        '--ctc-model',
        dest='ctc_model',
        type=str,
        choices=['turboflash'],
        default=None,
        help='Signal-to-concentration model (validator parity): turboflash.'
    )
    parser.add_argument(
        '--turbo-nph',
        dest='turbo_nph',
        type=int,
        default=None,
        help='TurboFLASH nph (1-based ky=0 line index within readout train). Used when --ctc-model=turboflash.'
    )
    parser.add_argument(
        '--roi-method',
        dest='roi_method',
        type=str,
        choices=['ai', 'deterministic', 'geometry', 'file'],
        default=None,
        help='ROI extraction method for input-function ROIs (ai|deterministic|file).'
    )
    parser.add_argument(
        '--roi-aif-mat',
        dest='roi_aif_mat',
        type=str,
        default=None,
        help='Path to .mat file containing arterial ROI mask (BW_input/BW_input_big).'
    )
    parser.add_argument(
        '--roi-aif-conc-mat',
        dest='roi_aif_conc_mat',
        type=str,
        default=None,
        help='Optional path to .mat file containing arterial concentration curve (c_input). If provided, p-brain will write CTC directly.'
    )
    parser.add_argument(
        '--roi-sss-mat',
        dest='roi_sss_mat',
        type=str,
        default=None,
        help='Path to .mat file containing SSS ROI mask (BW_input/BW_input_big).'
    )
    parser.add_argument(
        '--roi-sss-conc-mat',
        dest='roi_sss_conc_mat',
        type=str,
        default=None,
        help='Optional path to .mat file containing SSS concentration curve (c_input). If provided, p-brain will write CTC directly.'
    )
    parser.add_argument(
        '--roi-tscc-mat',
        dest='roi_tscc_mat',
        type=str,
        default=None,
        help='Optional path to .mat file containing the TSCC input function (c_input) used for modelling. If provided, TSCC generation will import this curve directly.'
    )

    parser.add_argument(
        '--compare-reference',
        dest='compare_reference',
        action='store_true',
        help='Enable best-effort QA comparisons (writes montage PNGs when matching reference .mat files are found).'
    )

    parser.add_argument(
        '--compare-matlab',
        dest='compare_matlab',
        action='store_true',
        help='Compare T1/M0 outputs against a reference MATLAB .mat (e.g. T1_M0_plusError_maps_.mat) when present.'
    )
    parser.add_argument(
        '--compare-matlab-path',
        dest='compare_matlab_path',
        type=str,
        default=None,
        help='Optional explicit path to the MATLAB T1/M0 reference .mat. If omitted, p-brain searches the subject folder.'
    )

    parser.add_argument(
        '--t1m0-only',
        dest='t1m0_only',
        action='store_true',
        help='Run only T1/M0 fitting (option 1) and exit (skips the rest of the pipeline).'
    )
    parser.add_argument(
        '--t1m0-force',
        dest='t1m0_force',
        action='store_true',
        help='When used with --t1m0-only, delete cached T1/M0 outputs in Analysis/Fitting before running.'
    )
    return parser.parse_args()


def _maybe_force_delete_t1m0_outputs(analysis_directory: str) -> None:
    fitting_dir = os.path.join(analysis_directory, 'Fitting')
    if not os.path.isdir(fitting_dir):
        return
    # Remove only the T1/M0 fit cache outputs.
    names = [
        'voxel_matrix.pkl',
        'voxel_T1_matrix.pkl',
        'voxel_M0_matrix.pkl',
        't1_map.nii.gz',
        'm0_map.nii.gz',
        't1_map_in_dce.nii.gz',
        'm0_map_in_dce.nii.gz',
    ]
    for name in names:
        for candidate in (name, f'._{name}'):
            path = os.path.join(fitting_dir, candidate)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


def _set_pk_model(raw: str) -> None:
    """Normalise PK model selection and push into env + settings."""

    val = str(raw).strip().lower()

    # Canonical validated keys: patlak | tikhonov | both
    if val in {
        'both',
        'all',
        'patlak_then_tikhonov',
        'patlak-then-tikhonov',
        'patlak_tikhonov',
        'patlak_tikhonov_fast',
        'patlak-then-tikhonov-fast',
        'patlak_then_tikhonov_fast',
    }:
        canonical = 'both'
    elif val in {
        'tikhonov',
        'tikhonov_fast',
        'tikhonov_only',
        'tikhonov-only-fast',
        'tik_fast',
        'tik-fast',
        'tikfast',
        'two_compartment',
        '2comp',
        'two-comp',
        'two-compartment',
    }:
        canonical = 'tikhonov'
    elif val == 'patlak':
        canonical = 'patlak'
    else:
        canonical = 'both'

    os.environ['P_BRAIN_MODEL'] = canonical
    settings.KINETIC_MODEL = canonical


def _apply_pk_model_flags(args) -> None:
    model_raw = None
    # Preserve flag order: scan argv for patlak/tik-fast flags in sequence.
    seq = []
    for tok in sys.argv:
        t = tok.strip().lower()
        if t in {'--patlak', '--patlak-only'}:
            seq.append('patlak')
        elif t in {'--tik-fast', '--tikhonov-fast'}:
            seq.append('tikhonov')

    if getattr(args, 'pk_model', None):
        model_raw = args.pk_model
    elif getattr(args, 'patlak_then_tik_fast', False) or seq == ['patlak', 'tikhonov']:
        model_raw = 'both'
    elif seq and all(step == 'patlak' for step in seq):
        model_raw = 'patlak'
    elif seq and all(step == 'tikhonov' for step in seq):
        model_raw = 'tikhonov'
    elif getattr(args, 'tik_fast', False):
        model_raw = 'tikhonov'
    elif getattr(args, 'patlak_only', False):
        model_raw = 'patlak'

    if model_raw:
        _set_pk_model(normalise_pk_model(model_raw))


def manual_cli_loop(option, data_directory, analysis_directory, nifti_directory,
                    image_directory, filenames, parameters, pseudo: bool = False):
    """Run the classic CLI interface.

    When ``pseudo`` is True the prompt text changes to reflect that the automatic
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

    # Apply p-brain-web Defaults early so subsequent CLI flags can override them.
    if getattr(args, 'defaults_json', None):
        try:
            from utils.defaults import apply_defaults_json

            apply_defaults_json(str(args.defaults_json), args=args)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to apply --defaults-json={args.defaults_json}: {exc}")

    # QA comparisons are opt-in (can be enabled via CLI flag or env var).
    # Downstream modules should gate all comparison PNG output behind this.
    if getattr(args, 'compare_reference', False):
        os.environ['P_BRAIN_COMPARE_REFERENCE'] = '1'

    # MATLAB T1/M0 comparison (legacy flag restored).
    # opt01_T1_fit implements the compare routine and is gated behind env vars.
    if getattr(args, 'compare_matlab', False):
        os.environ['P_BRAIN_COMPARE_MATLAB'] = '1'
        # Keep the older env name for backward compatibility.
        os.environ['P_BRAIN_T1M0_COMPARE'] = '1'
        if getattr(args, 'compare_matlab_path', None):
            os.environ['P_BRAIN_T1M0_COMPARE_MAT_PATH'] = str(args.compare_matlab_path)

    # Keep behaviour consistent with enumerator.py/roi_only.py: treat --data-dir as
    # the root of the dataset folders and export it for downstream modules.
    if getattr(args, 'data_dir', None):
        os.environ['P_BRAIN_DATA_DIR'] = os.path.abspath(str(args.data_dir))

    # PK model selection (Patlak-only, Tikhonov-fast-only, or combined).
    _apply_pk_model_flags(args)

    if args.flip_angle is not None:
        raw = str(args.flip_angle).strip()
        if raw.lower() == "auto" or raw == "":
            settings.FLIP_ANGLE_SETTING = "auto"
            settings.FLIP_ANGLE_DEG = None
        else:
            try:
                settings.FLIP_ANGLE_SETTING = raw
                settings.FLIP_ANGLE_DEG = float(raw)
            except ValueError:
                # Fall back to metadata resolution.
                settings.FLIP_ANGLE_SETTING = "auto"
                settings.FLIP_ANGLE_DEG = None

    if getattr(args, 'roi_method', None) is not None:
        raw = str(args.roi_method).strip().lower()
        if raw == 'geometry':
            raw = 'deterministic'
        settings.ROI_METHOD = raw
        os.environ['P_BRAIN_ROI_METHOD'] = raw

    if getattr(args, 'roi_aif_mat', None):
        settings.ROI_AIF_MAT = str(args.roi_aif_mat)
        os.environ['P_BRAIN_ROI_AIF_MAT'] = str(args.roi_aif_mat)
    if getattr(args, 'roi_aif_conc_mat', None):
        settings.ROI_AIF_CONC_MAT = str(args.roi_aif_conc_mat)
        os.environ['P_BRAIN_ROI_AIF_CONC_MAT'] = str(args.roi_aif_conc_mat)
    if getattr(args, 'roi_sss_mat', None):
        settings.ROI_SSS_MAT = str(args.roi_sss_mat)
        os.environ['P_BRAIN_ROI_SSS_MAT'] = str(args.roi_sss_mat)
    if getattr(args, 'roi_sss_conc_mat', None):
        settings.ROI_SSS_CONC_MAT = str(args.roi_sss_conc_mat)
        os.environ['P_BRAIN_ROI_SSS_CONC_MAT'] = str(args.roi_sss_conc_mat)
    if getattr(args, 'roi_tscc_mat', None):
        settings.ROI_TSCC_MAT = str(args.roi_tscc_mat)
        os.environ['P_BRAIN_ROI_TSCC_MAT'] = str(args.roi_tscc_mat)

    if args.t1_fit is not None:
        settings.T1_FIT_MODE = str(args.t1_fit).strip().lower() or "auto"
    if args.vfa_glob is not None:
        settings.VFA_FILE_GLOB = str(args.vfa_glob).strip() or settings.VFA_FILE_GLOB

    if getattr(args, 'ctc_model', None) is not None:
        settings.CTC_MODEL = str(args.ctc_model).strip().lower() or settings.CTC_MODEL
        os.environ['P_BRAIN_CTC_MODEL'] = settings.CTC_MODEL
    if getattr(args, 'turbo_nph', None) is not None:
        settings.TURBOFLASH_NPH = int(args.turbo_nph)
        os.environ['P_BRAIN_TURBOFLASH_NPH'] = str(int(settings.TURBOFLASH_NPH))

    # NOTE: Do NOT auto-switch the T1/M0 recovery model based on the
    # signal->concentration model.
    #
    # p-brain-web exposes T1 recovery model as an explicit user-facing Default.
    # The validated inversion-recovery fit must remain the default unless the
    # caller explicitly sets `P_BRAIN_T1_RECOVERY_MODEL` (or passes a Defaults
    # config that sets it).

    # Respect a pinned lambda by disabling auto-selection; otherwise keep/enable L-curve.
    if args.tikhonov_lambda is not None:
        settings.TIKHONOV_LAMBDA = args.tikhonov_lambda
        settings.AUTO_LAMBDA = False
    elif args.enable_lcurve:
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
        modules = [plotting, opt01_T1_fit, AI_tissue_functions,
                   opt03_time_shifting, opt02_input_functions,
                   opt04_tissue_function, opt05_BBB_parameters, opt00_images]

        # Only import the TensorFlow-backed AI input-function module when needed.
        if (getattr(settings, "ROI_METHOD", "ai") or "ai").strip().lower() == "ai":
            import modules.AI_input_functions as AI_input_functions

            modules.append(AI_input_functions)
        for m in modules:
            m.turbo_mode = enabled

    turbo_env = os.environ.get("PBRAIN_TURBO") == "1"
    set_turbo_mode(turbo_env)

    data_root = os.environ.get('P_BRAIN_DATA_DIR', './Data')
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

    # Fast path: run only T1/M0 fitting and exit.
    if getattr(args, 't1m0_only', False):
        set_turbo_mode(True)

        # In this mode we want deterministic behaviour based on the inputs that
        # exist on disk *for this dataset*, even if settings.T1_FIT_MODE was
        # previously set (e.g. via env var or a prior run in the same process).
        if getattr(args, 't1_fit', None) is None:
            try:
                has_ir = bool(discover_ir_series(nifti_directory))
            except Exception:
                has_ir = False
            try:
                has_vfa = bool(discover_vfa_series(nifti_directory, patterns=getattr(settings, "VFA_FILE_GLOB", None)))
            except Exception:
                has_vfa = False
            if has_ir:
                settings.T1_FIT_MODE = 'ir'
            elif has_vfa:
                settings.T1_FIT_MODE = 'vfa'
            else:
                settings.T1_FIT_MODE = 'none'
            parameters = global_parameters()

        if getattr(args, 't1m0_force', False):
            _maybe_force_delete_t1m0_outputs(analysis_directory)
        T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)

        # Optional best-effort comparison vs MATLAB reference outputs.
        # Enabled via --compare-matlab (sets env vars above).
        if (os.environ.get('P_BRAIN_COMPARE_MATLAB') or '').strip() in {'1', 'true', 'yes'} or (
            (os.environ.get('P_BRAIN_T1M0_COMPARE') or '').strip() in {'1', 'true', 'yes'}
        ):
            try:
                compare_t1m0_to_matlab(
                    subject_root=data_directory,
                    analysis_directory=analysis_directory,
                    image_directory=image_directory,
                    mat_path=os.environ.get('P_BRAIN_T1M0_COMPARE_MAT_PATH') or None,
                )
            except Exception as e:
                try:
                    log_auto(f"Compare(MATLAB): {e}", level="warning")
                except Exception:
                    pass

        save_run_settings(analysis_directory, parameters)
        settings.save_runtime_metadata(analysis_directory)
        return

    # Fast path: run only CTC/input-function extraction and exit.
    if getattr(args, 'ctc_only', False):
        set_turbo_mode(True)
        _hooks_installed = False
        try:
            install_auto_logging_hooks()
            _hooks_installed = True
            with auto_logging_suppressed():
                print_banner()
            log_auto("CTC-only path initialised.", level="info")

            log_process_start("T1 fitting (CTC-only)")
            T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            log_process_end("T1 fitting (CTC-only)")

            log_process_start("Input function extraction (CTC-only)")
            run_input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            log_process_end("Input function extraction (CTC-only)")

            save_run_settings(analysis_directory, parameters)
            settings.save_runtime_metadata(analysis_directory)
            return
        finally:
            if _hooks_installed:
                uninstall_auto_logging_hooks()

    # Fast path: run only modelling (PK) and exit.
    if getattr(args, 'model_only', False):
        set_turbo_mode(True)
        _hooks_installed = False
        try:
            install_auto_logging_hooks()
            _hooks_installed = True
            with auto_logging_suppressed():
                print_banner()
            log_auto("PK-only path initialised.", level="info")
            log_process_start("Tissue kinetic modelling (PK-only)")
            tissue_function_AI(
                analysis_directory,
                nifti_directory,
                image_directory,
                filenames,
                parameters,
                compute_diffusion=getattr(args, "diffusion", False),
            )
            log_process_end("Tissue kinetic modelling (PK-only)")
            save_run_settings(analysis_directory, parameters)
            settings.save_runtime_metadata(analysis_directory)
            return
        finally:
            if _hooks_installed:
                uninstall_auto_logging_hooks()

    # Auto-select the T1 fitting mode based on which inputs exist.
    if getattr(settings, "T1_FIT_MODE", "auto") == "auto":
        has_ir = bool(discover_ir_series(nifti_directory))
        has_vfa = bool(discover_vfa_series(nifti_directory, patterns=getattr(settings, "VFA_FILE_GLOB", None)))
        if has_ir:
            settings.T1_FIT_MODE = "ir"
        elif has_vfa:
            settings.T1_FIT_MODE = "vfa"
        else:
            settings.T1_FIT_MODE = "none"
        parameters = global_parameters()

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

            if (os.environ.get('P_BRAIN_COMPARE_MATLAB') or '').strip() in {'1', 'true', 'yes'} or (
                (os.environ.get('P_BRAIN_T1M0_COMPARE') or '').strip() in {'1', 'true', 'yes'}
            ):
                try:
                    compare_t1m0_to_matlab(
                        subject_root=data_directory,
                        analysis_directory=analysis_directory,
                        image_directory=image_directory,
                        mat_path=os.environ.get('P_BRAIN_T1M0_COMPARE_MAT_PATH') or None,
                    )
                    log_auto("Compare(MATLAB): wrote metrics/montage", level="info")
                except Exception as e:
                    log_auto(f"Compare(MATLAB) error: {e}", level="warning")

            try:
                rep, paths = qc.run_and_persist(
                    stage="t1_fit",
                    analysis_directory=analysis_directory,
                    nifti_directory=nifti_directory,
                    image_directory=image_directory,
                    settings_snapshot={
                        "ROI_METHOD": getattr(settings, "ROI_METHOD", None),
                        "INPUT_FUNCTION_USE_SSS": getattr(settings, "INPUT_FUNCTION_USE_SSS", None),
                    },
                )
                log_auto(f"QC(t1_fit): {rep.overallStatus} ({paths.get('latestStage')})", level=("success" if rep.overallStatus == "pass" else ("warning" if rep.overallStatus == "warn" else "error")))
                if rep.overallStatus == "fail" and (os.environ.get("P_BRAIN_QC_ENFORCE") or "").strip() in {"1", "true", "yes"}:
                    raise RuntimeError(f"QC failed for t1_fit: {paths.get('latestStage')}")
            except Exception as e:
                log_auto(f"QC(t1_fit) error: {e}", level="warning")

            log_process_start("Input function extraction")
            orig_roi_method = getattr(settings, "ROI_METHOD", None)
            tried_fallback = False
            try:
                run_input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            except InputFunctionUserInteractionRequired as e:
                # User-driven ROI fallback requested; stop the auto pipeline without marking failure.
                log_auto(f"WAITING(input_functions): {e}", level="warning")
                try:
                    log_process_end("Input function extraction")
                except Exception:
                    pass
                raise SystemExit(int(PBRAIN_WAITING_FOR_ROI_EXIT_CODE))
            except Exception as e:
                try:
                    is_deterministic = str(orig_roi_method).strip().lower() in {"deterministic", "geometry"}
                except Exception:
                    is_deterministic = False
                try:
                    is_file = str(orig_roi_method).strip().lower() == "file"
                except Exception:
                    is_file = False
                if (not is_deterministic) and (not is_file):
                    tried_fallback = True
                    log_auto(
                        f"Mitigation(input_functions): retrying with ROI_METHOD=deterministic after error: {e}",
                        level="warning",
                    )
                    try:
                        settings.ROI_METHOD = "deterministic"
                    except Exception:
                        pass
                    run_input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
                else:
                    raise
            log_process_end("Input function extraction")

            log_process_start("Time-shifted concentration curves (TSCC)")
            try:
                # Always generate TSCC so downstream modelling/web UI can rely on it.
                opt03_time_shifting.time_shifting(analysis_directory, nifti_directory, image_directory)
            finally:
                log_process_end("Time-shifted concentration curves (TSCC)")

            try:
                rep, paths = qc.run_and_persist(
                    stage="input_functions",
                    analysis_directory=analysis_directory,
                    nifti_directory=nifti_directory,
                    image_directory=image_directory,
                    settings_snapshot={
                        "ROI_METHOD": getattr(settings, "ROI_METHOD", None),
                        "INPUT_FUNCTION_USE_SSS": getattr(settings, "INPUT_FUNCTION_USE_SSS", None),
                        "MITIGATION": "retry_roi_method_deterministic" if tried_fallback else None,
                        "ORIG_ROI_METHOD": orig_roi_method,
                    },
                )
                log_auto(f"QC(input_functions): {rep.overallStatus} ({paths.get('latestStage')})", level=("success" if rep.overallStatus == "pass" else ("warning" if rep.overallStatus == "warn" else "error")))
                if rep.overallStatus == "fail" and not tried_fallback:
                    try:
                        is_deterministic = str(orig_roi_method).strip().lower() in {"deterministic", "geometry"}
                    except Exception:
                        is_deterministic = False
                    try:
                        is_file = str(orig_roi_method).strip().lower() == "file"
                    except Exception:
                        is_file = False
                    if (not is_deterministic) and (not is_file):
                        tried_fallback = True
                        log_auto(
                            "Mitigation(input_functions): QC failed; retrying with ROI_METHOD=deterministic",
                            level="warning",
                        )
                        try:
                            settings.ROI_METHOD = "deterministic"
                        except Exception:
                            pass
                        run_input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
                        rep, paths = qc.run_and_persist(
                            stage="input_functions",
                            analysis_directory=analysis_directory,
                            nifti_directory=nifti_directory,
                            image_directory=image_directory,
                            settings_snapshot={
                                "ROI_METHOD": getattr(settings, "ROI_METHOD", None),
                                "INPUT_FUNCTION_USE_SSS": getattr(settings, "INPUT_FUNCTION_USE_SSS", None),
                                "MITIGATION": "retry_roi_method_deterministic_after_qc_fail",
                                "ORIG_ROI_METHOD": orig_roi_method,
                            },
                        )
                        log_auto(f"QC(input_functions) after mitigation: {rep.overallStatus} ({paths.get('latestStage')})", level=("success" if rep.overallStatus == "pass" else ("warning" if rep.overallStatus == "warn" else "error")))
                if rep.overallStatus == "fail" and (os.environ.get("P_BRAIN_QC_ENFORCE") or "").strip() in {"1", "true", "yes"}:
                    raise RuntimeError(f"QC failed for input_functions: {paths.get('latestStage')}")
            except Exception as e:
                log_auto(f"QC(input_functions) error: {e}", level="warning")

            log_process_start("Tissue kinetic modelling")
            tissue_function_AI(
                analysis_directory,
                nifti_directory,
                image_directory,
                filenames,
                parameters,
                compute_diffusion=args.diffusion,
            )
            log_process_end("Tissue kinetic modelling")

            # The tissue_function_AI umbrella covers segmentation + tissue_ctc + modelling.
            for st in ("time_shift", "segmentation", "tissue_ctc", "modelling"):
                try:
                    rep, paths = qc.run_and_persist(
                        stage=st,
                        analysis_directory=analysis_directory,
                        nifti_directory=nifti_directory,
                        image_directory=image_directory,
                        settings_snapshot={
                            "ROI_METHOD": getattr(settings, "ROI_METHOD", None),
                            "INPUT_FUNCTION_USE_SSS": getattr(settings, "INPUT_FUNCTION_USE_SSS", None),
                        },
                    )
                    log_auto(f"QC({st}): {rep.overallStatus} ({paths.get('latestStage')})", level=("success" if rep.overallStatus == "pass" else ("warning" if rep.overallStatus == "warn" else "error")))
                    if rep.overallStatus == "fail" and (os.environ.get("P_BRAIN_QC_ENFORCE") or "").strip() in {"1", "true", "yes"}:
                        raise RuntimeError(f"QC failed for {st}: {paths.get('latestStage')}")
                except Exception as e:
                    log_auto(f"QC({st}) error: {e}", level="warning")

            if args.diffusion:
                try:
                    rep, paths = qc.run_and_persist(
                        stage="diffusion",
                        analysis_directory=analysis_directory,
                        nifti_directory=nifti_directory,
                        image_directory=image_directory,
                        settings_snapshot={
                            "ROI_METHOD": getattr(settings, "ROI_METHOD", None),
                            "INPUT_FUNCTION_USE_SSS": getattr(settings, "INPUT_FUNCTION_USE_SSS", None),
                        },
                    )
                    log_auto(f"QC(diffusion): {rep.overallStatus} ({paths.get('latestStage')})", level=("success" if rep.overallStatus == "pass" else ("warning" if rep.overallStatus == "warn" else "error")))
                    if rep.overallStatus == "fail" and (os.environ.get("P_BRAIN_QC_ENFORCE") or "").strip() in {"1", "true", "yes"}:
                        raise RuntimeError(f"QC failed for diffusion: {paths.get('latestStage')}")
                except Exception as e:
                    log_auto(f"QC(diffusion) error: {e}", level="warning")

            log_process_start("Segmented M0/T1 rendering")
            opt01_T1_fit.generate_segmented_m0_t1_maps(
                analysis_directory,
                image_directory,
                nifti_directory,
            )
            log_process_end("Segmented M0/T1 rendering")
            log_auto("Fully automatic analysis pipeline completed.", level="success")
        elif mode == 'pseudo':
            set_turbo_mode(False)
            print_banner()
            T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            run_input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            opt03_time_shifting.time_shifting(analysis_directory, nifti_directory, image_directory)
            tissue_function_AI(
                analysis_directory,
                nifti_directory,
                image_directory,
                filenames,
                parameters,
                compute_diffusion=args.diffusion,
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
