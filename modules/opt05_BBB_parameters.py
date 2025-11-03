from utils.plotting import *
from utils.mapping import *
from utils.loading import *
import utils.settings as settings
from .kinetic_models import (
    extended_tofts_tikhonov,
    construct_convolution_matrix,
    tikhonov_regularization,
    extended_tofts_model,
    pick_lambda_via_l_curve,
    residue_to_cbf,
)
import numpy as np
from termcolor import colored

turbo_mode = True  # When True, suppress interactive plotting

def permeability_user_interface(analysis_directory):
    input_source = settings.INPUT_FUNCTION_SOURCE
    subtype_mapping_artery = {
        "lica": "Left Interior Carotid",
        "rica": "Right Interior Carotid",
        "bas": "Basilar",
        "lmca": "Left Middle Cerebral",
        "rmca": "Right Middle Cerebral",
        "max": "Max"
    }
    if input_source == 'RICA':
        subtype_mapping_artery = {
            "rica": "Right Interior Carotid",
            "max": "Max"
        }
    
    subtype_mapping_tissue = {
        "w": "White Matter",
        "g": "Grey Matter",
        "b": "Boundary"
    }
    
    subtype_tissue, subtype_artery, slice_tissue, slice_artery = None, None, None, None
    
    for type_choice in ['artery', 'tissue']:
        subtype_mapping = subtype_mapping_artery if type_choice == 'artery' else subtype_mapping_tissue

        if type_choice == 'artery':
            if input_source == 'RICA':
                base_path = os.path.join(analysis_directory, 'CTC Data', 'Artery')
            else:
                base_path = os.path.join(analysis_directory, 'TSCC Data')
        else:
            base_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue')

        available_subtypes = {}
        if os.path.isdir(base_path):
            for subtype in os.listdir(base_path):
                subtype_path = os.path.join(base_path, subtype)
                if not os.path.isdir(subtype_path):
                    continue
                npy_files = [s for s in os.listdir(subtype_path) if s.endswith('.npy')]
                if not npy_files:
                    continue
                if type_choice == 'artery' and input_source == 'RICA':
                    slices = [
                        s.split('_')[-1].split('.')[0]
                        for s in npy_files
                        if s.lower().startswith('ctc_slice_')
                    ]
                    if slices:
                        available_subtypes[subtype] = sorted(set(slices), key=int)
                else:
                    slices = [s.split('_')[-1].split('.')[0] for s in npy_files]
                    available_subtypes[subtype] = slices

        if type_choice == 'artery' and input_source == 'RICA' and available_subtypes:
            available_subtypes['Max'] = ['auto']

        available_subtypes = {
            k: v for k, v in available_subtypes.items() if k in subtype_mapping.values()
        }
        
        print('=-=-==-=-==-=-==-=-==-=-==-=-=-==-=-==-=--=-==-=-==-=-==-=-==-=-==-=-=')
        print(f'Please identify {type_choice}:')
        for abbr, full_name in subtype_mapping.items():
            if full_name in available_subtypes:
                print(colored(f'{abbr}', 'cyan') +' : ' +colored(f'{full_name}', 'white'))
        print('=-=-==-=-==-=-==-=-==-=-==-=-=-==-=-==-=--=-==-=-==-=-==-=-==-=-==-=-=')
        
        choice_str = input('[!] Enter your choice: ')
        chosen_subtype = subtype_mapping.get(choice_str.lower())
        
        if type_choice == 'artery':
            if input_source == 'RICA':
                if chosen_subtype == "Max":
                    chosen_venous_slice = None
                    chosen_arterial_slice = 'auto'
                    print('[!] Automatically selecting the pure arterial curve with the highest peak.')
                else:
                    arterial_slices = available_subtypes.get(chosen_subtype, [])
                    if len(arterial_slices) == 1:
                        chosen_arterial_slice = arterial_slices[0]
                        print(f'[!] Automatically picking only available arterial slice: {chosen_arterial_slice}')
                    else:
                        print(f'[!] Available arterial slices: {", ".join(arterial_slices)}')
                        chosen_arterial_slice = input('[!] Enter the arterial slice index: ')
                    chosen_venous_slice = None
            else:
                if chosen_subtype == "Max":
                    max_dir = os.path.join(analysis_directory, 'TSCC Data', 'Max')
                    npy_files = [f for f in os.listdir(max_dir) if f.endswith('.npy') and not f.startswith('.')]
                    if not npy_files:
                        raise FileNotFoundError(f"No .npy files found in {max_dir}.")
                    max_file = npy_files[0]
                    chosen_venous_slice, chosen_arterial_slice = max_file.split('_')[2:4]
                    chosen_arterial_slice = chosen_arterial_slice.split('.')[0]
                else:
                    venous_slices = set()
                    arterial_slices = set()
                    for filename in os.listdir(os.path.join(analysis_directory, 'TSCC Data', chosen_subtype)):
                        if filename.startswith("TSCC_slice_"):
                            venous, arterial = filename.split("_")[2:4]
                            venous_slices.add(venous)
                            arterial_slices.add(arterial.split('.')[0])

                    print(f'[!] Available venous slices: {", ".join(venous_slices)}')
                    chosen_venous_slice = input('[!] Enter the venous slice index: ')
                    print(f'[!] Available arterial slices: {", ".join(arterial_slices)}')
                    chosen_arterial_slice = input('[!] Enter the arterial slice index: ')

            subtype_artery, slice_artery = chosen_subtype, (chosen_venous_slice, chosen_arterial_slice)
        else:
            available_slices = available_subtypes.get(chosen_subtype, [])
            if len(available_slices) == 1:
                chosen_slice = available_slices[0]
                print(f'[!] Automatically picking only available slice: slice {chosen_slice}')
            else:
                print(f'[!] Available slices: {", ".join(available_slices)}')
                chosen_slice = input('[!] Enter the slice index: ')
                if chosen_slice == 'seg2roi':
                    chosen_slice = '_seg2roi'
            subtype_tissue, slice_tissue = chosen_subtype, chosen_slice

    return subtype_tissue, subtype_artery, slice_tissue, slice_artery




def patlak_analysis(c_tissue, c_input, time, subtype_tissue, image_directory):
    frame_no = len(time)
    delta_t = np.diff(time)
    y_patlak = np.zeros(frame_no)
    x_patlak = np.zeros(frame_no)
    
    for i in range(frame_no - 1):
        if c_input[i] != 0:
            y_patlak[i] = c_tissue[i] / c_input[i]
            x_patlak[i] = np.sum(c_input[:i+1] * delta_t[:i+1]) / c_input[i]
    
    # Removing zero elements
    non_zero_indices = np.nonzero(y_patlak)
    y_patlak = y_patlak[non_zero_indices]
    x_patlak = x_patlak[non_zero_indices]
    
    calc_max = np.max(x_patlak)
    calc_min = calc_max / 3
    idx = np.where((x_patlak >= calc_min) & (x_patlak <= calc_max))
    x, y = x_patlak[idx], y_patlak[idx]
    
    Ki = np.dot(x - np.mean(x), y - np.mean(y)) / np.dot(x - np.mean(x), x - np.mean(x))
    lambda_ = np.mean(y) - Ki * np.mean(x)
    
    SD_Ki = np.sqrt(np.dot(y - (lambda_ + Ki * x), y - (lambda_ + Ki * x)) / (np.dot(x - np.mean(x), x - np.mean(x)) * (len(x) - 2)))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_patlak, y_patlak, c='black', s=2)
    plt.plot(x_patlak, lambda_ + Ki * x_patlak, c='red', linestyle='--')
    plt.xlabel('$x_{\\mathrm{patlak}}$')
    plt.ylabel('$y_{\\mathrm{patlak}}$')
    plt.ylim(min(y_patlak), max(y))
    plt.title(f'Patlak Plot ({subtype_tissue}) '
          f'$K_i = {round(Ki*6000, 5)} \pm {round(SD_Ki*6000, 5)}$ ml/100g/min, '
          f'$\\lambda = {round(lambda_*100, 5)}$ ml/100g')
    plt.grid(True)
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.savefig(os.path.join(image_directory, 'Fit', f'patlak_{subtype_tissue}'),dpi=200)
    if not turbo_mode:
        close_plot_after_delay_plt(3)
        plt.show()
    return Ki, lambda_, SD_Ki


def extended_tofts_model(t, Ktrans, ve, vp, Cp):
    integrand = np.zeros_like(t)
    for i in range(len(t)):
        min_len = min(len(Cp[:i+1]), len(t[:i+1]))
        integral = np.trapz(Cp[:min_len] * np.exp(-(t[i] - t[:min_len]) * Ktrans / ve), x=t[:min_len])
        integrand[i] = integral
    min_len = min(len(integrand), len(Cp))
    return Ktrans * integrand[:min_len] + vp * Cp[:min_len]

def compute_average_permeability(c_in, c_out, time_array, baseline_point,
                                 lambd=settings.TIKHONOV_LAMBDA):
    if c_in.shape[0] != c_out.shape[0]:
        raise ValueError("The number of time points in c_in and c_out must be the same.")

    Ktrans, _, _ = extended_tofts_tikhonov(c_in, c_out, time_array, lambd=lambd)
    Ktrans_fitted_mM_min = Ktrans * 6000
    std_dev_Ktrans_mM_min = np.nan

    return Ktrans_fitted_mM_min, std_dev_Ktrans_mM_min


def BBB_parameters(analysis_directory, image_directory):  # Ki from ROI
    time_points_s = np.load(os.path.join(analysis_directory, 'Fitting', 'time_points_s.npy'))
    subtype_tissue, subtype_artery, slice_tissue, (venous_slice, arterial_slice) = permeability_user_interface(analysis_directory)

    selected_subtype_artery = subtype_artery
    C_t = np.load(os.path.join(analysis_directory, 'CTC Data', 'Tissue', subtype_tissue, f'CTC_slice_{slice_tissue}.npy'))
    C_a, input_metadata = get_input_function_curve(
        analysis_directory,
        subtype=subtype_artery,
        venous_slice=venous_slice,
        arterial_slice=arterial_slice
    )

    actual_artery = input_metadata.get('artery_subtype', subtype_artery)
    actual_venous = input_metadata.get('venous_slice', venous_slice)
    actual_arterial = input_metadata.get('arterial_slice', arterial_slice)
    venous_label = 'N/A' if actual_venous is None else str(actual_venous)
    arterial_label = 'N/A' if actual_arterial is None else str(actual_arterial)

    C_t = C_t[0:len(C_a)]
    time_points_s = time_points_s[0:len(C_a)]
    
    model = settings.KINETIC_MODEL.lower()
    use_two_compartment = model in ('two_compartment', 'both')
    use_patlak = model in ('patlak', 'both')

    if use_two_compartment:
        if settings.AUTO_LAMBDA and settings.AUTO_LAMBDA_VALUE is not None:
            lambd = settings.AUTO_LAMBDA_VALUE
        else:
            lambd = settings.TIKHONOV_LAMBDA
        Ktrans, ve, vp = extended_tofts_tikhonov(C_a, C_t, time_points_s,
                                                 lambd=lambd)
        Ki = Ktrans * 6000
        lamda = ve * 100
        SD_Ki = np.nan
        fit_curve = extended_tofts_model(time_points_s, Ktrans, ve, vp, C_a)
        delta_t = np.diff(time_points_s)[0]
        A = construct_convolution_matrix(C_a, delta_t)
        if settings.AUTO_LAMBDA and settings.AUTO_LAMBDA_VALUE is not None:
            cbf_lambda = settings.AUTO_LAMBDA_VALUE
        else:
            cbf_lambda = settings.TIKHONOV_LAMBDA
        residue = tikhonov_regularization(A, C_t, cbf_lambda)
        CBF = residue_to_cbf(residue[0])
        print(f'[!] Two-compartment Ki: {Ki:.5f} ml/100g/min, '
              f'lambda: {lamda:.5f} ml/100g, vp: {vp:.5f}, CBF: {CBF:.5f}')
        save_values(
            Ki, SD_Ki, lamda, Ki, 0.0,
            subtype_tissue, slice_tissue,
            actual_artery, venous_label, arterial_label,
            analysis_directory, suffix='_tikhonov'
        )

    if use_patlak:
        Ki, lamda, SD_Ki = patlak_analysis(C_t, C_a, time_points_s,
                                          subtype_tissue, image_directory)
        baseline_point = find_shifted_baseline(C_t)+1
        if settings.AUTO_LAMBDA and settings.AUTO_LAMBDA_VALUE is not None:
            lam_fit = settings.AUTO_LAMBDA_VALUE
        else:
            lam_fit = settings.TIKHONOV_LAMBDA
        P, P_std = compute_average_permeability(
            C_a, C_t, time_points_s,
            baseline_point=baseline_point, lambd=lam_fit)
        print('[!] Advanced computation of permeability: (', P, '+-', P_std,
              ') ml/100g/min')
        print(f'[!] Ki: {Ki*6000} ml/100g min, lambda: {lamda*100} ml/100g, SD_Ki: {SD_Ki*6000}')
        save_values(Ki, SD_Ki, lamda, P, P_std, subtype_tissue, slice_tissue,
                    actual_artery, venous_label, arterial_label,
                    analysis_directory, suffix='_patlak')

    if selected_subtype_artery == 'Max':
        if use_two_compartment:
            json1 = os.path.join(analysis_directory, 'values_tikhonov.json')
            json2 = os.path.join(analysis_directory, 'max_info.json')
            replace_max_with_artery_type_and_delete(json1, json2)
        if use_patlak:
            json1 = os.path.join(analysis_directory, 'values_patlak.json')
            json2 = os.path.join(analysis_directory, 'max_info.json')
            replace_max_with_artery_type_and_delete(json1, json2)
    restart_prompt = input('[!] Repeat analysis? (y/n): ')
    if restart_prompt.lower() == 'y':
        subtype_tissue, subtype_artery, slice_tissue, (venous_slice, arterial_slice) = permeability_user_interface(analysis_directory)

        selected_subtype_artery = subtype_artery

        C_t = np.load(os.path.join(analysis_directory, 'CTC Data', 'Tissue', subtype_tissue, f'CTC_slice_{slice_tissue}.npy'))
        C_a, input_metadata = get_input_function_curve(
            analysis_directory,
            subtype=subtype_artery,
            venous_slice=venous_slice,
            arterial_slice=arterial_slice
        )

        actual_artery = input_metadata.get('artery_subtype', subtype_artery)
        actual_venous = input_metadata.get('venous_slice', venous_slice)
        actual_arterial = input_metadata.get('arterial_slice', arterial_slice)
        venous_label = 'N/A' if actual_venous is None else str(actual_venous)
        arterial_label = 'N/A' if actual_arterial is None else str(actual_arterial)

        C_t = C_t[0:len(C_a)]
        time_points_s = time_points_s[0:len(C_a)]
        model = settings.KINETIC_MODEL.lower()
        use_two_compartment = model in ('two_compartment', 'both')
        use_patlak = model in ('patlak', 'both')

        if use_two_compartment:
            if settings.AUTO_LAMBDA and settings.AUTO_LAMBDA_VALUE is not None:
                lambd = settings.AUTO_LAMBDA_VALUE
            else:
                lambd = settings.TIKHONOV_LAMBDA
            Ktrans, ve, vp = extended_tofts_tikhonov(
                C_a, C_t, time_points_s, lambd=lambd)
            Ki = Ktrans * 6000
            lamda = ve * 100
            SD_Ki = np.nan
            fit_curve = extended_tofts_model(time_points_s, Ktrans, ve, vp, C_a)
            delta_t = np.diff(time_points_s)[0]
            A = construct_convolution_matrix(C_a, delta_t)
            if settings.AUTO_LAMBDA and settings.AUTO_LAMBDA_VALUE is not None:
                cbf_lambda = settings.AUTO_LAMBDA_VALUE
            else:
                cbf_lambda = settings.TIKHONOV_LAMBDA
            residue = tikhonov_regularization(A, C_t, cbf_lambda)
            CBF = residue_to_cbf(residue[0])
            print(f'[!] Two-compartment Ki: {Ki:.5f} ml/100g/min, '
                  f'lambda: {lamda:.5f} ml/100g, vp: {vp:.5f}, CBF: {CBF:.5f}')
            save_values(
                Ki, SD_Ki, lamda, Ki, 0.0,
                subtype_tissue, slice_tissue,
                actual_artery, venous_label, arterial_label,
                analysis_directory, suffix='_tikhonov'
            )

        if use_patlak:
            Ki, lamda, SD_Ki = patlak_analysis(C_t, C_a, time_points_s, subtype_tissue, image_directory)
            baseline_point = find_shifted_baseline(C_t)+1
            if settings.AUTO_LAMBDA and settings.AUTO_LAMBDA_VALUE is not None:
                lam_fit = settings.AUTO_LAMBDA_VALUE
            else:
                lam_fit = settings.TIKHONOV_LAMBDA
            P, P_std = compute_average_permeability(
                C_a, C_t, time_points_s,
                baseline_point=baseline_point, lambd=lam_fit)
            print('[!] Advanced computation of permeability: (', P, '+-', P_std, ') ml/100g/min')
            print(f'[!] Ki: {Ki*6000} ml/100g min, lambda: {lamda*100} ml/100g, SD_Ki: {SD_Ki*6000}')
            save_values(Ki, SD_Ki, lamda, P, P_std, subtype_tissue, slice_tissue,
                        actual_artery, venous_label, arterial_label,
                        analysis_directory, suffix='_patlak')

        if selected_subtype_artery == 'Max':
            if use_two_compartment:
                json1 = os.path.join(analysis_directory, 'values_tikhonov.json')
                json2 = os.path.join(analysis_directory, 'max_info.json')
                replace_max_with_artery_type_and_delete(json1, json2)
            if use_patlak:
                json1 = os.path.join(analysis_directory, 'values_patlak.json')
                json2 = os.path.join(analysis_directory, 'max_info.json')
                replace_max_with_artery_type_and_delete(json1, json2)
        leaver()  
    else:
        leaver()  
