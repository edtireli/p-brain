
from utils.mapping import *
from utils.loading import *
import nibabel as nib
from termcolor import colored
from utils.plotting import *
from utils.qc import detect_truncated_bolus
import glob
import json
import time

import utils.settings as settings

turbo_mode = True  # When True, suppress interactive plotting


def plot_transformed_curves(shifted_vein_curve, shifted_artery_curve, slice_index, arterial_slice_index, vein_top2_peaks, time_points_s, analysis_directory, image_directory, subtype='test', scaling=1, time_shift=1):
    subtype=subtype[1]
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    fs = 15
    cutoff = 4.0
    order = 3
    smoothed_vein = butter_lowpass_filter(shifted_vein_curve, cutoff, fs, order)
    plt.figure(figsize=(10, 5))
    if scaling == 1:
        plt.title(f" Time-shifted Concentration Curve for {subtype} (Veinous Slice {slice_index}, Aterial Slice {arterial_slice_index})", fontproperties=prop, fontsize=16)
    elif scaling != 1:
        plt.title(f"Rescaled & Time-shifted Concentration Curve for {subtype} (Veinous Slice {slice_index}, Aterial Slice {arterial_slice_index})", fontproperties=prop, fontsize=16)
    if scaling != 1:
        plt.plot(time_points_s[0:len(shifted_vein_curve)], shifted_vein_curve, label=f'Rescaled ({round(scaling,1)}) & Time-Shifted ({time_shift} s) Vein Curve', color=blaa)
    elif scaling == 1:    
        plt.plot(time_points_s[0:len(shifted_vein_curve)], shifted_vein_curve, label=f'Time-Shifted ({time_shift} s) Vein Curve', color=blaa)
    plt.plot(time_points_s[0:len(shifted_artery_curve)], shifted_artery_curve, label=f'Artery Curve', color=roed, alpha=0.75, linestyle='dotted')

    plt.xlabel('Time (s)', fontproperties=prop, fontsize=14)
    plt.ylabel('Concentration (mM)', fontproperties=prop, fontsize=14)
    plt.legend()
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Time Shifted Concentration Curves', subtype, f'TSCC_slice_{slice_index}_{arterial_slice_index}.png'), dpi=300)
    np.save(os.path.join(analysis_directory, 'TSCC Data', subtype, f'TSCC_slice_{slice_index}_{arterial_slice_index}.npy'), shifted_vein_curve)
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay_plt(3)
        plt.show()


def get_available_arteries(analysis_directory):
    artery_types = {'lica': 'Left Interior Carotid', 'rica': 'Right Interior Carotid', 'b': 'Basilar', 'lmca': 'Left Middle Cerebral', 'rmca': 'Right Middle Cerebral'}
    available_arteries = {}
    available_slices = {}
    
    for code, artery in artery_types.items():
        artery_dir = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery)
        if os.path.exists(artery_dir) and any(fname.endswith('.npy') for fname in os.listdir(artery_dir)):
            available_arteries[code] = artery
            slice_indexes = []
            for i in range(1, 11):
                shifted_slice_file = os.path.join(artery_dir, f'CTC_shifted_slice_{i}.npy')
                slice_file = os.path.join(artery_dir, f'CTC_slice_{i}.npy')
                if os.path.exists(shifted_slice_file):
                    slice_indexes.append(i)
                elif os.path.exists(slice_file):
                    slice_indexes.append(i)
            available_slices[code] = slice_indexes

    return available_arteries, available_slices


def display_available_arteries(available_arteries):
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    print('['+colored('!', 'cyan')+'] Please identify the '+colored('anatomical structure', 'cyan')+' for each selected ROI:')
    for i, artery in available_arteries.items():
        print(f'{i}: '+colored(f'{artery}', 'cyan')+' Artery')
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))



def get_available_slices(analysis_directory, structure_type, structure_subtype):
    available_slices = []
    for i in range(1, 11):
        slice_file = os.path.join(analysis_directory, 'CTC Data', structure_type, structure_subtype, f'CTC_slice_{i}.npy')
        if os.path.exists(slice_file):
            available_slices.append(i)
    return available_slices



from scipy.signal import correlate, find_peaks

def align_first_peaks(vein_curve, artery_curve, radius=10, double_peak_radius=3, num_peaks=None):
    cross_corr = correlate(vein_curve, artery_curve)
    shift = np.argmax(cross_corr) - len(vein_curve) + 1

    if shift >= 0:
        aligned_vein_curve = vein_curve[shift:]
    else:
        aligned_vein_curve = np.concatenate([vein_curve[-shift:], np.zeros(-shift)])

    vein_peaks, _ = find_peaks(aligned_vein_curve)
    artery_peaks, _ = find_peaks(artery_curve)

    if num_peaks is None:
        num_peaks = int(getattr(settings, "NUMBER_OF_PEAKS", 2))

    def filter_double_peaks(peaks, curve, radius, n_peaks):
        sorted_peaks = sorted(peaks, key=lambda x: curve[x], reverse=True)
        top_peaks = []
        
        for p in sorted_peaks:
            if all(abs(p - tp) > radius for tp in top_peaks):
                top_peaks.append(p)
                if len(top_peaks) >= n_peaks:
                    break

        return top_peaks

    vein_top2_peaks = filter_double_peaks(vein_peaks, aligned_vein_curve, radius, num_peaks)
    artery_top2_peaks = filter_double_peaks(artery_peaks, artery_curve, radius, num_peaks)

    rescaled = 1

    for v_peak, a_peak in zip(vein_top2_peaks, artery_top2_peaks):
        if aligned_vein_curve[v_peak] < artery_curve[a_peak]:
            scaling_factor = artery_curve[a_peak] / aligned_vein_curve[v_peak]
            aligned_vein_curve *= scaling_factor
            rescaled = scaling_factor
            break

    is_double_peak = any(abs(vein_top2_peaks[i] - vein_top2_peaks[j]) <= double_peak_radius for i in range(len(vein_top2_peaks)) for j in range(i+1, len(vein_top2_peaks)))

    if is_double_peak:
        shift += 1
        aligned_vein_curve = aligned_vein_curve[1:]

    return aligned_vein_curve, vein_top2_peaks, rescaled, shift

def remove_trailing_zeros(arr):
    non_zero_idx = np.where(arr != 0)[0]
    if len(non_zero_idx) == 0:
        return np.array([])
    last_non_zero = non_zero_idx[-1]
    return arr[:last_non_zero+1]


def shift_curve_to_zero_start(curve):
    curve = np.asarray(curve)
    if curve.size == 0:
        return curve

    start_value = curve[0]
    if not np.isfinite(start_value):
        # If the first element is NaN/Inf, avoid propagating it as an offset.
        # Let downstream logic decide how to handle a non-finite curve.
        return curve

    return curve - start_value


def select_arterial_slice(available_slices):
    print("Available arterial slices:", available_slices)
    selected_slice = int(input("Select an arterial slice: "))
    if selected_slice not in available_slices:
        print("Invalid choice, exiting.")
        exit(1)
    return selected_slice

def select_venous_slice(available_slices):
    print("Available venous slices:", available_slices)
    selected_slice = int(input("Select a venous slice: "))
    if selected_slice not in available_slices:
        print("Invalid choice, exiting.")
        exit(1)
    return selected_slice



def find_max_npy_file(analysis_directory, num_peaks=None, peak_tolerance=0.20, peak_separation=10):
    subtypes = ["Left Interior Carotid", "Right Interior Carotid", "Basilar", "Left Middle Cerebral", "Right Middle Cerebral"]

    if num_peaks is None:
        num_peaks = int(getattr(settings, "NUMBER_OF_PEAKS", 2))

    def _top_peaks(curve, n_peaks, separation):
        peaks, _ = find_peaks(curve)
        if len(peaks) == 0:
            return []
        sorted_peaks = sorted(peaks, key=lambda x: curve[x], reverse=True)
        chosen = []
        for p in sorted_peaks:
            if not np.isfinite(curve[p]):
                continue
            if curve[p] <= 0:
                continue
            if all(abs(p - cp) > separation for cp in chosen):
                chosen.append(p)
                if len(chosen) >= n_peaks:
                    break
        return chosen

    def _load_artery_curve(artery_subtype, arterial_slice_index):
        artery_dir = os.path.join(analysis_directory, 'CTC Data', 'Artery', artery_subtype)
        for filename in (f'CTC_shifted_slice_{arterial_slice_index}.npy', f'CTC_slice_{arterial_slice_index}.npy'):
            path = os.path.join(artery_dir, filename)
            if os.path.exists(path):
                return np.load(path)
        return None

    def _peaks_consistent(vein_curve, artery_curve, n_peaks, tolerance, separation):
        if artery_curve is None or len(vein_curve) == 0 or len(artery_curve) == 0:
            return False

        vein_peaks = _top_peaks(vein_curve, n_peaks, separation)
        artery_peaks = _top_peaks(artery_curve, n_peaks, separation)
        if len(vein_peaks) < n_peaks or len(artery_peaks) < n_peaks:
            return False

        ratios = []
        for v_peak, a_peak in zip(vein_peaks, artery_peaks):
            v = float(vein_curve[v_peak])
            a = float(artery_curve[a_peak])
            if not np.isfinite(v) or not np.isfinite(a) or a == 0:
                return False
            ratios.append(v / a)

        ratios = np.asarray(ratios, dtype=float)
        median_ratio = float(np.median(ratios))
        if not np.isfinite(median_ratio) or median_ratio == 0:
            return False

        rel_dev = np.abs(ratios - median_ratio) / abs(median_ratio)
        return bool(np.all(rel_dev <= tolerance))

    def _is_truncated_tscc(curve: np.ndarray) -> bool:
        try:
            # Truncation in any dominant bolus peak should disqualify the candidate.
            truncated, _details = detect_truncated_bolus(
                curve,
                num_peaks=num_peaks,
                peak_fraction=0.99,
                plateau_min_points=3,
            )
            return bool(truncated)
        except Exception:
            return False

    best_any = {
        'value': float('-inf'),
        'file_path': "",
        'subtype': "",
        'slice_index': -1,
        'arterial_slice_index': -1,
    }
    best_consistent = {
        'value': float('-inf'),
        'file_path': "",
        'subtype': "",
        'slice_index': -1,
        'arterial_slice_index': -1,
    }

    best_any_truncated = best_any.copy()
    best_consistent_truncated = best_consistent.copy()

    for subtype in subtypes:
        file_paths = glob.glob(os.path.join(analysis_directory, 'TSCC Data', subtype, '*.npy'))
        for file_path in file_paths:
            arr = np.load(file_path)
            curr_max = float(np.max(arr))

            is_trunc = _is_truncated_tscc(arr)

            base = os.path.basename(file_path)
            split_filename = base.split('_')
            if len(split_filename) < 4:
                continue
            slice_index = int(split_filename[-2])
            arterial_slice_index = int(split_filename[-1].split('.npy')[0])

            if is_trunc:
                if curr_max > best_any_truncated['value']:
                    best_any_truncated.update({
                        'value': curr_max,
                        'file_path': file_path,
                        'subtype': subtype,
                        'slice_index': slice_index,
                        'arterial_slice_index': arterial_slice_index,
                    })
            else:
                if curr_max > best_any['value']:
                    best_any.update({
                        'value': curr_max,
                        'file_path': file_path,
                        'subtype': subtype,
                        'slice_index': slice_index,
                        'arterial_slice_index': arterial_slice_index,
                    })

            artery_curve = _load_artery_curve(subtype, arterial_slice_index)
            if _peaks_consistent(arr, artery_curve, num_peaks, peak_tolerance, peak_separation):
                if is_trunc:
                    if curr_max > best_consistent_truncated['value']:
                        best_consistent_truncated.update({
                            'value': curr_max,
                            'file_path': file_path,
                            'subtype': subtype,
                            'slice_index': slice_index,
                            'arterial_slice_index': arterial_slice_index,
                        })
                else:
                    if curr_max > best_consistent['value']:
                        best_consistent.update({
                            'value': curr_max,
                            'file_path': file_path,
                            'subtype': subtype,
                            'slice_index': slice_index,
                            'arterial_slice_index': arterial_slice_index,
                        })

    # Prefer consistent non-truncated; then any non-truncated; finally fall back to truncated
    # so the pipeline still runs even if every candidate looks clipped.
    if best_consistent['file_path']:
        best = best_consistent
    elif best_any['file_path']:
        best = best_any
    elif best_consistent_truncated['file_path']:
        best = best_consistent_truncated
    else:
        best = best_any_truncated

    return (
        best['file_path'],
        best['value'],
        best['subtype'],
        best['slice_index'],
        best['arterial_slice_index'],
    )


def plot_transformed_curves_max(shifted_vein_curve, slice_index, artery_index, vein_top2_peaks, subtype='test', time_points_s=1, analysis_directory='dir', image_directory = 'dir'):
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    fs = 10
    cutoff = 3.0
    order = 3
    smoothed_vein = butter_lowpass_filter(shifted_vein_curve, cutoff, fs, order)

    plt.figure(figsize=(10, 5))
    plt.title(f"Rescaled & Time-shifted Concentration Curve for {subtype} (Veinous Slice {slice_index}, Aterial Slice {artery_index})", fontproperties=prop, fontsize=16)
    plt.plot(time_points_s[0:len(shifted_vein_curve)], smoothed_vein, label=f'Time-Shifted Vein Curve', color=blaa1)
    plt.scatter(time_points_s[0:len(shifted_vein_curve)], shifted_vein_curve, label=f'Time-Shifted Vein Curve', color=roed, s=5)
    
    plt.xlabel('Time (s)', fontproperties=prop, fontsize=14)
    plt.ylabel('Concentration (mM)', fontproperties=prop, fontsize=14)
    plt.legend()
    plt.grid(which='minor', alpha=0.25)
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(image_directory, 'Time Shifted Concentration Curves', 'Max', f'TSCC_slice_{slice_index}_{artery_index}.png'), dpi=300)
    np.save(os.path.join(analysis_directory, 'TSCC Data', 'Max', f'TSCC_slice_{slice_index}_{artery_index}.npy'), shifted_vein_curve)
    if not turbo_mode:
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        close_plot_after_delay_plt(3)
        plt.show()



def time_shifting(analysis_directory, nifti_directory, image_directory):
    """Stage-runner entrypoint (non-interactive).

    Historically this module prompted for artery selection and slice iteration.
    That breaks non-interactive runs and can crash later when no Max candidate
    is selected (slice index -1).

    Delegate to the lightweight, non-interactive TSCC implementation.
    """

    try:
        from modules.time_shifting import time_shifting as _tscc
    except Exception:
        # Fallback to local implementation if import fails.
        _tscc = None

    if _tscc is not None:
        artery = getattr(settings, 'INPUT_FUNCTION_ARTERY', None)
        return _tscc(analysis_directory, nifti_directory, image_directory, artery=artery)

    # If we cannot import the new implementation, keep legacy behaviour but avoid crashing.
    print(colored('[!] TSCC: falling back to legacy opt03_time_shifting implementation.', 'yellow'))

    time_points_s = np.load(os.path.join(analysis_directory, 'Fitting', 'time_points_s.npy'))
    available_arteries, slices = get_available_arteries(analysis_directory)
    if not available_arteries:
        print(colored('[!] No available arteries found; cannot time-shift.', 'yellow'))
        return

    # Default artery from settings when available.
    artery_choice = None
    try:
        pref = str(getattr(settings, 'INPUT_FUNCTION_ARTERY', '') or '').strip().upper()
        if pref in available_arteries:
            artery_choice = pref
    except Exception:
        artery_choice = None
    if artery_choice is None:
        artery_choice = list(available_arteries.values())[0]

    subtype = str(artery_choice)
    arterial_slices = get_available_slices(analysis_directory, 'Artery', artery_choice)
    venous_slices = get_available_slices(analysis_directory, 'Vein', 'Sinus Sagittalis')
    if not arterial_slices or not venous_slices:
        print(colored('[!] Missing artery/vein slices; cannot time-shift.', 'yellow'))
        return

    for arterial_slice in arterial_slices:
        for venous_slice in venous_slices:
            vein_curve, artery_curve = load_curves(venous_slice, arterial_slice, artery_choice, analysis_directory)
            aligned_vein_curve, peaks, rescaled, time_shift = align_first_peaks(vein_curve, artery_curve)
            aligned_vein_curve_no_zeros = remove_trailing_zeros(aligned_vein_curve)
            if aligned_vein_curve_no_zeros.size == 0 or not np.all(np.isfinite(aligned_vein_curve_no_zeros)):
                continue
            aligned_vein_curve_no_zeros_shifted = shift_curve_to_zero_start(aligned_vein_curve_no_zeros)
            if aligned_vein_curve_no_zeros_shifted.size == 0:
                continue
            plot_transformed_curves(
                aligned_vein_curve_no_zeros_shifted,
                artery_curve,
                venous_slice,
                arterial_slice,
                time_points_s=time_points_s,
                analysis_directory=analysis_directory,
                image_directory=image_directory,
                subtype=subtype,
                vein_top2_peaks=peaks,
                scaling=rescaled,
                time_shift=time_shift,
            )

    print('[!] Finding maximum')
    time.sleep(1)
    max_file_path, max_value, max_subtype, max_slice_index, max_arterial_slice_index = find_max_npy_file(analysis_directory)
    if not max_subtype or int(max_slice_index) < 0 or int(max_arterial_slice_index) < 0:
        print(colored('[!] No valid Max TSCC candidate found; skipping Max outputs.', 'yellow'))
        return
    tscc_path = os.path.join(
        analysis_directory,
        'TSCC Data',
        str(max_subtype),
        f'TSCC_slice_{int(max_slice_index)}_{int(max_arterial_slice_index)}.npy',
    )
    if not os.path.exists(tscc_path):
        print(colored(f'[!] Missing TSCC file for Max selection: {tscc_path}', 'yellow'))
        return
    corresponding_vein_curve = np.load(tscc_path)
    [os.remove(f) for f in glob.glob(os.path.join(analysis_directory, 'TSCC Data', 'Max', '*.npy'))]
    plot_transformed_curves_max(
        corresponding_vein_curve,
        slice_index=int(max_slice_index),
        artery_index=int(max_arterial_slice_index),
        vein_top2_peaks=[0, 0],
        subtype=str(max_subtype),
        time_points_s=time_points_s,
        analysis_directory=analysis_directory,
        image_directory=image_directory,
    )
    values = [f'Max artery type: {max_subtype}']
    with open(os.path.join(analysis_directory, 'max_info.json'), 'w') as f:
        json.dump(values, f)