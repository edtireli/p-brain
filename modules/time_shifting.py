
from utils.mapping import *
from utils.loading import *
import nibabel as nib
from termcolor import colored
from utils.plotting import *
import glob
import json
import time


def plot_transformed_curves(shifted_vein_curve, shifted_artery_curve, slice_index, arterial_slice_index, vein_top2_peaks, time_points_s, analysis_directory, image_directory, subtype='test', scaling=1, time_shift=1):
    subtype=subtype[1]
    def on_esc(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    fs = 15
    cutoff = 4.0
    order = 3
    smoothed_vein = butter_lowpass_filter(shifted_vein_curve, cutoff, fs, order)
    smoothed_artery = butter_lowpass_filter(shifted_artery_curve, cutoff, fs, order)
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

    plt.savefig(os.path.join(image_directory, 'Time Shifted Concentration Curves', subtype, f'TSCC_slice_{slice_index}_{arterial_slice_index}.png'), dpi=200)
    np.save(os.path.join(analysis_directory, 'TSCC Data', subtype, f'TSCC_slice_{slice_index}_{arterial_slice_index}.npy'), shifted_vein_curve)
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
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
                slice_file = os.path.join(artery_dir, f'CTC_slice_{i}.npy')
                if os.path.exists(slice_file):
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

def select_slices(available_arterial_slices, available_venous_slices):
    print("Available arterial slices:", available_arterial_slices)
    arterial_slice = int(input("Select an arterial slice: "))
    
    print("Available venous slices:", available_venous_slices)
    venous_slice = int(input("Select a venous slice: "))
    
    return arterial_slice, venous_slice


from scipy.signal import correlate, find_peaks

def align_first_peaks(vein_curve, artery_curve, radius=10, double_peak_radius=3):
    cross_corr = correlate(vein_curve, artery_curve)
    shift = np.argmax(cross_corr) - len(vein_curve) + 1

    if shift >= 0:
        aligned_vein_curve = vein_curve[shift:]
    else:
        aligned_vein_curve = np.concatenate([vein_curve[-shift:], np.zeros(-shift)])

    vein_peaks, _ = find_peaks(aligned_vein_curve)
    artery_peaks, _ = find_peaks(artery_curve)

    def filter_double_peaks(peaks, curve, radius):
        sorted_peaks = sorted(peaks, key=lambda x: curve[x], reverse=True)
        top2_peaks = []
        
        for p in sorted_peaks:
            if all(abs(p - tp) > radius for tp in top2_peaks):
                top2_peaks.append(p)
                if len(top2_peaks) == 2:
                    break

        return top2_peaks

    vein_top2_peaks = filter_double_peaks(vein_peaks, aligned_vein_curve, radius)
    artery_top2_peaks = filter_double_peaks(artery_peaks, artery_curve, radius)

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
    start_value = curve[0]
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



def find_max_npy_file(analysis_directory):
    subtypes = ["Left Interior Carotid", "Right Interior Carotid", "Basilar", "Left Middle Cerebral", "Right Middle Cerebral"]
    max_value = float('-inf')
    max_file_path = ""
    max_subtype = ""
    max_slice_index = -1
    max_arterial_slice_index = -1  

    for subtype in subtypes:
        file_paths = glob.glob(os.path.join(analysis_directory, 'TSCC Data', subtype, '*.npy'))

        for file_path in file_paths:
            arr = np.load(file_path)
            curr_max = np.max(arr)

            if curr_max > max_value:
                max_value = curr_max
                max_file_path = file_path
                max_subtype = subtype
                split_filename = file_path.split('_')
                max_slice_index = int(split_filename[-2])  
                max_arterial_slice_index = int(split_filename[-1].split('.npy')[0])  

    return max_file_path, max_value, max_subtype, max_slice_index, max_arterial_slice_index  


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

    plt.savefig(os.path.join(image_directory, 'Time Shifted Concentration Curves', 'Max', f'TSCC_slice_{slice_index}_{artery_index}.png'), dpi=200)
    np.save(os.path.join(analysis_directory, 'TSCC Data', 'Max', f'TSCC_slice_{slice_index}_{artery_index}.npy'), shifted_vein_curve)
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.show()



def time_shifting(analysis_directory, nifti_directory, image_directory):
    time_points_s = np.load(os.path.join(analysis_directory,'Fitting', 'time_points_s.npy'))

    available_arteries, slices = get_available_arteries(analysis_directory)
    display_available_arteries(available_arteries)
    choice = input('[' + colored('!', 'cyan') + '] Enter the number corresponding to your choice: ')        
    subtype = choice2subtype(choice)
    artery_choice = available_arteries[choice]

    arterial_slices = get_available_slices(analysis_directory, 'Artery', artery_choice)
    venous_slices = get_available_slices(analysis_directory, 'Vein', 'Sinus Sagittalis')

    auto_choice = input("[!] Automatically iterate through all slices? (y/n): ")
    
    if auto_choice.lower() == 'y':
        for arterial_slice in arterial_slices:
            for venous_slice in venous_slices:
                vein_curve, artery_curve = load_curves(venous_slice, arterial_slice, artery_choice, analysis_directory)
                aligned_vein_curve, peaks, rescaled, time_shift = align_first_peaks(vein_curve, artery_curve)
                aligned_vein_curve_no_zeros = remove_trailing_zeros(aligned_vein_curve)
                aligned_vein_curve_no_zeros_shifted = shift_curve_to_zero_start(aligned_vein_curve_no_zeros)
                plot_transformed_curves(aligned_vein_curve_no_zeros_shifted, artery_curve, venous_slice, arterial_slice, time_points_s = time_points_s, analysis_directory = analysis_directory, image_directory = image_directory, subtype=subtype, vein_top2_peaks=peaks, scaling=rescaled, time_shift=time_shift)
    elif auto_choice.lower() == 'n':
        arterial_slice = select_arterial_slice(arterial_slices) 
        venous_slice = select_venous_slice(venous_slices)  
        
        vein_curve, artery_curve = load_curves(venous_slice, arterial_slice, artery_choice)
        aligned_vein_curve, peaks, rescaled, time_shift = align_first_peaks(vein_curve, artery_curve)
        aligned_vein_curve_no_zeros = remove_trailing_zeros(aligned_vein_curve)
        aligned_vein_curve_no_zeros_shifted = shift_curve_to_zero_start(aligned_vein_curve_no_zeros)
        plot_transformed_curves(aligned_vein_curve_no_zeros_shifted, artery_curve, venous_slice, arterial_slice, time_points_s = time_points_s, analysis_directory = analysis_directory, image_directory = image_directory, subtype=subtype, vein_top2_peaks=peaks, scaling=rescaled, time_shift=time_shift)

    print('[!] Finding maximum')
    time.sleep(1)
    
    max_file_path, max_value, max_subtype, max_slice_index, max_arterial_slice_index = find_max_npy_file(analysis_directory)
    max_curve = np.load(max_file_path)
    corresponding_vein_curve = np.load(os.path.join(analysis_directory, 'TSCC Data', max_subtype, f'TSCC_slice_{max_slice_index}_{max_arterial_slice_index}.npy'))
    [os.remove(f) for f in glob.glob(os.path.join(analysis_directory, 'TSCC Data', 'Max', '*.npy'))]
    plot_transformed_curves_max(corresponding_vein_curve, slice_index=max_slice_index, artery_index = max_arterial_slice_index, vein_top2_peaks=[0,0], subtype=max_subtype, time_points_s = time_points_s, analysis_directory = analysis_directory, image_directory = image_directory)
    values = [f'Max artery type: {max_subtype}']
    with open(os.path.join(analysis_directory, 'max_info.json'), 'w') as f:
        json.dump(values, f)