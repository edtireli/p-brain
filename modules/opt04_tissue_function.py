from utils.plotting import *
from utils.loading import *
import nibabel as nib
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from matplotlib.path import Path
from collections import defaultdict
from scipy.ndimage import zoom
from termcolor import colored
from utils.mapping import *
import glob


def patlak_analysis_plotting(c_tissue, c_input, time):
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
    
    return Ki, lambda_, SD_Ki, x_patlak, y_patlak


def extended_tofts_model(t, Ktrans, ve, vp, Cp):
    integrand = np.zeros_like(t)
    for i in range(len(t)):
        min_len = min(len(Cp[:i+1]), len(t[:i+1]))
        integral = np.trapz(Cp[:min_len] * np.exp(-(t[i] - t[:min_len]) * Ktrans / ve), x=t[:min_len])
        integrand[i] = integral
    min_len = min(len(integrand), len(Cp))
    return Ktrans * integrand[:min_len] + vp * Cp[:min_len]



def find_major_peaks(gradient, radius=10):
    """
    Finds the indices of the two major peaks in the given 1D array based on the gradient.
    
    Parameters:
        gradient (numpy.ndarray): The 1D array containing the gradient data.
        radius (int): The radius around the peaks for filtering out subdominant peaks.
        
    Returns:
        list: The indices of the two major peaks.
    """
    # Identify peaks
    peak_indices = argrelextrema(gradient, np.greater)[0]
    peak_values = gradient[peak_indices]
    
    # Sort peaks by value
    sorted_peak_indices = [x for _, x in sorted(zip(peak_values, peak_indices), reverse=True)]
    
    # Extract the two major peaks based on radius
    major_peaks = []
    for peak in sorted_peak_indices:
        if all(abs(peak - mp) >= radius for mp in major_peaks):
            major_peaks.append(peak)
            if len(major_peaks) >= 2:
                break
    return major_peaks


def find_baseline_point_advanced(y_data, fs=15, cutoff=4.0, order=3, radius=10):
    """
    Finds the baseline point in the given 1D array of y-values based on advanced filtering and gradient analysis.
    
    Parameters:
        y_data (numpy.ndarray): The 1D array containing the data.
        fs (int): Sampling frequency for the low-pass filter.
        cutoff (float): Cutoff frequency for the low-pass filter.
        order (int): Order of the low-pass filter.
        radius (int): The radius around the peaks for filtering out subdominant peaks.
        
    Returns:
        int: The index of the baseline point.
    """
    # Ignore the first point
    y_data = y_data[1:]
    
    # Apply the low-pass filter
    y_filtered = butter_lowpass_filter(y_data, cutoff, fs, order)
    
    # Compute the gradient of the filtered data
    gradient_filtered = np.gradient(y_filtered)
    
    # Find the major peaks in the gradient
    major_peaks_gradient = find_major_peaks(gradient_filtered, radius)
    
    # Find the baseline points as the points right before the major peaks in the gradient
    baseline_points_gradient = [peak - 1 for peak in major_peaks_gradient]
    
    # Select the baseline point with the smaller index
    baseline_point = min(baseline_points_gradient) if baseline_points_gradient else None
    
    # Adjust the index due to ignoring the first point
    if baseline_point is not None:
        baseline_point += 1
    
    return baseline_point



def compute_average_permeability(c_in, c_out, time_array, baseline_point):
    if c_in.shape[0] != c_out.shape[0]:
        raise ValueError("The number of time points in c_in and c_out must be the same.")
    
    initial_guess = [0.5/6000, 0.2, 0.05]
    
    popt, pcov = curve_fit(lambda t, Ktrans, ve, vp: extended_tofts_model(t, Ktrans, ve, vp, c_in),
                        time_array, c_out, p0=initial_guess)
    
    Ktrans_fitted, ve_fitted, vp_fitted = popt
    
    residuals = c_out - extended_tofts_model(time_array, Ktrans_fitted, ve_fitted, vp_fitted, c_in)
    std_dev_Ktrans = np.sqrt(np.diag(pcov))[0]
    
    # Convert to mM/min
    Ktrans_fitted_mM_min = Ktrans_fitted * 6000
    std_dev_Ktrans_mM_min = std_dev_Ktrans * 6000
    
    return Ktrans_fitted_mM_min, std_dev_Ktrans_mM_min


def plot_rois_and_curves(selected_voxels, data_4d, data_3d, T1_matrix, M0_matrix, choice = 1, analysis_directory='dir', image_directory='dir', time_points_s = 1):
    num_rois = sum(len(roi_list) for roi_list in selected_voxels.values())
    gs = gridspec.GridSpec(3, num_rois, height_ratios=[1, 1.5, 1])
    fig = plt.figure(figsize=(20, 12))
    idx = 0
    for slice_index, roi_voxels_list in selected_voxels.items():
        for roi_num, roi_voxels in enumerate(roi_voxels_list):
            all_C_t = []
            all_unnormalized_C_t = []
            roi_voxels_downsampled = np.floor_divide(roi_voxels, 2)
            for (x, y) in roi_voxels_downsampled:
                voxel_time_course = data_4d[x, y, slice_index, :]
                T1 = T1_matrix[x, y, slice_index]
                M0 = M0_matrix[x, y, slice_index]
                C_t_0 = compute_CTC(voxel_time_course, T1, r1=4000, TD=120, m0=M0, slice=slice_index, prints=False)
                baseline_point = find_baseline_point_advanced(C_t_0)
                C_t = custom_shifter(C_t_0, baseline_point)
                all_C_t.append(C_t)
                all_unnormalized_C_t.append(C_t_0)
            avg_C_t_0 = np.mean(all_C_t, axis=0)
            baseline_point = find_baseline_point_advanced(avg_C_t_0) - 1
            avg_C_t = custom_shifter(avg_C_t_0, baseline_point)
            
            # Get Patlak data
            max_file = os.listdir(os.path.join(analysis_directory, 'TSCC Data', 'Max'))[0]
            chosen_venous_slice, chosen_arterial_slice = max_file.split('_')[2:4]
            chosen_arterial_slice = chosen_arterial_slice.split('.')[0]
            C_a = np.load(os.path.join(analysis_directory, 'TSCC Data', 'Max', f'TSCC_slice_{chosen_venous_slice}_{chosen_arterial_slice}.npy'))
            C_t = avg_C_t[0:len(C_a)]
            time_points = time_points_s[0:len(C_a)]
            Ki, lambda_, SD_Ki, x_patlak, y_patlak = patlak_analysis_plotting(C_t, C_a, time_points)
            baseline_point_f = find_shifted_baseline(C_t)+1
            P, P_std = compute_average_permeability(C_a, C_t, time_points_s, baseline_point=baseline_point_f)
            
            ax1 = plt.subplot(gs[0, idx])
            ax1.plot(avg_C_t)
            ax1.set_title(f'Concentration (Slice {slice_index+1} - ROI {roi_num+1})', fontsize=8)
            ax1.grid(True)
            
            ax2 = plt.subplot(gs[1, idx])
            ax2.imshow(data_3d[:, :, slice_index], cmap='magma', origin='lower')
            for x, y in roi_voxels:
                rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
                ax2.add_patch(rect)
            ax2.set_title(f'T2 Image (Slice {slice_index+1} - ROI {roi_num+1})', fontsize=8)
            
            ax3 = plt.subplot(gs[2, idx])
            ax3.scatter(x_patlak, y_patlak, c='black', s=2)
            ax3.plot(x_patlak, lambda_ + Ki * x_patlak, c='red', linestyle='--')
            ax3.set_ylim(min(y_patlak), max(y_patlak))
            ax3.set_title(f'$K_i = {round(Ki*6000, 5)}$, $\\lambda = {round(lambda_*100, 5)}$', fontsize=8)
            ax3.set_xlabel(f'P = {round(P, 5)}')
            ax3.grid(True)
            
            idx += 1

    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    if choice == 2:
        plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', f'Grey_Matter.png'), dpi=200) 
    elif choice == 1:
        plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', f'White_Matter.png'), dpi=200)   
    elif choice == 3: 
        plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', f'Mixed_Matter.png'), dpi=200)    
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.show()
    plt.tight_layout()
    plt.close()

def plot_time_intensity_curves_and_CTC_t2(data, data2, roi_voxels, roi_voxels_upscaled, slice_index, r1=4000, TD=120, type='test', subtype='test', skipshift=False, time_points_s = 1, analysis_directory = 'dir', image_directory = 'dir'):
    N = data.shape[0]
    
    all_C_t = []
    all_unnormalized_C_t = []
    T1_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')), -1, axes=(0, 1))
    M0_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')), -1, axes=(0, 1))

    for (x, y) in roi_voxels:
        voxel_time_course = data[x, y, slice_index, :]
        T1 = T1_matrix[x, y, slice_index]
        M0 = M0_matrix[x, y, slice_index]
        C_t_0 = compute_CTC(voxel_time_course, T1, TD, r1=r1, m0=M0, slice=slice_index, prints=False)
        baseline_point = find_baseline_point_advanced(C_t_0)
        C_t = custom_shifter(C_t_0, baseline_point)
        all_C_t.append(C_t)  
        all_unnormalized_C_t.append(C_t_0 )
    # Averaging all the C_t curves
    avg_C_t_0 = np.mean(all_C_t, axis=0)
    avg_unnormalized_C_t_0 = np.mean(all_unnormalized_C_t, axis=0)
    baseline_point = find_baseline_point_advanced(avg_C_t_0)-1
    print('[!] Baseline point chosen: ', baseline_point)
    avg_C_t = custom_shifter(avg_C_t_0, baseline_point)
    
    fs = 15
    cutoff = 4.0
    order = 3
    


    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})
    axs[0].plot(time_points_s, avg_C_t, color='k', label='Normalised')
    axs[0].set_xlabel('Time (sec)', fontproperties=prop, fontsize=12)
    axs[0].set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
    axs[0].set_title(f'Normalised {type} Concentration (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()

    #np.save(os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{slice_index+1}_unshifted.npy'), avg_C_t)
    # Equilibrium Magnetisation Map
    axs[1].plot(time_points_s, avg_unnormalized_C_t_0, color='k', label='Un-Normalised')
    #axs[1].scatter(time_points_s, avg_unnormalized_C_t_0, color='r', label='Normalised')
    axs[1].axvline(time_points_s[baseline_point], color='red', linestyle = '--')
    axs[1].set_xlabel('Time (sec)', fontproperties=prop, fontsize=12)
    axs[1].set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
    axs[1].set_title(f'Un-Normalised {type} Concentration (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)

    axs[1].grid(which='minor', alpha=0.25)
    axs[1].minorticks_on()
    plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', type, f'CTC+ROI_slice_{slice_index+1}_normalisation.png'), dpi=200)
    
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.show()
    plt.close()

    shift_manual = input('[!] Manually shift baseline point? (y/n): ')
    if shift_manual.lower() == 'y':
        indices = list(range(0, len(avg_unnormalized_C_t_0)))
        ax = plt.gca()  # Get current axes
        ax.plot(indices,avg_unnormalized_C_t_0, color='k', label='Un-Normalised')
        ax.scatter(indices,avg_unnormalized_C_t_0, color='r', label='Un-Normalised', s=5)
        ax.axvline(baseline_point, color='red', linestyle = '--')
        ax.set_xlabel('Time (sec)', fontproperties=prop, fontsize=12)
        ax.set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
        ax.set_title(f'Un-Normalised {type} Concentration (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)

        ax.grid(which='minor', alpha=0.25)
        ax.minorticks_on()
        plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
        plt.show()
        plt.close()
        baseline_point = int(input('[!] Pick baseline point: '))
        avg_C_t = custom_shifter(avg_C_t_0, baseline_point)


    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})
    smoothed_values = butter_lowpass_filter(avg_C_t, cutoff, fs, order)
    
    # Concentration-Time Curve
    axs[0].plot(time_points_s, smoothed_values, color='black')
    axs[0].scatter(time_points_s, avg_C_t, color='r', s=5)
    axs[0].set_xlabel('Time (sec)', fontproperties=prop, fontsize=12)
    axs[0].set_ylabel('Concentration (mM)', fontproperties=prop, fontsize=12)
    axs[0].set_title(f'Average Concentration-Time Curve (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
    axs[0].grid(which='minor', alpha=0.25)
    axs[0].minorticks_on()

    #np.save(os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{slice_index+1}_unshifted.npy'), avg_C_t)
    # Equilibrium Magnetisation Map
    axs[1].imshow(data2[:, :, slice_index], cmap='magma', origin='lower')
    for x,y in roi_voxels_upscaled:
        rect = Rectangle((y, x), 1, 1, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
        axs[1].add_patch(rect)
    axs[1].set_title(f'T2-weighted Image (Slice {slice_index + 1})', fontproperties=prop, fontsize=14)
    plt.savefig(os.path.join(image_directory, 'Concentration Time Curves', 'Tissue', type, f'CTC+ROI_slice_{slice_index+1}.png'), dpi=200)
    
    plt.gcf().canvas.mpl_connect('key_press_event', on_esc)
    plt.show()
    plt.close()

    np.save(os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{slice_index+1}.npy'), avg_C_t)
    return avg_C_t




class ROISelector_tissue:
    def __init__(self, data, slice_index=None):
        self.data = data
        # Set the slice_index to the middle of the data if not provided
        if slice_index is None:
            self.slice_index = data.shape[2] // 2
        else:
            self.slice_index = slice_index
        self.roi_points = []
        self.roi_slices = defaultdict(list)
        self.zoom_level = 0
        self.fig, self.ax = plt.subplots()
        self.redraw()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def onclick(self, event):
        if event.inaxes != self.ax: return
        x, y = int(event.xdata), int(event.ydata)
        if event.key == 'shift':
            self.roi_points.append(self.roi_points[0]) 
            self.redraw()
        elif event.key == 'z':
            self.zoom_center = (x, y)
            self.zoom_level = (self.zoom_level + 1) % 5
            self.redraw()
        else:
            self.roi_points.append((x, y))
            self.redraw()

    def on_key(self, event):
        if event.key == 'escape':
            plt.close(self.fig)
        elif event.key == 'left':
            self.slice_index = (self.slice_index - 1) % self.data.shape[2]
            self.redraw()
        elif event.key == 'right':
            self.slice_index = (self.slice_index + 1) % self.data.shape[2]
            self.redraw()
        elif event.key == 'enter':
            self.find_enclosed_voxels()
            self.roi_points = []
            self.redraw()

    def find_enclosed_voxels(self):
        N = self.data.shape[0]
        path = Path(self.roi_points)
        x, y = np.meshgrid(np.arange(N), np.arange(N))
        points = np.column_stack((x.ravel(), y.ravel()))
        mask = path.contains_points(points)
        mask = mask.reshape(N, N)
        enclosed_voxels = np.argwhere(mask)
        self.roi_slices[self.slice_index].append(enclosed_voxels) 


    def get_current_frame(self, data):
        return data[:, :, self.slice_index]

    def redraw(self):
        self.ax.clear()
        frame = self.get_current_frame(self.data)

        # Calculate aspect ratio to stretch the image
        y_size, x_size = frame.shape
        aspect_ratio = x_size / y_size

        # Stretch the image to fill a square plot
        self.ax.imshow(frame, cmap='viridis', origin='lower', aspect=aspect_ratio)

        if self.zoom_level > 0 and self.zoom_center:
            # Adjust the zoom center and zoom limits
            x_center, y_center = self.zoom_center
            zoom_factor = 2 ** self.zoom_level
            x_zoom = min(x_size, y_size * aspect_ratio) / zoom_factor
            y_zoom = min(y_size, x_size / aspect_ratio) / zoom_factor

            x_start = max(0, x_center - x_zoom / 2)
            x_end = min(x_size, x_center + x_zoom / 2)
            y_start = max(0, y_center - y_zoom / 2)
            y_end = min(y_size, y_center + y_zoom / 2)

            self.ax.set_xlim(x_start, x_end)
            self.ax.set_ylim(y_start, y_end)

        self.title = self.ax.set_title(f'Slice {self.slice_index + 1}', fontsize=15)    
        if self.roi_points:
            x, y = zip(*self.roi_points)
            self.ax.plot(x, y, 'r-', markersize=0.5, alpha=0.75)
            self.ax.plot(x, y, 'ro', markersize=2)
            self.ax.fill(x, y, 'r', alpha=0.3)

        self.fig.canvas.draw()
        
    def get_selected_voxels(self):
        return self.roi_slices


def start_roi_selection_tissue(filename_t2, filename_dce, rotate_AC=True, time_points=1, analysis_directory='dir', image_directory='dir'):
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-Instructions-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    print("1. Left " +colored('click', 'cyan') +" to select ROI points.")
    print("2. Press " +colored('shift', 'cyan') +" to close the ROI.")
    print("3. Press " +colored('enter', 'cyan') +" to save the current ROI.")
    print("4. Use " +colored('left/right', 'cyan') +" arrows to change slices.")
    print("6. Press " +colored('z', 'cyan') +" to zoom in/out.")
    print("7. Press " +colored('Esc', 'red') +" to close the GUI.")
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    print(filename_dce)
    data_3d = nib.load(filename_t2).get_fdata()
    if rotate_AC==True:
        data_3d = np.rot90(data_3d, k=-1, axes=(0, 1))
    selector = ROISelector_tissue(data_3d)
    selected_voxels = selector.get_selected_voxels()

    data_4d = nib.load(filename_dce).get_fdata()
    if rotate_AC==True:
        data_4d = np.rot90(data_4d, k=-1, axes=(0, 1))

    print('['+colored('!', 'cyan')+'] Please identify the selected ' +colored('anatomical', 'red') +' structure:')
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    print(colored('w ', 'red')+": White Matter")
    print(colored('g ', 'red')+": Grey Matter")
    print(colored('m ', 'red')+": Mixed Matter")
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))

    choice_str = input('['+colored('!', 'cyan')+'] Enter the ' +colored('letter', 'cyan') +' corresponding to your choice: ')
    choice = choicestr2int_tissue(choice_str)
    if choice !=3:
        type = choice2type_tissue(choice_str)
        T1_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')), -1, axes=(0, 1))
        M0_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')), -1, axes=(0, 1))
        t1_map_upscaled = zoom(T1_matrix, (2, 2, 1), order=3)
        m0_map_upscaled = zoom(M0_matrix, (2, 2, 1), order=3)

        num_rois = sum(len(roi_list) for roi_list in selected_voxels.values())
        if num_rois > 1:
            plot_rois_and_curves(selected_voxels, data_4d, data_3d, T1_matrix, M0_matrix, time_points_s = time_points, choice = 3, analysis_directory= analysis_directory, image_directory = image_directory)
            
            selected_str = input("Select the index of the ROI curve you want to proceed with (format: slice-roi): ")
            try:
                selected_slice_idx, selected_roi_idx = map(int, selected_str.split('-'))
            except ValueError:
                print("Invalid format. Exiting.")
                return
            selected_slice_idx -= 1
            if selected_slice_idx not in selected_voxels:
                print("Invalid slice index. Exiting.")
                return

            if selected_roi_idx > len(selected_voxels[selected_slice_idx]) or selected_roi_idx < 1:
                print("Invalid ROI index. Exiting.")
                return

            selected_roi_voxels = selected_voxels[selected_slice_idx][selected_roi_idx - 1]
            selected_roi_voxels_downsampled = np.floor_divide(selected_roi_voxels, 2)
            curve = plot_time_intensity_curves_and_CTC_t2(data_4d, data_3d, selected_roi_voxels_downsampled, selected_roi_voxels, selected_slice_idx, type=type, skipshift=False)
            correction_prompt = input('[!] Correct tissue concentration curve of anomalous behavior? (y/n): ')
            if correction_prompt == 'y':
                correction_text = f'{type} signal jump corrected. '
                notes_path = os.path.join(analysis_directory, 'analysis_notes.txt')
                with open(notes_path, 'a') as f:
                    f.write(correction_text)    
                curve_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{selected_slice_idx+1}.npy')
                curve_path2 = glob.glob(os.path.join(analysis_directory, 'TSCC Data', 'Max', '*.npy'))[0]
                while True:
                    editor = ConcentrationCurveEditor(curve, curve_path2, curve_path)
                    corrected_curve_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{selected_slice_idx+1}.npy')
                    plot_corrected_tissue_curve(editor.data, data_3d, selected_roi_voxels, selected_slice_idx, type=type)
                    break
        else: 
            for slice_index, roi_list in selected_voxels.items():
                for roi_voxels in roi_list:
                    roi_voxels_downsampled = roi_voxels // 2
                    curve = plot_time_intensity_curves_and_CTC_t2(data_4d, data_3d, roi_voxels_downsampled, roi_voxels, slice_index, type=type, skipshift=False, analysis_directory= analysis_directory, image_directory = image_directory)
                    correction_prompt = input('[!] Correct tissue concentration curve of anomalous behavior? (y/n): ')
                    if correction_prompt == 'y':
                        curve_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{slice_index+1}.npy')
                        curve_path2 = glob.glob(os.path.join(analysis_directory, 'TSCC Data', 'Max', '*.npy'))[0]
                        while True:
                            editor = ConcentrationCurveEditor(curve, curve_path2, curve_path)
                            corrected_curve_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{slice_index+1}.npy')
                            plot_corrected_tissue_curve(editor.data, data_3d, roi_voxels, slice_index, type=type, analysis_directory= analysis_directory, image_directory = image_directory)
                            break
    elif choice==3:
        T1_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')), -1, axes=(0, 1))
        M0_matrix = np.rot90(load_from_pickle(os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')), -1, axes=(0, 1))
        t1_map_upscaled = zoom(T1_matrix, (2, 2, 1), order=3)
        m0_map_upscaled = zoom(M0_matrix, (2, 2, 1), order=3)

        num_rois = sum(len(roi_list) for roi_list in selected_voxels.values())
        plot_rois_and_curves(selected_voxels, data_4d, data_3d, T1_matrix, M0_matrix, time_points_s = time_points, choice = 3, analysis_directory= analysis_directory, image_directory = image_directory)
        
        selected_str_grey = input("Select the Grey Matter index (format: slice-roi): ")
        selected_str_white = input("Select the White Matter index (format: slice-roi): ")
        type_str = ['Grey Matter', 'White Matter']
        selected_str = [selected_str_grey, selected_str_white]
        for i in range(len(selected_str)):
            type = type_str[i]
            try:
                selected_slice_idx, selected_roi_idx = map(int, selected_str[i].split('-'))
            except ValueError:
                print("Invalid format. Exiting.")
                return

            # Decrement the slice index by 1 to make it 0-based
            selected_slice_idx -= 1

            # Now selected_slice_idx contains the slice index and selected_roi_idx contains the ROI index
            # Validate these indices
            if selected_slice_idx not in selected_voxels:
                print("Invalid slice index. Exiting.")
                return

            if selected_roi_idx > len(selected_voxels[selected_slice_idx]) or selected_roi_idx < 1:
                print("Invalid ROI index. Exiting.")
                return

            # Now you can proceed
            selected_roi_voxels = selected_voxels[selected_slice_idx][selected_roi_idx - 1]
            selected_roi_voxels_downsampled = np.floor_divide(selected_roi_voxels, 2)
            curve = plot_time_intensity_curves_and_CTC_t2(data_4d, data_3d, selected_roi_voxels_downsampled, selected_roi_voxels, selected_slice_idx, type=type, skipshift=False, time_points_s = time_points, analysis_directory = analysis_directory, image_directory = image_directory)
            correction_prompt = input('[!] Correct tissue concentration curve of anomalous behavior? (y/n): ')
            if correction_prompt == 'y':
                curve_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{selected_slice_idx+1}.npy')
                curve_path2 = glob.glob(os.path.join(analysis_directory, 'TSCC Data', 'Max', '*.npy'))[0]
                while True:
                    editor = ConcentrationCurveEditor(curve, curve_path2, curve_path)
                    corrected_curve_path = os.path.join(analysis_directory, 'CTC Data', 'Tissue', type, f'CTC_slice_{selected_slice_idx+1}.npy')
                    plot_corrected_tissue_curve(editor.data, data_3d, selected_roi_voxels, selected_slice_idx, type=type, time_points_s=time_points, image_directory=image_directory)
                    break



def tissue_function(analysis_directory, nifti_directory, image_directory, filenames):
    t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename, \
    flair_3D_filename, axial_flair_3D_filename, axial_t2_2D_filename, dce_filename = filenames
    filename_t2 = os.path.join(nifti_directory, axial_t2_2D_filename)
    filename_dce = os.path.join(nifti_directory, dce_filename)
    time_points_s = np.load(os.path.join(analysis_directory,'Fitting', 'time_points_s.npy'))
    #np.save(os.path.join(analysis_directory, 'time_points_s.npy'), time_points_s)
    start_roi_selection_tissue(filename_t2, filename_dce, rotate_AC=True, time_points=time_points_s, analysis_directory=analysis_directory, image_directory=image_directory)
    rerun = input('[!] Repeat analysis? (y/n): ')
    if rerun == 'y':
        filename_t2 = os.path.join(nifti_directory, axial_t2_2D_filename)
        filename_dce = os.path.join(nifti_directory, dce_filename)
        time_points_s = np.load(os.path.join(analysis_directory,'Fitting', 'time_points_s.npy'))
        start_roi_selection_tissue(filename_t2, filename_dce, rotate_AC=True, time_points=time_points_s, analysis_directory=analysis_directory, image_directory=image_directory)
        leaver()
    else: 
        leaver()


