from utils.loading import *
import nibabel as nib 
import numpy as np
from termcolor import colored
from utils.fonts import *
from utils.mapping import *
from utils.plotting import *
from matplotlib.path import Path
import glob

class ROISelector:
    def __init__(self, data, slice_index=0, frame_index=0):
        self.data = data
        self.normalized_data = (data - data.min()) / (data.max() - data.min())
        self.slice_index = slice_index
        self.frame_index = frame_index
        self.frame_index = 15 #np.argmax(data[:, :, self.slice_index, :].sum(axis=(0, 1)))
        self.roi_points = []
        self.roi_slices = {}
        self.zoom_level = 0
        self.zoom_center = None
        self.fig, self.ax = plt.subplots()
        # Connect event handlers for mouse clicks and key presses
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.redraw()
        plt.show()

    def onclick(self, event):
        if event.inaxes != self.ax: return
        x, y = int(event.xdata), int(event.ydata)
        if event.key == 'shift':
            self.roi_points.append(self.roi_points[0])  # Close the ROI
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
            return
        elif event.key == 'left':
            self.slice_index = (self.slice_index - 1) % self.data.shape[2]
            self.redraw()
        elif event.key == 'right':
            self.slice_index = (self.slice_index + 1) % self.data.shape[2]
            self.redraw()
        elif event.key == 'down':
            self.frame_index = (self.frame_index - 1) % self.data.shape[3]
            self.redraw()
        elif event.key == 'up':
            self.frame_index = (self.frame_index + 1) % self.data.shape[3]
            self.redraw()
        elif event.key == 'enter':
            self.find_enclosed_voxels()
            self.extract_peak_voxel_ITC()
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
        self.roi_slices[self.slice_index] = enclosed_voxels
        print('')
        print(f"Slice {self.slice_index+1}: Number of enclosed voxels: {len(self.roi_slices[self.slice_index])}")

    def extract_peak_voxel_ITC(self):
        max_intensity = -1
        max_voxel = None
        for (x, y) in self.roi_slices.get(self.slice_index, []):
            voxel_ITC = self.data[x, y, self.slice_index, :]
            peak_intensity = max(voxel_ITC)
            if peak_intensity > max_intensity:
                max_intensity = peak_intensity
                max_voxel = (x, y)
        print(f"Slice {self.slice_index+1}: Voxel with highest peak: {max_voxel}, Peak Intensity: {round(max_intensity,1)}")


    def get_current_frame(self, data):
        return data[:, :, self.slice_index, self.frame_index]

    def redraw(self):
        self.ax.clear()
        frame = self.get_current_frame(self.normalized_data)
        self.ax.imshow(frame, cmap='viridis', origin='lower')
        if self.zoom_level > 0 and self.zoom_center:
            x_center, y_center = self.zoom_center
            x_size, y_size = frame.shape
            zoom_factor = 1 / (2 ** self.zoom_level)
            x_zoom, y_zoom = x_size * zoom_factor, y_size * zoom_factor
            x_start, x_end = max(0, x_center - x_zoom), min(x_size, x_center + x_zoom)
            y_start, y_end = max(0, y_center - y_zoom), min(y_size, y_center + y_zoom)
            self.ax.set_xlim(x_start, x_end)
            self.ax.set_ylim(y_start, y_end)
        self.title = self.ax.set_title(f'Slice {self.slice_index + 1}, Frame {self.frame_index + 1}', fontproperties=prop, fontsize=15)
        if self.roi_points:
            x, y = zip(*self.roi_points)
            self.ax.plot(x, y, 'r-', markersize=0.5, alpha=0.75)
            self.ax.plot(x, y, 'ro', markersize=2)
            self.ax.fill(x, y, 'r', alpha=0.3)
        self.fig.canvas.draw()


    def get_selected_voxels(self):
        return self.roi_slices
    def get_all_selected_voxels(self):
        return self.roi_slices


def start_roi_selection(filename, rotate_AC=True, time=1, analysis='dir', image='dir', nifti='dir', filenames='filenames', IsVFA=False):
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-Instructions-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    print("1. Left " +colored('click', 'cyan') +" to select ROI points.")
    print("2. Press " +colored('shift', 'cyan') +" to close the ROI.")
    print("3. Press " +colored('enter', 'cyan') +" to save the current ROI.")
    print("4. Use " +colored('left/right', 'cyan') +" arrows to change slices.")
    print("5. Use " +colored('up/down', 'cyan') +" arrows to change frames.")
    print("6. Press " +colored('z', 'cyan') +" to zoom in/out.")
    print("7. Press " +colored('Esc', 'red') +" to close the GUI.")
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    data_4d = nib.load(filename).get_fdata()
    if rotate_AC==True:
        data_4d = np.rot90(data_4d, k=-1, axes=(0, 1))
    selector = ROISelector(data_4d)
    selected_voxels = selector.get_all_selected_voxels()
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    print('['+colored('!', 'cyan')+'] Please identify the selected ' +colored('anatomical', 'red') +' structure:')
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    print(colored('s', 'red')+"   : Sinus Sagittalis Vein")
    print(colored('lica', 'red')+": Left Interior Carotid Artery")
    print(colored('rica', 'red')+": Right Interior Carotid Artery")
    print(colored('b', 'red')+"   : Basilar Artery")
    print(colored('lmca', 'red')+": Left Middle Cerebral Artery")
    print(colored('rmca', 'red')+": Right Middle Cerebral Artery")
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    choice_str =input('['+colored('!', 'cyan')+'] Enter the ' +colored('letter', 'cyan') +' corresponding to your choice: ')
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    choice = choicestr2int(choice_str)
    
    type, subtype = choice2type(choice)    
    [os.remove(f) for f in glob.glob(os.path.join(analysis, 'CTC Data', type, subtype, '*.npy'))]
    [os.remove(f) for f in glob.glob(os.path.join(analysis, 'ITC Data', type, subtype, '*.npy'))]
    for slice_index, roi_voxels in selected_voxels.items():
        print(f"Processing slice {slice_index} with {len(roi_voxels)} voxels.")  # Debugging output
        print(f"roi_voxels sample: {roi_voxels[:5]}")
        plot_time_intensity_curves(data_4d, roi_voxels, slice_index, selector.frame_index, time, analysis, image, type=type, subtype=subtype)
        plot_time_intensity_curves_and_CTC(data_4d, roi_voxels, slice_index, selector.frame_index, time, analysis, image, nifti, type=type, subtype=subtype, IsVFA=IsVFA, filenames=filenames)
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white')) 

    rerun = input('[!] Repeat analysis? (y/n): ')
    print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
    if rerun == 'y':
        selector = ROISelector(data_4d)
        selected_voxels = selector.get_all_selected_voxels()
        print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
        print('['+colored('!', 'cyan')+'] Please identify the selected ' +colored('anatomical', 'red') +' structure:')
        print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
        print(colored('s', 'red')+": Sinus Sagittalis Vein")
        print(colored('lica', 'red')+": Left Interior Carotid Artery")
        print(colored('rica', 'red')+": Right Interior Carotid Artery")
        print(colored('b', 'red')+": Basilar Artery")
        print(colored('lmca', 'red')+": Left Middle Cerebral")
        print(colored('rmca', 'red')+": Right Middle Cerebral")
        print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
        choice_str =input('['+colored('!', 'cyan')+'] Enter the number corresponding to your choice: ')
        choice = choicestr2int(choice_str)
        
        type, subtype = choice2type(choice)
        [os.remove(f) for f in glob.glob(os.path.join(analysis, 'CTC Data', type, subtype, '*.npy'))]
        [os.remove(f) for f in glob.glob(os.path.join(analysis, 'ITC Data', type, subtype, '*.npy'))]
        for slice_index, roi_voxels in selected_voxels.items():
            plot_time_intensity_curves(data_4d, roi_voxels, slice_index, selector.frame_index, time, analysis, image, type=type, subtype=subtype)
            plot_time_intensity_curves_and_CTC(data_4d, roi_voxels, slice_index, selector.frame_index, time, analysis, image, nifti, type=type, subtype=subtype, IsVFA=IsVFA, filenames=filenames)
        print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))  

        rerun2 = input('[!] Repeat analysis? (y/n): ')
        print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
        if rerun2 == 'y':
            selector = ROISelector(data_4d)
            selected_voxels = selector.get_all_selected_voxels()
            print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
            print('['+colored('!', 'cyan')+'] Please identify the selected ' +colored('anatomical', 'red') +' structure:')
            print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
            print(colored('s', 'red')+": Sinus Sagittalis Vein")
            print(colored('lica', 'red')+": Left Interior Carotid Artery")
            print(colored('rica', 'red')+": Right Interior Carotid Artery")
            print(colored('b', 'red')+": Basilar Artery")
            print(colored('lmca', 'red')+": Left Middle Cerebral")
            print(colored('rmca', 'red')+": Right Middle Cerebral")
            print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))
            choice_str =input('['+colored('!', 'cyan')+'] Enter the number corresponding to your choice: ')
            choice = choicestr2int(choice_str)
            
            type, subtype = choice2type(choice)    
            [os.remove(f) for f in glob.glob(os.path.join(analysis, 'CTC Data', type, subtype, '*.npy'))]
            [os.remove(f) for f in glob.glob(os.path.join(analysis, 'ITC Data', type, subtype, '*.npy'))]
            for slice_index, roi_voxels in selected_voxels.items():
                plot_time_intensity_curves(data_4d, roi_voxels, slice_index, selector.frame_index, time, analysis, image, type=type, subtype=subtype)
                plot_time_intensity_curves_and_CTC(data_4d, roi_voxels, slice_index, selector.frame_index, time, analysis, image, nifti, type=type, subtype=subtype, IsVFA=IsVFA, filenames=filenames)
            print(colored('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=', 'white'))  

    

def refresh_nifti_directory(nifti_directory):
    return os.listdir(nifti_directory)

def input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters):
    t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename, \
        flair_3D_filename, axial_flair_3D_filename, axial_t2_2D_filename, dce_filename = filenames
    refresh_nifti_directory(nifti_directory)
    
    IsVFA, IsIR, _, _, _, _, _ = parameters
    filename = os.path.join(nifti_directory, dce_filename)
    nifti_img = nib.load(filename)
    TR = nifti_img.header.get_zooms()[-1] #*1e3
    num_volumes = nifti_img.shape[-1]
    total_scan_duration = TR * num_volumes #*1e-3
    time_points_s = np.linspace(0, total_scan_duration, num_volumes)
    np.save(os.path.join(analysis_directory,'Fitting', 'time_points_s.npy'), time_points_s)
    start_roi_selection(filename, rotate_AC=True, time=time_points_s, analysis=analysis_directory, image=image_directory, nifti=nifti_directory, IsVFA=IsVFA, filenames=filenames)