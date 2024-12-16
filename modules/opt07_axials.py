import os
import re
import subprocess
import time
import nibabel as nib

def decompress_nii_gz(nii_gz_path):
    if not os.path.exists(nii_gz_path + '.gz'):
        print(f"File not found: {nii_gz_path + '.gz'}")
        return None

    try:
        img = nib.load(nii_gz_path + '.gz')
        nii_path = nii_gz_path
        nib.save(img, nii_path)
        print(f"Decompressed to: {nii_path}")
        os.remove(os.path.join(nii_gz_path+'.gz'))
        return nii_path
    
    except Exception as e:
        print(f"Error decompressing file: {e}")
        return None
    
class ImageProcessor:
    def __init__(self, nifti_directory):
        self.nifti_directory = nifti_directory

    def find_matching_file(self, pattern):
        for root, _, files in os.walk(self.nifti_directory):
            for file in files:
                if re.fullmatch(pattern, file, re.IGNORECASE):
                    return os.path.join(root, file)
        return None

def file_exists(filename):
    return os.path.exists(filename)

def execute_command(command):
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}\n{e}")

def check_axial(nifti_directory, filenames):
    processor = ImageProcessor(nifti_directory)
    
    t1_3D_filename, axial_t1_3D_filename, t2_3D_filename, axial_t2_3D_filename, \
    flair_3D_filename, axial_flair_3D_filename_regex, axial_t2_2D_filename, dce_filename = filenames

    # Explicit filenames for output
    axial_t1_3D_output = "axVWIPcs_T1W_3D_TFE_32channel.nii"
    axial_t2_3D_output = "axVWIPcs_3D_Brain_VIEW_T2_32chSHC.nii"

    axial_flair_3D_filename = processor.find_matching_file(axial_flair_3D_filename_regex)
    flair_full_path = os.path.join(nifti_directory, flair_3D_filename)
    t2_full_path = os.path.join(nifti_directory, t2_3D_filename)
    t1_full_path = os.path.join(nifti_directory, t1_3D_filename)

    # Check for axially reconstructed T1 and T2 files
    if file_exists(os.path.join(nifti_directory, axial_t1_3D_output)) and \
       file_exists(os.path.join(nifti_directory, axial_t2_3D_output)):
<<<<<<< HEAD
        #print("Axially reconstructed T2 and T1 files already exist.")
        #time.sleep(3)
=======
        print("Axially reconstructed T2 and T1 files already exist.")
        time.sleep(3)
>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21
        return

    if not flair_full_path or not axial_flair_3D_filename:
        print("FLAIR or axial FLAIR reconstruction not found.")
        time.sleep(3)
        return

    print("[!] Generating T1 & T2 axial reconstructions")
    axial_t1_output_path = os.path.join(nifti_directory, axial_t1_3D_output)
    axial_t2_output_path = os.path.join(nifti_directory, axial_t2_3D_output)

    execute_command(f"flirt -in {t1_full_path} -ref {axial_flair_3D_filename} -applyxfm -usesqform -out {axial_t1_output_path}")
    execute_command(f"flirt -in {t2_full_path} -ref {axial_flair_3D_filename} -applyxfm -usesqform -out {axial_t2_output_path}")
    
    decompress_nii_gz(axial_t1_output_path)
    decompress_nii_gz(axial_t2_output_path)
    print("[!] Finished.")
<<<<<<< HEAD
    #time.sleep(2)
=======
    time.sleep(2)
>>>>>>> a0de673fc033368a127dc6bae55e4b3363958e21
