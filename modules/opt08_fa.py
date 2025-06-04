import os
import re
import numpy as np
import nibabel as nib
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table


def find_dwi_files(nifti_directory):
    for root, _, files in os.walk(nifti_directory):
        for file in files:
            if re.search(r'dwi', file, re.IGNORECASE) and file.endswith('.nii'):
                base = os.path.splitext(os.path.join(root, file))[0]
                bval = base + '.bval'
                bvec = base + '.bvec'
                if os.path.exists(bval) and os.path.exists(bvec):
                    return os.path.join(root, file), bval, bvec
    return None, None, None


def compute_fa(nifti_directory, analysis_directory):
    dwi_path, bval_path, bvec_path = find_dwi_files(nifti_directory)
    if dwi_path is None:
        print('[!] No DWI data found; skipping FA computation')
        return

    print(f'[!] Computing FA from {os.path.basename(dwi_path)}')
    img = nib.load(dwi_path)
    data = img.get_fdata()

    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    gtab = gradient_table(bvals, bvecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)

    fa = tenfit.fa.astype(np.float32)
    fa_img = nib.Nifti1Image(fa, img.affine, img.header)
    out_path = os.path.join(analysis_directory, 'FA_map.nii.gz')
    nib.save(fa_img, out_path)
    print(f'[!] Saved FA map to {out_path}')

    mean_fa = np.nanmean(fa)
    with open(os.path.join(analysis_directory, 'fa_mean.txt'), 'w') as f:
        f.write(f'{mean_fa}\n')
    print(f'[!] Mean FA: {mean_fa:.4f}')
