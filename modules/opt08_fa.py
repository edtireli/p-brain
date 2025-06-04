import os
import re
import numpy as np
import nibabel as nib
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table


def find_wm_mask(nifti_directory):
    """Return path to a white matter mask within ``nifti_directory`` if found."""
    for root, _, files in os.walk(nifti_directory):
        for file in files:
            if file.lower() == "wm.nii" or file.lower() == "wm.nii.gz":
                return os.path.join(root, file)
    return None


def find_dwi_files(nifti_directory):
    """Locate a DWI NIfTI file and its corresponding ``.bval`` and ``.bvec`` files.

    The conversion step may create ``.nii`` *or* ``.nii.gz`` files.  This
    function therefore checks for both extensions and strips them correctly when
    forming the paths to the gradient files.
    """

    for root, _, files in os.walk(nifti_directory):
        for file in files:
            if re.search(r"dwi", file, re.IGNORECASE) and (
                file.endswith(".nii") or file.endswith(".nii.gz")
            ):
                # Handle both .nii and .nii.gz extensions when deriving the base
                base = os.path.splitext(os.path.join(root, file))[0]
                if file.endswith(".nii.gz"):
                    base = os.path.splitext(base)[0]

                bval = base + ".bval"
                bvec = base + ".bvec"

                if os.path.exists(bval) and os.path.exists(bvec):
                    return os.path.join(root, file), bval, bvec

    return None, None, None


def compute_fa(nifti_directory, analysis_directory):
    dwi_path, bval_path, bvec_path = find_dwi_files(nifti_directory)
    if dwi_path is None:
        print("[!] No DWI data found; skipping FA computation")
        return

    print(f"[!] Computing FA from {os.path.basename(dwi_path)}")
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
    out_path = os.path.join(analysis_directory, "FA_map.nii.gz")
    nib.save(fa_img, out_path)
    print(f"[!] Saved FA map to {out_path}")

    mean_fa = np.nanmean(fa)
    with open(os.path.join(analysis_directory, "fa_mean.txt"), "w") as f:
        f.write(f"{mean_fa}\n")
    print(f"[!] Mean FA: {mean_fa:.4f}")

    wm_mask_path = find_wm_mask(nifti_directory)
    if wm_mask_path:
        wm_mask = nib.load(wm_mask_path).get_fdata() > 0
        mean_fa_wm = np.nanmean(fa[wm_mask])
        with open(os.path.join(analysis_directory, "fa_mean_wm.txt"), "w") as f:
            f.write(f"{mean_fa_wm}\n")
        print(f"[!] Mean WM FA: {mean_fa_wm:.4f}")
        fa_wm = fa * wm_mask
        fa_wm_img = nib.Nifti1Image(fa_wm.astype(np.float32), img.affine, img.header)
        wm_out_path = os.path.join(analysis_directory, "FA_WM_map.nii.gz")
        nib.save(fa_wm_img, wm_out_path)
        print(f"[!] Saved WM FA map to {wm_out_path}")
    else:
        print("[!] WM mask not found; skipping WM-specific FA computation")
