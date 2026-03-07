"""Check if Hemisure data has the same TI/DCE slope mismatch."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import nibabel as nib
except ImportError:
    print("nibabel not available")
    sys.exit(1)

hemi_dir = '/Volumes/T5_EVO_EDT/data'
for d in sorted(os.listdir(hemi_dir)):
    nifti_d = os.path.join(hemi_dir, d, 'NIfTI')
    if not os.path.isdir(nifti_d):
        continue
    nii_files = os.listdir(nifti_d)
    ti_files = sorted([f for f in nii_files if 'TI_' in f and f.endswith('.nii') and 'real' not in f and 'imag' not in f and not f.startswith('._')])
    dce_files = [f for f in nii_files if ('perf' in f.lower() or 'dce' in f.lower() or 'hperf' in f.lower()) and f.endswith('.nii') and 'real' not in f and 'imag' not in f and not f.startswith('._')]
    if ti_files and dce_files:
        ti0 = nib.load(os.path.join(nifti_d, ti_files[0]))
        ti_last = nib.load(os.path.join(nifti_d, ti_files[-1]))
        dce = nib.load(os.path.join(nifti_d, dce_files[0]))
        ti0_slope = getattr(ti0.dataobj, 'slope', None)
        ti_last_slope = getattr(ti_last.dataobj, 'slope', None)
        dce_slope = getattr(dce.dataobj, 'slope', None)
        print(f"{d}: TI[0] slope={ti0_slope}, TI[-1] slope={ti_last_slope}, DCE slope={dce_slope}, "
              f"DCE/TI[0]={float(dce_slope)/float(ti0_slope):.4f}")
        break  # just check one dataset
