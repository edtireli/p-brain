"""Debug script to investigate BBB CTC issues."""
import numpy as np
import pickle
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nibabel as nib

base = '/Users/edt/Desktop/hiroki_testBBB/BBB20220415pts5/Analysis'
nifti_dir = '/Users/edt/Desktop/hiroki_testBBB/BBB20220415pts5/NIfTI'

# Load T1 / M0
with open(os.path.join(base, 'Fitting', 'voxel_T1_matrix.pkl'), 'rb') as f:
    T1 = pickle.load(f)
with open(os.path.join(base, 'Fitting', 'voxel_M0_matrix.pkl'), 'rb') as f:
    M0 = pickle.load(f)

# Load DCE
dce_path = os.path.join(nifti_dir, 'WIPhperf120long.nii')
dce_img = nib.load(dce_path)
dce_data = np.asarray(dce_img.dataobj, dtype=np.float32)

print("=" * 70)
print("1. BASIC INFO")
print("=" * 70)
print(f"T1 shape={T1.shape}, M0 shape={M0.shape}")
print(f"DCE shape={dce_data.shape}, slope={getattr(dce_img.dataobj, 'slope', 'N/A')}")
for s in range(T1.shape[2]):
    vt = T1[:,:,s]; vm = M0[:,:,s]
    ok = np.isfinite(vt) & (vt > 0)
    print(f"  Slice {s+1}: T1 median={np.median(vt[ok]):.1f}ms, "
          f"M0 median={np.median(vm[ok]):.1f}, valid={ok.sum()}")

# JSON sidecar
json_path = dce_path.replace('.nii', '.json')
js = {}
if os.path.exists(json_path):
    with open(json_path) as jf:
        js = json.load(jf)
    print(f"\nJSON sidecar found. Key fields:")
    for k in ['FlipAngle', 'RepetitionTime', 'EchoTime',
              'MagneticFieldStrength', 'Manufacturer', 'PulseSequenceType']:
        if k in js:
            print(f"  {k}: {js[k]}")

print()
print("=" * 70)
print("2. M0 vs DCE BASELINE SCALE MISMATCH")
print("=" * 70)
print("The turboflash formula computes S / (M0 * sin(alpha)).")
print("If M0 and S are on different scales, this ratio will be wrong.")
print()

flip_deg = float(js.get('FlipAngle', 30.0))
sin_a = float(np.sin(np.radians(flip_deg)))
print(f"FlipAngle = {flip_deg} deg, sin(alpha) = {sin_a:.6f}")
print()

for s in range(dce_data.shape[2]):
    mask = np.isfinite(T1[:,:,s]) & (T1[:,:,s] > 0)
    dce_bl = np.mean(dce_data[:,:,s,2:10], axis=-1)  # baseline DCE signal
    dce_pk = np.max(dce_data[:,:,s,25:50], axis=-1)   # peak DCE signal

    m0_vals = M0[:,:,s][mask]
    dce_bl_vals = dce_bl[mask]
    dce_pk_vals = dce_pk[mask]

    ratio_bl = dce_bl_vals / (m0_vals * sin_a)
    ratio_pk = dce_pk_vals / (m0_vals * sin_a)

    n_bad_bl = int(np.sum(ratio_bl >= 1.0))
    n_bad_pk = int(np.sum(ratio_pk >= 1.0))

    print(f"Slice {s+1}:")
    print(f"  M0 median: {np.median(m0_vals):.1f}")
    print(f"  DCE_baseline median: {np.median(dce_bl_vals):.1f}")
    print(f"  DCE_peak median: {np.median(dce_pk_vals):.1f}")
    print(f"  M0/DCE_bl scale ratio: {np.median(m0_vals) / np.median(dce_bl_vals):.4f}")
    print(f"  S_bl/(M0*sin) median: {np.median(ratio_bl):.4f}  "
          f"(>= 1.0: {n_bad_bl}/{mask.sum()} = {100*n_bad_bl/mask.sum():.1f}%)")
    print(f"  S_pk/(M0*sin) median: {np.median(ratio_pk):.4f}  "
          f"(>= 1.0: {n_bad_pk}/{mask.sum()} = {100*n_bad_pk/mask.sum():.1f}%)")
    print()

print("=" * 70)
print("3. TI SERIES SCALING vs DCE SCALING")
print("=" * 70)
print("If TI NIfTIs have different proxy slopes than DCE, M0 will be on wrong scale.")
print()

ti_files = sorted([f for f in os.listdir(nifti_dir)
                    if f.startswith('WIPTI_') and f.endswith('.nii')
                    and '_real' not in f and '_imaginary' not in f])
for tf in ti_files:
    ti_img = nib.load(os.path.join(nifti_dir, tf))
    ti_slope = getattr(ti_img.dataobj, 'slope', None)
    ti_data = np.asarray(ti_img.dataobj, dtype=np.float32)
    print(f"  {tf}: slope={ti_slope:.6f}, shape={ti_data.shape}, "
          f"center_mean={np.mean(ti_data[80:180,80:180,2]):.1f}")

dce_slope = getattr(dce_img.dataobj, 'slope', None)
print(f"\n  DCE: slope={dce_slope:.6f}, baseline_center_mean={np.mean(dce_data[80:180,80:180,2,2:10]):.1f}")

# Check if there's a mismatch
if ti_files:
    ti0_img = nib.load(os.path.join(nifti_dir, ti_files[0]))
    ti0_slope = float(getattr(ti0_img.dataobj, 'slope', 1.0))
    dce_slope_f = float(dce_slope) if dce_slope else 1.0
    scale_ratio = dce_slope_f / ti0_slope
    print(f"\n  *** DCE_slope / TI_slope = {scale_ratio:.4f} ***")
    if abs(scale_ratio - 1.0) > 0.01:
        print(f"  >>> MISMATCH! M0 was fit from TI data at scale {ti0_slope:.6f},")
        print(f"      but DCE signal is at scale {dce_slope_f:.6f}.")
        print(f"      The ratio S/(M0*sin) will be off by factor {scale_ratio:.4f}.")
        print(f"      THIS IS LIKELY THE BUG.")
