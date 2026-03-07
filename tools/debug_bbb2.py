"""Debug part 2: verify M0 vs DCE scale and test recalibration."""
import numpy as np
import pickle
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nibabel as nib

base = '/Users/edt/Desktop/hiroki_testBBB/BBB20220415pts5/Analysis'
nifti_dir = '/Users/edt/Desktop/hiroki_testBBB/BBB20220415pts5/NIfTI'

with open(os.path.join(base, 'Fitting', 'voxel_T1_matrix.pkl'), 'rb') as f:
    T1 = pickle.load(f)
with open(os.path.join(base, 'Fitting', 'voxel_M0_matrix.pkl'), 'rb') as f:
    M0 = pickle.load(f)

dce_path = os.path.join(nifti_dir, 'WIPhperf120long.nii')
dce_img = nib.load(dce_path)
dce_data = np.asarray(dce_img.dataobj, dtype=np.float32)

# Load JSON for flip angle
json_path = dce_path.replace('.nii', '.json')
with open(json_path) as jf:
    js = json.load(jf)
flip_deg = float(js.get('FlipAngle', 30.0))
TR_s = float(js.get('RepetitionTime', 1.22656))

print(f"Flip angle: {flip_deg} deg, TR: {TR_s} s")
alpha_rad = np.radians(flip_deg)
sin_a = np.sin(alpha_rad)
cos_a = np.cos(alpha_rad)

print()
print("=" * 70)
print("1. COMPUTE EXPECTED M0 FROM DCE BASELINE + T1")
print("=" * 70)
print("If S_0 = M0*sin(a)*(1-E1)/(1-cos(a)*E1) where E1=exp(-TR/T1)")
print("Then M0 = S_0 * (1-cos(a)*E1) / (sin(a)*(1-E1))")
print()

for s in range(dce_data.shape[2]):
    mask = np.isfinite(T1[:,:,s]) & (T1[:,:,s] > 100)  # T1 > 100ms
    t1_ms = T1[:,:,s][mask]
    m0_ir = M0[:,:,s][mask]
    s_baseline = np.mean(dce_data[:,:,s,2:10], axis=-1)[mask]
    
    # Expected M0 from DCE baseline
    t1_s = t1_ms / 1000.0
    E1 = np.exp(-TR_s / t1_s)
    m0_expected = s_baseline * (1.0 - cos_a * E1) / (sin_a * (1.0 - E1))
    
    scale = m0_ir / m0_expected
    
    print(f"Slice {s+1}:")
    print(f"  M0_IR median: {np.median(m0_ir):.1f}")
    print(f"  M0_expected (from DCE bl): {np.median(m0_expected):.1f}")
    print(f"  M0_IR / M0_expected: {np.median(scale):.4f}")
    print(f"  => IR M0 is {np.median(scale):.1f}x too large for DCE scale")
    print()

print("=" * 70)
print("2. TEST: RECALIBRATE M0 AND RECOMPUTE CTC")
print("=" * 70)
# Use M0_expected (derived from DCE baseline + T1) instead of M0_IR
# for a representative artery voxel

# Find a high-enhancement voxel on slice 5 (which had best CTC)
for s in [0, 1, 4]:  # slices 1, 2, 5
    mask_3d = np.isfinite(T1[:,:,s]) & (T1[:,:,s] > 100)
    baseline_2d = np.mean(dce_data[:,:,s,2:10], axis=-1)
    peak_2d = np.max(dce_data[:,:,s,25:50], axis=-1)
    enhance = np.where(mask_3d & (baseline_2d > 100), peak_2d / baseline_2d, 0)
    yx = np.unravel_index(np.argmax(enhance), enhance.shape)
    bx, by = yx
    
    sig = dce_data[bx, by, s, :]
    t1_val = T1[bx, by, s]
    m0_ir_val = M0[bx, by, s]
    
    # Recalibrate M0
    t1_s_val = t1_val / 1000.0
    E1_val = np.exp(-TR_s / t1_s_val)
    s_bl = float(np.mean(sig[2:10]))
    m0_recal = s_bl * (1.0 - cos_a * E1_val) / (sin_a * (1.0 - E1_val))
    
    # CTC with original M0
    TD_s = 0.120  # 120ms
    r1_s = 4.0
    r1_pre = 1.0 / t1_s_val
    
    def compute_ctc(signal, m0, t1_s):
        ratio = signal / (m0 * sin_a)
        inside = 1.0 - ratio
        valid = inside > 0
        c = np.full_like(signal, np.nan)
        r1_pre = 1.0 / t1_s
        c[valid] = (-1.0 / (r1_s * TD_s)) * (np.log(inside[valid]) + TD_s * r1_pre)
        bl = np.nanmean(c[2:10])
        if not np.isfinite(bl):
            bl = 0.0
        c = c - bl
        c[:10] = 0
        return np.nan_to_num(c, nan=0.0)
    
    ctc_orig = compute_ctc(sig, m0_ir_val, t1_s_val)
    ctc_recal = compute_ctc(sig, m0_recal, t1_s_val)
    
    print(f"Slice {s+1} voxel ({bx},{by}) enhancement={enhance[bx,by]:.2f}x:")
    print(f"  T1={t1_val:.1f}ms  M0_IR={m0_ir_val:.1f}  M0_recal={m0_recal:.1f}  ratio={m0_ir_val/m0_recal:.2f}")
    print(f"  CTC with M0_IR:    peak={np.max(ctc_orig[10:100]):.4f} mM")
    print(f"  CTC with M0_recal: peak={np.max(ctc_recal[10:100]):.4f} mM")
    print()

print("=" * 70)
print("3. CHECK: Does the pipeline use M0 from IR or does it recalibrate?")
print("=" * 70)
# Check what the _turboflash_raw_concentration formula actually does
# c = (-1/(r1*TD)) * (ln(1 - S/(M0*sin(a))) + TD/T1)
# For pre-contrast: S_0/(M0*sin) should ideally give a ratio < 1
# where ln(1 - S_0/(M0*sin)) + TD/T1 ≈ 0 (baseline)
# If M0 is too large: S/(M0*sin) << 1, log(1-x) ≈ -x for small x
# So c ≈ (-1/(r1*TD))*(-S/(M0*sin) + TD/T1)
# The constant TD/T1 term dominates and after baseline subtraction 
# only the small -S/(M0*sin) variation remains -> tiny CTC values

# With correct M0: S/(M0*sin) is closer to the SPGR steady-state value
# and the log nonlinearity properly converts signal changes to concentration

mask_s5 = np.isfinite(T1[:,:,4]) & (T1[:,:,4] > 100)
s_bl_all = np.mean(dce_data[:,:,4,2:10], axis=-1)[mask_s5]
m0_all = M0[:,:,4][mask_s5]
ratio_all = s_bl_all / (m0_all * sin_a)
print(f"Slice 5: S_baseline/(M0*sin) median = {np.median(ratio_all):.4f}")
print(f"  This should be close to (1-E1)/(1-cos(a)*E1) = "
      f"{(1-np.exp(-TR_s/1.308))/(1-cos_a*np.exp(-TR_s/1.308)):.4f}")
print(f"  for T1=1308ms, TR={TR_s}s, alpha={flip_deg}deg")
print()

# What if we just divide M0 by the scale mismatch factor?
print("=" * 70)
print("4. SIMPLER FIX: Estimate scale factor from DCE baseline vs M0")
print("=" * 70)
for s in range(dce_data.shape[2]):
    mask = np.isfinite(T1[:,:,s]) & (T1[:,:,s] > 100)
    t1_s = T1[:,:,s][mask] / 1000.0
    m0_ir = M0[:,:,s][mask]
    s_bl = np.mean(dce_data[:,:,s,2:10], axis=-1)[mask]
    
    # Expected signal from SPGR equation: S = M0*sin(a)*(1-E1)/(1-cos(a)*E1)
    E1 = np.exp(-TR_s / t1_s)
    s_expected = m0_ir * sin_a * (1 - E1) / (1 - cos_a * E1)
    
    # Scale factor = actual DCE signal / expected signal from IR M0
    scale = s_bl / s_expected
    print(f"Slice {s+1}: scale_factor median = {np.median(scale):.4f}, "
          f"mean = {np.mean(scale):.4f}, std = {np.std(scale):.4f}")
    print(f"  => multiply M0 by {np.median(scale):.4f} to match DCE scale")
    print(f"  OR equivalently, M0_corrected = M0 * {np.median(scale):.4f}")
