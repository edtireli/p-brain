#!/usr/bin/env python3
"""Survey: which segmentation labels are CNN ROIs near?"""
import numpy as np
import nibabel as nib
import os
import sys
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_deterministic_vs_cnn import load_cnn_rois, find_dce_nifti

data_dir = '/Volumes/T5_EVO_EDT/data'
datasets = sorted([d for d in os.listdir(data_dir) if d.startswith('20')])

LABEL_NAMES = {
    0: 'Background', 2: 'L-Cerebral-WM', 3: 'L-Cerebral-Cortex', 4: 'L-Lat-Ventricle',
    5: 'L-Inf-Lat-Ventricle', 7: 'L-Cerebellum-WM', 8: 'L-Cerebellum-Cortex',
    10: 'L-Thalamus', 11: 'L-Caudate', 12: 'L-Putamen', 13: 'L-Pallidum',
    14: '3rd-Ventricle', 15: '4th-Ventricle', 16: 'Brain-Stem',
    17: 'L-Hippocampus', 18: 'L-Amygdala', 24: 'CSF',
    26: 'L-Accumbens', 28: 'L-VentralDC',
    41: 'R-Cerebral-WM', 42: 'R-Cerebral-Cortex', 43: 'R-Lat-Ventricle',
    44: 'R-Inf-Lat-Ventricle', 46: 'R-Cerebellum-WM', 47: 'R-Cerebellum-Cortex',
    49: 'R-Thalamus', 50: 'R-Caudate', 51: 'R-Putamen', 52: 'R-Pallidum',
    53: 'R-Hippocampus', 54: 'R-Amygdala', 58: 'R-Accumbens', 60: 'R-VentralDC',
}
# Add DKT cortical labels
for i in range(1000, 1036):
    LABEL_NAMES[i] = f'ctx-L-{i-1000}'
for i in range(2000, 2036):
    LABEL_NAMES[i] = f'ctx-R-{i-2000}'

art_label_on = []
vein_label_on = []
art_rel_pos = []
vein_rel_pos = []
# Per-dataset artery centroids: (ds, z, dist_to_nearest_label, nearest_label)
art_centroid_info = []
vein_centroid_info = []
n_proc = 0

for ds_name in datasets:
    ds = os.path.join(data_dir, ds_name)
    nifti_dir = os.path.join(ds, 'NIfTI')
    analysis_dir = os.path.join(ds, 'Analysis')
    if not os.path.isdir(nifti_dir) or not os.path.isdir(analysis_dir):
        continue
    seg_path = os.path.join(nifti_dir, 'segmentation/segmentation/mri/aparc.DKTatlas+aseg.deep_in_DCE.nii.gz')
    if not os.path.exists(seg_path):
        continue
    cnn_rois = load_cnn_rois(analysis_dir)
    if not cnn_rois:
        continue
    seg = np.asarray(nib.load(seg_path).dataobj)

    for (vtype, sub, z1), vox in cnn_rois.items():
        z0 = z1 - 1
        if z0 < 0 or z0 >= seg.shape[2]:
            continue
        seg_slice = seg[:, :, z0]
        brain_slice = seg_slice > 0
        if not brain_slice.any():
            continue
        br, bc = np.where(brain_slice)
        centroid_r, centroid_c = br.mean(), bc.mean()

        # ROI centroid
        roi_r, roi_c = vox[:, 0].mean(), vox[:, 1].mean()

        # Find nearest segmentation label to ROI centroid
        # Build distance from ROI centroid and check labels at nearby voxels
        nearby_labels = {}
        for vi in range(min(vox.shape[0], 200)):
            r, c = int(vox[vi, 0]), int(vox[vi, 1])
            if r < 0 or r >= seg.shape[0] or c < 0 or c >= seg.shape[1]:
                continue
            label_at = int(seg_slice[r, c])
            dr = r - centroid_r
            dc = c - centroid_c
            if vtype == 'artery':
                art_label_on.append(label_at)
                art_rel_pos.append((dr, dc))
            else:
                vein_label_on.append(label_at)
                vein_rel_pos.append((dr, dc))

        # Distance from ROI centroid to nearest voxel of each label
        # Check a 40px neighborhood
        r0 = max(0, int(roi_r) - 40)
        r1 = min(seg.shape[0], int(roi_r) + 40)
        c0 = max(0, int(roi_c) - 40)
        c1 = min(seg.shape[1], int(roi_c) + 40)
        patch = seg_slice[r0:r1, c0:c1]
        for lab in np.unique(patch):
            if lab == 0:
                continue
            lab_vox = np.argwhere(patch == lab)
            lab_vox_abs = lab_vox + np.array([r0, c0])
            dists = np.sqrt((lab_vox_abs[:, 0] - roi_r)**2 + (lab_vox_abs[:, 1] - roi_c)**2)
            min_dist = float(dists.min())
            if vtype == 'artery':
                art_centroid_info.append((ds_name, z1, int(lab), min_dist))
            else:
                vein_centroid_info.append((ds_name, z1, int(lab), min_dist))

    n_proc += 1
    if n_proc % 20 == 0:
        print(f'  {n_proc} datasets...')

print(f'\nProcessed {n_proc} datasets')
print(f'Artery voxels: {len(art_label_on)}, Vein voxels: {len(vein_label_on)}')

print('\n=== ARTERY: labels voxels sit ON ===')
uniq, counts = np.unique(art_label_on, return_counts=True)
for idx in np.argsort(-counts)[:20]:
    lab, cnt = int(uniq[idx]), int(counts[idx])
    name = LABEL_NAMES.get(lab, f'Label-{lab}')
    pct = 100 * cnt / len(art_label_on)
    print(f'  {name} ({lab}): {cnt} ({pct:.1f}%)')

print('\n=== VEIN: labels voxels sit ON ===')
uniq, counts = np.unique(vein_label_on, return_counts=True)
for idx in np.argsort(-counts)[:20]:
    lab, cnt = int(uniq[idx]), int(counts[idx])
    name = LABEL_NAMES.get(lab, f'Label-{lab}')
    pct = 100 * cnt / len(vein_label_on)
    print(f'  {name} ({lab}): {cnt} ({pct:.1f}%)')

art_pos = np.array(art_rel_pos)
vein_pos = np.array(vein_rel_pos)
print('\n=== ARTERY position relative to brain centroid ===')
print(f'  dr: {art_pos[:,0].mean():.1f} +/- {art_pos[:,0].std():.1f} (range {art_pos[:,0].min():.0f} to {art_pos[:,0].max():.0f})')
print(f'  dc: {art_pos[:,1].mean():.1f} +/- {art_pos[:,1].std():.1f} (range {art_pos[:,1].min():.0f} to {art_pos[:,1].max():.0f})')
print('\n=== VEIN position relative to brain centroid ===')
print(f'  dr: {vein_pos[:,0].mean():.1f} +/- {vein_pos[:,0].std():.1f} (range {vein_pos[:,0].min():.0f} to {vein_pos[:,0].max():.0f})')
print(f'  dc: {vein_pos[:,1].mean():.1f} +/- {vein_pos[:,1].std():.1f} (range {vein_pos[:,1].min():.0f} to {vein_pos[:,1].max():.0f})')

# Distance from nearest key labels to artery centroids
print('\n=== ARTERY: mean distance to nearest key structures ===')
ci = np.array([(lab, dist) for _, _, lab, dist in art_centroid_info])
key_lab_ids = [10, 49, 12, 51, 11, 50, 13, 52, 16, 41, 2, 42, 3]
for kl in key_lab_ids:
    mask = ci[:, 0] == kl
    if mask.sum() > 0:
        d = ci[mask, 1]
        name = LABEL_NAMES.get(kl, f'Label-{kl}')
        print(f'  {name} ({kl}): mean={d.mean():.1f}px, median={np.median(d):.1f}px, present in {mask.sum()} roi-slices')

print('\n=== VEIN: mean distance to nearest key structures ===')
ci_v = np.array([(lab, dist) for _, _, lab, dist in vein_centroid_info])
for kl in key_lab_ids + [4, 43, 24]:
    mask = ci_v[:, 0] == kl
    if mask.sum() > 0:
        d = ci_v[mask, 1]
        name = LABEL_NAMES.get(kl, f'Label-{kl}')
        print(f'  {name} ({kl}): mean={d.mean():.1f}px, median={np.median(d):.1f}px, present in {mask.sum()} roi-slices')
