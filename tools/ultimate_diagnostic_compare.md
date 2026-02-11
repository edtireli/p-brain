# Ultimate diagnostic comparison (p-brain vs MATLAB)

Single-script, repeatable artifact for comparing p-brain outputs against MATLAB for a specific subject.

## What it generates

- One PNG grid (default: slice 5) containing:
  - R1 (MATLAB `r1_map`) vs p-brain-derived R1 (`1/T1` or `1000/T1` heuristic) + absolute difference
  - M0 (MATLAB `m0_map`) vs p-brain M0 + absolute difference
  - Input concentration curve plot:
    - p-brain TSCC (SSS-shifted from RICA) vs MATLAB LICA slice2 scaled (`c_input`)
  - Dedicated perfusion comparison rows (each: p-brain | MATLAB | |diff|):
    - CBF (ml/100g/min): MATLAB `CBF_Tik` is converted from 1/s via `*6000`
    - CBV (ml/100g): MATLAB `Vd` is converted via `*100` (default)
    - MTT (s): compared directly
  - Patlak maps:
    - p-brain Ki scaled by `*6000` (1/s → ml/100g/min)
    - p-brain vp scaled by `*100` (fraction → %)

## Orientation rule

- MATLAB-exported *maps* are rotated **90° clockwise** before displaying/diffing to match p-brain/NIfTI orientation.
- Curves are not rotated.

## Example (20240321x2_flot)

```bash
cd /Users/edt/Desktop/p-brain
python3 tools/ultimate_diagnostic_compare.py \
  --out "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/ultimate_diagnostic_slice5.png" \
  --slice 5 \
  --mat-t1m0 "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/T1_M0_plusError_maps_.mat" \
  --pbrain-t1 "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/Fitting/voxel_T1_matrix.pkl" \
  --pbrain-m0 "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/Fitting/voxel_M0_matrix.pkl" \
  --mat-curve "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/conc_methodT1_map_input_MRsignal_M_fix_slice2_LICA_slice2_scaled.mat" \
  --pbrain-tscc-root "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/TSCC Data" \
  --pbrain-time "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/Fitting/time_points_s.npy" \
  --mat-tik "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/CBF_maps_tik_PaReLLeL_offset200_3DGauss0mm_frames250_slice1-10_MR contrast agent_Lambda_values_AUTO_TCBF_ 22_ml_mg_min_.mat" \
  --pbrain-cbf "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/CBF_per_voxel_tikhonov.nii.gz" \
  --pbrain-mtt "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/mtt_map.nii.gz" \
  --pbrain-ki "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/Ki_per_voxel_patlak.nii.gz" \
  --pbrain-vp "/Volumes/T5_EVO_EDT/hemisure/20240321x2_flot/Analysis/vp_per_voxel_patlak.nii.gz" \
  --scale-cbf 1
```

## Notes

- If you have a dedicated p-brain CBV NIfTI, pass `--pbrain-cbv <path>`. Otherwise CBV is derived as `CBF*(MTT/60)`.
- If p-brain CBF is later confirmed to be in 1/s, re-run with `--scale-cbf 6000`.
- MATLAB map keys/scales can be overridden via `--mat-key-*` / `--mat-scale-*`.
