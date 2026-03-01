#!/usr/bin/env python3
"""Debug TurboFLASH signal->concentration conversion for a single voxel.

Example:
  python tools/ctc_debug_voxel.py \
	--dce "/path/to/dce.nii.gz" \
	--t1  "/path/to/Analysis/Fitting/t1_map_in_dce.nii.gz" \
	--m0  "/path/to/Analysis/Fitting/m0_map_in_dce.nii.gz" \
	--x 120 --y 96 --z 9

Notes:
- Uses the same conversion function as the pipeline: `utils.plotting.turboflash`.
- Prints ratio/inside statistics that explain flat-zero or negative CTC.
"""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np

from utils.loading import resolve_flip_angle_deg, resolve_turboflash_ti_s
from utils.plotting import turboflash


def _p(name: str, v) -> str:
	if v is None:
		return f"{name}=None"
	try:
		if np.isscalar(v):
			return f"{name}={float(v):.6g}"
	except Exception:
		pass
	return f"{name}={v}"


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--dce", required=True, help="Path to DCE 4D NIfTI (.nii/.nii.gz)")
	ap.add_argument("--t1", required=True, help="Path to T1 map aligned to DCE grid (ms)")
	ap.add_argument("--m0", required=True, help="Path to M0 map aligned to DCE grid")
	ap.add_argument("--x", type=int, required=True)
	ap.add_argument("--y", type=int, required=True)
	ap.add_argument("--z", type=int, required=True)
	ap.add_argument(
		"--td-ms",
		type=float,
		default=None,
		help="Override TD (ms). Default: resolve from sidecar or 120ms",
	)
	ap.add_argument(
		"--flip-angle-deg",
		type=float,
		default=None,
		help="Override flip angle (deg). Default: resolve from sidecar",
	)
	ap.add_argument("--baseline-frames", type=int, default=None, help="Override baseline frames (default from settings)")
	args = ap.parse_args()

	dce_img = nib.load(args.dce)
	dce = np.asarray(dce_img.dataobj, dtype=float)
	if dce.ndim != 4:
		raise SystemExit(f"Expected 4D DCE, got shape {dce.shape}")

	t1 = np.asarray(nib.load(args.t1).get_fdata(), dtype=float)
	m0 = np.asarray(nib.load(args.m0).get_fdata(), dtype=float)
	if t1.shape != dce.shape[:3] or m0.shape != dce.shape[:3]:
		raise SystemExit(f"T1/M0 must match DCE spatial shape {dce.shape[:3]}, got T1={t1.shape} M0={m0.shape}")

	x, y, z = int(args.x), int(args.y), int(args.z)
	if not (0 <= x < dce.shape[0] and 0 <= y < dce.shape[1] and 0 <= z < dce.shape[2]):
		raise SystemExit(f"Voxel out of bounds for DCE shape {dce.shape}: ({x},{y},{z})")

	s = np.asarray(dce[x, y, z, :], dtype=float)
	t1_ms = float(t1[x, y, z])
	m0_v = float(m0[x, y, z])

	flip = args.flip_angle_deg
	if flip is None:
		flip = resolve_flip_angle_deg(args.dce, default=None)

	td_ms = args.td_ms
	if td_ms is None:
		td_s = resolve_turboflash_ti_s(args.dce, default=0.12)
		td_ms = float(td_s) * 1e3

	sin_th = float(np.sin(np.radians(float(flip if flip is not None else 30.0))))
	if np.isfinite(m0_v) and m0_v != 0 and np.isfinite(sin_th) and sin_th != 0:
		ratio = s / (m0_v * sin_th)
	else:
		ratio = np.full_like(s, np.nan)
	inside = 1.0 - ratio

	c = turboflash(
		s,
		t1_ms,
		TD=float(td_ms),
		r1=4000,
		m0=m0_v,
		flip_angle_deg=flip,
		baseline_frames=args.baseline_frames,
	)

	print(" ".join([
		_p("flip_deg", flip),
		_p("TD_ms", td_ms),
		_p("T1_ms", t1_ms),
		_p("M0", m0_v),
	]))
	print(f"S[min,max]=({np.nanmin(s):.6g},{np.nanmax(s):.6g})")
	print(f"ratio[min,max]=({np.nanmin(ratio):.6g},{np.nanmax(ratio):.6g})")
	print(f"inside[min,max]=({np.nanmin(inside):.6g},{np.nanmax(inside):.6g})")
	print(f"inside<=0: {int(np.count_nonzero(inside <= 0))}/{inside.size}")
	print(f"CTC[min,max]=({np.nanmin(c):.6g},{np.nanmax(c):.6g})")
	print(f"CTC nonzero: {int(np.count_nonzero(np.asarray(c) != 0))}/{np.size(c)}")

	out = os.environ.get("P_BRAIN_CTC_DEBUG_SAVE")
	if out:
		out = os.path.expanduser(out)
		np.save(out, np.asarray(c, dtype=np.float32))
		print(f"Saved CTC to: {out}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())