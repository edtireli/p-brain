"""``python -m pbrain.demo`` — synthesise a phantom, run the pipeline, montage outputs.

Builds a small 64×64×10 phantom with four nested geometric tissue regions
(CSF / GM-like / WM-like / vessel). Each region has known T1, M0, Ki, vp,
CBF, MTT, CTH. An analytical gamma-variate AIF is convolved per-region
with the chosen kinetic to produce ground-truth Cₜ(t), then converted
back to MRI signal via the inversion-recovery → SPGR forward model that
the pipeline inverts. Inputs are written under ``demo/phantom/raw/`` and
the full pipeline runs into ``demo/phantom/derivatives/``. Final step:
Adaptive montages of every produced parameter map land in ``demo/maps/``.

The script is self-contained and only depends on the package's runtime
deps (numpy, scipy, matplotlib, nibabel).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

DEMO_ROOT = Path(__file__).resolve().parents[2] / "demo"


# ───────────────────────────── phantom ────────────────────────────────


def _gamma_variate(t: np.ndarray, A: float, alpha: float, t0: float, beta: float) -> np.ndarray:
    """A·(t−t0)^α exp(−(t−t0)/β) for t > t0, else 0."""
    out = np.zeros_like(t, dtype=float)
    tau = t - t0
    pos = tau > 0
    out[pos] = A * (tau[pos] ** alpha) * np.exp(-tau[pos] / beta)
    return out


def _make_ca(t: np.ndarray) -> np.ndarray:
    """Analytical AIF in mM: main bolus + recirculation hump + slow tail."""
    main = _gamma_variate(t, A=2.0, alpha=3.0, t0=10.0, beta=2.5)
    recirc = 0.30 * _gamma_variate(t, A=2.0, alpha=4.0, t0=22.0, beta=4.5)
    tail = 0.18 * np.exp(-(t - 30.0) / 180.0) * (t > 30.0)
    return main + recirc + tail


def _convolve(ca: np.ndarray, kernel: np.ndarray, dt: float) -> np.ndarray:
    """Trapezoidal convolution Cₐ ⊛ kernel."""
    full = np.convolve(ca, kernel, mode="full")[: ca.size] * dt
    return full


def _patlak_ct(ca: np.ndarray, t: np.ndarray, *, ki_per_min: float, vp: float) -> np.ndarray:
    """Cₜ(t) = vp·Cₐ(t) + (Ki/60) ∫₀ᵗ Cₐ(τ)dτ."""
    dt = float(t[1] - t[0])
    integ = np.cumsum(ca) * dt
    return vp * ca + (ki_per_min / 60.0) * integ


def _make_phantom(dim_xy: int = 64, dim_z: int = 10) -> tuple[np.ndarray, dict]:
    """Build a label volume with concentric ellipsoidal regions."""
    yy, xx, zz = np.mgrid[0:dim_xy, 0:dim_xy, 0:dim_z]
    cx, cy, cz = dim_xy / 2, dim_xy / 2, dim_z / 2
    rx = (xx - cx) / (0.45 * dim_xy)
    ry = (yy - cy) / (0.45 * dim_xy)
    rz = (zz - cz) / (0.45 * dim_z)
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    # Use FreeSurfer-compatible labels so the `preloaded` tissue_roi plug-in
    # picks them up under its default region_map.
    labels = np.zeros_like(r, dtype=np.int32)
    labels[r <= 1.00] = 1001     # outer ring → cortical GM (DKT)
    labels[r <= 0.65] =    2     # mid ring   → WM (FreeSurfer Left-Cerebral-WM)
    labels[r <= 0.30] =   11     # inner      → subcortical GM (Left-Caudate)
    # Vessel: small bright off-centre sphere. Kept OUTSIDE the parcellation
    # (label 0 here, separate mask file) so it doesn't pollute the tissue
    # aggregation; the AIF plug-in finds it via the separate vessel mask.
    rv = np.sqrt((xx - cx + 0.18 * dim_xy) ** 2 +
                 (yy - cy + 0.10 * dim_xy) ** 2 +
                 (zz - cz) ** 2)
    vessel = rv <= 2.5
    return labels, vessel, {
        "1001": "cortical_GM",
        "2":    "WM",
        "11":   "subcortical_GM",
        "vessel_mask": "outside parcellation, used by from_file AIF",
    }


# Tissue ground truth (per label).  Vessel uses a separate boolean mask
# (not a parcellation label) so the AIF plug-in can find it independently.
_TISSUE_TRUTH = {
    # label : (T1_ms, M0_arb, ki, vp,  cbf,  mtt, cth)
    1001: (1330.0, 100.0,  0.5, 0.045, 55.0, 4.5, 3.0),   # cortical GM
       2: ( 850.0,  80.0,  0.1, 0.020, 22.0, 6.5, 4.5),   # WM
      11: (1300.0, 100.0,  0.7, 0.040, 50.0, 5.0, 3.5),   # subcortical GM
}
_VESSEL_TRUTH = (1500.0, 120.0, 0.0, 0.900, 0.0, 0.0, 0.0)


def _make_signal_volume(
    labels: np.ndarray, vessel: np.ndarray, t_s: np.ndarray, *,
    flip_angle_deg: float = 8.0, tr_s: float = 0.01118,
    r1_per_s_mM: float = 4.0,
    noise_pct: float = 1.0, baseline_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Forward model: produce Cₜ(t) per tissue, convert to SPGR signal."""
    X, Y, Z = labels.shape
    T = int(t_s.size)
    ca = _make_ca(t_s)

    sig = np.zeros((X, Y, Z, T), dtype=np.float32)
    rng = np.random.default_rng(seed=7)

    # Per-tissue forward: parcellation labels + the separate vessel mask.
    region_specs: list[tuple[np.ndarray, tuple]] = []
    for lbl, spec in _TISSUE_TRUTH.items():
        region_specs.append((labels == lbl, spec))
    region_specs.append((vessel, _VESSEL_TRUTH))

    for mask, (T1_ms, M0, ki, vp, cbf, mtt, cth) in region_specs:
        if not mask.any():
            continue
        # Cₜ(t) under Patlak — vessel uses vp=0.9, no Ki, so Cₜ≈0.9·Cₐ.
        ct = _patlak_ct(ca, t_s, ki_per_min=ki, vp=vp)
        # SPGR signal at flip angle a, TR, with R1(t) = 1/T1 + r1·Cₜ.
        a = np.deg2rad(flip_angle_deg)
        T1_s = T1_ms / 1000.0
        E1_pre = np.exp(-tr_s / T1_s)
        S_pre = M0 * np.sin(a) * (1 - E1_pre) / (1 - np.cos(a) * E1_pre)
        R1_t = 1.0 / T1_s + r1_per_s_mM * ct
        E1_t = np.exp(-tr_s * R1_t)
        S_t = M0 * np.sin(a) * (1 - E1_t) / (1 - np.cos(a) * E1_t)
        # Scale to baseline so signal starts near S_pre
        S_t = S_t * (S_pre / S_t[: baseline_frames].mean())
        # Broadcast to voxels with multiplicative noise
        noise = 1.0 + (noise_pct / 100.0) * rng.standard_normal(size=(int(mask.sum()), T))
        sig[mask] = (S_t[None, :] * noise).astype(np.float32)

    truth = {
        str(lbl): {
            "T1_ms": v[0], "M0": v[1],
            "ki_mL_per_100g_min": v[2], "vp_fraction": v[3],
            "cbf_mL_per_100g_min": v[4], "mtt_s": v[5], "cth_s": v[6],
        }
        for lbl, v in _TISSUE_TRUTH.items()
    }
    return sig, ca, truth


# ───────────────────── DWI synthesis (anisotropic) ────────────────────


def _icosahedral_dirs() -> np.ndarray:
    """12 evenly-spaced unit vectors on the upper hemisphere (icosahedron vertices)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0,  1,  phi], [0,  1, -phi], [0, -1,  phi], [0, -1, -phi],
        [1,  phi, 0], [1, -phi, 0], [-1,  phi, 0], [-1, -phi, 0],
        [phi, 0,  1], [phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1],
    ], dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    # Restrict to upper hemisphere (z ≥ 0) by flipping where needed
    v[v[:, 2] < 0] *= -1.0
    return v


def _diffusion_tensor(eigvals: tuple[float, float, float],
                      axis: np.ndarray) -> np.ndarray:
    """Build a rank-2 diffusion tensor with principal axis ``axis``.

    Constructs an orthonormal basis with ``axis`` as the first eigenvector;
    the other two eigenvectors are arbitrary perpendicular directions
    (eigvals[1], eigvals[2] are usually equal so this doesn't matter).
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    # Pick a non-collinear vector for Gram-Schmidt
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = seed - axis * (seed @ axis)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(axis, e2)
    R = np.column_stack([axis, e2, e3])
    D = R @ np.diag(eigvals) @ R.T
    return D


# Diffusion ground truth (per region).  Tensors give realistic FA / MD.
# Plus a small free-water fraction so DKI sees non-Gaussian decay and
# kurtosis maps are non-zero.
_DIFF_TRUTH = {
    1001: dict(
        S0=400.0, eigvals=(0.90e-3, 0.70e-3, 0.70e-3),    # cortical GM, FA ≈ 0.15
        axis=np.array([0.0, 0.0, 1.0]), fw_frac=0.15,
    ),
    2: dict(
        S0=350.0, eigvals=(1.70e-3, 0.30e-3, 0.30e-3),    # WM, FA ≈ 0.70, x-aligned
        axis=np.array([1.0, 0.0, 0.0]), fw_frac=0.05,
    ),
    11: dict(
        S0=420.0, eigvals=(1.00e-3, 0.75e-3, 0.65e-3),    # deep GM, FA ≈ 0.20
        axis=np.array([0.0, 1.0, 0.0]), fw_frac=0.12,
    ),
}
_DIFF_VESSEL = dict(
    S0=600.0, eigvals=(3.0e-3, 3.0e-3, 3.0e-3),           # near-free-water
    axis=np.array([0.0, 0.0, 1.0]), fw_frac=0.95,
)


def _make_dwi_volume(
    labels: np.ndarray, vessel: np.ndarray, *,
    snr: float = 30.0,
    rng_seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a 30-direction 2-shell DWI volume + bvals/bvecs.

    Two-shell scheme: 6 × b=0, 12 × b=1000, 12 × b=2500.  DKI requires
    multi-shell so we go ≥ 2 non-zero b-values; b=2500 ensures the
    non-Gaussian regime is reached.

    Per voxel signal (two-compartment, gives non-zero kurtosis):

        S(b, g) = S0 · [(1−f)·exp(−b·gᵀ·D·g) + f·exp(−b·D_iso)]

    where D is the region tensor, f is the free-water fraction, and
    D_iso = 3.0e-3 mm²/s.
    """
    dirs = _icosahedral_dirs()
    bvals_list = [0.0] * 6 + [1000.0] * 12 + [2500.0] * 12
    bvecs_list = (
        [np.zeros(3)] * 6
        + [v for v in dirs]
        + [v for v in dirs]
    )
    bvals = np.array(bvals_list, dtype=float)
    bvecs = np.stack(bvecs_list, axis=0).astype(float)

    X, Y, Z = labels.shape
    N = bvals.size
    sig = np.zeros((X, Y, Z, N), dtype=np.float32)
    rng = np.random.default_rng(rng_seed)
    D_iso = 3.0e-3

    region_specs: list[tuple[np.ndarray, dict]] = []
    for lbl, spec in _DIFF_TRUTH.items():
        region_specs.append((labels == lbl, spec))
    region_specs.append((vessel, _DIFF_VESSEL))

    for mask, spec in region_specs:
        if not mask.any():
            continue
        D = _diffusion_tensor(spec["eigvals"], spec["axis"])
        f = float(spec["fw_frac"])
        S0 = float(spec["S0"])
        for n in range(N):
            b, g = bvals[n], bvecs[n]
            if b <= 50:
                s = S0
            else:
                adc_tissue = float(g @ D @ g)
                s_tissue = S0 * np.exp(-b * adc_tissue)
                s_fw = S0 * np.exp(-b * D_iso)
                s = (1 - f) * s_tissue + f * s_fw
            sig[mask, n] = s

    # Rician-ish noise: add Gaussian to real + imag and take magnitude
    sigma = float(sig[sig > 0].mean()) / snr
    real = sig + sigma * rng.standard_normal(sig.shape).astype(np.float32)
    imag = sigma * rng.standard_normal(sig.shape).astype(np.float32)
    sig = np.sqrt(real * real + imag * imag).astype(np.float32)
    return sig, bvals, bvecs


def _make_ir_volume(labels: np.ndarray, vessel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inversion-recovery series at known TIs — used by the IR T1 fitter."""
    TI_s = np.array([0.04, 0.12, 0.30, 0.80, 2.50], dtype=float)
    X, Y, Z = labels.shape
    T = TI_s.size
    ir = np.zeros((X, Y, Z, T), dtype=np.float32)
    region_specs = list(_TISSUE_TRUTH.items()) + [(("vessel",), _VESSEL_TRUTH)]
    for key, spec in region_specs:
        T1_ms, M0 = spec[0], spec[1]
        if isinstance(key, int):
            mask = labels == key
        else:
            mask = vessel
        if not mask.any():
            continue
        T1_s = T1_ms / 1000.0
        S = M0 * np.abs(1 - 2 * np.exp(-TI_s / T1_s))
        ir[mask] = S[None, :].astype(np.float32)
    return ir, TI_s


# ───────────────────────────── pipeline driver ────────────────────────


def _write_phantom(root: Path) -> tuple[Path, dict]:
    raw = root / "raw" / "phantom"
    raw.mkdir(parents=True, exist_ok=True)

    dt = 2.463
    T = 250
    t_s = np.arange(T, dtype=float) * dt
    labels, vessel, label_map = _make_phantom()
    sig, ca, truth = _make_signal_volume(labels, vessel, t_s)
    ir, ti_s = _make_ir_volume(labels, vessel)
    T1 = np.zeros_like(labels, dtype=np.float32)
    for lbl, (T1_ms, *_) in _TISSUE_TRUTH.items():
        T1[labels == lbl] = T1_ms
    T1[vessel] = _VESSEL_TRUTH[0]

    affine = np.diag([2.0, 2.0, 3.0, 1.0]).astype(float)
    nib.save(nib.Nifti1Image(sig, affine), raw / "dce.nii.gz")
    nib.save(nib.Nifti1Image(ir, affine), raw / "ir.nii.gz")
    nib.save(nib.Nifti1Image(T1, affine), raw / "t1.nii.gz")
    nib.save(nib.Nifti1Image(labels.astype(np.int32), affine), raw / "labels.nii.gz")
    nib.save(nib.Nifti1Image(vessel.astype(np.uint8), affine), raw / "vessel_mask.nii.gz")

    # DWI: anisotropic per-region tensors → exercises dti/dki/csd/dki_micro/fwdti.
    dwi_data, dwi_bvals, dwi_bvecs = _make_dwi_volume(labels, vessel)
    nib.save(nib.Nifti1Image(dwi_data, affine), raw / "dwi.nii.gz")
    np.savetxt(raw / "dwi.bval", dwi_bvals[None, :], fmt="%.1f")
    np.savetxt(raw / "dwi.bvec", dwi_bvecs.T, fmt="%.6f")     # FSL row-major (3, N)

    np.save(raw / "ca_truth.npy", ca)
    np.save(raw / "t_s.npy", t_s)
    np.save(raw / "ti_s.npy", ti_s)
    (raw / "truth.json").write_text(json.dumps({"labels": label_map, "tissues": truth}, indent=2))

    print(f"  wrote phantom inputs to {raw}")
    return raw, label_map


def _run_pipeline(subject_dir: Path, raw: Path) -> None:
    """Drive ``pbrain run`` programmatically against the phantom inputs."""
    from pbrain.cli.run import main as run_main

    subject_dir.mkdir(parents=True, exist_ok=True)
    # Stage links
    for f in raw.iterdir():
        link = subject_dir / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f)

    argv = [
        "--subject-dir", str(subject_dir),
        "--dce",          str(raw / "dce.nii.gz"),
        "--ir",           str(raw / "ir.nii.gz"),
        "--t1",           str(raw / "t1.nii.gz"),
        "--dwi",          str(raw / "dwi.nii.gz"),       # → diffusion auto-defaults to 'default'
        "--bvals",        str(raw / "dwi.bval"),
        "--bvecs",        str(raw / "dwi.bvec"),
        "--aif",          "from_file",
        "--tissue-roi",   "preloaded",
        "--models",       "patlak,tikhonov",
        "--aggregations", "voxelwise,parcel,region,slice_wise",
        "--path-scheme",  "bids_like",
        "--device",       "cpu",
        "--opt",          f"aif.from_file.mask_path={raw / 'vessel_mask.nii.gz'}",
        "--opt",          f"tissue_roi.preloaded.parcellation_path={raw / 'labels.nii.gz'}",
    ]
    print("  pbrain run", " ".join(argv))
    rc = run_main(argv)
    if rc != 0:
        raise RuntimeError(f"pbrain run failed with exit code {rc}")


# ───────────────────────────── montage ────────────────────────────────


def _montage(maps_dir: Path, subject_dir: Path) -> None:
    """Montage every map under derivatives/{kinetic,diffusion}/*/voxelwise/,
    using the same adaptive renderer the pipeline uses."""
    from pbrain.diagnostics.montage import montage_volume

    candidates: list[tuple[str, Path]] = []   # (namespace, parent_dir)
    for nm in ("07_kinetic", "kinetic", "diffusion"):
        for cand in subject_dir.rglob(nm):
            if cand.is_dir():
                candidates.append((nm.replace("07_", ""), cand))
    if not candidates:
        raise FileNotFoundError("no kinetic/diffusion stage outputs found")

    maps_dir.mkdir(parents=True, exist_ok=True)
    for namespace, derivs in candidates:
        for model_dir in sorted(derivs.iterdir()):
            vox_dir = model_dir / "voxelwise"
            if not model_dir.is_dir() or not vox_dir.exists():
                continue
            for nii in sorted(vox_dir.glob("*.nii.gz")):
                arr = np.asarray(nib.load(str(nii)).dataobj, dtype=float)
                if arr.ndim != 3:
                    continue
                name = nii.stem.replace(".nii", "")
                out = maps_dir / f"{namespace}_{model_dir.name}__{name}.png"
                montage_volume(arr, out, title=f"{namespace}/{model_dir.name} — {name}")
                print(f"    wrote {out.relative_to(DEMO_ROOT.parent)}")


# ───────────────────────────── main ────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pbrain.demo")
    p.add_argument("--out", type=Path, default=DEMO_ROOT,
                   help="root output directory (default: <repo>/demo).")
    p.add_argument("--clean", action="store_true",
                   help="wipe out the existing demo dir first.")
    args = p.parse_args(argv)

    if args.clean and args.out.exists():
        print(f"removing {args.out}")
        shutil.rmtree(args.out)

    print(f"== pbrain.demo  → {args.out} ==")
    args.out.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] synthesise phantom")
    raw, _ = _write_phantom(args.out)

    print("\n[2/3] run pipeline")
    subject_dir = args.out / "phantom"
    _run_pipeline(subject_dir, raw)

    print("\n[3/3] montage maps")
    _montage(args.out / "maps", subject_dir)

    print(f"\ndone — outputs at {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
