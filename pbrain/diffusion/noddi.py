"""NODDI — Neurite Orientation Dispersion and Density via AMICO.

Zhang 2012 three-compartment microstructure model (intracellular,
extracellular, isotropic free water). NEEDS multi-shell DWI with at
least one high-b shell (b ≥ 2000 s/mm² recommended).

Outputs:

* ``icvf`` — intra-cellular volume fraction (≈ neurite density)
* ``odi``  — orientation dispersion index (0 = single direction, 1 = isotropic)
* ``iso``  — isotropic (CSF / free-water) volume fraction

Implementation uses AMICO (Daducci 2015) — orders of magnitude faster
than the original NODDI MATLAB toolbox or dipy's NODDI by precomputing
a kernel dictionary. The plug-in is gated on AMICO being importable;
without it, raises a clear error pointing at ``pip install dmri-amico``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import DWIInputs, DiffusionResult


@dataclass(frozen=True, slots=True)
class _NODDI:
    key:         ClassVar[str] = "noddi"
    name:        ClassVar[str] = "NODDI (Neurite Orientation Dispersion)"
    description: ClassVar[str] = (
        "Three-compartment microstructure model (Zhang 2012) via AMICO. "
        "Needs multi-shell DWI with a high-b shell (≥2000). Outputs "
        "intracellular volume fraction, orientation dispersion, and "
        "isotropic free-water fraction."
    )
    accepts:  ClassVar[dict[str, type]] = {
        "signal": np.ndarray, "bvals": np.ndarray, "bvecs": np.ndarray,
    }
    produces: ClassVar[dict[str, type]] = {
        "icvf": np.ndarray, "odi": np.ndarray, "iso": np.ndarray,
    }
    outputs: ClassVar[tuple[str, ...]] = ("icvf", "odi", "iso")
    units: ClassVar[dict[str, str]] = {
        "icvf": "(fraction)", "odi": "(unitless)", "iso": "(fraction)",
    }
    primary_map: ClassVar[str] = "icvf"

    def fit(self, inputs: DWIInputs, **opts: Any) -> DiffusionResult:
        try:
            import amico
        except ImportError as exc:
            raise RuntimeError(
                "NODDI plug-in requires AMICO. "
                "Install with: pip install dmri-amico"
            ) from exc

        import nibabel as nib
        import tempfile

        signal = np.asarray(inputs.signal, dtype=np.float32)
        bvals = np.asarray(inputs.bvals, dtype=float)
        bvecs = np.asarray(inputs.bvecs, dtype=float)
        if inputs.mask is None:
            raise RuntimeError("NODDI requires a brain mask")
        mask = np.asarray(inputs.mask, dtype=np.uint8)
        affine = np.asarray(inputs.affine, dtype=float)

        shells = sorted({int(round(b / 100.0) * 100) for b in bvals if b > 50})
        if len(shells) < 2 or max(shells) < 1500:
            raise RuntimeError(
                f"NODDI needs ≥2 non-zero shells incl. one ≥1500 s/mm²; got {shells}."
            )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dwi_p = td / "dwi.nii.gz"
            mask_p = td / "mask.nii.gz"
            scheme_p = td / "scheme.txt"
            nib.save(nib.Nifti1Image(signal, affine), str(dwi_p))
            nib.save(nib.Nifti1Image(mask, affine), str(mask_p))

            # AMICO scheme file (FSL-like, columns bvec_x bvec_y bvec_z bval)
            with scheme_p.open("w") as f:
                f.write("VERSION: BVAL\n")
                for v, b in zip(bvecs, bvals):
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {b:.2f}\n")

            amico.core.setup()
            ae = amico.Evaluation(str(td), "noddi_run")
            ae.load_data(dwi_filename=str(dwi_p), scheme_filename=str(scheme_p),
                         mask_filename=str(mask_p), b0_thr=50.0)
            ae.set_model("NODDI")
            ae.generate_kernels(regenerate=True)
            ae.load_kernels()
            ae.fit()
            ae.save_results()

            out_dir = td / "noddi_run" / "AMICO" / "NODDI"
            icvf = np.asarray(nib.load(str(out_dir / "fit_NDI.nii.gz")).dataobj, dtype=np.float32)
            odi = np.asarray(nib.load(str(out_dir / "fit_ODI.nii.gz")).dataobj, dtype=np.float32)
            iso = np.asarray(nib.load(str(out_dir / "fit_FWF.nii.gz")).dataobj, dtype=np.float32)

        m = mask.astype(bool)
        for arr in (icvf, odi, iso):
            arr[~m] = np.nan

        return DiffusionResult(
            maps={"icvf": icvf, "odi": odi, "iso": iso},
            units=dict(self.units),
            aux={"shells": shells, "backend": "amico"},
        )


PLUGIN = _NODDI()
