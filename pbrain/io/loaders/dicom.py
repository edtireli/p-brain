"""DICOM loader.

Wraps ``dcm2niix`` (the same converter the paper uses for PAR/REC). It
handles every DICOM flavour dcm2niix supports:

* a **single classic DICOM file** (``.dcm`` / ``.ima`` / no extension);
* a **directory holding a DICOM series** — dcm2niix discovers the sibling
  slices automatically (``-r`` is implied for a directory argument);
* **enhanced / multi-frame DICOM** — a single file whose frames are packed
  into one object; dcm2niix unpacks it to a 3-D/4-D NIfTI like any other.

Output selection: dcm2niix may emit several NIfTIs (e.g. a multi-frame
object that splits by echo/phase). We return the **largest** volume — for a
DCE series this is the full 4-D dynamic. If a specific sub-series is needed,
pre-convert with ``dcm2niix`` manually and pass the resulting ``.nii.gz``
to the NIfTI loader instead.

Errors are explicit and actionable:

* dcm2niix not on ``PATH`` → install hint (conda-forge / brew / apt);
* dcm2niix runs but yields no NIfTI (unsupported/secondary-capture/private
  enhanced object) → a clear message recommending manual pre-conversion.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .base import ImageLoader, Series4D
from .nifti import PLUGIN as _NIFTI


def _is_dicom(path: Path) -> bool:
    """Cheap DICOM check: file extension or first 132 bytes hint."""
    if path.is_dir():
        # A directory of DICOMs: look for any *.dcm or any file with the magic.
        for child in path.iterdir():
            if child.is_file() and _is_dicom_file(child):
                return True
        return False
    return _is_dicom_file(path)


def _is_dicom_file(path: Path) -> bool:
    if path.suffix.lower() in (".dcm", ".ima"):
        return True
    try:
        with path.open("rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class _DicomLoader:
    key: ClassVar[str] = "dicom"
    name: ClassVar[str] = "DICOM loader"
    description: ClassVar[str] = (
        "Wraps dcm2niix to read a single DICOM file or directory of "
        "DICOM series. Returns the largest discovered NIfTI."
    )
    accepts: ClassVar[dict[str, type]] = {"path": Path}
    produces: ClassVar[dict[str, type]] = {"series": Series4D}
    extensions: ClassVar[tuple[str, ...]] = (".dcm", ".ima", "/")

    def detect(self, path: Path) -> bool:
        return _is_dicom(path)

    def load(self, path: Path, **opts: Any) -> Series4D:
        path = Path(path)
        if not shutil.which("dcm2niix"):
            raise RuntimeError(
                "dcm2niix is required for DICOM loading but was not found on PATH. "
                "Install via `brew install dcm2niix` (macOS), `apt install dcm2niix` "
                "(Linux), or download from https://github.com/rordenlab/dcm2niix"
            )

        with tempfile.TemporaryDirectory(prefix="pbrain_dicom_") as tmp:
            tmp_dir = Path(tmp)
            # ``-z y`` gzip; ``-f out`` fixed stem. For a directory we add
            # ``-r y`` so dcm2niix recurses into a nested series tree; a single
            # (incl. enhanced/multi-frame) file converts in place.
            cmd = ["dcm2niix", "-z", "y", "-o", str(tmp_dir), "-f", "out"]
            if path.is_dir():
                cmd += ["-r", "y"]
            cmd.append(str(path))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(
                    f"dcm2niix failed for {path}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            niftis = sorted(tmp_dir.glob("out*.nii.gz")) or sorted(tmp_dir.glob("out*.nii"))
            if not niftis:
                raise RuntimeError(
                    f"dcm2niix produced no NIfTI for {path}. This usually means "
                    "an unsupported object (secondary capture, a private/"
                    "non-image enhanced-DICOM SOP class, or a non-DICOM file). "
                    "Pre-convert manually with `dcm2niix -z y -o <dir> <input>` "
                    "and load the resulting .nii.gz, or check the dcm2niix log:\n"
                    f"{result.stdout.strip()}"
                )
            # dcm2niix can split one input into several NIfTIs (echoes, phases,
            # a multi-frame object that fans out). The full DCE dynamic is the
            # largest volume; return it. (Pre-convert manually to pick another.)
            largest = max(niftis, key=lambda p: p.stat().st_size)
            return _NIFTI.load(largest)


PLUGIN = _DicomLoader()
