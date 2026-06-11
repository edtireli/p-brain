"""DICOM loader.

Wraps ``dcm2niix`` (the same converter the paper uses for PAR/REC). Pass
either a single ``.dcm`` file or a directory holding a DICOM series;
dcm2niix discovers siblings automatically.

Raises ``RuntimeError`` with a clear install hint if dcm2niix is not on
``PATH``.
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
            cmd = ["dcm2niix", "-z", "y", "-o", str(tmp_dir), "-f", "out", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(
                    f"dcm2niix failed for {path}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            niftis = sorted(tmp_dir.glob("out*.nii.gz")) or sorted(tmp_dir.glob("out*.nii"))
            if not niftis:
                raise RuntimeError(f"dcm2niix produced no NIfTI output in {tmp_dir}")
            largest = max(niftis, key=lambda p: p.stat().st_size)
            return _NIFTI.load(largest)


PLUGIN = _DicomLoader()
