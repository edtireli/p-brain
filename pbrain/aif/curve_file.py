"""AIF from a precomputed 1-D concentration curve file.

Loads a concentration-time curve **verbatim** from ``.npy`` / ``.npz`` /
``.mat`` / ``.json`` and returns it as the input function — no
extraction, no baseline re-shift. Use this to re-use an AIF produced by
an external or legacy pipeline (e.g. old-pbrain's ``TSCC-Max`` curve at
``Analysis/TSCC Data/Max/TSCC_slice_*.npy``) so the kinetic results
bit-match that pipeline.

Unlike :mod:`pbrain.aif.from_file`, which loads a *mask* and averages
the DCE signal inside it, this plug-in loads the *curve itself*.

Options
-------
curve_path : str | Path
    File holding the 1-D curve. A *directory* is also accepted: the
    first ``*.npy`` (lexically sorted, ``._``-prefixed AppleDouble files
    skipped) inside it is loaded — this matches old-pbrain's
    ``TSCC Data/Max`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import AIFExtractor, InputFunction


def _load_curve(path: Path) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        npys = sorted(p for p in path.glob("*.npy") if not p.name.startswith("._"))
        if not npys:
            raise FileNotFoundError(f"No .npy curve file in directory {path}")
        path = npys[0]
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        d = np.load(path)
        arr = d[d.files[0]]
    elif suffix == ".mat":
        from scipy.io import loadmat
        d = loadmat(str(path))
        arr = next(np.asarray(v) for k, v in d.items() if not k.startswith("__"))
    elif suffix == ".json":
        import json
        with open(path) as f:
            arr = np.asarray(json.load(f), dtype=float)
    else:
        raise ValueError(f"Unsupported curve format: {suffix!r} ({path})")
    return np.asarray(arr, dtype=float).ravel()


@dataclass(frozen=True, slots=True)
class _CurveFileAIF:
    key: ClassVar[str] = "curve_file"
    name: ClassVar[str] = "AIF from precomputed curve file"
    description: ClassVar[str] = (
        "Load a 1-D concentration-time curve from .npy/.npz/.mat/.json and "
        "use it verbatim as the AIF. Re-uses an external/legacy input "
        "function (e.g. old-pbrain TSCC-Max) for bit-parity."
    )
    accepts: ClassVar[dict[str, type]] = {"curve_path": Path}
    produces: ClassVar[dict[str, type]] = {"input_function": InputFunction}

    def extract(
        self,
        dce_data: np.ndarray,
        t_s: np.ndarray,
        dce_affine: np.ndarray,
        *,
        t1_map: np.ndarray | None = None,
        m0_map: np.ndarray | None = None,
        t1w_volume: np.ndarray | None = None,
        t1w_affine: np.ndarray | None = None,
        concentration_data: np.ndarray | None = None,
        **opts: Any,
    ) -> InputFunction:
        curve_path = opts.get("curve_path")
        if curve_path is None:
            raise ValueError(
                "curve_file AIF requires opts['curve_path'] "
                "(path to a .npy/.npz/.mat/.json curve, or a directory of .npy)"
            )
        c_a = _load_curve(Path(curve_path))
        t = np.asarray(t_s, dtype=float).ravel()
        if t.size:
            n = int(min(c_a.size, t.size))
            c_a, t = c_a[:n], t[:n]
        mask = np.zeros(tuple(dce_data.shape[:3]), dtype=bool)
        return InputFunction(
            c_a=c_a,
            t_s=t,
            mask=mask,
            source="file",
            meta={"curve_path": str(curve_path), "n_points": int(c_a.size)},
        )


PLUGIN = _CurveFileAIF()
