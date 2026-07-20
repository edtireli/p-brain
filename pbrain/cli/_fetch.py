"""Download the optional Zenodo assets — CNN AIF weights and the example dataset.

These are deliberately *not* bundled in the pip package (weights are large; data
is patient-derived and versioned separately). ``pbrain fetch-weights`` /
``pbrain fetch-data`` (and the ``pbrain setup`` wizard) pull them on demand into
the per-user asset directory (:func:`pbrain._paths.weights_dir`), which the CNN
AIF loader searches automatically.

Configuring the source (until the Zenodo record IDs are baked in below):
    * weights: set ``$PBRAIN_WEIGHTS_URL`` to the record's files base URL, e.g.
      ``https://zenodo.org/records/<ID>/files`` — each ``.keras`` file is then
      fetched from ``<base>/<name>?download=1``.
    * data:    set ``$PBRAIN_DATA_URL`` to a single archive URL, e.g.
      ``https://zenodo.org/records/<ID>/files/example.zip?download=1``.
Once the records are public, fill ``_ZENODO_WEIGHTS_BASE`` / ``_ZENODO_DATA_URL``
and end users need no env vars.
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

from pbrain._paths import weights_dir

# ── Zenodo sources ─────────────────────────────────────────────────────────────
# Trained CNN weights (~1.15 GB across the four .keras files) and the example
# dataset (sub-01, ~99 MB). Override either with $PBRAIN_WEIGHTS_URL /
# $PBRAIN_DATA_URL if the records ever move.
_ZENODO_WEIGHTS_BASE = "https://zenodo.org/records/15697443/files"
_ZENODO_DATA_URL = "https://zenodo.org/records/20826857/files/pbrain_example_sub-01.zip?download=1"

# The four .keras files the CNN AIF (`cnn_sss_shifted`, the default) loads.
WEIGHTS_FILES = (
    "slice_classifier_model_rica.keras",
    "rica_roi_model.keras",
    "ss_slice_classifier.keras",
    "ss_roi_model.keras",
)


def _weights_base() -> str:
    return os.environ.get("PBRAIN_WEIGHTS_URL", _ZENODO_WEIGHTS_BASE).rstrip("/")


def _data_url() -> str:
    return os.environ.get("PBRAIN_DATA_URL", _ZENODO_DATA_URL)


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


def _download(url: str, dest: Path) -> bool:
    """Stream ``url`` to ``dest`` (atomic via a .part temp). Returns success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "p-brain"})
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310 — fixed https Zenodo URLs
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = 100 * got / total
                        print(f"\r    {dest.name}: {pct:5.1f}%  ({_human(got)}/{_human(total)})",
                              end="", flush=True)
                    else:
                        print(f"\r    {dest.name}: {_human(got)}", end="", flush=True)
            print()
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001 — network/IO errors -> clean message, no traceback
        print(f"\n    failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def fetch_weights(dest: Path | None = None, *, force: bool = False) -> int:
    """Download the CNN AIF ``.keras`` weights into the user weights dir.

    Skips files already present (unless ``force``). Returns a process exit code
    (0 = all weights present afterwards). ``pbrain fetch-weights`` entry point.
    """
    target = Path(dest) if dest is not None else weights_dir()
    base = _weights_base()
    if not base:
        print("No weights URL configured. Set $PBRAIN_WEIGHTS_URL to the Zenodo\n"
              "record's files base (e.g. https://zenodo.org/records/<ID>/files),\n"
              "or hard-code _ZENODO_WEIGHTS_BASE in pbrain/cli/_fetch.py.")
        return 1

    target.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(WEIGHTS_FILES)} CNN weight file(s) -> {target}")
    ok = True
    for name in WEIGHTS_FILES:
        out = target / name
        if out.exists() and not force:
            print(f"    {name}: already present, skipping")
            continue
        if not _download(f"{base}/{name}?download=1", out):
            ok = False
    have = sum((target / n).exists() for n in WEIGHTS_FILES)
    print(f"Weights: {have}/{len(WEIGHTS_FILES)} present in {target}")
    return 0 if (ok and have == len(WEIGHTS_FILES)) else 1


def fetch_data(dest: Path | None = None, *, force: bool = False) -> int:
    """Download (and unzip, if a .zip) the example dataset. ``pbrain fetch-data``."""
    url = _data_url()
    if not url:
        print("No data URL configured. Set $PBRAIN_DATA_URL to the Zenodo archive\n"
              "URL (e.g. https://zenodo.org/records/<ID>/files/example.zip?download=1),\n"
              "or hard-code _ZENODO_DATA_URL in pbrain/cli/_fetch.py.")
        return 1

    target = Path(dest) if dest is not None else (Path.cwd() / "pbrain-example-data")
    target.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "pbrain-data.bin"
    archive = target / name
    if archive.exists() and not force:
        print(f"    {name}: already present, skipping download")
    elif not _download(url, archive):
        return 1

    if zipfile.is_zipfile(archive):
        print(f"    unzipping {name} -> {target}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    print(f"Example data ready in {target}")

    # Locate the subject wherever it landed (the archive may nest it a folder
    # deep) and print a ready-to-run, weights-free command so the user does not
    # have to reconstruct the paths or flags.
    hits = sorted(target.rglob("sub-01_dce.nii.gz"))
    if hits:
        sub = hits[0].parent
        # Build the argument groups once, then join for the current shell. Windows
        # PowerShell/cmd do not understand the POSIX "\" line-continuation, so we
        # print a single line there; POSIX shells get the readable multi-line form.
        parts = [
            "pbrain run",
            f'--subject-dir "{sub}"',
            f'--dce "{sub / "sub-01_dce.nii.gz"}" --relax "{sub / "sub-01_ir.nii.gz"}"',
            f'--aif curve_file --opt aif.curve_file.curve_path="{sub / "sub-01_aif.npy"}"',
            f'--tissue-roi preloaded --opt tissue_roi.preloaded.parcellation_path='
            f'"{sub / "sub-01_parcellation.nii.gz"}"',
            # median_curve is not optional here: expected_outputs/ carries a
            # pooled-curve Ki reference, which only that aggregation produces.
            "--models patlak,tikhonov "
            "--aggregations median_curve,region,parcel,voxelwise",
        ]
        print("\nRun it now (weights-free: no CNN weights and no FreeSurfer/SynthSeg needed):\n")
        if os.name == "nt":
            print("  " + " ".join(parts))   # one line: safe for PowerShell and cmd
        else:
            print("  " + " \\\n    ".join(parts))
        exp = next(iter(target.rglob("expected_outputs")), None)
        if exp is not None:
            print(f'\nThen compare "{sub / "derivatives"}" against "{exp}".')
    return 0


def _main(argv: list[str], *, what: str) -> int:
    """Tiny arg shim for the ``fetch-weights`` / ``fetch-data`` subcommands."""
    dest = None
    force = False
    rest = list(argv)
    while rest:
        a = rest.pop(0)
        if a in ("-f", "--force"):
            force = True
        elif a in ("-o", "--out", "--dest") and rest:
            dest = Path(rest.pop(0))
        elif a in ("-h", "--help"):
            print(f"usage: pbrain {what} [--out DIR] [--force]")
            return 0
        else:
            print(f"unknown option: {a}", file=sys.stderr)
            return 2
    return fetch_weights(dest, force=force) if what == "fetch-weights" else fetch_data(dest, force=force)
