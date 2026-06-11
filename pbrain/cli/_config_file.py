"""Config-file loading for ``pbrain run --config study.{toml,yaml}``.

A config file is a versionable, shareable alternative to a long CLI
command. Example (TOML)::

    subject_dir = "/data/sub-01"

    [inputs]
    dce = "dce.nii.gz"
    t1  = "t1.nii.gz"
    dwi = "dwi.nii.gz"

    [pipeline]
    t1m0          = "preloaded"
    aif           = "cnn_sss_shifted"
    tissue_roi    = "preloaded"
    models        = ["patlak", "tikhonov_bayes"]
    diffusion     = "default"
    aggregations  = ["region", "parcel"]
    path_scheme   = "bids_like"
    device        = "auto"

    [acquisition]
    flip_angle_deg = 8.0
    tr_s           = 0.01118

    # Per-plug-in options — the same keys as --opt <plug-point>.<plugin>.<key>
    [options]
    "models.tikhonov_bayes.uncertainty_samples" = 200
    "t1_m0.preloaded.t1_ms_path" = "/data/sub-01/t1_map.nii.gz"

The equivalent YAML is accepted too (requires ``pyyaml``). CLI flags
always override the file. Returns a flat dict matching the argparse
destinations so the caller can merge it as defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in (".toml",):
        import tomllib
        return tomllib.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "YAML config needs pyyaml — `pip install pyyaml`, or use a "
                ".toml file (TOML support is built in)."
            ) from exc
        return yaml.safe_load(text) or {}
    raise SystemExit(f"--config: unsupported file type {suffix!r} (use .toml/.yaml)")


def load_config_file(path: Path) -> dict[str, Any]:
    """Parse a config file into a flat dict of argparse-style defaults.

    Keys returned mirror the ``pbrain run`` flags (``subject_dir``, ``dce``,
    ``models``, ``aif``, …) plus ``opt`` (a list of ``a.b.c=v`` strings) and
    ``inputs`` (path overrides). Lists are joined to comma strings to match
    the CLI's string flags.
    """
    raw = _read(Path(path))
    out: dict[str, Any] = {}

    # top-level scalars
    for k in ("subject_dir",):
        if k in raw:
            out[k] = raw[k]

    inputs = raw.get("inputs", {}) or {}
    for k in ("dce", "t1", "ir", "dwi", "bvals", "bvecs"):
        if k in inputs:
            out[k] = inputs[k]

    pipe = raw.get("pipeline", {}) or {}
    # name-mapping: config key → argparse dest
    pmap = {
        "t1m0": "t1m0", "aif": "aif", "tissue_roi": "tissue_roi",
        "signal_to_conc": "signal_to_conc", "normaliser": "normaliser",
        "models": "models", "diffusion": "diffusion",
        "aggregations": "aggregations", "path_scheme": "path_scheme",
        "device": "device",
    }
    for ck, dest in pmap.items():
        if ck in pipe:
            v = pipe[ck]
            out[dest] = ",".join(map(str, v)) if isinstance(v, list) else str(v)

    acq = raw.get("acquisition", {}) or {}
    for ck, dest in (("flip_angle_deg", "flip_angle_deg"), ("tr_s", "tr_s"),
                     ("r1_per_s_mM", "r1_per_s_mM"),
                     ("baseline_frames", "baseline_frames"), ("dt_s", "dt_s")):
        if ck in acq:
            out[dest] = acq[ck]

    # [options] → repeated --opt strings
    opts = raw.get("options", {}) or {}
    out["opt"] = [f"{k}={v}" for k, v in opts.items()]

    return out
