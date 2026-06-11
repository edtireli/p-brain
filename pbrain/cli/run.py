"""``python -m pbrain run`` — execute the DCE-MRI pipeline on a subject.

Usage::

    python -m pbrain run \\
        --subject-dir <path>          \\
        --dce dce.nii.gz              \\
        --t1 t1.nii.gz                \\
        --ir ir.nii.gz                \\
        --aif deterministic           \\
        --tissue-roi voxelwise        \\
        --models patlak,tikhonov      \\
        --aggregations voxelwise,parcel,region,slice_wise \\
        --path-scheme bids_like

Override any plug-in option with ``--opt <plug-point>.<plugin>.<key>=<value>``,
e.g. ``--opt aif.deterministic.n_voxels=64``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pbrain.core import Config, Pipeline
from pbrain.io.path_schemes import REGISTRY as PATH_SCHEMES
from pbrain.stages import default_stages


def _parse_opts(opt_pairs: list[str]) -> dict[str, dict[str, str]]:
    """Parse ``--opt plug.point.key=value`` pairs into nested config."""
    out: dict[str, dict[str, str]] = {}
    for pair in opt_pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--opt expects key=value form; got {pair!r}")
        key, val = pair.split("=", 1)
        parts = key.split(".")
        if len(parts) < 3:
            raise SystemExit(
                f"--opt key must be <plug-point>.<plugin>.<opt>; got {key!r}"
            )
        plug_point, plugin = parts[0], parts[1]
        opt_name = ".".join(parts[2:])
        out.setdefault(f"{plug_point}.{plugin}", {})[opt_name] = _coerce(val)
    return out


def _coerce(val: str):
    """Cast a CLI string into int/float/bool when it looks like one."""
    lo = val.strip().lower()
    if lo in ("true", "false"):
        return lo == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbrain run", description=__doc__)
    p.add_argument("--config", type=Path, default=None,
                   help="Config file (.toml or .yaml) supplying any of the "
                        "flags below. CLI flags override the file. See "
                        "docs/ADDING_PLUGINS.md for the schema.")
    # Not required at the parser level — may be supplied by --config.
    p.add_argument("--subject-dir", type=Path, default=None,
                   help="Subject root directory (outputs land under here).")
    p.add_argument("--dce", type=Path, default=None, help="DCE 4-D series.")
    p.add_argument("--t1", type=Path, default=None, help="T1-weighted anatomical volume.")
    p.add_argument("--ir", type=Path, default=None, help="Inversion-recovery series.")
    p.add_argument("--dwi", type=Path, default=None,
                   help="DWI 4-D series (NIfTI). Sidecars .bval / .bvec auto-detected; "
                        "override via --bvals / --bvecs. Required when --diffusion is set.")
    p.add_argument("--bvals", type=Path, default=None, help="FSL .bval sidecar.")
    p.add_argument("--bvecs", type=Path, default=None, help="FSL .bvec sidecar.")

    p.add_argument("--t1m0", default="inversion_recovery",
                   help="T1/M0 fitter plug-in key.")
    p.add_argument("--aif", default="cnn_sss_shifted",
                   help="AIF extractor plug-in key (default reproduces paper §4.5.1: "
                        "CNN SSS time-shifted to rICA).")
    p.add_argument("--tissue-roi", default="voxelwise",
                   help="Tissue ROI provider plug-in key.")
    p.add_argument("--signal-to-conc", default="saturation_recovery",
                   help="Signal-to-concentration plug-in key.")
    p.add_argument("--normaliser", default="legacy_shift",
                   help="Curve-normaliser plug-in key.")
    p.add_argument("--models", default="patlak,tikhonov",
                   help="Comma-separated list of kinetic-model keys.")
    p.add_argument("--diffusion", default=None,
                   help="Comma-separated list of diffusion-model keys "
                        "(e.g. 'dti,dki,csd'), or one of the bundle "
                        "keywords 'default' / 'all'. If --dwi is given "
                        "and this flag is omitted, defaults to 'default' "
                        "(shell-aware: dti always; dki/dki_micro/csd/"
                        "fwdti added for multi-shell data). Empty string "
                        "explicitly disables the diffusion track.")
    p.add_argument("--aggregations", default="voxelwise,parcel,region",
                   help="Comma-separated list of aggregator keys.")
    p.add_argument("--path-scheme", default="bids_like",
                   choices=sorted(PATH_SCHEMES.keys()),
                   help="Output path-scheme plug-in key.")
    p.add_argument("--opt", action="append", default=[], metavar="P.PLUGIN.OPT=VAL",
                   help="Override a plug-in option (repeatable).")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "mps", "cuda", "auto"],
                   help="Compute device. 'mps' requires tensorflow-metal "
                        "(for CNN AIF) and/or torch with MPS. 'auto' picks "
                        "the best available.")

    p.add_argument("--force", action="store_true",
                   help="Re-run all stages even if cached manifests exist.")
    p.add_argument("--quiet", action="store_true", help="Warnings/errors only.")
    p.add_argument("--verbose", action="store_true", help="Debug-level detail.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Also append all log records to this file.")

    p.add_argument("--flip-angle-deg", type=float, default=30.0)
    p.add_argument("--tr-s", type=float, default=0.01118)
    p.add_argument("--r1-per-s-mM", type=float, default=4.0)
    p.add_argument("--baseline-frames", type=int, default=5)
    p.add_argument("--dt-s", type=float, default=2.463)
    return p


def _expand_diffusion_bundle(spec: str | None, dwi_path: Path | None) -> tuple[str, ...]:
    """Resolve ``--diffusion`` to a concrete tuple of model keys.

    Rules (most specific first):

    * ``spec = "" (empty string)`` → ``()``  (explicit "no diffusion").
    * ``spec = None`` (flag omitted):
        - if ``--dwi`` given  → behaves like ``"default"``
        - else                → ``()``  (no DWI, no diffusion).
    * ``"default"`` → shell-aware sensible set: ``dti`` always, plus
      ``dki, dki_micro, csd, fwdti`` if the DWI is multi-shell.
    * ``"all"`` → every plug-in supported by the data (adds ``noddi``
      when a high-b shell ≥ 1500 is present *and* AMICO importable).
    * Otherwise: comma-list, validated against the registry.
    """
    from pbrain.diffusion import REGISTRY as DIFF
    from pbrain.io.loaders.dwi import load_dwi

    if spec == "":
        return ()
    if spec is None:
        if dwi_path is None:
            return ()
        spec = "default"
    spec = spec.strip()

    if spec not in {"default", "all"}:
        keys = tuple(s for s in spec.split(",") if s)
        unknown = [k for k in keys if k not in DIFF]
        if unknown:
            raise SystemExit(
                f"--diffusion: unknown plug-in(s) {unknown}. "
                f"Available: {sorted(DIFF.keys())}"
            )
        return keys

    if dwi_path is None or not dwi_path.exists():
        raise SystemExit(
            "--diffusion 'default'/'all' requires --dwi pointing to a DWI volume."
        )
    # Inspect shells once.
    try:
        dwi = load_dwi(dwi_path)
    except Exception as exc:
        raise SystemExit(f"--diffusion: could not load DWI to choose bundle: {exc}")

    nonzero_shells = [s for s in dwi.shells if s > 50]
    is_multi = len(nonzero_shells) >= 2
    has_high_b = any(s >= 1500 for s in nonzero_shells)

    chosen: list[str] = ["dti"]
    if is_multi:
        chosen += ["dki", "dki_micro", "csd", "fwdti"]
        if has_high_b:
            chosen.append("rsi")           # RSI needs multi-shell with a high-b shell
    if spec == "all" and is_multi and has_high_b:
        try:
            import amico  # noqa: F401
            chosen.append("noddi")
        except Exception:
            print("--diffusion all: AMICO not installed → skipping NODDI "
                  "(pip install dmri-amico to enable).", file=sys.stderr)
    print(f"diffusion bundle '{spec}': shells={dwi.shells} "
          f"→ {chosen}", file=sys.stderr)
    return tuple(chosen)


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Config file supplies defaults; anything also given on the CLI wins.
    if args.config is not None:
        from ._config_file import load_config_file
        file_defaults = load_config_file(args.config)
        file_opts = file_defaults.pop("opt", [])
        # Dests explicitly present on the CLI (so the file does not override).
        cli_dests = {tok.lstrip("-").replace("-", "_").split("=")[0]
                     for tok in argv if tok.startswith("--")}
        path_dests = {"subject_dir", "dce", "t1", "ir", "dwi", "bvals", "bvecs"}
        for dest, val in file_defaults.items():
            if dest in cli_dests:
                continue
            setattr(args, dest, Path(val) if dest in path_dests else val)
        # File --opt entries append to (don't replace) any CLI --opt.
        args.opt = list(file_opts) + list(args.opt)

    if args.subject_dir is None:
        raise SystemExit("--subject-dir is required (CLI or --config).")

    # ``{subject}`` in any input path is replaced with the subject dir name —
    # lets one config file drive a whole cohort whose inputs live elsewhere,
    # e.g. dce = "/data/{subject}/NIfTI/dce.nii.gz".
    subj_name = args.subject_dir.name
    for dest in ("dce", "t1", "ir", "dwi", "bvals", "bvecs"):
        v = getattr(args, dest, None)
        if v is not None and "{subject}" in str(v):
            setattr(args, dest, Path(str(v).replace("{subject}", subj_name)))

    if args.dce is None:
        raise SystemExit("--dce is required (CLI or --config).")

    from pbrain.core import configure_logging, get_logger, level_from_flags
    configure_logging(level_from_flags(args.quiet, args.verbose),
                      log_file=args.log_file)
    log = get_logger("run")

    plugin_options = _parse_opts(args.opt)
    # Apply {subject} templating to plug-in option string values too.
    for grp in plugin_options.values():
        for k, v in list(grp.items()):
            if isinstance(v, str) and "{subject}" in v:
                grp[k] = v.replace("{subject}", subj_name)
    plugin_options.setdefault("inputs.paths", {})
    plugin_options["inputs.paths"]["dce"] = str(args.dce)
    if args.t1:
        plugin_options["inputs.paths"]["t1_anatomical"] = str(args.t1)
    if args.ir:
        plugin_options["inputs.paths"]["ir"] = str(args.ir)
    if args.dwi:
        plugin_options["inputs.paths"]["dwi"] = str(args.dwi)
    if args.bvals:
        plugin_options["inputs.paths"]["bvals"] = str(args.bvals)
    if args.bvecs:
        plugin_options["inputs.paths"]["bvecs"] = str(args.bvecs)

    diffusion_models = _expand_diffusion_bundle(args.diffusion, args.dwi)

    # Resolve + configure the accelerator before any plug-in initialises.
    from pbrain.core import resolve_device, configure_tf_device, probe_devices
    resolved = resolve_device(args.device)
    if args.device != resolved:
        log.info("device: requested=%r → resolved=%r", args.device, resolved)
    log.info("device: using %r  (probe: %s)", resolved, probe_devices())
    if resolved != "cpu":
        configure_tf_device(resolved)

    config = Config(
        t1_m0_fitter=args.t1m0,
        aif_extractor=args.aif,
        tissue_roi_provider=args.tissue_roi,
        signal_to_conc=args.signal_to_conc,
        normaliser=args.normaliser,
        kinetic_models=tuple(s for s in args.models.split(",") if s),
        diffusion_models=diffusion_models,
        analysis_levels=tuple(args.aggregations.split(",")),
        path_scheme=args.path_scheme,
        flip_angle_deg=args.flip_angle_deg,
        tr_s=args.tr_s,
        r1_per_s_mM=args.r1_per_s_mM,
        baseline_frames=args.baseline_frames,
        dt_s=args.dt_s,
        subject_id=args.subject_dir.name,
        data_root=args.subject_dir.parent,
        device=resolved,
        plugin_options=plugin_options,
    )

    pipeline = Pipeline(stages=default_stages(),
                        path_scheme=PATH_SCHEMES[args.path_scheme])

    log.info("Running pipeline on %s", args.subject_dir)
    records = pipeline.run(args.subject_dir, config, force=args.force)

    log.info("Stage completion:")
    for name, rec in records.items():
        log.info("  %-18s %d artefacts -> %s",
                 name, len(rec.output.artefacts), rec.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
