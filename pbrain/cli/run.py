"""``python -m pbrain run`` — execute the DCE-MRI pipeline on a subject.

Usage::

    python -m pbrain run \\
        --subject-dir <path>          \\
        --dce dce.nii.gz              \\
        --t1 t1.nii.gz                \\
        --relax ir.nii.gz             \\
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
import os
import sys
from pathlib import Path

from pbrain.core import Config, Pipeline
from pbrain.io.path_schemes import REGISTRY as PATH_SCHEMES
from pbrain.stages import default_stages


def _resolve_subject_input(value: str, subject_dir: Path, kind: str) -> Path | None:
    """Resolve a ``--dce/--t1/--ir`` value that may not be a full path.

    A researcher running many subjects shouldn't have to paste a different
    absolute path per subject when the *protocol name* (or a converted
    filename) is constant. ``value`` is tried, in order, as:

    1. an existing path (back-compatible — full paths still work);
    2. a filename relative to the subject dir or its parent
       (e.g. ``ir_stack.nii.gz``);
    3. ``"auto"`` → protocol-based discovery (DCE=``hperf*``, a 3-D T1
       anatomical, or assemble the ``TI_*`` series for the IR);
    4. otherwise a **protocol-name substring** (e.g. ``--dce hperf``,
       ``--t1 T1W_3D_TFE``), matched against the raw PAR headers.

    Searches the subject dir and its parent (so it works whether
    ``--subject-dir`` is the raw scan folder or a ``…/pbrain`` output dir).
    Returns the resolved path, or ``None`` if nothing matched.
    """
    from pbrain.io.subject_discovery import (
        _pars, find_dce, find_t1_anatomical, protocol_name,
    )

    sval = str(value).strip()
    is_auto = sval.lower() == "auto"
    roots = [subject_dir, subject_dir.parent]

    if not is_auto:
        p = Path(sval).expanduser()
        if p.exists():                                   # (1) explicit path
            return p
        for root in roots:                               # (2) filename in a root
            if (root / sval).exists():
                return root / sval

    for root in roots:                                   # (3)/(4) discover / protocol
        if not root.is_dir():
            continue
        if kind == "ir":
            from pbrain.io.ir_assembly import assemble_ir, find_ir_files
            if find_ir_files(root):
                out = subject_dir / "ir_assembled.nii.gz"
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    assemble_ir(root, out)
                return out
            continue
        if is_auto:
            hit = find_dce(root) if kind == "dce" else find_t1_anatomical(root)
        else:
            sl = sval.lower()
            hit = next((par for par in _pars(root)
                        if sl in protocol_name(par).lower()), None)
        if hit is not None:
            return hit
    return None


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
    from ._profiles import PROFILES
    p.add_argument("--profile", default=None, choices=sorted(PROFILES),
                   help="Bundled preset of plug-ins + acquisition params for a data "
                        "class (e.g. 'mouse'). CLI flags and --config override it; the "
                        "human default (no --profile) is unchanged.")
    p.add_argument("--animal", default="human", choices=["human", "mouse"],
                   help="Target species. 'human' (default) applies no overlay — the "
                        "paper defaults are untouched. 'mouse' natively applies the "
                        "mouse profile (equivalent to --profile mouse); the two flags "
                        "are mutually exclusive.")
    # Not required at the parser level — may be supplied by --config.
    p.add_argument("--subject-dir", type=Path, default=None,
                   help="Subject root directory (raw scans live here; outputs land "
                        "under here unless --derivatives-subdir nests them).")
    p.add_argument("--derivatives-subdir", default=None, metavar="SUBDIR",
                   help="Nest the derivatives tree under <subject-dir>/SUBDIR/ "
                        "instead of directly under <subject-dir>/. E.g. 'pbrain' "
                        "writes <subject-dir>/pbrain/derivatives/, keeping pipeline "
                        "outputs namespaced apart from the raw scans. Inputs are "
                        "still discovered from the real --subject-dir. Also settable "
                        "via the PBRAIN_DERIVATIVES_SUBDIR env var; unset (default) "
                        "preserves <subject-dir>/derivatives/.")
    p.add_argument("--dce", type=Path, default=None,
                   help="DCE 4-D series: a path, a filename in the subject dir, a "
                        "protocol-name substring (e.g. 'hperf'), or 'auto'.")
    p.add_argument("--t1", type=Path, default=None,
                   help="T1-weighted anatomical: path / filename / protocol substring "
                        "(e.g. 'T1W_3D_TFE') / 'auto'.")
    p.add_argument("--relax", "--ir", "--vfa", dest="ir", type=Path, default=None,
                   help="Baseline relaxometry series for T1/M0 fitting: an inversion-recovery "
                        "or variable-flip-angle acquisition (path / filename / 'auto'). "
                        "Aliases: --ir (inversion recovery), --vfa (variable flip angle).")
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
    p.add_argument("--tissue-roi", default=None,
                   help="Tissue ROI provider plug-in key. Default (auto): SynthSeg "
                        "parcellation when a --t1 scan and mri_synthseg are available, "
                        "otherwise a dependency-free whole-brain mask.")
    p.add_argument("--signal-to-conc", default="saturation_recovery",
                   help="Signal-to-concentration plug-in key.")
    p.add_argument("--normaliser", default="identity",
                   help="Curve-normaliser plug-in key (identity = raw curves; the "
                        "Patlak model does its own baseline shift and keeps the AIF raw).")
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
    p.add_argument("--aggregations", default="median_curve,voxelwise,parcel,region",
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
    p.add_argument("--dry-run", action="store_true",
                   help="Show the resolved pipeline plan and exit without computing.")
    p.add_argument("--mode", default="auto", choices=["auto", "verify", "manual"],
                   help="auto: run through (default) · verify: open a browser review at each "
                        "decision checkpoint (AIF, …) to confirm or nudge · manual: draw the "
                        "ROIs yourself. Reject a review to stop; ⇥ cycles it live.")
    p.add_argument("--assist", action="store_true",
                   help="Optional local-LLM help (metadata only, never the numbers): propose the "
                        "input series, explain a failure, summarise QC. Needs Ollama; no-ops without.")
    p.add_argument("--vision", action="store_true",
                   help="After the run, cross-check the AIF with the HF vision localiser "
                        "(SSS / carotids from the DCE max-projection). Advisory QC only — never "
                        "alters the run. Needs a vision backend; no-ops without.")
    p.add_argument("--quiet", action="store_true", help="Warnings/errors only.")
    p.add_argument("--verbose", action="store_true", help="Debug-level detail.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Also append all log records to this file.")

    p.add_argument("--flip-angle-deg", type=float, default=30.0)
    p.add_argument("--tr-s", type=float, default=0.01118)
    p.add_argument("--r1-per-s-mM", type=float, default=4.0)
    p.add_argument("--baseline-frames", type=int, default=5)
    p.add_argument("--dt-s", type=float, default=2.463)
    p.add_argument("--aif-blood-t1-ms", type=float, default=0.0,
                   help="Optional AIF flow correction: convert AIF voxels with this "
                        "fixed blood T1 (ms; 3 T ≈ 1600) instead of the per-voxel fit. "
                        "Default 0 = voxelwise fitted T1 everywhere.")
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
    """``pbrain run`` entry point — run the full pipeline on one subject.

    Parses ``argv`` into a :class:`~pbrain.core.config.Config`, builds the
    default stage pipeline, and runs it over the subject's scans. Returns
    the process exit code.
    """
    from pbrain._console import ensure_utf8_console
    ensure_utf8_console()

    parser = _build_parser()
    args = parser.parse_args(argv)

    # --animal is a species-level alias for --profile. 'mouse' selects the mouse
    # profile natively; 'human' (default) selects nothing, so the paper defaults
    # are untouched. The two flags are mutually exclusive to avoid a silent
    # override of an explicit --profile.
    if args.animal != "human":
        if args.profile is not None:
            raise SystemExit(
                "--animal and --profile are mutually exclusive; pass only one."
            )
        args.profile = args.animal

    # A --profile preset supplies defaults for a whole acquisition class (e.g.
    # mouse). Same merge rule as --config: applied only where the CLI is silent,
    # and BEFORE --config so precedence is CLI > --config > --profile > default.
    # A human run (args.profile is None) never enters here, so paper defaults hold.
    if args.profile is not None:
        from ._profiles import resolve_profile
        prof = resolve_profile(args.profile)
        prof_opts = prof.pop("opt", [])
        cli_dests = {tok.lstrip("-").replace("-", "_").split("=")[0]
                     for tok in argv if tok.startswith("--")}
        for dest, val in prof.items():
            if dest not in cli_dests:
                setattr(args, dest, val)
        args.opt = list(prof_opts) + list(args.opt)

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

    # Auto-detect scope: point `pbrain run` at a folder of subjects and it fans out
    # as a cohort; at one subject and it runs that. Explicit inputs or --config force
    # single-subject; the env guard stops a fanned-out child from re-detecting.
    if (not getattr(args, "dry_run", False)
            and getattr(args, "config", None) is None
            and not os.environ.get("_PBRAIN_COHORT_CHILD")
            and all(getattr(args, _k, None) in (None, "auto") for _k in ("dce", "t1", "ir"))):
        from pbrain.io import layout as _layout
        _res = _layout.resolve(args.subject_dir)
        if _res.kind == "unknown" and getattr(args, "assist", False):
            _res = _assist_layout(args.subject_dir)   # propose → confirm → freeze → re-resolve
        if _res.kind == "cohort":
            from pbrain.cli.cohort import run_layout as _run_layout
            return _run_layout(_res, args)

    # --dry-run / `pbrain plan`: show the resolved pipeline without touching scans.
    # Runs before input discovery and config building, so a plan is cheap and needs
    # no real data — just the flags that select each stage's plug-in.
    if getattr(args, "dry_run", False):
        from pbrain._ui import show_plan
        from pbrain.stages import default_stages as _ds
        diff = f" · diffusion {args.diffusion}" if args.diffusion else ""
        header = [
            ("subject", f"{args.subject_dir.name} · {args.subject_dir} · {args.animal}"),
            ("pipeline", f"aif {args.aif} · models {args.models}{diff} · {args.device}"),
        ]
        chosen = {
            "load": args.path_scheme, "t1_m0": args.t1m0,
            "signal_to_conc": args.signal_to_conc, "aif": args.aif,
            "tissue_roi": args.tissue_roi or "auto", "normalisation": args.normaliser,
            "kinetic": args.models, "diffusion": args.diffusion or "—",
            "summary": args.aggregations, "diagnostics": "auto",
        }
        show_plan(header, [(s.name, chosen.get(s.name, "")) for s in _ds()])
        return 0

    # --assist: a local model proposes the input-series mapping from the scan
    # headers (metadata only, never computation). It SUGGESTS; you confirm on a
    # TTY; the resolved paths get recorded in the manifest, so a re-run is
    # deterministic. Fills only inputs you left unset or as "auto".
    if getattr(args, "assist", False):
        from pbrain import _assist
        from pbrain.io.subject_discovery import series_table
        from pbrain._ui import console as _con
        _c = _con(stderr=False)
        if not _assist.available():
            from pbrain import _ollama
            _ollama.guide(_c)          # first-time onboarding: install / start / pull
        if not _assist.available():
            _c.print("  [pb.dim]○ --assist: no model yet — continuing without it[/]")
        else:
            _rows = series_table(args.subject_dir)
            if _rows:
                _c.print(f"  [pb.accent]▸ assist[/]  [pb.mut]{_assist.model()} · reading {len(_rows)} series[/]")
                _prop = _assist.identify_series(_rows)
                # show the human-readable protocol name, not the PAR file id; keep the
                # id + dynamics as a dim tag so it's still traceable
                _proto = {r["id"]: (r.get("protocol") or r.get("file") or r["id"]) for r in _rows}
                _by = {r["id"]: r["file"] for r in _rows}
                _dyn = {r["id"]: r.get("dynamics", "") for r in _rows}

                def _label(_id):
                    if not _id or _id not in _proto:
                        return "[pb.dim]— (none)[/]"
                    _tag = f"{_id}" + (f" · {_dyn[_id]} dyn" if _dyn.get(_id) else "")
                    return f"[pb.lite]{_proto[_id]}[/]  [pb.dim]{_tag}[/]"

                def _show(p):
                    _c.print(f"    [pb.ink]DCE[/]  [pb.accent]→[/]  {_label(p.get('dce'))}")
                    _c.print(f"    [pb.ink]T1 [/]  [pb.accent]→[/]  {_label(p.get('t1_anatomical'))}")
                    _irs = p.get("ir") or []
                    _c.print(f"    [pb.ink]IR [/]  [pb.accent]→[/]  {_label(_irs[0]) if _irs else '[pb.dim]— (none)[/]'}")
                    for _x in _irs[1:]:
                        _c.print(f"          {_label(_x)}")
                    if p.get("why"):
                        _c.print(f"    [pb.dim]{p['why']}[/]")

                _ok = True
                while _prop:
                    _show(_prop)
                    if not sys.stdin.isatty():
                        break
                    _ans = input("    accept · [e]dit · [n]o  › ").strip().lower()
                    if _ans in ("n", "no"):
                        _ok = False
                        break
                    if _ans and _ans[0] == "e":
                        _fb = input("    what's off? (e.g. 't1 should be the sagittal one; axt1w is a reformat')  › ").strip()
                        if _fb:
                            _c.print("    [pb.mut]re-reading with your correction…[/]")
                            _rev = _assist.revise_series(_rows, _prop, _fb)
                            if _rev:
                                _prop = _rev
                            else:
                                _c.print("    [pb.dim]couldn't revise — keeping the last mapping[/]")
                        continue
                    break   # empty / y / anything else → accept

                if _ok and _prop:
                    _dce, _t1 = _prop.get("dce"), _prop.get("t1_anatomical")
                    if (args.dce is None or str(args.dce).lower() == "auto") and _dce in _by:
                        args.dce = args.subject_dir / _by[_dce]
                    if (args.t1 is None or str(args.t1).lower() == "auto") and _t1 in _by:
                        args.t1 = args.subject_dir / _by[_t1]
                    if args.ir is None:
                        args.ir = Path("auto")
                else:
                    _c.print("    [pb.dim]kept your inputs / heuristic auto[/]")

    # Output nesting. By default the derivatives tree lands directly under
    # <subject-dir>/derivatives/. --derivatives-subdir (or the
    # PBRAIN_DERIVATIVES_SUBDIR env var) nests it under a subfolder, e.g.
    # "pbrain" -> <subject-dir>/pbrain/derivatives/, so pipeline outputs stay
    # namespaced apart from the raw scans. Inputs are still resolved against the
    # real --subject-dir below, so raw-scan discovery is unaffected.
    _subdir = args.derivatives_subdir
    if _subdir is None:
        _subdir = os.environ.get("PBRAIN_DERIVATIVES_SUBDIR", "")
    _subdir = str(_subdir).strip().strip("/")
    output_dir = args.subject_dir / _subdir if _subdir else args.subject_dir

    # ``{subject}`` in any input path is replaced with the subject dir name —
    # lets one config file drive a whole cohort whose inputs live elsewhere,
    # e.g. dce = "/data/{subject}/NIfTI/dce.nii.gz".
    subj_name = args.subject_dir.name
    for dest in ("dce", "t1", "ir", "dwi", "bvals", "bvecs"):
        v = getattr(args, dest, None)
        if v is not None and "{subject}" in str(v):
            setattr(args, dest, Path(str(v).replace("{subject}", subj_name)))

    # Resolve researcher-friendly inputs: a full path still works, but --dce /
    # --t1 / --ir may also be a filename, a protocol-name substring, or "auto"
    # (constant across subjects, unlike the per-subject scan-numbered paths).
    for dest, kind in (("dce", "dce"), ("t1", "t1"), ("ir", "ir")):
        v = getattr(args, dest, None)
        if v is None:
            continue
        resolved = _resolve_subject_input(str(v), args.subject_dir, kind)
        if resolved is None:
            if str(v).strip().lower() == "auto":
                setattr(args, dest, None)   # unresolved 'auto' → try the layout fallback below
                continue
            raise SystemExit(
                f"--{dest} {str(v)!r}: not an existing path, a filename in "
                f"{args.subject_dir}/, or a protocol name found there."
            )
        setattr(args, dest, resolved)

    # Layout fallback: fill still-unset inputs from the detected on-disk layout
    # (flat-NIfTI / BIDS single subjects; PAR/REC 'auto' is resolved above).
    if args.dce is None or args.t1 is None or args.ir is None:
        from pbrain.io import layout as _layout
        _lres = _layout.resolve(args.subject_dir)
        if _lres.kind == "subject":
            _linp = _layout.inputs_for(args.subject_dir, _lres.adapter)
            for _k in ("dce", "t1", "ir"):
                if getattr(args, _k) is None and _linp.get(_k):
                    setattr(args, _k, _linp[_k])

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
    from pbrain.core import resolve_device, configure_tf_device
    # auto_install: if a Metal backend can be provisioned *safely* (compatible TF),
    # do it rather than just warning; a too-new TF is detected and left untouched.
    resolved = resolve_device(args.device, auto_install=True,
                              log=lambda m: log.info("device: %s", m))
    if args.device != resolved:
        log.info("device: requested=%r → resolved=%r", args.device, resolved)
    log.info("device: using %r", resolved)
    if resolved != "cpu":
        configure_tf_device(resolved)

    # Tissue-ROI default: prefer a SynthSeg parcellation when it can actually run
    # (a --t1 scan is supplied and mri_synthseg is on PATH); otherwise fall back to
    # a dependency-free whole-brain mask. An explicit --tissue-roi is always honoured.
    tissue_roi_provider = args.tissue_roi
    if tissue_roi_provider is None:
        import shutil
        if args.t1 is not None and shutil.which("mri_synthseg"):
            tissue_roi_provider = "synthseg"
        else:
            tissue_roi_provider = "voxelwise"
            why = "no --t1 volume" if args.t1 is None else "mri_synthseg not on PATH"
            log.info("tissue-roi: %s -> whole-brain mask (pass --tissue-roi synthseg to force)", why)

    config = Config(
        t1_m0_fitter=args.t1m0,
        aif_extractor=args.aif,
        tissue_roi_provider=tissue_roi_provider,
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
        aif_blood_t1_ms=args.aif_blood_t1_ms,
        subject_id=args.subject_dir.name,
        data_root=args.subject_dir.parent,
        device=resolved,
        mode=args.mode,
        plugin_options=plugin_options,
    )

    stages = default_stages()
    pipeline = Pipeline(stages=stages,
                        path_scheme=PATH_SCHEMES[args.path_scheme])

    from pbrain._ui import make_run_reporter, RunReporter
    import time as _time

    diff = f" · diffusion {args.diffusion}" if args.diffusion else ""
    header = [
        ("subject", f"{args.subject_dir.name} · {args.subject_dir} · {args.animal}"),
        ("pipeline", f"{len(stages)} stages · aif {args.aif} · models {args.models}"
                     f"{diff} · {resolved}"),
        ("output", f"{output_dir}/derivatives"),
    ]
    reporter = make_run_reporter(header, mode=args.mode)
    active = isinstance(reporter, RunReporter)
    if not active:                       # plain (piped / --quiet) path, unchanged
        if output_dir != args.subject_dir:
            log.info("Running pipeline on %s  (outputs -> %s/derivatives)",
                     args.subject_dir, output_dir)
        else:
            log.info("Running pipeline on %s", args.subject_dir)

    t0 = _time.perf_counter()
    try:
        with reporter:
            records = pipeline.run(output_dir, config, force=args.force,
                                   reporter=reporter if active else None)
    except Exception as exc:
        # A stage failed. The cockpit already marked it ✗; show a clean, actionable
        # error instead of dumping a traceback (pass --verbose for the full trace).
        if getattr(args, "assist", False):
            _assist_explain(exc, args)
        from pbrain._ui import console as _console
        con = _console(stderr=True)
        con.print(f"\n  [pb.fail]✗ run failed[/]  [pb.ink]{exc}[/]")
        if getattr(args, "verbose", False):
            import traceback
            con.print(f"[pb.dim]{traceback.format_exc()}[/]")
        return 1

    if active:
        extras = [("models", args.models.replace(",", " · "))]
        if args.diffusion:
            extras.append(("diff", args.diffusion))
        extras.append(("levels", args.aggregations.replace(",", " · ")))
        from pbrain import _clock
        reporter.summary(max(0.0, _time.perf_counter() - t0 - _clock.paused_total()),
                         extras, str(output_dir))
    else:
        log.info("Stage completion:")
        for name, rec in records.items():
            log.info("  %-18s %d artefacts -> %s",
                     name, len(rec.output.artefacts), rec.manifest_path)

    if getattr(args, "assist", False):
        _assist_qc(records)
    if getattr(args, "vision", False):
        _vision_qc(output_dir)
    return 0


def _assist_layout(root):
    """Unknown on-disk layout + ``--assist``: let a local model *propose* how the
    folder maps to subjects and inputs (from names only), confirm/correct it, and
    **freeze** the result to ``pbrain.layout.toml`` so re-runs are deterministic.
    Returns the (re-)resolved layout."""
    from pbrain.io import layout as _layout
    from pbrain import _assist
    from pbrain._ui import console
    con = console(stderr=False)
    if not _assist.available():
        from pbrain import _ollama
        _ollama.guide(con)
    if not _assist.available():
        con.print("  [pb.dim]○ --assist: no model — can't propose a layout[/]")
        return _layout.resolve(root)

    tree = _layout.gather_tree(root)
    con.print(f"  [pb.accent]▸ assist[/]  [pb.mut]{_assist.model()} · reading the folder layout[/]")
    prop = _assist.propose_layout(tree)

    def _show(p):
        subs = p.get("subjects", [])
        con.print(f"    [pb.ink]{p.get('kind', 'subject')}[/]  [pb.mut]{len(subs)} subject(s)[/]")
        for s in subs[:12]:
            got = "  ".join(f"[pb.accent]{k}[/]" if s.get(k) else f"[pb.dim]{k}·—[/]"
                            for k in ("dce", "t1", "ir"))
            con.print(f"      [pb.lite]{s.get('dir', '.')}[/]   {got}")
        if len(subs) > 12:
            con.print(f"      [pb.dim]… and {len(subs) - 12} more[/]")
        if p.get("why"):
            con.print(f"    [pb.dim]{p['why']}[/]")

    while prop and prop.get("subjects"):
        _show(prop)
        if not sys.stdin.isatty():
            break
        ans = input("    freeze & use · [e]dit · [n]o  › ").strip().lower()
        if ans in ("n", "no"):
            return _layout.resolve(root)
        if ans and ans[0] == "e":
            fb = input("    what's off? (e.g. 'these are 2 subjects; dce is the perf/ folder')  › ").strip()
            if fb:
                con.print("    [pb.mut]re-reading with your correction…[/]")
                rev = _assist.revise_layout(tree, prop, fb)
                if rev:
                    prop = rev
            continue
        break

    if prop and prop.get("subjects"):
        dest = _layout.write_frozen(root, prop)
        con.print(f"    [pb.accent]●[/] froze layout → [pb.ink]{dest}[/]  "
                  f"[pb.dim](later runs read this — the model is not re-asked)[/]")
    return _layout.resolve(root)


def _assist_explain(exc: Exception, args) -> None:
    from pbrain import _assist
    if not _assist.available():
        return
    ctx = f"aif={args.aif} models={args.models} device={args.device} dce={args.dce}"
    txt = _assist.explain_error(type(exc).__name__, str(exc)[:400], ctx)
    if txt:
        from rich.panel import Panel
        from pbrain._ui import console
        console(stderr=True).print(Panel(txt.strip(), title="[pb.accent]assist · what went wrong[/]",
                                         title_align="left", border_style="pb.deep", padding=(0, 1)))


def _vision_qc(output_dir) -> None:
    """Advisory only: localise the AIF vessels from the DCE max-projection with the
    HF vision model and cross-check the CNN AIF the run actually used. Never touches
    the numbers or alters the run; silent no-op if no vision backend / no volume.

    The SSS line reports agreement with the CNN AIF (the CNN marks the sinus). The
    carotid lines report a confidence signal — peak enhancement relative to the SSS:
    a true ICA enhances far harder than the venous sinus on first pass, so a ratio
    near 1 means 'not clearly arterial, defer to manual review', not a confident hit."""
    from pbrain import _ollama
    from pbrain import aif_vision as V
    from pbrain._ui import console
    if not V.vlm_available():
        return
    conc = V.canonical_glob(output_dir, "concentration.nii.gz")
    if not conc:
        return
    mask = V.canonical_glob(output_dir, "aif_mask.nii.gz")
    repo = _ollama.recommend()["vision"]["repo"]
    con = console(stderr=True)
    con.print(f"  [pb.accent]▸ vision[/]  [pb.mut]{repo} · cross-checking AIF (advisory)…[/]")
    res = V.find_aif(conc, repo, str(output_dir), cnn_mask=mask)
    if not res:
        con.print("  [pb.dim]vision: no localisation returned.[/]")
        return
    sss = res.get("sss")
    sss_peak = sss["peak"] if sss else None
    lines = []
    for t in V.TARGETS:
        r = res.get(t)
        if not r:
            lines.append(f"[pb.dim]{t:6s} not found[/]")
            continue
        if t == "sss":
            d = r.get("dist_cnn")
            agree = "" if d is None else (f" · [pb.accent]{d} vox from CNN AIF ✓[/]" if d <= 6
                                          else f" · [pb.warn]{d} vox from CNN AIF[/]")
            lines.append(f"[pb.ink]{t:6s}[/] voxel {r['voxel']} · peak {r['peak']}{agree}")
        else:
            ratio = (r["peak"] / sss_peak) if sss_peak else 0.0
            tag = "[pb.accent]✓ arterial[/]" if ratio >= 2.0 else "[pb.warn]low-confidence (not clearly arterial)[/]"
            lines.append(f"[pb.ink]{t:6s}[/] voxel {r['voxel']} · peak {r['peak']} · {ratio:.1f}× SSS · {tag}")
    from rich.panel import Panel
    con.print(Panel("\n".join(lines), title="[pb.accent]vision · AIF cross-check[/]",
                    title_align="left", border_style="pb.deep", padding=(0, 1)))


def _assist_qc(records) -> None:
    import json
    from pbrain import _assist
    if not _assist.available():
        return
    facts = []
    for name, rec in records.items():
        try:
            m = json.loads(Path(rec.manifest_path).read_text())
        except Exception:
            continue
        qc = m.get("qc", {}) or {}
        md = m.get("metadata", {}) or {}
        bits = [f"status={qc.get('status', '?')}"]
        for src in (md, qc):
            for k, v in src.items():
                if isinstance(v, (int, float, str)) and k not in ("status",) and len(str(v)) < 40:
                    bits.append(f"{k}={v}")
        for msg in (qc.get("messages") or [])[:1]:
            bits.append("note=" + str(msg)[:80])
        facts.append(f"{name}: " + ", ".join(bits[:8]))
    txt = _assist.summarize_qc("\n".join(facts))
    if txt:
        from rich.panel import Panel
        from pbrain._ui import console
        console(stderr=True).print(Panel(txt.strip(), title="[pb.accent]assist · QC summary[/]",
                                         title_align="left", border_style="pb.deep", padding=(0, 1)))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
