"""``python -m pbrain`` dispatcher."""

import sys

from .run import main as run_main


def _help() -> int:
    print(
        "usage: pbrain <command> [options]\n\n"
        "commands:\n"
        "  run                 Run the DCE-MRI pipeline on a single subject (--assist for optional LLM help).\n"
        "  methods             Draft the Methods paragraph from a run's provenance (needs Ollama).\n"
        "  assist [--vision]   Set up the local model backend (install Ollama, pull models for your hardware).\n"
        "  plan                Show the resolved pipeline for a run without computing (run --dry-run).\n"
        "  layout <path>       Preview what's detected at a path: subject vs cohort, layout, and inputs.\n"
        "  run-cohort          Run the pipeline over many subjects (parallel, resumable).\n"
        "  list                Overview of registered plug-ins (every plug-point).\n"
        "  list <plug-point>   Detailed contract for one plug-point (e.g. `list models`).\n"
        "  theme [name]        Show or set the colour theme (clay, teal, green, red, …).\n"
        "  tone [single|two|three]  Shading of the banner brain glyph.\n"
        "  setup               Get everything ready to run: check + install deps, dcm2niix, FreeSurfer, GPU, Zenodo assets.\n"
        "  check-deps          Verify third-party Python deps; offer to pip-install missing ones.\n"
        "  fetch-weights       Download the CNN AIF weights from Zenodo (for the default 'cnn_sss_shifted' AIF).\n"
        "  fetch-data          Download the example test dataset from Zenodo.\n"
    )
    return 0


def _registries():
    from pbrain.io.loaders import REGISTRY as LOADERS
    from pbrain.io.path_schemes import REGISTRY as PATH_SCHEMES
    from pbrain.t1_m0 import REGISTRY as T1M0
    from pbrain.aif import REGISTRY as AIF
    from pbrain.tissue_roi import REGISTRY as TROI
    from pbrain.signal_to_conc import REGISTRY as STC
    from pbrain.normalisation import REGISTRY as NORM
    from pbrain.models import REGISTRY as MODELS
    from pbrain.diffusion import REGISTRY as DIFF
    from pbrain.aggregation import REGISTRY as AGG
    from pbrain.diagnostics import REGISTRY as DIAG
    return [
        ("loaders",          LOADERS),
        ("path_schemes",     PATH_SCHEMES),
        ("t1_m0",            T1M0),
        ("aif",              AIF),
        ("tissue_roi",       TROI),
        ("signal_to_conc",   STC),
        ("normalisation",    NORM),
        ("models",           MODELS),
        ("diffusion",        DIFF),
        ("aggregation",      AGG),
        ("diagnostics",      DIAG),
    ]


def _list_overview() -> int:
    from pbrain._ui import console
    con = console(stderr=False)
    con.print("  [pb.accent]▸ plug-points[/]  [pb.dim]— pbrain list <plug-point> for the full contract[/]\n")
    for label, reg in _registries():
        keys = ", ".join(sorted(reg.keys())) or "(none)"
        con.print(f"  [pb.accent]{label:<16}[/] [pb.mut]{keys}[/]")
    return 0


def _list_detail(plug_point: str) -> int:
    regs = dict(_registries())
    if plug_point not in regs:
        print(f"unknown plug-point: {plug_point!r}", file=sys.stderr)
        print(f"available: {', '.join(regs.keys())}", file=sys.stderr)
        return 2

    reg = regs[plug_point]
    if not reg:
        print(f"(no plug-ins registered under {plug_point!r})")
        return 0

    print(f"== {plug_point} ==\n")
    for key in sorted(reg.keys()):
        plug = reg[key]
        print(f"  {key}")
        if getattr(plug, "name", ""):
            print(f"    name        : {plug.name}")
        if getattr(plug, "description", ""):
            print(f"    description : {plug.description}")
        if plug_point in ("models", "diffusion"):
            outputs = getattr(plug, "outputs", ()) or ()
            units = getattr(plug, "units", {}) or {}
            primary = getattr(plug, "primary_map", None)
            print(f"    outputs     : {', '.join(outputs) if outputs else '(none)'}")
            if units:
                print(f"    units       : {dict(units)}")
            if primary:
                print(f"    primary_map : {primary!r}")
            from pbrain.diagnostics import REGISTRY as DIAG
            if callable(getattr(plug, "diagnose", None)):
                diag_src = "model.diagnose()"
            elif key in DIAG:
                diag_src = f"diagnostics.{key}"
            else:
                diag_src = "generic fallback"
            print(f"    diagnostic  : {diag_src}")
        if plug_point == "diagnostics":
            mk = getattr(plug, "model_key", "")
            if mk:
                print(f"    model_key   : {mk}")
        print()
    return 0


def _theme_cmd(rest) -> int:
    """`pbrain theme` lists the palettes with swatches; `pbrain theme <name>` sets
    it (persisted to ~/.config/pbrain/config.json). Clay is the default."""
    from pbrain._ui import console
    from pbrain import _palette as P
    con = console(stderr=False)
    if rest:
        if P.set_theme(rest[0]):
            con.print(f"  [pb.accent]●[/] theme set to [pb.ink]{rest[0]}[/]  [pb.dim](takes effect next command)[/]")
            return 0
        con.print(f"  [pb.fail]✗[/] unknown theme: [pb.ink]{rest[0]}[/]")
    cur = P.active_name()
    con.print("  [pb.accent]▸ themes[/]  [pb.dim]— pbrain theme <name> to set[/]\n")
    for name, (base, deep, lite) in P.PALETTES.items():
        (r, g, b), (dr, dg, db), (lr, lg, lb) = P.rgb(base), P.rgb(deep), P.rgb(lite)
        sw = (f"[rgb({dr},{dg},{db})]███[/][rgb({r},{g},{b})]███[/][rgb({lr},{lg},{lb})]███[/]")
        mark = "  [pb.accent](active)[/]" if name == cur else ""
        con.print(f"  {sw}  [pb.ink]{name}[/]{mark}")
    con.print()
    return 0


def _tone_cmd(rest) -> int:
    """`pbrain tone [single|two|three]` — shading of the banner brain glyph."""
    from pbrain._ui import console
    from pbrain import _palette as P
    con = console(stderr=False)
    if rest:
        if P.set_tone(rest[0]):
            con.print(f"  [pb.accent]●[/] brain tone set to [pb.ink]{rest[0]}[/]")
            return 0
        con.print(f"  [pb.fail]✗[/] unknown tone: [pb.ink]{rest[0]}[/]  [pb.dim](single · two · three)[/]")
    cur = P.active_tone()
    con.print("  [pb.accent]▸ tone[/]  [pb.dim]— pbrain tone <single|two|three>[/]")
    for t in P.TONES:
        con.print(f"    [pb.ink]{t}[/]" + ("  [pb.accent](active)[/]" if t == cur else ""))
    return 0


def _methods_cmd(rest) -> int:
    """`pbrain methods --subject-dir X` — draft the Methods sentences from a run's
    recorded provenance, via the optional local model (grounded, editable)."""
    import argparse
    import glob
    import json
    from pathlib import Path
    from pbrain import _assist
    from pbrain._ui import console
    p = argparse.ArgumentParser(prog="pbrain methods")
    p.add_argument("--subject-dir", required=True, type=Path)
    p.add_argument("--derivatives-subdir", default="")
    a = p.parse_args(rest)
    con = console(stderr=False)
    if not _assist.available():
        con.print("  [pb.fail]✗[/] methods needs a local model (Ollama running with a model).")
        return 1
    root = (a.subject_dir / a.derivatives_subdir / "derivatives") if a.derivatives_subdir else a.subject_dir / "derivatives"
    prov = None
    for mf in sorted(glob.glob(str(root / "**" / "manifest.json"), recursive=True)):
        try:
            d = json.loads(Path(mf).read_text())
            if d.get("config"):
                prov = d["config"]
                break
        except Exception:
            continue
    if not prov:
        con.print(f"  [pb.fail]✗[/] no run provenance under {root} — run pbrain on this subject first.")
        return 1
    txt = _assist.methods_text(json.dumps(prov, indent=1))
    if txt:
        from rich.panel import Panel
        con.print(Panel(txt.strip(), title="[pb.accent]methods (draft)[/]", title_align="left",
                        border_style="pb.deep", padding=(0, 1)))
        con.print("  [pb.dim]draft from provenance — verify against your acquisition before publishing.[/]")
    return 0


def _aif_locate_cmd(rest) -> int:
    """`pbrain aif-locate --subject-dir X` — vision-assisted AIF/VIF localisation
    on a run's concentration volume: propose SSS / R-ICA / L-ICA regions with a HF
    vision model, refine to the peak voxel, cross-check against the CNN."""
    import argparse
    from pathlib import Path
    from pbrain import aif_vision as V
    from pbrain import _ollama
    from pbrain._ui import console
    p = argparse.ArgumentParser(prog="pbrain aif-locate")
    p.add_argument("--subject-dir", required=True, type=Path)
    p.add_argument("--derivatives-subdir", default="")
    a = p.parse_args(rest)
    con = console(stderr=False)
    root = (a.subject_dir / a.derivatives_subdir / "derivatives") if a.derivatives_subdir else a.subject_dir / "derivatives"
    conc = V.canonical_glob(root, "concentration.nii.gz")
    if not conc:
        con.print(f"  [pb.fail]✗[/] no concentration volume under {root} — run pbrain on this subject first.")
        return 1
    vis = _ollama.recommend()["vision"]
    if not V.vlm_available():
        con.print("  [pb.warn]vision backend not installed.[/]")
        con.print(f"    install:  [pb.ink]{vis['install']}[/]")
        con.print(f"    model:    [pb.ink]{vis['repo']}[/]  [pb.dim](downloads from HuggingFace on first use)[/]")
        return 1
    con.print(f"  [pb.accent]▸ aif-locate[/]  [pb.mut]{vis['repo']}[/]")
    cnn_mask = V.canonical_glob(root, "aif_mask.nii.gz")
    res = V.find_aif(conc, vis["repo"], str(root), cnn_mask=cnn_mask)
    if not res:
        con.print("  [pb.fail]✗[/] the vision model returned no localisation.")
        return 1
    cnn = res.get("_cnn")
    for t in V.TARGETS:
        r = res.get(t)
        if r:
            con.print(f"    [pb.ink]{t}[/] {r['region']} · voxel {r['voxel']} · peak {r['peak']} · "
                      f"cluster {r['cluster']} · [pb.mut]dist→CNN {r.get('dist_cnn')}[/]")
        else:
            con.print(f"    [pb.dim]{t}: not found[/]")
    con.print(f"  [pb.dim]montage: {res.get('_montage')} · CNN peak voxel: {cnn}[/]")
    return 0


def _layout_cmd(rest) -> int:
    """`pbrain layout <path>` — show what p-Brain detects at a path: single subject
    or cohort, which layout (PAR/REC · flat-NIfTI · BIDS), and each subject's inputs.
    Read-only — a safe way to preview what `pbrain run <path>` would do."""
    import argparse
    from pathlib import Path
    from pbrain.io import layout as L
    from pbrain._ui import console
    p = argparse.ArgumentParser(prog="pbrain layout")
    p.add_argument("path", type=Path)
    a = p.parse_args(rest)
    con = console(stderr=False)
    res = L.resolve(a.path)
    if res.kind == "unknown":
        con.print(f"  [pb.warn]?[/] no recognised layout at [pb.ink]{a.path}[/] "
                  f"[pb.dim](tried PAR/REC · flat-NIfTI · BIDS)[/]")
        con.print("    [pb.dim]with a local model, `--assist` can propose one for an unfamiliar layout.[/]")
        return 1
    con.print(f"  [pb.accent]▸ layout[/]  [pb.ink]{res.kind}[/]  "
              f"[pb.mut]{res.adapter} · {res.n} subject(s)[/]")
    for s in res.subjects[:15]:
        inp = L.inputs_for(s, res.adapter)
        got = "  ".join((f"[pb.accent]{k}[/]" if inp.get(k) else f"[pb.dim]{k}·—[/]")
                        for k in ("dce", "t1", "ir"))
        con.print(f"    [pb.ink]{s.name}[/]   {got}")
    if res.n > 15:
        con.print(f"    [pb.dim]… and {res.n - 15} more[/]")
    return 0


def _check_deps() -> int:
    from pbrain.cli._deps import check_and_install
    return check_and_install()


def main() -> int:
    """Top-level ``pbrain`` entry point (the ``[project.scripts]`` target).

    Dispatches the first CLI argument to the matching sub-command
    (``run``, ``cohort``, ``list``, ``setup``, …). Returns the process
    exit code.
    """
    from pbrain._console import ensure_utf8_console
    ensure_utf8_console()

    from pbrain._banner import print_banner
    print_banner()

    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        return _help()
    cmd = argv[0]
    rest = argv[1:]
    if cmd == "run":
        return run_main(rest)
    if cmd == "plan":
        return run_main(rest + ["--dry-run"])
    if cmd in ("run-cohort", "cohort"):
        from .cohort import main as cohort_main
        return cohort_main(rest)
    if cmd == "list":
        if rest:
            return _list_detail(rest[0])
        return _list_overview()
    if cmd == "theme":
        return _theme_cmd(rest)
    if cmd == "tone":
        return _tone_cmd(rest)
    if cmd == "methods":
        return _methods_cmd(rest)
    if cmd == "assist":
        from pbrain import _ollama
        from pbrain._ui import console
        _ollama.guide(console(stderr=False), want_vision="--vision" in rest)
        return 0
    if cmd == "aif-locate":
        return _aif_locate_cmd(rest)
    if cmd == "layout":
        return _layout_cmd(rest)
    if cmd == "check-deps":
        return _check_deps()
    if cmd == "setup":
        from pbrain.cli._setup import setup
        return setup()
    if cmd in ("fetch-weights", "fetch-data"):
        from pbrain.cli._fetch import _main as fetch_main
        return fetch_main(rest, what=cmd)
    from pbrain._ui import console
    con = console(stderr=True)
    con.print(f"  [pb.fail]✗[/] unknown command: [pb.ink]{cmd}[/]  [pb.dim](try: pbrain help)[/]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
