"""``python -m pbrain`` dispatcher."""

import sys

from .run import main as run_main


def _help() -> int:
    print(
        "usage: python -m pbrain <command> [options]\n\n"
        "commands:\n"
        "  run                 Run the DCE-MRI pipeline on a single subject.\n"
        "  run-cohort          Run the pipeline over many subjects (parallel, resumable).\n"
        "  list                Overview of registered plug-ins (every plug-point).\n"
        "  list <plug-point>   Detailed contract for one plug-point (e.g. `list models`).\n"
        "  check-deps          Verify third-party Python deps; offer to pip-install missing ones.\n"
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
    for label, reg in _registries():
        keys = ", ".join(sorted(reg.keys())) or "(none)"
        print(f"{label:18}  {keys}")
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


def _check_deps() -> int:
    from pbrain.cli._deps import check_and_install
    return check_and_install()


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        return _help()
    cmd = argv[0]
    rest = argv[1:]
    if cmd == "run":
        return run_main(rest)
    if cmd in ("run-cohort", "cohort"):
        from .cohort import main as cohort_main
        return cohort_main(rest)
    if cmd == "list":
        if rest:
            return _list_detail(rest[0])
        return _list_overview()
    if cmd == "check-deps":
        return _check_deps()
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
