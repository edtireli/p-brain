# p-Brain modular architecture

> **Status:** the `pbrain/` framework is now the codebase. The previous
> monolithic stack (`main.py` / `enumerator.py` / `models/` / `modules/`
> / `utils/`) has been removed from the working tree — it lives in git
> history only. This document describes the clean modular design; see
> [`ADDING_PLUGINS.md`](ADDING_PLUGINS.md) to extend it.

---

## Why this rewrite

The paper (Tireli et al., *p-Brain: Advanced Neuroimaging platform*)
claims a modular framework with stable file-based interfaces between
stages. The original code grew organically into the opposite: monolithic
`main.py` / `enumerator.py` scripts, two duplicate Tikhonov solvers,
kinetic models with incompatible APIs (Patlak / Tikhonov tuple wrappers
disagreed on argument order, gamma was class-based), global mutable
`settings.py` driving every stage by side-effect, and 1167 lines of
config glue. The rewrite replaces all of that with auto-discovered
single-file plug-ins and file-manifest stage interfaces.

The `pbrain/` package is a clean-room implementation that delivers what
the paper actually promised: nine substitutable stages, one shared
contract, auto-discovery, explicit configuration, and parity-tested
numerics.

## Nine plug-points, one contract

| Plug-point | Folder | Default | Plug-ins shipped |
|---|---|---|---|
| Input loaders | `pbrain/io/loaders/` | `nifti` | `nifti`, `parrec`, `dicom` (via dcm2niix) |
| Output path schemes | `pbrain/io/path_schemes/` | `bids_like` | `bids_like`, `legacy` |
| T1/M0 fitters | `pbrain/t1_m0/` | `inversion_recovery` | `inversion_recovery`, `vfa_spgr` |
| AIF/VIF extractors | `pbrain/aif/` | `deterministic` | `deterministic`, `from_file`, `cnn`*, `manual`* |
| Tissue ROI providers | `pbrain/tissue_roi/` | `synthseg` | `voxelwise`, `synthseg`, `fastsurfer`*, `manual`* |
| Signal → concentration | `pbrain/signal_to_conc/` | `saturation_recovery` | `saturation_recovery`, `vfa_spgr` |
| Curve normalisers | `pbrain/normalisation/` | `baseline_p95` | `baseline_p95` |
| Kinetic models | `pbrain/models/` | `patlak`, `tikhonov` | `patlak`, `tikhonov`, `extended_tofts`, `gamma`** |
| Analysis aggregators | `pbrain/aggregation/` | `voxelwise`, `parcel`, `region` | `voxelwise`, `parcel`, `region`, `slice_wise` |

\* *Stub: registered with the Plugin contract; `run()` raises
`NotImplementedError` with a pointer to the legacy module to port from.
These need GPU + real-data verification before they should be marked
production-ready.*

\** *`pbrain/models/gamma.py` is gitignored (and so are `egamma.py` /
`gamma_gain.py` if you drop them in). Auto-discovery picks them up
when present on disk; on a fresh clone from `main` they simply aren't
there and the registry omits them with zero special-casing.*

## The contract

Every plug-in is a frozen dataclass that satisfies one Protocol::

    @runtime_checkable
    class Plugin(Protocol):
        key: ClassVar[str]
        name: ClassVar[str]
        description: ClassVar[str]
        accepts: ClassVar[dict[str, type]]
        produces: ClassVar[dict[str, type]]

Each plug-point folder defines a sub-Protocol that refines `Plugin`
with a typed entry point — e.g. `KineticModel.fit(inputs: CurveInputs)
-> ModelResult`, `ImageLoader.load(path: Path) -> Series4D`.

Plug-ins expose a module-level `PLUGIN` attribute. The folder's
`__init__.py` is two lines::

    from pbrain.core import discover
    from .base import KineticModel
    REGISTRY = discover(__name__, __file__, expected_protocol=KineticModel)

`discover()` scans `*.py` in the folder (skipping `_*.py` and
`base.py`), imports each module, indexes whatever exposes `PLUGIN`,
runtime-checks it against the expected Protocol, and complains loudly
on duplicate keys. **There is no `register()` call, no manual list to
update, no dispatcher to edit.** Drop a file in → it appears in the
registry; delete or `.gitignore` it → it's gone.

## How outputs are dynamic per model

`ModelResult.maps` is `dict[str, np.ndarray]` — the keys are *the
model's choice*:

* Patlak fills `{"ki", "vb"}`.
* Tikhonov fills `{"cbf", "mtt", "cth", "lambda_opt"}`.
* Gamma fills its own set.

Downstream aggregators consume this dict by iterating its keys; they
never hard-code a parameter name. The paper's "dynamic number of
variables" requirement is met because the schema is the model's, not
the framework's.

## The orchestrator + stage manifests

Stages are small classes wired together by
`pbrain.core.Pipeline.run(subject_dir, config)`. Each stage:

1. Reads upstream `manifest.json` files (it never reads sibling files
   directly — paths come from the PathScheme + the upstream manifest).
2. Calls the plug-in selected by `config`.
3. Writes its outputs to disk and persists a fresh `manifest.json`
   listing them.

The default 9-stage pipeline lives in `pbrain.cli.stages.default_stages()`.
Custom pipelines pass a different stage list to `Pipeline`.

## Configuration

A single immutable `Config` dataclass (`pbrain.core.config.Config`)
holds every knob. Stages receive `config: Config` explicitly — no
`import settings`, no env-var side channels.

Per-plug-in options live in `config.plugin_options` keyed by
`"<plug-point>.<plugin-key>"`, e.g.::

    Config(plugin_options={
        "aif.deterministic": {"n_voxels": 64},
        "models.tikhonov":   {"n_lambdas": 201},
    })

The CLI accepts the same form via `--opt aif.deterministic.n_voxels=64`.

## Output layout (BIDS-like default)

    <subject>/derivatives/
      01_load/                         {dce.nii.gz, dce_time_s.npy, dce_meta.json, manifest.json}
      02_t1m0/inversion_recovery/      {t1_map_ms.nii.gz, m0_map.nii.gz, manifest.json}
      03_aif/deterministic/            {aif_signal.npy, aif_mask.nii.gz, aif.json, manifest.json}
      04_tissue_roi/synthseg/          {parcellation.nii.gz, labels.json, manifest.json}
      05_signal_to_conc/saturation_recovery/ {ct.nii.gz, manifest.json}
      06_normalisation/baseline_p95/   {aif_normalised.npy, ct_normalised.nii.gz, manifest.json}
      07_kinetic/
        patlak/
          voxelwise/                   {ki.nii.gz, vb.nii.gz, histogram_summary.json, manifest.json}
          parcel/                      {ki.csv, vb.csv, manifest.json}
          region/                      {ki.json, vb.json, manifest.json}
          slice_wise/                  {ki_slice.csv, vb_slice.csv, manifest.json}
        tikhonov/
          voxelwise/                   {cbf.nii.gz, mtt.nii.gz, cth.nii.gz, lambda_opt.nii.gz, …, manifest.json}
          parcel/                      ⟨same shape⟩
          region/                      ⟨same shape⟩
          slice_wise/                  ⟨same shape⟩
        gamma/                         ⟨only present if gamma.py is on disk⟩

The `legacy` PathScheme (opt-in via `--path-scheme legacy`) reproduces
the original `Analysis/Fitting/`, `Images/AI_patlak/`, … directory tree
for backward compatibility with the p-Brain Platform app.

## Validation universe

The MATLAB-reference, regu-toolbox, comparison scripts, and any other
parity-checking infrastructure live in `validation/` (gitignored). They
stay on disk for ongoing development but never reach the public branch.

## Branch strategy

* `main` — the legacy pipeline; ships the paper-published code as-is.
* `dev/modular-framework` — this work. Once parity tests pass against
  legacy on the validation dataset under `/Volumes/T5_EVO_EDT/data`,
  merge with `--no-ff` so the refactor is one reviewable square.

## Running it

Discover what's installed::

    python -m pbrain list

End-to-end run::

    python -m pbrain run \
        --subject-dir /path/to/sub-01 \
        --dce /path/to/dce.nii.gz \
        --t1 /path/to/t1.nii.gz \
        --ir /path/to/ir.nii.gz \
        --aif deterministic \
        --tissue-roi voxelwise \
        --models patlak,tikhonov \
        --aggregations voxelwise,parcel,region,slice_wise

Override any plug-in option::

    --opt aif.deterministic.n_voxels=64 \
    --opt models.tikhonov.n_lambdas=201

## Tests

    .venv/bin/python -m pytest tests/test_pbrain_*.py -v

Coverage:

* `test_pbrain_contracts.py` — discovery, Plugin/sub-Protocol
  conformance, key uniqueness, presence of expected default plug-ins.
* `test_pbrain_models_parity.py` — **the critical gate**: every
  ported model must match the legacy implementation to 1e-10 / 1e-12
  on synthetic phantoms.
* `test_pbrain_path_schemes.py` — deterministic path computation.
* `test_pbrain_loaders.py` — NIfTI round-trip on synthesised volumes.
* `test_pbrain_t1m0.py` — IR fit recovers known T1 within ms.
* `test_pbrain_signal_to_conc.py` — monotonicity and edge cases for
  paper Eq. 2.
* `test_pbrain_normalisation.py` — baseline subtract + p95 rescale.
* `test_pbrain_aggregation.py` — voxel/parcel/region/slice output
  shapes and contents.
* `test_pbrain_aif.py` — deterministic AIF picks high-peak voxels;
  stubs raise clear NotImplementedError.

## What still needs porting

* `pbrain/aif/cnn.py` — wire the two-stage CNN inference (slice
  classifier + per-slice U-Net) from `modules/AI_input_functions.py`.
  Needs the CNN weights in `AI/` and a GPU + real-data verification
  pass.
* `pbrain/aif/manual.py` — port the Tkinter/matplotlib drawing widgets
  from `modules/manual_tissue_roi.py` and adjacent files.
* `pbrain/tissue_roi/fastsurfer.py` — subprocess wrapper for the
  FastSurfer DL parcellation backend.
* `pbrain/tissue_roi/manual.py` — same drawing-widget port as above.

Each stub is registered, contract-conformant, and raises
`NotImplementedError` with a precise pointer to the legacy file to
port from.
