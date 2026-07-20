<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/p-brain.gif" alt="p-Brain" width="680"/>
</p>

<p align="center">
  <a href="https://github.com/edtireli/p-brain/actions/workflows/ci.yml"><img src="https://github.com/edtireli/p-brain/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-D97757?style=flat-square" alt="Python 3.10–3.12"/>
  <img src="https://img.shields.io/badge/license-MIT-D97757?style=flat-square" alt="MIT"/>
  <img src="https://img.shields.io/badge/macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-D97757?style=flat-square" alt="macOS · Linux · Windows"/>
</p>

<p align="center">
  Automated quantitative DCE-MRI of cerebral perfusion, microvasculature, and
  blood–brain-barrier permeability. Point it at a subject's scans and it produces
  the full derivatives tree — T1/M0 mapping, arterial-input-function extraction,
  signal-to-concentration, and pharmacokinetic and diffusion modelling — as
  voxel-, tissue-, and parcel-level maps. No notebook, GUI, or server.
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#example-run">Example run</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#data-layouts">Data layouts</a> ·
  <a href="#commands">Commands</a> ·
  <a href="#assist">Assist</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#principles">Principles</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center"><img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/divider.svg" width="520" alt=""/></p>

**p-Brain** (the *p* is for **p**erfusion and **p**ermeability) is a
cross-platform command-line tool for quantitative DCE-MRI. It serves two
purposes. As shipped, it is a validated, ready-to-run pipeline you point at real
scanner data to get quantitative maps (Ki, CBF, MTT, CTH, Ktrans, FA, and
others). It is also a framework you extend: each step — kinetic model, AIF,
segmentation backend, or a whole stage — is a self-contained plug-in, so adding
your own means writing a single file with no changes to the core. Drop a model
into `pbrain/models/`, call it with `--models yourmodel`, and it runs on every
subject, is aggregated to every anatomical level, is written as NIfTI/CSV/JSON,
and is given diagnostics automatically.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Install

```bash
pip install p-brain
pbrain --help
```

Optional extras pull in heavier dependencies only when you need them:

```bash
pip install "p-brain[cnn]"        # TensorFlow — the CNN arterial-input-function (default AIF)
pip install "p-brain[metal]"      # Apple-Silicon GPU for the TF stages (pins TF 2.16 + tensorflow-metal)
pip install "p-brain[diffusion]"  # dipy — the diffusion track (DTI, DKI, CSD, tractography)
pip install "p-brain[dicom]"      # pydicom — DICOM input
pip install "p-brain[all]"        # everything at once
```

On Apple Silicon, `[metal]` is how the CNN AIF and SynthSeg run on the GPU —
Apple's `tensorflow-metal` plugin only supports TensorFlow ≤ 2.16, so this pins
the matched pair. Install it into a **fresh** environment (a newer TensorFlow
already present can't be safely downgraded in place). On Linux, GPU works through
`[cnn]` with a CUDA build of TensorFlow. Without a GPU backend, `--device mps/cuda`
cleanly falls back to CPU (identical results, only slower).

Runs on Linux, macOS, and Windows with Python 3.10–3.12. From a clone,
`pip install -e ".[dev]"` installs it in editable mode.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Quickstart

```bash
pbrain setup            # interactive: install extras, offer to fetch weights and data
pbrain fetch-weights    # CNN weights           (Zenodo 10.5281/zenodo.15697443)
pbrain fetch-data       # example subject sub-01 (Zenodo 10.5281/zenodo.20826857, ~99 MB)
```

`pbrain fetch-data` downloads a real subject and prints a ready-to-run,
weights-free command. Then, on your own data:

```bash
# One flag on raw Philips PAR/REC: inputs are auto-discovered by protocol name
pbrain run --subject-dir /data/20230403x2

# A whole study: each sub-directory is a subject, run in parallel
pbrain run-cohort --cohort /data/patients /data/controls --workers 4
```

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Try it: the example subject

The quickest way to confirm _p_-Brain works end-to-end, on **Linux, macOS, or
Windows**, with no CNN weights and no FreeSurfer or SynthSeg. The example ships
its own AIF curve and parcellation, so nothing extra is downloaded.

```bash
pip install p-brain
pbrain fetch-data          # downloads sub-01 (~99 MB), then prints the exact run command
```

`pbrain fetch-data` locates the data and prints a ready-to-run, weights-free
command with the correct paths for your machine, formatted for your shell (a
single line on Windows, so it pastes into PowerShell as-is). Copy, paste, run.
It has the form:

```bash
pbrain run \
  --subject-dir <data>/sub-01 \
  --dce  <data>/sub-01/sub-01_dce.nii.gz  --relax <data>/sub-01/sub-01_ir.nii.gz \
  --aif  curve_file      --opt aif.curve_file.curve_path=<data>/sub-01/sub-01_aif.npy \
  --tissue-roi preloaded --opt tissue_roi.preloaded.parcellation_path=<data>/sub-01/sub-01_parcellation.nii.gz \
  --models patlak,tikhonov --aggregations median_curve,region,parcel,voxelwise
```

`--dce` and `--relax` may be omitted — the subject directory is auto-discovered
— but they are spelled out here so the command is unambiguous about what it read.

Results are written under `sub-01/derivatives/`. Compare
`07_kinetic/patlak/region/ki.csv` (BBB Ki and vb) and
`07_kinetic/tikhonov/region/cbf.json` (CBF and MTT) against the bundled
`expected_outputs/`. The values should agree to within about 2 percent.

> **Windows.** The command runs the same way in PowerShell or `cmd`. Put it on
> one line, or replace each trailing backslash with a backtick:
>
> ```powershell
> pbrain run --subject-dir data\sub-01 `
>   --dce data\sub-01\sub-01_dce.nii.gz --relax data\sub-01\sub-01_ir.nii.gz `
>   --models patlak,tikhonov --aggregations median_curve,region,parcel,voxelwise
> ```
>
> You do not need dcm2niix, FreeSurfer, or the CNN weights to run the example.

**No download at all.** `python -m pbrain.demo` synthesises a small phantom and
runs the entire pipeline in seconds — a self-contained check that your install
works, on any operating system.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Example run

On an interactive terminal, `pbrain run` opens a live cockpit — the brain draws
itself in, then an analysis panel tracks the pipeline while log lines scroll
above it and a status line shows where it is:

```
  ⢀⣴⣿⣿⣿⣷⣦⡀
  ⠈⢻⣿⣿⣿⣿⣿⠿   p-Brain
  ⠀⠀⠉⠉⠹⣿⠟⠀   perfusion & permeability

  reading DCE … dcm2niix
  DCE shape · 256×256×10 · 250 frames
  fitting T1 map … median 1642 ms
╭──────────────────────── analysis ────────────────────────╮
│  ✓ load            DCE 256×256×10·250, IR 8 TIs           │
│  ✓ t1_m0           T1 median 1642 ms                      │
│  ✓ signal_to_conc  saturation_recovery                    │
│  ◆ aif             extracting … ROI 704 vox · peak 3.2 mM │
│  ○ tissue_roi      synthseg                               │
│  ○ kinetic         patlak · tikhonov                      │
╰───────────────────────────────────────────────────────────╯
 ⠴ p-Brain · aif · 22s · CNN rICA, 3 slices        auto ⇥
```

Piped or redirected (`| tee`, `> log`, `--quiet`), the cockpit is suppressed and
you get plain, greppable log lines instead — the numeric results are identical
either way.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> How it works

Ten self-contained stages, each cached to the derivatives tree and re-runnable in
isolation:

```
 load ──▶ t1_m0 ──▶ signal_to_conc ──▶ aif ──▶ tissue_roi ──▶ normalisation
   │      (T1/M0)   (signal → mM)     (CNN /   (SynthSeg      (baseline,
   │                                   auto)    parcels)       Gd leak)
   ▼
 kinetic ──▶ diffusion ──▶ summary ──▶ diagnostics
 (patlak,    (DTI/DKI/     (voxel ·    (per-map QC
  tikhonov,   CSD, tracts)  tissue ·    overlays)
  tofts…)                   parcel)
```

Every stage is a plug-in resolved at run time; `pbrain list` shows all
plug-points, `pbrain list models` drills into one. Results are written at three
levels (voxel, tissue-region, parcel) in NIfTI, CSV, and JSON, with a diagnostics
montage per map.

**Inputs are normalised first.** The `load` stage converts whatever you point it at
— Philips PAR/REC or DICOM (via `dcm2niix`), NIfTI passed straight through — into the
4-D NIfTI the rest of the pipeline runs on. Every downstream stage sees the same
canonical volume, so the analysis is source-format-agnostic.

<p align="center">
  <img src="https://raw.githubusercontent.com/edtireli/p-brain/main/docs/img/output_maps_voxel.png" alt="Voxelwise Ki, vb, CBF, MTT and CBV from one automated pbrain run" width="760">
</p>
<p align="center"><sub>Voxelwise K<sub>i</sub>, v<sub>b</sub>, CBF, MTT and CBV from a single automated <code>pbrain run</code>.</sub></p>

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Data layouts

`pbrain run <path>` works out **what you pointed it at** — one subject or a whole
folder — and **which on-disk convention** it follows, then runs accordingly:

```bash
pbrain run /data/20230403x2        # one subject → runs it
pbrain run /data/patients          # a folder of subjects → fans out as a cohort
pbrain layout /data/patients       # preview what it detects (read-only, no run)
```

Three layouts are recognised out of the box:

| layout | a subject looks like | inputs resolved from |
|---|---|---|
| **PAR/REC** | Philips `*.PAR` with an `hperf*` DCE | protocol names |
| **flat-NIfTI** | `dce.nii.gz` (+ optional `t1`, `ir`) | file names |
| **BIDS** | `dataset_description.json` · `sub-*` · `anat/*_T1w` | BIDS entities |

Detection is deterministic and tries *single-subject first*, so a subject you've
already run once (it has a `pbrain/` output tree) isn't read as a one-subject
cohort.

**Don't know the layout, or which file is which? `--assist` works it out.** Point
it at a folder no built-in convention recognises and a local model reads the *file
tree* — names only, no pixel data — to propose which folders are subjects and
which files are the DCE, T1, and IR. You accept the proposal or correct it in
plain words ("the T1 is the MP-RAGE, not the FLAIR"). The confirmed layout is
**frozen to `pbrain.layout.toml`** at the data root, so later runs read it directly
— point-at-anything convenience with a reproducible pipeline. More in
[Assist](#assist).

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Commands

| command | description |
|---|---|
| `pbrain run <path>` | run one subject, or a folder of them — auto-detected |
| `pbrain layout <path>` | preview the detected layout (subject vs cohort, inputs) |
| `pbrain cohort --cohort A B` | run a study explicitly, in parallel (`--workers N`) |
| `pbrain plan …` | show the resolved pipeline plan without computing |
| `pbrain list [models\|aif\|…]` | all plug-points, or one in detail |
| `pbrain setup` | detect tooling, install extras, fetch weights/data |
| `pbrain fetch-weights` · `fetch-data` | download the CNN weights · the example subject |
| `pbrain methods --subject-dir X` | draft a Methods paragraph from a run's provenance |
| `pbrain assist` | set up the optional local model backend (Ollama) |
| `pbrain theme <name>` · `tone <n>` | banner/log palette · one-, two-, or three-tone brain |
| `pbrain check-deps` | verify the numeric core and report optional extras |

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Assist

Running a subject usually means knowing which series is the DCE, which is the T1,
where the IR sits, and how your folders are arranged. **`--assist` lifts that
requirement** — an optional local-model layer that reads your acquisition
parameters and file tree and proposes the mapping, so you can run a subject, or a
whole unfamiliar archive, without hand-writing a single path.

```bash
pbrain run --subject-dir X --assist
```

What it does:

- **Finds your inputs for you.** From the scan headers it proposes which series is
  the DCE, T1, and IR; for a folder no built-in layout recognises, it reads the
  file tree (names only) and proposes which folders are subjects and which files
  are each input. You **accept it, or correct it in plain words** — "the T1 is the
  MP-RAGE" — and it re-proposes. A confirmed layout is frozen to
  `pbrain.layout.toml`, so later runs read it directly.
- **Explains a failure.** When a stage errors, it turns the traceback into a
  plain-language cause and a concrete fix.
- **Writes the boilerplate.** It summarises each stage's QC in a sentence and
  drafts a Methods paragraph from the run's provenance, ready to edit for a paper.

Two things keep it safe to leave on:

- **It reads text, not data.** The model sees scan headers, run provenance, QC
  statistics, and error messages — not voxel values — and it computes no result.
  The maps come from the deterministic pipeline. A suggestion that would change a
  run, such as which series is the DCE, takes effect only after you confirm it, and
  is written to the manifest so a re-run reproduces the same numbers.
- **It's local and opt-in.** Assist talks only to a model running on your machine
  via [Ollama](https://ollama.com). With none configured, the assist features stay
  off and the run is unchanged. First use walks you through installing Ollama and
  pulling a model sized to your hardware, and asks before it downloads anything.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Live controls

During an interactive run:

- **Review mode.** `--mode auto` (default) runs straight through. `--mode verify`
  opens a **browser review** at each decision checkpoint so you can confirm the
  suggested choice; `--mode manual` additionally lets you correct it. The
  checkpoints:
  - **AIF** — the input-function curve and its vessel ROI on the DCE slice (with a
    motion-correction toggle and the candidate vessels); confirm, drag the max
    voxel, or draw the ROI yourself.
  - **Baseline** — the pre-contrast baseline point on the first-peak region of the
    mean curve; confirm or drag it.
  - **Tissue segmentation** — the parcellation mask across slices; confirm, or draw
    exclusion regions to cut artefacts out of the tissue mask.
  - **Per kinetic model** — the model's own plot (Patlak plot; tissue-curve fit for
    the residue/compartment models) with its parameter values. In manual mode some
    models expose editable fit parameters (e.g. Patlak's fit-window and regression)
    that **re-fit live** on confirm. Any model gains a review for free by defining
    `review()`.
  - **Diffusion** — a scalar-map summary (median FA/MD/… + the primary map's central
    slice) for every diffusion model.

  Reject any review to stop the run; **⇥ Tab** cycles the mode live, shown in the
  status line, and the run timers pause while a review is open. Everything is served
  locally (no data leaves your machine) and your choice is recorded in the manifest,
  so a re-run reproduces it.
- **Ctrl-C** stops cleanly at any point; completed stages are cached, so the next
  run picks up where it left off.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Configuration

Flags override a config file, which overrides defaults.

```bash
pbrain run --config study.toml --subject-dir X
pbrain run --subject-dir X --models patlak,tikhonov --opt models.patlak.regression=ols
```

`--config` takes TOML (built-in) or YAML (`pip install pyyaml`). Any plug-in
option is settable with `--opt <plugin>.<key>.<opt>=<value>` or in the file.
Acquisition parameters (flip angle, TR) are read from the scan sidecar when
present and can be overridden per run.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Add your own

Every plug-point (kinetic models, AIF extractors, signal-to-concentration,
segmentation backends, diffusion models, whole stages) is a single file that
registers itself. A kinetic model is a dataclass declaring what it `accepts`,
what it `produces`, and an `extract`/`fit` method:

```python
# pbrain/models/my_model.py — then: pbrain run --models my_model
@dataclass(frozen=True, slots=True)
class MyModel:
    key = "my_model"
    produces = {"ki_map": np.ndarray}
    def fit(self, conc, aif, t_s, **opts) -> dict: ...
```

It is then aggregated to every level, written in every format, and given
diagnostics with no further wiring.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Models

Set any option with `--opt models.<key>.<opt>=<value>`. Defaults are what you get
without setting anything.

| model | produces | notes |
|---|---|---|
| `patlak` | **Ki**, **vp** | Patlak graphical analysis; robust Huber slope, smart tail detection |
| `tikhonov` | **CBF**, **MTT**, **CTH** | regularised residue deconvolution; GCV / L-curve / evidence λ selection |
| `extended_tofts` | **Ktrans**, **ve**, **vp**, **kep** | constrained Levenberg–Marquardt; no tuning for the default fit |

The list is meant to be extended — see [Add your own](#add-your-own). The
diffusion track (`--diffusion`) adds FA, MD, and tractography via dipy.

### Aggregation levels

Every model is fitted once and then summarised at whichever levels you ask for
with `--aggregations` (comma-separated, any combination):

| level | writes | use it for |
|---|---|---|
| `voxelwise` | one NIfTI per output map | maps, figures, further voxel analysis |
| `parcel` | one CSV per map, one row per parcel label | per-structure tables |
| `region` | parcels collapsed into broader regions | the headline GM / WM / cerebellum numbers |
| `median_curve` | one fit of the pooled ROI curve | the article's ROI-curve method; less noise-sensitive than averaging voxel fits |
| `slice_wise` | per-slice distributions | slice-direction trends and QC (paper Fig. 7) |

`pbrain list` prints every registered plug-in — models, aggregations, AIF
extractors, and the rest — for the version you actually have installed.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Principles

<img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/dot.svg" width="14" alt=""/> **Plug-in architecture.** Kinetic models, AIF extractors, signal-to-concentration methods, segmentation backends, and stages are self-registering modules. Adding one is a single file; the core is unchanged.

<img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/dot.svg" width="14" alt=""/> **The optional model reads text, not data.** Input mapping, QC summaries, methods text, and error explanations come from scan headers and run provenance, not voxel values, and the model computes no result. A suggestion that would alter a run takes effect only once you confirm it, and is recorded in the manifest.

<img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/dot.svg" width="14" alt=""/> **Runs are reproducible.** Inputs, plug-in choices, and parameters are recorded per run; an `--assist`-resolved layout is written to `pbrain.layout.toml`. The same inputs and settings produce the same outputs.

<img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/dot.svg" width="14" alt=""/> **One input representation.** `load` converts PAR/REC, DICOM, or NIfTI to a 4-D NIfTI; every downstream stage operates on that volume.

<img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/dot.svg" width="14" alt=""/> **Output is separable from computation.** The interactive cockpit is TTY-gated; a redirected or `--quiet` run emits plain text with identical numeric results.

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Requirements

- Linux, macOS, or Windows with Python 3.10–3.12
- `dcm2niix` on PATH for PAR/REC and DICOM conversion
- Optional: TensorFlow (`[cnn]`), dipy (`[diffusion]`), Ollama (assist)

## <img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/mark.svg" width="20" alt=""/> Citation

If p-Brain contributes to your work, please cite the accompanying paper (Tireli
et al.) and this repository. See
[`LICENSE`](https://github.com/edtireli/p-brain/blob/main/LICENSE) for terms.

<p align="center"><img src="https://cdn.jsdelivr.net/gh/edtireli/p-brain@main/assets/divider.svg" width="520" alt=""/></p>

<p align="center">
  MIT · Built by <b>Edis Devin Tireli</b> · Functional Imaging Unit, Rigshospitalet · University of Copenhagen
</p>
