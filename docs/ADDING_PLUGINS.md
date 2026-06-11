# Extending p-Brain

Everything in p-Brain is a plug-in. This guide shows how to add a new model,
a new method, a new pipeline stage, or a whole new plug-point — each with a
copy-paste template. The golden rule: **you add files, you never edit the
core.**

- [The universal contract](#the-universal-contract)
- [Add a kinetic model](#add-a-kinetic-model)
- [Add a diffusion model](#add-a-diffusion-model)
- [Add any other method (AIF, normaliser, …)](#add-any-other-method)
- [Add a diagnostic plot](#add-a-diagnostic-plot)
- [Add a pipeline stage](#add-a-pipeline-stage)
- [Add a whole new plug-point](#add-a-whole-new-plug-point)
- [Options & config](#options--config)
- [Verifying your plug-in](#verifying-your-plug-in)

---

## The universal contract

Every plug-in is a module that exports a module-level `PLUGIN` object with
five class attributes (`pbrain/core/plugin.py`):

```python
key:         ClassVar[str]              # unique registry name within the plug-point
name:        ClassVar[str]              # short human label
description: ClassVar[str]              # one-liner
accepts:     ClassVar[dict[str, type]]  # {input_name: type}
produces:    ClassVar[dict[str, type]]  # {output_name: type}
```

Each plug-point adds a small sub-Protocol with the typed entry method (`fit`,
`extract`, `aggregate`, …) and any extra attributes. Discovery (`discover()`)
scans the plug-point directory, imports every `*.py` that isn't `_`-prefixed
or `base.py`, and registers whatever exposes `PLUGIN`. Duplicate keys are a
hard error.

---

## Add a kinetic model

One file: `pbrain/models/my_model.py`.

```python
from dataclasses import dataclass
from typing import Any, ClassVar
import numpy as np
from .base import CurveInputs, ModelResult

@dataclass(frozen=True, slots=True)
class MyModel:
    key:         ClassVar[str] = "my_model"
    name:        ClassVar[str] = "My kinetic model"
    description: ClassVar[str] = "What it computes, in one line."
    accepts:     ClassVar[dict] = {"c_tissue": np.ndarray, "c_input": np.ndarray, "t_s": np.ndarray}
    produces:    ClassVar[dict] = {"p1": np.ndarray, "p2": np.ndarray}
    outputs:     ClassVar[tuple] = ("p1", "p2")            # MUST match maps keys
    units:       ClassVar[dict] = {"p1": "mL/100g/min", "p2": "fraction"}

    # Optional extras (sensible defaults if omitted):
    primary_map: ClassVar[str] = "p1"        # which map the voxel diagnostic samples

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        c_t = np.asarray(inputs.c_tissue, dtype=float)   # (T,) or (T, V)
        c_a = np.asarray(inputs.c_input,  dtype=float)   # (T,)
        t   = np.asarray(inputs.t_s,      dtype=float)   # (T,)
        single = c_t.ndim == 1
        if single:
            c_t = c_t.reshape(-1, 1)

        # ... your maths → p1, p2 of shape (V,) ...
        p1 = ...; p2 = ...

        if inputs.mask is not None:           # voxelwise: NaN outside the brain
            m = np.asarray(inputs.mask, bool).ravel()
            p1 = np.where(m, p1, np.nan); p2 = np.where(m, p2, np.nan)

        maps = {"p1": np.asarray(p1[0]) if single else p1,
                "p2": np.asarray(p2[0]) if single else p2}
        return ModelResult(maps=maps, units=dict(self.units))

    # Optional: lets the generic diagnostic overlay a fitted curve.
    def predict(self, params: dict[str, float], c_input, t_s) -> np.ndarray:
        ...

PLUGIN = MyModel()
```

Contract rules enforced at runtime by `KineticStage`:
- `set(result.maps) ⊇ set(self.outputs)` — every declared output must be
  produced (extra maps, e.g. optional uncertainty, are allowed).
- `c_tissue` is `(T,)` for an ROI-mean fit or `(T, V)` for voxelwise; handle
  both. `inputs.mask` (shape `(V,)`) selects voxels to fit.

Run it: `python -m pbrain run --models my_model,patlak ...`. It aggregates at
every level and gets voxel/tissue/parcel diagnostics for free.

---

## Add a diffusion model

Identical pattern in `pbrain/diffusion/my_dmodel.py`, against
`pbrain/diffusion/base.py`:

```python
from .base import DWIInputs, DiffusionResult
# inputs.signal (X,Y,Z,N), inputs.bvals (N,), inputs.bvecs (N,3),
# inputs.affine (4,4), inputs.mask (X,Y,Z)
def fit(self, inputs: DWIInputs, **opts) -> DiffusionResult:
    ...
    return DiffusionResult(maps={...}, units={...})
```

Gate on data requirements inside `fit` (e.g. raise if single-shell). The
DiffusionStage resamples scalar maps to the parcellation grid automatically.

---

## Add any other method

AIF extractor, T1/M0 fitter, normaliser, signal-to-conc, tissue ROI,
aggregator, loader, path scheme — each is a directory under `pbrain/` with a
`base.py` Protocol. Open the existing plug-ins for the exact entry signature,
then drop a file exporting `PLUGIN`. Example — a new AIF method:

```python
# pbrain/aif/my_aif.py
from .base import AIFExtractor, InputFunction
@dataclass(frozen=True, slots=True)
class MyAIF:
    key: ClassVar[str] = "my_aif"; name = ...; description = ...
    accepts = {"dce_data": np.ndarray}; produces = {"input_function": InputFunction}
    def extract(self, dce_data, t_s, dce_affine, **opts) -> InputFunction:
        ...
PLUGIN = MyAIF()
```

`python -m pbrain run --aif my_aif ...`.

---

## Add a diagnostic plot

A model gets a diagnostic three ways, most-specific first:

1. Define `diagnose(self, ctx)` on the model itself (single-file).
2. Drop `pbrain/diagnostics/<model_key>.py` with `model_key = "<key>"` and a
   `plot(self, ctx)` method.
3. Do nothing — the **generic fallback** renders AIF + Cₜ + a parameter table.

`ctx` is a `DiagnosticContext` (`c_tissue`, `c_input`, `t_s`, `out_path`,
`model_opts`, `label`). Write a PNG to `ctx.out_path`. The diagnostic stage
calls it for the representative voxel, each tissue class, and each parcel.

---

## Add a pipeline stage

Stages are a plug-point too. Drop `pbrain/stages/my_stage.py`:

```python
from dataclasses import dataclass
from pbrain.core import StageContext, StageOutput

@dataclass
class MyStage:
    name: str = "my_stage"                       # manifest name it produces
    requires: tuple = ("kinetic", "tissue_roi")  # upstream stage names it reads
    plugin_key: str = "my_stage"

    def run(self, ctx: StageContext) -> StageOutput:
        kinetic = ctx.upstream_manifests["kinetic"]   # parsed Manifest
        # ... read upstream outputs, do work, write files ...
        return StageOutput(artefacts={"my_output": path})

PLUGIN = MyStage()
```

`resolve_pipeline()` topologically sorts by `requires`, so `my_stage` runs
after `kinetic` and `tissue_roi` automatically — no list to edit. Confirm with
`python -m pbrain list` (stages appear) and inspect the resolved order.

---

## Add a whole new plug-point

Rarely needed, but turnkey. Create `pbrain/<plugpoint>/`:

```
pbrain/myplug/
  base.py        # @runtime_checkable Protocol + Inputs/Result dataclasses
  __init__.py    # REGISTRY = discover(__name__, __file__, expected_protocol=MyProto)
  default.py     # first plug-in exporting PLUGIN
```

`__init__.py` is two lines:

```python
from pbrain.core import discover
from .base import MyProto
REGISTRY = discover(__name__, __file__, expected_protocol=MyProto)
```

Add a `Stage` (above) that consumes the registry, and it's wired in. To expose
it on the CLI, add the registry to `pbrain/cli/__main__.py`'s `_registries()`
so `pbrain list` shows it.

---

## Options & config

There is exactly **one** place for plug-in parameters: `plugin_options`, keyed
`"<plug-point>.<plugin>.<opt>"`. Set them on the CLI or in a config file — never
add fields to `core/Config`.

```bash
--opt models.my_model.alpha=2.5
```
```toml
[options]
"models.my_model.alpha" = 2.5
```

Inside `fit`, they arrive as `**opts` (`opts["alpha"]`). Provide defaults via
`opts.get("alpha", 2.5)`.

---

## Verifying your plug-in

```bash
python -m pbrain list <plug-point>        # confirm it's discovered, see its contract
python -m pbrain.demo --clean             # full-pipeline smoke test on a phantom
python -m pytest tests/ -q                # the suite
```

A good plug-in PR adds a test in `tests/test_pbrain_*.py` asserting the
contract (declared outputs == produced maps) and a sanity check on a synthetic
curve. See `tests/test_pbrain_contracts.py` for the pattern.
```
