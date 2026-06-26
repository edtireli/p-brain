# Architecture (for contributors)

This page is the orientation for anyone extending p-Brain. The two
in-repo reference documents go deeper:
[`ARCHITECTURE.md`](https://github.com/edtireli/p-brain/blob/main/docs/ARCHITECTURE.md)
(full design rationale + output layout) and
[`ADDING_PLUGINS.md`](https://github.com/edtireli/p-brain/blob/main/docs/ADDING_PLUGINS.md)
(copy-paste plug-in templates).

## The shape of the system

p-Brain is a **pipeline of nine substitutable stages** connected by
**file-based manifests**. Each stage reads its upstream stages'
`manifest.json` files, calls one plug-in selected by the run `Config`,
writes its outputs to disk, and persists a fresh `manifest.json`. Stages
never read sibling files directly — paths always come from the
`PathScheme` plus the upstream manifest. That is the paper's "stable
file-based interface" promise, and it is what lets any single stage be
swapped, re-run, or inspected in isolation.

```
load -> t1_m0 -> signal_to_conc -> aif -> tissue_roi
     -> normalisation -> kinetic -> (diffusion) -> summary -> diagnostics
```

`resolve_pipeline()` topologically sorts stages by their `requires`
tuples, so adding a stage that declares its upstream dependencies wires
itself into the right place with no list to edit.

## Nine plug-points, one contract

Every plug-point is a folder under `pbrain/` (`models/`, `aif/`,
`tissue_roi/`, `aggregation/`, ...). Each contains:

- `base.py` — a `@runtime_checkable` Protocol that refines the universal
  [`Plugin`](api/core.md) contract with a typed entry method
  (`fit`, `extract`, `aggregate`, `convert`, ...) and the plug-in's
  Inputs/Result dataclasses.
- `__init__.py` — two lines that build the registry:

  ```python
  from pbrain.core import discover
  from .base import KineticModel
  REGISTRY = discover(__name__, __file__, expected_protocol=KineticModel)
  ```

- one `<key>.py` per implementation, exposing a module-level `PLUGIN`.

[`discover()`](api/core.md) scans the folder (skipping `_*.py` and
`base.py`), imports each module, indexes whatever exposes `PLUGIN`,
runtime-checks it against the expected Protocol, and errors loudly on
duplicate keys. **There is no `register()` call and no dispatcher to
edit.** Drop a file in and it appears in the registry; delete or
`.gitignore` it and it is gone.

The universal `Plugin` contract is five class attributes — `key`,
`name`, `description`, `accepts`, `produces` — and each plug-point adds
its own typed method. See [Plug-in contracts](api/contracts.md) for
every Protocol.

## Dynamic outputs

A model's `ModelResult.maps` is a `dict[str, np.ndarray]` whose keys are
**the model's own choice** (Patlak gives `{"ki", "vb"}`; Tikhonov gives
`{"cbf", "mtt", "cth", "lambda_opt"}`). Aggregators iterate those keys
generically and never hard-code a parameter name, so a new model's
outputs flow through aggregation, NIfTI/CSV/JSON writing, and
diagnostics automatically.

## Configuration

A single immutable [`Config`](api/core.md) dataclass holds every knob and
is passed to each stage explicitly — no global `settings` module. Per
plug-in options live in `config.plugin_options`, keyed
`"<plug-point>.<plugin-key>"`, set on the CLI as
`--opt models.tikhonov.n_lambdas=201`. There are intentionally **no
per-model fields** in `Config`; research groups add their own keys with
zero core changes.

## Adding a plug-in (the one-file rule)

Adding a kinetic model is one file in `pbrain/models/`:

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
    produces:    ClassVar[dict] = {"p1": np.ndarray}
    outputs:     ClassVar[tuple] = ("p1",)            # MUST match maps keys
    units:       ClassVar[dict] = {"p1": "mL/100g/min"}

    def fit(self, inputs: CurveInputs, **opts: Any) -> ModelResult:
        # inputs.c_tissue is (T,) or (T, V); inputs.mask selects voxels
        ...
        return ModelResult(maps={"p1": p1}, units=dict(self.units))

PLUGIN = MyModel()
```

Run it with `pbrain run --models my_model,patlak ...`; it aggregates at
every level and gets voxel/tissue/parcel diagnostics for free. The same
"drop a file exporting `PLUGIN`" pattern applies to every plug-point —
AIF extractors, T1/M0 fitters, normalisers, signal-to-conc converters,
tissue-ROI providers, aggregators, loaders, diagnostics, and even whole
new stages or plug-points. The full templates (diffusion model,
diagnostic plot, new stage, new plug-point) are in
[`ADDING_PLUGINS.md`](https://github.com/edtireli/p-brain/blob/main/docs/ADDING_PLUGINS.md).

## Verifying a plug-in

```bash
pbrain list <plug-point>      # confirm it's discovered, see its contract
python -m pbrain.demo --clean # full weights-free pipeline smoke test
pytest tests/ -q              # the suite
```

A good plug-in PR adds a `tests/test_pbrain_*.py` test asserting the
contract (declared `outputs` == produced `maps` keys) plus a sanity
check on a synthetic curve.
