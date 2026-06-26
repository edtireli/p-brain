# p-Brain API reference

_p_-Brain is a **cross-platform Python command-line tool** for automated
DCE-MRI and diffusion-MRI research. The `pbrain` CLI is the product; the
optional macOS desktop app is just one front-end on top of it.

This site is the rendered **API reference**, generated directly from the
package docstrings. For the user-facing guide (install, how to run, the
output tree, citation) see the [project README on
GitHub](https://github.com/edtireli/p-brain#readme).

## Install

```bash
pip install p-brain          # core install — installs the `pbrain` command
pbrain --help
```

Optional extras: `[cnn]` (TensorFlow CNN AIF), `[diffusion]` (dipy),
`[dicom]` (pydicom; DICOM input also needs the `dcm2niix` binary),
`[docs]` (this site), `[dev]` (tests), `[all]`.

## How it fits together

p-Brain is a **nine-stage pipeline of auto-discovered single-file
plug-ins** with file-manifest interfaces between stages. To understand or
extend the design, start with the [architecture page](architecture-overview.md);
to look up a class or function, use the **API reference** in the
navigation:

- **[Core](api/core.md)** — `Config`, `Pipeline`, `Stage`, `Manifest`,
  `discover`, the `Plugin` contract.
- **[Plug-in contracts](api/contracts.md)** — the typed Protocols every
  plug-point implements (AIF, T1/M0, signal→conc, normaliser, tissue
  ROI, aggregator, loader, diagnostic).
- **[Kinetic models](api/models.md)** — Patlak, Tikhonov, extended
  Tofts, inverse-Gaussian, Mittag-Leffler, Stieltjes.
- **[Diffusion models](api/diffusion.md)** — the DWI track.
- **[Pipeline stages](api/stages.md)** — the built-in stages.
- **[QC](api/qc.md)** — voxel-wise CNR and other quality checks.
