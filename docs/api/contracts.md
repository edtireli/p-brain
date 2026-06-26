# Plug-in contracts

Each plug-point refines the universal [`Plugin`](core.md) contract with a
typed Protocol and its Inputs/Result dataclasses. Implement the Protocol,
expose a module-level `PLUGIN`, and the plug-in is auto-discovered.

## Input loaders

::: pbrain.io.loaders.base

## T1 / M0 fitters

::: pbrain.t1_m0.base

## AIF / VIF extractors

::: pbrain.aif.base

## Signal-to-concentration converters

::: pbrain.signal_to_conc.base

## Curve normalisers

::: pbrain.normalisation.base

## Tissue-ROI providers

::: pbrain.tissue_roi.base

## Aggregators

::: pbrain.aggregation.base

## Diagnostics

::: pbrain.diagnostics.base
