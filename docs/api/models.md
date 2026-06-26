# Kinetic models

The shipped pharmacokinetic models. Each implements the `KineticModel`
Protocol (see [Contract](#contract) below) and returns a `ModelResult`
whose `maps` keys match its declared `outputs`.

## Contract

::: pbrain.models.base

## Patlak

::: pbrain.models.patlak

## Tikhonov deconvolution

::: pbrain.models.tikhonov

## Extended Tofts

::: pbrain.models.extended_tofts

## Inverse-Gaussian residue

::: pbrain.models.inverse_gaussian

## Mittag-Leffler (fractional) residue

::: pbrain.models.mittag_leffler

## Stieltjes transfer function

::: pbrain.models.stieltjes
