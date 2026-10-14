---
icon: material/chart-bell-curve-cumulative
---

# :material-chart-bell-curve-cumulative: Fitting

A fit starts from data and a model and returns the model parameters with
their uncertainties. These notebooks show how to fit with EasyScience
and, above all, what to do with the result: derived quantities with
their uncertainties, computed with the arithmetic and the functions of
`es.math`. They assume that you know the [Basics](../basics/index.md).

## A suggested path

- [Fitting introduction](../fitting-intro.ipynb) – Fit a straight line
  to simulated data, then work out quantities that follow from the
  fitted slope. _Start here._
- [SANS sphere](sans-sphere.ipynb) – Fit the scattering of spheres to
  small-angle neutron scattering data, then derive the diameter, the
  volume, the radius of gyration and a Guinier approximation. _Next._
- [QENS Lorentzian](qens-lorentzian.ipynb) – Fit a Lorentzian, convolved
  with the instrument resolution, to quasi-elastic neutron scattering
  data, then derive the relaxation time, a decay curve and the
  resolution. _Next._
- [Bayesian QENS](bayesian-qens.ipynb) – Sample the posterior of the
  QENS model and compare its derived quantities with the first-order
  uncertainties. _Advanced._

## How the notebooks work

- Each part ends with a few **exercises**. The hints and the solution
  are folded away and open on a click, so you can read straight through
  or try each exercise first.
- A solution is a worked answer: its code runs and the result is shown.
- The SANS and QENS notebooks download their data the first time they
  run, so they need an internet connection. So do the interactive
  charts.
- Each notebook can be downloaded and run on your computer; see
  [how to run the tutorials](../../installation-and-setup/index.md#how-to-run-tutorials).
