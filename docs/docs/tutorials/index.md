---
icon: material/school
---

# :material-school: Tutorials

This section presents a collection of **Jupyter Notebook** tutorials
that demonstrate how to use EasyScience for various tasks. These
tutorials serve as self-contained, step-by-step **guides** to help users
grasp the workflow of data analysis using EasyScience.

Instructions on how to run the tutorials are provided in the
[:material-cog-box: Installation & Setup](../installation-and-setup/index.md#how-to-run-tutorials)
section of the documentation.

The tutorials are organized into the following categories:

## Basics

Short examples of how to do maths with EasyScience parameters, with
exercises. Units, uncertainties and bounds are propagated for you.

- [Arithmetic](basics/maths-arithmetic.ipynb) – Addition, subtraction,
  multiplication, division and powers of numbers with units,
  uncertainties and bounds.
- [Exponential and logarithm](basics/maths-exp-log.ipynb) – `exp`,
  `log`, `log10`, `sqrt` and the error functions `erf` and `erfc`.
- [Trigonometric functions](basics/maths-trigonometric.ipynb) – `sin`,
  `cos`, `tan` and `sinc`, with Bragg's law as the running example.
- [Inverse trigonometric functions](basics/maths-inverse-trigonometric.ipynb)
  – `arcsin`, `arccos`, `arctan` and `arctan2`, and their domains.
- [Hyperbolic functions](basics/maths-hyperbolic.ipynb) – `sinh`,
  `cosh`, `tanh` and their inverses.

## Workshops & Schools

- [Fitting introduction](fitting-intro.ipynb) – A short introduction to
  fitting: fit a straight line, then compute quantities that derive from
  the fitted parameters, with their uncertainty.
- [Fitting QENS](fitting-qens.ipynb) – A tutorial demonstrating how to
  fit a quasielastic neutron scattering (QENS) data using EasyScience
  framework.
- [Fitting SANS](fitting-sans.ipynb) – A tutorial demonstrating how to
  fit a small-angle neutron scattering (SANS) data using EasyScience
  framework.
- [Bayesian analysis](fitting-bayesian.ipynb) – A tutorial showing how
  to run Bayesian MCMC sampling on a model with EasyScience to obtain
  full posterior distributions for the parameters.
- [Progress Callback](progress-callback.ipynb) – A tutorial showing how
  to monitor fitting progress across minimizer backends and update a
  notebook UI while a fit is running.
