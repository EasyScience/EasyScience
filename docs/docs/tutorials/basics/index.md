---
icon: material/calculator-variant
---

# :material-calculator-variant: Basics

These short notebooks show how to work with EasyScience numbers: values
that carry a **unit**, an **uncertainty** and, for fit parameters,
**bounds**. You need Python and a little NumPy; no earlier knowledge of
EasyScience is required.

Every notebook stands alone, so you can read them in any order. If you
are new, follow the path below.

## A suggested path

**1. Numbers with units and uncertainties**

- [Arithmetic](maths-arithmetic.ipynb) – Add, subtract, multiply and
  divide numbers with units, uncertainties and bounds, and see how each
  of them carries over to the result. _Start here._

**2. Functions**

Read the ones you need; each stands alone.

- [Exponential and logarithm](maths-exp-log.ipynb) – `exp`, `log`,
  `log10`, `sqrt` and the error functions `erf` and `erfc`.
- [Trigonometric functions](maths-trigonometric.ipynb) – `sin`, `cos`,
  `tan` and `sinc`, with Bragg's law as the running example.
- [Inverse trigonometric functions](maths-inverse-trigonometric.ipynb) –
  `arcsin`, `arccos`, `arctan` and `arctan2`, and the values they
  refuse.
- [Hyperbolic functions](maths-hyperbolic.ipynb) – `sinh`, `cosh`,
  `tanh` and their inverses.

**3. Trusting a result**

- [Error propagation](maths-error-propagation.ipynb) – How the choice of
  formula changes the uncertainty of a result, with the area of a table
  as the running example. _A step further._
- [Correlation](maths-correlation.ipynb) – Correlated measurements and
  fit parameters, and what ignoring the correlation costs. _A step
  further._

When you are comfortable with these, continue with the
[Fitting introduction](../fitting-intro.ipynb), which uses the same
numbers on the result of a fit.

## How the notebooks work

- Each part ends with a few **exercises**. The hints and the solution
  are folded away and open on a click, so you can read straight through
  or try each exercise first.
- A solution is a worked answer: its code runs and the result is shown.
- The charts are interactive, and they need an internet connection to
  load.
- Each notebook can be downloaded and run on your computer; see
  [how to run the tutorials](../../installation-and-setup/index.md#how-to-run-tutorials).
