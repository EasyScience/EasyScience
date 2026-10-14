---
icon: material/lightbulb-outline
---

# :material-lightbulb-outline: Concepts

EasyScience works with numbers that know what they are: a value, a unit,
an uncertainty and, for numbers that are fitted, the range they may
take. This page defines the ideas that the tutorials rely on.

## Numbers with a unit and an uncertainty

A **`DescriptorNumber`** is a number with a value, a unit and an
uncertainty. It takes part in arithmetic and in the functions of
`es.math`, but it cannot be fitted. It is the right choice for a
constant, or for a measured value that no fit will touch.

A **`Parameter`** is a `DescriptorNumber` that can be fitted. It adds
two things: **bounds** and a **fixed/free** switch. A free parameter is
varied by a fit; a fixed one is not.

The **uncertainty** is stored as a **variance**, in the square of the
unit. The attribute `error` is its square root, in the unit itself. The
[arithmetic notebook](../tutorials/basics/maths-arithmetic.ipynb)
explains the difference between an error and an uncertainty.

The **bounds** `min` and `max` are the range that a fit may explore. For
Bayesian sampling the same range is the uniform prior of the parameter.

**Units** are checked. Numbers whose units have different dimensions
cannot be added, a length and a time for example, while a length in
metres and one in centimetres can: the result takes the unit of the
first. A product or a quotient gets the combined unit, and
`convert_unit` expresses a number, with its uncertainty and bounds, in
another unit.

## Results of calculations

Arithmetic and the functions of `es.math` return a **new** number. Its
value, unit, uncertainty and bounds are worked out from the numbers that
went in. The uncertainty is propagated to first order and assumes that
the inputs are independent. Two consequences follow:

- a result is a **snapshot**: it does not change when a number it came
  from changes afterwards,
- a quantity that depends on the same number more than once, or on
  correlated numbers, can have an unreliable uncertainty. The notebooks
  on
  [error propagation](../tutorials/basics/maths-error-propagation.ipynb)
  and [correlation](../tutorials/basics/maths-correlation.ipynb) show
  when and by how much.

## Models and fitting

A **model** is a class derived from `ModelBase` that holds its
parameters as properties and is called with the independent variable. A
**`Fitter`** takes the model and finds the free parameters that make it
describe the data. The available minimizers are LMFit, Bumps, DFO and
Minuit, selected with `AvailableMinimizers`.

After a fit every free parameter holds its fitted value and its
uncertainty, and can be used like any other number. A **`Sampler`**
explores the full posterior distribution of the parameters instead of a
single best value.

The [Fitting](../tutorials/fitting/index.md) tutorials show all of this
on data.
