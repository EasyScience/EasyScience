---
icon: material/information-slab-circle
---

# :material-information-slab-circle: Introduction

## Description

**EasyScience** is the foundation of the framework, providing reusable
building blocks for scientific libraries and applications aimed at
making data analysis easier.

**EasyScience** is developed as a Python library.

<!--
**EasyScience** is developed using the
[EasyScience framework](https://easyscience.org).
-->

## Who is it for

**EasyScience** is the foundation for scientific libraries and
applications. It is for developers who build on its numbers, which carry
a **unit**, an **uncertainty** and **bounds**, and for scientists who
use them directly to analyse data. It provides:

- numbers (`Parameter`, `DescriptorNumber`) that know their unit,
  uncertainty and bounds, with arithmetic and mathematical functions
  that carry all three along,
- fitting of a model to data with several minimizers, and Bayesian
  sampling of the posterior.

## A first example

Bragg's law gives the lattice spacing from a wavelength and the Bragg
angle. Each number has an uncertainty, and the result has one too:

```python
import easyscience as es
from easyscience import DescriptorNumber

wavelength = DescriptorNumber(name='wavelength', value=1.54, unit='angstrom', variance=0.01**2)
angle = DescriptorNumber(name='angle', value=22.5, unit='deg', variance=0.1**2)

d = wavelength / (2 * es.math.sin(angle))
d.convert_unit('nm')
print(f'd = {d.value:.4f} ± {d.error:.4f} {d.unit}')
```

```text
d = 0.2012 ± 0.0016 nm
```

## Where to start

- **New to EasyScience?** Follow the
  [Installation & Setup](../installation-and-setup/index.md), then the
  [Basics](../tutorials/basics/index.md) tutorials.
- **Want to fit data?** Continue with the
  [Fitting](../tutorials/fitting/index.md) tutorials.
- **Building on EasyScience?** Read the
  [Concepts](../user-guide/concepts.md) and the
  [API Reference](../api-reference/index.md).
- **Following a workshop?** See
  [Workshops & Schools](../tutorials/workshops/index.md).

## License

**EasyScience** library is released under the
[BSD 3-Clause License](https://raw.githubusercontent.com/easyscience/core/master/LICENSE).

## Releases

The latest version of the **EasyScience** library is
[{{ vars.release_version }}](https://github.com/easyscience/core/releases/latest).

For a complete list of new features, bug fixes, and improvements, see
the
[GitHub Releases page](https://github.com/easyscience/core/releases).

## Citation

If you use **EasyScience** library in your work, please cite the
specific version you used.

All official releases of the **EasyScience** library are archived on
Zenodo, each with a version-specific Digital Object Identifier (DOI).

Citation details in various styles (e.g., APA, MLA) and formats (e.g.,
BibTeX, JSON) are available on the
[Zenodo archive page](https://doi.org/10.5281/zenodo.19918609).

## Contributing

We welcome contributions of any kind!

**EasyScience** is intended to be a community-driven, open-source
project supported by a diverse group of contributors.

The project is maintained by the
[European Spallation Source (ESS)](https://ess.eu).

If you would like to report a bug or request a new feature, please use
the [GitHub Issue Tracker](https://github.com/easyscience/core/issues)
(A free GitHub account is required.)

To contribute code, documentation, or tests, please see our
[:material-account-plus: Contributing Guidelines](https://github.com/easyscience/core/blob/master/CONTRIBUTING.md)
for detailed development instructions.

## Get in Touch

For general questions or feedback, please contact us at
[support@easyscience.org](mailto:support@easyscience.org).
