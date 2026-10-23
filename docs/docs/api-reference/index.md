---
icon: material/code-braces-box
---

# :material-code-braces-box: API Reference

This section contains the reference detailing the functions and modules
available in EasyScience.

- [base_classes](base_classes.md) – Core abstract and helper base
  classes used to build EasyScience objects (e.g. `NewBase`,
  `ModelBase`, `EasyList`, `SerializerBase`).
- [fitting](fitting.md) – Fitting and sampling, including `Fitter`,
  `Sampler` and the available minimizers.
- [global_object](global_object.md) – Global singleton providing shared
  services (logger, object registry, undo/redo stack, script manager).
- [job](job.md) – Job and experiment abstractions for running and
  organizing analyses.
- [utils](utils.md) – Miscellaneous utility functions and helpers (class
  tools, decorators, type helpers).
- [variable](variable.md) – Descriptor types and variable abstractions
  (e.g. `DescriptorNumber`, `Parameter`, `DescriptorArray`). All of them
  are `NewBase` objects, serialized with `to_dict`/`from_dict`.
