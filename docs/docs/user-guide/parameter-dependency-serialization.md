# Serializing Dependent Parameters

A `Parameter` can depend on other parameters through an expression (see
`Parameter.make_dependent_on` and `Parameter.from_dependency`). This
page explains how such parameters are written to a dictionary or JSON
and how their dependencies are rebuilt when they are loaded.

## Quick Start

Serialize every parameter involved with `to_dict`:

```python
import json

from easyscience import Parameter

a = Parameter(value=2.0, unit='m', min=0, max=10, display_name='a')
b = Parameter.from_dependency(
    dependency_expression='2 * a',
    dependency_map={'a': a},
    display_name='b',
)

with open('parameters.json', 'w') as f:
    json.dump({'a': a.to_dict(), 'b': b.to_dict()}, f, indent=2)
```

To load them, deserialize all parameters first. Then resolve the
dependencies in a single step:

```python
import json

from easyscience import Parameter
from easyscience.variable.parameter_dependency_resolver import (
    resolve_all_parameter_dependencies,
)

with open('parameters.json') as f:
    params_dict = json.load(f)

new_a = Parameter.from_dict(params_dict['a'])
new_b = Parameter.from_dict(params_dict['b'])

resolve_all_parameter_dependencies({'a': new_a, 'b': new_b})

new_a.value = 5.0
print(new_b.value)  # 10.0
```

!!! warning "Always resolve after loading"

    `Parameter.from_dict` does **not** restore the dependency on its
    own. Until `resolve_all_parameter_dependencies` (or
    `Parameter.resolve_pending_dependencies`) is called, a loaded
    dependent parameter behaves as an independent parameter holding the
    value it had when it was saved.

Alternatively, if all you have is a dictionary of serialized parameters,
the convenience helper `deserialize_and_resolve_parameters` combines
both steps:

```python
from easyscience.variable.parameter_dependency_resolver import (
    deserialize_and_resolve_parameters,
)

params = deserialize_and_resolve_parameters(params_dict)
params['a'].value = 3.0
print(params['b'].value)  # 6.0
```

## What Is Written

An independent parameter is written with its constructor arguments
(`value`, `unit`, `variance`, `min`, `max`, `fixed`, `description`,
`url`, and `display_name`/`unique_name` when they were set explicitly).
A dependent parameter gets the following additional fields:

| Field                            | Meaning                                                                             |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| `_dependency_string`             | The dependency expression, e.g. `"2 * a"`.                                          |
| `_dependency_map_serializer_ids` | Maps each symbol in the expression to the serializer id of the object it refers to. |
| `_independent`                   | `false`, marking the parameter as dependent.                                        |
| `_desired_unit`                  | Only present if a `desired_unit` was given when making the dependency.              |

Any parameter that something else depends on also writes its own
`__serializer_id`. This is a UUID assigned the first time another
parameter depends on it. It is what `_dependency_map_serializer_ids`
refers to:

```json
{
  "@module": "easyscience.variable.parameter",
  "@class": "Parameter",
  "value": 4.0,
  "unit": "m",
  "display_name": "b",
  "_dependency_string": "2 * a",
  "_independent": false,
  "_dependency_map_serializer_ids": {
    "a": "a48db37b-519b-4bea-9873-aa3cc5b2ec76"
  }
}
```

Serializer ids are independent of `unique_name`. A loaded parameter can
therefore receive a fresh unique name, avoiding a clash with objects
that already exist in the session, and its dependencies still find each
other.

## How Loading Works

1. `Parameter.from_dict` creates each parameter as an ordinary
   independent parameter and restores its `__serializer_id`.
2. For dependent parameters, the expression, the serializer-id map and
   the desired unit are kept on the object as pending dependency
   information.
3. `resolve_all_parameter_dependencies(obj)` walks `obj` and finds every
   parameter with pending information. `obj` can be a single
   `Parameter`, a list, tuple or dict of them, or an object exposing
   them as public attributes or properties. For each parameter it looks
   up the referenced objects by serializer id among all objects alive in
   the session and calls `make_dependent_on`. It then removes the
   pending information.

Because resolution happens only after everything is loaded, parameters
can be deserialized in **any order**. A dependent parameter may come
before the parameters it depends on.

To check what still needs resolving, use
`get_parameters_with_pending_dependencies`:

```python
from easyscience.variable.parameter_dependency_resolver import (
    get_parameters_with_pending_dependencies,
)

pending = get_parameters_with_pending_dependencies({'a': new_a, 'b': new_b})
print(f'{len(pending)} parameter(s) still need resolving')
```

## Errors

`resolve_all_parameter_dependencies` tries every pending parameter and
then raises a single `ValueError` listing all failures. The typical
cause is a referenced parameter that was not loaded:

```text
ValueError: Failed to resolve dependencies for 1 parameter(s):
Failed to resolve dependencies for parameter 'Parameter_0' (display_name: 'b', serializer_id: 'unknown'): Cannot find parameter with serializer_id 'f9040c2a-...'
```

An invalid expression, or one that would create a cyclic dependency, is
reported the same way, with the underlying error message appended.

## Best Practices

- **Serialize the whole dependency graph.** Every parameter referenced
  by a dependency expression must be saved and loaded together with the
  parameters that depend on it.
- **Load everything, then resolve once.** Deserialize all parameters
  first and call `resolve_all_parameter_dependencies` on the collection
  as a whole.
- **Keep the referenced objects alive.** Lookup goes through the global
  object map, so a parameter that has been garbage-collected cannot be
  found.
- **Do not load the same data twice in one session.** Serializer ids are
  restored verbatim. If the original parameters, or an earlier load of
  the same file, are still alive, a dependency can be bound to that
  older object instead of the one just loaded.

!!! note "Parameters nested in models"

    Dependency information is written by `Parameter.to_dict`. Calling
    `to_dict` on a `ModelBase` that *contains* dependent parameters
    does not currently go through `Parameter.to_dict`, so the
    dependency fields are not included. Until this is addressed,
    serialize dependent parameters directly as shown above.
