# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import weakref
from typing import Iterator


class ObjectRegistry:
    """Global registry of EasyScience objects, indexed by ``unique_name``.

    Registered objects must have a ``unique_name`` attribute. Values are
    held weakly, so an entry disappears once nothing else references the
    object.

    Supports ``name in registry``, ``registry[name]``, ``len(registry)``
    and iteration over the registered names.
    """

    def __init__(self):
        # A dict with object names as keys and a list of their object
        # types as values, with weak references
        self._store = weakref.WeakValueDictionary()

    def __getitem__(self, name: str) -> object:
        return self._store[name]

    def __contains__(self, name: object) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[str]:
        # Iterate over a snapshot, so objects collected mid-loop are safe
        return iter(list(self._store))

    def __len__(self) -> int:
        return len(self._store)

    def names(self) -> list[str]:
        """All registered unique names, including retained old names."""
        return list(self._store)

    def objects(self) -> list[object]:
        """All registered objects that are still alive.

        The list holds strong references, so it keeps those objects alive
        until it is released. An object renamed via ``unique_name`` keeps
        its old name registered, so it can appear more than once.
        """
        return list(self._store.values())

    def add(self, obj: object) -> None:
        """Register ``obj`` under its ``unique_name``.

        Parameters
        ----------
        obj : object
            The object to register. It must have a ``unique_name``
            attribute.

        Raises
        ------
        ValueError
            If the name is already registered.
        """
        name = obj.unique_name
        if name in self._store:
            raise ValueError(f'Object name {name} already exists in the registry.')
        self._store[name] = obj

    def discard(self, name: str) -> None:
        """Remove ``name`` from the registry, if present."""
        self._store.pop(name, None)

    def clear(self) -> None:
        """Empty the registry."""
        self._store.clear()

    def __repr__(self) -> str:
        return f'ObjectRegistry object of {len(self._store)} entries.'
