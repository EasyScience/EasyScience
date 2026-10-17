# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import weakref


class Map:
    """Global registry of EasyScience objects, indexed by ``unique_name``.

    Registered objects must have a ``unique_name`` attribute. Values are
    held weakly, so an entry disappears once nothing else references the
    object.
    """

    def __init__(self):
        self._store = weakref.WeakValueDictionary()

    def vertices(self) -> list[str]:
        """All registered unique names, including retained old names."""
        return list(self._store)

    def objects(self) -> list[object]:
        """All registered objects that are still alive.

        The list holds strong references, so it keeps those objects alive
        until it is released. An object renamed via ``unique_name`` keeps
        its old name registered, so it can appear more than once.
        """
        return list(self._store.values())

    def get_item_by_key(self, item_id: str) -> object:
        try:
            return self._store[item_id]
        except KeyError:
            raise ValueError('Item not in map.') from None

    def add_vertex(self, obj: object) -> None:
        name = obj.unique_name
        if name in self._store:
            raise ValueError(f'Object name {name} already exists in the map.')
        self._store[name] = obj

    def prune(self, key: str) -> None:
        self._store.pop(key, None)

    def _clear(self) -> None:
        """Empty the map. Used by tests and by downstream project resets."""
        self._store.clear()

    def __repr__(self) -> str:
        return f'Map object of {len(self._store)} vertices.'
