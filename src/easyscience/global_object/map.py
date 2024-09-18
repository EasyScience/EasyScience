#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import gc
import sys
import weakref
from typing import List
from typing import Optional


class _EntryList(list):
    def __init__(self, *args, my_type=None, **kwargs):
        super(_EntryList, self).__init__(*args, **kwargs)
        self.__known_types = {'argument', 'created', 'created_internal', 'returned'}
        self.finalizer = None
        self._type = []
        if my_type in self.__known_types:
            self._type.append(my_type)

    def __repr__(self) -> str:
        s = 'Map entry of type: '
        if self._type:
            s += ', '.join(self._type)
        else:
            s += 'Undefined'
        s += '. With'
        if self.finalizer is None:
            s += 'out'
        s += 'a finalizer.'
        return s

    def __delitem__(self, key):
        super(_EntryList, self).__delitem__(key)

    def remove_type(self, old_type: str):
        if old_type in self.__known_types and old_type in self._type:
            self._type.remove(old_type)

    def reset_type(self, default_type: str = None):
        self._type = []
        self.type = default_type

    @property
    def type(self) -> List[str]:
        return self._type

    @type.setter
    def type(self, value: str):
        if value in self.__known_types and value not in self._type:
            self._type.append(value)

    @property
    def is_argument(self) -> bool:
        return 'argument' in self._type

    @property
    def is_created(self) -> bool:
        return 'created' in self._type

    @property
    def is_created_internal(self) -> bool:
        return 'created_internal' in self._type

    @property
    def is_returned(self) -> bool:
        return 'returned' in self._type


class Map:
    def __init__(self):
        # A dictionary of object names and their corresponding objects
        self._store = weakref.WeakValueDictionary()
        # A dict with object names as keys and a list of their object types as values, with weak references
        self.__type_dict = {}

    def vertices(self) -> List[str]:
        """returns the vertices of a map"""
        return list(self._store.keys())

    def edges(self):
        """returns the edges of a map"""
        return self.__generate_edges()

    @property
    def argument_objs(self) -> List[str]:
        return self._nested_get('argument')

    @property
    def created_objs(self) -> List[str]:
        return self._nested_get('created')

    @property
    def created_internal(self) -> List[str]:
        return self._nested_get('created_internal')

    @property
    def returned_objs(self) -> List[str]:
        return self._nested_get('returned')

    def _nested_get(self, obj_type: str) -> List[str]:
        """Access a nested object in root by key sequence."""
        return [key for key, item in self.__type_dict.items() if obj_type in item.type]

    def get_item_by_key(self, item_id: str) -> object:
        if item_id in self._store.keys():
            return self._store[item_id]
        raise ValueError('Item not in map.')

    def is_known(self, vertex: object) -> bool:
        # All objects should have a 'unique_name' attribute
        return vertex.unique_name in self._store.keys()

    def find_type(self, vertex: object) -> List[str]:
        if self.is_known(vertex):
            return self.__type_dict[vertex.unique_name].type

    def reset_type(self, obj, default_type: str):
        if obj.unique_name in self.__type_dict.keys():
            self.__type_dict[obj.unique_name].reset_type(default_type)

    def change_type(self, obj, new_type: str):
        if obj.unique_name in self.__type_dict.keys():
            self.__type_dict[obj.unique_name].type = new_type

    def add_vertex(self, obj: object, obj_type: str = None):
        name = obj.unique_name
        if name in self._store.keys():
            raise ValueError(f'Object name {name} already exists in the graph.')
        self._store[name] = obj
        self.__type_dict[name] = _EntryList()  # Add objects type to the list of types
        self.__type_dict[name].finalizer = weakref.finalize(self._store[name], self.prune, name)
        self.__type_dict[name].type = obj_type

    def add_edge(self, start_obj: object, end_obj: object):
        if start_obj.unique_name in self.__type_dict.keys():
            self.__type_dict[start_obj.unique_name].append(end_obj.unique_name)
        else:
            raise AttributeError('Start object not in map.')

    def get_edges(self, start_obj) -> List[str]:
        if start_obj.unique_name in self.__type_dict.keys():
            return list(self.__type_dict[start_obj.unique_name])
        else:
            raise AttributeError

    def __generate_edges(self) -> list:
        """A static method generating the edges of the
        map. Edges are represented as sets
        with one (a loop back to the vertex) or two
        vertices
        """
        edges = []
        for vertex in self.__type_dict:
            for neighbour in self.__type_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def prune_vertex_from_edge(self, parent_obj, child_obj):
        vertex1 = parent_obj.unique_name
        if child_obj is None:
            return
        vertex2 = child_obj.unique_name

        if vertex1 in self.__type_dict.keys() and vertex2 in self.__type_dict[vertex1]:
            del self.__type_dict[vertex1][self.__type_dict[vertex1].index(vertex2)]

    def prune(self, key: str):
        if key in self.__type_dict.keys():
            del self.__type_dict[key]
            del self._store[key]

    def find_isolated_vertices(self) -> list:
        """returns a list of isolated vertices."""
        graph = self.__type_dict
        isolated = []
        for vertex in graph:
            print(isolated, vertex)
            if not graph[vertex]:
                isolated += [vertex]
        return isolated

    def find_path(self, start_vertex: str, end_vertex: str, path=[]) -> list:
        """find a path from start_vertex to end_vertex
        in map"""

        graph = self.__type_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex, end_vertex, path)
                if extended_path:
                    return extended_path
        return []

    def find_all_paths(self, start_vertex: str, end_vertex: str, path=[]) -> list:
        """find all paths from start_vertex to
        end_vertex in map"""

        graph = self.__type_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, end_vertex, path)
                for p in extended_paths:
                    paths.append(p)
        return paths

    def reverse_route(self, end_vertex: str, start_vertex: Optional[str] = None) -> List:
        """
        In this case we have an object and want to know the connections to get to another in reverse.
        We might not know the start_object. In which case we follow the shortest path to a base vertex.
        :param end_obj:
        :type end_obj:
        :param start_obj:
        :type start_obj:
        :return:
        :rtype:
        """
        path_length = sys.maxsize
        optimum_path = []
        if start_vertex is None:
            # We now have to find where to begin.....
            for possible_start, vertices in self.__type_dict.items():
                if end_vertex in vertices:
                    temp_path = self.find_path(possible_start, end_vertex)
                    if len(temp_path) < path_length:
                        path_length = len(temp_path)
                        optimum_path = temp_path
        else:
            optimum_path = self.find_path(start_vertex, end_vertex)
        optimum_path.reverse()
        return optimum_path

    def is_connected(self, vertices_encountered=None, start_vertex=None) -> bool:
        """determines if the map is connected"""
        if vertices_encountered is None:
            vertices_encountered = set()
        graph = self.__type_dict
        vertices = list(graph.keys())
        if not start_vertex:
            # chose a vertex from graph as a starting point
            start_vertex = vertices[0]
        vertices_encountered.add(start_vertex)
        if len(vertices_encountered) != len(vertices):
            for vertex in graph[start_vertex]:
                if vertex not in vertices_encountered and self.is_connected(vertices_encountered, vertex):
                    return True
        else:
            return True
        return False

    def _clear(self):
        """Reset the map to an empty state."""
        for vertex in self.vertices():
            self.prune(vertex)
        gc.collect()
        self.__type_dict = {}

    def __repr__(self) -> str:
        return f'Map object of {len(self._store)} vertices.'
