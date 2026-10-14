# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import gc
from unittest.mock import MagicMock

import pytest

from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import NewBase
from easyscience.global_object.map import Map
from easyscience.global_object.map import _EntryList


class TestEntryList:
    """Test the _EntryList helper class"""

    def test_init_default(self):
        """Test _EntryList initialization with defaults"""
        entry = _EntryList()
        assert entry._type == []
        assert entry.finalizer is None

    def test_init_with_known_type(self):
        """Test _EntryList initialization with known type"""
        entry = _EntryList(my_type='created')
        assert 'created' in entry._type

    def test_init_with_unknown_type(self):
        """Test _EntryList initialization with unknown type"""
        entry = _EntryList(my_type='unknown')
        assert entry._type == []

    def test_repr(self):
        """Test string representation"""
        entry = _EntryList()
        repr_str = str(entry)
        assert 'Undefined' in repr_str
        assert 'finalizer' in repr_str

        entry.type = 'created'
        repr_str = str(entry)
        assert 'created' in repr_str
        assert 'finalizer' in repr_str

    def test_type_property_getter(self):
        """Test type property getter"""
        entry = _EntryList()
        assert entry.type == []

    def test_type_property_setter_valid(self):
        """Test type property setter with valid types"""
        entry = _EntryList()
        entry.type = 'created'
        assert 'created' in entry.type

        entry.type = 'argument'
        assert 'argument' in entry.type
        assert 'created' in entry.type  # Should be additive

    def test_type_property_setter_invalid(self):
        """Test type property setter with invalid type"""
        entry = _EntryList()
        entry.type = 'invalid'
        assert entry.type == []

    def test_type_property_setter_duplicate(self):
        """Test type property setter doesn't add duplicates"""
        entry = _EntryList()
        entry.type = 'created'
        entry.type = 'created'  # Try to add again
        assert entry.type.count('created') == 1

    def test_remove_type(self):
        """Test removing a type"""
        entry = _EntryList()
        entry.type = 'created'
        entry.type = 'argument'

        entry.remove_type('created')
        assert 'created' not in entry.type
        assert 'argument' in entry.type

    def test_remove_type_not_present(self):
        """Test removing a type that's not present"""
        entry = _EntryList()
        entry.type = 'created'
        entry.remove_type('argument')  # Not present
        assert 'created' in entry.type

    def test_reset_type(self):
        """Test resetting types"""
        entry = _EntryList()
        entry.type = 'created'
        entry.type = 'argument'

        entry.reset_type('returned')
        assert entry.type == ['returned']

    def test_reset_type_default_none(self):
        """Test resetting types with no default"""
        entry = _EntryList()
        entry.type = 'created'

        entry.reset_type()
        assert entry.type == []

    def test_boolean_properties(self):
        """Test boolean property helpers"""
        entry = _EntryList()

        # Test all false initially
        assert not entry.is_argument
        assert not entry.is_created
        assert not entry.is_created_internal
        assert not entry.is_returned

        # Test each property
        entry.type = 'argument'
        assert entry.is_argument

        entry.type = 'created'
        assert entry.is_created

        entry.type = 'created_internal'
        assert entry.is_created_internal

        entry.type = 'returned'
        assert entry.is_returned


class TestMap:
    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    @pytest.fixture
    def base_object(self):
        return NewBase(display_name='test')

    @pytest.fixture
    def parameter_object(self):
        return Parameter(display_name='test2', value=2)

    def test_add_vertex(self, clear, base_object, parameter_object):
        # When Then Expect
        assert len(global_object.map._store) == 2
        assert len(global_object.map._Map__type_dict) == 2

    def test_clear(self, clear, base_object):
        # When
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        # Then
        global_object.map._clear()
        # Expect
        assert len(global_object.map._store) == 0
        assert global_object.map._Map__type_dict == {}

    def test_weakref(self, clear):
        # When
        test_obj = NewBase(display_name='test')
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        # Then
        del test_obj
        gc.collect()
        # Expect
        assert len(global_object.map._store) == 0
        assert len(global_object.map._Map__type_dict) == 0

    def test_vertices(self, clear, base_object, parameter_object):
        # When Then Expect
        assert global_object.map.vertices() == [
            base_object.unique_name,
            parameter_object.unique_name,
        ]

    def test_get_item_by_key(self, clear, base_object, parameter_object):
        # When Then Expect
        assert global_object.map.get_item_by_key(base_object.unique_name) == base_object
        assert global_object.map.get_item_by_key(parameter_object.unique_name) == parameter_object

    @pytest.mark.parametrize(
        'cls, kwargs', [(NewBase, {'display_name': 'test'}), (Parameter, {'value': 2.0})]
    )
    def test_identical_unique_names_exception(self, clear, cls, kwargs):
        # When
        test_obj = cls(unique_name='test', **kwargs)
        # Then Expect
        with pytest.raises(ValueError):
            test_obj2 = cls(unique_name='test', **kwargs)

    def test_unique_name_change_still_in_map(self, clear, base_object, parameter_object):
        # When
        assert global_object.map.get_item_by_key('NewBase_0') == base_object
        assert global_object.map.get_item_by_key('Parameter_0') == parameter_object
        # Then
        base_object.unique_name = 'test3'
        parameter_object.unique_name = 'test4'
        # Expect
        assert global_object.map.get_item_by_key('NewBase_0') == base_object
        assert global_object.map.get_item_by_key('Parameter_0') == parameter_object
        assert global_object.map.get_item_by_key('test3') == base_object
        assert global_object.map.get_item_by_key('test4') == parameter_object

    def test_add_vertex_duplicate_name_error(self, clear, base_object):
        """Test that adding vertex with duplicate name raises error"""
        # When/Then
        with pytest.raises(ValueError, match='already exists'):
            # Try to add another object with same unique_name
            duplicate_obj = NewBase(display_name='duplicate', unique_name=base_object.unique_name)

    def test_add_vertex_with_object_type(self, clear):
        """Test adding vertex with specific object type"""
        # Given
        obj = NewBase(display_name='test')

        # When - Object is automatically added during construction
        # Then
        assert global_object.map.is_known(obj)
        assert 'created' in global_object.map.find_type(obj)

    def test_is_known_object(self, clear, base_object):
        """Test is_known method"""
        # When/Then
        assert global_object.map.is_known(base_object) is True

        # Test with unknown object
        unknown_obj = MagicMock()
        unknown_obj.unique_name = 'unknown'
        assert global_object.map.is_known(unknown_obj) is False

    def test_find_type_known_object(self, clear, base_object):
        """Test find_type method"""
        # When/Then
        types = global_object.map.find_type(base_object)
        assert isinstance(types, list)
        assert 'created' in types

    def test_find_type_unknown_object(self, clear):
        """Test find_type with unknown object"""
        # Given
        unknown_obj = MagicMock()
        unknown_obj.unique_name = 'unknown'

        # When/Then
        result = global_object.map.find_type(unknown_obj)
        assert result is None

    def test_returned_objs_access_safe_under_modification(self, clear):
        """Ensure accessing returned_objs doesn't raise when entries change size during iteration."""
        objs = [NewBase(display_name=f'race_{i}') for i in range(8)]
        # Mark all as returned
        for o in objs:
            global_object.map.change_type(o, 'returned')

        # Repeatedly access returned_objs while deleting objects and forcing GC to
        # try to trigger concurrent modification. This used to raise RuntimeError.
        for _ in range(200):
            _ = global_object.map.returned_objs  # should not raise
            if _ and objs:
                # delete one object and collect to trigger finalizer/prune
                del objs[0]
                gc.collect()
        # If we got here without exceptions, consider the access safe
        assert True

    def test_reset_type(self, clear, base_object):
        """Test resetting object type"""
        # Given
        original_types = global_object.map.find_type(base_object)

        # When
        global_object.map.reset_type(base_object, 'argument')

        # Then
        new_types = global_object.map.find_type(base_object)
        assert new_types == ['argument']
        assert new_types != original_types

    def test_change_type(self, clear, base_object):
        """Test changing object type"""
        # Given
        original_types = global_object.map.find_type(base_object)
        original_count = len(original_types)

        # When
        global_object.map.change_type(base_object, 'argument')

        # Then
        new_types = global_object.map.find_type(base_object)
        assert 'argument' in new_types
        assert len(new_types) >= original_count  # Should be at least the same or more

    def test_add_edge(self, clear, base_object, parameter_object):
        """Test adding edges between objects"""
        # When
        global_object.map.add_edge(base_object, parameter_object)

        # Then
        edges = global_object.map.get_edges(base_object)
        assert parameter_object.unique_name in edges

    def test_add_edge_unknown_start_object(self, clear, parameter_object):
        """Test adding edge with unknown start object"""
        # Given
        unknown_obj = MagicMock()
        unknown_obj.unique_name = 'unknown'

        # When/Then
        with pytest.raises(AttributeError, match='Start object not in map'):
            global_object.map.add_edge(unknown_obj, parameter_object)

    def test_get_edges(self, clear, base_object, parameter_object):
        """Test getting edges for an object"""
        # Given
        global_object.map.add_edge(base_object, parameter_object)

        # When
        edges = global_object.map.get_edges(base_object)

        # Then
        assert isinstance(edges, list)
        assert parameter_object.unique_name in edges

    def test_get_edges_unknown_object(self, clear):
        """Test getting edges for unknown object"""
        # Given
        unknown_obj = MagicMock()
        unknown_obj.unique_name = 'unknown'

        # When/Then
        with pytest.raises(AttributeError):
            global_object.map.get_edges(unknown_obj)

    def test_prune_vertex_from_edge(self, clear, base_object, parameter_object):
        """Test removing edge between objects"""
        # Given
        global_object.map.add_edge(base_object, parameter_object)
        assert parameter_object.unique_name in global_object.map.get_edges(base_object)

        # When
        global_object.map.prune_vertex_from_edge(base_object, parameter_object)

        # Then
        edges = global_object.map.get_edges(base_object)
        assert parameter_object.unique_name not in edges

    def test_prune_vertex_from_edge_none_child(self, clear, base_object):
        """Test pruning edge with None child"""
        # When/Then - Should not raise error
        global_object.map.prune_vertex_from_edge(base_object, None)

    def test_prune_vertex(self, clear, base_object):
        """Test pruning a vertex completely"""
        # Given
        unique_name = base_object.unique_name
        assert global_object.map.is_known(base_object)

        # When
        global_object.map.prune(unique_name)

        # Then
        assert not global_object.map.is_known(base_object)
        assert unique_name not in global_object.map.vertices()

    def test_edges_generation(self, clear, base_object, parameter_object):
        """Test edge generation"""
        # Given
        global_object.map.add_edge(base_object, parameter_object)

        # When
        edges = global_object.map.edges()

        # Then
        assert isinstance(edges, list)
        expected_edge = {base_object.unique_name, parameter_object.unique_name}
        assert expected_edge in edges

    def test_type_filtering_properties(self, clear):
        """Test type filtering properties"""
        # Given
        obj1 = NewBase(display_name='obj1')  # 'created' type
        obj2 = Parameter(display_name='obj2', value=1)  # 'created' type

        global_object.map.change_type(obj1, 'argument')
        global_object.map.change_type(obj2, 'returned')

        # When/Then
        argument_objs = global_object.map.argument_objs
        created_objs = global_object.map.created_objs
        returned_objs = global_object.map.returned_objs

        assert obj1.unique_name in argument_objs
        assert obj1.unique_name in created_objs  # Should still be there
        assert obj2.unique_name in created_objs
        assert obj2.unique_name in returned_objs

    def test_find_path_simple(self, clear, base_object, parameter_object):
        """Test finding path between objects"""
        # Given
        global_object.map.add_edge(base_object, parameter_object)

        # When
        path = global_object.map.find_path(base_object.unique_name, parameter_object.unique_name)

        # Then
        assert path == [base_object.unique_name, parameter_object.unique_name]

    def test_find_path_same_vertex(self, clear, base_object):
        """Test finding path to same vertex"""
        # When
        path = global_object.map.find_path(base_object.unique_name, base_object.unique_name)

        # Then
        assert path == [base_object.unique_name]

    def test_find_path_no_path(self, clear, base_object, parameter_object):
        """Test finding path when no path exists"""
        # When - No edge added
        path = global_object.map.find_path(base_object.unique_name, parameter_object.unique_name)

        # Then
        assert path == []

    def test_find_all_paths(self, clear, base_object, parameter_object):
        """Test finding all paths between objects"""
        # Given
        global_object.map.add_edge(base_object, parameter_object)

        # When
        paths = global_object.map.find_all_paths(
            base_object.unique_name, parameter_object.unique_name
        )

        # Then
        assert len(paths) == 1
        assert paths[0] == [base_object.unique_name, parameter_object.unique_name]

    def test_reverse_route_with_start(self, clear, base_object, parameter_object):
        """Test reverse route with specified start"""
        # Given
        global_object.map.add_edge(base_object, parameter_object)

        # When
        route = global_object.map.reverse_route(
            parameter_object.unique_name, base_object.unique_name
        )

        # Then
        assert route == [parameter_object.unique_name, base_object.unique_name]

    def test_reverse_route_without_start(self, clear, base_object, parameter_object):
        """Test reverse route without specified start"""
        # Given
        global_object.map.add_edge(base_object, parameter_object)

        # When
        route = global_object.map.reverse_route(parameter_object.unique_name)

        # Then
        assert len(route) >= 1
        assert route[0] == parameter_object.unique_name

    def test_is_connected_single_vertex(self, clear, base_object):
        """Test connectivity with single vertex"""
        # When/Then
        assert global_object.map.is_connected() is True

    def test_is_connected_multiple_vertices(self, clear, base_object, parameter_object):
        """Test connectivity with multiple connected vertices"""
        # Given
        global_object.map.add_edge(base_object, parameter_object)

        # When/Then
        assert global_object.map.is_connected() is True

    def test_map_repr(self, clear, base_object, parameter_object):
        """Test map string representation"""
        # When
        repr_str = str(global_object.map)

        # Then
        assert 'Map object' in repr_str
        assert '2' in repr_str  # Should show vertex count

    def test_get_item_by_key_not_found(self, clear):
        """Test getting item by non-existent key"""
        # When/Then
        with pytest.raises(ValueError, match='Item not in map'):
            global_object.map.get_item_by_key('non_existent')

    def test_clear_with_finalizers(self, clear):
        """Test clearing map properly calls finalizers"""
        # Given
        obj = NewBase(display_name='test')
        original_count = len(global_object.map.vertices())

        # When
        global_object.map._clear()

        # Then
        assert len(global_object.map.vertices()) == 0

    def test_map_initialization(self):
        """Test Map initialization"""
        # When
        test_map = Map()

        # Then
        assert len(test_map.vertices()) == 0
        assert test_map.edges() == []

    def test_vertices_retry_on_runtime_error(self, clear):
        """Test that vertices() retries when RuntimeError occurs during iteration.

        This tests the thread-safety fix for WeakValueDictionary modification
        during iteration (e.g., by garbage collection).
        """
        # Given
        test_map = Map()

        # Create a mock _store that raises RuntimeError on first iteration attempt
        call_count = 0
        original_store = test_map._store

        class MockWeakValueDict:
            def __init__(self):
                self.data = {}
                self.iteration_count = 0

            def __iter__(self):
                self.iteration_count += 1
                if self.iteration_count == 1:
                    # First iteration raises RuntimeError (simulates GC interference)
                    raise RuntimeError('dictionary changed size during iteration')
                # Subsequent iterations succeed
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        mock_store = MockWeakValueDict()
        test_map._store = mock_store

        # When
        vertices = test_map.vertices()

        # Then
        assert vertices == []
        assert mock_store.iteration_count == 2  # Should have retried once

    def test_add_vertex_cleans_stale_type_dict_entry(self, clear):
        """Test that add_vertex cleans up stale __type_dict entries.

        This can happen when a weak reference was collected but the finalizer
        hasn't run yet, and a new object is created with the same unique_name.
        """
        # Given
        test_map = Map()

        # Manually add a stale entry to __type_dict (simulating GC collected but finalizer not run)
        stale_name = 'StaleObject_0'
        test_map._Map__type_dict[stale_name] = _EntryList()

        # Create a mock object with the same unique_name
        mock_obj = MagicMock()
        mock_obj.unique_name = stale_name

        # When - Adding the object should clean up the stale entry first
        test_map.add_vertex(mock_obj, 'created')

        # Then - Object should be added successfully
        assert stale_name in test_map._store
        assert stale_name in test_map._Map__type_dict
        assert test_map._Map__type_dict[stale_name].type == ['created']

    def test_prune_key_not_in_store(self, clear):
        """Test that prune handles case when key is not in _store.

        This defensive check prevents KeyError when the weak reference has
        already been garbage collected but __type_dict entry remains.
        """
        # Given
        test_map = Map()

        # Manually add entry to __type_dict without corresponding _store entry
        orphan_key = 'OrphanObject_0'
        test_map._Map__type_dict[orphan_key] = _EntryList()

        # When - Pruning should not raise error
        test_map.prune(orphan_key)

        # Then - Entry should be removed from __type_dict
        assert orphan_key not in test_map._Map__type_dict

    def test_prune_key_in_both_dicts(self, clear, base_object):
        """Test that prune removes key from both _store and __type_dict."""
        # Given
        unique_name = base_object.unique_name
        assert unique_name in global_object.map._store
        assert unique_name in global_object.map._Map__type_dict

        # When
        global_object.map.prune(unique_name)

        # Then
        assert unique_name not in global_object.map._Map__type_dict
        # Note: _store entry may or may not exist depending on weak ref state

    def test_prune_nonexistent_key(self, clear):
        """Test that prune handles nonexistent key gracefully."""
        # When/Then - Should not raise error
        global_object.map.prune('nonexistent_key')

    def test_reset_type_unknown_object(self, clear):
        """Test reset_type with object not in map."""
        # Given
        unknown_obj = MagicMock()
        unknown_obj.unique_name = 'unknown'

        # When/Then - Should not raise error
        global_object.map.reset_type(unknown_obj, 'argument')

    def test_change_type_unknown_object(self, clear):
        """Test change_type with object not in map."""
        # Given
        unknown_obj = MagicMock()
        unknown_obj.unique_name = 'unknown'

        # When/Then - Should not raise error
        global_object.map.change_type(unknown_obj, 'argument')

    def test_find_path_start_not_in_graph(self, clear):
        """Test find_path when start vertex is not in graph."""
        # When
        path = global_object.map.find_path('nonexistent', 'also_nonexistent')

        # Then
        assert path == []

    def test_find_all_paths_start_not_in_graph(self, clear):
        """Test find_all_paths when start vertex is not in graph."""
        # When
        paths = global_object.map.find_all_paths('nonexistent', 'also_nonexistent')

        # Then
        assert paths == []

    def test_find_isolated_vertices(self, clear, base_object, parameter_object):
        """Test finding isolated vertices (vertices with no outgoing edges)."""
        # Given - No edges added, both objects are isolated

        # When
        isolated = global_object.map.find_isolated_vertices()

        # Then
        assert base_object.unique_name in isolated
        assert parameter_object.unique_name in isolated

    def test_find_isolated_vertices_with_edges(self, clear, base_object, parameter_object):
        """Test finding isolated vertices when some have edges."""
        # Given
        global_object.map.add_edge(base_object, parameter_object)

        # When
        isolated = global_object.map.find_isolated_vertices()

        # Then
        # base_object has an edge, so it's not isolated
        assert base_object.unique_name not in isolated
        # parameter_object has no outgoing edges, so it's isolated
        assert parameter_object.unique_name in isolated

    def test_prune_vertex_from_edge_edge_not_exists(self, clear, base_object, parameter_object):
        """Test pruning edge that doesn't exist."""
        # Given - No edge added between objects

        # When/Then - Should not raise error
        global_object.map.prune_vertex_from_edge(base_object, parameter_object)

    def test_prune_vertex_from_edge_parent_not_in_map(self, clear, parameter_object):
        """Test pruning edge when parent is not in map."""
        # Given
        unknown_obj = MagicMock()
        unknown_obj.unique_name = 'unknown'

        # When/Then - Should not raise error (vertex1 not in type_dict)
        global_object.map.prune_vertex_from_edge(unknown_obj, parameter_object)

    def test_created_internal_property(self, clear):
        """Test created_internal property."""
        # Given
        obj = NewBase(display_name='internal_obj')
        global_object.map.change_type(obj, 'created_internal')

        # When
        internal_objs = global_object.map.created_internal

        # Then
        assert obj.unique_name in internal_objs

    def test_clear_empties_both_dicts(self, clear, base_object, parameter_object):
        """Test that _clear() properly empties both _store and __type_dict."""
        # Given
        assert len(global_object.map._store) == 2
        assert len(global_object.map._Map__type_dict) == 2

        # When
        global_object.map._clear()

        # Then
        assert len(global_object.map._store) == 0
        assert len(global_object.map._Map__type_dict) == 0

    def test_entry_list_delitem(self):
        """Test _EntryList __delitem__ method."""
        # Given
        entry = _EntryList()
        entry.append('item1')
        entry.append('item2')
        entry.append('item3')

        # When
        del entry[1]

        # Then
        assert len(entry) == 2
        assert 'item2' not in entry
        assert 'item1' in entry
        assert 'item3' in entry

    def test_entry_list_repr_with_finalizer(self):
        """Test _EntryList repr when finalizer is set."""
        # Given
        entry = _EntryList()
        entry.type = 'created'
        entry.finalizer = MagicMock()  # Non-None finalizer

        # When
        repr_str = str(entry)

        # Then
        assert 'created' in repr_str
        assert 'With a finalizer' in repr_str

    def test_entry_list_remove_type_unknown(self):
        """Test removing a type that's not in known types."""
        # Given
        entry = _EntryList()
        entry.type = 'created'

        # When - Try to remove unknown type
        entry.remove_type('unknown_type')

        # Then - Should not change anything
        assert 'created' in entry.type
