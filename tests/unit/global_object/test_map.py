# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import gc

import pytest

from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import NewBase
from easyscience.global_object.map import Map


class _Named:
    """Minimal weak-referenceable object that can be registered in a Map."""

    def __init__(self, unique_name):
        self.unique_name = unique_name


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
        assert global_object.map.objects() == [base_object, parameter_object]

    def test_clear(self, clear, base_object):
        # When
        assert global_object.map.vertices() == [base_object.unique_name]
        # Then
        global_object.map._clear()
        # Expect
        assert global_object.map.vertices() == []
        assert global_object.map.objects() == []

    def test_weakref(self, clear):
        # When
        test_obj = NewBase(display_name='test')
        assert global_object.map.vertices() == [test_obj.unique_name]
        # Then
        del test_obj
        gc.collect()
        # Expect
        assert global_object.map.vertices() == []

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

    def test_get_item_by_key_not_found(self, clear):
        """Test getting item by non-existent key"""
        # When/Then
        with pytest.raises(ValueError, match='Item not in map'):
            global_object.map.get_item_by_key('non_existent')

    @pytest.mark.parametrize(
        'cls, kwargs', [(NewBase, {'display_name': 'test'}), (Parameter, {'value': 2.0})]
    )
    def test_identical_unique_names_exception(self, clear, cls, kwargs):
        # When
        test_obj = cls(unique_name='test', **kwargs)
        # Then Expect
        with pytest.raises(ValueError):
            test_obj2 = cls(unique_name='test', **kwargs)

    def test_add_vertex_duplicate_name_error(self, clear, base_object):
        """Test that adding vertex with duplicate name raises error"""
        # When/Then
        with pytest.raises(ValueError, match='already exists'):
            # Try to add another object with same unique_name
            duplicate_obj = NewBase(display_name='duplicate', unique_name=base_object.unique_name)

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

    def test_prune_vertex(self, clear, base_object):
        """Test pruning a vertex completely"""
        # Given
        unique_name = base_object.unique_name
        assert unique_name in global_object.map.vertices()

        # When
        global_object.map.prune(unique_name)

        # Then
        assert unique_name not in global_object.map.vertices()
        assert base_object not in global_object.map.objects()

    def test_prune_nonexistent_key(self, clear):
        """Test that prune handles nonexistent key gracefully."""
        # When/Then - Should not raise error
        global_object.map.prune('nonexistent_key')

    def test_map_repr(self, clear, base_object, parameter_object):
        """Test map string representation"""
        # When
        repr_str = str(global_object.map)

        # Then
        assert 'Map object' in repr_str
        assert '2' in repr_str  # Should show vertex count

    def test_map_initialization(self):
        """Test Map initialization"""
        # When
        test_map = Map()

        # Then
        assert len(test_map.vertices()) == 0

    def test_prune_then_reuse_name(self):
        """Pruning a reused name removes the new object, even after the old one is collected."""
        # Given
        test_map = Map()
        obj_a = _Named('n')
        test_map.add_vertex(obj_a)
        test_map.prune('n')
        obj_b = _Named('n')
        test_map.add_vertex(obj_b)

        # When
        del obj_a
        gc.collect()
        test_map.prune('n')

        # Then
        assert 'n' not in test_map.vertices()
        assert obj_b not in test_map.objects()

    def test_clear_then_reuse_name(self):
        """Pruning a name reused after clearing removes the new object."""
        # Given
        test_map = Map()
        obj_a = _Named('n')
        test_map.add_vertex(obj_a)
        test_map._clear()
        obj_b = _Named('n')
        test_map.add_vertex(obj_b)

        # When
        del obj_a
        gc.collect()
        test_map.prune('n')

        # Then
        assert 'n' not in test_map.vertices()
        assert obj_b not in test_map.objects()

    def test_objects_contains_live_objects_only(self):
        """Objects that have been collected are not returned."""
        # Given
        test_map = Map()
        kept = _Named('kept')
        released = _Named('released')
        test_map.add_vertex(kept)
        test_map.add_vertex(released)

        # When
        del released
        gc.collect()

        # Then
        assert test_map.objects() == [kept]

    def test_objects_lists_renamed_object_once_per_name(self, clear, base_object):
        """A renamed object keeps its old name, so it is returned once for each name."""
        # When
        base_object.unique_name = 'renamed'

        # Then
        objects = global_object.map.objects()
        assert len(objects) == 2
        assert all(obj is base_object for obj in objects)
