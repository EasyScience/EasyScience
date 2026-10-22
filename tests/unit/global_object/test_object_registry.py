# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import gc

import pytest

from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import NewBase
from easyscience.global_object.object_registry import ObjectRegistry


class _Named:
    """Minimal weak-referenceable object that can be registered in an ObjectRegistry."""

    def __init__(self, unique_name):
        self.unique_name = unique_name


class TestObjectRegistry:
    @pytest.fixture
    def clear(self):
        global_object.registry.clear()

    @pytest.fixture
    def base_object(self):
        return NewBase(display_name='test')

    @pytest.fixture
    def parameter_object(self):
        return Parameter(display_name='test2', value=2)

    def test_add(self, clear, base_object, parameter_object):
        # When Then Expect
        assert global_object.registry.objects() == [base_object, parameter_object]

    def test_clear(self, clear, base_object):
        # When
        assert global_object.registry.names() == [base_object.unique_name]
        # Then
        global_object.registry.clear()
        # Expect
        assert global_object.registry.names() == []
        assert global_object.registry.objects() == []

    def test_weakref(self, clear):
        # When
        test_obj = NewBase(display_name='test')
        assert global_object.registry.names() == [test_obj.unique_name]
        # Then
        del test_obj
        gc.collect()
        # Expect
        assert global_object.registry.names() == []

    def test_names(self, clear, base_object, parameter_object):
        # When Then Expect
        assert global_object.registry.names() == [
            base_object.unique_name,
            parameter_object.unique_name,
        ]

    def test_getitem(self, clear, base_object, parameter_object):
        # When Then Expect
        assert global_object.registry[base_object.unique_name] == base_object
        assert global_object.registry[parameter_object.unique_name] == parameter_object

    def test_getitem_not_found(self, clear):
        """Test getting item by non-existent key"""
        # When/Then
        with pytest.raises(KeyError):
            global_object.registry['non_existent']

    def test_contains_len_iter(self, clear, base_object, parameter_object):
        # When Then Expect
        assert base_object.unique_name in global_object.registry
        assert 'non_existent' not in global_object.registry
        assert len(global_object.registry) == 2
        assert list(global_object.registry) == [
            base_object.unique_name,
            parameter_object.unique_name,
        ]

    def test_iter_survives_collection_mid_loop(self):
        """Iteration uses a snapshot, so collecting an object mid-loop is safe."""
        # Given
        test_registry = ObjectRegistry()
        objs = [_Named('a'), _Named('b')]
        for obj in objs:
            test_registry.add(obj)
        del obj

        # When
        seen = []
        for name in test_registry:
            seen.append(name)
            objs.clear()
            gc.collect()

        # Then
        assert seen == ['a', 'b']
        assert len(test_registry) == 0

    @pytest.mark.parametrize(
        'cls, kwargs', [(NewBase, {'display_name': 'test'}), (Parameter, {'value': 2.0})]
    )
    def test_identical_unique_names_exception(self, clear, cls, kwargs):
        # When
        test_obj = cls(unique_name='test', **kwargs)
        # Then Expect
        with pytest.raises(ValueError):
            test_obj2 = cls(unique_name='test', **kwargs)

    def test_add_duplicate_name_error(self, clear, base_object):
        """Test that adding an object with a duplicate name raises error"""
        # When/Then
        with pytest.raises(ValueError, match='already exists'):
            # Try to add another object with same unique_name
            duplicate_obj = NewBase(display_name='duplicate', unique_name=base_object.unique_name)

    def test_unique_name_change_still_in_registry(self, clear, base_object, parameter_object):
        # When
        assert global_object.registry['NewBase_0'] == base_object
        assert global_object.registry['Parameter_0'] == parameter_object
        # Then
        base_object.unique_name = 'test3'
        parameter_object.unique_name = 'test4'
        # Expect
        assert global_object.registry['NewBase_0'] == base_object
        assert global_object.registry['Parameter_0'] == parameter_object
        assert global_object.registry['test3'] == base_object
        assert global_object.registry['test4'] == parameter_object

    def test_discard(self, clear, base_object):
        """Test discarding a name completely"""
        # Given
        unique_name = base_object.unique_name
        assert unique_name in global_object.registry

        # When
        global_object.registry.discard(unique_name)

        # Then
        assert unique_name not in global_object.registry
        assert base_object not in global_object.registry.objects()

    def test_discard_nonexistent_key(self, clear):
        """Test that discard handles nonexistent key gracefully."""
        # When/Then - Should not raise error
        global_object.registry.discard('nonexistent_key')

    def test_registry_repr(self, clear, base_object, parameter_object):
        """Test registry string representation"""
        # When
        repr_str = str(global_object.registry)

        # Then
        assert 'ObjectRegistry object' in repr_str
        assert '2' in repr_str  # Should show entry count

    def test_registry_initialization(self):
        """Test ObjectRegistry initialization"""
        # When
        test_registry = ObjectRegistry()

        # Then
        assert len(test_registry) == 0

    def test_discard_then_reuse_name(self):
        """Discarding a reused name removes the new object, even after the old one is collected."""
        # Given
        test_registry = ObjectRegistry()
        obj_a = _Named('n')
        test_registry.add(obj_a)
        test_registry.discard('n')
        obj_b = _Named('n')
        test_registry.add(obj_b)

        # When
        del obj_a
        gc.collect()
        test_registry.discard('n')

        # Then
        assert 'n' not in test_registry
        assert obj_b not in test_registry.objects()

    def test_clear_then_reuse_name(self):
        """Discarding a name reused after clearing removes the new object."""
        # Given
        test_registry = ObjectRegistry()
        obj_a = _Named('n')
        test_registry.add(obj_a)
        test_registry.clear()
        obj_b = _Named('n')
        test_registry.add(obj_b)

        # When
        del obj_a
        gc.collect()
        test_registry.discard('n')

        # Then
        assert 'n' not in test_registry
        assert obj_b not in test_registry.objects()

    def test_objects_contains_live_objects_only(self):
        """Objects that have been collected are not returned."""
        # Given
        test_registry = ObjectRegistry()
        kept = _Named('kept')
        released = _Named('released')
        test_registry.add(kept)
        test_registry.add(released)

        # When
        del released
        gc.collect()

        # Then
        assert test_registry.objects() == [kept]

    def test_objects_lists_renamed_object_once_per_name(self, clear, base_object):
        """A renamed object keeps its old name, so it is returned once for each name."""
        # When
        base_object.unique_name = 'renamed'

        # Then
        objects = global_object.registry.objects()
        assert len(objects) == 2
        assert all(obj is base_object for obj in objects)
