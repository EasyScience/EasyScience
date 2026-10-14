# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easyscience import Parameter
from easyscience import global_object
from easyscience.global_object.global_object import GlobalObject
from easyscience.variable import DescriptorBool


class TestGlobalObject:
    def test_init(self):
        # When Then
        global_object = GlobalObject()

        # Expect
        assert global_object.log == GlobalObject().log
        assert global_object.map == GlobalObject().map
        assert global_object.stack == GlobalObject().stack
        assert global_object.debug == GlobalObject().debug

    def test_generate_unique_name(self):
        # When Then
        global_object = GlobalObject()
        name = global_object.generate_unique_name('name_prefix')

        # Expect
        assert name == 'name_prefix_0'

    def test_generate_unique_name_already_taken(self):
        # When
        global_object = GlobalObject()
        # Block the other_name_prefix_2 name
        keep_due_toweakref_1 = DescriptorBool(
            display_name='test', value=True, unique_name='other_name_prefix_2'
        )
        keep_due_toweakref_2 = DescriptorBool(
            display_name='test', value=True, unique_name='other_name_prefix_a_3'
        )
        keep_due_toweakref_3 = DescriptorBool(
            display_name='test', value=True, unique_name='almost_other_name_prefix_3'
        )

        # Then
        name = global_object.generate_unique_name('other_name_prefix')

        # Expect
        assert name == 'other_name_prefix_3'

    @pytest.fixture
    def clear_global_map(self):
        """Clear global map before and after each test"""
        global_object.map._clear()
        yield
        global_object.map._clear()

    def test_singleton_behavior(self):
        """Test that GlobalObject is truly a singleton"""
        # When
        obj1 = GlobalObject()
        obj2 = GlobalObject()

        # Then
        assert obj1 is obj2
        assert id(obj1) == id(obj2)
        assert obj1.map is obj2.map
        assert obj1.log is obj2.log

    def test_instantiate_stack(self, clear_global_map):
        """Test stack instantiation"""
        # Given
        global_obj = GlobalObject()

        # When
        global_obj.instantiate_stack()

        # Then
        assert global_obj.stack is not None
        from easyscience.global_object.undo_redo import UndoStack

        assert isinstance(global_obj.stack, UndoStack)

    def test_debug_property(self):
        """Test debug property access"""
        # Given
        global_obj = GlobalObject()

        # When/Then
        assert global_obj.debug is False  # Default value

        # Test that it's the same across instances
        global_obj2 = GlobalObject()
        assert global_obj.debug is global_obj2.debug

    def test_generate_unique_name_empty_map(self, clear_global_map):
        """Test unique name generation with empty map"""
        # Given
        global_obj = GlobalObject()

        # When
        name = global_obj.generate_unique_name('test')

        # Then
        assert name == 'test_0'

    def test_generate_unique_name_with_gaps(self, clear_global_map):
        """Test unique name generation with gaps in numbering"""
        # Given
        global_obj = GlobalObject()
        # Create objects with non-sequential names
        keep1 = DescriptorBool(display_name='test', value=True, unique_name='prefix_0')
        keep2 = DescriptorBool(display_name='test', value=True, unique_name='prefix_2')
        keep3 = DescriptorBool(display_name='test', value=True, unique_name='prefix_5')

        # When
        name = global_obj.generate_unique_name('prefix')

        # Then - Should pick the highest + 1
        assert name == 'prefix_6'

    def test_generate_unique_name_non_numeric_suffix(self, clear_global_map):
        """Test that non-numeric suffixes are ignored"""
        # Given
        global_obj = GlobalObject()
        keep1 = DescriptorBool(display_name='test', value=True, unique_name='prefix_abc')
        keep2 = DescriptorBool(display_name='test', value=True, unique_name='prefix_1')

        # When
        name = global_obj.generate_unique_name('prefix')

        # Then
        assert name == 'prefix_2'

    def test_generate_unique_name_similar_prefixes(self, clear_global_map):
        """Test that similar but different prefixes don't interfere"""
        # Given
        global_obj = GlobalObject()
        keep1 = DescriptorBool(display_name='test', value=True, unique_name='test_param_5')
        keep2 = DescriptorBool(display_name='test', value=True, unique_name='test_parameter_10')

        # When
        name1 = global_obj.generate_unique_name('test_param')
        name2 = global_obj.generate_unique_name('test_parameter')

        # Then
        assert name1 == 'test_param_6'
        assert name2 == 'test_parameter_11'

    def test_integration_with_parameter_creation(self, clear_global_map):
        """Test GlobalObject integration with Parameter creation"""
        # Given
        global_obj = GlobalObject()

        # When
        param1 = Parameter(display_name='test1', value=1.0)
        param2 = Parameter(display_name='test2', value=2.0)

        # Then
        assert len(global_obj.map.vertices()) == 2
        assert param1.unique_name in global_obj.map.vertices()
        assert param2.unique_name in global_obj.map.vertices()

        # Test retrieval
        retrieved1 = global_obj.map.get_item_by_key(param1.unique_name)
        assert retrieved1 is param1
