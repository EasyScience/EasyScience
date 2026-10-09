# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easyscience import global_object
from easyscience.base_classes import NewBase


class NestedNewBase(NewBase):
    def __init__(self, unique_name=None, display_name=None, nested_new_base=None):
        super().__init__(unique_name=unique_name, display_name=display_name)
        self.nested_new_base = nested_new_base or NewBase()


class TestNewBase:
    @pytest.fixture
    def clear(self):
        # Clear the global object map before each test
        global_object.map._clear()

    def test_constructor_defaults(self, clear):
        # When Then
        obj = NewBase()
        # Expect
        assert obj._unique_name == 'NewBase_0'
        assert obj._default_unique_name is True
        assert obj._display_name is None
        assert global_object.map.get_item_by_key('NewBase_0') is obj

    def test_constructor_with_arguments(self):
        # When Then
        obj = NewBase(unique_name='custom_name', display_name='My Object')
        # Expect
        assert obj._unique_name == 'custom_name'
        assert obj._default_unique_name is False
        assert obj._display_name == 'My Object'
        assert global_object.map.get_item_by_key('custom_name') is obj

    def test_constructor_invalid_unique_name(self):
        # When Then Expect
        with pytest.raises(TypeError, match='Unique name has to be a string.'):
            NewBase(unique_name=123)

    def test_constructor_invalid_display_name(self):
        # When Then Expect
        with pytest.raises(TypeError, match='Display name must be a string or None'):
            NewBase(display_name=456)

    def test_arg_spec(self):
        # When
        obj = NewBase()
        # Then
        arg_spec = obj._arg_spec
        # Expect
        assert 'unique_name' in arg_spec
        assert 'display_name' in arg_spec

    def test_unique_name_setter(self, clear):
        # When
        obj = NewBase()
        # Then
        obj.unique_name = 'new_unique_name'
        # Expect
        assert obj.unique_name == 'new_unique_name'
        assert obj._default_unique_name is False
        assert global_object.map.get_item_by_key('new_unique_name') is obj
        assert global_object.map.get_item_by_key('NewBase_0') is obj

    def test_unique_name_setter_invalid(self):
        # When
        obj = NewBase()
        # Then Expect
        with pytest.raises(TypeError, match='Unique name has to be a string.'):
            obj.unique_name = 789

    def test_default_display_name(self):
        # When
        obj = NewBase()
        # Then
        display_name = obj.display_name
        # Expect
        assert display_name == obj.unique_name
        assert obj._display_name is None

    def test_display_name_setter(self):
        # When
        obj = NewBase()
        # Then
        obj.display_name = 'Pretty Name'
        # Expect
        assert obj.display_name == 'Pretty Name'
        assert obj._display_name == 'Pretty Name'

    def test_display_name_setter_invalid(self):
        # When
        obj = NewBase()
        # Then Expect
        with pytest.raises(TypeError, match='Display name must be a string or None'):
            obj.display_name = 101112

    def test_to_dict_full_params(self):
        # When
        obj = NewBase(unique_name='test_name', display_name='Test Object')
        # Then
        obj_dict = obj.to_dict()
        # Expect
        assert isinstance(obj_dict, dict)
        assert obj_dict['unique_name'] == 'test_name'
        assert obj_dict['display_name'] == 'Test Object'
        assert obj_dict['@module'] == 'easyscience.base_classes.new_base'
        assert obj_dict['@class'] == 'NewBase'
        assert '@version' in obj_dict

    def test_to_dict_default_params(self):
        # When
        obj = NewBase()
        # Then
        obj_dict = obj.to_dict()
        # Expect
        assert isinstance(obj_dict, dict)
        assert obj_dict['@module'] == 'easyscience.base_classes.new_base'
        assert obj_dict['@class'] == 'NewBase'
        assert '@version' in obj_dict
        assert 'unique_name' not in obj_dict
        assert 'display_name' not in obj_dict

    def test_to_dict_with_skip(self):
        # When
        obj = NewBase(unique_name='skip_test', display_name='Skip Test Object')
        # Then
        obj_dict = obj.to_dict(skip=['display_name'])
        # Expect
        assert isinstance(obj_dict, dict)
        assert obj_dict['unique_name'] == 'skip_test'
        assert 'display_name' not in obj_dict
        assert obj_dict['@module'] == 'easyscience.base_classes.new_base'
        assert obj_dict['@class'] == 'NewBase'
        assert '@version' in obj_dict

    def test_to_dict_nested_new_base(self):
        # When
        obj = NewBase()
        parent_obj = NestedNewBase(unique_name='parent', display_name='Parent Object')
        parent_obj.nested_new_base = obj
        # Then
        obj_dict = parent_obj.to_dict()
        # Expect
        assert isinstance(obj_dict, dict)
        assert 'nested_new_base' in obj_dict
        assert 'unique_name' in obj_dict
        assert 'display_name' in obj_dict
        assert 'unique_name' not in obj_dict['nested_new_base']
        assert 'display_name' not in obj_dict['nested_new_base']

    def test_to_dict_nested_new_base_reverse(self):
        # When
        obj = NewBase(unique_name='child', display_name='Child Object')
        parent_obj = NestedNewBase()
        parent_obj.nested_new_base = obj
        # Then
        obj_dict = parent_obj.to_dict()
        # Expect
        assert isinstance(obj_dict, dict)
        assert 'nested_new_base' in obj_dict
        assert 'unique_name' not in obj_dict
        assert 'display_name' not in obj_dict
        assert 'unique_name' in obj_dict['nested_new_base']
        assert 'display_name' in obj_dict['nested_new_base']


    def test_from_dict(self):
        # When
        obj_dict = {
            '@module': 'easyscience.base_classes.new_base',
            '@class': 'NewBase',
            'unique_name': 'from_dict_name',
            'display_name': 'From Dict Object',
        }
        # Then
        obj = NewBase.from_dict(obj_dict)
        # Expect
        assert isinstance(obj, NewBase)
        assert obj.unique_name == 'from_dict_name'
        assert obj.display_name == 'From Dict Object'
        assert global_object.map.get_item_by_key('from_dict_name') is obj

    def test_from_dict_not_easyscience(self):
        # When
        obj_dict = {
            '@module': 'some.other.module',
            '@class': 'NewBase',
            'unique_name': 'invalid_from_dict',
            'display_name': 'Invalid From Dict Object',
        }
        # Then Expect
        with pytest.raises(
            ValueError, match='Input must be a dictionary representing an EasyScience object.'
        ):
            NewBase.from_dict(obj_dict)

    def test_from_dict_wrong_class(self):
        # When
        obj_dict = {
            '@module': 'easyscience.base_classes.new_base',
            '@class': 'SomeOtherClass',
            'unique_name': 'wrong_class_name',
            'display_name': 'Wrong Class Object',
        }
        # Then Expect
        with pytest.raises(
            ValueError,
            match='Class name in dictionary does not match the expected class: NewBase.',
        ):
            NewBase.from_dict(obj_dict)

    def test__dir__(self):
        # When
        obj = NewBase()
        dir_list = dir(obj)
        # Then
        expected_attributes = [
            'to_dict',
            'unique_name',
            'display_name',
            'from_dict',
        ]
        # Expect
        for attr in expected_attributes:
            assert attr in dir_list

    def test_copy(self):
        # When
        obj = NewBase(unique_name='original_name', display_name='Original Object')
        # Then
        obj_copy = obj.__copy__()
        # Expect
        assert isinstance(obj_copy, NewBase)
        assert obj_copy.unique_name != obj.unique_name
        assert obj_copy.display_name == obj.display_name
        assert global_object.map.get_item_by_key(obj_copy.unique_name) is obj_copy

    def test_deepcopy(self):
        # When
        obj = NewBase(unique_name='deepcopy_name', display_name='Deep Copy Object')
        # Then
        import copy

        obj_deepcopy = copy.deepcopy(obj)
        # Expect
        assert isinstance(obj_deepcopy, NewBase)
        assert obj_deepcopy.unique_name != obj.unique_name
        assert obj_deepcopy.display_name == obj.display_name
        assert global_object.map.get_item_by_key(obj_deepcopy.unique_name) is obj_deepcopy
