# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easyscience import global_object
from easyscience.base_classes import EasyBase


class TestEasyBase:
    @pytest.fixture
    def clear(self):
        # Clear the global object registry before each test
        global_object.registry.clear()

    def test_constructor_defaults(self, clear):
        # When Then
        obj = EasyBase()
        # Expect
        assert obj._unique_name == 'EasyBase_0'
        assert obj._default_unique_name is True
        assert obj._display_name is None
        assert global_object.registry['EasyBase_0'] is obj

    def test_constructor_with_arguments(self):
        # When Then
        obj = EasyBase(unique_name='custom_name', display_name='My Object')
        # Expect
        assert obj._unique_name == 'custom_name'
        assert obj._default_unique_name is False
        assert obj._display_name == 'My Object'
        assert global_object.registry['custom_name'] is obj

    def test_constructor_invalid_unique_name(self):
        # When Then Expect
        with pytest.raises(TypeError, match='Unique name has to be a string.'):
            EasyBase(unique_name=123)

    def test_constructor_invalid_display_name(self):
        # When Then Expect
        with pytest.raises(TypeError, match='Display name must be a string or None'):
            EasyBase(display_name=456)

    def test_no_arg_spec_override(self):
        # When
        obj = EasyBase()
        # Expect
        assert not hasattr(obj, '_arg_spec')

    def test_to_dict_includes_keyword_only_args(self, clear):
        # When
        class KeywordOnly(EasyBase):
            def __init__(self, value, *args, unit='', unique_name=None, **kwargs):
                super().__init__(unique_name=unique_name)
                self.value = value
                self.unit = unit

        obj = KeywordOnly(1.0, unit='m', unique_name='kw')
        # Then
        d = obj.to_dict()
        # Expect
        assert d['value'] == 1.0
        assert d['unit'] == 'm'
        assert d['unique_name'] == 'kw'

    def test_unique_name_setter(self, clear):
        # When
        obj = EasyBase()
        # Then
        obj.unique_name = 'new_unique_name'
        # Expect
        assert obj.unique_name == 'new_unique_name'
        assert obj._default_unique_name is False
        assert global_object.registry['new_unique_name'] is obj
        assert 'EasyBase_0' not in global_object.registry

    def test_unique_name_setter_invalid(self):
        # When
        obj = EasyBase()
        # Then Expect
        with pytest.raises(TypeError, match='Unique name has to be a string.'):
            obj.unique_name = 789

    def test_default_display_name(self):
        # When
        obj = EasyBase()
        # Then
        display_name = obj.display_name
        # Expect
        assert display_name == obj.unique_name
        assert obj._display_name is None

    def test_display_name_setter(self):
        # When
        obj = EasyBase()
        # Then
        obj.display_name = 'Pretty Name'
        # Expect
        assert obj.display_name == 'Pretty Name'
        assert obj._display_name == 'Pretty Name'

    def test_display_name_setter_invalid(self):
        # When
        obj = EasyBase()
        # Then Expect
        with pytest.raises(TypeError, match='Display name must be a string or None'):
            obj.display_name = 101112

    def test_to_dict_full_params(self):
        # When
        obj = EasyBase(unique_name='test_name', display_name='Test Object')
        # Then
        obj_dict = obj.to_dict()
        # Expect
        assert isinstance(obj_dict, dict)
        assert obj_dict['unique_name'] == 'test_name'
        assert obj_dict['display_name'] == 'Test Object'
        assert obj_dict['@module'] == 'easyscience.base_classes.easy_base'
        assert obj_dict['@class'] == 'EasyBase'
        assert '@version' in obj_dict

    def test_to_dict_default_params(self):
        # When
        obj = EasyBase()
        # Then
        obj_dict = obj.to_dict()
        # Expect
        assert isinstance(obj_dict, dict)
        assert obj_dict['@module'] == 'easyscience.base_classes.easy_base'
        assert obj_dict['@class'] == 'EasyBase'
        assert '@version' in obj_dict
        assert 'unique_name' not in obj_dict
        assert 'display_name' not in obj_dict

    def test_to_dict_does_not_mutate_skip(self):
        obj = EasyBase()
        skip = ['foo']
        obj.to_dict(skip=skip)
        assert skip == ['foo']

    def test_to_dict_nested_object_keeps_own_names(self):
        """A parent without names must not strip names from a nested object."""

        class Holder(EasyBase):
            def __init__(self, item=None, unique_name=None, display_name=None):
                super().__init__(unique_name=unique_name, display_name=display_name)
                self._item = item

            @property
            def item(self):
                return self._item

        inner = EasyBase(unique_name='inner', display_name='Inner')
        outer = Holder(item=inner)  # auto unique_name, no display_name

        d = outer.to_dict()

        assert 'unique_name' not in d
        assert 'display_name' not in d
        assert d['item']['unique_name'] == 'inner'
        assert d['item']['display_name'] == 'Inner'

    def test_to_dict_with_skip(self):
        # When
        obj = EasyBase(unique_name='skip_test', display_name='Skip Test Object')
        # Then
        obj_dict = obj.to_dict(skip=['display_name'])
        # Expect
        assert isinstance(obj_dict, dict)
        assert obj_dict['unique_name'] == 'skip_test'
        assert 'display_name' not in obj_dict
        assert obj_dict['@module'] == 'easyscience.base_classes.easy_base'
        assert obj_dict['@class'] == 'EasyBase'
        assert '@version' in obj_dict

    def test_from_dict(self):
        # When
        obj_dict = {
            '@module': 'easyscience.base_classes.easy_base',
            '@class': 'EasyBase',
            'unique_name': 'from_dict_name',
            'display_name': 'From Dict Object',
        }
        # Then
        obj = EasyBase.from_dict(obj_dict)
        # Expect
        assert isinstance(obj, EasyBase)
        assert obj.unique_name == 'from_dict_name'
        assert obj.display_name == 'From Dict Object'
        assert global_object.registry['from_dict_name'] is obj

    def test_from_dict_not_easyscience(self):
        # When
        obj_dict = {
            '@module': 'some.other.module',
            '@class': 'EasyBase',
            'unique_name': 'invalid_from_dict',
            'display_name': 'Invalid From Dict Object',
        }
        # Then Expect
        with pytest.raises(
            ValueError, match='Input must be a dictionary representing an EasyScience object.'
        ):
            EasyBase.from_dict(obj_dict)

    def test_from_dict_wrong_class(self):
        # When
        obj_dict = {
            '@module': 'easyscience.base_classes.easy_base',
            '@class': 'SomeOtherClass',
            'unique_name': 'wrong_class_name',
            'display_name': 'Wrong Class Object',
        }
        # Then Expect
        with pytest.raises(
            ValueError,
            match='Class name in dictionary does not match the expected class: EasyBase.',
        ):
            EasyBase.from_dict(obj_dict)

    def test__dir__(self):
        # When
        obj = EasyBase()
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
        obj = EasyBase(unique_name='original_name', display_name='Original Object')
        # Then
        obj_copy = obj.__copy__()
        # Expect
        assert isinstance(obj_copy, EasyBase)
        assert obj_copy.unique_name != obj.unique_name
        assert obj_copy.display_name == obj.display_name
        assert global_object.registry[obj_copy.unique_name] is obj_copy

    def test_deepcopy(self):
        # When
        obj = EasyBase(unique_name='deepcopy_name', display_name='Deep Copy Object')
        # Then
        import copy

        obj_deepcopy = copy.deepcopy(obj)
        # Expect
        assert isinstance(obj_deepcopy, EasyBase)
        assert obj_deepcopy.unique_name != obj.unique_name
        assert obj_deepcopy.display_name == obj.display_name
        assert global_object.registry[obj_deepcopy.unique_name] is obj_deepcopy
