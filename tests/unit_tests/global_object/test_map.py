#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from easyscience.global_object.map import Map
from easyscience.Objects.Variable import Parameter
from easyscience.Objects.ObjectClasses import BaseObj
import pytest
import gc
from easyscience import global_object

class TestMap:
    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_clear(self, clear):
        test_obj = BaseObj(name="test")
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        assert global_object.map._name_iterator_dict == {"BaseObj": 0}
        global_object.map._clear()
        assert len(global_object.map._store) == 0
        assert global_object.map._Map__type_dict == {}
        assert global_object.map._name_iterator_dict == {}

    def test_add_vertex(self, clear):
        test_obj = BaseObj(name="test")
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        assert global_object.map._name_iterator_dict == {"BaseObj": 0}

    def test_weakref(self, clear):
        test_obj = BaseObj(name="test")
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        del test_obj
        gc.collect()
        assert len(global_object.map._store) == 0
        assert len(global_object.map._Map__type_dict) == 0

    def test_vertices(self, clear):
        test_obj = BaseObj(name="test")
        test_obj2 = Parameter(value=2.0, name="test2")
        assert global_object.map.vertices() == ["BaseObj_0", "Parameter_0"]

    def test_get_item_by_key(self, clear):
        test_obj = BaseObj(name="test")
        test_obj2 = Parameter(value=2.0, name="test2")
        assert global_object.map.get_item_by_key(test_obj.unique_name) == test_obj
        assert global_object.map.get_item_by_key(test_obj2.unique_name) == test_obj2

    def test_get_name_iterator(self, clear):
        assert global_object.map._get_name_iterator("BaseObj") == 0
        assert global_object.map._get_name_iterator("Parameter") == 0
        test_obj = BaseObj(name="test")
        test_obj2 = Parameter(value=2.0, name="test2")
        assert global_object.map._get_name_iterator("BaseObj") == 2
        assert global_object.map._get_name_iterator("Parameter") == 2

    @pytest.mark.parametrize("cls, kwargs", [(BaseObj, {}), (Parameter, {"value": 2.0})])
    def test_identical_unique_names_exception(self, clear, cls, kwargs):
        test_obj = cls(name="test", unique_name="test", **kwargs)
        with pytest.raises(ValueError):
            test_obj2 = cls(name="test2", unique_name="test", **kwargs)

    # test unique_name change

