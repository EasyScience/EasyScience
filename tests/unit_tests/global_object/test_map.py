#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from easyscience.global_object.map import Map
from easyscience.Objects.Variable import Parameter
from easyscience.Objects.ObjectClasses import BaseObj
import pytest
from easyscience import global_object

class TestMap:
    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_clear(self, clear):
        test_obj = BaseObj("test")
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        assert global_object.map._name_iterator_dict == {"BaseObj": 0}
        global_object.map._clear()
        assert len(global_object.map._store) == 0
        assert global_object.map._Map__type_dict == {}
        assert global_object.map._name_iterator_dict == {}

    def test_add_vertex(self, clear):
        test_obj = BaseObj("test")
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        assert global_object.map._name_iterator_dict == {"BaseObj": 0}

    @pytest.mark.parametrize("name", ["test", "test2", "test3"])
    def test_clear_fixture(self, name, clear):
        test_obj= BaseObj(name, unique_name=name)
        assert len(global_object.map._store) == 1