#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from easyscience.Objects.Graph import Graph
from easyscience.Objects.Variable import Parameter
from easyscience.Objects.ObjectClasses import BaseObj
import pytest
from easyscience import borg

class TestGraph:
    @pytest.fixture
    def clear(self):
        borg.map._clear()

    def test_clear(self, clear):
        test_obj = BaseObj("test")
        assert len(borg.map._store) == 1
        assert len(borg.map._Graph__graph_dict) == 1
        assert borg.map._name_iterator_dict == {"BaseObj": 0}
        borg.map._clear()
        assert len(borg.map._store) == 0
        assert borg.map._Graph__graph_dict == {}
        assert borg.map._name_iterator_dict == {}

    def test_add_vertex(self, clear):
        test_obj = BaseObj("test")
        assert len(borg.map._store) == 1
        assert len(borg.map._Graph__graph_dict) == 1
        assert borg.map._name_iterator_dict == {"BaseObj": 0}

    @pytest.mark.parametrize("name", ["test", "test2", "test3"])
    def test_clear_fixture(self, name, clear):
        test_obj= BaseObj(name, unique_name=name)
        assert len(borg.map._store) == 1