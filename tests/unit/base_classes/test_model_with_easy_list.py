# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Behaviour of a ModelBase that holds its parameters in an EasyList."""

import copy

import numpy as np
import pytest
from easy_test_models import CoefficientModel

from easyscience import global_object
from easyscience.base_classes import EasyList
from easyscience.variable import Parameter


@pytest.fixture(autouse=True)
def clear():
    """Clear global object map before each test."""
    global_object.map._clear()


@pytest.mark.parametrize(
    'coefficients',
    [(1.0,), (1.0, 2.0), (1.0, 2.0, 3.0), (-1.0, -2.0, -3.0), (0.72, 6.48, -0.48)],
)
def test_numbers_become_parameters(coefficients):
    model = CoefficientModel(coefficients=coefficients)

    assert all(isinstance(c, Parameter) for c in model.coefficients)
    assert [c.value for c in model.coefficients] == list(coefficients)
    assert [c.display_name for c in model.coefficients] == [
        f'c{i}' for i in range(len(coefficients))
    ]
    x = np.linspace(0, 10, 100)
    assert np.allclose(model(x), np.polyval(coefficients, x))


def test_default_initialization():
    model = CoefficientModel(display_name='empty')

    assert model.display_name == 'empty'
    assert len(model.coefficients) == 0
    assert model.coefficients._protected_types == [Parameter]


def test_parameters_are_kept():
    p0 = Parameter(value=5.0, display_name='a')
    p1 = Parameter(value=2.0, display_name='b')

    model = CoefficientModel(coefficients=[p0, p1, 1.0])

    assert model.coefficients[0] is p0
    assert model.coefficients[1] is p1
    assert model.coefficients[2].display_name == 'c2'


def test_easy_list_is_used_directly():
    collection = EasyList(Parameter(value=1.0), Parameter(value=2.0), protected_types=Parameter)

    model = CoefficientModel(coefficients=collection)

    assert model.coefficients is collection


def test_get_fit_parameters():
    """Parameters inside the EasyList are exposed for fitting."""
    model = CoefficientModel(coefficients=[1.0, 2.0])

    assert model.get_fit_parameters() == list(model.coefficients)


def test_dict_round_trip():
    model = CoefficientModel(display_name='round_trip', coefficients=[1.0, 2.0, 3.0])

    new_model = CoefficientModel.from_dict(model.to_dict())

    assert new_model.display_name == 'round_trip'
    assert [c.value for c in new_model.coefficients] == [1.0, 2.0, 3.0]
    assert [c.display_name for c in new_model.coefficients] == ['c0', 'c1', 'c2']


def test_dict_round_trip_unnamed_model_keeps_coefficient_names():
    """The model's own missing names must not be skipped on nested items."""
    model = CoefficientModel(
        coefficients=[Parameter(1.0, display_name='c0', unique_name='my_c0'), 2.0]
    )

    d = model.to_dict()
    global_object.map._clear()  # the explicit name is kept, so free it first
    new_model = CoefficientModel.from_dict(d)

    assert 'unique_name' not in d
    assert 'display_name' not in d
    assert [c.display_name for c in new_model.coefficients] == ['c0', 'c1']
    assert new_model.coefficients[0].unique_name == 'my_c0'


@pytest.mark.parametrize('copier', [copy.copy, copy.deepcopy], ids=['copy', 'deepcopy'])
def test_copy_with_named_coefficients(copier):
    """Copies get fresh coefficient identities, even for explicit names."""
    model = CoefficientModel(
        coefficients=[Parameter(1.0, unique_name='coefficient'), Parameter(2.0)]
    )

    data = model.to_dict(skip=['unique_name'])
    assert all('unique_name' not in c for c in data['coefficients']['data'])

    new_model = copier(model)

    assert [c.value for c in new_model.coefficients] == [1.0, 2.0]
    assert new_model.coefficients[0].unique_name != 'coefficient'
    assert all(new is not old for new, old in zip(new_model.coefficients, model.coefficients))
