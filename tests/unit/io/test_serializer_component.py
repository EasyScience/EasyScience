# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import Type

import numpy as np
import pytest

import easyscience
from easyscience import DescriptorNumber
from easyscience import Parameter

dp_param_dict = {
    'argnames': 'dp_kwargs, dp_cls',
    'argvalues': (
        [
            {
                '@module': DescriptorNumber.__module__,
                '@class': DescriptorNumber.__name__,
                '@version': easyscience.__version__,
                'name': 'test',
                'value': 1.0,
                'variance': None,
                'unit': 'dimensionless',
                'description': '',
                'url': '',
                'display_name': 'test',
            },
            DescriptorNumber,
        ],
        [
            {
                '@module': Parameter.__module__,
                '@class': Parameter.__name__,
                '@version': easyscience.__version__,
                'name': 'test',
                'unit': 'km',
                'value': 1.0,
                'variance': 0.0,
                'min': -np.inf,
                'max': np.inf,
                'fixed': False,
                'url': 'https://www.boo.com',
                'description': '',
                'display_name': 'test',
            },
            Parameter,
        ],
    ),
    'ids': ['DescriptorNumber', 'Parameter'],
}

_skip_opt = [[], None] + [k for k in dp_param_dict['argvalues'][0][0].keys() if k[0] != '@']
skip_dict = {
    'argnames': 'skip',
    'argvalues': _skip_opt,
    'ids': ['skip_' + str(opt) for opt in _skip_opt],
}


def check_dict(check, item):
    if isinstance(check, dict) and isinstance(item, dict):
        for key in check.keys():
            assert key in item.keys()
            check_dict(check[key], item[key])
    else:
        assert isinstance(item, type(check))
        assert item == check


@pytest.mark.parametrize(**skip_dict)
@pytest.mark.parametrize(**dp_param_dict)
def test_variable_to_dict_methods(dp_kwargs: dict, dp_cls: Type[DescriptorNumber], skip):
    data_dict = {k: v for k, v in dp_kwargs.items() if k[0] != '@'}

    obj = dp_cls(**data_dict)

    dp_kwargs = deepcopy(dp_kwargs)

    if isinstance(skip, str):
        del dp_kwargs[skip]

    if not isinstance(skip, list):
        skip = [skip]

    enc = obj.to_dict(skip=skip)

    expected_keys = set(dp_kwargs.keys())
    obtained_keys = set(enc.keys())

    dif = expected_keys.difference(obtained_keys)

    assert len(dif) == 0

    check_dict(dp_kwargs, enc)
