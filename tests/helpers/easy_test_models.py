# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
Test-only models shared across the test suite.

``tests/conftest.py`` puts this directory on ``sys.path``, so tests
import from here with ``from easy_test_models import ...``. The module
name starts with ``easy`` on purpose: the serializer only treats a dict
as an EasyScience object (and dispatches to ``from_dict``) when its
``@module`` starts with ``easy``, so this keeps serializer round trips
working for these models.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from easyscience.base_classes import EasyList
from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter


class CoefficientModel(ModelBase):
    """
    A minimal model holding its parameters in an ``EasyList``.

    It exercises nested containers in serialization, copying, fitting
    parameter discovery and calculator bindings. Plain numbers become
    ``Parameter`` objects named ``c0``, ``c1``, ... The model evaluates
    as a polynomial with the coefficients in ``numpy.polyval`` order.

    Parameters
    ----------
    display_name : str | None, default=None
        Display name of the model.
    coefficients : Iterable[float | Parameter] | EasyList | None, default=None
        Initial coefficients, either numbers/``Parameter`` objects or a
        ready-made ``EasyList``.
    unique_name : str | None, default=None
        Unique name of the model.
    """

    def __init__(
        self,
        display_name: str | None = None,
        coefficients: Iterable[float | Parameter] | EasyList | None = None,
        unique_name: str | None = None,
    ):
        super().__init__(unique_name=unique_name, display_name=display_name)
        if isinstance(coefficients, EasyList):
            self._coefficients = coefficients
            return
        self._coefficients = EasyList(protected_types=Parameter)
        for index, item in enumerate(coefficients or []):
            if not isinstance(item, Parameter):
                item = Parameter(value=float(item), display_name=f'c{index}')
            self._coefficients.append(item)

    @property
    def coefficients(self) -> EasyList:
        """The coefficient parameters."""
        return self._coefficients

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.polyval([c.value for c in self.coefficients], x)


class PairModel(ModelBase):
    """
    Model holding two parameters as plain attributes.

    Used to check that a nested ``Parameter`` is serialized through its
    own ``to_dict``, so dependency information survives a round trip.

    Following the ``ModelBase.from_dict`` convention, a plain number is
    wrapped in a new ``Parameter`` while a ``Parameter`` is used as is.

    Parameters
    ----------
    a : float | Parameter, default=0.0
        First parameter.
    b : float | Parameter, default=0.0
        Second parameter, possibly dependent on ``a``.
    unique_name : str | None, default=None
        Unique name of the model.
    display_name : str | None, default=None
        Display name of the model.
    """

    def __init__(
        self,
        a: float | Parameter = 0.0,
        b: float | Parameter = 0.0,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        super().__init__(unique_name=unique_name, display_name=display_name)
        self._a = a if isinstance(a, Parameter) else Parameter(value=float(a), display_name='a')
        self._b = b if isinstance(b, Parameter) else Parameter(value=float(b), display_name='b')

    @property
    def a(self) -> Parameter:
        """First parameter."""
        return self._a

    @property
    def b(self) -> Parameter:
        """Second parameter."""
        return self._b
