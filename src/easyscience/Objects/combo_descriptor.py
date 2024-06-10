#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import numbers
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import TypeVar
from typing import Union

import numpy as np

from easyscience import ureg
from easyscience.Objects import Descriptor
from easyscience.Utils.classTools import addProp
from easyscience.Utils.UndoRedo import property_stack_deco

Q_ = ureg.Quantity
V = TypeVar('V', bound=Descriptor)

class ComboDescriptor(Descriptor):
    """
    This class is an extension of a ``EasyScience.Object.Base.Descriptor``. This class has a selection of values which can
    be checked against. For example, combo box styling.
    """

    def __init__(self, *args, available_options: list = None, **kwargs):
        super(ComboDescriptor, self).__init__(*args, **kwargs)
        if available_options is None:
            available_options = []
        self._available_options = available_options

        # We have initialized from the Descriptor class where value has it's own undo/redo decorator
        # This needs to be bypassed to use the Parameter undo/redo stack
        fun = self.__class__.value.fset
        if hasattr(fun, 'func'):
            fun = getattr(fun, 'func')
        self.__previous_set: Callable[[V, Union[numbers.Number, np.ndarray]], Union[numbers.Number, np.ndarray]] = fun

        # Monkey patch the unit and the value to take into account the new max/min situation
        addProp(
            self,
            'value',
            fget=self.__class__.value.fget,
            fset=self.__class__._property_value.fset,
            fdel=self.__class__.value.fdel,
        )

    @property
    def _property_value(self) -> Union[numbers.Number, np.ndarray]:
        return self.value

    @_property_value.setter
    @property_stack_deco
    def _property_value(self, set_value: Union[numbers.Number, np.ndarray, Q_]):
        """
        Verify value against constraints. This hasn't really been implemented as fitting is tricky.

        :param set_value: value to be verified
        :return: new value from constraint
        """
        if isinstance(set_value, Q_):
            set_value = set_value.magnitude
        # Save the old state and create the new state
        old_value = self._value
        state = self._borg.stack.enabled
        if state:
            self._borg.stack.force_state(False)
        try:
            new_value = old_value
            if set_value in self.available_options:
                new_value = set_value
        finally:
            self._borg.stack.force_state(state)

        # Restore to the old state
        self.__previous_set(self, new_value)

    @property
    def available_options(self) -> List[Union[numbers.Number, np.ndarray, Q_]]:
        return self._available_options

    @available_options.setter
    @property_stack_deco
    def available_options(self, available_options: List[Union[numbers.Number, np.ndarray, Q_]]) -> None:
        self._available_options = available_options

    def as_dict(self, **kwargs) -> Dict[str, Any]:
        import json

        d = super().as_dict(**kwargs)
        d['name'] = self.name
        d['available_options'] = json.dumps(self.available_options)
        return d
