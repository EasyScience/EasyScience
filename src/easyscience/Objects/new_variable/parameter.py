#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import copy
import numbers

# import warnings
import weakref

# from copy import deepcopy
# from inspect import getfullargspec
from types import MappingProxyType
from typing import Any

# from typing import TYPE_CHECKING
# from typing import Any
# from typing import Callable
# from typing import Dict
# from typing import List
from typing import Optional

# from typing import Set
from typing import Tuple

# from typing import Type
# from typing import TypeVar
from typing import Union

import numpy as np
import scipp as sc

from easyscience import borg

# from easyscience import pint
# from easyscience import ureg
from easyscience.fitting.Constraints import SelfConstraint

# from easyscience.Objects.core import ComponentSerializer
# from easyscience.Utils.classTools import addProp
from easyscience.Utils.Exceptions import CoreSetException
from easyscience.Utils.UndoRedo import property_stack_deco

from .descriptor_number import DescriptorNumber


class Parameter(DescriptorNumber):
    """
    This class is an extension of a ``EasyScience.Object.Base.Descriptor``. Where the descriptor was for static objects,
    a `Parameter` is for dynamic objects. A parameter has the ability to be used in fitting and
    has additional fields to facilitate this.
    """

    #    _constructor = M_

    def __init__(
        self,
        name: str,
        value: numbers.Number,
        unit: Optional[Union[str, sc.Unit]] = '',
        variance: Optional[numbers.Number] = 0.0,
        min: Optional[numbers.Number] = -np.Inf,
        max: Optional[numbers.Number] = np.Inf,
        fixed: Optional[bool] = False,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        callback: Optional[property] = None,
        enabled: Optional[bool] = True,
        parent: Optional[Any] = None,
    ):
        """
        This class is an extension of a ``EasyScience.Object.Base.Descriptor``. Where the descriptor was for static
        objects, a `Parameter` is for dynamic objects. A parameter has the ability to be used in fitting and has
        additional fields to facilitate this.

        :param name: Name of this obj
        :param value: Value of this object
        :param error: Error associated as sigma for this parameter
        :param min: Minimum value for fitting
        :param max: Maximum value for fitting
        :param fixed: Should this parameter vary when fitting?
        :param kwargs: Key word arguments for the `Descriptor` class.

        .. code-block:: python

             from easyscience.Objects.Base import Parameter
             # Describe a phase
             phase_basic = Parameter('phase', 3)
             # Describe a phase with a unit
             phase_unit = Parameter('phase', 3, units,='rad/s')

        .. note::
            Undo/Redo functionality is implemented for the attributes `value`, `error`, `min`, `max`, `fixed`
        """
        # Set the error
        #        self._args = {'value': value, 'units': '', 'error': error}

        if value < min:
            raise ValueError(f'{value=} can not be less than {min=}')
        if value > max:
            raise ValueError(f'{value=} can not be greater than {max=}')
        #        if variance < 0:
        #            raise ValueError('`variance` must be positive')

        #        self._value: sc.scalar  # set in super().__init__

        super().__init__(
            name=name,
            value=value,
            unit=unit,
            variance=variance,
            description=description,
            url=url,
            display_name=display_name,
            #            callback=callback,
            #            enabled=enabled,
            parent=parent,
        )

        if callback is None:
            callback = property()
        self._callback = callback
        if self._callback.fdel is not None:
            weakref.finalize(self, self._callback.fdel)

        #        self._args['units'] = str(self.unit)

        # # Warnings if we are given a boolean
        # if isinstance(value, bool):
        #     warnings.warn(
        #         'Boolean values are not officially supported in Parameter. Use a Descriptor instead',
        #         UserWarning,
        #     )

        # Create additional fitting elements
        self._min = sc.scalar(float(min), unit=unit)
        self._max = sc.scalar(float(max), unit=unit)
        self._fixed = fixed
        self._enabled = enabled
        self._initial_scalar = copy.deepcopy(self._scalar)
        self._constraints = {
            'user': {},
            'builtin': {
                'min': SelfConstraint(self, '>=', '_min'),
                'max': SelfConstraint(self, '<=', '_max'),
            },
            'virtual': {},
        }
        # # This is for the serialization. Otherwise we wouldn't catch the values given to `super()`
        # self._kwargs = kwargs

        # # We have initialized from the Descriptor class where value has it's own undo/redo decorator
        # # This needs to be bypassed to use the Parameter undo/redo stack
        # fun = self.__class__.value.fset
        # if hasattr(fun, 'func'):
        #     fun = getattr(fun, 'func')
        # self.__previous_set: Callable[
        #     [V, Union[numbers.Number, np.ndarray]],
        #     Union[numbers.Number, np.ndarray],
        # ] = fun

        # # Monkey patch the unit and the value to take into account the new max/min situation
        # addProp(
        #     self,
        #     'value',
        #     fget=self.__class__.value.fget,
        #     fset=self.__class__._property_value.fset,
        #     fdel=self.__class__.value.fdel,
        # )

    # # Property from DescriptorNumber
    # def _raw_value_property_setter(self, value: numbers.Number) -> None:
    #     old_value = self._value
    #     # self._value = self.__class__._constructor(value=set_value, units=self._args['units'], error=self._args['error'])

    #     # First run the built in constraints. i.e. min/max
    #     constraint_type = self.builtin_constraints
    #     #        constraint_type: MappingProxyType[str, C] = self.builtin_constraints
    #     #        new_value = self.__constraint_runner(constraint_type, set_value)
    #     # Then run any user constraints.
    #     constraint_type: dict = self.user_constraints
    #     state = self._borg.stack.enabled
    #     if state:
    #         self._borg.stack.force_state(False)
    #     # try:
    #     #     new_value = self.__constraint_runner(constraint_type, new_value)
    #     # finally:
    #     #     self._borg.stack.force_state(state)

    #     # # And finally update any virtual constraints
    #     # constraint_type: dict = self._constraints['virtual']
    #     # _ = self.__constraint_runner(constraint_type, new_value)

    #     # # Restore to the old state
    #     # # self._value = old_value
    #     # self.__previous_set(self, new_value)

    # @property
    # #    def _property_value(self) -> Union[numbers.Number, np.ndarray, M_]:
    # def _property_value(self) -> Union[numbers.Number, np.ndarray]:
    #     return self.value

    # @_property_value.setter
    # @property_stack_deco
    # def _property_value(self, set_value: Union[numbers.Number, np.ndarray]) -> None:
    #     #    def _property_value(self, set_value: Union[numbers.Number, np.ndarray, M_]) -> None:
    #     """
    #     Verify value against constraints. This hasn't really been implemented as fitting is tricky.

    #     :param set_value: value to be verified
    #     :return: new value from constraint
    #     """
    #     #        if isinstance(set_value, M_):
    #     #            set_value = set_value.magnitude.nominal_value
    #     # Save the old state and create the new state
    #     old_value = self._value
    #     # self._value = self.__class__._constructor(value=set_value, units=self._args['units'], error=self._args['error'])

    #     # First run the built in constraints. i.e. min/max
    #     constraint_type = self.builtin_constraints
    #     #        constraint_type: MappingProxyType[str, C] = self.builtin_constraints
    #     new_value = self.__constraint_runner(constraint_type, set_value)
    #     # Then run any user constraints.
    #     constraint_type: dict = self.user_constraints
    #     state = self._borg.stack.enabled
    #     if state:
    #         self._borg.stack.force_state(False)
    #     try:
    #         new_value = self.__constraint_runner(constraint_type, new_value)
    #     finally:
    #         self._borg.stack.force_state(state)

    #     # And finally update any virtual constraints
    #     constraint_type: dict = self._constraints['virtual']
    #     _ = self.__constraint_runner(constraint_type, new_value)

    #     # Restore to the old state
    #     # self._value = old_value
    #     self.__previous_set(self, new_value)

    @property
    def full_value(self) -> sc.scalar:
        """
        Get the value of self as a scipp scalar. This is should be usable for most cases.
        If a scipp scalar is not acceptable then the raw value can be obtained through `obj.value`.

        :return: Value of self with unit and variance.
        """
        # Also should reference for undo/redo
        if self._callback.fget is not None:
            scalar = self._callback.fget()
            if scalar != self._scalar:
                self._scalar = scalar
        return self._scalar

    @full_value.setter
    @property_stack_deco
    def full_value(self, scalar: sc.scalar) -> None:
        """
        Set the value of self. This creates a scipp scalar with a unit.

        :param full_value: New value of self
        """
        if not self.enabled:
            if borg.debug:
                raise CoreSetException(f'{str(self)} is not enabled.')
            return
        self._scalar = scalar
        if self._callback.fset is not None:
            self._callback.fset(scalar)

    @property
    def value(self) -> numbers.Number:
        """
        Get the value of self as a Number.

        :return: Value of self without unit.
        """
        # Also should reference for undo/redo
        if self._callback.fget is not None:
            scalar = self._callback.fget()
            if scalar.value != self._scalar.value:
                self._scalar.value = scalar.value
        return self._scalar.value

    @value.setter
    @property_stack_deco
    def value(self, value: numbers.Number) -> None:
        """
        Set the value of self. This only update the value of the scipp scalar.

        :param value: New value of self
        """
        if not self.enabled:
            if borg.debug:
                raise CoreSetException(f'{str(self)} is not enabled.')
            return

        self._scalar.value = value
        if 1:
            # First run the built in constraints. i.e. min/max
            #        constraint_type = self.builtin_constraints
            #        constraint_type: MappingProxyType[str, C] = self.builtin_constraints
            value = self._constraint_runner(self.builtin_constraints, value)
            # Then run any user constraints.
            #        constraint_type: dict = self.user_constraints

            stack_state = self._borg.stack.enabled
            if stack_state:
                self._borg.stack.force_state(False)
            try:
                value = self._constraint_runner(self.user_constraints, value)
            finally:
                self._borg.stack.force_state(stack_state)

            # And finally update any virtual constraints
            #        constraint_type: dict = self._constraints['virtual']
            value = self._constraint_runner(self._constraints['virtual'], value)

        self._scalar.value = value
        if self._callback.fset is not None:
            self._callback.fset(self._scalar)

    def convert_unit(self, unit_str: str) -> None:
        """
        Perform unit conversion. The value, max and min can change on unit change.

        :param new_unit: new unit
        :return: None
        """
        #        old_unit = self._value.unit  # str(self._args['units'])
        super().convert_unit(unit_str)
        new_unit = sc.Unit(unit_str)
        self._min = self._min.to(unit=new_unit)
        self._max = self._max.to(unit=new_unit)
        # Deal with min/max.
        #        if not self.value.unitless and old_unit != 'dimensionless':

    #        self._min = sc.to_unit(self._min * old_unit, unit_str).value
    #        self._max = sc.to_unit(self._max * old_unit, unit_str).value

    #        # Log the new converted error
    #        self._args['error'] = self.value.error.magnitude

    @property
    def min(self) -> numbers.Number:
        """
        Get the minimum value for fitting.

        :return: minimum value
        """
        return self._min.value

    @min.setter
    @property_stack_deco
    def min(self, value: numbers.Number) -> None:
        """
        Set the minimum value for fitting.
        - implements undo/redo functionality.

        :param value: new minimum value
        :return: None
        """
        if value <= self.value:
            self._min.value = value
        else:
            raise ValueError(f'The current value ({self.value}) is less than the desired min value ({value}).')

    @property
    def max(self) -> numbers.Number:
        """
        Get the maximum value for fitting.

        :return: maximum value
        """
        return self._max.value

    @max.setter
    @property_stack_deco
    def max(self, value: numbers.Number) -> None:
        """
        Get the maximum value for fitting.
        - implements undo/redo functionality.

        :param value: new maximum value
        :return: None
        """
        if value >= self.value:
            self._max.value = value
        else:
            raise ValueError(f'The current value ({self.value}) is greater than the desired max value ({value}).')

    @property
    def fixed(self) -> bool:
        """
        Can the parameter vary while fitting?

        :return: True = fixed, False = can vary
        """
        return self._fixed

    @fixed.setter
    @property_stack_deco
    def fixed(self, fixed: bool) -> None:
        """
        Change the parameter vary while fitting state.
        - implements undo/redo functionality.

        :param fixed: True = fixed, False = can vary
        """
        if not self.enabled:
            if self._borg.stack.enabled:
                # Remove the recorded change from the stack
                self._borg.stack.pop()
            if borg.debug:
                raise CoreSetException(f'{str(self)} is not enabled.')
            return
        if not isinstance(fixed, bool):
            raise ValueError(f'{fixed=} must be a boolean. Got {type(fixed)}')
        self._fixed = fixed

    @property
    def error(self) -> float:
        """
        The standard deviation for the parameter.

        :return: Error associated with parameter
        """
        return float(np.sqrt(self._scalar.variance))

    @error.setter
    @property_stack_deco
    def error(self, value: float) -> None:
        """
        Set the standard deviation for the parameter.

        :param value: New error value
        """
        if value < 0:
            raise ValueError(f'{value} must be positive')
        self._scalar.variance = value**2

    @property
    def bounds(self) -> Tuple[numbers.Number, numbers.Number]:
        """
        Get the bounds of the parameter.

        :return: Tuple of the parameters minimum and maximum values
        """
        return self.min, self.max

    @bounds.setter
    def bounds(self, new_bound: Tuple[numbers.Number, numbers.Number]) -> None:
        """
        Set the bounds of the parameter. *This will also enable the parameter*.

        :param new_bound: New bounds. This should be a tuple of (min, max).
        """
        # # Macro checking and opening for undo/redo
        # close_macro = False
        # if self._borg.stack.enabled:
        #     self._borg.stack.beginMacro('Setting bounds')
        #     close_macro = True
        # # Have we only been given a single number (MIN)?
        # if isinstance(new_bound, numbers.Number):
        #     self.min = new_bound
        # # Have we been given a tuple?
        # if isinstance(new_bound, tuple):
        old_min = self.min
        old_max = self.max
        new_min, new_max = new_bound

        try:
            self.min = new_min
            self.max = new_max
        except ValueError:
            self.min = old_min
            self.max = old_max
            raise ValueError(f'Current paramter value: {self._scalar.value} must be within {new_bound=}')

        # Enable the parameter if needed
        if not self.enabled:
            self.enabled = True
        # Free parameter if needed
        if self.fixed:
            self.fixed = False

        # # Close the macro if we opened it
        # if close_macro:
        #     self._borg.stack.endMacro()

    @property
    def builtin_constraints(self):
        #    def builtin_constraints(self) -> MappingProxyType[str, C]:
        """
        Get the built in constrains of the object. Typically these are the min/max

        :return: Dictionary of constraints which are built into the system
        """
        return MappingProxyType(self._constraints['builtin'])

    @property
    def user_constraints(self):
        #    def user_constraints(self) -> Dict[str, C]:
        """
        Get the user specified constrains of the object.

        :return: Dictionary of constraints which are user supplied
        """
        return self._constraints['user']

    @user_constraints.setter
    def user_constraints(self, constraints_dict) -> None:
        self._constraints['user'] = constraints_dict

    # def user_constraints(self, constraints_dict: Dict[str, C]) -> None:
    #     self._constraints['user'] = constraints_dict

    def _quick_set(
        self,
        set_value: float,
        run_builtin_constraints: bool = False,
        run_user_constraints: bool = False,
        run_virtual_constraints: bool = False,
    ) -> None:
        """
        This is a quick setter for the parameter. It bypasses all the checks and constraints,
        just setting the value and issuing the interface callbacks.

        WARNING: This is a dangerous function and should only be used when you know what you are doing.
        """
        # First run the built-in constraints. i.e. min/max
        if run_builtin_constraints:
            constraint_type: MappingProxyType = self.builtin_constraints
            set_value = self._constraint_runner(constraint_type, set_value)
        # Then run any user constraints.
        if run_user_constraints:
            constraint_type: dict = self.user_constraints
            state = self._borg.stack.enabled
            if state:
                self._borg.stack.force_state(False)
            try:
                set_value = self._constraint_runner(constraint_type, set_value)
            finally:
                self._borg.stack.force_state(state)
        if run_virtual_constraints:
            # And finally update any virtual constraints
            constraint_type: dict = self._constraints['virtual']
            _ = self._constraint_runner(constraint_type, set_value)

        # Finally set the value
        self._property_value._magnitude._nominal_value = set_value
        self._args['value'] = set_value
        if self._callback.fset is not None:
            self._callback.fset(set_value)

    def _constraint_runner(
        self,
        this_constraint_type,
        #        this_constraint_type: Union[dict, MappingProxyType[str, C]],
        value: numbers.Number,
    ) -> float:
        for constraint in this_constraint_type.values():
            if constraint.external:
                constraint()
                continue

            constained_value = constraint(no_set=True)
            if constained_value != value:
                if borg.debug:
                    print(f'Constraint `{constraint}` has been applied')
                self._scalar.value = constained_value
                # self._value = self.__class__._constructor(
                #     value=this_new_value,
                #     units=self._args['units'],
                #     error=self._args['error'],
                # )
            value = constained_value
        return value

    @property
    def enabled(self) -> bool:
        """
        Logical property to see if the objects value can be directly set.

        :return: Can the objects value be set
        """
        return self._enabled

    @enabled.setter
    @property_stack_deco
    def enabled(self, value: bool) -> None:
        """
        Enable and disable the direct setting of an objects value field.

        :param value: True - objects value can be set, False - the opposite
        """
        self._enabled = value

    # Just to get return type right
    def __copy__(self) -> Parameter:
        new_obj = super().__copy__()
        new_obj._callback = self._callback
        return new_obj

    def __repr__(self) -> str:
        """
        Return printable representation of a Parameter object.
        """
        super_str = super().__repr__()
        super_str = super_str[:-1]
        s = []
        if self.variance:
            super_str += f' +/-{self.variance}'
        if self.fixed:
            super_str += ' (fixed)'
        s.append(super_str)
        s.append('bounds=[%s:%s]' % (repr(self.min), repr(self.max)))
        return '%s>' % ', '.join(s)

    def __float__(self) -> float:
        return float(self._scalar.value)
