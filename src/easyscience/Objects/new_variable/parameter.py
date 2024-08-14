#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

import copy
import numbers
import weakref
from collections import namedtuple
from types import MappingProxyType
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipp as sc
from scipp import UnitError
from scipp import Variable

from easyscience import global_object
from easyscience.fitting.Constraints import ConstraintBase
from easyscience.fitting.Constraints import SelfConstraint
from easyscience.global_object.undo_redo import property_stack_deco
from easyscience.Utils.Exceptions import CoreSetException

from .descriptor_number import DescriptorNumber

Constraints = namedtuple('Constraints', ['user', 'builtin', 'virtual'])


class Parameter(DescriptorNumber):
    """
    A Parameter is a DescriptorNumber which can be used in fitting. It has additional fields to facilitate this.
    """

    # Used by serializer
    _REDIRECT = DescriptorNumber._REDIRECT
    _REDIRECT['callback'] = None

    def __init__(
        self,
        name: str,
        value: numbers.Number,
        unit: Optional[Union[str, sc.Unit]] = '',
        variance: Optional[numbers.Number] = 0.0,
        min: Optional[numbers.Number] = -np.inf,
        max: Optional[numbers.Number] = np.inf,
        fixed: Optional[bool] = False,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        callback: property = property(),
        enabled: Optional[bool] = True,
        parent: Optional[Any] = None,
    ):
        """
        This class is an extension of a `DescriptorNumber`. Where the descriptor was for static
        objects, a `Parameter` is for dynamic objects. A parameter has the ability to be used in fitting and has
        additional fields to facilitate this.

        :param name: Name of this object
        :param value: Value of this object
        :param unit: This object can have a physical unit associated with it
        :param variance: The variance of the value
        :param min: The minimum value for fitting
        :param max: The maximum value for fitting
        :param fixed: Can the parameter vary while fitting?
        :param description: A brief summary of what this object is
        :param url: Lookup url for documentation/information
        :param display_name: The name of the object as it should be displayed
        :param enabled: Can the objects value be set
        :param parent: The object which is the parent to this one

        .. note::
            Undo/Redo functionality is implemented for the attributes `value`, `error`, `min`, `max`, `fixed`
        """
        if not isinstance(min, numbers.Number):
            raise TypeError('`min` must be a number')
        if not isinstance(max, numbers.Number):
            raise TypeError('`max` must be a number')
        if value < min:
            raise ValueError(f'{value=} can not be less than {min=}')
        if value > max:
            raise ValueError(f'{value=} can not be greater than {max=}')
        if min == max:
            raise ValueError('The min and max bounds cannot be identical. Please use fixed=True instead to fix the value.')
        if not isinstance(fixed, bool):
            raise TypeError('`fixed` must be either True or False')

        super().__init__(
            name=name,
            value=value,
            unit=unit,
            variance=variance,
            unique_name=unique_name,
            description=description,
            url=url,
            display_name=display_name,
            parent=parent,
        )

        self._callback = callback  # Callback is used by interface to link to model
        if self._callback.fdel is not None:
            weakref.finalize(self, self._callback.fdel)

        # Create additional fitting elements
        self._min = sc.scalar(float(min), unit=unit)
        self._max = sc.scalar(float(max), unit=unit)
        self._fixed = fixed
        self._enabled = enabled
        self._initial_scalar = copy.deepcopy(self._scalar)
        builtin_constraint = {
            # Last argument in constructor is the name of the property holding the value of the constraint
            'min': SelfConstraint(self, '>=', 'min'),
            'max': SelfConstraint(self, '<=', 'max'),
        }
        self._constraints = Constraints(builtin=builtin_constraint, user={}, virtual={})

    @property
    def value_no_call_back(self) -> numbers.Number:
        """
        Get the currently hold value of self surpassing call back.

        :return: Value of self without unit.
        """
        return self._scalar.value

    @property
    def full_value(self) -> Variable:
        """
        Get the value of self as a scipp scalar. This is should be usable for most cases.
        If a scipp scalar is not acceptable then the raw value can be obtained through `obj.value`.

        :return: Value of self with unit and variance.
        """
        if self._callback.fget is not None:
            scalar = self._callback.fget()
            if scalar != self._scalar:
                self._scalar = scalar
        return self._scalar

    @full_value.setter
    @property_stack_deco
    def full_value(self, scalar: Variable) -> None:
        """
        Set the value of self. This creates a scipp scalar with a unit.

        :param full_value: New value of self
        """
        if not self.enabled:
            if global_object.debug:
                raise CoreSetException(f'{str(self)} is not enabled.')
            return
        if not isinstance(scalar, Variable) and len(scalar.dims) == 0:
            raise TypeError(f'{scalar=} must be a Scipp scalar')
        if not isinstance(scalar.value, numbers.Number) or isinstance(scalar.value, bool):
            raise TypeError('value of Scipp scalar must be a number')
        self._scalar = scalar
        if self._callback.fset is not None:
            self._callback.fset(scalar)

    @property
    def value(self) -> numbers.Number:
        """
        Get the value of self as a Number.

        :return: Value of self without unit.
        """
        if self._callback.fget is not None:
            existing_value = self._callback.fget()
            if existing_value != self._scalar.value:
                self._scalar.value = existing_value
        return self._scalar.value

    @value.setter
    @property_stack_deco
    def value(self, value: numbers.Number) -> None:
        """
        Set the value of self. This only update the value of the scipp scalar.

        :param value: New value of self
        """
        if not self.enabled:
            if global_object.debug:
                raise CoreSetException(f'{str(self)} is not enabled.')
            return

        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise TypeError(f'{value=} must be a number')

        # Need to set the value for constraints to be functional
        self._scalar.value = float(value)
        #        if self._callback.fset is not None:
        #            self._callback.fset(self._scalar.value)

        # Deals with min/max
        value = self._constraint_runner(self.builtin_constraints, self._scalar.value)

        # Deals with user constraints
        # Changes should not be registrered in the undo/redo stack
        stack_state = self._global_object.stack.enabled
        if stack_state:
            self._global_object.stack.force_state(False)
        try:
            value = self._constraint_runner(self.user_constraints, value)
        finally:
            self._global_object.stack.force_state(stack_state)

        value = self._constraint_runner(self._constraints.virtual, value)

        self._scalar.value = value
        if self._callback.fset is not None:
            self._callback.fset(self._scalar.value)

    def convert_unit(self, unit_str: str) -> None:
        """
        Perform unit conversion. The value, max and min can change on unit change.

        :param new_unit: new unit
        :return: None
        """
        super().convert_unit(unit_str)
        new_unit = sc.Unit(unit_str)  # unit_str is tested in super method
        self._min = self._min.to(unit=new_unit)
        self._max = self._max.to(unit=new_unit)

    @property
    def min(self) -> numbers.Number:
        """
        Get the minimum value for fitting.

        :return: minimum value
        """
        return self._min.value

    @min.setter
    @property_stack_deco
    def min(self, min_value: numbers.Number) -> None:
        """
        Set the minimum value for fitting.
        - implements undo/redo functionality.

        :param min_value: new minimum value
        :return: None
        """
        if not isinstance(min_value, numbers.Number):
            raise TypeError('`min` must be a number')
        if min_value == self._max.value:
            raise ValueError('The min and max bounds cannot be identical. Please use fixed=True instead to fix the value.')
        if min_value <= self.value:
            self._min.value = min_value
        else:
            raise ValueError(f'The current value ({self.value}) is smaller than the desired min value ({min_value}).')

    @property
    def max(self) -> numbers.Number:
        """
        Get the maximum value for fitting.

        :return: maximum value
        """
        return self._max.value

    @max.setter
    @property_stack_deco
    def max(self, max_value: numbers.Number) -> None:
        """
        Get the maximum value for fitting.
        - implements undo/redo functionality.

        :param max_value: new maximum value
        :return: None
        """
        if not isinstance(max_value, numbers.Number):
            raise TypeError('`max` must be a number')
        if max_value == self._min.value:
            raise ValueError('The min and max bounds cannot be identical. Please use fixed=True instead to fix the value.')
        if max_value >= self.value:
            self._max.value = max_value
        else:
            raise ValueError(f'The current value ({self.value}) is greater than the desired max value ({max_value}).')

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
            if self._global_object.stack.enabled:
                # Remove the recorded change from the stack
                self._global_object.stack.pop()
            if global_object.debug:
                raise CoreSetException(f'{str(self)} is not enabled.')
            return
        if not isinstance(fixed, bool):
            raise ValueError(f'{fixed=} must be a boolean. Got {type(fixed)}')
        self._fixed = fixed

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

    @property
    def builtin_constraints(self) -> Dict[str, SelfConstraint]:
        """
        Get the built in constrains of the object. Typically these are the min/max

        :return: Dictionary of constraints which are built into the system
        """
        return MappingProxyType(self._constraints.builtin)

    @property
    def user_constraints(self) -> Dict[str, ConstraintBase]:
        """
        Get the user specified constrains of the object.

        :return: Dictionary of constraints which are user supplied
        """
        return self._constraints.user

    @user_constraints.setter
    def user_constraints(self, constraints_dict: Dict[str, ConstraintBase]) -> None:
        self._constraints.user = constraints_dict

    def _constraint_runner(
        self,
        this_constraint_type,
        value: numbers.Number,
    ) -> float:
        for constraint in this_constraint_type.values():
            if constraint.external:
                constraint()
                continue

            constained_value = constraint(no_set=True)
            if constained_value != value:
                if global_object.debug:
                    print(f'Constraint `{constraint}` has been applied')
                self._scalar.value = constained_value
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

    def __copy__(self) -> Parameter:
        new_obj = super().__copy__()
        new_obj._callback = property()
        return new_obj

    def __repr__(self) -> str:
        """
        Return printable representation of a Parameter object.
        """
        super_str = super().__repr__()
        super_str = super_str[:-1]
        s = []
        if self.fixed:
            super_str += ' (fixed)'
        s.append(super_str)
        s.append('bounds=[%s:%s]' % (repr(self.min), repr(self.max)))
        return '%s>' % ', '.join(s)

    # Seems redundant
    # def __float__(self) -> float:
    #     return float(self._scalar.value)
    
    def __add__(self, other: Union[DescriptorNumber, Parameter, numbers.Number]) -> Parameter:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError("Numbers can only be added to dimensionless values")
            new_full_value = self.full_value + other
            min_value = self.min + other
            max_value = self.max + other
            name = f"{self.name} + {other}"
        elif isinstance(other, DescriptorNumber): # Parameter inherits from DescriptorNumber and is also handled here 
            other_unit = other.unit
            try:
                other.convert_unit(self.unit)
            except UnitError:
                raise UnitError(f"Values with units {self.unit} and {other.unit} cannot be added") from None
            new_full_value = self.full_value + other.full_value
            min_value = self.min + other.min if isinstance(other, Parameter) else self.min + other.value
            max_value = self.max + other.max if isinstance(other, Parameter) else self.max + other.value
            name = f"{self.name} + {other.name}"
            other.convert_unit(other_unit)
        else: 
            return NotImplemented
        return Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)

    def __radd__(self, other: Union[DescriptorNumber, numbers.Number]) -> Parameter:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError("Numbers can only be added to dimensionless values")
            new_full_value = self.full_value + other
            min_value = self.min + other
            max_value = self.max + other
            name = f"{other} + {self.name}"
        elif isinstance(other, DescriptorNumber): # Parameter inherits from DescriptorNumber and is also handled here 
            original_unit = self.unit
            try:
                self.convert_unit(other.unit)
            except UnitError:
                raise UnitError(f"Values with units {other.unit} and {self.unit} cannot be added") from None
            new_full_value = self.full_value + other.full_value
            min_value = self.min + other.value
            max_value = self.max + other.value
            name = f"{other.name} + {self.name}"
            self.convert_unit(original_unit)
        else:
            return NotImplemented
        return Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)

    def __sub__(self, other: Union[DescriptorNumber, Parameter, numbers.Number]) -> Parameter:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError("Numbers can only be subtracted from dimensionless values")
            new_full_value = self.full_value - other
            min_value = self.min - other
            max_value = self.max - other
            name = f"{self.name} - {other}"
        elif isinstance(other, DescriptorNumber): # Parameter inherits from DescriptorNumber and is also handled here
            other_unit = other.unit
            try:
                other.convert_unit(self.unit)
            except UnitError:
                raise UnitError(f"Values with units {self.unit} and {other.unit} cannot be subtracted") from None
            new_full_value = self.full_value - other.full_value
            if isinstance(other, Parameter):
                min_value = self.min - other.max if other.max != np.inf else -np.inf
                max_value = self.max - other.min if other.min != -np.inf else np.inf
            else:
                min_value = self.min - other.value
                max_value = self.max - other.value
            name = self.name+" - "+other.name
            other.convert_unit(other_unit)
        else: 
            return NotImplemented
        return Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)
    
    def __rsub__(self, other: Union[DescriptorNumber, numbers.Number]) -> Parameter:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError("Numbers can only be subtracted from dimensionless values")
            new_full_value = other - self.full_value
            min_value = other - self.max
            max_value = other - self.min
            name = f"{other} - {self.name}"
        elif isinstance(other, DescriptorNumber):   # Parameter inherits from DescriptorNumber and is also handled here
            original_unit = self.unit
            try:
                self.convert_unit(other.unit)
            except UnitError:
                raise UnitError(f"Values with units {other.unit} and {self.unit} cannot be subtracted") from None
            new_full_value = other.full_value - self.full_value
            min_value = other.value - self.max 
            max_value = other.value - self.min 
            name = f"{other.name} - {self.name}"
            self.convert_unit(original_unit)
        else:
            return NotImplemented
        return Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)
    
    def __mul__(self, other: Union[DescriptorNumber, Parameter, numbers.Number]) -> Parameter:
        if isinstance(other, numbers.Number):
            new_full_value = self.full_value * other
            name = f"{self.name} * {other}"
            if other == 0:
                return DescriptorNumber.from_scipp(name=name, full_value=new_full_value)
            combinations = [self.min * other, self.max * other]
        elif isinstance(other, DescriptorNumber): # Parameter inherits from DescriptorNumber and is also handled here
            new_full_value = self.full_value * other.full_value
            name = f"{self.name} * {other.name}"
            if other.value == 0 and isinstance(other, DescriptorNumber):
                return DescriptorNumber.from_scipp(name=name, full_value=new_full_value)
            if isinstance(other, Parameter):
                combinations = []
                for first, second in [(self.min, other.min), (self.min, other.max), (self.max, other.min), (self.max, other.max)]:  # noqa: E501
                    if first == 0 and np.isinf(second):
                        combinations.append(0)
                    elif second == 0 and np.isinf(first):
                        combinations.append(0)
                    else:
                        combinations.append(first * second)
            else:
                combinations = [self.min * other.value, self.max * other.value]
        else: 
            return NotImplemented
        min_value = min(combinations)
        max_value = max(combinations)
        parameter = Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)
        parameter.convert_unit(parameter._base_unit())
        return parameter
            
    def __rmul__(self, other: Union[DescriptorNumber, numbers.Number]) -> Parameter:
        if isinstance(other, numbers.Number):
            new_full_value = other * self.full_value
            name = f"{other} * {self.name}"
            if other == 0:
                return DescriptorNumber.from_scipp(name=name, full_value=new_full_value)
            combinations = [other * self.min, other * self.max]
        elif isinstance(other, DescriptorNumber):   # Parameter inherits from DescriptorNumber and is also handled here
            new_full_value = other.full_value * self.full_value
            name = f"{other.name} * {self.name}"
            if other.value == 0:
                return DescriptorNumber.from_scipp(name=name, full_value=new_full_value)
            combinations = [self.min * other.value, self.max * other.value]
        else:
            return NotImplemented
        min_value = min(combinations)
        max_value = max(combinations)
        parameter = Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)
        parameter.convert_unit(parameter._base_unit())
        return parameter
    
    def __truediv__(self, other: Union[DescriptorNumber, Parameter, numbers.Number]) -> Parameter:
        if isinstance(other, numbers.Number):
            original_other = other
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            new_full_value = self.full_value / other
            combinations = [self.min / other, self.max / other]
            name = f"{self.name} / {original_other}"
        elif isinstance(other, DescriptorNumber):  # Parameter inherits from DescriptorNumber and is also handled here
            original_value = other.value
            if original_value == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            new_full_value = self.full_value / other.full_value
            if isinstance(other, Parameter):
                if (other.min < 0 and other.max > 0):
                    combinations = [-np.inf, np.inf]
                elif other.min == 0:
                    if (self.min < 0 and self.max > 0):
                        combinations = [-np.inf, np.inf]
                    elif self.min >= 0:
                        combinations = [self.min/other.max, np.inf]
                    elif self.max <= 0:
                        combinations = [-np.inf, self.max/other.max]
                elif other.max == 0:
                    if (self.min < 0 and self.max > 0):
                        combinations = [-np.inf, np.inf]
                    elif self.min >= 0:
                        combinations = [-np.inf, self.min/other.min]
                    elif self.max <= 0:
                        combinations = [self.max/other.min, np.inf]
                else:
                    combinations = [self.min/other.min, self.max/other.max, self.min/other.max, self.max/other.min]
            else:
                combinations = [self.min / other.value, self.max / other.value]
            name = f"{self.name} / {other.name}"
            other.value = original_value
        else:
            return NotImplemented
        min_value = min(combinations)
        max_value = max(combinations)
        parameter = Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)
        parameter.convert_unit(parameter._base_unit())
        return parameter
    
    def __rtruediv__(self, other: Union[DescriptorNumber, numbers.Number]) -> Parameter:
        original_self = self.value
        if original_self == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        if isinstance(other, numbers.Number):
            new_full_value = other / self.full_value
            other_value = other
            name = f"{other} / {self.name}"
            if other_value == 0:
                return DescriptorNumber.from_scipp(name=name, full_value=new_full_value)
        elif isinstance(other, DescriptorNumber): # Parameter inherits from DescriptorNumber and is also handled here
            new_full_value = other.full_value / self.full_value
            other_value = other.value
            name = other.name+" / "+self.name
            if other_value == 0:
                return DescriptorNumber.from_scipp(name=name, full_value=new_full_value)
        else:
            return NotImplemented
        if (self.min < 0 and self.max > 0):
            combinations = [-np.inf, np.inf]
        elif self.min == 0:
            if other_value > 0:
                combinations = [other_value/self.max, np.inf]
            elif other_value < 0:
                combinations = [-np.inf, other_value/self.max]
        elif self.max == 0:
            if other_value > 0:
                combinations = [-np.inf, other_value/self.min]
            elif other_value < 0:
                combinations = [other_value/self.min, np.inf]
        else:
            combinations = [other_value / self.min, other_value / self.max]
        min_value = min(combinations)
        max_value = max(combinations)
        parameter = Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)
        parameter.convert_unit(parameter._base_unit())
        self.value = original_self
        return parameter
    
    def __pow__(self, other: Union[DescriptorNumber, numbers.Number]) -> Parameter:
        if isinstance(other, numbers.Number):
            exponent = other
            name = f"{self.name} ** {other}"
        elif isinstance(other, DescriptorNumber): # Parameter inherits from DescriptorNumber and is also handled here
            if other.unit != 'dimensionless':
                raise UnitError("Exponents must be dimensionless")
            if other.variance is not None:
                raise ValueError("Exponents must not have variance")
            exponent = other.value
            name = f"{self.name} ** {other.name}"
        else:
            return NotImplemented
        
        try:
            new_full_value = self.full_value ** exponent
        except Exception as message:
            raise message from None
        
        if np.isnan(new_full_value.value):
            raise ValueError("The result of the exponentiation is not a number")
        if exponent == 0:
            return DescriptorNumber.from_scipp(name=name, full_value=new_full_value)
        elif exponent < 0:
            if self.min < 0 and self.max > 0:
                combinations = [-np.inf, np.inf]
            elif self.min == 0:
                combinations = [self.max ** exponent, np.inf]
            elif self.max == 0:
                combinations = [-np.inf, self.min ** exponent]
            else:
                combinations = [self.min ** exponent, self.max ** exponent]
        else:
            combinations = [self.min ** exponent, self.max ** exponent]
        if exponent % 2 == 0:
            if self.min < 0 and self.max > 0:
                combinations.append(0)
            combinations = [abs(combination) for combination in combinations]
        elif exponent % 1 != 0:
            if self.min < 0:
                combinations.append(0)
            combinations = [combination for combination in combinations if combination >= 0]
        min_value = min(combinations)
        max_value = max(combinations)
        return Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)
    
    def __neg__(self) -> Parameter:
        new_full_value = -self.full_value
        name = f"-{self.name}"
        min_value = -self.max
        max_value = -self.min
        return Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)
    
    def __abs__(self) -> Parameter:
        new_full_value = abs(self.full_value)
        name = f"abs({self.name})"
        combinations = [abs(self.min), abs(self.max)]
        if self.min < 0 and self.max > 0:
            combinations.append(0)
        min_value = min(combinations)
        max_value = max(combinations)
        return Parameter.from_scipp(name=name, full_value=new_full_value, min=min_value, max=max_value)