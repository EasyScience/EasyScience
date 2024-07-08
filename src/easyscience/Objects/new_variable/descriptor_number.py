from __future__ import annotations

import numbers
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import scipp as sc
from scipp import Variable

from easyscience.global_object.undo_redo import property_stack_deco

from .descriptor_base import DescriptorBase


class DescriptorNumber(DescriptorBase):
    """
    A `Descriptor` for Number values with units.  The internal representation is a scipp scalar.
    """

    def __init__(
        self,
        name: str,
        value: numbers.Number,
        unit: Optional[Union[str, sc.Unit]] = '',
        variance: Optional[numbers.Number] = None,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        parent: Optional[Any] = None,
    ):
        """Constructor for the DescriptorNumber class

        param name: Name of the descriptor
        param value: Value of the descriptor
        param unit: Unit of the descriptor
        param variance: Variance of the descriptor
        param description: Description of the descriptor
        param url: URL of the descriptor
        param display_name: Display name of the descriptor
        param parent: Parent of the descriptor

        .. note:: Undo/Redo functionality is implemented for the attributes `variance` and `value`.
        """
        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise TypeError(f'{value=} must be a number')
        if variance is not None:
            if not isinstance(variance, numbers.Number) or isinstance(variance, bool):
                raise TypeError(f'{variance=} must be a number or None')
            if variance < 0:
                raise ValueError(f'{variance=} must be positive')
            variance = float(variance)
        if not isinstance(unit, sc.Unit) and not isinstance(unit, str):
            raise TypeError(f'{unit=} must be a scipp unit or a string representing a valid scipp unit')
        try:
            self._scalar = sc.scalar(float(value), unit=unit, variance=variance)
        except Exception as message:
            raise ValueError(message)
        super().__init__(
            name=name,
            unique_name=unique_name,
            description=description,
            url=url,
            display_name=display_name,
            parent=parent,
        )

    @classmethod
    def from_scipp(cls, name: str, full_value: Variable, **kwargs) -> DescriptorNumber:
        """
        Create a DescriptorNumber from a scipp constant.

        :param name: Name of the descriptor
        :param value: Value of the descriptor as a scipp scalar
        :param kwargs: Additional parameters for the descriptor
        :return: DescriptorNumber
        """
        if not isinstance(full_value, Variable):
            raise TypeError(f'{full_value=} must be a scipp scalar')
        if len(full_value.dims) != 0:
            raise TypeError(f'{full_value=} must be a scipp scalar')
        return cls(name=name, value=full_value.value, unit=full_value.unit, variance=full_value.variance, **kwargs)


    @property
    def full_value(self) -> Variable:
        """
        Get the value of self as a scipp scalar. This is should be usable for most cases.

        :return: Value of self with unit.
        """
        return self._scalar

    @full_value.setter
    def full_value(self, full_value: Variable) -> None:
        raise AttributeError(f'Full_value is read-only. Change the value and variance seperately. or create a new {self.__class__.__name__}.')  # noqa: E501
    
    @property
    def value(self) -> numbers.Number:
        """
        Get the value. This should be usable for most cases. The full value can be obtained from `obj.full_value`.

        :return: Value of self with unit.
        """
        return self._scalar.value

    @value.setter
    @property_stack_deco
    def value(self, value: numbers.Number) -> None:
        """
        Set the value of self. This should be usable for most cases. The full value can be obtained from `obj.full_value`.

        :param value: New value of self
        """
        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise TypeError(f'{value=} must be a number')
        self._scalar.value = value

    @property
    def unit(self) -> str:
        """
        Get the unit.

        :return: Unit as a string.
        """
        return str(self._scalar.unit)

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(f'Unit is read-only. Use convert_unit to change the unit between allowed types or create a new {self.__class__.__name__} with the desired unit.')  # noqa: E501

    @property
    def variance(self) -> float:
        """
        Get the variance.

        :return: variance.
        """
        return self._scalar.variance

    @variance.setter
    @property_stack_deco
    def variance(self, variance_float: float) -> None:
        """
        Set the variance.

        :param variance_float: Variance as a float
        """
        if variance_float is not None:
            if not isinstance(variance_float, numbers.Number):
                raise TypeError(f'{variance_float=} must be a number or None')
            if variance_float < 0:
                raise ValueError(f'{variance_float=} must be positive')
            variance_float = float(variance_float)
        self._scalar.variance = variance_float


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
        if value is not None:
            if not isinstance(value, numbers.Number):
                raise TypeError(f'{value=} must be a number or None')
            if value < 0:
                raise ValueError(f'{value=} must be positive')
            value = float(value)
        self._scalar.variance = value**2

    def convert_unit(self, unit_str: str):
        """
        Convert the value from one unit system to another.

        :param unit_str: New unit in string form
        """
        if not isinstance(unit_str, str):
            raise TypeError(f'{unit_str=} must be a string representing a valid scipp unit')
        try:
            new_unit = sc.Unit(unit_str)
        except Exception as message:
            raise ValueError(message)
        self._scalar = self._scalar.to(unit=new_unit)

    # Just to get return type right
    def __copy__(self) -> DescriptorNumber:
        return super().__copy__()
   
    def __repr__(self) -> str:
        """Return printable representation."""
        string='<'
        string+= self.__class__.__name__+' '
        string+=f"'{self._name}': "
        string+= f'{self._scalar.value:.4f}'
        if self.variance:
            string += f' \u00B1 {self.error:.4f}'
        obj_unit = self._scalar.unit
        if obj_unit == 'dimensionless':
            obj_unit = ''
        else:
            obj_unit = f' {obj_unit}'
        string+= obj_unit
        string+='>'
        return string
        # return f"<{class_name} '{obj_name}': {obj_value:0.04f}{obj_unit}>"

    def as_dict(self) -> Dict[str, Any]:
        raw_dict = super().as_dict()
        raw_dict['value'] = self._scalar.value
        raw_dict['unit'] = str(self._scalar.unit)
        raw_dict['variance'] = self._scalar.variance
        return raw_dict
