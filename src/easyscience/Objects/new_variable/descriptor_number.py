from __future__ import annotations

import numbers
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import scipp as sc
from scipp import Variable

from easyscience.Utils.UndoRedo import property_stack_deco

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
        variance: Optional[float] = None,
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

        .. note:: Undo/Redo functionality is implemented for the attributes `full_value`, `unit`, `variance` and `value`.
        """
        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise ValueError(f'{value=} must be type numeric')
        if variance is not None and variance < 0:
            raise ValueError(f'{variance=} must be positive')
        super().__init__(
            name=name,
            description=description,
            url=url,
            display_name=display_name,
            parent=parent,
        )
        self._scalar = sc.scalar(float(value), unit=unit, variance=variance)

    @property
    def full_value(self) -> Variable:
        """
        Get the value of self as a scipp scalar. This is should be usable for most cases.

        :return: Value of self with unit.
        """
        return self._scalar

    @full_value.setter
    @property_stack_deco
    def full_value(self, full_value: Variable) -> None:
        """
        Set the full value of self. This creates a scipp scalar with a unit.

        :param value: New value of self
        """
        self._scalar = full_value

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
            raise ValueError(f'{value=} must be type numbers.Number')
        self._scalar.value = value

    @property
    def unit(self) -> str:
        """
        Get the unit.

        :return: Unit as a string.
        """
        return str(self._scalar.unit)

    @unit.setter
    @property_stack_deco
    def unit(self, unit_str: str) -> None:
        """
        Set the unit to a new one.  Value remains unchanged.

        :param unit_str: String representation of the unit required. i.e `m/s`
        """
        self._scalar.unit = sc.Unit(unit_str)

    @property
    def variance(self) -> float:
        """
        Get the variance.

        :return: variance.
        """
        return self._scalar.variance

    @variance.setter
    @property_stack_deco
    def variance(self, variance_float: str) -> None:
        """
        Set the variance.

        :param variance_float: Variance as a float
        """
        self._scalar.variance = variance_float

    def convert_unit(self, unit_str: str):
        """
        Convert the value from one unit system to another.

        :param unit_str: New unit in string form
        """
        new_unit = sc.Unit(unit_str)
        self._scalar = self._scalar.to(unit=new_unit)

    # Just to get return type right
    def __copy__(self) -> DescriptorNumber:
        return super().__copy__()

    def __repr__(self) -> str:
        """Return printable representation."""
        class_name = self.__class__.__name__
        obj_name = self._name
        obj_value = self._scalar.value
        obj_unit = self._scalar.unit
        if obj_unit == 'dimensionless':
            obj_unit = ''
        else:
            obj_unit = f' {obj_unit}'
        return f"<{class_name} '{obj_name}': {obj_value:0.04f}{obj_unit}>"

    def as_dict(self) -> Dict[str, Any]:
        raw_dict = super().as_dict()
        raw_dict['value'] = self._scalar.value
        raw_dict['unit'] = str(self._scalar.unit)
        raw_dict['variance'] = self._scalar.variance
        return raw_dict
