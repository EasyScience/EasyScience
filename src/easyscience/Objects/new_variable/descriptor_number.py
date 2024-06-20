from __future__ import annotations

import numbers

# from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import scipp as sc
from easyscience import borg

# from easyscience import pint
# from easyscience import ureg
from easyscience.Utils.Exceptions import CoreSetException
from easyscience.Utils.UndoRedo import property_stack_deco

from .descriptor_base import DescriptorBase


class DescriptorNumber(DescriptorBase):
    def __init__(
        self,
        name: str,
        value: numbers.Number,
        unit: Optional[Union[str, sc.Unit]] = '',
        variance: Optional[float] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        callback: Optional[property] = None,
        enabled: Optional[bool] = True,
        parent: Optional[Any] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            url=url,
            display_name=display_name,
            callback=callback,
            enabled=enabled,
            parent=parent,
        )
        self._value = sc.scalar(float(value), unit=unit, variance=variance)

    @property
    def value(self) -> sc.scalar:
        """
        Get the value of self as a pint. This is should be usable for most cases. If a pint
        is not acceptable then the raw value can be obtained through `obj.raw_value`.

        :return: Value of self with unit.
        """
        # Cached property? Should reference callback.
        # Also should reference for undo/redo
        if self._callback.fget is not None:
            value = self._callback.fget()
            if value != self._value:
                self._value = value
        return self._value

    @value.setter
    @property_stack_deco
    def value(self, value: sc.scalar) -> None:
        """
        Set the value of self. This creates a pint with a unit.

        :param value: New value of self
        """
        if not self.enabled:
            if borg.debug:
                raise CoreSetException(f'{str(self)} is not enabled.')
            return
        self._value = value
        if self._callback.fset is not None:
            self._callback.fset(value)

    @property
    def unit(self) -> str:
        """
        Get the unit.

        :return: Unit as a string.
        """
        return str(self._value.unit)

    @unit.setter
    @property_stack_deco
    def unit(self, unit_str: str) -> None:
        """
        Set the unit to a new one.  Value remains the same.

        :param unit_str: String representation of the unit required. i.e `m/s`
        """
        self._value.unit = sc.Unit(unit_str)

    def convert_unit(self, unit_str: str):
        """
        Convert the value from one unit system to another. You will should use
        `compatible_units` to see if your new unit is compatible.

        :param unit_str: New unit in string form
        """
        new_unit = sc.Unit(unit_str)
        self._value = self._value.to(unit=new_unit)

    #        self._value.unit = new_unit

    @property
    def variance(self) -> float:
        """
        Get the variance.

        :return: variance.
        """
        return self._value.variance

    @variance.setter
    @property_stack_deco
    def variance(self, variance_float: str) -> None:
        """
        Set the variance.

        :param variance_float: Variance as a float
        """
        self._value.variance = variance_float

    # # @cached_property
    # @property
    # def compatible_units(self) -> List[str]:
    #     """
    #     Returns all possible units for which the current unit can be converted.

    #     :return: Possible conversion units
    #     """
    #     return [str(u) for u in self._scalar.unit.compatible_units()]

    @property
    def raw_value(self) -> numbers.Number:
        """
        Return the raw value of self without a unit.

        :return: The raw value of self
        """
        return self._value.value

    @raw_value.setter
    @property_stack_deco
    def raw_value(self, value: numbers.Number) -> None:
        """
        Set the raw value.

        :param value: value to set
        """
        self._raw_value_property_setter(value)

    # Needed by child classes
    def _raw_value_property_setter(self, value: numbers.Number) -> None:
        self._value.value = value

    def __copy__(self) -> DescriptorNumber:
        return super().__copy__()

    def __repr__(self) -> str:
        """Return printable representation."""
        class_name = self.__class__.__name__
        obj_name = self._name
        obj_value = self._value.value
        obj_unit = self._value.unit
        return f"<{class_name} '{obj_name}': {obj_value:0.04f}{obj_unit}>"

    def as_dict(self) -> Dict[str, Any]:
        raw_dict = super().as_dict()
        raw_dict['value'] = self._value.value
        raw_dict['unit'] = str(self._value.unit)
        raw_dict['variance'] = self._value.variance
        return raw_dict
