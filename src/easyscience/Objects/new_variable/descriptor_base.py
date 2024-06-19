#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

# import numbers
# import warnings
import abc
import weakref

# from copy import deepcopy
# from inspect import getfullargspec
# from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any

# from typing import Callable
# from typing import Dict
# from typing import List
from typing import Optional

# from typing import Set
# from typing import Tuple
# from typing import Type
# from typing import TypeVar
# from typing import Union
# import numpy as np
from easyscience import borg

# from easyscience import pint
from easyscience import ureg

# from easyscience.Fitting.Constraints import SelfConstraint
from easyscience.Objects.core import ComponentSerializer

# from easyscience.Utils.classTools import addProp
# from easyscience.Utils.Exceptions import CoreSetException
from easyscience.Utils.UndoRedo import property_stack_deco

if TYPE_CHECKING:
    from easyscience.Utils.typing import C

Q_ = ureg.Quantity
M_ = ureg.Measurement


class DescriptorBase(ComponentSerializer, metaclass=abc.ABCMeta):
    """
    This is the base of all variable descriptions for models. It contains all information to describe a single
    unique property of an object. This description includes a name and value as well as optionally a unit, description
    and url (for reference material). Also implemented is a callback so that the value can be read/set from a linked
    library object.

    A `Descriptor` is typically something which describes part of a model and is non-fittable and generally changes the
    state of an object.
    """

    #    _constructor = Q_
    _borg = borg
    # _REDIRECT = {
    #     'value': lambda obj: obj.raw_value,
    #     'units': lambda obj: obj._args['units'],
    #     'parent': None,
    #     'callback': None,
    #     '_finalizer': None,
    # }

    def __init__(
        self,
        name: str,
        # value: Any,
        # units: Optional[Union[str, ureg.Unit]] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        callback: Optional[property] = None,
        enabled: Optional[bool] = True,
        parent: Optional[Any] = None,
    ):  # noqa: S107
        """
        This is the base of all variable descriptions for models. It contains all information to describe a single
        unique property of an object. This description includes a name and value as well as optionally a unit,
        description and url (for reference material). Also implemented is a callback so that the value can be read/set
        from a linked library object.

        A `Descriptor` is typically something which describes part of a model and is non-fittable and generally changes
        the state of an object.

        Units are provided by pint: https://github.com/hgrecco/pint

        :param name: Name of this object
        :param value: Value of this object
        :param units: This object can have a physical unit associated with it
        :param description: A brief summary of what this object is
        :param url: Lookup url for documentation/information
        :param callback: The property which says how the object is linked to another one
        :param parent: The object which is the parent to this one

        .. code-block:: python

             from easyscience.Objects.Base import Descriptor
             # Describe a color by text
             color_text = Descriptor('fav_colour', 'red')
             # Describe a color by RGB
             color_num = Descriptor('fav_colour', [1, 0, 0])

        .. note:: Undo/Redo functionality is implemented for the attributes `value`, `unit` and `display name`.
        """
        # if not hasattr(self, '_args'):
        #     self._args = {'value': None, 'units': ''}

        # Let the collective know we've been assimilated
        self._parent = parent
        self._borg.map.add_vertex(self, obj_type='created')
        # Make the connection between self and parent
        if parent is not None:
            self._borg.map.add_edge(parent, self)

        self._name: str = name

        # # Attach units if necessary
        # if isinstance(units, ureg.Unit):
        #     self._units = ureg.Quantity(1, units=deepcopy(units))
        # elif isinstance(units, (str, type(None))):
        #     self._units = ureg.parse_expression(units)
        # else:
        #     raise AttributeError
        # # Clunky method of keeping self.value up to date
        # self._type = type(value)
        # self.__isBooleanValue = isinstance(value, bool)
        # if self.__isBooleanValue:
        #     value = int(value)
        # self._args['value'] = value
        # self._args['units'] = str(self.unit)
        # self._value = self.__class__._constructor(**self._args)

        self._enabled = enabled

        if description is None:
            description = ''
        self._description: str = description

        self._display_name: str = display_name

        if url is None:
            url = ''
        self._url: str = url

        if callback is None:
            callback = property()
        self._callback: property = callback
        #        self.user_data: dict = {}

        finalizer = None
        if self._callback.fdel is not None:
            weakref.finalize(self, self._callback.fdel)
        self._finalizer = finalizer

    # @property
    # def _arg_spec(self) -> Set[str]:
    #     base_cls = getattr(self, '__old_class__', self.__class__)
    #     mro = base_cls.__mro__
    #     idx = mro.index(ComponentSerializer)
    #     names = set()
    #     for i in range(idx):
    #         cls = mro[i]
    #         if hasattr(cls, '_CORE'):
    #             spec = getfullargspec(cls.__init__)
    #             names = names.union(set(spec.args[1:]))
    #     return names

    # def __reduce__(self):
    #     """
    #     Make the class picklable. Due to the nature of the dynamic class definitions special measures need to be taken.

    #     :return: Tuple consisting of how to make the object
    #     :rtype: tuple
    #     """
    #     state = self.encode()
    #     cls = self.__class__
    #     if hasattr(self, '__old_class__'):
    #         cls = self.__old_class__
    #     return cls.from_dict, (state,)

    @property
    def display_name(self) -> str:
        """
        Get a pretty display name.

        :return: The pretty display name.
        """
        display_name = self._display_name
        if display_name is None:
            display_name = self._name
        return display_name

    @display_name.setter
    @property_stack_deco
    def display_name(self, name_str: str) -> None:
        """
        Set the pretty display name.

        :param name_str: Pretty display name of the object.
        :return: None
        """
        self._display_name = name_str

    # @property
    # def unit(self) -> pint.UnitRegistry:
    #     """
    #     Get the unit associated with the object.

    #     :return: Unit associated with self in `pint` form.
    #     """
    #     return self._units.units

    # @unit.setter
    # @property_stack_deco
    # def unit(self, unit_str: str):
    #     """
    #     Set the unit to a new one.

    #     :param unit_str: String representation of the unit required. i.e `m/s`
    #     :return: None
    #     """
    #     if not isinstance(unit_str, str):
    #         unit_str = str(unit_str)
    #     new_unit = ureg.parse_expression(unit_str)
    #     self._units = new_unit
    #     self._args['units'] = str(new_unit)
    #     self._value = self.__class__._constructor(**self._args)

    # @property
    # def value(self) -> Any:
    #     """
    #     Get the value of self as a pint. This is should be usable for most cases. If a pint
    #     is not acceptable then the raw value can be obtained through `obj.raw_value`.

    #     :return: Value of self with unit.
    #     """
    #     # Cached property? Should reference callback.
    #     # Also should reference for undo/redo
    #     if self._callback.fget is not None:
    #         try:
    #             value = self._callback.fget()
    #             if hasattr(self._value, 'magnitude'):
    #                 if value != self._value.magnitude:
    #                     self.__deepValueSetter(value)
    #             elif value != self._value:
    #                 self.__deepValueSetter(value)

    #         except Exception as e:
    #             raise ValueError(f'Unable to return value:\n{e}')
    #     r_value = self._value
    #     if self.__isBooleanValue:
    #         r_value = bool(r_value)
    #     return r_value

    # def __deepValueSetter(self, value: Any):
    #     """
    #     Set the value of self. This creates a pint with a unit.

    #     :param value: New value of self
    #     :return: None
    #     """
    #     # TODO there should be a callback to the collective, logging this as a return(if from a non `EasyScience` class)
    #     if hasattr(value, 'magnitude'):
    #         value = value.magnitude
    #         if hasattr(value, 'nominal_value'):
    #             value = value.nominal_value
    #     self._type = type(value)
    #     self.__isBooleanValue = isinstance(value, bool)
    #     if self.__isBooleanValue:
    #         value = int(value)
    #     self._args['value'] = value
    #     self._value = self.__class__._constructor(**self._args)

    # @value.setter
    # @property_stack_deco
    # def value(self, value: Any):
    #     """
    #     Set the value of self. This creates a pint with a unit.

    #     :param value: New value of self
    #     :return: None
    #     """
    #     if not self.enabled:
    #         if borg.debug:
    #             raise CoreSetException(f'{str(self)} is not enabled.')
    #         return
    #     self.__deepValueSetter(value)
    #     if self._callback.fset is not None:
    #         try:
    #             self._callback.fset(value)
    #         except Exception as e:
    #             raise CoreSetException(e)

    # @property
    # def raw_value(self) -> Any:
    #     """
    #     Return the raw value of self without a unit.

    #     :return: The raw value of self
    #     """
    #     value = self._value
    #     if hasattr(value, 'magnitude'):
    #         value = value.magnitude
    #         if hasattr(value, 'nominal_value'):
    #             value = value.nominal_value
    #     if self.__isBooleanValue:
    #         value = bool(value)
    #     return value

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

    # def convert_unit(self, unit_str: str):
    #     """
    #     Convert the value from one unit system to another. You will should use
    #     `compatible_units` to see if your new unit is compatible.

    #     :param unit_str: New unit in string form
    #     """
    #     new_unit = ureg.parse_expression(unit_str)
    #     self._value = self._value.to(new_unit)
    #     self._units = new_unit
    #     self._args['value'] = self.raw_value
    #     self._args['units'] = str(self.unit)

    # # @cached_property
    # @property
    # def compatible_units(self) -> List[str]:
    #     """
    #     Returns all possible units for which the current unit can be converted.

    #     :return: Possible conversion units
    #     """
    #     return [str(u) for u in self.unit.compatible_units()]

    # def __repr__(self):
    #     """Return printable representation of a Descriptor/Parameter object."""
    #     class_name = self.__class__.__name__
    #     obj_name = self.name
    #     if self.__isBooleanValue:
    #         obj_value = self.raw_value
    #     else:
    #         obj_value = self._value.magnitude
    #     if isinstance(obj_value, float):
    #         obj_value = '{:0.04f}'.format(obj_value)
    #     obj_units = ''
    #     if not self.unit.dimensionless:
    #         obj_units = ' {:~P}'.format(self.unit)
    #     out_str = f"<{class_name} '{obj_name}': {obj_value}{obj_units}>"
    #     return out_str

    # def to_obj_type(self, data_type: Type[Parameter], *kwargs):
    #     """
    #     Convert between a `Parameter` and a `Descriptor`.

    #     :param data_type: class constructor of what we want to be
    #     :param kwargs: Additional keyword/value pairs for conversion
    #     :return: self as a new type
    #     """
    #     pickled_obj = self.encode()
    #     pickled_obj.update(kwargs)
    #     if '@class' in pickled_obj.keys():
    #         pickled_obj['@class'] = data_type.__name__
    #     return data_type.from_dict(pickled_obj)

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return printable representation of a Descriptor/Parameter object."""

    def __copy__(self) -> DescriptorBase:
        """Return a copy of the Descriptor/Parameter object."""
        new_obj = self.__class__.from_dict(self.as_dict())
        new_obj._callback = self._callback
        return new_obj
