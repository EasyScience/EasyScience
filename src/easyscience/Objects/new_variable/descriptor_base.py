#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

import abc
import weakref
from typing import Any
from typing import Optional

from easyscience import borg
from easyscience.Objects.core import ComponentSerializer
from easyscience.Utils.UndoRedo import property_stack_deco


class DescriptorBase(ComponentSerializer, metaclass=abc.ABCMeta):
    """
    This is the base of all variable descriptions for models. It contains all information to describe a single
    unique property of an object. This description includes a name and value as well as optionally a unit, description
    and url (for reference material). Also implemented is a callback so that the value can be read/set from a linked
    library object.

    A `Descriptor` is typically something which describes part of a model and is non-fittable and generally changes the
    state of an object.
    """

    _borg = borg

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        #        callback: Optional[property] = None,
        # enabled: Optional[bool] = True,
        parent: Optional[Any] = None,
    ):
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
        # Let the collective know we've been assimilated
        self._parent = parent
        self._borg.map.add_vertex(self, obj_type='created')
        # Make the connection between self and parent
        if parent is not None:
            self._borg.map.add_edge(parent, self)

        self._name: str = name

        #        self._enabled = enabled

        if description is None:
            description = ''
        self._description: str = description

        self._display_name: str = display_name

        if url is None:
            url = ''
        self._url: str = url

        # if callback is None:
        #     callback = property()
        # self._callback: property = callback

        #        finalizer = None
        # if self._callback.fdel is not None:
        #     weakref.finalize(self, self._callback.fdel)

    #        self._finalizer = finalizer

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
    # def enabled(self) -> bool:
    #     """
    #     Logical property to see if the objects value can be directly set.

    #     :return: Can the objects value be set
    #     """
    #     return self._enabled

    # @enabled.setter
    # @property_stack_deco
    # def enabled(self, value: bool) -> None:
    #     """
    #     Enable and disable the direct setting of an objects value field.

    #     :param value: True - objects value can be set, False - the opposite
    #     """
    #     self._enabled = value

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return printable representation of a Descriptor/Parameter object."""

    def __copy__(self) -> DescriptorBase:
        """Return a copy of the Descriptor/Parameter object."""
        new_obj = self.__class__.from_dict(self.as_dict())
        #        new_obj._callback = self._callback
        return new_obj
