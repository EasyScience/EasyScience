# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from easyscience import global_object
from easyscience.base_classes.new_base import NewBase
from easyscience.global_object.undo_redo import property_stack


class DescriptorBase(NewBase, metaclass=abc.ABCMeta):
    """
    This is the base of all variable descriptions for models.

    It contains all information to describe a single unique property of
    an object. This description includes a value as well as optionally a
    unit, description and url (for reference material). Also implemented
    is a callback so that the value can be read/set from a linked
    library object.

    A ``Descriptor`` is typically something which describes part of a
    model and is non-fittable and generally changes the state of an
    object.
    """

    _global_object = global_object

    def __init__(
        self,
        *,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        """
        This is the base of variables for models.

        It contains all information to describe a single unique property
        of an object. This description includes a description and url
        (for reference material).

        A ``Descriptor`` is typically something which describes part of
        a model and is non-fittable and generally changes the state of
        an object.

        Parameters
        ----------
        unique_name : Optional[str], default=None
            Unique identifier for this object. By default, None.
        description : Optional[str], default=None
            A brief summary of what this object is. By default, None.
        url : Optional[str], default=None
            Lookup url for documentation/information. By default, None.
        display_name : Optional[str], default=None
            A pretty name for the object. Falls back to ``unique_name``
            when not given. By default, None.

        Raises
        ------
        TypeError
            If any optional string field has an invalid type.
        """

        super().__init__(unique_name=unique_name, display_name=display_name)

        if description is not None and not isinstance(description, str):
            raise TypeError('Description must be a string or None')
        if description is None:
            description = ''
        self._description: str = description

        if url is not None and not isinstance(url, str):
            raise TypeError('url must be a string')
        if url is None:
            url = ''
        self._url: str = url

    @property
    def description(self) -> str:
        """
        Get the description of the object.

        Returns
        -------
        str
            Description of the object.
        """
        return self._description

    @description.setter
    def description(self, description: Optional[str]) -> None:
        """
        Set the description of the object.

        Parameters
        ----------
        description : Optional[str]
            Description of the object.

        Raises
        ------
        TypeError
            If ``description`` is neither a string nor ``None``.
        """
        if description is not None and not isinstance(description, str):
            raise TypeError('Description must be a string or None')
        self._description = description

    @property
    def url(self) -> str:
        """
        Get the url of the object.

        Returns
        -------
        str
            Url of the object.
        """
        return self._url

    @url.setter
    def url(self, url: Optional[str]) -> None:
        """
        Set the url of the object.

        Parameters
        ----------
        url : Optional[str]
            Url of the object.

        Raises
        ------
        TypeError
            If ``url`` is neither a string nor ``None``.
        """
        if url is not None and not isinstance(url, str):
            raise TypeError('url must be a string')
        self._url = url

    @property
    @abc.abstractmethod
    def value(self) -> Any:
        """Get the value of the object."""

    @value.setter
    @abc.abstractmethod
    def value(self, value: Any) -> None:
        """Set the value of the object."""

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return printable representation of the object."""
