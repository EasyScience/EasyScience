from __future__ import annotations

from typing import Any
from typing import Optional

# from easyscience import borg
# from easyscience.Utils.Exceptions import CoreSetException
from easyscience.Utils.UndoRedo import property_stack_deco

from .descriptor_base import DescriptorBase


class DescriptorStr(DescriptorBase):
    def __init__(
        self,
        name: str,
        string: str,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        # callback: Optional[property] = None,
        #        enabled: Optional[bool] = True,
        parent: Optional[Any] = None,
    ):
        if not isinstance(string, str):
            raise ValueError(f'{string=} must be type str')
        super().__init__(
            name=name,
            description=description,
            url=url,
            display_name=display_name,
            # callback=callback,
            #            enabled=enabled,
            parent=parent,
        )
        self._string = string

    @property
    def value(self) -> str:
        """
        Get the value of self as a pint. This is should be usable for most cases. If a pint
        is not acceptable then the raw value can be obtained through `obj.raw_value`.

        :return: Value of self with unit.
        """
        # Cached property? Should reference callback.
        # Also should reference for undo/redo
        # if self._callback.fget is not None:
        #     string = self._callback.fget()
        #     if string != self._string:
        #         self._string = string
        return self._string

    @value.setter
    @property_stack_deco
    def value(self, value: str) -> None:
        """
        Set the value of self. This creates a pint with a unit.

        :param value: New value of self
        :return: None
        """
        # if not self._enabled:
        #     if borg.debug:
        #         raise CoreSetException(f'{str(self)} is not enabled.')
        #     return
        self._string = value
        # if self._callback.fset is not None:
        #     self._callback.fset(value)

    def __repr__(self) -> str:
        """Return printable representation."""
        class_name = self.__class__.__name__
        obj_name = self._name
        obj_value = self._string
        return f"<{class_name} '{obj_name}': {obj_value}>"

    def __copy__(self) -> DescriptorStr:
        return super().__copy__()
