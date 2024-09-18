#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

from typing import TYPE_CHECKING
from typing import List
from typing import Tuple

from easyscience import global_object
from easyscience.global_object.hugger.property import LoggedProperty

if TYPE_CHECKING:
    from easyscience.Objects.ObjectClasses import BV
    from easyscience.Objects.ObjectClasses import B


def addLoggedProp(inst: BV, name: str, *args, **kwargs) -> None:
    cls = type(inst)
    annotations = getattr(cls, '__annotations__', False)
    if not hasattr(cls, '__perinstance'):
        cls = type(cls.__name__, (cls,), {'__module__': inst.__module__})
        cls.__perinstance = True
        if annotations:
            cls.__annotations__ = annotations
        inst.__old_class__ = inst.__class__
        inst.__class__ = cls
    setattr(cls, name, LoggedProperty(*args, **kwargs))


def addProp(inst: BV, name: str, *args, **kwargs) -> None:
    cls = type(inst)
    annotations = getattr(cls, '__annotations__', False)
    if not hasattr(cls, '__perinstance'):
        cls = type(cls.__name__, (cls,), {'__module__': __name__})
        cls.__perinstance = True
        if annotations:
            cls.__annotations__ = annotations
        inst.__old_class__ = inst.__class__
        inst.__class__ = cls

    setattr(cls, name, property(*args, **kwargs))


def removeProp(inst: BV, name: str) -> None:
    cls = type(inst)
    if not hasattr(cls, '__perinstance'):
        cls = type(cls.__name__, (cls,), {'__module__': __name__})
        cls.__perinstance = True
        inst.__old_class__ = inst.__class__
        inst.__class__ = cls
    delattr(cls, name)


def generatePath(model_obj: B, skip_first: bool = False) -> Tuple[List[int], List[str]]:
    pars = model_obj.get_parameters()
    start_idx = 0 + int(skip_first)
    unique_names = []
    names = []
    for par in pars:
        route = global_object.map.reverse_route(par.unique_name, model_obj.unique_name)
        objs = [getattr(global_object.map.get_item_by_key(r), 'name') for r in route]
        objs.reverse()
        names.append('.'.join(objs[start_idx:]))
        unique_names.append(par.unique_name)
    return unique_names, names
