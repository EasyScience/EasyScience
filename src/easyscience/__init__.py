#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = 'github.com/wardsimon'

import warnings

import numpy as np  # noqa: F401  This is used in the other codebases that uses easyscience
import pint

from easyscience.__version__ import __version__ as __version__
from easyscience.global_object import GlobalObject

ureg = pint.UnitRegistry()
global_object = GlobalObject()
global_object.instantiate_stack()
global_object.stack.enabled = False

# alias for global_object, remove later
def __getattr__(name):
    if name == 'borg':
        warnings.warn("The 'borg' has been renamed to 'global_object', this alias will be deprecated in the future", DeprecationWarning)  # noqa: E501
        print("The 'borg' has been renamed to 'global_object', this alias will be deprecated in the future")
        return global_object