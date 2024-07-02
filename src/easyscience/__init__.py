#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = 'github.com/wardsimon'

import numpy as np  # noqa: F401  This is used in the other codebases that uses easyscience
import pint

from easyscience.__version__ import __version__ as __version__
from easyscience.global_object import GlobalObject

default_fitting_engine = 'lmfit'

ureg = pint.UnitRegistry()
global_object = GlobalObject()
global_object.instantiate_stack()
global_object.stack.enabled = False
