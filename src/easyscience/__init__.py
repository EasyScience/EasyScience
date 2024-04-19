#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = 'github.com/wardsimon'

from importlib import metadata

import numpy as np  # noqa: F401  This is used in the other codebases that uses easyscience
import pint

from easyscience.Objects.Borg import Borg

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = '0.0.0'

default_fitting_engine = 'lmfit'

ureg = pint.UnitRegistry()
borg = Borg()
borg.instantiate_stack()
borg.stack.enabled = False
