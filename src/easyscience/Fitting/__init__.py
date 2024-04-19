#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/easyscience

__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

import warnings

imported = -1
try:
    from easyscience.Fitting.lmfit import lmfit  # noqa: F401, E402

    imported += 1
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn("lmfit has not been installed.", ImportWarning, stacklevel=2)
try:
    from easyscience.Fitting.bumps import bumps  # noqa: F401, E402

    imported += 1
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn("bumps has not been installed.", ImportWarning, stacklevel=2)
try:
    from easyscience.Fitting.DFO_LS import DFO  # noqa: F401, E402

    imported += 1
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn("dfo-ls has not been installed.", ImportWarning, stacklevel=2)

from easyscience.Fitting.fitting_template import FittingTemplate  # noqa: E402

engines: list = FittingTemplate._engines
