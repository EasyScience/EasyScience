#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/easyscience

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import warnings

from .minimizer_base import MinimizerBase  # noqa: E402
from .utils import FitError  # noqa: F401, E402
from .utils import FitResults  # noqa: F401, E402

imported = -1
try:
    from .minimizer_lmfit import LMFit  # noqa: F401, E402

    imported += 1
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('lmfit has not been installed.', ImportWarning, stacklevel=2)
try:
    from .minimizer_bumps import Bumps  # noqa: F401, E402

    imported += 1
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('bumps has not been installed.', ImportWarning, stacklevel=2)
try:
    from .minimizer_dfo import DFO  # noqa: F401, E402

    imported += 1
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('dfo-ls has not been installed.', ImportWarning, stacklevel=2)

engines: list = MinimizerBase._engines
