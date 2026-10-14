# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import version

from .global_object import GlobalObject

# Must be executed before any other imports
global_object = GlobalObject()
global_object.instantiate_stack()
global_object.stack.enabled = False


from .base_classes import EasyList  # noqa: E402
from .base_classes import ModelBase  # noqa: E402
from .fitting import AvailableMinimizers  # noqa: E402
from .fitting import Fitter  # noqa: E402
from .variable import DescriptorNumber  # noqa: E402
from .variable import Parameter  # noqa: E402
from .variable import ParameterCovariance  # noqa: E402

# Must come after the imports above: math needs the variable and base classes to be initialised
from . import math  # noqa: E402  # isort: skip

__version__ = version('easyscience')

__all__ = [
    '__version__',
    'global_object',
    'ModelBase',
    'EasyList',
    'AvailableMinimizers',
    'Fitter',
    'DescriptorNumber',
    'Parameter',
    'ParameterCovariance',
]
