# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Mathematical functions for EasyScience ``Parameter`` objects.

Use them through the package namespace, for example
``es.math.sin(angle)``. Each function takes ``Parameter`` objects and
returns a new ``Parameter`` with the value, the propagated variance and
exact fitting bounds. The implementation lives in
``easyscience.variable.math``; this module only re-exports its public
names.
"""

from easyscience.variable.math import *  # noqa: F403
from easyscience.variable.math import __all__  # noqa: F401
