# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest


def test_calculators_package_removed():
    # The interface factory was removed in 3.0 (#294); calculators are
    # owned by the technique libraries.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('easyscience.fitting.calculators')
