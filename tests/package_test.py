# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Easyscience contributors (https://github.com/EasyScience)
import easyscience as pkg


def test_has_version():
    assert hasattr(pkg, '__version__') # noqa S101
