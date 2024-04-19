# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Easyscience contributors (https://github.com/EasyScience)

# flake8: noqa
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib
