# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the engine-agnostic helpers in ``engine_base.py``."""

import numpy as np
import pytest

from easyscience.fitting.engine_base import EngineBase

validate_arrays = EngineBase.validate_arrays


class TestValidateArrays:
    @staticmethod
    def _data():
        return {
            'x': np.array([1.0, 2.0]),
            'y': np.array([0.1, 0.2]),
            'weights': np.array([1.0, 1.0]),
        }

    @pytest.mark.parametrize(
        'overrides, match',
        [
            ({'y': np.array([0.1])}, 'x and y must have the same shape'),
            ({'x': np.array([1.0, np.nan])}, 'x cannot contain NaN'),
            ({'x': np.array([1.0, np.inf])}, 'x cannot contain NaN'),
            ({'y': np.array([0.1, np.nan])}, 'y cannot contain NaN'),
            ({'y': np.array([0.1, np.inf])}, 'y cannot contain NaN'),
            ({'weights': np.array([1.0])}, 'Weights must have the same shape'),
            ({'weights': np.array([1.0, np.nan])}, 'Weights cannot be NaN'),
            ({'weights': np.array([1.0, np.inf])}, 'Weights cannot be NaN'),
            ({'weights': np.array([1.0, 0.0])}, 'Weights must be strictly positive'),
            ({'weights': np.array([1.0, -1.0])}, 'Weights must be strictly positive'),
            ({'x': np.array([1.0, None], dtype=object)}, 'x must hold numeric values'),
            ({'y': np.array(['a', 'b'])}, 'y must hold numeric values'),
            ({'weights': np.asarray(None)}, 'weights must hold numeric values'),
        ],
    )
    def test_invalid_arrays_raise(self, overrides, match):
        data = self._data()
        data.update(overrides)
        with pytest.raises(ValueError, match=match):
            validate_arrays(**data)

    def test_valid_arrays_pass(self):
        validate_arrays(**self._data())
