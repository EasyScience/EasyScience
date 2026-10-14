# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the DREAM run-settings validation helper."""

import pytest

from easyscience.fitting.samplers.validation import validate_run_settings


class TestValidateRunSettings:
    @pytest.mark.parametrize(
        'kwargs, match',
        [
            ({'samples': 0}, 'samples must be a positive integer'),
            ({'samples': -1}, 'samples must be a positive integer'),
            ({'samples': 10.0}, 'samples must be a positive integer'),
            # bool is an int subclass; True must not sneak through as 1.
            ({'samples': True}, 'samples must be a positive integer'),
            ({'burn': -1}, 'burn must be a non-negative integer'),
            ({'burn': 1.5}, 'burn must be a non-negative integer'),
            ({'burn': True}, 'burn must be a non-negative integer'),
            ({'burn': False}, 'burn must be a non-negative integer'),
            ({'thin': 0}, 'thin must be a positive integer'),
            ({'thin': 2.0}, 'thin must be a positive integer'),
            ({'thin': True}, 'thin must be a positive integer'),
        ],
    )
    def test_invalid_settings_raise(self, kwargs, match):
        settings = {'samples': 10, 'burn': 0, 'thin': 1}
        settings.update(kwargs)
        with pytest.raises(ValueError, match=match):
            validate_run_settings(**settings)

    def test_valid_settings_pass(self):
        validate_run_settings(samples=1, burn=0, thin=1)
        validate_run_settings(samples=10000, burn=2000, thin=10)
