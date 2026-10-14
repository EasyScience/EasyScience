# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

import numpy as np
import pytest

from easyscience import Parameter
from easyscience.fitting.minimizers.minimizer_bumps import Bumps
from easyscience.fitting.minimizers.minimizer_lmfit import LMFit

COVARIANCE = np.array([[0.04, 0.03], [0.03, 0.09]])


@pytest.fixture
def minimizer() -> LMFit:
    minimizer = LMFit(
        obj='obj',
        fit_function='fit_function',
        minimizer_enum=MagicMock(package='lmfit', method='leastsq'),
    )
    minimizer._cached_pars = {
        'a': Parameter(1.0, display_name='a'),
        'b': Parameter(2.0, display_name='b'),
    }
    return minimizer


def lmfit_result(covar, var_names, redchi=2.0, errorbars=True, **extra):
    result = MagicMock()
    result.covar = covar
    result.var_names = var_names
    result.redchi = redchi
    result.errorbars = errorbars
    result.scale_covar = extra.get('scale_covar', True)
    return result


class TestStoreFitCovariance:
    def test_a_valid_covariance_is_attached(self, minimizer):
        # When
        parameters = list(minimizer._cached_pars.values())

        # Then
        record = minimizer._store_fit_covariance(COVARIANCE, parameters)

        # Expect
        assert record is minimizer._covariance
        assert all(parameter.covariance is record for parameter in parameters)

    def test_none_removes_the_previous_record(self, minimizer):
        # When
        parameters = list(minimizer._cached_pars.values())
        minimizer._store_fit_covariance(COVARIANCE, parameters)

        # Then
        result = minimizer._store_fit_covariance(None, parameters)

        # Expect
        assert result is None
        assert minimizer._covariance is None
        assert all(parameter.covariance is None for parameter in parameters)

    def test_an_invalid_covariance_gives_none(self, minimizer):
        # When
        parameters = list(minimizer._cached_pars.values())

        # Then
        result = minimizer._store_fit_covariance(np.diag([1.0, -1.0]), parameters)

        # Expect
        assert result is None
        assert minimizer._covariance is None

    def test_results_hold_the_record_and_the_keys(self, minimizer):
        # When
        parameters = list(minimizer._cached_pars.values())
        record = minimizer._store_fit_covariance(COVARIANCE, parameters[::-1])

        # Then
        found, labels = minimizer._covariance_results()

        # Expect
        assert found is record
        assert labels == ['pb', 'pa']

    def test_no_record_gives_empty_results(self, minimizer):
        # When Then Expect
        assert minimizer._covariance_results() == (None, [])


class TestLMFitCovariance:
    def test_the_scaling_is_undone(self, minimizer):
        # When
        result = lmfit_result(2.0 * COVARIANCE, ['pa', 'pb'], redchi=2.0)

        # Then
        covariance, parameters = minimizer._lmfit_covariance(result)

        # Expect
        np.testing.assert_allclose(covariance, COVARIANCE)
        assert parameters == list(minimizer._cached_pars.values())

    def test_without_scaling_nothing_is_divided(self, minimizer):
        # When
        result = lmfit_result(COVARIANCE, ['pa', 'pb'], scale_covar=False)

        # Then
        covariance, _ = minimizer._lmfit_covariance(result)

        # Expect
        np.testing.assert_allclose(covariance, COVARIANCE)

    def test_parameters_follow_var_names_not_the_cache_order(self, minimizer):
        # When
        result = lmfit_result(2.0 * COVARIANCE, ['pb', 'pa'], redchi=2.0)

        # Then
        _, parameters = minimizer._lmfit_covariance(result)

        # Expect
        assert parameters == [minimizer._cached_pars['b'], minimizer._cached_pars['a']]

    @pytest.mark.parametrize(
        'result',
        [
            lmfit_result(COVARIANCE, ['pa', 'pb'], errorbars=False),
            lmfit_result(None, ['pa', 'pb']),
            lmfit_result(COVARIANCE, ['pa', 'pz']),
            lmfit_result(COVARIANCE, ['pa', 'pb'], redchi=0.0),
            lmfit_result(COVARIANCE, ['pa', 'pb'], redchi=float('nan')),
        ],
    )
    def test_no_covariance(self, minimizer, result):
        # When Then Expect
        assert minimizer._lmfit_covariance(result) == (None, [])


class TestBumpsCovariance:
    def test_the_driver_covariance_is_an_array(self):
        # When
        driver = MagicMock()
        driver.cov.return_value = COVARIANCE

        # Then Expect
        np.testing.assert_allclose(Bumps._driver_covariance(driver), COVARIANCE)

    @pytest.mark.parametrize('error', [np.linalg.LinAlgError, ValueError, FloatingPointError])
    def test_a_failing_driver_gives_none(self, error):
        # When
        driver = MagicMock()
        driver.cov.side_effect = error

        # Then Expect
        assert Bumps._driver_covariance(driver) is None

    def test_the_record_is_stored_in_the_order_of_the_problem(self):
        # When
        minimizer = Bumps(
            obj='obj',
            fit_function='fit_function',
            minimizer_enum=MagicMock(package='bumps', method='lm'),
        )
        minimizer._cached_pars = {
            'a': Parameter(0.0, display_name='a'),
            'b': Parameter(0.0, display_name='b'),
        }
        fit_result = MagicMock()
        fit_result.x = [1.0, 2.0]
        fit_result.dx = np.array([0.2, 0.3])
        fit_result.cov = COVARIANCE

        # Then
        minimizer._set_parameter_fit_result(fit_result, False, ['b', 'a'])

        # Expect
        record = minimizer._covariance
        assert record.parameters == [minimizer._cached_pars['b'], minimizer._cached_pars['a']]
        np.testing.assert_allclose(record.covariance, COVARIANCE)
