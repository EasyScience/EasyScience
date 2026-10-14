# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easyscience import Parameter
from easyscience.fitting.samplers.covariance import posterior_covariance

COVARIANCE = np.array([[0.04, 0.03], [0.03, 0.09]])


@pytest.fixture
def parameters():
    return {
        'alpha': Parameter(1.0, display_name='alpha'),
        'beta': Parameter(2.0, display_name='beta'),
    }


@pytest.fixture
def draws():
    return np.random.default_rng(1).multivariate_normal([1.0, 2.0], COVARIANCE, size=20000)


class TestPosteriorCovariance:
    def test_covariance_of_the_draws(self, parameters, draws):
        # When Then
        record = posterior_covariance(parameters, ['alpha', 'beta'], draws)

        # Expect
        np.testing.assert_allclose(record.covariance, COVARIANCE, atol=3e-3)
        assert record.source == 'posterior'
        assert record.names == ('alpha', 'beta')
        assert record.samples.shape == draws.shape

    def test_columns_follow_the_given_names(self, parameters, draws):
        # When Then
        record = posterior_covariance(parameters, ['beta', 'alpha'], draws[:, ::-1])

        # Expect
        assert record.parameters == [parameters['beta'], parameters['alpha']]
        np.testing.assert_allclose(record.covariance, COVARIANCE[::-1, ::-1], atol=3e-3)
        np.testing.assert_allclose(
            record.submatrix([parameters['alpha'], parameters['beta']]), COVARIANCE, atol=3e-3
        )

    def test_it_is_not_attached_to_the_parameters(self, parameters, draws):
        # When
        record = posterior_covariance(parameters, ['alpha', 'beta'], draws)

        # Then Expect
        assert parameters['alpha'].covariance is None
        record.attach()
        assert parameters['alpha'].covariance is record

    def test_an_unknown_name_gives_none(self, parameters, draws):
        # When Then Expect
        assert posterior_covariance(parameters, ['alpha', 'gamma'], draws) is None

    def test_too_few_or_bad_draws_give_none(self, parameters, draws):
        # When Then Expect
        assert posterior_covariance(parameters, ['alpha', 'beta'], draws[:1]) is None
        assert posterior_covariance(parameters, ['alpha', 'beta'], draws[:, :1]) is None
        bad = draws.copy()
        bad[0, 0] = np.nan
        assert posterior_covariance(parameters, ['alpha', 'beta'], bad) is None
