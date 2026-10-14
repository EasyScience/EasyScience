# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy import stats

from easyscience import Parameter
from easyscience.fitting.prior_transform import PriorTransform


@pytest.fixture
def parameters():
    return [
        Parameter(1.0, min=0.0, max=4.0, display_name='flat'),
        Parameter(
            1.0, display_name='bell', prior={'distribution': 'norm', 'loc': 1.0, 'scale': 0.5}
        ),
        Parameter(
            2.0,
            min=1e-3,
            max=1e3,
            display_name='scale',
            prior={'distribution': 'loguniform', 'a': 1e-3, 'b': 1e3},
        ),
    ]


class TestPriorTransform:
    def test_the_corners_of_the_cube_are_the_bounds(self, parameters):
        # When
        transform = PriorTransform(parameters)

        # Then
        low, high = transform(np.zeros(3)), transform(np.ones(3))

        # Expect
        assert low[0] == 0.0 and high[0] == 4.0
        assert low[2] == pytest.approx(1e-3) and high[2] == pytest.approx(1e3)

    def test_a_point_and_many_points(self, parameters):
        # When
        transform = PriorTransform(parameters)
        u = np.random.default_rng(0).uniform(size=(50, 3))

        # Then
        many = transform(u)
        one = transform(u[0])

        # Expect
        assert many.shape == (50, 3)
        np.testing.assert_allclose(one, many[0])

    def test_the_inverse_undoes_the_transform(self, parameters):
        # When
        transform = PriorTransform(parameters)
        u = np.random.default_rng(1).uniform(0.01, 0.99, size=(40, 3))

        # Then
        back = transform.inverse(transform(u))

        # Expect
        np.testing.assert_allclose(back, u, atol=1e-8)

    def test_uniform_points_follow_each_prior(self, parameters):
        # When
        transform = PriorTransform(parameters)
        u = np.random.default_rng(2).uniform(size=(20000, 3))

        # Then
        theta = transform(u)

        # Expect
        assert stats.kstest(theta[:, 0], stats.uniform(0, 4).cdf).statistic < 0.02
        assert stats.kstest(theta[:, 1], stats.norm(1.0, 0.5).cdf).statistic < 0.02
        assert stats.kstest(theta[:, 2], stats.loguniform(1e-3, 1e3).cdf).statistic < 0.02

    def test_the_log_prior_is_the_sum_and_minus_infinity_outside(self, parameters):
        # When
        transform = PriorTransform(parameters)

        # Then
        inside = transform.log_prior(np.array([1.0, 1.0, 2.0]))
        outside = transform.log_prior(np.array([5.0, 1.0, 2.0]))

        # Expect
        expected = (
            -np.log(4.0)
            + stats.norm(1.0, 0.5).logpdf(1.0)
            + stats.loguniform(1e-3, 1e3).logpdf(2.0)
        )
        assert inside == pytest.approx(expected)
        assert outside == -np.inf

    def test_a_flat_prior_needs_finite_bounds(self):
        # When
        parameter = Parameter(1.0, display_name='open')

        # Then Expect
        with pytest.raises(ValueError, match="'open'"):
            PriorTransform([parameter])

    def test_a_proper_prior_does_not_need_bounds(self):
        # When
        parameter = Parameter(1.0, prior={'distribution': 'norm', 'loc': 1.0, 'scale': 1.0})

        # Then
        transform = PriorTransform([parameter])

        # Expect
        assert transform(np.array([0.5]))[0] == pytest.approx(1.0)
        assert len(transform) == 1

    def test_the_wrong_number_of_coordinates_is_rejected(self, parameters):
        # When
        transform = PriorTransform(parameters)

        # Then Expect
        with pytest.raises(ValueError, match='coordinates'):
            transform(np.zeros(2))
