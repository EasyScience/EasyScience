# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pickle

import numpy as np
import pytest
from scipy import stats

from easyscience import Parameter
from easyscience.fitting.minimizers.bumps_utils import to_bumps_parameter
from easyscience.fitting.minimizers.bumps_utils.prior import PriorBounds

NORMAL = {'distribution': 'norm', 'loc': 1.0, 'scale': 0.2}


@pytest.fixture
def bounds():
    return PriorBounds(Parameter(1.0, min=0.0, max=2.0, prior=NORMAL).prior_distribution)


class TestPriorBounds:
    def test_the_limits_are_the_bounds_of_the_parameter(self, bounds):
        # When Then Expect
        assert bounds.limits == (0.0, 2.0)
        assert bounds.dof == 1

    def test_nllf_is_minus_the_log_density(self, bounds):
        # When Then
        inside, outside = bounds.nllf(1.1), bounds.nllf(2.5)

        # Expect
        mass = stats.norm(1.0, 0.2).cdf(2.0) - stats.norm(1.0, 0.2).cdf(0.0)
        assert inside == pytest.approx(-stats.norm(1.0, 0.2).logpdf(1.1) + np.log(mass))
        assert outside == np.inf

    def test_unit_interval_round_trip(self, bounds):
        # When
        x = np.linspace(0.1, 1.9, 7)

        # Then
        back = bounds.put01(bounds.get01(x))

        # Expect
        np.testing.assert_allclose(back, x, atol=1e-9)

    def test_random_values_lie_inside_the_limits(self, bounds):
        # When Then
        values = bounds.random(500)

        # Expect
        assert values.min() >= 0.0 and values.max() <= 2.0

    def test_it_can_be_pickled(self, bounds):
        # When Then
        restored = pickle.loads(pickle.dumps(bounds))

        # Expect
        assert restored.nllf(1.1) == pytest.approx(bounds.nllf(1.1))


class TestToBumpsParameter:
    def test_without_a_prior_the_bounds_are_uniform(self):
        # When
        result = to_bumps_parameter(Parameter(1.0, min=0.0, max=2.0))

        # Then Expect
        assert not isinstance(result.prior, PriorBounds)
        assert result.bounds == (0.0, 2.0)

    def test_with_a_prior_bumps_gets_the_truncated_prior(self):
        # When
        result = to_bumps_parameter(Parameter(1.0, min=0.0, max=2.0, prior=NORMAL))

        # Then Expect
        assert isinstance(result.prior, PriorBounds)
        assert result.bounds == (0.0, 2.0)
        assert result.prior.nllf(1.0) < result.prior.nllf(1.5)

    def test_a_prior_works_without_bounds(self):
        # When
        result = to_bumps_parameter(Parameter(1.0, prior=NORMAL))

        # Then Expect
        assert result.bounds == (-np.inf, np.inf)
        assert np.isfinite(result.prior.nllf(1.0))
