# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipy import stats

from easyscience.variable.prior import Prior
from easyscience.variable.prior import validate_prior

NORMAL = {'distribution': 'norm', 'loc': 1.0, 'scale': 0.5}


class TestValidation:
    def test_a_valid_prior_is_copied(self):
        # When
        spec = dict(NORMAL)

        # Then
        checked = validate_prior(spec)
        spec['loc'] = 99.0

        # Expect
        assert checked == NORMAL

    @pytest.mark.parametrize('spec', [None, 'norm', [('distribution', 'norm')], {'loc': 1.0}])
    def test_rejects_something_that_is_not_a_prior(self, spec):
        # When Then Expect
        with pytest.raises(TypeError):
            validate_prior(spec)

    @pytest.mark.parametrize(
        'spec',
        [
            {'distribution': 'nonsense'},
            {'distribution': 'bernoulli'},
            {'distribution': '__name__'},
            {'distribution': 'norm', 'scale': -1.0},
            {'distribution': 'norm', 'shape': 2.0},
            {'distribution': 'beta', 'a': 2.0},
        ],
    )
    def test_rejects_an_unusable_distribution(self, spec):
        # When Then Expect
        with pytest.raises(ValueError):
            validate_prior(spec)

    def test_rejects_a_non_numeric_argument(self):
        # When Then Expect
        with pytest.raises(TypeError, match='loc'):
            validate_prior({'distribution': 'norm', 'loc': 'one'})


class TestTruncation:
    def test_the_density_is_renormalised_inside_the_bounds(self):
        # When
        prior = Prior(NORMAL, 0.0, 2.0)
        x = np.linspace(0.0, 2.0, 200001)

        # Then
        area = np.trapezoid(np.exp(prior.logpdf(x)), x)

        # Expect
        assert area == pytest.approx(1.0, abs=1e-6)
        assert prior.logpdf(-0.1) == -np.inf
        assert prior.logpdf(2.1) == -np.inf

    def test_without_finite_bounds_the_prior_is_the_distribution(self):
        # When
        prior = Prior(NORMAL, -np.inf, np.inf)

        # Then Expect
        assert prior.logpdf(1.3) == pytest.approx(stats.norm(1.0, 0.5).logpdf(1.3))
        assert prior.is_proper

    def test_ppf_inverts_cdf(self):
        # When
        prior = Prior({'distribution': 'cauchy', 'loc': 0.5, 'scale': 0.2}, -1.0, 3.0)
        x = np.linspace(-0.9, 2.9, 30)

        # Then
        back = prior.ppf(prior.cdf(x))

        # Expect
        np.testing.assert_allclose(back, x, atol=1e-9)

    def test_ppf_stays_inside_the_bounds(self):
        # When
        prior = Prior(NORMAL, 0.5, 1.5)

        # Then
        values = prior.ppf(np.linspace(0.0, 1.0, 101))

        # Expect
        assert values.min() >= 0.5
        assert values.max() <= 1.5
        assert values[0] == pytest.approx(0.5)
        assert values[-1] == pytest.approx(1.5)

    def test_samples_follow_the_truncated_distribution(self):
        # When
        prior = Prior(NORMAL, 1.0, np.inf)
        samples = prior.rvs(20000, np.random.default_rng(0))

        # Then
        statistic = stats.kstest(samples, prior.cdf).statistic

        # Expect
        assert samples.min() >= 1.0
        assert statistic < 0.02

    def test_a_prior_without_mass_in_the_bounds_is_rejected(self):
        # When Then Expect
        with pytest.raises(ValueError, match='no probability'):
            Prior({'distribution': 'norm', 'loc': 0.0, 'scale': 1e-3}, 50.0, 60.0)


class TestUniform:
    def test_the_default_is_uniform_between_the_bounds(self):
        # When
        prior = Prior(None, 2.0, 6.0)

        # Then Expect
        assert prior.is_uniform
        assert prior.is_proper
        assert prior.spec is None
        assert prior.logpdf(3.0) == pytest.approx(-np.log(4.0))
        assert prior.logpdf(7.0) == -np.inf
        assert prior.ppf(0.25) == pytest.approx(3.0)
        assert prior.cdf(5.0) == pytest.approx(0.75)

    def test_infinite_bounds_are_improper(self):
        # When
        prior = Prior(None, -np.inf, np.inf)

        # Then Expect
        assert not prior.is_proper
        assert prior.logpdf(1.0) == 0.0
        with pytest.raises(ValueError, match='finite bounds'):
            prior.ppf(0.5)
        with pytest.raises(ValueError, match='finite bounds'):
            prior.cdf(0.5)
