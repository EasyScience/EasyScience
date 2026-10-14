# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np
import pytest

from easyscience import AvailableMinimizers
from easyscience import Fitter
from easyscience import Parameter
from easyscience.base_classes import ModelBase
from easyscience.fitting.sampler import Sampler

pytestmark = pytest.mark.fast

PRIOR = {'distribution': 'norm', 'loc': 5.0, 'scale': 0.05}


class Line(ModelBase):
    """A straight line whose parameters have no bounds."""

    def __init__(self, slope_prior=None, slope=2.0):
        super().__init__()
        self._slope = Parameter(slope, display_name='slope', prior=slope_prior)
        self._intercept = Parameter(1.0, display_name='intercept')

    slope = property(lambda self: self._slope)
    intercept = property(lambda self: self._intercept)

    def __call__(self, x):
        return self._slope.value * x + self._intercept.value


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 4.0, 8)
    sigma = np.full_like(x, 3.0)
    return x, 2.0 * x + 1.0 + rng.normal(scale=sigma), sigma


def fit(model, data, enum):
    x, y, sigma = data
    fitter = Fitter(model, model)
    fitter.switch_minimizer(enum)
    return fitter.fit(x=x, y=y, weights=1 / sigma)


class TestBumpsFitWithAPrior:
    def test_a_strong_prior_pulls_the_fit(self, data):
        # When
        free, pulled = Line(), Line(slope_prior=PRIOR)

        # Then
        fit(free, data, AvailableMinimizers.Bumps_simplex)
        fit(pulled, data, AvailableMinimizers.Bumps_simplex)

        # Expect
        assert abs(free.slope.value - 5.0) > 1.0
        assert pulled.slope.value == pytest.approx(5.0, abs=0.3)

    def test_a_prior_that_agrees_with_the_data_changes_little(self, data):
        # When
        free = Line()
        fit(free, data, AvailableMinimizers.Bumps_simplex)
        weak = Line(slope_prior={'distribution': 'norm', 'loc': free.slope.value, 'scale': 100.0})

        # Then
        fit(weak, data, AvailableMinimizers.Bumps_simplex)

        # Expect
        assert weak.slope.value == pytest.approx(free.slope.value, abs=0.05)


class TestSamplerWithAPrior:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_the_posterior_follows_the_prior_when_the_data_are_weak(self, data):
        # When
        x, y, sigma = data
        free, pulled = Line(), Line(slope_prior=PRIOR, slope=5.0)

        # Then
        free_draws = Sampler(free, free, x, y, 1 / sigma).sample(samples=3000, burn=300, thin=1)
        pulled_draws = Sampler(pulled, pulled, x, y, 1 / sigma).sample(
            samples=3000, burn=300, thin=1
        )

        # Expect
        names = pulled_draws.param_names
        slope = pulled_draws.draws[:, names.index(pulled.slope.unique_name)]
        free_slope = free_draws.draws[:, free_draws.param_names.index(free.slope.unique_name)]
        assert abs(np.mean(free_slope) - 5.0) > 1.0
        assert np.mean(slope) == pytest.approx(5.0, abs=0.4)
        assert np.std(slope) < 0.5 * np.std(free_slope)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_a_truncated_prior_never_leaves_the_bounds(self, data):
        # When
        x, y, sigma = data
        model = Line()
        model.slope.min, model.slope.max = 0.0, 3.0
        model.slope.prior = {'distribution': 'norm', 'loc': 1.0, 'scale': 5.0}

        # Then
        results = Sampler(model, model, x, y, 1 / sigma).sample(samples=1500, burn=200, thin=1)

        # Expect
        slope = results.draws[:, results.param_names.index(model.slope.unique_name)]
        assert slope.min() >= 0.0
        assert slope.max() <= 3.0


class TestEnginesThatIgnoreAPrior:
    @pytest.mark.parametrize(
        'enum',
        [AvailableMinimizers.LMFit, AvailableMinimizers.Minuit, AvailableMinimizers.DFO_leastsq],
    )
    def test_they_warn_and_use_the_bounds_only(self, data, enum, caplog):
        # When
        model = Line(slope_prior=PRIOR)

        # Then
        with caplog.at_level(logging.WARNING):
            fit(model, data, enum)

        # Expect
        assert any('ignores the priors of slope' in record.message for record in caplog.records)
        assert abs(model.slope.value - 5.0) > 1.0

    def test_no_warning_without_a_prior(self, data, caplog):
        # When
        model = Line()

        # Then
        with caplog.at_level(logging.WARNING):
            fit(model, data, AvailableMinimizers.LMFit)

        # Expect
        assert not any('ignores the priors' in record.message for record in caplog.records)
