# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import iminuit
import numpy as np
import pytest

from easyscience import AvailableMinimizers
from easyscience import Fitter
from easyscience import Parameter
from easyscience.base_classes import ModelBase

pytestmark = pytest.mark.fast

START = {'amplitude': 5.0, 'center': 0.0, 'width': 1.0, 'background': 0.0}
NAMES = list(START)


class Peak(ModelBase):
    """A Gaussian peak on a flat background."""

    def __init__(self, amplitude, center, width, background):
        super().__init__()
        self._amplitude = Parameter(amplitude, display_name='amplitude', min=0.0, max=100.0)
        self._center = Parameter(center, display_name='center', min=-1.0, max=1.0)
        self._width = Parameter(width, display_name='width', min=1e-3, max=2.0)
        self._background = Parameter(background, display_name='background', min=-5.0, max=5.0)

    amplitude = property(lambda self: self._amplitude)
    center = property(lambda self: self._center)
    width = property(lambda self: self._width)
    background = property(lambda self: self._background)

    def __call__(self, x):
        peak = self._amplitude.value * np.exp(
            -0.5 * ((x - self._center.value) / self._width.value) ** 2
        )
        return peak + self._background.value


@pytest.fixture
def data():
    rng = np.random.default_rng(seed=1)
    x = np.linspace(-3.0, 3.0, 80)
    sigma = np.full_like(x, 0.4)
    truth = Peak(amplitude=8.0, center=0.3, width=0.6, background=1.0)
    return x, truth(x) + rng.normal(scale=sigma), sigma


def fit_with(data, enum: AvailableMinimizers, **kwargs):
    x, y, sigma = data
    model = Peak(**START)
    fitter = Fitter(model, model)
    fitter.switch_minimizer(enum)
    result = fitter.fit(x=x, y=y, weights=1 / sigma, **kwargs)
    return model, result


def values(model: Peak) -> np.ndarray:
    return np.array([getattr(model, name).value for name in NAMES])


def errors(model: Peak) -> np.ndarray:
    return np.array([getattr(model, name).error for name in NAMES])


def test_values_equal_lmfit(data) -> None:
    # When Then
    minuit, _ = fit_with(data, AvailableMinimizers.Minuit)
    lmfit, _ = fit_with(data, AvailableMinimizers.LMFit)

    # Expect
    assert values(minuit) == pytest.approx(values(lmfit), rel=1e-3)


def test_default_errors_equal_bumps(data) -> None:
    # lmfit and bumps estimate the covariance from the Jacobian (Gauss-Newton); Minuit uses the
    # full Hessian, so a strongly non-linear parameter such as the width differs by a few per cent.
    # When Then
    minuit, _ = fit_with(data, AvailableMinimizers.Minuit)
    bumps, _ = fit_with(data, AvailableMinimizers.Bumps_lm)

    # Expect
    assert errors(minuit) == pytest.approx(errors(bumps), rel=0.05)


def test_scale_covar_errors_equal_lmfit(data) -> None:
    # Same few per cent as above, from Gauss-Newton against the full Hessian.
    # When Then
    minuit, _ = fit_with(data, AvailableMinimizers.Minuit, scale_covar=True)
    lmfit, _ = fit_with(data, AvailableMinimizers.LMFit)

    # Expect
    assert errors(minuit) == pytest.approx(errors(lmfit), rel=0.05)


def test_engine_result_exposes_the_covariance(data) -> None:
    # When Then
    model, result = fit_with(data, AvailableMinimizers.Minuit)

    # Expect
    assert isinstance(result.engine_result, iminuit.Minuit)
    covariance = np.asarray(result.engine_result.covariance)
    assert covariance.shape == (4, 4)
    assert np.sort(np.sqrt(np.diag(covariance))) == pytest.approx(np.sort(errors(model)), rel=1e-9)
