# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
``es.math`` applied to the parameters of a fit.

``es.math`` takes scalar ``Parameter`` objects, so it cannot be applied over an ``x`` array
inside a model. Its use is to derive quantities, with uncertainties, from the parameters a fit
produces. These tests fit small models and check the derived value and variance against closed
forms and against sampling the fitted distribution.

The models are test-only, like ``AbsSin`` in ``test_fitter.py``:

* ``StraightLine``: dimensionless parameters, so every dimensionless function can be applied.
* ``AngleLine``: a free parameter in degrees, for the trigonometric functions and ``sinc``.
* ``Quadrature``: ``a cos + b sin`` with uncorrelated ``a`` and ``b``, for amplitude and phase.
* ``GaussianPeak``: parametrised by its FWHM; the width is derived with ``log`` and ``sqrt``
  *inside* ``__call__`` on every evaluation (so a ``Parameter`` is created per call), and an
  in-window fraction is derived after the fit with ``erf``.
"""

import gc
import math
import time

import numpy as np
import pytest

import easyscience as es
from easyscience import AvailableMinimizers
from easyscience import Fitter
from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import ModelBase

pytestmark = pytest.mark.fast

SIGMA = 0.05  # standard deviation of the noise added to the synthetic data
OMEGA = 2.0 * np.pi  # angular frequency of the Quadrature model (period 1)

_erf = np.vectorize(math.erf)
_erfc = np.vectorize(math.erfc)


@pytest.fixture(autouse=True)
def clear():
    global_object.map._clear()


# ----------------------------------------------------------------------
# Test-only models
# ----------------------------------------------------------------------


class StraightLine(ModelBase):
    """y = slope * x + intercept"""

    def __init__(self, slope: float, intercept: float):
        super().__init__()
        self._slope = Parameter(slope)
        self._intercept = Parameter(intercept)

    @property
    def slope(self) -> Parameter:
        return self._slope

    @slope.setter
    def slope(self, value: float) -> None:
        self._slope.value = value

    @property
    def intercept(self) -> Parameter:
        return self._intercept

    @intercept.setter
    def intercept(self, value: float) -> None:
        self._intercept.value = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.slope.value * x + self.intercept.value


class AngleLine(ModelBase):
    """y = tan(angle) * x + intercept, with the angle a free parameter in degrees."""

    def __init__(self, angle: float, intercept: float):
        super().__init__()
        self._angle = Parameter(angle, unit='deg', min=1.0, max=80.0)
        self._intercept = Parameter(intercept)

    @property
    def angle(self) -> Parameter:
        return self._angle

    @angle.setter
    def angle(self, value: float) -> None:
        self._angle.value = value

    @property
    def intercept(self) -> Parameter:
        return self._intercept

    @intercept.setter
    def intercept(self, value: float) -> None:
        self._intercept.value = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tan(np.deg2rad(self.angle.value)) * x + self.intercept.value


class Quadrature(ModelBase):
    """y = a cos(OMEGA x) + b sin(OMEGA x)"""

    def __init__(self, a: float, b: float):
        super().__init__()
        self._a = Parameter(a)
        self._b = Parameter(b)

    @property
    def a(self) -> Parameter:
        return self._a

    @a.setter
    def a(self, value: float) -> None:
        self._a.value = value

    @property
    def b(self) -> Parameter:
        return self._b

    @b.setter
    def b(self, value: float) -> None:
        self._b.value = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.a.value * np.cos(OMEGA * x) + self.b.value * np.sin(OMEGA * x)


class GaussianPeak(ModelBase):
    """
    y = amplitude * exp(-(x - center)**2 / (2 sigma**2)), with sigma derived from the FWHM.

    ``sigma()`` uses ``es.math`` (``log``, ``sqrt``) and so creates new ``Parameter`` objects on
    every call; ``__call__`` calls it on every evaluation.
    """

    def __init__(self, amplitude: float, center: float, fwhm: float):
        super().__init__()
        self._amplitude = Parameter(amplitude)
        self._center = Parameter(center)
        self._fwhm = Parameter(fwhm, min=0.1, max=10.0)
        self._two = Parameter(2.0)  # a constant, so that log() receives a Parameter

    @property
    def amplitude(self) -> Parameter:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._amplitude.value = value

    @property
    def center(self) -> Parameter:
        return self._center

    @center.setter
    def center(self, value: float) -> None:
        self._center.value = value

    @property
    def fwhm(self) -> Parameter:
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value: float) -> None:
        self._fwhm.value = value

    def sigma(self) -> Parameter:
        """sigma = FWHM / (2 sqrt(2 ln 2)), with value and variance propagated."""
        return self._fwhm / (2 * es.math.sqrt(2 * es.math.log(self._two)))

    def fraction_within(self, window: float) -> Parameter:
        """Fraction of the area within +-window of the center: erf(window / (sqrt(2) sigma))."""
        return es.math.erf(window / (self.sigma() * math.sqrt(2.0)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sigma = self.sigma().value
        return self.amplitude.value * np.exp(-((x - self.center.value) ** 2) / (2 * sigma**2))


FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def fit(model: ModelBase, x: np.ndarray, y: np.ndarray) -> None:
    """Fit ``model`` to ``y`` with weights = 1/sigma (the repo's convention)."""
    result = Fitter(model, model).fit(x, y, weights=np.full_like(x, 1.0 / SIGMA))
    assert result.success


def noisy(values: np.ndarray, seed: int) -> np.ndarray:
    return values + np.random.default_rng(seed).normal(0.0, SIGMA, values.size)


# ----------------------------------------------------------------------
# Dimensionless functions on the parameters of a fitted line
# ----------------------------------------------------------------------

# (function, parameter it is applied to, reference, derivative of the reference).
# slope is about 0.5 (inside (-1, 1)) and intercept about 2.0 (above 1), so every function
# is evaluated well inside its domain.
UNARY_CASES = [
    ('exp', 'slope', np.exp, np.exp),
    ('erf', 'slope', _erf, lambda x: 2 / np.sqrt(np.pi) * np.exp(-(x**2))),
    ('erfc', 'slope', _erfc, lambda x: -2 / np.sqrt(np.pi) * np.exp(-(x**2))),
    ('sinh', 'slope', np.sinh, np.cosh),
    ('cosh', 'slope', np.cosh, np.sinh),
    ('tanh', 'slope', np.tanh, lambda x: 1 / np.cosh(x) ** 2),
    ('arcsinh', 'slope', np.arcsinh, lambda x: 1 / np.sqrt(x**2 + 1)),
    ('arcsin', 'slope', np.arcsin, lambda x: 1 / np.sqrt(1 - x**2)),
    ('arccos', 'slope', np.arccos, lambda x: -1 / np.sqrt(1 - x**2)),
    ('arctan', 'slope', np.arctan, lambda x: 1 / (1 + x**2)),
    ('arctanh', 'slope', np.arctanh, lambda x: 1 / (1 - x**2)),
    ('log', 'intercept', np.log, lambda x: 1 / x),
    ('log10', 'intercept', np.log10, lambda x: 1 / (x * np.log(10))),
    ('sqrt', 'intercept', np.sqrt, lambda x: 0.5 / np.sqrt(x)),
    ('arccosh', 'intercept', np.arccosh, lambda x: 1 / np.sqrt(x**2 - 1)),
]
UNARY_IDS = [case[0] for case in UNARY_CASES]


@pytest.fixture
def fitted_line():
    x = np.linspace(0.0, 5.0, 200)
    model = StraightLine(slope=0.1, intercept=1.0)
    fit(model, x, noisy(0.5 * x + 2.0, seed=1))
    return model, x


class TestDimensionlessFunctionsOnFittedLine:
    @pytest.mark.parametrize(
        ('name', 'attribute', 'reference', 'derivative'), UNARY_CASES, ids=UNARY_IDS
    )
    def test_value_and_variance_follow_the_fitted_parameter(
        self, fitted_line, name, attribute, reference, derivative
    ):
        # WHEN
        model, _ = fitted_line
        fitted = getattr(model, attribute)

        # THEN
        derived = getattr(es.math, name)(fitted)

        # EXPECT
        assert fitted.variance > 0  # the fit produced an uncertainty to propagate
        assert derived.value == pytest.approx(float(reference(fitted.value)), rel=1e-12)
        expected_variance = float(derivative(fitted.value)) ** 2 * fitted.variance
        assert derived.variance == pytest.approx(expected_variance, rel=1e-9)

    @pytest.mark.parametrize(
        ('name', 'attribute', 'reference', 'derivative'), UNARY_CASES, ids=UNARY_IDS
    )
    def test_variance_agrees_with_sampling_the_fitted_distribution(
        self, fitted_line, name, attribute, reference, derivative
    ):
        # WHEN
        model, _ = fitted_line
        fitted = getattr(model, attribute)
        rng = np.random.default_rng(7)
        samples = rng.normal(fitted.value, np.sqrt(fitted.variance), 200_000)

        # THEN
        derived = getattr(es.math, name)(fitted)

        # EXPECT
        # An independent check of the first-order (delta) method: the spread of f(samples).
        assert np.std(reference(samples)) == pytest.approx(np.sqrt(derived.variance), rel=0.02)

    @pytest.mark.parametrize(('alias', 'canonical'), [('asin', 'arcsin'), ('atanh', 'arctanh')])
    def test_aliases_give_the_same_result_as_the_canonical_function(
        self, fitted_line, alias, canonical
    ):
        # WHEN
        model, _ = fitted_line

        # THEN
        via_alias = getattr(es.math, alias)(model.slope)
        via_canonical = getattr(es.math, canonical)(model.slope)

        # EXPECT
        assert (via_alias.value, via_alias.variance) == (
            via_canonical.value,
            via_canonical.variance,
        )


# ----------------------------------------------------------------------
# A fitted angle in degrees
# ----------------------------------------------------------------------


@pytest.fixture
def fitted_angle():
    x = np.linspace(0.0, 5.0, 200)
    model = AngleLine(angle=20.0, intercept=1.0)
    fit(model, x, noisy(np.tan(np.deg2rad(30.0)) * x + 2.0, seed=2))
    return model


class TestFittedAngle:
    def test_fit_recovers_the_angle_within_its_uncertainty(self, fitted_angle):
        # WHEN THEN
        angle = fitted_angle.angle

        # EXPECT
        assert abs(angle.value - 30.0) < 5 * np.sqrt(angle.variance)

    @pytest.mark.parametrize(
        ('name', 'derivative'),
        [
            ('sin', np.cos),
            ('cos', lambda t: -np.sin(t)),
            ('tan', lambda t: 1 / np.cos(t) ** 2),
        ],
    )
    def test_trigonometric_value_and_variance_with_the_angle_in_degrees(
        self, fitted_angle, name, derivative
    ):
        # WHEN
        angle = fitted_angle.angle
        radians = np.deg2rad(angle.value)

        # THEN
        derived = getattr(es.math, name)(angle)

        # EXPECT
        # d f / d(deg) = f'(rad) * pi / 180, so the variance picks up (pi / 180) ** 2.
        expected_variance = float(derivative(radians)) ** 2 * angle.variance * np.deg2rad(1.0) ** 2
        assert derived.unit == 'dimensionless'
        assert derived.value == pytest.approx(float(getattr(np, name)(radians)), rel=1e-12)
        assert derived.variance == pytest.approx(expected_variance, rel=1e-9)

    @pytest.mark.parametrize(
        ('name', 'expected_min', 'expected_max'),
        [
            ('sin', np.sin(np.deg2rad(1.0)), np.sin(np.deg2rad(80.0))),
            ('cos', np.cos(np.deg2rad(80.0)), np.cos(np.deg2rad(1.0))),
            ('tan', np.tan(np.deg2rad(1.0)), np.tan(np.deg2rad(80.0))),
        ],
    )
    def test_bounds_of_the_fitted_angle_are_mapped_through_the_function(
        self, fitted_angle, name, expected_min, expected_max
    ):
        # WHEN
        angle = fitted_angle.angle  # bounds [1, 80] deg survive the fit

        # THEN
        derived = getattr(es.math, name)(angle)

        # EXPECT
        assert derived.min == pytest.approx(expected_min, abs=1e-12)
        assert derived.max == pytest.approx(expected_max, abs=1e-12)

    def test_sinc_of_the_fitted_angle_after_converting_to_radians(self, fitted_angle):
        # WHEN
        angle = fitted_angle.angle
        angle.convert_unit('rad')  # sinc accepts rad or dimensionless, not deg

        # THEN
        derived = es.math.sinc(angle)

        # EXPECT
        x = angle.value
        expected_variance = ((x * np.cos(x) - np.sin(x)) / x**2) ** 2 * angle.variance
        assert derived.value == pytest.approx(np.sin(x) / x, rel=1e-12)
        assert derived.variance == pytest.approx(expected_variance, rel=1e-9)


# ----------------------------------------------------------------------
# Amplitude and phase of a fitted a cos + b sin
# ----------------------------------------------------------------------


@pytest.fixture
def fitted_quadrature():
    x = np.arange(0.0, 4.0, 0.01)  # four whole periods, endpoint excluded
    model = Quadrature(a=1.0, b=0.1)
    fit(model, x, noisy(1.5 * np.cos(OMEGA * x) + 0.8 * np.sin(OMEGA * x), seed=3))
    return model, x


class TestFittedQuadrature:
    def test_premise_a_and_b_are_uncorrelated(self, fitted_quadrature):
        # WHEN
        _, x = fitted_quadrature
        design = np.column_stack([np.cos(OMEGA * x), np.sin(OMEGA * x)])

        # THEN
        covariance = np.linalg.inv(design.T @ design)
        correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])

        # EXPECT
        # Over whole periods cos and sin are orthogonal; the delta method relies on this.
        assert abs(correlation) < 1e-9

    def test_amplitude_from_parameter_arithmetic_and_sqrt(self, fitted_quadrature):
        # WHEN
        model, _ = fitted_quadrature
        a, b = model.a, model.b

        # THEN
        amplitude = es.math.sqrt(a**2 + b**2)

        # EXPECT
        expected_value = np.hypot(a.value, b.value)
        expected_variance = (a.value**2 * a.variance + b.value**2 * b.variance) / expected_value**2
        assert amplitude.value == pytest.approx(expected_value, rel=1e-12)
        assert amplitude.variance == pytest.approx(expected_variance, rel=1e-9)
        assert abs(amplitude.value - 1.7) < 5 * np.sqrt(amplitude.variance)

    def test_phase_from_arctan2(self, fitted_quadrature):
        # WHEN
        model, _ = fitted_quadrature
        a, b = model.a, model.b

        # THEN
        phase = es.math.arctan2(b, a)

        # EXPECT
        r2 = a.value**2 + b.value**2
        expected_variance = (a.value**2 * b.variance + b.value**2 * a.variance) / r2**2
        assert phase.unit == 'rad'
        assert phase.value == pytest.approx(np.arctan2(b.value, a.value), rel=1e-12)
        assert phase.variance == pytest.approx(expected_variance, rel=1e-9)
        assert abs(phase.value - np.arctan2(0.8, 1.5)) < 5 * np.sqrt(phase.variance)

    def test_phase_variance_agrees_with_sampling_uncorrelated_parameters(self, fitted_quadrature):
        # WHEN
        model, _ = fitted_quadrature
        a, b = model.a, model.b
        rng = np.random.default_rng(11)
        sampled_a = rng.normal(a.value, np.sqrt(a.variance), 200_000)
        sampled_b = rng.normal(b.value, np.sqrt(b.variance), 200_000)

        # THEN
        phase = es.math.arctan2(b, a)

        # EXPECT
        assert np.std(np.arctan2(sampled_b, sampled_a)) == pytest.approx(
            np.sqrt(phase.variance), rel=0.03
        )

    def test_phase_in_degrees_scales_value_and_variance(self, fitted_quadrature):
        # WHEN
        model, _ = fitted_quadrature
        a, b = model.a, model.b

        # THEN
        in_radians = es.math.arctan2(b, a)
        in_degrees = es.math.arctan2(b, a, unit='deg')

        # EXPECT
        assert in_degrees.unit == 'deg'
        assert in_degrees.value == pytest.approx(np.degrees(in_radians.value), rel=1e-12)
        assert in_degrees.variance == pytest.approx(
            in_radians.variance * np.degrees(1.0) ** 2, rel=1e-9
        )


# ----------------------------------------------------------------------
# Deriving values inside __call__ (a Parameter is created per evaluation)
# ----------------------------------------------------------------------

# NOTE(): Under DFO the library itself prints "RuntimeWarning: invalid value encountered in sqrt"
# (minimizer_base.py, in the error-matrix calculation). It is not caused by es.math, and it is
# deliberately not filtered here so that it stays visible.
#
# NOTE(): The uncertainty a fit reports depends on the minimizer. For the same data the variance of
# the fitted FWHM was about 6.2e-5 (LMFit), 6.7e-5 (Bumps) and 3.0e-6 (DFO). Whether DFO's error
# estimate is wrong is a question for the lead developer. A derived variance is only as good as the
# variance it starts from, so these tests compare a derived variance only with the variance of the
# same engine's fit, never across engines.
ENGINES = [
    None,
    AvailableMinimizers.LMFit,
    AvailableMinimizers.Bumps,
    AvailableMinimizers.DFO,
    AvailableMinimizers.Minuit,
]


def count_parameters() -> int:
    """Number of live ``Parameter`` objects (after a garbage collection)."""
    gc.collect()
    return sum(isinstance(obj, Parameter) for obj in gc.get_objects())


@pytest.fixture
def gaussian_data():
    x = np.linspace(-5.0, 5.0, 200)
    truth = GaussianPeak(amplitude=3.0, center=0.4, fwhm=1.8)
    return x, noisy(truth(x), seed=5)


class TestGaussianPeakWithDerivedWidth:
    def test_derived_width_value_and_variance_match_the_closed_form(self):
        # WHEN
        peak = GaussianPeak(amplitude=1.0, center=0.0, fwhm=1.8)
        peak.fwhm.variance = 0.01

        # THEN
        sigma = peak.sigma()

        # EXPECT
        # sigma = fwhm / c with c a constant, so var(sigma) = var(fwhm) / c**2.
        assert sigma.value == pytest.approx(1.8 / FWHM_PER_SIGMA, rel=1e-12)
        assert sigma.variance == pytest.approx(0.01 / FWHM_PER_SIGMA**2, rel=1e-9)

    def test_curve_equals_the_numpy_reference(self):
        # WHEN
        x = np.linspace(-5.0, 5.0, 50)
        peak = GaussianPeak(amplitude=3.0, center=0.4, fwhm=1.8)

        # THEN
        y = peak(x)

        # EXPECT
        sigma = 1.8 / FWHM_PER_SIGMA
        assert y == pytest.approx(3.0 * np.exp(-((x - 0.4) ** 2) / (2 * sigma**2)), rel=1e-12)

    @pytest.mark.parametrize('engine', ENGINES, ids=[str(engine) for engine in ENGINES])
    def test_fit_recovers_the_parameters_with_every_minimizer(self, gaussian_data, engine):
        # WHEN
        x, y = gaussian_data
        peak = GaussianPeak(amplitude=2.0, center=0.0, fwhm=1.0)
        fitter = Fitter(peak, peak)
        if engine is not None:
            try:
                fitter.switch_minimizer(engine)
            except AttributeError:
                pytest.skip(reason=f'{engine} is not installed')

        # THEN
        result = fitter.fit(x, y, weights=np.full_like(x, 1.0 / SIGMA))

        # EXPECT
        assert result.success
        assert peak.amplitude.value == pytest.approx(3.0, abs=0.05)
        assert peak.center.value == pytest.approx(0.4, abs=0.05)
        assert peak.fwhm.value == pytest.approx(1.8, abs=0.05)

    def test_in_window_fraction_after_the_fit(self, gaussian_data):
        # WHEN
        x, y = gaussian_data
        peak = GaussianPeak(amplitude=2.0, center=0.0, fwhm=1.0)
        fit(peak, x, y)
        window = 1.0

        # THEN
        fraction = peak.fraction_within(window)

        # EXPECT
        # u = window * c / (sqrt(2) fwhm) depends on the fwhm alone, so the delta method is exact
        # to first order: var(f) = (erf'(u) * du/dfwhm)**2 * var(fwhm), with du/dfwhm = -u/fwhm.
        fwhm = peak.fwhm
        u = window * FWHM_PER_SIGMA / (math.sqrt(2.0) * fwhm.value)
        slope = 2 / math.sqrt(math.pi) * math.exp(-(u**2)) * (-u / fwhm.value)
        assert fraction.value == pytest.approx(math.erf(u), rel=1e-12)
        assert fraction.variance == pytest.approx(slope**2 * fwhm.variance, rel=1e-9)

    def test_in_window_fraction_variance_agrees_with_sampling_the_fwhm(self, gaussian_data):
        # WHEN
        x, y = gaussian_data
        peak = GaussianPeak(amplitude=2.0, center=0.0, fwhm=1.0)
        fit(peak, x, y)
        rng = np.random.default_rng(13)
        fwhm_samples = rng.normal(peak.fwhm.value, np.sqrt(peak.fwhm.variance), 200_000)

        # THEN
        fraction = peak.fraction_within(1.0)

        # EXPECT
        sampled = _erf(1.0 * FWHM_PER_SIGMA / (math.sqrt(2.0) * fwhm_samples))
        assert np.std(sampled) == pytest.approx(np.sqrt(fraction.variance), rel=0.03)


class TestCostOfDerivingInsideCall:
    """
    ``GaussianPeak.__call__`` creates several ``Parameter`` objects per evaluation (the
    ``log``, ``sqrt`` and arithmetic intermediates). These tests pin that this is affordable and
    does not leak.
    """

    def test_parameters_created_per_call_are_garbage_collected(self):
        # WHEN
        x = np.linspace(-5.0, 5.0, 200)
        peak = GaussianPeak(amplitude=3.0, center=0.4, fwhm=1.8)
        peak(x)  # warm-up
        before = count_parameters()

        # THEN
        for _ in range(500):
            peak(x)
        after = count_parameters()

        # EXPECT
        assert after == before

    def test_a_single_evaluation_stays_cheap(self):
        # WHEN
        x = np.linspace(-5.0, 5.0, 200)
        peak = GaussianPeak(amplitude=3.0, center=0.4, fwhm=1.8)
        peak(x)  # warm-up
        durations = []

        # THEN
        for _ in range(200):
            start = time.perf_counter()
            peak(x)
            durations.append(time.perf_counter() - start)

        # EXPECT
        # About 1 ms on a development machine; the bound is generous for slow CI runners.
        assert np.median(durations) < 0.05

    def test_results_are_snapshots_so_deriving_once_goes_stale(self):
        # WHEN
        peak = GaussianPeak(amplitude=3.0, center=0.4, fwhm=1.8)
        derived_once = peak.sigma()
        stale_value = derived_once.value

        # THEN
        peak.fwhm = 3.0

        # EXPECT
        # Deriving in __init__ (once) would silently not follow the fit; derive in __call__.
        assert derived_once.value == stale_value
        assert peak.sigma().value == pytest.approx(3.0 / FWHM_PER_SIGMA, rel=1e-12)


# ----------------------------------------------------------------------
# Parameters that are correlated after a fit
# ----------------------------------------------------------------------


class TestCorrelatedParameters:
    """
    The slope and intercept of a line are strongly anticorrelated (about -0.87 here). The
    covariance of the fit is used when a function of both (``arctan2``) is propagated.
    """

    def test_premise_slope_and_intercept_are_strongly_correlated(self, fitted_line):
        # WHEN
        _, x = fitted_line
        design = np.column_stack([x, np.ones_like(x)])

        # THEN
        covariance = np.linalg.inv(design.T @ design) * SIGMA**2
        correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])

        # EXPECT
        assert correlation < -0.8

    def test_arctan2_of_correlated_parameters_matches_the_full_covariance(self, fitted_line):
        # WHEN
        model, x = fitted_line
        design = np.column_stack([x, np.ones_like(x)])
        covariance = np.linalg.inv(design.T @ design) * SIGMA**2  # true parameter covariance
        slope, intercept = model.slope.value, model.intercept.value
        r2 = slope**2 + intercept**2
        jacobian = np.array([-intercept / r2, slope / r2])  # d atan2(intercept, slope) / d(p)
        variance_with_covariance = float(jacobian @ covariance @ jacobian)

        # THEN
        derived = es.math.arctan2(model.intercept, model.slope)

        # EXPECT
        assert derived.variance == pytest.approx(variance_with_covariance, rel=0.2)
