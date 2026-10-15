# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for ``easyscience.math``.

Conventions encoded here (agreed with the lead developer):

* **Values are validated strictly.** A value outside a function's domain,
  or a result that is not finite, raises ``ValueError`` (such values break
  model-fitting loops).
* **Bounds are permissive.** Bounds never raise: bounds outside a domain are
  clipped to it, infinite bounds are accepted and bounds that cross an
  asymptote (``tan``) blow up to ``(-inf, inf)``.
* **Bounds are exact**, not sampled: they are compared against a brute-force
  dense evaluation of the function over the input interval.
"""

import importlib
import math
import warnings

import numpy as np
import pytest
import scipp as sc
from scipp import UnitError

import easyscience as es
import easyscience.variable as es_variable
from easyscience import DescriptorNumber
from easyscience import Parameter
from easyscience import global_object
from easyscience import math as es_math
from easyscience.variable import DescriptorArray

INF = np.inf
SINC_GLOBAL_MIN = -0.21723362821122166  # sinc(x) at its first minimum, x ~ 4.4934

UNARY_FUNCTIONS = [
    'exp',
    'log',
    'log10',
    'sqrt',
    'erf',
    'erfc',
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinc',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
]
ALL_FUNCTIONS = [*UNARY_FUNCTIONS, 'arctan2']


# ----------------------------------------------------------------------
# Helpers / reference implementations
# ----------------------------------------------------------------------


def make(value, unit='', *, min=-INF, max=INF, variance=0.0, name='x') -> Parameter:
    """Shorthand for building a ``Parameter`` with the bounds under test."""
    return Parameter(name=name, value=value, unit=unit, variance=variance, min=min, max=max)


def fn(name: str):
    return getattr(es_math, name)


def unnormalized_sinc(x):
    """sin(x)/x, the definition Scipp uses (``np.sinc`` is the normalised one)."""
    return np.sinc(np.asarray(x) / np.pi)


_erf = np.vectorize(math.erf)
_erfc = np.vectorize(math.erfc)


def dense_range(np_func, lo, hi, n=200_001):
    """Brute-force oracle: the range of ``np_func`` sampled on a dense grid."""
    xs = np.linspace(lo, hi, n)
    with np.errstate(all='ignore'):
        ys = np_func(xs)
    return float(np.min(ys)), float(np.max(ys))


def assert_bounds(result: Parameter, expected_min, expected_max, tol=1e-9):
    assert np.isclose(result.min, expected_min, rtol=0, atol=tol), (result.min, expected_min)
    assert np.isclose(result.max, expected_max, rtol=0, atol=tol), (result.max, expected_max)


# name -> (numpy/math reference, derivative of the reference, value, unit)
VALUE_CASES = {
    'exp': (np.exp, np.exp, 0.3, ''),
    'log': (np.log, lambda x: 1 / x, 2.0, ''),
    'log10': (np.log10, lambda x: 1 / (x * np.log(10)), 2.0, ''),
    'sqrt': (np.sqrt, lambda x: 0.5 / np.sqrt(x), 2.0, ''),
    'erf': (math.erf, lambda x: 2 / np.sqrt(np.pi) * np.exp(-(x**2)), 0.4, ''),
    'erfc': (math.erfc, lambda x: -2 / np.sqrt(np.pi) * np.exp(-(x**2)), 0.4, ''),
    'sin': (np.sin, np.cos, 0.7, 'rad'),
    'cos': (np.cos, lambda x: -np.sin(x), 0.7, 'rad'),
    'tan': (np.tan, lambda x: 1 / np.cos(x) ** 2, 0.7, 'rad'),
    'arcsin': (np.arcsin, lambda x: 1 / np.sqrt(1 - x**2), 0.4, ''),
    'arccos': (np.arccos, lambda x: -1 / np.sqrt(1 - x**2), 0.4, ''),
    'arctan': (np.arctan, lambda x: 1 / (1 + x**2), 0.4, ''),
    'sinc': (unnormalized_sinc, lambda x: (x * np.cos(x) - np.sin(x)) / x**2, 1.3, 'rad'),
    'sinh': (np.sinh, np.cosh, 0.4, ''),
    'cosh': (np.cosh, np.sinh, 0.4, ''),
    'tanh': (np.tanh, lambda x: 1 / np.cosh(x) ** 2, 0.4, ''),
    'arcsinh': (np.arcsinh, lambda x: 1 / np.sqrt(x**2 + 1), 0.4, ''),
    'arccosh': (np.arccosh, lambda x: 1 / np.sqrt(x**2 - 1), 1.7, ''),
    'arctanh': (np.arctanh, lambda x: 1 / (1 - x**2), 0.4, ''),
}


@pytest.fixture(autouse=True)
def clear():
    global_object.map._clear()


# ----------------------------------------------------------------------
# Value and variance
# ----------------------------------------------------------------------


class TestValueAndVariance:
    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_value_matches_reference(self, name):
        # WHEN
        reference, _, value, unit = VALUE_CASES[name]
        x = make(value, unit)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert isinstance(result, Parameter)
        assert result.value == pytest.approx(float(reference(value)), rel=1e-12)

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_variance_matches_delta_method(self, name):
        # WHEN
        # sigma_y^2 = f'(x)^2 * sigma_x^2. This also covers erf/erfc, for which
        # Scipp itself has no variance support and EasyScience must supply it.
        _, derivative, value, unit = VALUE_CASES[name]
        variance = 0.04
        x = make(value, unit, variance=variance)

        # THEN
        result = fn(name)(x)

        # EXPECT
        expected = float(derivative(value)) ** 2 * variance
        assert result.variance == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize('name', ['sin', 'cos', 'tan'])
    def test_variance_in_degrees_follows_the_unit_conversion(self, name):
        # WHEN
        _, derivative, _, _ = VALUE_CASES[name]
        angle_in_degrees, variance_in_degrees2 = 30.0, 1.0
        x = make(angle_in_degrees, 'deg', variance=variance_in_degrees2)

        # THEN
        result = fn(name)(x)

        # EXPECT
        # d f / d(deg) = f'(rad) * pi / 180, so the variance picks up (pi / 180) ** 2.
        radians = np.deg2rad(angle_in_degrees)
        expected = float(derivative(radians)) ** 2 * variance_in_degrees2 * np.deg2rad(1.0) ** 2
        assert result.variance == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    @pytest.mark.parametrize('variance', [0.0, None])
    def test_no_input_variance_gives_no_output_variance(self, name, variance):
        # WHEN
        _, _, value, unit = VALUE_CASES[name]
        x = make(value, unit, variance=variance)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert result.variance is None

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_inputs_are_not_modified(self, name):
        # WHEN
        _, _, value, unit = VALUE_CASES[name]
        x = make(value, unit, variance=0.04, min=value - 0.1, max=value + 0.1)
        before = (x.value, x.variance, x.min, x.max, x.unit)

        # THEN
        fn(name)(x)

        # EXPECT
        assert (x.value, x.variance, x.min, x.max, x.unit) == before

    def test_result_name_is_built_from_function_and_argument(self):
        # WHEN
        angle = make(0.5, 'rad', name='angle')
        y = make(1.0, name='y')
        x = make(2.0, name='x')

        # THEN
        sin_result = es_math.sin(angle)
        sinc_result = es_math.sinc(angle)
        arctan2_result = es_math.arctan2(y, x)

        # EXPECT
        assert sin_result.name == 'sin(angle)'
        assert sinc_result.name == 'sinc(angle)'
        assert arctan2_result.name == 'arctan2(y, x)'


# ----------------------------------------------------------------------
# Bounds: exact, compared against brute-force evaluation
# ----------------------------------------------------------------------

# (function, reference, lo, hi, value, unit)
ORACLE_CASES = [
    # sin: bounds exactly on an extremum (regression: a 100-point grid missed
    # sin(pi/2) = 1 and Parameter.from_scipp then raised "value > max")
    ('sin', np.sin, 0.0, np.pi, np.pi / 2, 'rad'),
    ('sin', np.sin, 0.1, 0.5, 0.3, 'rad'),  # monotonic piece only
    ('sin', np.sin, -10.0, 10.0, 1.0, 'rad'),  # several periods
    ('sin', np.sin, 1.0, 5.0, 2.0, 'rad'),  # contains a max and a min
    ('sin', np.sin, 2.0, 3.0, 2.5, 'rad'),  # decreasing piece only
    ('cos', np.cos, -1.0, 2.0, 0.5, 'rad'),
    ('cos', np.cos, 3.0, 7.0, 4.0, 'rad'),
    ('cos', np.cos, 0.2, 0.9, 0.5, 'rad'),
    ('sinc', unnormalized_sinc, -10.0, 10.0, 1.0, 'rad'),
    ('sinc', unnormalized_sinc, 3.0, 8.0, 5.0, 'rad'),  # first minimum and second maximum
    ('sinc', unnormalized_sinc, 0.5, 2.0, 1.0, 'rad'),
    ('sinc', unnormalized_sinc, -1.0, 20.0, 3.0, ''),  # dimensionless is accepted too
    ('sinc', unnormalized_sinc, 100.0, 110.0, 105.0, 'rad'),  # far-away, tiny lobes
    ('cosh', np.cosh, -1.0, 2.0, 0.5, ''),  # minimum inside
    ('cosh', np.cosh, 1.0, 2.0, 1.5, ''),  # monotonic piece only
    ('tan', np.tan, -1.0, 1.0, 0.5, 'rad'),
    ('exp', np.exp, -2.0, 3.0, 0.5, ''),
    ('log', np.log, 0.5, 8.0, 2.0, ''),
    ('log10', np.log10, 0.5, 8.0, 2.0, ''),
    ('sqrt', np.sqrt, 0.5, 8.0, 2.0, ''),
    ('erf', _erf, -1.0, 2.0, 0.5, ''),
    ('erfc', _erfc, -1.0, 2.0, 0.5, ''),  # decreasing
    ('sinh', np.sinh, -1.0, 2.0, 0.5, ''),
    ('tanh', np.tanh, -1.0, 2.0, 0.5, ''),
    ('arcsinh', np.arcsinh, -1.0, 2.0, 0.5, ''),
    ('arccosh', np.arccosh, 1.5, 6.0, 2.0, ''),
    ('arctanh', np.arctanh, -0.5, 0.8, 0.1, ''),
    ('arcsin', np.arcsin, -0.5, 0.8, 0.1, ''),
    ('arccos', np.arccos, -0.5, 0.7, 0.1, ''),  # decreasing
    ('arctan', np.arctan, -3.0, 5.0, 1.0, ''),
]


class TestBounds:
    @pytest.mark.parametrize(('name', 'reference', 'lo', 'hi', 'value', 'unit'), ORACLE_CASES)
    def test_bounds_match_brute_force_range(self, name, reference, lo, hi, value, unit):
        # WHEN
        x = make(value, unit, min=lo, max=hi)

        # THEN
        result = fn(name)(x)

        # EXPECT
        true_min, true_max = dense_range(reference, lo, hi)
        # Contains everything the dense sampling found (up to rounding) ...
        assert result.min <= true_min + 1e-12
        assert result.max >= true_max - 1e-12
        # ... and is tight: no looser than the sampling resolution.
        assert result.min >= true_min - 1e-6
        assert result.max <= true_max + 1e-6

    @pytest.mark.parametrize(
        ('name', 'value', 'unit', 'expected_min', 'expected_max'),
        [
            ('sin', 0.5, 'rad', -1.0, 1.0),
            ('cos', 0.5, 'rad', -1.0, 1.0),
            ('tan', 0.3, 'rad', -INF, INF),
            ('sinc', 0.5, 'rad', SINC_GLOBAL_MIN, 1.0),
            ('exp', 0.5, '', 0.0, INF),
            ('log', 2.0, '', -INF, INF),
            ('log10', 2.0, '', -INF, INF),
            ('sqrt', 2.0, '', 0.0, INF),
            ('erf', 0.5, '', -1.0, 1.0),
            ('erfc', 0.5, '', 0.0, 2.0),
            ('sinh', 0.5, '', -INF, INF),
            ('cosh', 0.5, '', 1.0, INF),
            ('tanh', 0.5, '', -1.0, 1.0),
            ('arcsinh', 0.5, '', -INF, INF),
            ('arccosh', 2.0, '', 0.0, INF),
            ('arctanh', 0.3, '', -INF, INF),
            ('arcsin', 0.3, '', -np.pi / 2, np.pi / 2),
            ('arccos', 0.3, '', 0.0, np.pi),
            ('arctan', 0.3, '', -np.pi / 2, np.pi / 2),
        ],
    )
    def test_default_infinite_bounds_are_accepted(
        self, name, value, unit, expected_min, expected_max
    ):
        # WHEN
        # A freshly created Parameter has bounds (-inf, inf); that must just work.
        x = make(value, unit)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert_bounds(result, expected_min, expected_max)

    def test_sin_with_degrees_bounds_are_computed_in_radians(self):
        # WHEN
        # Regression: numpy used to be fed 0..90 as if they were radians.
        angle = make(45.0, 'deg', min=0.0, max=90.0)

        # THEN
        result = es_math.sin(angle)

        # EXPECT
        assert result.value == pytest.approx(np.sqrt(0.5))
        assert_bounds(result, 0.0, 1.0)

    def test_sin_bounds_include_an_interior_maximum(self):
        # WHEN
        # sin peaks at 90 deg, which lies inside [15, 95] deg, so the upper bound is 1.0 and
        # NOT sin(95 deg) = 0.9962, the value at the endpoint.
        angle = make(45.0, 'deg', min=15.0, max=95.0)

        # THEN
        result = es_math.sin(angle)

        # EXPECT
        assert result.value == pytest.approx(0.707106781, abs=1e-9)
        assert_bounds(result, 0.258819045, 1.0)

    def test_sin_bounds_are_the_endpoint_values_when_no_extremum_is_inside(self):
        # WHEN
        # sin is increasing on all of [15, 85] deg, so the bounds are just sin(15) and sin(85).
        angle = make(45.0, 'deg', min=15.0, max=85.0)

        # THEN
        result = es_math.sin(angle)

        # EXPECT
        assert result.value == pytest.approx(0.707106781, abs=1e-9)
        assert_bounds(result, 0.258819045, 0.996194698)

    def test_cos_bounds_are_computed_in_radians_for_degree_input(self):
        # WHEN
        angle = make(30.0, 'deg', min=0.0, max=90.0)

        # THEN
        result = es_math.cos(angle)

        # EXPECT
        assert_bounds(result, 0.0, 1.0)

    def test_sin_with_degrees_and_infinite_bounds(self):
        # WHEN
        angle = make(45.0, 'deg')

        # THEN
        result = es_math.sin(angle)

        # EXPECT
        assert_bounds(result, -1.0, 1.0)

    def test_inverse_trig_bounds_are_in_radians(self):
        # WHEN
        x = make(0.5, min=0.0, max=1.0)

        # THEN
        result = es_math.arcsin(x)

        # EXPECT
        assert result.unit == 'rad'
        assert_bounds(result, 0.0, np.pi / 2)

    def test_sinc_global_minimum_is_found(self):
        # WHEN
        x = make(1.0, 'rad', min=-20.0, max=20.0)

        # THEN
        result = es_math.sinc(x)

        # EXPECT
        assert_bounds(result, SINC_GLOBAL_MIN, 1.0)

    def test_sinc_negative_one_sided_infinite_bounds(self):
        # WHEN
        x = make(-150.0, 'rad', max=-100.0)

        # THEN
        result = es_math.sinc(x)

        # EXPECT
        true_min, true_max = dense_range(unnormalized_sinc, -1000.0, -100.0)
        assert result.min <= true_min + 1e-12
        assert result.max >= true_max - 1e-12
        assert abs(result.min) < 0.01
        assert 0 < result.max < 0.01

    @pytest.mark.parametrize(
        ('value', 'lo', 'hi'),
        [(2e20, 1e20, INF), (-2e20, -INF, -1e20), (3e20, 1e20, 5e20)],
    )
    def test_sinc_with_huge_bounds_does_not_raise(self, value, lo, hi):
        # WHEN
        # Regression: int(floor(lo / pi)) overflowed numpy's integer range for lo >~ 5.8e19.
        x = make(value, 'rad', min=lo, max=hi)

        # THEN
        result = es_math.sinc(x)

        # EXPECT
        # The lobes are below 1e-19 out here, so the bounds are essentially [0, 0].
        assert abs(result.min) < 1e-19
        assert abs(result.max) < 1e-19

    def test_sinc_one_sided_infinite_bounds(self):
        # WHEN
        # Lobes decay like 1/x; the extremes beyond x=100 are tiny but present.
        x = make(150.0, 'rad', min=100.0)

        # THEN
        result = es_math.sinc(x)

        # EXPECT
        true_min, true_max = dense_range(unnormalized_sinc, 100.0, 1000.0)
        assert result.min <= true_min + 1e-12
        assert result.max >= true_max - 1e-12
        assert abs(result.min) < 0.01
        assert 0 < result.max < 0.01


# ----------------------------------------------------------------------
# Domain: strict on values, permissive on bounds
# ----------------------------------------------------------------------


class TestValueGuard:
    @pytest.mark.parametrize(
        ('name', 'value', 'unit'),
        [
            ('log', 0.0, ''),
            ('log', -1.0, ''),
            ('log10', 0.0, ''),
            ('log10', -2.0, ''),
            ('sqrt', -1.0, ''),
            ('arcsin', 1.5, ''),
            ('arcsin', -1.5, ''),
            ('arccos', 1.5, ''),
            ('arccos', -1.5, ''),
            ('arccosh', 0.5, ''),
            ('arctanh', 1.0, ''),
            ('arctanh', -1.0, ''),
            ('arctanh', 1.5, ''),
        ],
    )
    def test_value_outside_domain_raises(self, name, value, unit):
        # WHEN THEN
        x = make(value, unit)

        # EXPECT
        with pytest.raises(ValueError, match='outside the domain'):
            fn(name)(x)

    @pytest.mark.parametrize('name', ['exp', 'sinh', 'cosh'])
    def test_overflowing_result_raises(self, name):
        # WHEN THEN
        x = make(1000.0)

        # EXPECT
        with pytest.raises(ValueError, match='not finite'):
            fn(name)(x)

    @pytest.mark.parametrize('name', ['exp', 'sinh', 'cosh'])
    def test_overflowing_variance_raises_although_the_value_is_finite(self, name):
        # WHEN THEN
        # The value is finite at x=400, but the variance scales with exp(2x) and overflows.
        x = make(400.0, variance=1.0)
        with np.errstate(all='ignore'):
            assert np.isfinite(getattr(np, name)(400.0))

        # EXPECT
        with pytest.raises(ValueError, match='not finite'):
            fn(name)(x)

    @pytest.mark.parametrize('name', ['sqrt', 'arcsin', 'arccos', 'arccosh'])
    def test_infinite_variance_at_domain_edge_raises(self, name):
        # WHEN THEN
        # d/dx is infinite at the edge, so a non-zero input variance gives a
        # non-finite output variance.
        edge = {'sqrt': 0.0, 'arcsin': 1.0, 'arccos': 1.0, 'arccosh': 1.0}[name]
        x = make(edge, variance=0.01)

        # EXPECT
        with pytest.raises(ValueError, match='not finite'):
            fn(name)(x)

    @pytest.mark.parametrize(
        ('name', 'value', 'expected'),
        [
            ('sqrt', 0.0, 0.0),
            ('arcsin', 1.0, np.pi / 2),
            ('arcsin', -1.0, -np.pi / 2),
            ('arccos', 1.0, 0.0),
            ('arccos', -1.0, np.pi),
            ('arccosh', 1.0, 0.0),
        ],
    )
    def test_domain_edge_value_is_allowed_without_variance(self, name, value, expected):
        # WHEN
        x = make(value)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert result.value == pytest.approx(expected)

    def test_error_message_names_the_parameter(self):
        # WHEN THEN
        x = make(-1.0, name='my_param')

        # EXPECT
        with pytest.raises(ValueError, match='my_param'):
            es_math.log(x)


class TestPermissiveBounds:
    """Bounds that leave the domain are clipped to it, never an error or NaN."""

    @pytest.fixture(autouse=True)
    def no_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            yield

    def test_log_bounds_below_zero(self):
        # WHEN
        x = make(2.0, min=-1.0, max=5.0)

        # THEN
        result = es_math.log(x)

        # EXPECT
        assert_bounds(result, -INF, np.log(5.0))

    def test_log10_bounds_touching_zero(self):
        # WHEN
        x = make(10.0, min=0.0, max=100.0)

        # THEN
        result = es_math.log10(x)

        # EXPECT
        assert_bounds(result, -INF, 2.0)

    def test_sqrt_bounds_below_zero(self):
        # WHEN
        x = make(4.0, min=-4.0, max=9.0)

        # THEN
        result = es_math.sqrt(x)

        # EXPECT
        assert_bounds(result, 0.0, 3.0)

    def test_arcsin_bounds_beyond_one(self):
        # WHEN
        x = make(0.2, min=-3.0, max=3.0)

        # THEN
        result = es_math.arcsin(x)

        # EXPECT
        assert_bounds(result, -np.pi / 2, np.pi / 2)

    def test_arccos_bounds_beyond_one(self):
        # WHEN
        x = make(0.2, min=-3.0, max=3.0)

        # THEN
        result = es_math.arccos(x)

        # EXPECT
        assert_bounds(result, 0.0, np.pi)

    def test_arccosh_bounds_below_one(self):
        # WHEN
        x = make(3.0, min=-5.0, max=10.0)

        # THEN
        result = es_math.arccosh(x)

        # EXPECT
        assert_bounds(result, 0.0, np.arccosh(10.0))

    def test_arctanh_bounds_beyond_one(self):
        # WHEN
        x = make(0.3, min=-5.0, max=5.0)

        # THEN
        result = es_math.arctanh(x)

        # EXPECT
        assert_bounds(result, -INF, INF)

    @pytest.mark.parametrize('name', ['log', 'log10', 'sqrt', 'arcsin', 'arccos', 'arctanh'])
    def test_bounds_are_never_nan(self, name):
        # WHEN
        x = make(0.5, min=-10.0, max=10.0)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert not np.isnan(result.min)
        assert not np.isnan(result.max)


# ----------------------------------------------------------------------
# tan
# ----------------------------------------------------------------------


class TestTan:
    def test_bounds_without_asymptote(self):
        # WHEN
        angle = make(0.3, 'rad', min=-1.0, max=1.0)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert_bounds(result, np.tan(-1.0), np.tan(1.0))

    def test_bounds_crossing_asymptote_blow_up(self):
        # WHEN
        angle = make(0.3, 'rad', min=0.0, max=2.0)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert_bounds(result, -INF, INF)

    def test_bounds_crossing_asymptote_in_degrees(self):
        # WHEN
        angle = make(30.0, 'deg', min=0.0, max=100.0)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert_bounds(result, -INF, INF)

    def test_bounds_in_degrees_without_asymptote(self):
        # WHEN
        angle = make(30.0, 'deg', min=0.0, max=60.0)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert_bounds(result, 0.0, np.tan(np.pi / 3))

    def test_upper_bound_exactly_on_an_asymptote_is_infinite_on_that_side_only(self):
        # WHEN
        # tan rises from tan(0) = 0 to +inf as the angle approaches 90 deg.
        angle = make(45.0, 'deg', min=0.0, max=90.0)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert_bounds(result, 0.0, INF)

    def test_upper_bound_exactly_on_an_asymptote_in_radians(self):
        # WHEN
        angle = make(0.5, 'rad', min=0.0, max=np.pi / 2)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert_bounds(result, 0.0, INF)

    def test_lower_bound_exactly_on_an_asymptote_is_infinite_on_that_side_only(self):
        # WHEN
        # Just after 90 deg tan is -inf and rises to tan(120 deg) = -sqrt(3).
        angle = make(100.0, 'deg', min=90.0, max=120.0)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert_bounds(result, -INF, -np.sqrt(3.0))

    def test_bounds_on_two_asymptotes_cover_all_real_numbers(self):
        # WHEN
        angle = make(0.0, 'deg', min=-90.0, max=90.0)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert_bounds(result, -INF, INF)

    @pytest.mark.parametrize(
        ('value', 'unit'), [(np.pi / 2, 'rad'), (90.0, 'deg'), (-90.0, 'deg')]
    )
    def test_value_on_asymptote_raises(self, value, unit):
        # WHEN THEN
        angle = make(value, unit)

        # EXPECT
        with pytest.raises(ValueError, match='asymptote'):
            es_math.tan(angle)

    def test_value_within_one_sigma_of_asymptote_raises(self):
        # WHEN THEN
        angle = make(np.pi / 2 - 1e-6, 'rad', variance=1e-4)

        # EXPECT
        with pytest.raises(ValueError, match='asymptote'):
            es_math.tan(angle)

    def test_value_within_one_sigma_of_asymptote_in_degrees_raises(self):
        # WHEN THEN
        angle = make(89.9, 'deg', variance=1.0)

        # EXPECT
        with pytest.raises(ValueError, match='asymptote'):
            es_math.tan(angle)

    @pytest.mark.parametrize('value', [np.pi / 2 - 5e-8, np.pi / 2 + 5e-8])
    def test_value_inside_the_numerical_floor_raises(self, value):
        # WHEN THEN
        # With no variance the margin is the 1e-7 rad floor.
        angle = make(value, 'rad')

        # EXPECT
        with pytest.raises(ValueError, match='asymptote'):
            es_math.tan(angle)

    @pytest.mark.parametrize('value', [np.pi / 2 - 2e-7, np.pi / 2 + 2e-7])
    def test_value_just_outside_the_numerical_floor_is_allowed(self, value):
        # WHEN
        angle = make(value, 'rad')

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert np.isfinite(result.value)

    def test_value_within_one_sigma_of_asymptote_raises_when_sigma_exceeds_the_floor(self):
        # WHEN THEN
        # variance=1e-4 -> sigma = 0.01 rad, far above the 1e-7 rad floor; 0.009 rad away.
        angle = make(np.pi / 2 - 0.009, 'rad', variance=1e-4)

        # EXPECT
        with pytest.raises(ValueError, match='asymptote'):
            es_math.tan(angle)

    def test_value_just_beyond_one_sigma_of_asymptote_is_allowed(self):
        # WHEN
        # sigma = 0.01 rad; the asymptote is 0.011 rad away.
        angle = make(np.pi / 2 - 0.011, 'rad', variance=1e-4)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert np.isfinite(result.variance)

    @pytest.mark.parametrize(
        ('value', 'unit'), [(3 * np.pi / 2, 'rad'), (270.0, 'deg'), (-3 * np.pi / 2, 'rad')]
    )
    def test_asymptotes_other_than_plus_minus_pi_over_two_raise(self, value, unit):
        # WHEN THEN
        angle = make(value, unit)

        # EXPECT
        with pytest.raises(ValueError, match='asymptote'):
            es_math.tan(angle)

    @pytest.mark.parametrize('variance', [0.0, None])
    def test_missing_variance_is_guarded_like_zero_variance(self, variance):
        # WHEN THEN
        angle = make(np.pi / 2, 'rad', variance=variance)

        # EXPECT
        with pytest.raises(ValueError, match='asymptote'):
            es_math.tan(angle)

    def test_value_far_from_asymptote_in_sigma_is_allowed(self):
        # WHEN
        angle = make(1.0, 'rad', variance=1e-4)

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert np.isfinite(result.variance)

    def test_value_near_but_not_on_asymptote_is_allowed(self):
        # WHEN
        angle = make(np.pi / 2 - 1e-3, 'rad')

        # THEN
        result = es_math.tan(angle)

        # EXPECT
        assert result.value == pytest.approx(np.tan(np.pi / 2 - 1e-3))


# ----------------------------------------------------------------------
# arctan2 (binary; Scipp has no variance support for it)
# ----------------------------------------------------------------------


class TestArctan2:
    def test_value_is_the_angle_in_radians_and_unit_is_rad(self):
        # WHEN
        y = make(1.0, 'm', name='y')
        x = make(2.0, 'm', name='x')

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert result.value == pytest.approx(np.arctan2(1.0, 2.0))
        assert result.unit == 'rad'

    @pytest.mark.parametrize(('y', 'x'), [(1.0, 2.0), (1.0, -2.0), (-1.0, -2.0), (-1.0, 2.0)])
    def test_value_is_in_the_correct_quadrant(self, y, x):
        # WHEN
        y_param = make(y)
        x_param = make(x)

        # THEN
        result = es_math.arctan2(y_param, x_param)

        # EXPECT
        assert result.value == pytest.approx(np.arctan2(y, x))

    def test_variance_matches_delta_method(self):
        # WHEN
        y, x, var_y, var_x = 1.0, 2.0, 0.04, 0.09
        y_param = make(y, variance=var_y)
        x_param = make(x, variance=var_x)

        # THEN
        result = es_math.arctan2(y_param, x_param)

        # EXPECT
        expected = (x**2 * var_y + y**2 * var_x) / (x**2 + y**2) ** 2
        assert result.variance == pytest.approx(expected, rel=1e-12)

    def test_variance_with_only_one_argument_uncertain(self):
        # WHEN
        y = make(1.0, variance=0.04)
        x = make(2.0)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert result.variance == pytest.approx(2.0**2 * 0.04 / (1.0 + 4.0) ** 2, rel=1e-12)

    def test_no_variance_when_neither_argument_has_one(self):
        # WHEN
        y = make(1.0)
        x = make(2.0)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert result.variance is None

    def test_variance_at_origin_raises(self):
        # WHEN THEN
        y = make(0.0, variance=0.01)
        x = make(0.0, variance=0.01)

        # EXPECT
        with pytest.raises(ValueError, match='origin'):
            es_math.arctan2(y, x)

    @pytest.mark.parametrize('var_y', [0.0, None])
    @pytest.mark.parametrize('var_x', [0.0, None])
    def test_zero_and_missing_variance_are_equivalent(self, var_y, var_x):
        # WHEN
        y = make(1.0, variance=var_y)
        x = make(2.0, variance=var_x)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert result.variance is None

    def test_variance_with_only_x_uncertain(self):
        # WHEN
        y = make(1.0)
        x = make(2.0, variance=0.09)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert result.variance == pytest.approx(1.0**2 * 0.09 / (4.0 + 1.0) ** 2, rel=1e-12)

    def test_variance_with_one_argument_none_and_the_other_uncertain(self):
        # WHEN
        y = make(1.0, variance=None)
        x = make(2.0, variance=0.09)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert result.variance == pytest.approx(0.09 / 25.0, rel=1e-12)

    def test_only_x_uncertain_and_close_to_origin_raises(self):
        # WHEN THEN
        y = make(0.0)
        x = make(0.01, variance=0.01)

        # EXPECT
        with pytest.raises(ValueError, match='origin'):
            es_math.arctan2(y, x)

    def test_point_exactly_one_sigma_from_origin_raises(self):
        # WHEN THEN
        # sigma = sqrt(0.25) = 0.5 exactly, and the point is exactly 0.5 from the origin.
        y = make(0.0)
        x = make(0.5, variance=0.25)

        # EXPECT
        with pytest.raises(ValueError, match='origin'):
            es_math.arctan2(y, x)

    def test_point_just_outside_one_sigma_from_origin_is_allowed(self):
        # WHEN
        y = make(0.0)
        x = make(0.5 + 1e-6, variance=0.25)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert np.isfinite(result.variance)

    def test_origin_without_variance_is_allowed(self):
        # WHEN
        y = make(0.0)
        x = make(0.0)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert result.value == 0.0

    def test_point_within_one_sigma_of_origin_raises(self):
        # WHEN THEN
        y = make(0.01, variance=0.01)
        x = make(0.0)

        # EXPECT
        with pytest.raises(ValueError, match='origin'):
            es_math.arctan2(y, x)

    def test_point_far_from_origin_in_sigma_is_allowed(self):
        # WHEN
        y = make(1.0, variance=0.01)
        x = make(0.0)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert np.isfinite(result.variance)

    def test_mismatched_units_raise(self):
        # WHEN THEN
        y = make(1.0, 'm')
        x = make(2.0, 's')

        # EXPECT
        with pytest.raises(UnitError):
            es_math.arctan2(y, x)

    @pytest.mark.parametrize(
        ('y_bounds', 'x_bounds', 'y', 'x'),
        [
            ((0.5, 2.0), (1.0, 3.0), 1.0, 2.0),  # first quadrant
            ((-2.0, -0.5), (1.0, 3.0), -1.0, 2.0),  # fourth quadrant
            ((0.5, 2.0), (-3.0, -1.0), 1.0, -2.0),  # second quadrant
            ((-2.0, -0.5), (-3.0, -1.0), -1.0, -2.0),  # third quadrant
            ((0.0, 1.0), (0.0, 2.0), 0.5, 1.0),  # origin on a corner
            ((0.0, 1.0), (-1.0, 2.0), 0.5, 1.0),  # y >= 0, origin on an edge
            ((-1.0, 1.0), (0.0, 2.0), 0.5, 1.0),  # x >= 0, origin on an edge
        ],
    )
    def test_bounds_without_branch_cut_match_brute_force(self, y_bounds, x_bounds, y, x):
        # WHEN
        y_param = make(y, min=y_bounds[0], max=y_bounds[1])
        x_param = make(x, min=x_bounds[0], max=x_bounds[1])

        # THEN
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = es_math.arctan2(y_param, x_param)

        # EXPECT
        ys, xs = np.meshgrid(
            np.linspace(*y_bounds, 801), np.linspace(*x_bounds, 801), indexing='ij'
        )
        angles = np.arctan2(ys, xs)
        assert result.min <= angles.min() + 1e-12
        assert result.max >= angles.max() - 1e-12
        assert result.min >= angles.min() - 1e-2
        assert result.max <= angles.max() + 1e-2

    def test_bounds_crossing_branch_cut_warn_and_span_full_range(self):
        # WHEN
        # y crosses 0 while x < 0: the angle jumps between -pi and +pi.
        y = make(0.3, min=-1.0, max=1.0)
        x = make(-1.0, min=-2.0, max=-0.5)

        # THEN
        with pytest.warns(UserWarning, match='branch cut'):
            result = es_math.arctan2(y, x)

        # EXPECT
        assert_bounds(result, -np.pi, np.pi)

    def test_bounds_containing_origin_span_full_range(self):
        # WHEN
        y = make(0.3, min=-1.0, max=1.0)
        x = make(0.3, min=-1.0, max=1.0)

        # THEN
        with pytest.warns(UserWarning):
            result = es_math.arctan2(y, x)

        # EXPECT
        assert_bounds(result, -np.pi, np.pi)

    def test_one_sided_infinite_bounds_span_full_range_without_warning(self):
        # WHEN
        # y crosses 0 while x extends to -inf: the warning is only for fully finite bounds.
        y = make(0.3, min=-1.0, max=1.0)
        x = make(-2.0, min=-INF, max=-1.0)

        # THEN
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = es_math.arctan2(y, x)

        # EXPECT
        assert_bounds(result, -np.pi, np.pi)

    def test_default_infinite_bounds_span_full_range_without_warning(self):
        # WHEN
        # Unbounded parameters are the default; warning on every call would be noise.
        y = make(1.0)
        x = make(-2.0)

        # THEN
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = es_math.arctan2(y, x)

        # EXPECT
        assert_bounds(result, -np.pi, np.pi)

    def test_value_is_always_inside_bounds(self):
        # WHEN
        y = make(1.0, min=0.5, max=2.0)
        x = make(2.0, min=1.0, max=3.0)

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert result.min <= result.value <= result.max


# ----------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------


class TestUnits:
    @pytest.mark.parametrize('name', ['sin', 'cos', 'tan'])
    @pytest.mark.parametrize('unit', ['m', ''])
    def test_trig_requires_an_angle_unit(self, name, unit):
        # WHEN THEN
        x = make(0.5, unit)

        # EXPECT
        with pytest.raises(UnitError):
            fn(name)(x)

    @pytest.mark.parametrize('name', ['sin', 'cos'])
    def test_trig_result_is_dimensionless(self, name):
        # WHEN
        radians = make(0.5, 'rad')
        degrees = make(30.0, 'deg')

        # THEN
        result_rad = fn(name)(radians)
        result_deg = fn(name)(degrees)

        # EXPECT
        assert result_rad.unit == 'dimensionless'
        assert result_deg.unit == 'dimensionless'

    @pytest.mark.parametrize('name', ['arcsin', 'arccos', 'arctan'])
    def test_inverse_trig_returns_radians(self, name):
        # WHEN
        x = make(0.5)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert result.unit == 'rad'

    @pytest.mark.parametrize('name', ['arcsin', 'arccos', 'arctan'])
    def test_inverse_trig_requires_dimensionless_input(self, name):
        # WHEN THEN
        x = make(0.5, 'rad')

        # EXPECT
        with pytest.raises(UnitError):
            fn(name)(x)

    @pytest.mark.parametrize(
        'name',
        ['exp', 'log', 'log10', 'erf', 'erfc', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh'],
    )
    def test_dimensionless_only_functions_reject_units(self, name):
        # WHEN THEN
        x = make(1.5, 'm')

        # EXPECT
        with pytest.raises(UnitError):
            fn(name)(x)

    def test_arctanh_rejects_units(self):
        # WHEN THEN
        x = make(0.5, 'm')

        # EXPECT
        with pytest.raises(UnitError):
            es_math.arctanh(x)

    def test_sinc_accepts_radians_and_dimensionless(self):
        # WHEN
        radians = make(0.5, 'rad')
        dimensionless = make(0.5, '')

        # THEN
        result_rad = es_math.sinc(radians)
        result_dimensionless = es_math.sinc(dimensionless)

        # EXPECT
        assert result_rad.value == pytest.approx(np.sin(0.5) / 0.5)
        assert result_dimensionless.value == pytest.approx(np.sin(0.5) / 0.5)

    def test_sinc_rejects_degrees(self):
        # WHEN THEN
        angle = make(30.0, 'deg')

        # EXPECT
        with pytest.raises(UnitError):
            es_math.sinc(angle)

    def test_sqrt_of_a_square_unit(self):
        # WHEN
        area = make(4.0, 'm**2', min=1.0, max=9.0)

        # THEN
        result = es_math.sqrt(area)

        # EXPECT
        assert result.unit == 'm'
        assert result.value == pytest.approx(2.0)
        assert_bounds(result, 1.0, 3.0)

    def test_unit_error_wins_over_value_error(self):
        # WHEN THEN
        # Scipp validates units first, so the more fundamental error is reported.
        x = make(-1.0, 'm')

        # EXPECT
        with pytest.raises(UnitError):
            es_math.log(x)


# ----------------------------------------------------------------------
# Input types
# ----------------------------------------------------------------------


class TestInputTypes:
    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_plain_numbers_are_not_supported(self, name):
        # WHEN THEN
        x = 0.5

        # EXPECT
        with pytest.raises(TypeError):
            fn(name)(x)

    def test_scipp_variable_is_not_supported(self):
        # WHEN THEN
        x = sc.scalar(0.5, unit='rad')

        # EXPECT
        with pytest.raises(TypeError):
            es_math.sin(x)

    @pytest.mark.parametrize(('y', 'x'), [(1.0, 2.0), (1.0, make(2.0)), (make(1.0), 2.0)])
    def test_arctan2_requires_descriptor_numbers_or_parameters(self, y, x):
        # WHEN THEN
        # (y, x) contain at least one plain number

        # EXPECT
        with pytest.raises(TypeError):
            es_math.arctan2(y, x)


# ----------------------------------------------------------------------
# Invariants over random inputs
# ----------------------------------------------------------------------

# (value range, unit). Bounds are drawn around the value and may leave the domain.
RANDOM_RANGES = {
    'exp': ((-3.0, 3.0), ''),
    'log': ((0.1, 10.0), ''),
    'log10': ((0.1, 10.0), ''),
    'sqrt': ((0.1, 10.0), ''),
    'erf': ((-2.0, 2.0), ''),
    'erfc': ((-2.0, 2.0), ''),
    'sin': ((-10.0, 10.0), 'rad'),
    'cos': ((-10.0, 10.0), 'rad'),
    'tan': ((-1.4, 1.4), 'rad'),
    'arcsin': ((-0.9, 0.9), ''),
    'arccos': ((-0.9, 0.9), ''),
    'arctan': ((-5.0, 5.0), ''),
    'sinc': ((-10.0, 10.0), 'rad'),
    'sinh': ((-3.0, 3.0), ''),
    'cosh': ((-3.0, 3.0), ''),
    'tanh': ((-3.0, 3.0), ''),
    'arcsinh': ((-3.0, 3.0), ''),
    'arccosh': ((1.1, 10.0), ''),
    'arctanh': ((-0.9, 0.9), ''),
}


# Array-capable references where VALUE_CASES holds a scalar-only one (math.erf).
ARRAY_REFERENCE = {'erf': _erf, 'erfc': _erfc}


class TestInvariants:
    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_value_lies_within_result_bounds_and_bounds_are_not_nan(self, name):
        # WHEN
        (low, high), unit = RANDOM_RANGES[name]
        rng = np.random.default_rng(seed=1234)
        for _ in range(50):
            value = rng.uniform(low, high)
            lo = value - rng.uniform(0.01, 3.0)
            hi = value + rng.uniform(0.01, 3.0)
            x = make(value, unit, min=lo, max=hi, variance=0.001)

            # THEN
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                result = fn(name)(x)

            # EXPECT
            assert not np.isnan(result.min) and not np.isnan(result.max)
            assert result.min <= result.value <= result.max

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_random_bounds_match_brute_force_range(self, name):
        # WHEN
        # Unlike the invariant above (which _apply enforces by widening the bounds), this
        # compares against an independent dense evaluation, so a wrong strategy is caught.
        (low, high), unit = RANDOM_RANGES[name]
        reference = ARRAY_REFERENCE.get(name, VALUE_CASES[name][0])
        rng = np.random.default_rng(seed=2024)
        for _ in range(25):
            value = rng.uniform(low, high)
            lo, hi = rng.uniform(low, value), rng.uniform(value, high)
            if hi - lo < 0.05:
                continue
            x = make(value, unit, min=lo, max=hi)

            # THEN
            result = fn(name)(x)

            # EXPECT
            true_min, true_max = dense_range(reference, lo, hi, n=20_001)
            assert result.min <= true_min + 1e-9
            assert result.max >= true_max - 1e-9
            assert result.min >= true_min - 1e-3
            assert result.max <= true_max + 1e-3

    def test_arctan2_value_lies_within_result_bounds(self):
        # WHEN
        rng = np.random.default_rng(seed=99)
        for _ in range(50):
            y, x = rng.uniform(-3.0, 3.0, size=2)
            y_lo, y_hi = y - rng.uniform(0.01, 2.0), y + rng.uniform(0.01, 2.0)
            x_lo, x_hi = x - rng.uniform(0.01, 2.0), x + rng.uniform(0.01, 2.0)
            y_param = make(y, min=y_lo, max=y_hi)
            x_param = make(x, min=x_lo, max=x_hi)

            # THEN
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                result = es_math.arctan2(y_param, x_param)

            # EXPECT
            assert result.min <= result.value <= result.max


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


class TestPublicApi:
    @pytest.mark.parametrize('name', ALL_FUNCTIONS)
    def test_function_is_exported_from_easyscience_variable(self, name):
        # WHEN THEN
        exported = getattr(es_math, name)

        # EXPECT
        assert callable(exported)
        assert exported is getattr(es_math, name)

    @pytest.mark.parametrize('name', ALL_FUNCTIONS)
    def test_function_has_a_docstring(self, name):
        # WHEN
        function = fn(name)

        # THEN
        docstring = function.__doc__

        # EXPECT
        assert docstring and docstring.strip()

    def test_math_all_contains_only_strings(self):
        # WHEN
        exported_names = es_math.__all__

        # THEN
        all_are_strings = all(isinstance(entry, str) for entry in exported_names)

        # EXPECT
        assert all_are_strings
        for name in ALL_FUNCTIONS:
            assert name in exported_names


# ----------------------------------------------------------------------
# Output unit of the inverse trigonometric functions
# ----------------------------------------------------------------------

INVERSE_TRIG = ['arcsin', 'arccos', 'arctan']


class TestInverseOutputUnit:
    @pytest.mark.parametrize('name', INVERSE_TRIG)
    def test_default_unit_is_radians(self, name):
        # WHEN
        x = make(0.4, min=0.2, max=0.8)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert result.unit == 'rad'

    @pytest.mark.parametrize('name', INVERSE_TRIG)
    def test_explicit_radians_equals_the_default(self, name):
        # WHEN
        x = make(0.4, min=0.2, max=0.8, variance=0.04)

        # THEN
        default = fn(name)(x)
        explicit = fn(name)(x, unit='rad')

        # EXPECT
        assert explicit.unit == 'rad'
        assert (explicit.value, explicit.variance, explicit.min, explicit.max) == (
            default.value,
            default.variance,
            default.min,
            default.max,
        )

    @pytest.mark.parametrize('name', INVERSE_TRIG)
    def test_degrees_convert_value_and_bounds(self, name):
        # WHEN
        x = make(0.4, min=0.2, max=0.8)
        in_radians = fn(name)(x)

        # THEN
        in_degrees = fn(name)(x, unit='deg')

        # EXPECT
        assert in_degrees.unit == 'deg'
        assert in_degrees.value == pytest.approx(np.degrees(in_radians.value), rel=1e-12)
        assert in_degrees.min == pytest.approx(np.degrees(in_radians.min), rel=1e-12)
        assert in_degrees.max == pytest.approx(np.degrees(in_radians.max), rel=1e-12)

    @pytest.mark.parametrize('name', INVERSE_TRIG)
    def test_degrees_convert_variance_with_the_squared_factor(self, name):
        # WHEN
        _, derivative, value, _ = VALUE_CASES[name]
        variance = 0.04
        x = make(value, variance=variance)

        # THEN
        result = fn(name)(x, unit='deg')

        # EXPECT
        variance_in_rad2 = float(derivative(value)) ** 2 * variance
        assert result.variance == pytest.approx(variance_in_rad2 * np.degrees(1.0) ** 2, rel=1e-9)

    def test_arcsin_example_in_degrees(self):
        # WHEN
        x = make(0.5, min=0.0, max=1.0)

        # THEN
        result = es_math.arcsin(x, unit='deg')

        # EXPECT
        assert result.value == pytest.approx(30.0, abs=1e-9)
        assert_bounds(result, 0.0, 90.0)

    def test_arctan2_in_degrees(self):
        # WHEN
        y = make(1.0, 'm', min=0.5, max=2.0, name='y')
        x = make(1.0, 'm', min=1.0, max=3.0, name='x')

        # THEN
        result = es_math.arctan2(y, x, unit='deg')

        # EXPECT
        assert result.unit == 'deg'
        assert result.value == pytest.approx(45.0, abs=1e-9)
        assert result.min == pytest.approx(np.degrees(np.arctan2(0.5, 3.0)), rel=1e-12)
        assert result.max == pytest.approx(np.degrees(np.arctan2(2.0, 1.0)), rel=1e-12)

    def test_arctan2_variance_in_degrees(self):
        # WHEN
        y, x, var_y, var_x = 1.0, 2.0, 0.04, 0.09
        y_param = make(y, 'm', variance=var_y)
        x_param = make(x, 'm', variance=var_x)

        # THEN
        result = es_math.arctan2(y_param, x_param, unit='deg')

        # EXPECT
        variance_in_rad2 = (x**2 * var_y + y**2 * var_x) / (x**2 + y**2) ** 2
        assert result.variance == pytest.approx(variance_in_rad2 * np.degrees(1.0) ** 2, rel=1e-9)

    @pytest.mark.parametrize('name', INVERSE_TRIG)
    def test_a_unit_that_is_not_an_angle_raises(self, name):
        # WHEN THEN
        x = make(0.4)

        # EXPECT
        with pytest.raises(UnitError):
            fn(name)(x, unit='m')

    def test_arctan2_with_a_unit_that_is_not_an_angle_raises(self):
        # WHEN THEN
        y = make(1.0, 'm')
        x = make(2.0, 'm')

        # EXPECT
        with pytest.raises(UnitError):
            es_math.arctan2(y, x, unit='m')

    @pytest.mark.parametrize('name', INVERSE_TRIG)
    def test_unit_is_keyword_only(self, name):
        # WHEN THEN
        x = make(0.4)

        # EXPECT
        with pytest.raises(TypeError):
            fn(name)(x, 'deg')


# ----------------------------------------------------------------------
# Aliases
# ----------------------------------------------------------------------

ALIASES = {
    'asin': 'arcsin',
    'acos': 'arccos',
    'atan': 'arctan',
    'atan2': 'arctan2',
    'asinh': 'arcsinh',
    'acosh': 'arccosh',
    'atanh': 'arctanh',
}


class TestAliases:
    @pytest.mark.parametrize(('alias', 'canonical'), ALIASES.items())
    def test_alias_is_the_same_function_object(self, alias, canonical):
        # WHEN THEN
        alias_function = getattr(es_math, alias)

        # EXPECT
        assert alias_function is getattr(es_math, canonical)

    @pytest.mark.parametrize(('alias', 'canonical'), ALIASES.items())
    def test_alias_is_exported_from_easyscience_math(self, alias, canonical):
        # WHEN THEN
        exported = getattr(es_math, alias)

        # EXPECT
        assert exported is getattr(es_math, canonical)
        assert alias in es_math.__all__

    def test_math_names_are_not_exported_from_easyscience_variable(self):
        # WHEN
        # THEN
        exported = set(es_variable.__all__)

        # EXPECT
        assert exported.isdisjoint(ALL_FUNCTIONS)
        assert exported.isdisjoint(ALIASES)

    def test_result_is_named_after_the_canonical_function(self):
        # WHEN
        x = make(0.4, name='x')

        # THEN
        result = es_math.asin(x)

        # EXPECT
        assert result.name == 'arcsin(x)'

    def test_alias_accepts_the_unit_keyword(self):
        # WHEN
        x = make(0.5, min=0.0, max=1.0)

        # THEN
        result = es_math.asin(x, unit='deg')

        # EXPECT
        assert result.unit == 'deg'
        assert result.value == pytest.approx(30.0, abs=1e-9)

    def test_atan2_alias_keeps_the_y_then_x_argument_order(self):
        # WHEN
        y = make(1.0, name='y')
        x = make(-1.0, name='x')

        # THEN
        result = es_math.atan2(y, x)

        # EXPECT
        assert result.value == pytest.approx(3 * np.pi / 4)


# ----------------------------------------------------------------------
# Inheritance: Parameter and DescriptorNumber
# ----------------------------------------------------------------------


class MyParameter(Parameter):
    """A user subclass whose constructor needs an extra argument."""

    def __init__(self, name, value, tag, **kwargs):
        super().__init__(name, value, **kwargs)
        self.tag = tag


class MyDescriptor(DescriptorNumber):
    """A user subclass whose constructor needs an extra argument."""

    def __init__(self, name, value, tag, **kwargs):
        super().__init__(name, value, **kwargs)
        self.tag = tag


def descriptor(value, unit='', *, variance=0.0, name='x') -> DescriptorNumber:
    """Shorthand for a plain ``DescriptorNumber`` (no bounds)."""
    return DescriptorNumber(name=name, value=value, unit=unit, variance=variance)


class TestInheritance:
    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_descriptor_number_input_returns_a_descriptor_number(self, name):
        # WHEN
        _, _, value, unit = VALUE_CASES[name]
        plain = descriptor(value, unit, variance=0.04)
        parameter = make(value, unit, variance=0.04)

        # THEN
        result = fn(name)(plain)
        reference = fn(name)(parameter)

        # EXPECT
        assert type(result) is DescriptorNumber
        assert not hasattr(result, 'min')
        assert result.value == pytest.approx(reference.value, rel=1e-12)
        assert result.variance == pytest.approx(reference.variance, rel=1e-12)

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_parameter_input_still_returns_a_parameter(self, name):
        # WHEN
        _, _, value, unit = VALUE_CASES[name]
        x = make(value, unit)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert type(result) is Parameter

    def test_mixed_arctan2_returns_a_parameter_with_point_bounds(self):
        # WHEN
        y = make(1.0, min=0.5, max=2.0, name='y')
        x = descriptor(2.0, name='x')
        point_y = descriptor(1.0, name='y')
        range_x = make(2.0, min=1.0, max=3.0, name='x')

        # THEN
        result = es_math.arctan2(y, x)
        swapped = es_math.arctan2(point_y, range_x)

        # EXPECT
        assert type(result) is Parameter
        assert_bounds(result, np.arctan2(0.5, 2.0), np.arctan2(2.0, 2.0))
        assert type(swapped) is Parameter
        assert_bounds(swapped, np.arctan2(1.0, 3.0), np.arctan2(1.0, 1.0))

    def test_arctan2_of_two_descriptor_numbers_returns_a_descriptor_number(self):
        # WHEN
        y = descriptor(1.0, name='y')
        x = descriptor(1.0, name='x')

        # THEN
        result = es_math.arctan2(y, x)

        # EXPECT
        assert type(result) is DescriptorNumber
        assert result.value == pytest.approx(np.pi / 4)

    def test_user_subclasses_are_accepted_and_return_base_classes(self):
        # WHEN
        sub_parameter = MyParameter('p', 0.5, 'tag', unit='rad', min=0.0, max=1.0)
        sub_descriptor = MyDescriptor('d', 0.5, 'tag', unit='rad')

        # THEN
        from_parameter = es_math.sin(sub_parameter)
        from_descriptor = es_math.sin(sub_descriptor)

        # EXPECT
        assert type(from_parameter) is Parameter
        assert type(from_descriptor) is DescriptorNumber
        assert from_parameter.value == pytest.approx(np.sin(0.5))
        assert from_descriptor.value == pytest.approx(np.sin(0.5))

    def test_unit_keyword_on_a_descriptor_number(self):
        # WHEN
        x = descriptor(0.5, name='x')

        # THEN
        result = es_math.arcsin(x, unit='deg')

        # EXPECT
        assert type(result) is DescriptorNumber
        assert result.unit == 'deg'
        assert result.value == pytest.approx(30.0, abs=1e-9)

    def test_variance_in_degrees_for_a_descriptor_number(self):
        # WHEN
        angle = descriptor(30.0, 'deg', variance=1.0)

        # THEN
        result = es_math.sin(angle)

        # EXPECT
        expected = np.cos(np.deg2rad(30.0)) ** 2 * np.deg2rad(1.0) ** 2
        assert result.variance == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize(
        ('name', 'value', 'unit', 'message'),
        [
            ('tan', np.pi / 2, 'rad', 'asymptote'),
            ('log', -1.0, '', 'outside the domain'),
            ('exp', 1000.0, '', 'not finite'),
        ],
    )
    def test_guards_apply_to_a_descriptor_number(self, name, value, unit, message):
        # WHEN THEN
        x = descriptor(value, unit)

        # EXPECT
        with pytest.raises(ValueError, match=message):
            fn(name)(x)

    @pytest.mark.parametrize(
        'unsupported',
        [
            0.5,
            np.array([0.5]),
            sc.scalar(0.5, unit='rad'),
            DescriptorArray(name='arr', value=[0.5], unit='rad', dimensions=['x']),
        ],
        ids=['float', 'ndarray', 'scipp-variable', 'descriptor-array'],
    )
    def test_other_types_still_raise_type_error(self, unsupported):
        # WHEN THEN
        argument = unsupported

        # EXPECT
        with pytest.raises(TypeError, match='DescriptorNumber'):
            es_math.sin(argument)


# ----------------------------------------------------------------------
# The es.math namespace
# ----------------------------------------------------------------------


class TestEsMathNamespace:
    def test_es_math_is_the_easyscience_math_module(self):
        # WHEN THEN
        namespace = es.math

        # EXPECT
        assert namespace.__name__ == 'easyscience.math'

    def test_es_math_is_not_the_standard_library_math(self):
        # WHEN
        import math as stdlib_math

        # THEN
        namespace = es.math

        # EXPECT
        assert stdlib_math is not namespace
        assert stdlib_math.sin(0.0) == 0.0

    def test_es_math_all_equals_the_implementation_all(self):
        # WHEN THEN
        public_names = es.math.__all__

        # EXPECT
        assert public_names == es_math.__all__
        assert len(public_names) == 27
        assert all(isinstance(name, str) for name in public_names)

    @pytest.mark.parametrize('name', es_math.__all__)
    def test_es_math_exposes_the_same_function_objects(self, name):
        # WHEN THEN
        exposed = getattr(es.math, name)

        # EXPECT
        assert exposed is getattr(es_math, name)

    def test_es_math_exposes_nothing_beyond_the_public_names(self):
        # WHEN THEN
        # Names imported into the module (helpers, typing) are not part of the namespace.
        defined_here = {
            name
            for name, value in vars(es.math).items()
            if not name.startswith('_') and getattr(value, '__module__', None) == es.math.__name__
        }

        # EXPECT
        assert defined_here == set(es_math.__all__)

    def test_math_is_not_exported_by_a_star_import_of_the_package(self):
        # WHEN THEN
        top_level_all = es.__all__

        # EXPECT
        # `from easyscience import *` would otherwise replace the standard-library `math`.
        assert 'math' not in top_level_all

    def test_from_import_works(self):
        # WHEN THEN
        module = importlib.import_module('easyscience.math')

        # EXPECT
        assert module.sin is es_math.sin
        assert module.asin is es_math.arcsin


# ----------------------------------------------------------------------
# Collapsed result bounds
# ----------------------------------------------------------------------

COLLAPSED_BOUNDS_CASES = [
    # (function, value, unit, min, max, why the exact result bounds collapse)
    ('erf', 15.0, '', 10.0, 20.0, 'erf(10) == erf(20) == 1.0'),
    ('erf', 5.5, '', 5.0, 6.0, 'erf(5) and erf(6) differ by 1.5e-12 (< rtol 1e-9)'),
    ('tanh', 20.0, '', 15.0, 25.0, 'tanh saturates at 1.0'),
    ('exp', -900.0, '', -1000.0, -800.0, 'both bounds underflow to 0.0'),
    ('erfc', 35.0, '', 30.0, 40.0, 'both bounds underflow to 0.0'),
    ('cos', 0.0, 'rad', -1e-5, 1e-5, 'cos is quadratic at its maximum: [1 - 5e-11, 1]'),
    ('cosh', 0.0, '', -1e-5, 1e-5, 'cosh is quadratic at its minimum: [1, 1 + 5e-11]'),
    ('sin', np.pi / 2, 'rad', np.pi / 2 - 1e-5, np.pi / 2 + 1e-5, 'quadratic at the maximum'),
    ('arctan', 1e11, '', 1e10, 1e12, 'arctan saturates at pi/2'),
]
COLLAPSED_IDS = [f'{case[0]}[{case[3]:g},{case[4]:g}]' for case in COLLAPSED_BOUNDS_CASES]


class TestCollapsedBounds:
    """
    ``Parameter`` rejects ``min`` and ``max`` that are equal or within ``rtol=1e-9``. Valid
    inputs whose function saturates, underflows or is quadratic at an extremum would make the
    exact result bounds collapse, so such bounds are padded around the value (by twice that
    relative tolerance, or by an absolute 1e-12 if the value is 0).
    """

    @pytest.mark.parametrize(
        ('name', 'value', 'unit', 'lo', 'hi', 'why'), COLLAPSED_BOUNDS_CASES, ids=COLLAPSED_IDS
    )
    def test_collapsed_bounds_do_not_raise_and_contain_the_value(
        self, name, value, unit, lo, hi, why
    ):
        # WHEN
        x = make(value, unit, min=lo, max=hi)

        # THEN
        result = fn(name)(x)

        # EXPECT
        assert not np.isnan(result.min) and not np.isnan(result.max)
        assert result.min <= result.value <= result.max
        assert result.min < result.max

    @pytest.mark.parametrize(
        ('name', 'value', 'unit', 'lo', 'hi', 'why'), COLLAPSED_BOUNDS_CASES, ids=COLLAPSED_IDS
    )
    def test_padded_bounds_contain_the_true_range(self, name, value, unit, lo, hi, why):
        # WHEN
        x = make(value, unit, min=lo, max=hi)
        reference = ARRAY_REFERENCE.get(name, VALUE_CASES[name][0])

        # THEN
        result = fn(name)(x)

        # EXPECT
        true_min, true_max = dense_range(reference, lo, hi, n=20_001)
        assert result.min <= true_min
        assert result.max >= true_max

    @pytest.mark.parametrize(
        ('name', 'value', 'unit', 'lo', 'hi', 'why'), COLLAPSED_BOUNDS_CASES, ids=COLLAPSED_IDS
    )
    def test_padding_is_tiny(self, name, value, unit, lo, hi, why):
        # WHEN
        x = make(value, unit, min=lo, max=hi)

        # THEN
        result = fn(name)(x)

        # EXPECT
        # Twice the 1e-9 relative tolerance on each side, i.e. a width of about 4e-9 * |value|
        # (an absolute 2e-12 around a value of 0).
        assert result.max - result.min <= 1e-8 * max(abs(result.value), 1e-3)

    def test_zero_value_gets_an_absolute_pad(self):
        # WHEN
        # exp(-900) underflows to exactly 0.0, so a relative pad would again be zero.
        x = make(-900.0, min=-1000.0, max=-800.0)

        # THEN
        result = es_math.exp(x)

        # EXPECT
        assert result.value == 0.0
        assert_bounds(result, -1e-12, 1e-12, tol=1e-15)

    def test_bounds_that_do_not_collapse_are_left_exact(self):
        # WHEN
        x = make(0.5, min=0.0, max=1.0)

        # THEN
        result = es_math.erf(x)

        # EXPECT
        assert_bounds(result, 0.0, math.erf(1.0), tol=1e-15)
