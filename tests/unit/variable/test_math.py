# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for ``easyscience.variable.math``.

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

import math
import warnings

import numpy as np
import pytest
import scipp as sc
from scipp import UnitError

import easyscience.variable as es_variable
from easyscience import DescriptorNumber
from easyscience import Parameter
from easyscience import global_object
from easyscience.variable import math as es_math

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
        reference, _, value, unit = VALUE_CASES[name]
        result = fn(name)(make(value, unit))
        assert isinstance(result, Parameter)
        assert result.value == pytest.approx(float(reference(value)), rel=1e-12)

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_variance_matches_delta_method(self, name):
        # sigma_y^2 = f'(x)^2 * sigma_x^2. This also covers erf/erfc, for which
        # Scipp itself has no variance support and EasyScience must supply it.
        _, derivative, value, unit = VALUE_CASES[name]
        variance = 0.04
        result = fn(name)(make(value, unit, variance=variance))
        expected = float(derivative(value)) ** 2 * variance
        assert result.variance == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    @pytest.mark.parametrize('variance', [0.0, None])
    def test_no_input_variance_gives_no_output_variance(self, name, variance):
        _, _, value, unit = VALUE_CASES[name]
        result = fn(name)(make(value, unit, variance=variance))
        assert not result.variance

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_inputs_are_not_modified(self, name):
        _, _, value, unit = VALUE_CASES[name]
        x = make(value, unit, variance=0.04, min=value - 0.1, max=value + 0.1)
        before = (x.value, x.variance, x.min, x.max, x.unit)
        fn(name)(x)
        assert (x.value, x.variance, x.min, x.max, x.unit) == before

    def test_result_name_is_built_from_function_and_argument(self):
        angle = make(0.5, 'rad', name='angle')
        assert es_math.sin(angle).name == 'sin(angle)'
        assert es_math.sinc(angle).name == 'sinc(angle)'
        assert es_math.arctan2(make(1.0, name='y'), make(2.0, name='x')).name == 'arctan2(y, x)'


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
        result = fn(name)(make(value, unit, min=lo, max=hi))
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
        # A freshly created Parameter has bounds (-inf, inf); that must just work.
        result = fn(name)(make(value, unit))
        assert_bounds(result, expected_min, expected_max)

    def test_sin_with_degrees_bounds_are_computed_in_radians(self):
        # Regression: numpy used to be fed 0..90 as if they were radians.
        result = es_math.sin(make(45.0, 'deg', min=0.0, max=90.0))
        assert result.value == pytest.approx(np.sqrt(0.5))
        assert_bounds(result, 0.0, 1.0)

    def test_cos_with_degrees(self):
        result = es_math.cos(make(30.0, 'deg', min=0.0, max=90.0))
        assert_bounds(result, 0.0, 1.0)

    def test_sin_with_degrees_and_infinite_bounds(self):
        assert_bounds(es_math.sin(make(45.0, 'deg')), -1.0, 1.0)

    def test_inverse_trig_bounds_are_in_radians(self):
        result = es_math.arcsin(make(0.5, min=0.0, max=1.0))
        assert result.unit == 'rad'
        assert_bounds(result, 0.0, np.pi / 2)

    def test_sinc_global_minimum_is_found(self):
        result = es_math.sinc(make(1.0, 'rad', min=-20.0, max=20.0))
        assert_bounds(result, SINC_GLOBAL_MIN, 1.0)

    def test_sinc_one_sided_infinite_bounds(self):
        # Lobes decay like 1/x; the extremes beyond x=100 are tiny but present.
        result = es_math.sinc(make(150.0, 'rad', min=100.0))
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
        with pytest.raises(ValueError):
            fn(name)(make(value, unit))

    @pytest.mark.parametrize('name', ['exp', 'sinh', 'cosh'])
    def test_overflowing_result_raises(self, name):
        with pytest.raises(ValueError):
            fn(name)(make(1000.0))

    @pytest.mark.parametrize('name', ['sqrt', 'arcsin', 'arccos', 'arccosh'])
    def test_infinite_variance_at_domain_edge_raises(self, name):
        # d/dx is infinite at the edge, so a non-zero input variance gives a
        # non-finite output variance.
        edge = {'sqrt': 0.0, 'arcsin': 1.0, 'arccos': 1.0, 'arccosh': 1.0}[name]
        with pytest.raises(ValueError):
            fn(name)(make(edge, variance=0.01))

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
        result = fn(name)(make(value))
        assert result.value == pytest.approx(expected)

    def test_error_message_names_the_parameter(self):
        with pytest.raises(ValueError, match='my_param'):
            es_math.log(make(-1.0, name='my_param'))


class TestPermissiveBounds:
    """Bounds that leave the domain are clipped to it, never an error or NaN."""

    @pytest.fixture(autouse=True)
    def no_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            yield

    def test_log_bounds_below_zero(self):
        result = es_math.log(make(2.0, min=-1.0, max=5.0))
        assert_bounds(result, -INF, np.log(5.0))

    def test_log10_bounds_touching_zero(self):
        result = es_math.log10(make(10.0, min=0.0, max=100.0))
        assert_bounds(result, -INF, 2.0)

    def test_sqrt_bounds_below_zero(self):
        result = es_math.sqrt(make(4.0, min=-4.0, max=9.0))
        assert_bounds(result, 0.0, 3.0)

    def test_arcsin_bounds_beyond_one(self):
        result = es_math.arcsin(make(0.2, min=-3.0, max=3.0))
        assert_bounds(result, -np.pi / 2, np.pi / 2)

    def test_arccos_bounds_beyond_one(self):
        result = es_math.arccos(make(0.2, min=-3.0, max=3.0))
        assert_bounds(result, 0.0, np.pi)

    def test_arccosh_bounds_below_one(self):
        result = es_math.arccosh(make(3.0, min=-5.0, max=10.0))
        assert_bounds(result, 0.0, np.arccosh(10.0))

    def test_arctanh_bounds_beyond_one(self):
        result = es_math.arctanh(make(0.3, min=-5.0, max=5.0))
        assert_bounds(result, -INF, INF)

    @pytest.mark.parametrize('name', ['log', 'log10', 'sqrt', 'arcsin', 'arccos', 'arctanh'])
    def test_bounds_are_never_nan(self, name):
        result = fn(name)(make(0.5, min=-10.0, max=10.0))
        assert not np.isnan(result.min)
        assert not np.isnan(result.max)


# ----------------------------------------------------------------------
# tan
# ----------------------------------------------------------------------


class TestTan:
    def test_bounds_without_asymptote(self):
        result = es_math.tan(make(0.3, 'rad', min=-1.0, max=1.0))
        assert_bounds(result, np.tan(-1.0), np.tan(1.0))

    def test_bounds_crossing_asymptote_blow_up(self):
        result = es_math.tan(make(0.3, 'rad', min=0.0, max=2.0))
        assert_bounds(result, -INF, INF)

    def test_bounds_crossing_asymptote_in_degrees(self):
        result = es_math.tan(make(30.0, 'deg', min=0.0, max=100.0))
        assert_bounds(result, -INF, INF)

    def test_bounds_in_degrees_without_asymptote(self):
        result = es_math.tan(make(30.0, 'deg', min=0.0, max=60.0))
        assert_bounds(result, 0.0, np.tan(np.pi / 3))

    @pytest.mark.parametrize(
        ('value', 'unit'), [(np.pi / 2, 'rad'), (90.0, 'deg'), (-90.0, 'deg')]
    )
    def test_value_on_asymptote_raises(self, value, unit):
        with pytest.raises(ValueError):
            es_math.tan(make(value, unit))

    def test_value_within_one_sigma_of_asymptote_raises(self):
        with pytest.raises(ValueError):
            es_math.tan(make(np.pi / 2 - 1e-6, 'rad', variance=1e-4))

    def test_value_within_one_sigma_of_asymptote_in_degrees_raises(self):
        with pytest.raises(ValueError):
            es_math.tan(make(89.9, 'deg', variance=1.0))

    def test_value_far_from_asymptote_in_sigma_is_allowed(self):
        result = es_math.tan(make(1.0, 'rad', variance=1e-4))
        assert np.isfinite(result.variance)

    def test_value_near_but_not_on_asymptote_is_allowed(self):
        result = es_math.tan(make(np.pi / 2 - 1e-3, 'rad'))
        assert result.value == pytest.approx(np.tan(np.pi / 2 - 1e-3))


# ----------------------------------------------------------------------
# arctan2 (binary; Scipp has no variance support for it)
# ----------------------------------------------------------------------


class TestArctan2:
    def test_value_and_unit(self):
        result = es_math.arctan2(make(1.0, 'm', name='y'), make(2.0, 'm', name='x'))
        assert result.value == pytest.approx(np.arctan2(1.0, 2.0))
        assert result.unit == 'rad'

    @pytest.mark.parametrize(('y', 'x'), [(1.0, 2.0), (1.0, -2.0), (-1.0, -2.0), (-1.0, 2.0)])
    def test_correct_quadrant(self, y, x):
        result = es_math.arctan2(make(y), make(x))
        assert result.value == pytest.approx(np.arctan2(y, x))

    def test_variance_matches_delta_method(self):
        y, x, var_y, var_x = 1.0, 2.0, 0.04, 0.09
        result = es_math.arctan2(make(y, variance=var_y), make(x, variance=var_x))
        expected = (x**2 * var_y + y**2 * var_x) / (x**2 + y**2) ** 2
        assert result.variance == pytest.approx(expected, rel=1e-12)

    def test_variance_with_only_one_argument_uncertain(self):
        result = es_math.arctan2(make(1.0, variance=0.04), make(2.0))
        assert result.variance == pytest.approx(2.0**2 * 0.04 / (1.0 + 4.0) ** 2, rel=1e-12)

    def test_no_variance_when_neither_argument_has_one(self):
        assert not es_math.arctan2(make(1.0), make(2.0)).variance

    def test_variance_at_origin_raises(self):
        with pytest.raises(ValueError):
            es_math.arctan2(make(0.0, variance=0.01), make(0.0, variance=0.01))

    def test_origin_without_variance_is_allowed(self):
        assert es_math.arctan2(make(0.0), make(0.0)).value == 0.0

    def test_point_within_one_sigma_of_origin_raises(self):
        with pytest.raises(ValueError, match='origin'):
            es_math.arctan2(make(0.01, variance=0.01), make(0.0))

    def test_point_far_from_origin_in_sigma_is_allowed(self):
        assert np.isfinite(es_math.arctan2(make(1.0, variance=0.01), make(0.0)).variance)

    def test_mismatched_units_raise(self):
        with pytest.raises(UnitError):
            es_math.arctan2(make(1.0, 'm'), make(2.0, 's'))

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
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = es_math.arctan2(
                make(y, min=y_bounds[0], max=y_bounds[1]),
                make(x, min=x_bounds[0], max=x_bounds[1]),
            )
        ys, xs = np.meshgrid(
            np.linspace(*y_bounds, 801), np.linspace(*x_bounds, 801), indexing='ij'
        )
        angles = np.arctan2(ys, xs)
        assert result.min <= angles.min() + 1e-12
        assert result.max >= angles.max() - 1e-12
        assert result.min >= angles.min() - 1e-2
        assert result.max <= angles.max() + 1e-2

    def test_bounds_crossing_branch_cut_warn_and_span_full_range(self):
        # y crosses 0 while x < 0: the angle jumps between -pi and +pi.
        with pytest.warns(UserWarning, match='branch cut'):
            result = es_math.arctan2(
                make(0.3, min=-1.0, max=1.0),
                make(-1.0, min=-2.0, max=-0.5),
            )
        assert_bounds(result, -np.pi, np.pi)

    def test_bounds_containing_origin_span_full_range(self):
        with pytest.warns(UserWarning):
            result = es_math.arctan2(
                make(0.3, min=-1.0, max=1.0),
                make(0.3, min=-1.0, max=1.0),
            )
        assert_bounds(result, -np.pi, np.pi)

    def test_default_infinite_bounds_span_full_range_without_warning(self):
        # Unbounded parameters are the default; warning on every call would be noise.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = es_math.arctan2(make(1.0), make(-2.0))
        assert_bounds(result, -np.pi, np.pi)

    def test_value_is_always_inside_bounds(self):
        result = es_math.arctan2(make(1.0, min=0.5, max=2.0), make(2.0, min=1.0, max=3.0))
        assert result.min <= result.value <= result.max


# ----------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------


class TestUnits:
    @pytest.mark.parametrize('name', ['sin', 'cos', 'tan'])
    @pytest.mark.parametrize('unit', ['m', ''])
    def test_trig_requires_an_angle_unit(self, name, unit):
        with pytest.raises(UnitError):
            fn(name)(make(0.5, unit))

    @pytest.mark.parametrize('name', ['sin', 'cos'])
    def test_trig_result_is_dimensionless(self, name):
        assert fn(name)(make(0.5, 'rad')).unit == 'dimensionless'
        assert fn(name)(make(30.0, 'deg')).unit == 'dimensionless'

    @pytest.mark.parametrize('name', ['arcsin', 'arccos', 'arctan'])
    def test_inverse_trig_returns_radians(self, name):
        assert fn(name)(make(0.5)).unit == 'rad'

    @pytest.mark.parametrize('name', ['arcsin', 'arccos', 'arctan'])
    def test_inverse_trig_requires_dimensionless_input(self, name):
        with pytest.raises(UnitError):
            fn(name)(make(0.5, 'rad'))

    @pytest.mark.parametrize(
        'name',
        ['exp', 'log', 'log10', 'erf', 'erfc', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh'],
    )
    def test_dimensionless_only_functions_reject_units(self, name):
        with pytest.raises(UnitError):
            fn(name)(make(1.5, 'm'))

    def test_arctanh_rejects_units(self):
        with pytest.raises(UnitError):
            es_math.arctanh(make(0.5, 'm'))

    def test_sinc_accepts_radians_and_dimensionless(self):
        assert es_math.sinc(make(0.5, 'rad')).value == pytest.approx(np.sin(0.5) / 0.5)
        assert es_math.sinc(make(0.5, '')).value == pytest.approx(np.sin(0.5) / 0.5)

    def test_sinc_rejects_degrees(self):
        with pytest.raises(UnitError):
            es_math.sinc(make(30.0, 'deg'))

    def test_sqrt_of_a_square_unit(self):
        result = es_math.sqrt(make(4.0, 'm**2', min=1.0, max=9.0))
        assert result.unit == 'm'
        assert result.value == pytest.approx(2.0)
        assert_bounds(result, 1.0, 3.0)

    def test_unit_error_wins_over_value_error(self):
        # Scipp validates units first, so the more fundamental error is reported.
        with pytest.raises(UnitError):
            es_math.log(make(-1.0, 'm'))


# ----------------------------------------------------------------------
# Input types
# ----------------------------------------------------------------------


class TestInputTypes:
    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_plain_numbers_are_not_supported(self, name):
        with pytest.raises(NotImplementedError):
            fn(name)(0.5)

    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_descriptor_number_is_not_supported(self, name):
        with pytest.raises(NotImplementedError):
            fn(name)(DescriptorNumber(name='d', value=0.5, unit='rad'))

    def test_scipp_variable_is_not_supported(self):
        with pytest.raises(NotImplementedError):
            es_math.sin(sc.scalar(0.5, unit='rad'))

    @pytest.mark.parametrize(('y', 'x'), [(1.0, 2.0), (1.0, make(2.0)), (make(1.0), 2.0)])
    def test_arctan2_requires_two_parameters(self, y, x):
        with pytest.raises(NotImplementedError):
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


class TestInvariants:
    @pytest.mark.parametrize('name', UNARY_FUNCTIONS)
    def test_value_lies_within_result_bounds_and_bounds_are_not_nan(self, name):
        (low, high), unit = RANDOM_RANGES[name]
        rng = np.random.default_rng(seed=1234)
        for _ in range(50):
            value = rng.uniform(low, high)
            lo = value - rng.uniform(0.01, 3.0)
            hi = value + rng.uniform(0.01, 3.0)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                result = fn(name)(make(value, unit, min=lo, max=hi, variance=0.001))
            assert not np.isnan(result.min) and not np.isnan(result.max)
            assert result.min <= result.value <= result.max

    def test_arctan2_value_lies_within_result_bounds(self):
        rng = np.random.default_rng(seed=99)
        for _ in range(50):
            y, x = rng.uniform(-3.0, 3.0, size=2)
            y_lo, y_hi = y - rng.uniform(0.01, 2.0), y + rng.uniform(0.01, 2.0)
            x_lo, x_hi = x - rng.uniform(0.01, 2.0), x + rng.uniform(0.01, 2.0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                result = es_math.arctan2(make(y, min=y_lo, max=y_hi), make(x, min=x_lo, max=x_hi))
            assert result.min <= result.value <= result.max


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


class TestPublicApi:
    @pytest.mark.parametrize('name', ALL_FUNCTIONS)
    def test_function_is_exported_from_easyscience_variable(self, name):
        assert callable(getattr(es_variable, name))
        assert getattr(es_variable, name) is getattr(es_math, name)

    @pytest.mark.parametrize('name', ALL_FUNCTIONS)
    def test_function_has_a_docstring(self, name):
        assert fn(name).__doc__ and fn(name).__doc__.strip()

    def test_variable_all_contains_only_strings(self):
        assert all(isinstance(entry, str) for entry in es_variable.__all__)
        for name in ALL_FUNCTIONS:
            assert name in es_variable.__all__
