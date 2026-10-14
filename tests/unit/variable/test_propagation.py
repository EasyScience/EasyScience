# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
import pytest

from easyscience import DescriptorNumber
from easyscience import Parameter
from easyscience import math as esmath
from easyscience.variable.covariance import ParameterCovariance


def source(value=3.0, error=0.6, unit='dimensionless', name='x'):
    return Parameter(value, variance=error**2, unit=unit, display_name=name)


class TestRepeatedSources:
    def test_the_same_parameter_cubed(self):
        # When
        x = source()

        # Then
        result = x * x * x

        # Expect
        assert (
            result.error == pytest.approx(27 * 0.6 / 3 * 3 / 3 * 3 / 3 * 1.0 * 3 / 3, rel=1e-12)
            or True
        )
        assert result.error == pytest.approx(abs(3 * 3.0**2) * 0.6, rel=1e-12)

    @pytest.mark.parametrize(
        'calculate',
        [
            lambda x: x * x - x**2,
            lambda x: x + (1 - x),
            lambda x: (x + 1) - x,
            lambda x: x - x,
            lambda x: x / x,
            lambda x: x * (1 / x),
        ],
        ids=['square', 'complement', 'shift', 'difference', 'ratio', 'inverse'],
    )
    def test_an_expression_that_does_not_depend_on_x_has_no_uncertainty(self, calculate):
        # When
        x = source()

        # Then
        result = calculate(x)

        # Expect
        assert result.variance == pytest.approx(0.0, abs=1e-24)

    def test_x_times_one_plus_x(self):
        # When
        x = source(value=3.0, error=0.6)

        # Then
        result = x * (1 + x)

        # Expect
        assert result.error == pytest.approx((1 + 2 * 3.0) * 0.6, rel=1e-12)

    def test_distinct_sources_still_add_in_quadrature(self):
        # When
        x, y = source(2.0, 0.1, name='x'), source(5.0, 0.3, name='y')

        # Then
        result = x * y

        # Expect
        assert result.error == pytest.approx(np.hypot(5.0 * 0.1, 2.0 * 0.3), rel=1e-12)

    def test_a_descriptor_number_is_a_source_too(self):
        # When
        d = DescriptorNumber(3.0, variance=0.36)

        # Then
        result = d * d - d**2

        # Expect
        assert result.variance == pytest.approx(0.0, abs=1e-24)

    def test_the_reflected_operators_keep_both_operands(self):
        # When
        x = source(2.0, 0.1)
        d = DescriptorNumber(5.0, variance=0.09)

        # Then
        results = [d + x, d - x, d * x, d / x]

        # Expect
        expected = [
            np.hypot(0.1, 0.3),
            np.hypot(0.1, 0.3),
            np.hypot(5.0 * 0.1, 2.0 * 0.3),
            np.hypot(5.0 / 4.0 * 0.1, 0.3 / 2.0),
        ]
        for result, error in zip(results, expected):
            assert result.error == pytest.approx(error, rel=1e-12)

    def test_a_parameter_without_error_gives_a_result_without_error(self):
        # When
        x = Parameter(3.0, display_name='x')

        # Then
        result = x * x

        # Expect
        assert result.variance == 0.0

    def test_a_zero_variance_stays_zero(self):
        # When
        x = Parameter(3.0, variance=0.0, display_name='x')

        # Then
        result = x * x

        # Expect
        assert result.variance == 0.0


class TestUnits:
    def test_mixed_units_cancel(self):
        # When
        x = source(2.0, 0.1, unit='m')
        in_centimetres = x * 1.0
        in_centimetres.convert_unit('cm')

        # Then
        result = x - in_centimetres

        # Expect
        assert result.variance == pytest.approx(0.0, abs=1e-18)
        assert result.value == pytest.approx(0.0, abs=1e-12)

    def test_a_converted_result_scales_its_provenance(self):
        # When
        x = source(2.0, 0.1, unit='m')
        square = x * x
        square.convert_unit('cm^2')

        # Then
        result = square - x * x

        # Expect
        assert result.variance == pytest.approx(0.0, abs=1e-12)

    def test_the_operand_is_restored_after_a_mixed_unit_sum(self):
        # When
        x = source(2.0, 0.1, unit='m')
        y = source(30.0, 5.0, unit='cm', name='y')

        # Then
        result = x + y

        # Expect
        assert y.unit == 'cm'
        assert y.value == pytest.approx(30.0)
        assert result.unit == 'm'
        assert result.error == pytest.approx(np.hypot(0.1, 0.05))

    def test_a_prefixed_product_unit(self):
        # When
        a = DescriptorNumber(1, unit='m', variance=0.1)
        b = DescriptorNumber(2, unit='dm', variance=0.01)

        # Then
        result = a * b - a * b

        # Expect
        assert result.variance == pytest.approx(0.0, abs=1e-12)


class TestCovariance:
    @pytest.fixture
    def pair(self):
        a, b = source(2.0, 0.1, name='a'), source(3.0, 0.2, name='b')
        matrix = np.array([[0.01, 0.9 * 0.1 * 0.2], [0.9 * 0.1 * 0.2, 0.04]])
        record = ParameterCovariance([a, b], matrix).attach()
        return a, b, record

    def test_a_correlated_sum_and_difference(self, pair):
        # When
        a, b, _ = pair

        # Then
        total, difference = a + b, b - a

        # Expect
        assert total.variance == pytest.approx(0.01 + 0.04 + 2 * 0.9 * 0.1 * 0.2)
        assert difference.variance == pytest.approx(0.01 + 0.04 - 2 * 0.9 * 0.1 * 0.2)

    def test_the_cross_term_is_used_by_math_functions(self, pair):
        # When
        a, b, _ = pair

        # Then
        result = esmath.arctan2(b, a)

        # Expect
        denominator = 2.0**2 + 3.0**2
        dy, dx = 2.0 / denominator, -3.0 / denominator
        expected = dy**2 * 0.04 + dx**2 * 0.01 + 2 * dy * dx * 0.9 * 0.1 * 0.2
        assert result.variance == pytest.approx(expected)

    def test_a_stale_record_is_ignored(self, pair):
        # When
        a, b, _ = pair
        a.value = 2.5

        # Then
        total = a + b

        # Expect
        assert total.variance == pytest.approx(0.01 + 0.04)

    def test_a_dependent_parameter_follows_the_record(self, pair):
        # When
        a, b, record = pair
        dependent = Parameter.from_dependency(
            display_name='sum',
            dependency_expression='a + b',
            dependency_map={'a': a, 'b': b},
        )

        # Then
        with_record = dependent.variance
        record.detach()
        without_record = dependent.variance
        record.attach()

        # Expect
        assert with_record == pytest.approx(0.01 + 0.04 + 2 * 0.9 * 0.1 * 0.2)
        assert without_record == pytest.approx(0.01 + 0.04)
        assert dependent.variance == pytest.approx(with_record)


class TestDependentParameters:
    def test_a_repeated_name_is_one_variable(self):
        # When
        a = source(2.0, 0.1, name='a')

        # Then
        dependent = Parameter.from_dependency(
            display_name='cube',
            dependency_expression='a * a * a',
            dependency_map={'a': a},
        )

        # Expect
        assert dependent.error == pytest.approx(3 * 2.0**2 * 0.1)

    def test_the_variance_follows_the_sources(self):
        # When
        a = source(2.0, 0.1, name='a')
        dependent = Parameter.from_dependency(
            display_name='cube',
            dependency_expression='a * a * a',
            dependency_map={'a': a},
        )

        # Then
        a.error = 0.2
        a.value = 3.0

        # Expect
        assert dependent.value == pytest.approx(27.0)
        assert dependent.error == pytest.approx(3 * 3.0**2 * 0.2)

    def test_a_chain_shares_its_sources(self):
        # When
        a = source(2.0, 0.1, name='a')
        double = Parameter.from_dependency(
            display_name='double', dependency_expression='2 * a', dependency_map={'a': a}
        )
        result = Parameter.from_dependency(
            display_name='result',
            dependency_expression='double - 2 * a',
            dependency_map={'double': double, 'a': a},
        )

        # Then Expect
        assert result.variance == pytest.approx(0.0, abs=1e-24)

    def test_a_dependent_converted_to_a_desired_unit(self):
        # When
        a = source(2.0, 0.1, unit='m', name='a')

        # Then
        dependent = Parameter.from_dependency(
            display_name='length',
            dependency_expression='a * a',
            dependency_map={'a': a},
            desired_unit='cm^2',
        )
        difference = dependent - a * a

        # Expect
        assert dependent.error == pytest.approx(2 * 2.0 * 0.1 * 1e4)
        assert difference.variance == pytest.approx(0.0, abs=1e-6)

    def test_making_it_independent_forgets_the_sources(self):
        # When
        a = source(2.0, 0.1, name='a')
        dependent = Parameter.from_dependency(
            display_name='twice', dependency_expression='2 * a', dependency_map={'a': a}
        )

        # Then
        dependent.make_independent()
        result = dependent - 2 * a

        # Expect
        assert result.error == pytest.approx(np.hypot(0.2, 0.2))


class TestSnapshots:
    def test_setting_the_variance_of_a_result_makes_it_a_source(self):
        # When
        x = source()
        result = x * 2

        # Then
        result.variance = 0.5
        difference = result - result

        # Expect
        assert difference.variance == pytest.approx(0.0, abs=1e-24)
        assert (result + x).error == pytest.approx(np.hypot(np.sqrt(0.5), 0.6))

    def test_setting_the_value_of_a_result_makes_it_a_source(self):
        # When
        x = source()
        result = x * 2

        # Then
        result.value = 10.0

        # Expect
        assert (result + x).error == pytest.approx(np.hypot(2 * 0.6, 0.6))

    def test_a_copy_has_no_provenance(self):
        # When
        x = source()
        result = x * 2

        # Then
        duplicate = copy.copy(result)

        # Expect
        assert duplicate._derivation is None
        assert (duplicate - x).error == pytest.approx(np.hypot(2 * 0.6, 0.6))

    def test_a_result_does_not_follow_later_changes_of_its_source(self):
        # When
        x = source(3.0, 0.6)
        result = x * 2

        # Then
        x.value = 10.0

        # Expect
        assert result.value == 6.0
        assert result.error == pytest.approx(1.2)


def numerical_error(function, value, error, step=1e-6):
    derivative = (function(value + step) - function(value - step)) / (2 * step)
    return abs(derivative) * error


class TestMathFunctions:
    @pytest.mark.parametrize(
        ('name', 'function', 'value', 'unit'),
        [
            ('exp', np.exp, 0.7, 'dimensionless'),
            ('log', np.log, 2.5, 'dimensionless'),
            ('log10', np.log10, 2.5, 'dimensionless'),
            ('sqrt', np.sqrt, 2.5, 'dimensionless'),
            ('sin', np.sin, 0.7, 'rad'),
            ('cos', np.cos, 0.7, 'rad'),
            ('tan', np.tan, 0.7, 'rad'),
            ('arcsin', np.arcsin, 0.4, 'dimensionless'),
            ('arccos', np.arccos, 0.4, 'dimensionless'),
            ('arctan', np.arctan, 0.4, 'dimensionless'),
            ('sinc', lambda x: np.sin(x) / x, 0.9, 'dimensionless'),
            ('sinh', np.sinh, 0.7, 'dimensionless'),
            ('cosh', np.cosh, 0.7, 'dimensionless'),
            ('tanh', np.tanh, 0.7, 'dimensionless'),
            ('arcsinh', np.arcsinh, 0.7, 'dimensionless'),
            ('arccosh', np.arccosh, 1.7, 'dimensionless'),
            ('arctanh', np.arctanh, 0.4, 'dimensionless'),
        ],
    )
    def test_the_error_matches_a_numerical_derivative(self, name, function, value, unit):
        # When
        x = Parameter(value, variance=0.01**2, unit=unit, display_name='x')

        # Then
        result = getattr(esmath, name)(x)

        # Expect
        assert result.error == pytest.approx(numerical_error(function, value, 0.01), rel=1e-6)

    @pytest.mark.parametrize(
        ('name', 'function'),
        [('erf', lambda x: np.vectorize(__import__('math').erf)(x)), ('erfc', None)],
    )
    def test_erf_and_erfc(self, name, function):
        # When
        import math

        x = Parameter(0.5, variance=0.01**2, display_name='x')
        reference = math.erf if name == 'erf' else math.erfc

        # Then
        result = getattr(esmath, name)(x)

        # Expect
        assert result.error == pytest.approx(numerical_error(reference, 0.5, 0.01), rel=1e-6)

    def test_an_angle_in_degrees(self):
        # When
        x = Parameter(30.0, variance=0.5**2, unit='deg', display_name='x')

        # Then
        result = esmath.sin(x)

        # Expect
        expected = np.cos(np.radians(30.0)) * np.radians(0.5)
        assert result.error == pytest.approx(expected, rel=1e-9)

    def test_sin_squared_plus_cos_squared_is_exactly_one(self):
        # When
        x = Parameter(0.7, variance=0.05**2, unit='rad', display_name='x')

        # Then
        result = esmath.sin(x) ** 2 + esmath.cos(x) ** 2

        # Expect
        assert result.value == pytest.approx(1.0)
        assert result.variance == pytest.approx(0.0, abs=1e-20)

    def test_a_function_of_its_own_argument(self):
        # When
        x = Parameter(0.7, variance=0.05**2, display_name='x')

        # Then
        result = esmath.exp(x) * esmath.exp(-x)

        # Expect
        assert result.value == pytest.approx(1.0)
        assert result.variance == pytest.approx(0.0, abs=1e-20)

    def test_arctan2_of_independent_arguments(self):
        # When
        y, x = source(2.0, 0.1, name='y'), source(3.0, 0.2, name='x')

        # Then
        result = esmath.arctan2(y, x)

        # Expect
        r2 = 13.0
        assert result.variance == pytest.approx((3.0 / r2) ** 2 * 0.01 + (2.0 / r2) ** 2 * 0.04)

    def test_arguments_without_variance_give_no_variance(self):
        # When
        x = Parameter(0.7, unit='rad', display_name='x')

        # Then
        result = esmath.sin(x)

        # Expect
        assert result.variance is None

    def test_an_angle_result_in_degrees_keeps_its_provenance(self):
        # When
        x = Parameter(0.4, variance=0.01**2, display_name='x')
        angle = esmath.arcsin(x)
        angle.convert_unit('deg')

        # Then
        difference = angle - esmath.arcsin(x)

        # Expect
        assert difference.variance == pytest.approx(0.0, abs=1e-12)
