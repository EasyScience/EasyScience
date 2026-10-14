# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The error propagation and correlation tutorials.

These notebooks compare formulas and correlations, so besides the results of their exercises
the tests check the numbers that the text of the notebooks quotes, against closed forms.
"""

import math
import re

import pytest

from .notebooks import PROJECT_ROOT
from .notebooks import load_notebook
from .notebooks import source_of

pytestmark = pytest.mark.fast

PROPAGATION = 'basics/maths-error-propagation'
CORRELATION = 'basics/maths-correlation'

# Results are printed with four decimals (five for the contributions of Exercise 2).
PRINT_TOLERANCE = 5.1e-5

LINE = re.compile(r'^(?P<label>\w+) = (?P<value>-?\d+\.\d+) ± (?P<error>\d+\.\d+) (?P<rest>.*)$')


def _lines(output: str) -> list[tuple[str, float, float, str]]:
    parsed = []
    for line in output.splitlines():
        match = LINE.match(line)
        assert match, line
        parsed.append((
            match['label'],
            float(match['value']),
            float(match['error']),
            match['rest'],
        ))
    return parsed


def _quadrature(*terms: float) -> float:
    return math.sqrt(sum(term**2 for term in terms))


_PI_GRAVITY = math.pi**2  # 4 pi^2 * 1 m / (2 s)^2
_FIT_VARIANCE = (18 * 0.02) ** 2 + 0.3**2

# (notebook, exercise number) -> [(label, value, error, unit and bounds as printed), ...]
RESULT_LINES = {
    (PROPAGATION, 1): [
        (
            'area_from_diameter',
            math.pi * 1.2**2 / 4,
            math.pi * 1.2 / 2 * 0.01,
            'm^2   in [0.0000, inf]',
        ),
        (
            'area_from_circumference',
            3.77**2 / (4 * math.pi),
            3.77 / (2 * math.pi) * 0.01,
            'm^2   in [0.0000, inf]',
        ),
    ],
    (PROPAGATION, 3): [
        (
            'by_operators',
            0.4,
            _quadrature((1.3 - 2 * 0.8) * 0.01, 0.8 / 2 * 0.01),
            'm^2',
        ),
        (
            'by_helper',
            0.4,
            _quadrature((1.3 - 2 * 0.8) * 0.01, 0.8 / 2 * 0.01),
            'm^2',
        ),
    ],
    (PROPAGATION, 4): [
        (
            'g_one_swing',
            _PI_GRAVITY,
            _PI_GRAVITY * _quadrature(0.002 / 1.0, 2 * 0.1 / 2.0),
            'm/s^2',
        ),
        (
            'g_ten_swings',
            _PI_GRAVITY,
            _PI_GRAVITY * _quadrature(0.002 / 1.0, 2 * 0.1 / 20.0),
            'm/s^2',
        ),
    ],
    (CORRELATION, 1): [
        ('by_operators', 2.0, _quadrature(0.02, 0.03), 'm'),
        ('correlated', 2.0, math.sqrt(0.02**2 + 0.03**2 + 2 * 0.9 * 0.02 * 0.03), 'm'),
    ],
    (CORRELATION, 2): [
        ('by_operators', 2.24, _quadrature(0.05, 0.05), 'm'),
        ('correlated', 2.24, 0.05 * math.sqrt(2 * (1 - 0.95)), 'm'),
    ],
    (CORRELATION, 3): [
        ('by_operators', 1.5, 1.5 * _quadrature(0.05, 0.05), 'dimensionless'),
        (
            'correlated',
            1.5,
            1.5 * math.sqrt(0.05**2 + 0.05**2 - 2 * 0.9 * 0.05 * 0.05),
            'dimensionless',
        ),
    ],
    (CORRELATION, 4): [
        ('by_operators', 11.0, math.sqrt(_FIT_VARIANCE), 'dimensionless'),
        (
            'correlated',
            11.0,
            math.sqrt(_FIT_VARIANCE + 2 * 18 * -0.97 * 0.02 * 0.3),
            'dimensionless',
        ),
    ],
}


@pytest.mark.parametrize(('name', 'number'), list(RESULT_LINES))
def test_solution_matches_the_closed_form_reference(solutions, name, number):
    shown = _lines(solutions(name)[number - 1].output)
    expected = RESULT_LINES[name, number]

    assert len(shown) == len(expected)
    for (label, value, error, rest), (want_label, want_value, want_error, want_rest) in zip(
        shown, expected
    ):
        assert label == want_label
        assert value == pytest.approx(want_value, abs=PRINT_TOLERANCE)
        assert error == pytest.approx(want_error, abs=PRINT_TOLERANCE)
        assert rest == want_rest


def test_the_volume_exercise_ranks_the_sides_by_their_contribution(solutions):
    output = solutions(PROPAGATION)[1].output
    first, *contributions = output.splitlines()

    label, value, error, rest = _lines(first)[0]
    assert (label, rest) == ('volume', 'm^3')
    assert value == pytest.approx(0.5 * 0.2 * 0.1, abs=PRINT_TOLERANCE)
    assert error == pytest.approx(
        0.01 * _quadrature(0.2 * 0.1, 0.5 * 0.1, 0.5 * 0.2), abs=PRINT_TOLERANCE
    )
    assert contributions == [
        f'length contributes {0.2 * 0.1 * 0.01:.5f} m^3',
        f'width contributes {0.5 * 0.1 * 0.01:.5f} m^3',
        f'thickness contributes {0.5 * 0.2 * 0.01:.5f} m^3',
    ]


class TestTheTableExample:
    @pytest.fixture
    def namespace(self, solutions, notebook_environment):
        solutions(PROPAGATION)
        return notebook_environment[0].user_ns

    def test_the_three_methods_have_the_uncertainties_of_the_text(self, namespace):
        propagate = namespace['propagate']
        sigma = namespace['sigma']
        by_method = {
            label: propagate(func, values, [sigma, sigma])
            for label, _, func, values in namespace['methods']
        }

        assert [value for value, _ in by_method.values()] == pytest.approx([3.0, 3.0, 3.0])
        assert by_method['length and width'][1] == pytest.approx(0.025)
        assert by_method['length and perimeter'][1] == pytest.approx(0.0112, abs=5e-5)
        assert by_method['length and diagonal'][1] == pytest.approx(0.0353, abs=5e-5)

    def test_the_first_method_agrees_with_the_operators(self, namespace):
        assert namespace['area_1'].value == pytest.approx(3.0)
        assert namespace['area_1'].error == pytest.approx(0.025)

    def test_the_operators_agree_with_the_helper_when_a_measurement_is_used_twice(self, namespace):
        operators = namespace['by_operators']
        correct = namespace['propagate'](
            lambda length, perimeter: length * (perimeter / 2 - length),
            [2.0, 7.0],
            [0.01, 0.01],
        )

        assert operators.error == pytest.approx(correct[1], rel=1e-6)
        assert correct[1] == pytest.approx(_quadrature((3.5 - 4.0) * 0.01, 1.0 * 0.01))
        assert operators.error < 0.5 * _quadrature(
            (3.5 - 2.0) * 0.01, 2.0 * _quadrature(0.005, 0.01)
        )

    def test_the_small_example_of_the_text(self, namespace):
        assert namespace['x'].value == pytest.approx(3.0)
        assert (namespace['x'] * (1 + namespace['x'])).error == pytest.approx(0.7)
        value, uncertainty = namespace['propagate'](lambda x: x * (1 + x), [3.0], [0.1])
        assert (value, uncertainty) == pytest.approx((12.0, 0.7))

    def test_the_same_quantity_by_different_routes_is_exact(self, namespace):
        x = namespace['x']

        assert (x * x).error == pytest.approx(2 * 3.0 * 0.1)
        assert (x - x).error == pytest.approx(0.0)
        assert (x * x - x**2).error == pytest.approx(0.0, abs=1e-12)
        assert ((x + 1) - x).error == pytest.approx(0.0, abs=1e-12)


class TestTheCorrelationHelper:
    @pytest.fixture
    def namespace(self, solutions, notebook_environment):
        solutions(CORRELATION)
        return notebook_environment[0].user_ns

    @pytest.mark.parametrize('rho', [-1.0, -0.3, 0.0, 0.5, 1.0])
    def test_sum_and_difference_follow_the_textbook_formulas(self, namespace, rho):
        propagate = namespace['propagate']
        correlation = [[1, rho], [rho, 1]]
        sigmas = [0.02, 0.03]

        added = propagate(lambda a, b: a + b, [1.2, 0.8], sigmas, correlation)
        subtracted = propagate(lambda a, b: a - b, [1.2, 0.8], sigmas, correlation)

        assert added[0] == pytest.approx(2.0)
        assert added[1] == pytest.approx(math.sqrt(0.02**2 + 0.03**2 + 2 * rho * 0.02 * 0.03))
        assert subtracted[0] == pytest.approx(0.4)
        assert subtracted[1] == pytest.approx(math.sqrt(0.02**2 + 0.03**2 - 2 * rho * 0.02 * 0.03))

    def test_without_a_correlation_the_helper_agrees_with_the_operators(self, namespace):
        total = namespace['first'] + namespace['second']
        value, uncertainty = namespace['propagate'](lambda a, b: a + b, [1.2, 0.8], [0.02, 0.02])

        assert (value, uncertainty) == pytest.approx((total.value, total.error))

    def test_fully_correlated_errors_add_and_cancel(self, namespace):
        propagate = namespace['propagate']
        full = [[1, 1], [1, 1]]

        assert propagate(lambda a, b: a + b, [1.2, 0.8], [0.02, 0.02], full)[1] == pytest.approx(
            0.04
        )
        assert propagate(lambda a, b: a - b, [1.2, 0.8], [0.02, 0.02], full)[1] == pytest.approx(
            0.0, abs=1e-9
        )

    def test_the_fit_example_is_strongly_anticorrelated(self, namespace):
        assert namespace['correlation'] < -0.97
        assert namespace['at_20'].error > 3 * 0.1115

    def test_operators_with_the_covariance_agree_with_the_helper(self, namespace):
        rod, propagate = namespace['rod'], namespace['propagate']
        correct = propagate(lambda a, b: a + b, [1.2, 0.8], [0.02, 0.03], [[1, 0.9], [0.9, 1]])

        assert rod.value == pytest.approx(correct[0])
        assert rod.error == pytest.approx(correct[1], rel=1e-9)
        assert namespace['left_section'].correlation_with(namespace['right_section']) == (
            pytest.approx(0.9)
        )

    def test_the_fit_prediction_with_the_covariance_is_the_correct_one(self, namespace):
        with_covariance = namespace['at_20_with_covariance']

        assert with_covariance.error == pytest.approx(namespace['with_covariance'], rel=1e-6)
        assert with_covariance.error < 0.3 * namespace['at_20'].error

    def test_the_fit_prediction_matches_the_closed_form(self, namespace):
        covariance = namespace['covariance']
        x = 20.0
        expected = math.sqrt(
            covariance[0, 0] * x**2 + covariance[1, 1] + 2 * x * covariance[0, 1],
        )
        value, uncertainty = namespace['propagate'](
            lambda a, b: a * x + b,
            [namespace['slope'], namespace['intercept']],
            [namespace['slope_error'], namespace['intercept_error']],
            [[1, namespace['correlation']], [namespace['correlation'], 1]],
        )

        assert value == pytest.approx(namespace['slope'] * x + namespace['intercept'])
        assert uncertainty == pytest.approx(expected, rel=1e-6)


class TestWording:
    @pytest.mark.parametrize('name', [PROPAGATION, CORRELATION])
    def test_the_notebooks_use_the_word_uncertainty(self, name):
        text = '\n'.join(source_of(cell) for cell in load_notebook(name)['cells'])

        assert 'uncertainty' in text

    def test_the_correlation_notebook_says_how_to_state_a_correlation(self):
        text = '\n'.join(source_of(cell) for cell in load_notebook(CORRELATION)['cells'])

        assert 'ParameterCovariance' in text
        assert r'\rho = 0' in text
        assert 'stores a variance and no covariance' not in text

    @pytest.mark.parametrize('name', [PROPAGATION, CORRELATION])
    def test_the_basics_hub_lists_the_notebook(self, name):
        hub = (PROJECT_ROOT / 'docs' / 'docs' / 'tutorials' / 'basics' / 'index.md').read_text(
            encoding='utf-8'
        )

        assert f'{name.removeprefix("basics/")}.ipynb' in hub

    def test_the_tutorial_index_links_the_basics_hub(self):
        index = (PROJECT_ROOT / 'docs' / 'docs' / 'tutorials' / 'index.md').read_text(
            encoding='utf-8'
        )

        assert '(basics/index.md)' in index
