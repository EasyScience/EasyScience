# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Results shown by the exercise solutions of the tutorial notebooks.

The solutions are run when a notebook is executed, so the documentation shows whatever they
print. These tests replay the notebooks and compare each result with a closed-form reference
(value and error propagated by hand, as in the exercise hints), so that a change to the library
cannot silently change what the tutorials teach.
"""

import math
import re

import numpy as np
import pytest

from .notebooks import EXERCISES
from .notebooks import FITTING_NOTEBOOKS
from .notebooks import MATHS_NOTEBOOKS
from .notebooks import NOTEBOOKS
from .notebooks import Solution

pytestmark = pytest.mark.fast

# Results are printed with four decimals, so a reference can be off by half a unit of the last one.
PRINT_TOLERANCE = 5.1e-5

RESULT = re.compile(r'^(?P<value>-?\d+\.\d+) ± (?P<error>\d+\.\d+) (?P<rest>.*)$')


def _parse(output: str) -> tuple[float, float, str]:
    """Split ``value ± error unit [in [min, max]]`` into its parts."""
    match = RESULT.match(output)
    assert match, output
    return float(match['value']), float(match['error']), match['rest']


def _rad(degrees: float) -> float:
    return math.radians(degrees)


def _quadrature(*terms: float) -> float:
    return math.sqrt(sum(term**2 for term in terms))


# Arithmetic: sums, differences, quotients and powers.
_THETA = _rad(22.5)
_SIGMA_THETA = _rad(0.2)
_RATIO_SIGMA = 0.25 * _quadrature(0.01 / 0.3, 0.02 / 1.2)
_BRAGG_RATIO = 1.5406 / (2 * 3.1356)
_ARG_ERF = 1.0 / (0.8 * math.sqrt(2))
_SINC = math.sin(2.0) / 2.0
_DSINC = (2.0 * math.cos(2.0) - math.sin(2.0)) / 4.0
_TANH = math.tanh(0.8)

# (notebook, exercise number, value, error, unit and bounds as printed)
REFERENCES = [
    ('basics/maths-arithmetic', 1, 2.0, _quadrature(0.02, 0.03), 'm'),
    ('basics/maths-arithmetic', 2, 1.075, _quadrature(0.02, 0.005), 'm'),
    ('basics/maths-arithmetic', 3, 2.0, 2.0 * _quadrature(0.1 / 3.0, 0.05 / 1.5), 'm/s'),
    (
        'basics/maths-arithmetic',
        4,
        4 * math.pi,
        2 * math.pi * 2.0 * 0.1,
        'cm^2   in [0.0000, inf]',
    ),
    (
        'basics/maths-exp-log',
        1,
        math.exp(-1.0),
        math.exp(-1.0) * _quadrature(0.02 / 0.5, 0.1 / 2.0),
        'dimensionless   in [0.0000, inf]',
    ),
    (
        'basics/maths-exp-log',
        2,
        10 * math.log10(0.25),
        10 / math.log(10) * 10.0 / 250.0,
        'dimensionless',
    ),
    (
        'basics/maths-exp-log',
        3,
        0.8,
        0.02 / (2 * 0.8),
        'dimensionless   in [0.0000, inf]',
    ),
    (
        'basics/maths-exp-log',
        4,
        math.erf(_ARG_ERF),
        2 / math.sqrt(math.pi) * math.exp(-(_ARG_ERF**2)) * _ARG_ERF * _quadrature(0.05, 0.05),
        'dimensionless   in [-1.0000, 1.0000]',
    ),
    (
        'basics/maths-trigonometric',
        1,
        4 * math.pi * math.sin(_THETA) / 1.54,
        4 * math.pi * math.cos(_THETA) / 1.54 * _SIGMA_THETA,
        '1/Å   in [-8.1600, 8.1600]',
    ),
    (
        'basics/maths-trigonometric',
        2,
        1.54 / (2 * math.sin(_THETA)),
        1.54 * math.cos(_THETA) / (2 * math.sin(_THETA) ** 2) * _SIGMA_THETA,
        'Å',
    ),
    (
        'basics/maths-trigonometric',
        3,
        1.5 * math.tan(_rad(30.0)),
        _quadrature(
            math.tan(_rad(30.0)) * 0.01,
            1.5 / math.cos(_rad(30.0)) ** 2 * _rad(0.2),
        ),
        'm',
    ),
    (
        'basics/maths-trigonometric',
        4,
        _SINC**2,
        abs(2 * _SINC * _DSINC) * 0.1,
        'dimensionless   in [0.0000, 1.0000]',
    ),
    (
        'basics/maths-inverse-trigonometric',
        1,
        math.degrees(math.asin(_BRAGG_RATIO)),
        math.degrees(
            _BRAGG_RATIO * 0.0005 / 3.1356 / math.sqrt(1 - _BRAGG_RATIO**2),
        ),
        'deg   in [-90.0000, 90.0000]',
    ),
    (
        'basics/maths-inverse-trigonometric',
        2,
        math.degrees(math.atan(0.25)),
        math.degrees(_RATIO_SIGMA / (1 + 0.25**2)),
        'deg   in [-90.0000, 90.0000]',
    ),
    (
        'basics/maths-inverse-trigonometric',
        3,
        math.degrees(math.atan2(2.0, 1.5)),
        math.degrees(0.05 * math.hypot(1.5, 2.0) / (1.5**2 + 2.0**2)),
        'deg   in [-180.0000, 180.0000]',
    ),
    (
        'basics/maths-hyperbolic',
        1,
        _TANH,
        (1 - _TANH**2) * 0.05,
        'dimensionless   in [0.0000, 0.9951]',
    ),
    (
        'basics/maths-hyperbolic',
        2,
        math.atanh(0.5),
        0.02 / (1 - 0.5**2),
        'dimensionless',
    ),
]

# Exercises that print something other than a value with its error.
MESSAGES = {
    ('basics/maths-hyperbolic', 3): '1.0',
    ('basics/maths-hyperbolic', 4): (
        "The value of 'x' (0.5) is outside the domain [1.0, inf] of arccosh()."
    ),
}


@pytest.fixture
def solutions(run_notebook):
    cache: dict[str, list[Solution]] = {}

    def get(name: str) -> list[Solution]:
        if name not in cache:
            cache[name] = run_notebook(name)
        return cache[name]

    return get


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_every_exercise_shows_a_solution_with_code_and_output(solutions, name):
    shown = solutions(name)

    assert len(shown) == EXERCISES[name]
    for solution in shown:
        assert solution.exercise.startswith('Exercise')
        assert solution.code.strip()
        assert solution.output


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_exercise_titles_are_numbered_in_order(solutions, name):
    numbers = [int(re.match(r'Exercise (\d+)', s.exercise)[1]) for s in solutions(name)]

    assert numbers == list(range(1, EXERCISES[name] + 1))


@pytest.mark.parametrize(
    ('name', 'number', 'value', 'error', 'rest'),
    REFERENCES,
    ids=[f'{name.split("-", 1)[1]}-{number}' for name, number, *_ in REFERENCES],
)
def test_solution_matches_the_closed_form_reference(solutions, name, number, value, error, rest):
    shown_value, shown_error, shown_rest = _parse(solutions(name)[number - 1].output)

    assert shown_value == pytest.approx(value, abs=PRINT_TOLERANCE)
    assert shown_error == pytest.approx(error, abs=PRINT_TOLERANCE)
    assert shown_rest == rest


@pytest.mark.parametrize(('name', 'number'), list(MESSAGES))
def test_solution_prints_the_expected_message(solutions, name, number):
    assert solutions(name)[number - 1].output == MESSAGES[name, number]


def test_arcsin_outside_its_domain_names_the_intermediate_parameter(solutions):
    output = solutions('basics/maths-inverse-trigonometric')[3].output

    ratio = 1.5406 / (2 * 0.70)
    match = re.fullmatch(
        r"The value of 'ratio' \((?P<value>[\d.]+)\) is outside the domain "
        r'\[-1\.0, 1\.0\] of arcsin\(\)\.',
        output,
    )
    assert match, output
    assert float(match['value']) == pytest.approx(ratio)


@pytest.mark.parametrize(
    ('name', 'number'),
    [('basics/maths-hyperbolic', 3)],
)
def test_the_lowest_value_of_cosh_is_one(solutions, name, number):
    assert float(solutions(name)[number - 1].output) == 1.0


@pytest.mark.parametrize(
    ('name', 'number', 'text'),
    [
        ('basics/maths-exp-log', 1, 'The lower bound 0 appears because an exponential is never'),
        ('basics/maths-hyperbolic', 1, 'The upper bound is $\\tanh(3)$'),
        ('basics/maths-inverse-trigonometric', 4, 'Without the line that sets the name'),
    ],
)
def test_notes_are_shown_after_the_output(solutions, name, number, text):
    page = solutions(name)[number - 1].html

    assert page.index('Output:') < page.index(text)


@pytest.fixture
def fitted_line():
    """Weighted straight-line fit of the simulated data of the fitting tutorial, in closed form.

    The error is the one a least-squares fit reports: the covariance of the weighted fit scaled
    by the reduced chi-squared.
    """
    rng = np.random.default_rng(seed=20261013)
    noise = 0.3
    x = np.linspace(0.0, 10.0, 40)
    y = 0.6 * x + 1.0 + rng.normal(scale=noise, size=x.size)
    design = np.column_stack([x, np.ones_like(x)])
    covariance = np.linalg.inv(design.T @ design / noise**2)
    slope, intercept = covariance @ (design.T @ y / noise**2)
    residuals = (y - design @ [slope, intercept]) / noise
    reduced_chi_squared = residuals @ residuals / (x.size - 2)
    return slope, math.sqrt(covariance[0, 0] * reduced_chi_squared)


@pytest.mark.parametrize('name', FITTING_NOTEBOOKS)
def test_derived_quantities_follow_the_fitted_slope(solutions, fitted_line, name):
    slope, slope_error = fitted_line
    angle, inverse = solutions(name)

    angle_value, angle_error, angle_rest = _parse(angle.output)
    inverse_value, inverse_error, inverse_rest = _parse(inverse.output)

    assert angle_value == pytest.approx(math.atan(slope), abs=2e-4)
    assert angle_error == pytest.approx(slope_error / (1 + slope**2), abs=2e-4)
    assert angle_rest == 'rad   in [-1.5708, 1.5708]'
    assert inverse_value == pytest.approx(1 / slope, abs=2e-4)
    assert inverse_error == pytest.approx(slope_error / slope**2, abs=2e-4)
    assert inverse_rest == 'dimensionless'


def test_the_references_cover_every_numeric_exercise():
    covered = {(name, number) for name, number, *_ in REFERENCES} | set(MESSAGES)
    # The fitting tutorial is covered by its own test and the inverse-trigonometric message by
    # a dedicated one.
    expected = {
        (name, number) for name in MATHS_NOTEBOOKS for number in range(1, EXERCISES[name] + 1)
    } - {('basics/maths-inverse-trigonometric', 4)}

    assert covered == expected
