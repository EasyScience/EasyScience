# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
Fitting through an external, stateless calculator.

Core keeps the only copy of parameter state. A calculator reads the
model when asked to calculate and does not write back. These tests pin
that contract down with a plain class, without any core calculator
machinery.
"""

import numpy as np
import pytest
from easy_test_models import CoefficientModel

from easyscience import Parameter
from easyscience import global_object
from easyscience.fitting import Fitter
from easyscience.fitting.multi_fitter import MultiFitter


class PolynomialCalculator:
    """
    A stateless calculator owned by a technique library.

    It keeps no copy of any parameter: every evaluation reads the
    current values from the model it is handed.
    """

    def __init__(self):
        self.n_calls = 0

    def calculate(self, model: CoefficientModel, x: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        coefficients = [float(c.value) for c in model.coefficients]
        return np.polyval(coefficients, x)

    def fit_func_for(self, model: CoefficientModel):
        return lambda x: self.calculate(model, x)


@pytest.fixture(autouse=True)
def clear():
    global_object.map._clear()
    yield
    global_object.map._clear()
    if global_object.stack:
        global_object.stack.clear()
        global_object.stack.enabled = False


def noise(x: np.ndarray) -> np.ndarray:
    # Small, reproducible noise so every engine can estimate uncertainties
    return np.random.default_rng(0).normal(0.0, 0.01, size=x.shape)


def snapshot(model: CoefficientModel):
    return [
        (p.unique_name, p.value, p.variance, p.min, p.max, p.fixed, p.independent)
        for p in model.coefficients
    ]


def test_calculation_reads_current_values_and_leaves_model_untouched():
    # When
    model = CoefficientModel(coefficients=[2.0, 1.0])
    calculator = PolynomialCalculator()
    x = np.linspace(0, 1, 5)
    before = snapshot(model)

    # Then Expect
    np.testing.assert_allclose(calculator.calculate(model, x), 2.0 * x + 1.0)
    assert snapshot(model) == before

    # Then # Edits made through core are seen by the next evaluation
    model.coefficients[0].value = 3.0

    # Expect
    np.testing.assert_allclose(calculator.calculate(model, x), 3.0 * x + 1.0)


@pytest.mark.parametrize('fit_engine', ['LMFit', 'Bumps', 'DFO'])
def test_fit_with_dependent_parameter(fit_engine):
    # When # The intercept is tied to the slope, so only the slope is fitted
    slope = Parameter(1.0, display_name='slope', min=-10, max=10)
    intercept = Parameter.from_dependency('0.5 * s', {'s': slope}, display_name='intercept')
    model = CoefficientModel(coefficients=[slope, intercept])
    calculator = PolynomialCalculator()
    x = np.linspace(0, 10, 50)
    y = 3.0 * x + 1.5 + noise(x)
    weights = np.ones_like(x)

    fitter = Fitter(model, calculator.fit_func_for(model))
    fitter.switch_minimizer(fit_engine)

    # Then
    result = fitter.fit(x=x, y=y, weights=weights)

    # Expect
    assert result.success
    assert calculator.n_calls > 0
    assert [p.unique_name for p in model.get_fit_parameters()] == [slope.unique_name]
    assert slope.value == pytest.approx(3.0, abs=1e-2)
    assert intercept.value == pytest.approx(0.5 * slope.value)
    np.testing.assert_allclose(calculator.calculate(model, x), y, atol=0.1)


@pytest.mark.parametrize('fit_engine', ['LMFit', 'Bumps', 'DFO'])
def test_multi_fit_with_shared_parameter(fit_engine):
    # When # Two models share their intercept
    shared = Parameter(0.0, display_name='shared', min=-10, max=10)
    model_1 = CoefficientModel(
        coefficients=[Parameter(1.0, display_name='slope_1', min=-10, max=10), shared]
    )
    model_2 = CoefficientModel(
        coefficients=[Parameter(1.0, display_name='slope_2', min=-10, max=10), shared]
    )
    calculator = PolynomialCalculator()
    x_1 = np.linspace(0, 5, 30)
    x_2 = np.linspace(-5, 0, 40)
    y_1 = 2.0 * x_1 - 1.0 + noise(x_1)
    y_2 = -3.0 * x_2 - 1.0 + noise(x_2)

    fitter = MultiFitter(
        [model_1, model_2],
        [calculator.fit_func_for(model_1), calculator.fit_func_for(model_2)],
    )
    fitter.switch_minimizer(fit_engine)

    # Then
    results = fitter.fit(
        x=[x_1, x_2], y=[y_1, y_2], weights=[np.ones_like(x_1), np.ones_like(x_2)]
    )

    # Expect
    assert all(result.success for result in results)
    assert len({p.unique_name for p in fitter.fit_object.get_fit_parameters()}) == 3
    assert model_1.coefficients[0].value == pytest.approx(2.0, abs=1e-2)
    assert model_2.coefficients[0].value == pytest.approx(-3.0, abs=1e-2)
    assert shared.value == pytest.approx(-1.0, abs=1e-2)
    assert model_1.coefficients[1] is model_2.coefficients[1]


def test_fit_is_one_undo_step():
    # When
    if not global_object.stack:
        global_object.instantiate_stack()
    global_object.stack.enabled = True
    slope = Parameter(1.0, display_name='slope', min=-10, max=10)
    intercept = Parameter(0.0, display_name='intercept', min=-10, max=10)
    model = CoefficientModel(coefficients=[slope, intercept])
    calculator = PolynomialCalculator()
    x = np.linspace(0, 10, 50)
    y = 3.0 * x + 1.5
    global_object.stack.clear()

    fitter = Fitter(model, calculator.fit_func_for(model))

    # Then
    result = fitter.fit(x=x, y=y, weights=np.ones_like(x))

    # Expect
    assert result.success
    fitted = (slope.value, intercept.value)
    assert len(global_object.stack.history) == 1

    # Then
    global_object.stack.undo()

    # Expect
    assert (slope.value, intercept.value) == (1.0, 0.0)

    # Then
    global_object.stack.redo()

    # Expect
    assert (slope.value, intercept.value) == fitted
