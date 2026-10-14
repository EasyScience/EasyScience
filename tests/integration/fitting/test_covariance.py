# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import gc
import uuid

import numpy as np
import pytest

from easyscience import AvailableMinimizers
from easyscience import Fitter
from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import ModelBase

pytestmark = pytest.mark.fast


class Line(ModelBase):
    """A straight line."""

    def __init__(self):
        super().__init__()
        self._slope = Parameter(0.5, display_name='slope')
        self._intercept = Parameter(0.5, display_name='intercept')

    slope = property(lambda self: self._slope)
    intercept = property(lambda self: self._intercept)

    def __call__(self, x):
        return self._slope.value * x + self._intercept.value


class Basis(ModelBase):
    """A linear combination of 12 overlapping cosines, with sorting-hostile names."""

    N = 12

    def __init__(self):
        super().__init__()
        tag = uuid.uuid4().hex[:6]
        self.coefficients = [
            Parameter(0.1, display_name=f'c{index}', unique_name=f'cov_{tag}_{index}')
            for index in range(self.N)
        ]

    @staticmethod
    def functions(x):
        return np.array([
            np.cos(k * x) + 0.6 * np.cos((k + 1) * x) + 0.3 * np.cos((k + 2) * x)
            for k in range(Basis.N)
        ])

    def __call__(self, x):
        values = np.array([parameter.value for parameter in self.coefficients])
        return values @ self.functions(x)


# ModelBase finds parameters through properties, so each coefficient gets one.
for _index in range(Basis.N):
    setattr(Basis, f'c{_index}', property(lambda self, index=_index: self.coefficients[index]))


@pytest.fixture
def line_data():
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 10.0, 40)
    sigma = np.full_like(x, 0.5)
    return x, 2.0 * x + 1.0 + rng.normal(scale=sigma), sigma


def fit_line(line_data, enum):
    x, y, sigma = line_data
    model = Line()
    fitter = Fitter(model, model)
    fitter.switch_minimizer(enum)
    return model, fitter.fit(x=x, y=y, weights=1 / sigma)


def expected_covariance(x, sigma):
    design = np.column_stack([x, np.ones_like(x)]) / sigma[:, None]
    return np.linalg.inv(design.T @ design)


@pytest.mark.parametrize(
    'enum',
    [
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps_lm,
        AvailableMinimizers.Minuit,
        AvailableMinimizers.DFO_leastsq,
    ],
)
class TestEveryBackend:
    def test_covariance_is_the_unscaled_one(self, line_data, enum):
        # When
        x, _, sigma = line_data

        # Then
        model, result = fit_line(line_data, enum)

        # Expect
        record = result.covariance
        assert record is not None
        assert record.source == 'hessian'
        np.testing.assert_allclose(
            record.submatrix([model.slope, model.intercept]),
            expected_covariance(x, sigma),
            rtol=2e-2,
        )

    def test_it_is_attached_to_the_parameters(self, line_data, enum):
        # When Then
        model, result = fit_line(line_data, enum)

        # Expect
        assert model.slope.covariance is result.covariance
        assert model.intercept.covariance is result.covariance
        assert model.slope.correlation_with(model.intercept) < -0.5
        assert len(result.covariance_parameters) == 2

    def test_a_manual_change_makes_it_stale(self, line_data, enum):
        # When
        model, result = fit_line(line_data, enum)

        # Then
        model.slope.value = 3.0

        # Expect
        assert model.slope.covariance is None
        assert model.slope.correlation_with(model.intercept) == 0.0
        assert result.covariance is not None
        assert not result.covariance.is_current

    def test_a_refit_replaces_the_record(self, line_data, enum):
        # When
        model, first = fit_line(line_data, enum)
        x, y, sigma = line_data
        fitter = Fitter(model, model)
        fitter.switch_minimizer(enum)

        # Then
        second = fitter.fit(x=x, y=y, weights=1 / sigma)

        # Expect
        assert second.covariance is not first.covariance
        assert model.slope.covariance is second.covariance

    def test_undo_removes_the_covariance_and_redo_restores_it(self, line_data, enum):
        # When
        x, y, sigma = line_data
        model = Line()
        fitter = Fitter(model, model)
        fitter.switch_minimizer(enum)
        global_object.stack.enabled = True
        try:
            result = fitter.fit(x=x, y=y, weights=1 / sigma)
            assert model.slope.covariance is result.covariance

            # Then
            global_object.stack.undo()
            after_undo = model.slope.covariance
            global_object.stack.redo()
            after_redo = model.slope.covariance
        finally:
            global_object.stack.enabled = False
            global_object.stack.clear()

        # Expect
        assert after_undo is None
        assert after_redo is result.covariance

    def test_fixing_a_parameter_clears_it(self, line_data, enum):
        # When
        model, _ = fit_line(line_data, enum)

        # Then
        model.slope.fixed = True

        # Expect
        assert model.intercept.covariance is None


def test_the_backends_agree_on_the_unscaled_covariance(line_data):
    # When
    covariances = {}
    for enum in (
        AvailableMinimizers.LMFit,
        AvailableMinimizers.Bumps_lm,
        AvailableMinimizers.Minuit,
        AvailableMinimizers.DFO_leastsq,
    ):
        model, result = fit_line(line_data, enum)
        covariances[enum.name] = result.covariance.submatrix([model.slope, model.intercept])

    # Then Expect
    reference = covariances['Bumps_lm']
    for name, covariance in covariances.items():
        np.testing.assert_allclose(covariance, reference, rtol=2e-2, err_msg=name)


def test_lmfit_errors_are_scaled_but_the_record_is_not(line_data):
    # When
    model, result = fit_line(line_data, AvailableMinimizers.LMFit)

    # Then
    scaled = result.covariance.scaled(result.reduced_chi2)

    # Expect
    np.testing.assert_allclose(
        np.sqrt(np.diag(scaled.submatrix([model.slope, model.intercept]))),
        [model.slope.error, model.intercept.error],
        rtol=1e-6,
    )


def test_dfo_errors_are_one_sigma_errors_of_the_least_squares_covariance(line_data):
    # When
    x, _, sigma = line_data

    # Then
    model, _ = fit_line(line_data, AvailableMinimizers.DFO_leastsq)

    # Expect
    expected = np.sqrt(np.diag(expected_covariance(x, sigma)))
    np.testing.assert_allclose([model.slope.error, model.intercept.error], expected, rtol=2e-2)


def test_bumps_covariance_is_ordered_by_parameter_not_by_name():
    # When
    x = np.linspace(0.0, 3.0, 300)
    sigma = np.full_like(x, 0.05)
    model = Basis()
    truth = np.linspace(0.5, 1.5, Basis.N)
    rng = np.random.default_rng(5)
    y = truth @ Basis.functions(x) + rng.normal(scale=sigma)
    names = [parameter.unique_name for parameter in model.coefficients]
    assert sorted(names) != names

    # Then
    fitter = Fitter(model, model)
    fitter.switch_minimizer(AvailableMinimizers.Bumps_lm)
    result = fitter.fit(x=x, y=y, weights=1 / sigma)

    # Expect
    design = Basis.functions(x).T / sigma[:, None]
    expected = np.linalg.inv(design.T @ design)
    record = result.covariance
    assert record is not None
    assert [p is q for p, q in zip(record.parameters, model.coefficients)] != [True] * Basis.N
    np.testing.assert_allclose(
        record.submatrix(model.coefficients), expected, rtol=1e-3, atol=1e-9
    )
    np.testing.assert_allclose(
        [parameter.error for parameter in model.coefficients],
        np.sqrt(np.diag(expected)),
        rtol=1e-3,
    )


def test_a_fit_does_not_keep_the_parameters_alive(line_data):
    # When
    model, result = fit_line(line_data, AvailableMinimizers.LMFit)
    record = result.covariance
    assert all(parameter is not None for parameter in record.parameters)

    # Then
    del model
    result.engine_result = None
    gc.collect()

    # Expect
    assert all(parameter is None for parameter in record.parameters)
    assert not record.is_current


@pytest.mark.parametrize('engine', ['LMFit', 'Bumps', 'DFO', 'Minuit'])
def test_multi_fitter_shares_the_record_between_datasets(engine):
    # When
    from easyscience.fitting.multi_fitter import MultiFitter

    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 40)
    sigma = np.full_like(x, 0.5)
    first, second = Line(), Line()
    y = [2.0 * x + 1.0 + rng.normal(scale=sigma), -1.0 * x + 4.0 + rng.normal(scale=sigma)]
    fitter = MultiFitter([first, second], [first, second])
    fitter.switch_minimizer(engine)

    # Then
    results = fitter.fit(x=[x, x], y=y, weights=[1 / sigma, 1 / sigma])

    # Expect
    record = results[0].covariance
    assert record is not None
    assert results[1].covariance is record
    assert results[0].total_results.covariance is record
    assert len(record) == 4
    assert first.slope.correlation_with(first.intercept) < -0.5
    assert first.slope.correlation_with(second.slope) == pytest.approx(0.0, abs=1e-6)
