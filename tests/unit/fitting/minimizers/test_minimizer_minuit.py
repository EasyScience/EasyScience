# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging
from unittest.mock import MagicMock

import iminuit
import numpy as np
import pytest

from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.fitting.available_minimizers import AvailableMinimizers
from easyscience.fitting.minimizers.minimizer_minuit import Minuit
from easyscience.fitting.minimizers.utils import FitError


class TestMinuit:
    @pytest.fixture
    def minimizer(self) -> Minuit:
        minimizer = Minuit(
            obj='obj',
            fit_function='fit_function',
            minimizer_enum=MagicMock(package='minuit', method='migrad'),
        )
        return minimizer

    def test_init(self, minimizer: Minuit) -> None:
        assert minimizer.package == 'minuit'

    def test_init_exception(self) -> None:
        with pytest.raises(FitError):
            Minuit(
                obj='obj',
                fit_function='fit_function',
                minimizer_enum=MagicMock(package='minuit', method='not_migrad'),
            )

    def test_supported_methods(self) -> None:
        assert Minuit.supported_methods() == ['migrad', 'simplex']

    def test_all_methods(self) -> None:
        assert Minuit.all_methods() == ['migrad', 'simplex']

    @pytest.mark.parametrize(
        'weights',
        [
            np.array([1, 2, 3, 4]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.repeat(np.nan, 3),
            np.zeros(3),
            np.repeat(np.inf, 3),
            -np.ones(3),
        ],
        ids=['wrong_length', 'multidimensional', 'NaNs', 'zeros', 'Infs', 'negative'],
    )
    def test_fit_weight_exceptions(self, minimizer: Minuit, weights) -> None:
        # When Then Expect
        with pytest.raises(ValueError):
            minimizer.fit(x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), weights=weights)


class Line(ModelBase):
    """A straight line whose parameters start at zero."""

    def __init__(self, bounds: str = 'both'):
        super().__init__()
        limits = {'both': (-100.0, 100.0), 'none': (-np.inf, np.inf), 'lower': (0.0, np.inf)}
        low, high = limits[bounds]
        self._slope = Parameter(0.0, display_name='slope', min=low, max=high)
        self._intercept = Parameter(0.0, display_name='intercept', min=low, max=high)

    @property
    def slope(self) -> Parameter:
        return self._slope

    @property
    def intercept(self) -> Parameter:
        return self._intercept

    def __call__(self, x):
        return self._slope.value * x + self._intercept.value


def line_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 30)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.5, size=x.size)
    return x, y, np.ones_like(x)


def engine_for(model: Line, enum: AvailableMinimizers = AvailableMinimizers.Minuit) -> Minuit:
    return Minuit(obj=model, fit_function=model, minimizer_enum=enum)


class TestMinuitFit:
    def test_fit_recovers_the_least_squares_line(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()

        # Then
        engine_for(model).fit(x=x, y=y, weights=weights)

        # Expect
        slope, intercept = np.polyfit(x, y, 1)
        assert model.slope.value == pytest.approx(slope, abs=1e-6)
        assert model.intercept.value == pytest.approx(intercept, abs=1e-6)

    def test_default_tolerance_reaches_the_least_squares_minimum(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()

        # Then
        engine_for(model).fit(x=x, y=y, weights=weights)

        # Expect
        slope, intercept = np.polyfit(x, y, 1)
        assert model.slope.value == pytest.approx(slope, abs=1e-6)
        assert model.intercept.value == pytest.approx(intercept, abs=1e-6)

    # A one-sided limit makes Minuit transform the parameter, which caps the precision near 1e-5
    # whatever the tolerance; without limits the fit is exact to round-off.
    @pytest.mark.parametrize('bounds,precision', [('none', 1e-6), ('lower', 1e-4)])
    def test_unbounded_parameters_fit(self, bounds: str, precision: float) -> None:
        # When
        model = Line(bounds=bounds)
        x, y, weights = line_data()

        # Then
        engine_for(model).fit(x=x, y=y, weights=weights)

        # Expect
        slope, intercept = np.polyfit(x, y, 1)
        assert model.slope.value == pytest.approx(slope, abs=precision)
        assert model.intercept.value == pytest.approx(intercept, abs=precision)

    def test_fit_result_fields(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()

        # Then
        result = engine_for(model).fit(x=x, y=y, weights=weights)

        # Expect
        assert result.success is True
        assert result.n_evaluations > 0
        assert result.y_calc == pytest.approx(model(x))
        assert result.y_err == pytest.approx(1 / weights)
        assert result.minimizer_engine is Minuit
        assert isinstance(result.engine_result, iminuit.Minuit)
        assert set(result.p) == set(result.p0)
        assert len(result.p) == 2
        assert isinstance(result.iterations, int)
        assert result.iterations >= 0

    def test_fixed_parameter_is_not_fitted(self) -> None:
        # When
        model = Line()
        model.intercept.value = 3.0
        model.intercept.fixed = True
        x, y, weights = line_data()

        # Then
        result = engine_for(model).fit(x=x, y=y, weights=weights)

        # Expect
        assert model.intercept.value == 3.0
        assert len(result.engine_result.parameters) == 1
        assert model.slope.value == pytest.approx(np.sum(x * (y - 3.0)) / np.sum(x**2), abs=1e-6)


class TestMinuitErrors:
    def test_errors_are_the_unscaled_hesse_errors(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()

        # Then
        engine_for(model).fit(x=x, y=y, weights=weights)

        # Expect
        sxx = np.sum((x - x.mean()) ** 2)
        assert model.slope.error == pytest.approx(1 / np.sqrt(sxx), rel=1e-3)
        assert model.intercept.error == pytest.approx(
            np.sqrt(np.sum(x**2) / (x.size * sxx)), rel=1e-3
        )

    def test_scale_covar_multiplies_by_the_reduced_chi_square(self) -> None:
        # When
        x, y, weights = line_data()
        unscaled, scaled = Line(), Line()

        # Then
        engine_for(unscaled).fit(x=x, y=y, weights=weights)
        engine_for(scaled).fit(x=x, y=y, weights=weights, scale_covar=True)

        # Expect
        assert unscaled.slope.error > 0
        chi2 = np.sum(((y - unscaled(x)) * weights) ** 2)
        factor = np.sqrt(chi2 / (x.size - 2))
        assert scaled.slope.error == pytest.approx(unscaled.slope.error * factor, rel=1e-3)
        assert scaled.intercept.error == pytest.approx(unscaled.intercept.error * factor, rel=1e-3)

    def test_scale_covar_with_no_degrees_of_freedom_is_unchanged(self) -> None:
        # When
        x, y, weights = np.array([0.0, 1.0]), np.array([1.0, 3.0]), np.ones(2)
        unscaled, scaled = Line(), Line()

        # Then
        engine_for(unscaled).fit(x=x, y=y, weights=weights)
        engine_for(scaled).fit(x=x, y=y, weights=weights, scale_covar=True)

        # Expect
        assert unscaled.slope.error > 0
        assert scaled.slope.error == pytest.approx(unscaled.slope.error, rel=1e-6)
        assert scaled.intercept.error == pytest.approx(unscaled.intercept.error, rel=1e-6)

    def test_simplex_gives_no_errors(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()

        # Then
        engine_for(model, AvailableMinimizers.Minuit_simplex).fit(x=x, y=y, weights=weights)

        # Expect
        slope, intercept = np.polyfit(x, y, 1)
        assert model.slope.value == pytest.approx(slope, abs=5e-2)
        assert model.intercept.value == pytest.approx(intercept, abs=5e-2)
        assert model.slope.error is None
        assert model.intercept.error is None

    def test_stale_error_is_cleared(self) -> None:
        # When
        model = Line()
        model.slope.error = 5.0
        x, y, weights = line_data()

        # Then
        engine_for(model, AvailableMinimizers.Minuit_simplex).fit(x=x, y=y, weights=weights)

        # Expect
        assert model.slope.error is None

    def test_parameter_at_a_limit_warns_and_keeps_errors(self, caplog) -> None:
        # When
        model = Line()
        model.slope.max = 0.5
        x, y, weights = line_data()

        # Then
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.minuit'):
            engine_for(model).fit(x=x, y=y, weights=weights)

        # Expect
        assert model.slope.value == pytest.approx(0.5, abs=1e-6)
        assert 'limit' in caplog.text
        assert model.slope.error is not None


class FailingLine(Line):
    """A line whose model raises after a few evaluations."""

    def __init__(self, evaluations_before_failure: int = 5):
        super().__init__()
        self._left = evaluations_before_failure

    def __call__(self, x):
        self._left -= 1
        if self._left < 0:
            raise RuntimeError('the model failed')
        return super().__call__(x)


class TestMinuitRobustness:
    def test_whole_fit_is_one_undo_step(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()
        global_object.stack.clear()
        global_object.stack.enabled = True

        # Then
        try:
            engine_for(model).fit(x=x, y=y, weights=weights)
            fitted_slope = model.slope.value
            global_object.stack.undo()
            undone = (model.slope.value, model.intercept.value)
        finally:
            global_object.stack.enabled = False
            global_object.stack.clear()

        # Expect
        assert fitted_slope == pytest.approx(2.0, abs=0.2)
        assert undone == (0.0, 0.0)

    def test_exception_restores_start_values_and_raises_fit_error(self) -> None:
        # When
        model = FailingLine()
        x, y, weights = line_data()

        # Then Expect
        with pytest.raises(FitError):
            engine_for(model).fit(x=x, y=y, weights=weights)
        assert (model.slope.value, model.intercept.value) == (0.0, 0.0)
        assert (model.slope.error, model.intercept.error) == (0.0, 0.0)

    def test_no_free_parameters_raises_fit_error(self) -> None:
        # When
        model = Line()
        model.slope.fixed = True
        model.intercept.fixed = True
        x, y, weights = line_data()

        # Then Expect
        with pytest.raises(FitError, match='free'):
            engine_for(model).fit(x=x, y=y, weights=weights)

    def test_progress_callback_payload(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()
        payloads = []

        # Then
        engine = engine_for(model)
        engine.fit(x=x, y=y, weights=weights, progress_callback=payloads.append)

        # Expect
        assert len(payloads) > 0
        assert set(payloads[0]) == {
            'iteration',
            'chi2',
            'reduced_chi2',
            'parameter_values',
            'refresh_plots',
            'finished',
        }
        assert set(payloads[0]['parameter_values']) == set(engine._cached_pars)
        assert payloads[0]['finished'] is False
        assert payloads[-1]['iteration'] > payloads[0]['iteration']

    def test_max_evaluations_stops_the_fit_cleanly(self, caplog) -> None:
        # When
        model = Line()
        x, y, weights = line_data()

        # Then
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.minuit'):
            result = engine_for(model).fit(x=x, y=y, weights=weights, max_evaluations=3)

        # Expect
        assert result.success is False
        assert result.n_evaluations < 50
        assert np.isfinite(model.slope.value)
        assert 'max evaluations reached' in result.message
        assert 'max evaluations reached' in caplog.text
        assert model.slope.error is None
        assert model.intercept.error is None

    def test_tolerance_is_applied(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()

        # Then
        result = engine_for(model).fit(x=x, y=y, weights=weights, tolerance=1e-2)

        # Expect
        assert result.engine_result.tol == 1e-2


class TestMinuitReviewFixes:
    def test_the_model_is_not_evaluated_again_after_the_write_back(self) -> None:
        # When
        x, y, weights = line_data()
        reference = engine_for(Line()).fit(x=x, y=y, weights=weights)
        calls = []

        def late_failure(payload: dict) -> None:
            calls.append(payload['iteration'])
            if len(calls) > reference.n_evaluations:
                raise RuntimeError('a call after the minimization')

        # Then
        result = engine_for(Line()).fit(x=x, y=y, weights=weights, progress_callback=late_failure)

        # Expect
        assert len(calls) == result.n_evaluations
        assert calls[-1] == result.n_evaluations

    @pytest.mark.parametrize('keyword', ['bogus', 'minimizer_kwargs'])
    def test_unexpected_keyword_arguments_raise(self, keyword: str) -> None:
        # When
        model = Line()
        x, y, weights = line_data()

        # Then Expect
        with pytest.raises(TypeError, match=keyword):
            engine_for(model).fit(x=x, y=y, weights=weights, **{keyword: 1})

    def test_large_unweighted_data_converges_with_the_default_tolerance(self) -> None:
        # When
        x = np.linspace(0.0, 10.0, 50)
        invalid = []

        # Then
        for seed in range(20):
            rng = np.random.default_rng(seed)
            y = 1e6 * (2.0 * x + 1.0) + rng.normal(scale=5e4, size=x.size)
            result = engine_for(Line(bounds='none')).fit(x=x, y=y, weights=np.ones_like(x))
            if not result.success:
                invalid.append(seed)

        # Expect
        assert invalid == []


class TestMinuitRawAccess:
    def test_minos_moves_the_parameters_and_fcn_restores_them(self) -> None:
        # When
        model = Line()
        x, y, weights = line_data()
        result = engine_for(model).fit(x=x, y=y, weights=weights)
        fitted = (model.slope.value, model.intercept.value)
        errors = (model.slope.error, model.intercept.error)

        # Then: the cost function of the raw object writes into the parameters while it evaluates
        result.engine_result.minos()
        moved = (model.slope.value, model.intercept.value)
        result.engine_result.fcn(list(result.engine_result.values))

        # Expect
        assert moved != pytest.approx(fitted, abs=1e-3)
        assert (model.slope.value, model.intercept.value) == pytest.approx(fitted)
        assert (model.slope.error, model.intercept.error) == errors
