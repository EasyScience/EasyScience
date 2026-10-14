# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easyscience.fitting.minimizers.utils import FitResults


class TestFitResultsRepr:
    def _make_result(self, **overrides):
        r = FitResults()
        r.success = True
        r.x = np.array([1.0, 2.0, 3.0])
        r.y_obs = np.array([1.0, 2.0, 3.0])
        r.y_calc = np.array([1.1, 1.9, 3.05])
        r.y_err = np.array([0.1, 0.1, 0.1])
        r.p = {'pa': 1.234, 'pb': 5.678}
        r.n_evaluations = 42
        r.iterations = 7
        r.minimizer_engine = type('Bumps', (), {'__name__': 'Bumps'})
        for k, v in overrides.items():
            setattr(r, k, v)
        return r

    def test_repr_has_no_covariance_line_by_default(self):
        # When Then
        text = repr(self._make_result())

        # Expect
        assert 'covariance' not in text

    def test_repr_mentions_the_covariance(self):
        # When
        from easyscience import Parameter
        from easyscience.variable.covariance import ParameterCovariance

        parameters = [Parameter(1.0), Parameter(2.0)]
        result = self._make_result()
        result.covariance = ParameterCovariance(parameters, np.eye(2))

        # Then Expect
        assert 'covariance=hessian 2x2' in repr(result)

    def test_repr_contains_success(self):
        r = self._make_result()
        assert 'success=True' in repr(r)

    def test_repr_contains_n_pars_and_n_points(self):
        r = self._make_result()
        text = repr(r)
        assert 'n_pars=2' in text
        assert 'n_points=3' in text

    def test_repr_contains_chi2_values(self):
        r = self._make_result()
        text = repr(r)
        assert 'chi2=' in text
        assert 'reduced_chi2=' in text
        assert 'N/A' not in text

    def test_repr_shows_na_when_chi2_cannot_be_computed(self):
        r = self._make_result(y_err=np.array([0.0, 0.0, 0.0]))
        text = repr(r)
        assert 'chi2=N/A' in text
        assert 'reduced_chi2=N/A' in text

    def test_repr_contains_n_evaluations(self):
        r = self._make_result()
        assert 'n_evaluations=42' in repr(r)

    def test_repr_contains_iterations(self):
        r = self._make_result()
        assert 'iterations=7' in repr(r)

    def test_repr_contains_minimizer_name(self):
        r = self._make_result()
        assert 'minimizer=Bumps' in repr(r)

    def test_repr_minimizer_none(self):
        r = self._make_result(minimizer_engine=None)
        assert 'minimizer=None' in repr(r)

    def test_repr_includes_message_when_set(self):
        r = self._make_result(message='Fit stopped: reached maximum evaluations (3)')
        assert 'Fit stopped: reached maximum evaluations (3)' in repr(r)

    def test_repr_omits_message_when_empty(self):
        r = self._make_result(message='')
        assert 'message' not in repr(r)

    def test_repr_includes_parameters(self):
        r = self._make_result()
        text = repr(r)
        assert 'pa=1.234' in text
        assert 'pb=5.678' in text

    def test_repr_omits_parameters_when_empty(self):
        r = self._make_result(p={})
        assert 'parameters' not in repr(r)

    def test_repr_default_fit_results(self):
        r = FitResults()
        text = repr(r)
        assert 'success=False' in text
        assert 'n_pars=0' in text
        assert 'n_points=0' in text
        assert 'n_evaluations=None' in text
        assert 'iterations=None' in text
        assert 'chi2=N/A' in text
