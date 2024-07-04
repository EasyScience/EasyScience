import pytest
import numpy as np

from unittest.mock import MagicMock

from easyscience.fitting.minimizers.minimizer_base import MinimizerBase


class TestMinimizerBase():
    @pytest.fixture
    def minimizer(self):
        # This avoids the error: TypeError: Can't instantiate abstract class with abstract methods __init__
        MinimizerBase.__abstractmethods__ = set()

        minimizer = MinimizerBase(
            obj='obj',
            fit_function='fit_function', 
            method='method'
        )
        return minimizer
    
    def test_init(self, minimizer: MinimizerBase):
        assert minimizer._object == 'obj'
        assert minimizer._original_fit_function == 'fit_function'
        assert minimizer._method == 'method'
        assert minimizer._cached_pars == {}
        assert minimizer._cached_pars_vals == {}
        assert minimizer._cached_model == None
        assert minimizer._fit_function == None
        assert minimizer._constraints == []
    
    def test_evaluate(self, minimizer: MinimizerBase):
        # When
        minimizer._fit_function = MagicMock(return_value='fit_function_return')
        minimizer._prepare_parameters = MagicMock()

        # Then
        result = minimizer.evaluate('x', minimizer_parameters={'parms_key': 'parms_val'}, kwargs={'kwargs_key': 'kwargs_val'})

        # Expect
        assert result == 'fit_function_return'
        minimizer._fit_function.assert_called_once_with('x', parms_key='parms_val', kwargs={'kwargs_key': 'kwargs_val'})
        minimizer._prepare_parameters.assert_called_once_with({'parms_key': 'parms_val'})

    def test_evaluate_no_fit_function(self, minimizer: MinimizerBase):
        # When
        mock_fit_fucntion = MagicMock()
        minimizer._fit_function = None
        minimizer._prepare_parameters = MagicMock()
        minimizer._generate_fit_function = MagicMock(return_value=mock_fit_fucntion)

        # Then
        minimizer.evaluate('x', minimizer_parameters={'parms_key': 'parms_val'}, kwargs={'kwargs_key': 'kwargs_val'})

        # Expect
        mock_fit_fucntion.assert_called_once_with('x', parms_key='parms_val', kwargs={'kwargs_key': 'kwargs_val'})
        minimizer._prepare_parameters.assert_called_once_with({'parms_key': 'parms_val'})

    def test_evaluate_no_paramters(self, minimizer: MinimizerBase):
        # When
        minimizer._fit_function = MagicMock(return_value='fit_function_return')
        minimizer._prepare_parameters = MagicMock(return_value={'parms_key': 'parms_val'})

        # Then
        minimizer.evaluate('x')

        # Expect
        minimizer._prepare_parameters.assert_called_once_with({})
        minimizer._fit_function.assert_called_once_with('x', parms_key='parms_val')

    def test_evaluate_exception(self, minimizer: MinimizerBase):
        # When
        minimizer_parameters = 'not dict type'

        # Then Expect
        with pytest.raises(AttributeError):
            minimizer.evaluate('x', minimizer_parameters=minimizer_parameters)

    def test_prepare_parameters(self, minimizer: MinimizerBase):
        # When
        parameters = {
            'pa': 1,
            'pb': 2
        }

        minimizer._cached_pars = {
            'a': MagicMock(),
            'b': MagicMock(),
            'c': MagicMock()
        }
        minimizer._cached_pars['a'].raw_value = 3
        minimizer._cached_pars['b'].raw_value = 4
        minimizer._cached_pars['c'].raw_value = 5

        # Then
        parameters = minimizer._prepare_parameters(parameters)

        # Expect
        assert parameters == {
            'pa': 1,
            'pb': 2,
            'pc': 5
        }