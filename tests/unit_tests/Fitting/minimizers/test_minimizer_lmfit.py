import pytest

from inspect import Parameter as InspectParameter
from inspect import Signature
from inspect import _empty
from unittest.mock import MagicMock

import easyscience.fitting.minimizers.minimizer_lmfit

from easyscience.fitting.minimizers.minimizer_lmfit import LMFit
from easyscience.fitting.minimizers.minimizer_lmfit import _wrap_to_lm_signature
from easyscience.Objects.new_variable import Parameter
from lmfit import Parameter as LMParameter
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.fitting.minimizers.utils import FitError


class TestLMFit():
    @pytest.fixture
    def minimizer(self):
        minimizer = LMFit(
            obj='obj',
            fit_function='fit_function', 
            method='least_squares'
        )
        return minimizer

    def test_init_exception(self):
        with pytest.raises(FitError):
            LMFit(
                obj='obj',
                fit_function='fit_function', 
                method='method'
            )

    def test_make_model(self, minimizer: LMFit, monkeypatch):
        # When
        mock_lm_model = MagicMock()
        mock_LMModel = MagicMock(return_value=mock_lm_model)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "LMModel", mock_LMModel)
        minimizer._generate_fit_function = MagicMock(return_value='model')
        mock_parm_1 = MagicMock(LMParameter)
        mock_parm_1.value = 1.0
        mock_parm_1.min = -10.0
        mock_parm_1.max = 10.0
        mock_parm_2 = MagicMock(LMParameter)
        mock_parm_2.value = 2.0
        mock_parm_2.min = -20.0
        mock_parm_2.max = 20.0
        pars = {'key_1': mock_parm_1, 'key_2': mock_parm_2}

        # Then
        model = minimizer.make_model(pars=pars)
        
        # Expect
        minimizer._generate_fit_function.assert_called_once_with()
        mock_LMModel.assert_called_once_with('model', independent_vars=['x'], param_names=['pkey_1', 'pkey_2'])
        mock_lm_model.set_param_hint.assert_called_with('pkey_2', value=2.0, min=-20.0, max=20.0)
        assert mock_lm_model.set_param_hint.call_count == 2
        assert model == mock_lm_model

    def test_make_model_no_pars(self, minimizer: LMFit, monkeypatch):
        # When
        mock_lm_model = MagicMock()
        mock_LMModel = MagicMock(return_value=mock_lm_model)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "LMModel", mock_LMModel)
        minimizer._generate_fit_function = MagicMock(return_value='model')
        mock_parm_1 = MagicMock(Parameter)
        mock_parm_1.value = 1.0
        mock_parm_1.min = -10.0
        mock_parm_1.max = 10.0
        mock_parm_2 = MagicMock(Parameter)
        mock_parm_2.value = 2.0
        mock_parm_2.min = -20.0
        mock_parm_2.max = 20.0
        minimizer._cached_pars = {'key_1': mock_parm_1, 'key_2': mock_parm_2}

        # Then
        model = minimizer.make_model()
        
        # Expect
        minimizer._generate_fit_function.assert_called_once_with()
        mock_LMModel.assert_called_once_with('model', independent_vars=['x'], param_names=['pkey_1', 'pkey_2'])
        mock_lm_model.set_param_hint.assert_called_with('pkey_2', value=2.0, min=-20.0, max=20.0)
        assert mock_lm_model.set_param_hint.call_count == 2
        assert model == mock_lm_model

    def test_generate_fit_function_signatur(self, minimizer: LMFit, monkeypatch):
        # When
        mock_parm_1 = MagicMock(Parameter)
        mock_parm_1.value = 1.0
        mock_parm_1.error = 0.1
        mock_parm_2 = MagicMock(Parameter)
        mock_parm_2.value = 2.0
        mock_parm_2.error = 0.2
        mock_obj = MagicMock(BaseObj)
        mock_obj.get_fit_parameters = MagicMock(return_value=[mock_parm_1, mock_parm_2])
        minimizer._object = mock_obj

        mock_name_converter = MagicMock()
        mock_name_converter.get_key = MagicMock(side_effect=['key_1_converted', 'key_2_converted'])
        mock_NameConverter = MagicMock(return_value=mock_name_converter)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "NameConverter", mock_NameConverter)

        mock_wrap_to_lm_signature = MagicMock(return_value='signature')
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "_wrap_to_lm_signature", mock_wrap_to_lm_signature)

        minimizer._original_fit_function = MagicMock(return_value='fit_function_return')

        # Then
        fit_function = minimizer._generate_fit_function()

        # Expect
        assert fit_function.__signature__ == 'signature'

    def test_generate_fit_function_lm_fit_function(self, minimizer: LMFit, monkeypatch):
        # When
        mock_parm_1 = MagicMock(Parameter)
        mock_parm_1.value = 1.0
        mock_parm_1.error = 0.1
        mock_parm_2 = MagicMock(Parameter)
        mock_parm_2.value = 2.0
        mock_parm_2.error = 0.2
        mock_obj = MagicMock(BaseObj)
        mock_obj.get_fit_parameters = MagicMock(return_value=[mock_parm_1, mock_parm_2])
        minimizer._object = mock_obj

        mock_name_converter = MagicMock()
        mock_name_converter.get_key = MagicMock(side_effect=['key_1_converted', 'key_2_converted'])
        mock_NameConverter = MagicMock(return_value=mock_name_converter)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "NameConverter", mock_NameConverter)

        mock_wrap_to_lm_signature = MagicMock(return_value='signature')
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "_wrap_to_lm_signature", mock_wrap_to_lm_signature)

        minimizer._original_fit_function = MagicMock(return_value='fit_function_return')

        mock_constraint = MagicMock()
        minimizer.fit_constraints = MagicMock(return_value=[mock_constraint])

        fit_function = minimizer._generate_fit_function()

        # Then
        result = fit_function(1)

        # Expect
        result == 'fit_function_return'
        mock_constraint.assert_called_once_with()

    def test_fit(self, minimizer: LMFit):
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer.make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        result = minimizer.fit(x=1.0, y=2.0)

        # Expect
        assert result == 'gen_fit_results'
        mock_model.fit.assert_called_once_with(2.0, x=1.0, weights=0.7071067811865475, method='least_squares')
        minimizer.make_model.assert_called_once_with()
        minimizer._set_parameter_fit_result.assert_called_once_with('fit', False)
        minimizer._gen_fit_results.assert_called_once_with('fit')

    def test_fit_model(self, minimizer: LMFit):
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer.make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        minimizer.fit(x=1.0, y=2.0, model=mock_model)

        # Expect
        mock_model.fit.assert_called_once_with(2.0, x=1.0, weights=0.7071067811865475, method='least_squares')
        minimizer.make_model.assert_not_called()

    def test_fit_method(self, minimizer: LMFit):
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer.make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')
        minimizer.available_methods = MagicMock(return_value=['method_passed'])

        # Then
        minimizer.fit(x=1.0, y=2.0, method='method_passed')

        # Expect
        mock_model.fit.assert_called_once_with(2.0, x=1.0, weights=0.7071067811865475, method='method_passed')
        minimizer.available_methods.assert_called_once_with()

    def test_fit_kwargs(self, minimizer: LMFit):
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer.make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        minimizer.fit(x=1.0, y=2.0, minimizer_kwargs={'minimizer_key': 'minimizer_val'}, engine_kwargs={'engine_key': 'engine_val'})

        # Expect
        mock_model.fit.assert_called_once_with(2.0, x=1.0, weights=0.7071067811865475, method='least_squares', fit_kws={'minimizer_key': 'minimizer_val'}, engine_key='engine_val')

    def test_fit_exception(self, minimizer: LMFit):
        # When
        minimizer.make_model = MagicMock(side_effect=Exception('Exception'))
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then Expect
        with pytest.raises(FitError):
            minimizer.fit(x=1.0, y=2.0)
        
def test_wrap_fit_function():
    # When
    mock_parm_1 = MagicMock(Parameter)
    mock_parm_1.value = 1.0
    mock_parm_2 = MagicMock(Parameter)
    mock_parm_2.value = 2.0
    pars = {1: mock_parm_1, 2: mock_parm_2}


    # Then
    signature = _wrap_to_lm_signature(pars)
    
    # Expect
    wrapped_parameters = [
        InspectParameter('x', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty),
        InspectParameter('p1', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty, default=1.0),
        InspectParameter('p2', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty, default=2.0)
    ]
    expected_signature = Signature(wrapped_parameters)
    assert signature == expected_signature
