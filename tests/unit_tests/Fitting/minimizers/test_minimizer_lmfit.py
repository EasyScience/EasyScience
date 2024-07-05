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

class TestLMFit():
    @pytest.fixture
    def minimizer(self):
        minimizer = LMFit(
            obj='obj',
            fit_function='fit_function', 
            method='method'
        )
        return minimizer
    
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
