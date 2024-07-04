import pytest

from unittest.mock import MagicMock

import easyscience.fitting.minimizers.minimizer_lmfit

from easyscience.fitting.minimizers.minimizer_lmfit import LMFit
from easyscience.Objects.new_variable import Parameter
from lmfit import Parameter as LMParameter

class TestMinimizerBase():
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