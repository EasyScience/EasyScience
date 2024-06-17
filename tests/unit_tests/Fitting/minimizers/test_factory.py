from easyscience.fitting.minimizers.factory import factory
from easyscience.fitting.minimizers.factory import from_string_to_enum
from easyscience.fitting.minimizers.factory import AvailableMinimizers
from easyscience.fitting.minimizers import MinimizerBase
from unittest.mock import MagicMock
import pytest

class TestFactory:
    def pull_minminizer(self, minimizer: AvailableMinimizers) -> MinimizerBase:
        mock_fit_object = MagicMock()
        mock_fit_function = MagicMock()
        minimizer = factory(minimizer, mock_fit_object, mock_fit_function)
        return minimizer

    @pytest.mark.parametrize('minimizer_method,minimizer_enum', [('leastsq', AvailableMinimizers.LMFit), ('leastsq', AvailableMinimizers.LMFit_leastsq), ('powell', AvailableMinimizers.LMFit_powell), ('cobyla', AvailableMinimizers.LMFit_cobyla)])
    def test_factory_lm_fit(self, minimizer_method, minimizer_enum):
        minimizer = self.pull_minminizer(minimizer_enum)
        assert minimizer._method == minimizer_method
        assert minimizer.wrapping == 'lmfit'

    @pytest.mark.parametrize('minimizer_method,minimizer_enum', [('amoeba', AvailableMinimizers.Bumps), ('amoeba', AvailableMinimizers.Bumps_simplex), ('newton', AvailableMinimizers.Bumps_newton), ('lm', AvailableMinimizers.Bumps_lm)])
    def test_factory_bumps_fit(self, minimizer_method, minimizer_enum):
        minimizer = self.pull_minminizer(minimizer_enum)
        assert minimizer._method == minimizer_method
        assert minimizer.wrapping == 'bumps'

    @pytest.mark.parametrize('minimizer_method,minimizer_enum', [('leastsq', AvailableMinimizers.DFO), ('leastsq', AvailableMinimizers.DFO_leastsq)])
    def test_factory_dfo_fit(self, minimizer_method, minimizer_enum):
        minimizer = self.pull_minminizer(minimizer_enum)
        assert minimizer._method == minimizer_method
        assert minimizer.wrapping == 'dfo'

@pytest.mark.parametrize('minimizer_name,expected', [('lmfit', AvailableMinimizers.LMFit), ('lmfit-leastsq', AvailableMinimizers.LMFit_leastsq), ('lmfit-powell', AvailableMinimizers.LMFit_powell), ('lmfit-cobyla', AvailableMinimizers.LMFit_cobyla), ])
def test_from_string_to_enum_lmfit(self, minimizer_name, expected):
    assert from_string_to_enum(minimizer_name) == expected

@pytest.mark.parametrize('minimizer_name,expected', [('bumps', AvailableMinimizers.Bumps), ('bumps-simplex', AvailableMinimizers.Bumps_simplex), ('bumps-newton', AvailableMinimizers.Bumps_newton), ('bumps-lm', AvailableMinimizers.Bumps_lm)])
def test_from_string_to_enum_bumps(self, minimizer_name, expected):
    assert from_string_to_enum(minimizer_name) == expected

@pytest.mark.parametrize('minimizer_name,expected', [('dfo', AvailableMinimizers.DFO), ('dfo-leastsq', AvailableMinimizers.DFO_leastsq)])
def test_from_string_to_enum_dfo(self, minimizer_name, expected):
    assert from_string_to_enum(minimizer_name) == expected


def test_available_minimizers():
    assert AvailableMinimizers.LMFit
    assert AvailableMinimizers.LMFit_leastsq
    assert AvailableMinimizers.LMFit_powell
    assert AvailableMinimizers.LMFit_cobyla
    assert AvailableMinimizers.Bumps
    assert AvailableMinimizers.Bumps_simplex
    assert AvailableMinimizers.Bumps_newton
    assert AvailableMinimizers.Bumps_lm
    assert AvailableMinimizers.DFO
    assert AvailableMinimizers.DFO_leastsq
    assert len(AvailableMinimizers) == 10