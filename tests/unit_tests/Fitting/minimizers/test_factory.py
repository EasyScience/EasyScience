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


@pytest.mark.parametrize('minimizer_name,expected', [('LMFit', AvailableMinimizers.LMFit), ('LMFit_leastsq', AvailableMinimizers.LMFit_leastsq), ('LMFit_powell', AvailableMinimizers.LMFit_powell), ('LMFit_cobyla', AvailableMinimizers.LMFit_cobyla), ])
def test_from_string_to_enum_lmfit(minimizer_name, expected):
    assert from_string_to_enum(minimizer_name) == expected


@pytest.mark.parametrize('minimizer_name,expected', [('Bumps', AvailableMinimizers.Bumps), ('Bumps_simplex', AvailableMinimizers.Bumps_simplex), ('Bumps_newton', AvailableMinimizers.Bumps_newton), ('Bumps_lm', AvailableMinimizers.Bumps_lm)])
def test_from_string_to_enum_bumps(minimizer_name, expected):
    assert from_string_to_enum(minimizer_name) == expected


@pytest.mark.parametrize('minimizer_name,expected', [('DFO', AvailableMinimizers.DFO), ('DFO_leastsq', AvailableMinimizers.DFO_leastsq)])
def test_from_string_to_enum_dfo(minimizer_name, expected):
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