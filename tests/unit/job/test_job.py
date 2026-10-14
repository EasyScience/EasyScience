# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.fitting.calculators.interface_factory import InterfaceFactoryTemplate
from easyscience.job import AnalysisBase
from easyscience.job import ExperimentBase
from easyscience.job import JobBase
from easyscience.job import TheoreticalModelBase


@pytest.fixture(autouse=True)
def clear():
    global_object.map._clear()
    yield
    global_object.map._clear()


class Analysis(AnalysisBase):
    def calculate_theory(self, x, **kwargs):
        return super().calculate_theory(x, **kwargs)

    def fit(self, x, y, e, **kwargs):
        return super().fit(x, y, e, **kwargs)


class Job(JobBase):
    @JobBase.theoretical_model.setter
    def theoretical_model(self, theory):
        JobBase.theoretical_model.fset(self, theory)

    @JobBase.experiment.setter
    def experiment(self, experiment):
        JobBase.experiment.fset(self, experiment)

    @JobBase.analysis.setter
    def analysis(self, analysis):
        JobBase.analysis.fset(self, analysis)

    def calculate_theory(self, *args, **kwargs):
        return super().calculate_theory(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)


class CalculatorA:
    name = 'A'

    def create(self, model):
        return []

    def fit_func(self, *args, **kwargs):
        return None


class CalculatorB(CalculatorA):
    name = 'B'


@pytest.mark.parametrize('cls', [Analysis, ExperimentBase, Job, TheoreticalModelBase])
def test_job_classes_are_model_bases(cls):
    obj = cls(display_name='name', unique_name=f'{cls.__name__}_unique')

    assert isinstance(obj, ModelBase)
    assert obj.display_name == 'name'
    assert obj.unique_name == f'{cls.__name__}_unique'
    assert obj.get_all_variables() == []


class TestAnalysisBase:
    def test_defaults(self):
        analysis = Analysis(display_name='analysis')

        assert analysis.interface is None
        assert analysis.calculator is None
        assert analysis.minimizer is None
        assert str(analysis) == 'Analysis: analysis'

    def test_calculator_comes_from_interface(self):
        interface = InterfaceFactoryTemplate([CalculatorA, CalculatorB])

        analysis = Analysis(interface=interface)

        assert analysis.calculator == 'A'

    def test_calculator_setter_switches_interface(self):
        interface = InterfaceFactoryTemplate([CalculatorA, CalculatorB])
        analysis = Analysis(interface=interface)

        analysis.calculator = 'B'

        assert interface.current_interface_name == 'B'

    def test_minimizer_setter(self):
        analysis = Analysis()
        minimizer = object()

        analysis.minimizer = minimizer

        assert analysis.minimizer is minimizer

    def test_abstract_methods_raise(self):
        analysis = Analysis()
        x = np.array([1.0])

        with pytest.raises(NotImplementedError, match='calculate_theory'):
            analysis.calculate_theory(x)
        with pytest.raises(NotImplementedError, match='fit'):
            analysis.fit(x, x, x)

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            AnalysisBase()


class TestExperimentBase:
    def test_str(self):
        assert str(ExperimentBase(display_name='experiment')) == 'Experiment: experiment'

    def test_str_falls_back_to_unique_name(self):
        experiment = ExperimentBase(unique_name='exp_1')

        assert str(experiment) == 'Experiment: exp_1'


class TestTheoreticalModelBase:
    def test_str_not_implemented(self):
        with pytest.raises(NotImplementedError):
            str(TheoreticalModelBase())


class TestJobBase:
    def test_defaults(self):
        job = Job(display_name='job')

        assert job.theoretical_model is None
        assert job.experiment is None
        assert job.analysis is None

    @pytest.mark.parametrize(
        ('attribute', 'value_factory'),
        [
            ('theoretical_model', TheoreticalModelBase),
            ('experiment', ExperimentBase),
            ('analysis', Analysis),
        ],
    )
    def test_base_setters_are_abstract(self, attribute, value_factory):
        job = Job()

        with pytest.raises(NotImplementedError, match='setter not implemented'):
            setattr(job, attribute, value_factory())

    def test_abstract_methods_raise(self):
        job = Job()

        with pytest.raises(NotImplementedError, match='calculate_theory'):
            job.calculate_theory()
        with pytest.raises(NotImplementedError, match='fit'):
            job.fit()

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            JobBase()
