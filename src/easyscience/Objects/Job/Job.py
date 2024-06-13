#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from abc import ABCMeta
from abc import abstractmethod

from easyscience.Objects.Job.Analysis import AnalysisBase
from easyscience.Objects.Job.Experiment import ExperimentBase
from easyscience.Objects.Job.Theory import TheoryBase
from easyscience.Objects.ObjectClasses import BaseObj


class JobBase(BaseObj, metaclass=ABCMeta):
    """
    This virtual class allows for the creation of technique-specific Job objects.
    """
    def __init__(self, name: str, *args, **kwargs):
        super(JobBase, self).__init__(name, *args, **kwargs)
        self.name = name
        self._theory = None
        self._experiment = None
        self._analysis = None
        self._summary = None
        self._info = None

    """
    JobBase consists of Theory, Experiment, Analysis virtual classes.
    Summary and Info classes are included to store additional information.
    """
    @property
    def theory(self):
        return self._theory

    @theory.setter
    @abstractmethod
    def theory(self, theory: TheoryBase):
        raise NotImplementedError("theory setter not implemented")
   
    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    @abstractmethod
    def experiment(self, experiment: ExperimentBase):
        raise NotImplementedError("experiment setter not implemented")

    @property
    def analysis(self):
        return self._analysis

    @analysis.setter
    @abstractmethod
    def analysis(self, analysis: AnalysisBase):
        raise NotImplementedError("analysis setter not implemented")

    # TODO: extend derived classes to include Summary and Info
    # @property
    # def summary(self):
    #     return self._summary

    # @summary.setter
    # @abstractmethod
    # def summary(self, summary: SummaryBase):
    #     raise NotImplementedError("summary setter not implemented")
    
    # @property
    # def info(self):
    #     return self._info

    # @info.setter
    # @abstractmethod
    # def info(self, info: InfoBase):
    #     raise NotImplementedError("info setter not implemented")

    # @abstractmethod
    # def calculate_theory(self, *args, **kwargs):
    #     raise NotImplementedError("calculate_theory not implemented")
    #     #pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("fit not implemented")
        #pass
    