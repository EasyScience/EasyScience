#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from abc import ABCMeta
from abc import abstractmethod
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np

from .utils import FitResults


class MinimizerBase(metaclass=ABCMeta):
    """
    This template class is the basis for all fitting engines in `EasyScience`.
    """

    wrapping: str = None

    def __init__(self, obj, fit_function: Callable, method: Optional[str] = None):
        self._object = obj
        self._original_fit_function = fit_function
        self._method = method
        self._cached_pars = {}
        self._cached_pars_vals = {}
        self._cached_model = None
        self._fit_function = None
        self._constraints = []

    @property
    def all_constraints(self) -> list:
        return [*self._constraints, *self._object._constraints]

    def fit_constraints(self) -> list:
        return self._constraints

    def set_fit_constraint(self, constraints):
        self._constraints = constraints

    def add_fit_constraint(self, constraint):
        self._constraints.append(constraint)

    def remove_fit_constraint(self, index: int):
        del self._constraints[index]

    @abstractmethod
    def make_model(self, pars=None):
        """
        Generate an engine model from the supplied `fit_function` and parameters in the base object.

        :return: Callable model
        """

    @abstractmethod
    def _generate_fit_function(self) -> Callable:
        """
        Using the user supplied `fit_function`, wrap it in such a way we can update `Parameter` on
        iterations.
        """

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[Union[np.ndarray]] = None,
        model=None,
        parameters=None,
        method=None,
        **kwargs,
    ):
        """
        Perform a fit using the  engine.

        :param x: points to be calculated at
        :type x: np.ndarray
        :param y: measured points
        :type y: np.ndarray
        :param weights: Weights for supplied measured points
        :type weights: np.ndarray
        :param model: Optional Model which is being fitted to
        :param parameters: Optional parameters for the fit
        :param method: method for the minimizer to use.
        :type method: str
        :param kwargs: Additional arguments for the fitting function.
        :return: Fit results
        """

    def evaluate(self, x: np.ndarray, parameters: dict = None, **kwargs) -> np.ndarray:
        """
        Evaluate the fit function for values of x. Parameters used are either the latest or user supplied.
        If the parameters are user supplied, it must be in a dictionary of {'parameter_name': parameter_value,...}.

        :param x: x values for which the fit function will be evaluated
        :type x:  np.ndarray
        :param parameters: Dictionary of parameters which will be used in the fit function. They must be in a dictionary
         of {'parameter_name': parameter_value,...}
        :type parameters: dict
        :param kwargs: additional arguments
        :return: y values calculated at points x for a set of parameters.
        :rtype: np.ndarray
        """
        if self._fit_function is None:
            # This will also generate self._cached_pars
            self._fit_function = self._generate_fit_function()

        if not isinstance(parameters, (dict, type(None))):
            raise AttributeError

        pars = self._cached_pars
        new_parameters = parameters
        if new_parameters is None:
            new_parameters = {}
        for name, item in pars.items():
            fit_name = 'p' + str(name)
            if fit_name not in new_parameters.keys():
                new_parameters[fit_name] = item.raw_value

        return self._fit_function(x, **new_parameters, **kwargs)

    @abstractmethod
    def convert_to_pars_obj(self, par_list: Optional[Union[list]] = None):
        """
        Create an engine compatible container with the `Parameters` converted from the base object.

        :param par_list: If only a single/selection of parameter is required. Specify as a list
        :type par_list: List[str]
        :return: engine Parameters compatible object
        """

    @staticmethod
    @abstractmethod
    def convert_to_par_object(obj):
        """
        Convert an `EasyScience.Objects.Base.Parameter` object to an engine Parameter object.
        """

    @abstractmethod
    def _set_parameter_fit_result(self, fit_result):
        """
        Update parameters to their final values and assign a std error to them.

        :param fit_result: Fit object which contains info on the fit
        :return: None
        :rtype: noneType
        """

    @abstractmethod
    def _gen_fit_results(self, fit_results, **kwargs) -> 'FitResults':
        """
        Convert fit results into the unified `FitResults` format.

        :param fit_result: Fit object which contains info on the fit
        :return: fit results container
        :rtype: FitResults
        """

    @staticmethod
    def _error_from_jacobian(jacobian: np.ndarray, residuals: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        from scipy import stats

        JtJi = np.linalg.inv(np.dot(jacobian.T, jacobian))
        # 1.96 is a 95% confidence value
        error_matrix = np.dot(
            JtJi,
            np.dot(jacobian.T, np.dot(np.diag(residuals**2), np.dot(jacobian, JtJi))),
        )

        z = 1 - ((1 - confidence) / 2)
        z = stats.norm.pdf(z)
        error_matrix = z * np.sqrt(error_matrix)
        return error_matrix
