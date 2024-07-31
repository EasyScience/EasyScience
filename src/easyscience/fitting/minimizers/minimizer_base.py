#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from abc import ABCMeta
from abc import abstractmethod
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

# causes circular import when Parameter is imported
# from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.Variable import Parameter

from ..Constraints import ObjConstraint
from .utils import FitError
from .utils import FitResults

MINIMIZER_PARAMETER_PREFIX = 'p'


class MinimizerBase(metaclass=ABCMeta):
    """
    This template class is the basis for all minimizer engines in `EasyScience`.
    """

    wrapping: str = None

    def __init__(
        self, obj, fit_function: Callable, method: Optional[str] = None
    ):  # todo after constraint changes, add type hint: obj: BaseObj  # noqa: E501
        if method not in self.available_methods():
            raise FitError(f'Method {method} not available in {self.__class__}')
        self._object = obj
        self._original_fit_function = fit_function
        self._method = method
        self._cached_pars: Dict[str, Parameter] = {}
        self._cached_pars_vals: Dict[str, Tuple[float]] = {}
        self._cached_model = None
        self._fit_function = None
        self._constraints = []

    @property
    def all_constraints(self) -> List[ObjConstraint]:
        return [*self._constraints, *self._object._constraints]

    def fit_constraints(self) -> List[ObjConstraint]:
        return self._constraints

    def set_fit_constraint(self, constraints: List[ObjConstraint]):
        self._constraints = constraints

    def add_fit_constraint(self, constraint: ObjConstraint):
        self._constraints.append(constraint)

    def remove_fit_constraint(self, index: int) -> None:
        del self._constraints[index]

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        model: Optional[Callable] = None,
        parameters: Optional[Parameter] = None,
        method: Optional[str] = None,
        **kwargs,
    ) -> FitResults:
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

    def evaluate(self, x: np.ndarray, minimizer_parameters: dict[str, float] = None, **kwargs) -> np.ndarray:
        """
        Evaluate the fit function for values of x. Parameters used are either the latest or user supplied.
        If the parameters are user supplied, it must be in a dictionary of {'parameter_name': parameter_value,...}.

        :param x: x values for which the fit function will be evaluated
        :type x:  np.ndarray
        :param minimizer_parameters: Dictionary of parameters which will be used in the fit function. They must be in a dictionary
         of {'parameter_name': parameter_value,...}
        :type minimizer_parameters: dict
        :param kwargs: additional arguments
        :return: y values calculated at points x for a set of parameters.
        :rtype: np.ndarray
        """  # noqa: E501
        if minimizer_parameters is None:
            minimizer_parameters = {}
        if not isinstance(minimizer_parameters, dict):
            raise TypeError('minimizer_parameters must be a dictionary')

        if self._fit_function is None:
            # This will also generate self._cached_pars
            self._fit_function = self._generate_fit_function()

        minimizer_parameters = self._prepare_parameters(minimizer_parameters)

        return self._fit_function(x, **minimizer_parameters, **kwargs)

    @abstractmethod
    def convert_to_pars_obj(self, par_list: Optional[Union[list]] = None):
        """
        Create an engine compatible container with the `Parameters` converted from the base object.

        :param par_list: If only a single/selection of parameter is required. Specify as a list
        :type par_list: List[str]
        :return: engine Parameters compatible object
        """

    @abstractmethod
    def available_methods(self) -> List[str]:
        """
        Return a list of available methods for the engine.

        :return: List of available methods
        :rtype: List[str]
        """

    @staticmethod
    @abstractmethod
    def convert_to_par_object(obj):  # todo after constraint changes, add type hint: obj: BaseObj
        """
        Convert an `EasyScience.Objects.Base.Parameter` object to an engine Parameter object.
        """

    def _prepare_parameters(self, parameters: dict[str, float]) -> dict[str, float]:
        """
        Prepare the parameters for the minimizer.

        :param parameters: Dict of parameters for the minimizer with names as keys.
        """
        pars = self._cached_pars

        # TODO clean when full move to new_variable
        from easyscience.Objects.new_variable import Parameter as NewParameter

        for name, item in pars.items():
            parameter_name = MINIMIZER_PARAMETER_PREFIX + str(name)
            if parameter_name not in parameters.keys():
                # TODO clean when full move to new_variable
                if isinstance(item, NewParameter):
                    parameters[parameter_name] = item.value
                else:
                    parameters[parameter_name] = item.raw_value
        return parameters

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
