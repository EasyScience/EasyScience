#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience
import functools
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from .available_minimizers import AvailableMinimizers
from .available_minimizers import from_string_to_enum
from .minimizers import FitResults
from .minimizers import MinimizerBase
from .minimizers.factory import factory

DEFAULT_MINIMIZER = AvailableMinimizers.LMFit_leastsq


class Fitter:
    """
    Fitter is a class which makes it possible to undertake fitting utilizing one of the supported minimizers.
    """

    def __init__(self, fit_object, fit_function: Callable):
        self._fit_object = fit_object
        self._fit_function = fit_function
        self._dependent_dims: int = None
        self._tolerance: float = None
        self._max_evaluations: int = None

        self._minimizer: MinimizerBase = None  # set in _update_minimizer
        self._enum_current_minimizer: AvailableMinimizers = None  # set in _update_minimizer
        self._update_minimizer(DEFAULT_MINIMIZER)

    def fit_constraints(self) -> list:
        return self._minimizer.fit_constraints()

    def add_fit_constraint(self, constraint) -> None:
        self._minimizer.add_fit_constraint(constraint)

    def remove_fit_constraint(self, index: int) -> None:
        self._minimizer.remove_fit_constraint(index)

    def make_model(self, pars=None) -> Callable:
        return self._minimizer.make_model(pars)

    def evaluate(self, pars=None) -> np.ndarray:
        return self._minimizer.evaluate(pars)

    def convert_to_pars_obj(self, pars) -> object:
        return self._minimizer.convert_to_pars_obj(pars)

    # TODO: remove this method when we are ready to adjust the dependent products
    def initialize(self, fit_object, fit_function: Callable) -> None:
        """
        Set the model and callable in the calculator interface.

        :param fit_object: The EasyScience model object
        :param fit_function: The function to be optimized against.
        """
        self._fit_object = fit_object
        self._fit_function = fit_function
        self._update_minimizer(DEFAULT_MINIMIZER)

    # TODO: remove this method when we are ready to adjust the dependent products
    def create(self, minimizer_enum: Union[AvailableMinimizers, str] = DEFAULT_MINIMIZER) -> None:
        """
        Create the required minimizer.
        :param minimizer_enum: The enum of the minimization engine to create.
        """
        if isinstance(minimizer_enum, str):
            print(f'minimizer should be set with enum {minimizer_enum}')
            minimizer_enum = from_string_to_enum(minimizer_enum)
        self._update_minimizer(minimizer_enum)

    def switch_minimizer(self, minimizer_enum: Union[AvailableMinimizers, str]) -> None:
        """
        Switch minimizer and initialize.
        :param minimizer_enum: The enum of the minimizer to create and instantiate.
        """
        if isinstance(minimizer_enum, str):
            print(f'minimizer should be set with enum {minimizer_enum}')
            minimizer_enum = from_string_to_enum(minimizer_enum)

        constraints = self._minimizer.fit_constraints()
        self._update_minimizer(minimizer_enum)
        self._minimizer.set_fit_constraint(constraints)

    def _update_minimizer(self, minimizer_enum: AvailableMinimizers) -> None:
        self._minimizer = factory(minimizer_enum=minimizer_enum, fit_object=self._fit_object, fit_function=self.fit_function)
        self._enum_current_minimizer = minimizer_enum

    @property
    def available_minimizers(self) -> List[str]:
        """
        Get a list of the names of available fitting minimizers

        :return: List of available fitting minimizers
        :rtype: List[str]
        """
        return [minimize.name for minimize in AvailableMinimizers]

    @property
    def minimizer(self) -> MinimizerBase:
        """
        Get the current fitting minimizer object.

        :return:
        :rtype: MinimizerBase
        """
        return self._minimizer

    @property
    def tolerance(self) -> float:
        """
        Get the tolerance for the minimizer.

        :return: Tolerance for the minimizer
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        """
        Set the tolerance for the minimizer.

        :param tolerance: Tolerance for the minimizer
        """
        self._tolerance = tolerance

    @property
    def max_evaluations(self) -> int:
        """
        Get the maximal number of evaluations for the minimizer.

        :return: Maximal number of steps for the minimizer
        """
        return self._max_evaluations

    @max_evaluations.setter
    def max_evaluations(self, max_evaluations: int) -> None:
        """
        Set the maximal number of evaluations for the minimizer.

        :param max_evaluations: Maximal number of steps for the minimizer
        """
        self._max_evaluations = max_evaluations

    @property
    def fit_function(self) -> Callable:
        """
        The raw fit function that the optimizer will call (no wrapping)
        :return: Raw fit function
        """
        return self._fit_function

    @fit_function.setter
    def fit_function(self, fit_function: Callable) -> None:
        """
        Set the raw fit function to a new one.
        :param fit_function: New fit function
        :return: None
        """
        self._fit_function = fit_function
        self._update_minimizer(self._enum_current_minimizer)

    @property
    def fit_object(self):
        """
        The EasyScience object which will be used as a model
        :return: EasyScience Model
        """
        return self._fit_object

    @fit_object.setter
    def fit_object(self, fit_object) -> None:
        """
        Set the EasyScience object which wil be used as a model
        :param fit_object: New EasyScience object
        :return: None
        """
        self._fit_object = fit_object
        self._update_minimizer(self._enum_current_minimizer)

    def _fit_function_wrapper(self, real_x=None, flatten: bool = True) -> Callable:
        """
        Simple fit function which injects the real X (independent) values into the
        optimizer function. This will also flatten the results if needed.
        :param real_x: Independent x parameters to be injected
        :param flatten: Should the result be a flat 1D array?
        :return: Wrapped optimizer function.
        """
        fun = self._fit_function

        @functools.wraps(fun)
        def wrapped_fit_function(x, **kwargs):
            if real_x is not None:
                x = real_x
            dependent = fun(x, **kwargs)
            if flatten:
                dependent = dependent.flatten()
            return dependent

        return wrapped_fit_function

    @property
    def fit(self) -> Callable:
        """
        Property which wraps the current `fit` function from the fitting interface. This property return a wrapped fit
        function which converts the input data into the correct shape for the optimizer, wraps the fit function to
        re-constitute the independent variables and once the fit is completed, reshape the inputs to those expected.
        """

        @functools.wraps(self._minimizer.fit)
        def inner_fit_callable(
            x: np.ndarray,
            y: np.ndarray,
            weights: Optional[np.ndarray] = None,
            vectorized: bool = False,
            **kwargs,
        ) -> FitResults:
            """
            This is a wrapped callable which performs the actual fitting. It is split into
            3 sections, PRE/ FIT/ POST.
            - PRE = Reshaping the input data into the correct dimensions for the optimizer
            - FIT = Wrapping the fit function and performing the fit
            - POST = Reshaping the outputs so it is coherent with the inputs.
            """
            # Precompute - Reshape all independents into the correct dimensionality
            x_fit, x_new, y_new, weights, dims = self._precompute_reshaping(x, y, weights, vectorized)
            self._dependent_dims = dims

            # Fit
            fit_fun_org = self._fit_function
            fit_fun_wrap = self._fit_function_wrapper(x_new, flatten=True)  # This should be wrapped.

            # We change the  fit function, so have to  reset constraints
            constraints = self._minimizer.fit_constraints()
            self.fit_function = fit_fun_wrap
            self._minimizer.set_fit_constraint(constraints)
            f_res = self._minimizer.fit(
                x_fit,
                y_new,
                weights=weights,
                tolerance=self._tolerance,
                max_evaluations=self._max_evaluations,
                **kwargs,
            )

            # Postcompute
            fit_result = self._post_compute_reshaping(f_res, x, y)
            # Reset the function and constrains
            self.fit_function = fit_fun_org
            self._minimizer.set_fit_constraint(constraints)
            return fit_result

        return inner_fit_callable

    @staticmethod
    def _precompute_reshaping(
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray],
        vectorized: bool,
    ):
        """
        Check the dimensions of the inputs and reshape if necessary.
        :param x: ND matrix of dependent points
        :param y: N-1D matrix of independent points
        :param kwargs: Additional key-word arguments
        :return:
        """
        # Make sure that they are np arrays
        x_new = np.array(x)
        y_new = np.array(y)
        # Get the shape
        x_shape = x_new.shape
        # Check if the x data is 1D
        if len(x_shape) > 1:
            # It is ND data
            # Check if the data is vectorized. i.e. should x be [NxMx...x Ndims]
            if vectorized:
                # Assert that the shapes are the same
                if np.all(x_shape[:-1] != y_new.shape):
                    raise ValueError('The shape of the x and y data must be the same')
                # If so do nothing but note that the data is vectorized
                # x_shape = (-1,) # Should this be done?
            else:
                # Assert that the shapes are the same
                if np.prod(x_new.shape[:-1]) != y_new.size:
                    raise ValueError('The number of elements in x and y data must be the same')
                # Reshape the data to be [len(NxMx..), Ndims] i.e. flatten to columns
                x_new = x_new.reshape(-1, x_shape[-1], order='F')
        else:
            # Assert that the shapes are the same
            if np.all(x_shape != y_new.shape):
                raise ValueError('The shape of the x and y data must be the same')
            # It is 1D data
            x_new = x.flatten()
        # The optimizer needs a 1D array, flatten the y data
        y_new = y_new.flatten()
        if weights is not None:
            weights = np.array(weights).flatten()
        # Make a 'dummy' x array for the fit function
        x_for_fit = np.array(range(y_new.size))
        return x_for_fit, x_new, y_new, weights, x_shape

    @staticmethod
    def _post_compute_reshaping(fit_result: FitResults, x: np.ndarray, y: np.ndarray) -> FitResults:
        """
        Reshape the output of the fitter into the correct dimensions.
        :param fit_result: Output from the fitter
        :param x: Input x independent
        :param y: Input y dependent
        :return: Reshaped Fit Results
        """
        fit_result.x = x
        fit_result.y_obs = y
        fit_result.y_calc = np.reshape(fit_result.y_calc, y.shape)
        fit_result.y_err = np.reshape(fit_result.y_err, y.shape)
        return fit_result
