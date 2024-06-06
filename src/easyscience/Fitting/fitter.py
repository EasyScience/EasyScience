__author__ = 'github.com/wardsimon'
__version__ = '0.0.1'

import functools

#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience
from abc import ABCMeta
from types import FunctionType
from typing import Callable
from typing import List
from typing import Optional
from typing import TypeVar

import numpy as np

import easyscience.Fitting.minimizers as minimizers
from easyscience import default_fitting_engine

from .minimizers import FitResults
from .minimizers import FittingBase

_C = TypeVar('_C', bound=ABCMeta)
_M = TypeVar('_M', bound=FittingBase)


class Fitter:
    """
    Wrapper to the fitting engines
    """

    def __init__(self, fit_object=None, fit_function: Optional[Callable] = None):
        self._fit_object = fit_object
        self._fit_function = fit_function
        self._dependent_dims = None

        can_initialize = False
        # We can only proceed if both obj and func are not None
        if (fit_object is not None) & (fit_function is not None):
            can_initialize = True
        else:
            if (fit_object is not None) or (fit_function is not None):
                raise AttributeError

        self._engines: List[_C] = minimizers.engines
        self._current_engine: _C = None
        self.__engine_obj: _M = None
        self._is_initialized: bool = False
        self.create()

        fit_methods = [
            x
            for x, y in FittingBase.__dict__.items()
            if (isinstance(y, FunctionType) and not x.startswith('_')) and x != 'fit'
        ]
        for method_name in fit_methods:
            setattr(self, method_name, self.__pass_through_generator(method_name))

        if can_initialize:
            self.__initialize()

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

    def initialize(self, fit_object, fit_function: Callable):
        """
        Set the model and callable in the calculator interface.

        :param fit_object: The EasyScience model object
        :param fit_function: The function to be optimized against.
        :return: None
        """
        self._fit_object = fit_object
        self._fit_function = fit_function
        self.__initialize()

    def __initialize(self):
        """
        The real initialization. Setting the optimizer object properly
        :return: None
        """
        self.__engine_obj = self._current_engine(self._fit_object, self.fit_function)
        self._is_initialized = True

    def create(self, engine_name: str = default_fitting_engine):
        """
        Create a backend optimization engine.
        :param engine_name: The label of the optimization engine to create.
        :return: None
        """
        engines = self.available_engines
        if engine_name in engines:
            self._current_engine = self._engines[engines.index(engine_name)]
            self._is_initialized = False
        else:
            raise AttributeError(f"The supplied optimizer engine '{engine_name}' is unknown.")

    def switch_engine(self, engine_name: str):
        """
        Switch backend optimization engines and initialize.
        :param engine_name: The label of the optimization engine to create and instantiate.
        :return: None
        """
        # There isn't any state to carry over
        if not self._is_initialized:
            raise ReferenceError('The fitting engine must be initialized before switching')
        # Constrains are not carried over. Do it manually.
        constraints = self.__engine_obj._constraints
        self.create(engine_name)
        self.__initialize()
        self.__engine_obj._constraints = constraints

    @property
    def available_engines(self) -> List[str]:
        """
        Get a list of the names of available fitting engines

        :return: List of available fitting engines
        :rtype: List[str]
        """
        if minimizers.engines is None:
            raise ImportError('There are no available fitting engines. Install `lmfit` and/or `bumps`')
        return [engine.name for engine in minimizers.engines]

    @property
    def can_fit(self) -> bool:
        """
        Can a fit be performed. i.e has the object been created properly

        :return: Can a fit be performed
        :rtype: bool
        """
        return self._is_initialized

    @property
    def current_engine(self) -> _C:
        """
        Get the class object of the current fitting engine.

        :return: Class of the current fitting engine (based on the `FittingTemplate` class)
        :rtype: _T
        """
        return self._current_engine

    @property
    def engine(self) -> _M:
        """
        Get the current fitting engine object.

        :return:
        :rtype: _M
        """
        return self.__engine_obj

    @property
    def fit_function(self) -> Callable:
        """
        The raw fit function that the  optimizer will call (no wrapping)
        :return: Raw fit function
        """
        return self._fit_function

    @fit_function.setter
    def fit_function(self, fit_function: Callable):
        """
        Set the raw fit function to a new one.
        :param fit_function: New fit function
        :return: None
        """
        self._fit_function = fit_function
        self.__initialize()

    @property
    def fit_object(self):
        """
        The EasyScience object which will be used as a model
        :return: EasyScience Model
        """
        return self._fit_object

    @fit_object.setter
    def fit_object(self, fit_object):
        """
        Set the EasyScience object which wil be used as a model
        :param fit_object: New EasyScience object
        :return: None
        """
        self._fit_object = fit_object
        self.__initialize()

    def __pass_through_generator(self, name: str):
        """
        Attach the attributes of the calculator template to the current fitter instance.
        :param name: Attribute name to attach
        :return: Wrapped calculator interface object.
        """
        obj = self

        def inner(*args, **kwargs):
            if not obj.can_fit:
                raise ReferenceError('The fitting engine must first be initialized')
            func = getattr(obj.engine, name, None)
            if func is None:
                raise ValueError('The fitting engine does not have the attribute "{}"'.format(name))
            return func(*args, **kwargs)

        return inner

    @property
    def fit(self) -> Callable:
        """
        Property which wraps the current `fit` function from the fitting interface. This property return a wrapped fit
        function which converts the input data into the correct shape for the optimizer, wraps the fit function to
        re-constitute the independent variables and once the fit is completed, reshape the inputs to those expected.
        """

        @functools.wraps(self.engine.fit)
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
            # Check to see if we can perform a fit
            if not self.can_fit:
                raise ReferenceError('The fitting engine must first be initialized')

            # Precompute - Reshape all independents into the correct dimensionality
            x_fit, x_new, y_new, weights, dims, kwargs = self._precompute_reshaping(x, y, weights, vectorized, kwargs)
            self._dependent_dims = dims

            # Fit
            fit_fun = self._fit_function
            fit_fun_wrap = self._fit_function_wrapper(x_new, flatten=True)  # This should be wrapped.

            # We change the  fit function, so have to  reset constraints
            constraints = self.__engine_obj._constraints
            self.fit_function = fit_fun_wrap
            self.__engine_obj._constraints = constraints
            f_res = self.engine.fit(x_fit, y_new, weights=weights, **kwargs)

            # Postcompute
            fit_result = self._post_compute_reshaping(f_res, x, y, weights)
            # Reset the function and constrains
            self.fit_function = fit_fun
            self.__engine_obj._constraints = constraints
            return fit_result

        return inner_fit_callable

    @staticmethod
    def _precompute_reshaping(
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray],
        vectorized: bool,
        kwargs,
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
        return x_for_fit, x_new, y_new, weights, x_shape, kwargs

    @staticmethod
    def _post_compute_reshaping(fit_result: FitResults, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> FitResults:
        """
        Reshape the output of the fitter into the correct dimensions.
        :param fit_result: Output from the fitter
        :param x: Input x independent
        :param y: Input y dependent
        :return: Reshaped Fit Results
        """
        setattr(fit_result, 'x', x)
        setattr(fit_result, 'y_obs', y)
        setattr(fit_result, 'y_calc', np.reshape(fit_result.y_calc, y.shape))
        setattr(fit_result, 'y_err', np.reshape(fit_result.y_err, y.shape))
        return fit_result
