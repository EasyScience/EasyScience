# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Callable
from typing import List
from typing import Union

import numpy as np

from easyscience import global_object

from .available_minimizers import AvailableMinimizers
from .available_minimizers import from_string_to_enum
from .minimizers import FitResults
from .minimizers import MinimizerBase
from .minimizers.factory import factory
from .reshaping import inject_x
from .reshaping import reshape_dataset

DEFAULT_MINIMIZER = AvailableMinimizers.LMFit_leastsq


class Fitter:
    """
    Fitter is a class which makes it possible to undertake fitting
    utilizing one of the supported minimizers.
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

    def make_model(self, pars=None) -> Callable:
        return self._minimizer.make_model(pars)

    def evaluate(self, pars=None) -> np.ndarray:
        return self._minimizer.evaluate(pars)

    # TODO: remove this method when we are ready to adjust the dependent products
    def initialize(self, fit_object: object, fit_function: Callable) -> None:
        """
        Set the model and callable in the calculator interface.

        Parameters
        ----------
        fit_object : object
            The EasyScience model object.
        fit_function : Callable
            The function to be optimized against.
        """
        self._fit_object = fit_object
        self._fit_function = fit_function
        self._update_minimizer(DEFAULT_MINIMIZER)

    # TODO: remove this method when we are ready to adjust the dependent products
    def create(self, minimizer_enum: Union[AvailableMinimizers, str] = DEFAULT_MINIMIZER) -> None:
        """
        Create the required minimizer.

        Parameters
        ----------
        minimizer_enum : Union[AvailableMinimizers, str], default=DEFAULT_MINIMIZER
            The enum of the minimization engine to create. By default,
            DEFAULT_MINIMIZER.
        """
        if isinstance(minimizer_enum, str):
            global_object.log.getLogger('fitting').warning(
                'minimizer should be set with enum %s', minimizer_enum
            )
            minimizer_enum = from_string_to_enum(minimizer_enum)
        self._update_minimizer(minimizer_enum)

    def switch_minimizer(self, minimizer_enum: Union[AvailableMinimizers, str]) -> None:
        """
        Switch minimizer and initialize.

        Parameters
        ----------
        minimizer_enum : Union[AvailableMinimizers, str]
            The enum of the minimizer to create and instantiate.
        """
        if isinstance(minimizer_enum, str):
            global_object.log.getLogger('fitting').warning(
                'minimizer should be set with enum %s', minimizer_enum
            )
            minimizer_enum = from_string_to_enum(minimizer_enum)

        self._update_minimizer(minimizer_enum)

    def _update_minimizer(self, minimizer_enum: AvailableMinimizers) -> None:
        self._minimizer = factory(
            minimizer_enum=minimizer_enum,
            fit_object=self._fit_object,
            fit_function=self.fit_function,
        )
        self._enum_current_minimizer = minimizer_enum

    @property
    def available_minimizers(self) -> List[str]:
        """
        Get a list of the names of available fitting minimizers.

        Returns
        -------
        List[str]
            List of available fitting minimizers.
        """
        return [minimize.name for minimize in AvailableMinimizers]

    @property
    def minimizer(self) -> MinimizerBase:
        """
        Get the current fitting minimizer object.

        Returns
        -------
        MinimizerBase
        """
        return self._minimizer

    @property
    def tolerance(self) -> float:
        """
        Get the tolerance for the minimizer.

        Returns
        -------
        float
            Tolerance for the minimizer.
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        """
        Set the tolerance for the minimizer.

        Parameters
        ----------
        tolerance : float
            Tolerance for the minimizer.
        """
        self._tolerance = tolerance

    @property
    def max_evaluations(self) -> int:
        """
        Get the maximal number of evaluations for the minimizer.

        Returns
        -------
        int
            Maximal number of steps for the minimizer.
        """
        return self._max_evaluations

    @max_evaluations.setter
    def max_evaluations(self, max_evaluations: int) -> None:
        """
        Set the maximal number of evaluations for the minimizer.

        Parameters
        ----------
        max_evaluations : int
            Maximal number of steps for the minimizer.
        """
        self._max_evaluations = max_evaluations

    @property
    def fit_function(self) -> Callable:
        """
        Get the raw fit function that the optimizer will call.

        Returns
        -------
        Callable
            Raw fit function.
        """
        return self._fit_function

    @fit_function.setter
    def fit_function(self, fit_function: Callable) -> None:
        """
        Set the raw fit function to a new one.

        Parameters
        ----------
        fit_function : Callable
            New fit function.

        Returns
        -------
        None
            None.
        """
        self._fit_function = fit_function
        self._update_minimizer(self._enum_current_minimizer)

    @property
    def fit_object(self) -> object:
        """
        Get the EasyScience object used as a model.

        For a ``Fitter`` this is not one of the supplied fit
        objects but a read-only, indexable and iterable aggregate.

        Returns
        -------
        object
            EasyScience model object.
        """
        return self._fit_object

    @fit_object.setter
    def fit_object(self, fit_object: object) -> None:
        """
        Set the EasyScience object which wil be used as a model.

        Parameters
        ----------
        fit_object : object
            New EasyScience object.
        """
        self._fit_object = fit_object
        self._update_minimizer(self._enum_current_minimizer)

    def _fit_function_wrapper(
        self,
        real_x: np.ndarray | None = None,
        flatten: bool = True,
    ) -> Callable:
        """
        Wrap the fit function so it evaluates on the real X (independent)
        values instead of the optimizer's dummy x, flattening if needed.

        Parameters
        ----------
        real_x : np.ndarray | None, default=None
            Independent x parameters to be injected. By default, None.
        flatten : bool, default=True
            Should the result be a flat 1D array? By default, True.

        Returns
        -------
        Callable
            Wrapped optimizer function.
        """
        return inject_x(self._fit_function, real_x, flatten=flatten)

    @property
    def fit(self) -> Callable:
        """
        Property which wraps the current ``fit`` function from the
        fitting interface.

        This property return a wrapped fit function which converts the
        input data into the correct shape for the optimizer, wraps the
        fit function to re-constitute the independent variables and once
        the fit is completed, reshape the inputs to those expected.
        """

        @functools.wraps(self._minimizer.fit)
        def inner_fit_callable(
            x: np.ndarray,
            y: np.ndarray,
            weights: np.ndarray | None = None,
            vectorized: bool = False,
            progress_callback: Callable[[dict], None] | None = None,
            **kwargs,
        ) -> FitResults:
            """
            This is a wrapped callable which performs the actual
            fitting. It is split into.

            3 sections, PRE/ FIT/ POST.
            - PRE = Reshaping the input data into the correct dimensions for the optimizer
            - FIT = Wrapping the fit function and performing the fit
            - POST = Reshaping the outputs so it is coherent with the inputs.
            """
            # Precompute - Reshape all independents into the correct dimensionality
            x_fit, x_new, y_new, weights, dims = self._precompute_reshaping(
                x, y, weights, vectorized
            )
            self._dependent_dims = dims

            # Fit
            fit_fun_org = self._fit_function
            fit_fun_wrap = self._fit_function_wrapper(
                x_new, flatten=True
            )  # This should be wrapped.
            self.fit_function = fit_fun_wrap
            f_res = self._minimizer.fit(
                x_fit,
                y_new,
                weights=weights,
                tolerance=self._tolerance,
                max_evaluations=self._max_evaluations,
                progress_callback=progress_callback,
                **kwargs,
            )

            # Postcompute
            fit_result = self._post_compute_reshaping(f_res, x, y)
            # Reset the function
            self.fit_function = fit_fun_org
            return fit_result

        return inner_fit_callable

    _precompute_reshaping = staticmethod(reshape_dataset)

    @staticmethod
    def _post_compute_reshaping(
        fit_result: FitResults, x: np.ndarray, y: np.ndarray
    ) -> FitResults:
        """
        Reshape the output of the fitter into the correct dimensions.

        Parameters
        ----------
        fit_result : FitResults
            Output from the fitter.
        x : np.ndarray
            Input x independent.
        y : np.ndarray
            Input y dependent.

        Returns
        -------
        FitResults
            Reshaped Fit Results.
        """
        fit_result.x = x
        fit_result.y_obs = y
        fit_result.y_calc = np.reshape(fit_result.y_calc, y.shape)
        fit_result.y_err = np.reshape(fit_result.y_err, y.shape)
        return fit_result
