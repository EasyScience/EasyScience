#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

import inspect
from typing import Callable
from typing import List
from typing import Optional

import numpy as np
from bumps.fitters import FIT_AVAILABLE_IDS
from bumps.fitters import fit as bumps_fit
from bumps.names import Curve
from bumps.names import FitProblem
from bumps.parameter import Parameter as BumpsParameter

from easyscience.Objects.Variable import Parameter

from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults


class Bumps(MinimizerBase):  # noqa: S101
    """
    This is a wrapper to Bumps: https://bumps.readthedocs.io/
    It allows for the Bumps fitting engine to use parameters declared in an `EasyScience.Objects.Base.BaseObj`.
    """

    wrapping = 'bumps'

    def __init__(self, obj, fit_function: Callable, method: Optional[str] = None):
        """
        Initialize the fitting engine with a `BaseObj` and an arbitrary fitting function.

        :param obj: Object containing elements of the `Parameter` class
        :type obj: BaseObj
        :param fit_function: function that when called returns y values. 'x' must be the first
                            and only positional argument. Additional values can be supplied by
                            keyword/value pairs
        :type fit_function: Callable
        """
        super().__init__(obj=obj, fit_function=fit_function, method=method)
        self._cached_pars_order = ()
        self._p_0 = {}

    def _make_model(self, pars: Optional[List[BumpsParameter]] = None) -> Callable:
        """
        Generate a bumps model from the supplied `fit_function` and parameters in the base object.
        Note that this makes a callable as it needs to be initialized with *x*, *y*, *weights*

        :return: Callable to make a bumps Curve model
        :rtype: Callable
        """
        fit_func = self._generate_fit_function()

        def outer(obj):
            def make_func(x, y, weights):
                par = {}
                if not pars:
                    for name, item in obj._cached_pars.items():
                        par[MINIMIZER_PARAMETER_PREFIX + str(name)] = obj.convert_to_par_object(item)
                else:
                    for item in pars:
                        par[MINIMIZER_PARAMETER_PREFIX + item.unique_name] = obj.convert_to_par_object(item)
                return Curve(fit_func, x, y, dy=weights, **par)

            return make_func

        return outer(self)

    def _generate_fit_function(self) -> Callable:
        """
        Using the user supplied `fit_function`, wrap it in such a way we can update `Parameter` on
        iterations.

        :return: a fit function which is compatible with bumps models
        :rtype: Callable
        """
        # Original fit function
        func = self._original_fit_function
        # Get a list of `Parameters`
        self._cached_pars_vals = {}
        for parameter in self._object.get_fit_parameters():
            key = parameter.unique_name
            self._cached_pars[key] = parameter
            self._cached_pars_vals[key] = (parameter.value, parameter.error)

        # Make a new fit function
        def fit_function(x: np.ndarray, **kwargs):
            """
            Wrapped fit function which now has a bumps compatible form

            :param x: array of data points to be calculated
            :type x: np.ndarray
            :param kwargs: key word arguments
            :return: points calculated at `x`
            :rtype: np.ndarray
            """
            # Update the `Parameter` values and the callback if needed
            for name, value in kwargs.items():
                par_name = name[1:]
                if par_name in self._cached_pars.keys():
                    ## TODO clean when full move to new_variable
                    from easyscience.Objects.new_variable import Parameter

                    if isinstance(self._cached_pars[par_name], Parameter):
                        if self._cached_pars[par_name].value != value:
                            self._cached_pars[par_name].value = value
                    else:
                        if self._cached_pars[par_name].raw_value != value:
                            self._cached_pars[par_name].value = value

                    # update_fun = self._cached_pars[par_name]._callback.fset
                    # if update_fun:
                    #     update_fun(value)
            # TODO Pre processing here
            for constraint in self.fit_constraints():
                constraint()
            return_data = func(x)
            # TODO Loading or manipulating data here
            return return_data

        # Fake the function signature.
        # This is done as lmfit wants the function to be in the form:
        # f = (x, a=1, b=2)...
        # Where we need to be generic. Note that this won't hold for much outside of this scope.

        ## TODO clean when full move to new_variable
        from easyscience.Objects.new_variable import Parameter

        if isinstance(parameter, Parameter):
            default_value = parameter.value
        else:
            default_value = parameter.raw_value

        self._cached_pars_order = tuple(self._cached_pars.keys())
        params = [
            inspect.Parameter('x', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=inspect._empty),
            *[
                inspect.Parameter(
                    MINIMIZER_PARAMETER_PREFIX + str(name),
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=inspect._empty,
                    default=default_value,
                )
                for name in self._cached_pars_order
            ],
        ]
        # Sign the function
        fit_function.__signature__ = inspect.Signature(params)
        self._fit_function = fit_function
        return fit_function

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        model: Optional[Callable] = None,
        parameters: Optional[Parameter] = None,
        method: Optional[str] = None,
        minimizer_kwargs: Optional[dict] = None,
        engine_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit using the lmfit engine.

        :param x: points to be calculated at
        :type x: np.ndarray
        :param y: measured points
        :type y: np.ndarray
        :param weights: Weights for supplied measured points * Not really optional*
        :type weights: np.ndarray
        :param model: Optional Model which is being fitted to
        :type model: lmModel
        :param parameters: Optional parameters for the fit
        :type parameters: List[BumpsParameter]
        :param kwargs: Additional arguments for the fitting function.
        :param method: Method for minimization
        :type method: str
        :return: Fit results
        :rtype: ModelResult
        """
        default_method = {}
        if self._method is not None:
            default_method = {'method': self._method}
        if method is not None and method in self.available_methods():
            default_method['method'] = method

        if weights is None:
            weights = np.sqrt(np.abs(y))

        if engine_kwargs is None:
            engine_kwargs = {}

        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        # else:
        #     minimizer_kwargs = {"fit_kws": minimizer_kwargs}
        minimizer_kwargs.update(engine_kwargs)

        if model is None:
            model = self._make_model(pars=parameters)
            model = model(x, y, weights)
        self._cached_model = model

        ## TODO clean when full move to new_variable
        from easyscience.Objects.new_variable import Parameter

        if isinstance(self._cached_pars[list(self._cached_pars.keys())[0]], Parameter):
            self._p_0 = {f'p{key}': self._cached_pars[key].value for key in self._cached_pars.keys()}
        else:
            self._p_0 = {f'p{key}': self._cached_pars[key].raw_value for key in self._cached_pars.keys()}

        problem = FitProblem(model)
        # Why do we do this? Because a fitting template has to have global_object instantiated outside pre-runtime
        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False

        try:
            model_results = bumps_fit(problem, **default_method, **minimizer_kwargs, **kwargs)
            self._set_parameter_fit_result(model_results, stack_status)
            results = self._gen_fit_results(model_results)
        except Exception as e:
            for key in self._cached_pars.keys():
                self._cached_pars[key].value = self._cached_pars_vals[key][0]
            raise FitError(e)
        return results

    def convert_to_pars_obj(self, par_list: Optional[List] = None) -> List[BumpsParameter]:
        """
        Create a container with the `Parameters` converted from the base object.

        :param par_list: If only a single/selection of parameter is required. Specify as a list
        :type par_list: List[str]
        :return: bumps Parameters list
        :rtype: List[BumpsParameter]
        """
        if par_list is None:
            # Assume that we have a BaseObj for which we can obtain a list
            par_list = self._object.get_fit_parameters()
        pars_obj = [self.__class__.convert_to_par_object(obj) for obj in par_list]
        return pars_obj

    # For some reason I have to double staticmethod :-/
    @staticmethod
    def convert_to_par_object(obj) -> BumpsParameter:
        """
        Convert an `EasyScience.Objects.Base.Parameter` object to a bumps Parameter object

        :return: bumps Parameter compatible object.
        :rtype: BumpsParameter
        """

        ## TODO clean when full move to new_variable
        from easyscience.Objects.new_variable import Parameter

        if isinstance(obj, Parameter):
            value = obj.value
        else:
            value = obj.raw_value

        return BumpsParameter(
            name=MINIMIZER_PARAMETER_PREFIX + obj.unique_name,
            value=value,
            bounds=[obj.min, obj.max],
            fixed=obj.fixed,
        )

    def _set_parameter_fit_result(self, fit_result, stack_status: bool):
        """
        Update parameters to their final values and assign a std error to them.

        :param fit_result: Fit object which contains info on the fit
        :return: None
        :rtype: noneType
        """
        from easyscience import global_object

        pars = self._cached_pars

        if stack_status:
            for name in pars.keys():
                pars[name].value = self._cached_pars_vals[name][0]
                pars[name].error = self._cached_pars_vals[name][1]
            global_object.stack.enabled = True
            global_object.stack.beginMacro('Fitting routine')

        for index, name in enumerate(self._cached_model._pnames):
            dict_name = name[1:]
            pars[dict_name].value = fit_result.x[index]
            pars[dict_name].error = fit_result.dx[index]
        if stack_status:
            global_object.stack.endMacro()

    def _gen_fit_results(self, fit_results, **kwargs) -> FitResults:
        """
        Convert fit results into the unified `FitResults` format

        :param fit_result: Fit object which contains info on the fit
        :return: fit results container
        :rtype: FitResults
        """

        results = FitResults()
        for name, value in kwargs.items():
            if getattr(results, name, False):
                setattr(results, name, value)
        results.success = fit_results.success
        pars = self._cached_pars
        item = {}
        for index, name in enumerate(self._cached_model._pnames):
            dict_name = name[1:]

            ## TODO clean when full move to new_variable
            from easyscience.Objects.new_variable import Parameter

            if isinstance(pars[dict_name], Parameter):
                item[name] = pars[dict_name].value
            else:
                item[name] = pars[dict_name].raw_value

        results.p0 = self._p_0
        results.p = item
        results.x = self._cached_model.x
        results.y_obs = self._cached_model.y
        results.y_calc = self.evaluate(results.x, minimizer_parameters=results.p)
        results.y_err = self._cached_model.dy
        # results.residual = results.y_obs - results.y_calc
        # results.goodness_of_fit = np.sum(results.residual**2)
        results.minimizer_engine = self.__class__
        results.fit_args = None
        results.engine_result = fit_results
        # results.check_sanity()
        return results

    def available_methods(self) -> List[str]:
        return FIT_AVAILABLE_IDS
