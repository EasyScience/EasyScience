#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from inspect import Parameter as InspectParameter
from inspect import Signature
from inspect import _empty
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from lmfit import Model as LMModel
from lmfit import Parameter as LMParameter
from lmfit import Parameters as LMParameters
from lmfit.model import ModelResult

from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults


class LMFit(MinimizerBase):  # noqa: S101
    """
    This is a wrapper to the extended Levenberg-Marquardt Fit: https://lmfit.github.io/lmfit-py/
    It allows for the lmfit fitting engine to use parameters declared in an `EasyScience.Objects.Base.BaseObj`.
    """

    wrapping = 'lmfit'

    def __init__(self, obj, fit_function: Callable, method: Optional[str] = None):
        """
        Initialize the minimizer with the `BaseObj` and the `fit_function` to be used.

        :param obj: Base object which contains the parameters to be fitted
        :type obj: BaseObj
        :param fit_function: Function which will be fitted to the data
        :type fit_function: Callable
        :param method: Method to be used by the minimizer
        :type method: str
        """
        super().__init__(obj=obj, fit_function=fit_function, method=method)

    def make_model(self, pars: Optional[LMParameters] = None) -> LMModel:
        """
        Generate a lmfit model from the supplied `fit_function` and parameters in the base object.

        :return: Callable lmfit model
        :rtype: LMModel
        """
        # Generate the fitting function
        fit_func = self._generate_fit_function()
        self._fit_function = fit_func

        if pars is None:
            pars = self._cached_pars
        # Create the model
        model = LMModel(
            fit_func,
            independent_vars=['x'],
            param_names=[MINIMIZER_PARAMETER_PREFIX + str(key) for key in pars.keys()],
        )
        # Assign values from the `Parameter` to the model
        for name, item in pars.items():
            if isinstance(item, LMParameter):
                value = item.value
            else:
                ## TODO clean when full move to new_variable
                from easyscience.Objects.new_variable import Parameter

                if isinstance(item, Parameter):
                    value = item.value
                else:
                    value = item.raw_value

            model.set_param_hint(MINIMIZER_PARAMETER_PREFIX + str(name), value=value, min=item.min, max=item.max)

        # Cache the model for later reference
        self._cached_model = model
        return model

    def _generate_fit_function(self) -> Callable:
        """
        Using the user supplied `fit_function`, wrap it in such a way we can update `Parameter` on
        iterations.

        :return: a fit function which is compatible with lmfit models
        :rtype: Callable
        """
        # Get a list of `Parameters`
        self._cached_pars = {}
        self._cached_pars_vals = {}
        for parameter in self._object.get_fit_parameters():
            key = parameter.unique_name
            self._cached_pars[key] = parameter
            self._cached_pars_vals[key] = (parameter.value, parameter.error)

        # Make a lm fit function
        def lm_fit_function(x: np.ndarray, **kwargs):
            """
            Fit function with a lmfit compatible signature.

            :param x: array of data points to be calculated
            :type x: np.ndarray
            :param kwargs: key word arguments
            :return: points, `f(x)`, calculated at `x`
            :rtype: np.ndarray
            """
            # Update the `Parameter` values and the callback if needed
            # TODO THIS IS NOT THREAD SAFE :-(
            for name, value in kwargs.items():
                par_name = name[1:]
                if par_name in self._cached_pars.keys():
                    # This will take into account constraints

                    ## TODO clean when full move to new_variable
                    from easyscience.Objects.new_variable import Parameter

                    if isinstance(self._cached_pars[par_name], Parameter):
                        if self._cached_pars[par_name].value != value:
                            self._cached_pars[par_name].value = value
                    else:
                        if self._cached_pars[par_name].raw_value != value:
                            self._cached_pars[par_name].value = value

                    # Since we are calling the parameter fset will be called.
            # TODO Pre processing here
            for constraint in self.fit_constraints():
                constraint()
            return_data = self._original_fit_function(x)
            # TODO Loading or manipulating data here
            return return_data

        lm_fit_function.__signature__ = _wrap_to_lm_signature(self._cached_pars)
        return lm_fit_function

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        model: Optional[LMModel] = None,
        parameters: Optional[LMParameters] = None,
        method: Optional[str] = None,
        minimizer_kwargs: Optional[dict] = None,
        engine_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit using the lmfit engine.

        :param method:
        :type method:
        :param x: points to be calculated at
        :type x: np.ndarray
        :param y: measured points
        :type y: np.ndarray
        :param weights: Weights for supplied measured points
        :type weights: np.ndarray
        :param model: Optional Model which is being fitted to
        :type model: LMModel
        :param parameters: Optional parameters for the fit
        :type parameters: LMParameters
        :param minimizer_kwargs: Arguments to be passed directly to the minimizer
        :type minimizer_kwargs: dict
        :param kwargs: Additional arguments for the fitting function.
        :return: Fit results
        :rtype: ModelResult
        """
        default_method = {}
        if self._method is not None:
            default_method = {'method': self._method}
        if method is not None:
            if method in self.available_methods():
                default_method['method'] = method
            else:
                raise FitError(f'Method {method} not available in {self.__class__}')

        if weights is None:
            weights = 1 / np.sqrt(np.abs(y))

        if engine_kwargs is None:
            engine_kwargs = {}

        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        else:
            minimizer_kwargs = {'fit_kws': minimizer_kwargs}
        minimizer_kwargs.update(engine_kwargs)

        # Why do we do this? Because a fitting template has to have global_object instantiated outside pre-runtime
        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False

        try:
            if model is None:
                model = self.make_model()

            model_results = model.fit(y, x=x, weights=weights, **default_method, **minimizer_kwargs, **kwargs)
            self._set_parameter_fit_result(model_results, stack_status)
            results = self._gen_fit_results(model_results)
        except Exception as e:
            for key in self._cached_pars.keys():
                self._cached_pars[key].value = self._cached_pars_vals[key][0]
            raise FitError(e)
        return results

    def convert_to_pars_obj(self, parameters: Optional[List] = None) -> LMParameters:
        """
        Create an lmfit compatible container with the `Parameters` converted from the base object.

        :param parameters: If only a single/selection of parameter is required. Specify as a list
        :return: lmfit Parameters compatible object
        """
        if parameters is None:
            # Assume that we have a BaseObj for which we can obtain a list
            parameters = self._object.get_fit_parameters()
        lm_parameters = LMParameters().add_many([self.convert_to_par_object(parameter) for parameter in parameters])
        return lm_parameters

    @staticmethod
    def convert_to_par_object(parameter) -> LMParameter:
        """
        Convert an `EasyScience.Objects.Base.Parameter` object to a lmfit Parameter object.

        :return: lmfit Parameter compatible object.
        :rtype: LMParameter
        """
        ## TODO clean when full move to new_variable
        from easyscience.Objects.new_variable import Parameter as NewParameter
        if isinstance(parameter, NewParameter):
            value = parameter.value
        else:
            value = parameter.raw_value

        return LMParameter(
            MINIMIZER_PARAMETER_PREFIX + parameter.unique_name,
            value=value,
            vary=not parameter.fixed,
            min=parameter.min,
            max=parameter.max,
            expr=None,
            brute_step=None,
        )

    def _set_parameter_fit_result(self, fit_result: ModelResult, stack_status: bool):
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
        for name in pars.keys():
            pars[name].value = fit_result.params[MINIMIZER_PARAMETER_PREFIX + str(name)].value
            if fit_result.errorbars:
                pars[name].error = fit_result.params[MINIMIZER_PARAMETER_PREFIX + str(name)].stderr
            else:
                pars[name].error = 0.0
        if stack_status:
            global_object.stack.endMacro()

    def _gen_fit_results(self, fit_results: ModelResult, **kwargs) -> FitResults:
        """
        Convert fit results into the unified `FitResults` format.
        See https://github.com/lmfit/lmfit-py/blob/480072b9f7834b31ff2ca66277a5ad31246843a4/lmfit/model.py#L1272

        :param fit_result: Fit object which contains info on the fit
        :return: fit results container
        :rtype: FitResults
        """
        results = FitResults()
        for name, value in kwargs.items():
            if getattr(results, name, False):
                setattr(results, name, value)

        # We need to unify return codes......
        results.success = fit_results.success
        results.y_obs = fit_results.data
        # results.residual = fit_results.residual
        results.x = fit_results.userkws['x']
        results.p = fit_results.values
        results.p0 = fit_results.init_values
        # results.goodness_of_fit = fit_results.chisqr
        results.y_calc = fit_results.best_fit
        results.y_err = 1 / fit_results.weights
        results.minimizer_engine = self.__class__
        results.fit_args = None

        results.engine_result = fit_results
        # results.check_sanity()
        return results

    def available_methods(self) -> List[str]:
        return [
            'least_squares',
            'leastsq',
            'differential_evolution',
            'basinhopping',
            'ampgo',
            'nelder',
            'lbfgsb',
            'powell',
            'cg',
            'newton',
            'cobyla',
            'bfgs',
        ]


def _wrap_to_lm_signature(parameters: Dict) -> Signature:
    """
    Wrap the function signature.
    This is done as lmfit wants the function to be in the form:
    f = (x, a=1, b=2)...
    Where we need to be generic. Note that this won't hold for much outside of this scope.
    """
    wrapped_parameters = []
    wrapped_parameters.append(InspectParameter('x', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty))
    from easyscience.Objects.new_variable import Parameter as NewParameter
    for name, parameter in parameters.items():
        ## TODO clean when full move to new_variable
        if isinstance(parameter, NewParameter):
            default_value = parameter.value
        else:
            default_value = parameter.raw_value

        wrapped_parameters.append(
            InspectParameter(
                MINIMIZER_PARAMETER_PREFIX + str(name),
                InspectParameter.POSITIONAL_OR_KEYWORD,
                annotation=_empty,
                default=default_value,
            )
        )
    return Signature(wrapped_parameters)
