#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import dfols
import numpy as np

# causes circular import when Parameter is imported
# from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.Variable import Parameter

from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults


class DFO(MinimizerBase):
    """
    This is a wrapper to Derivative Free Optimisation for Least Square: https://numericalalgorithmsgroup.github.io/dfols/
    """

    wrapping = 'dfo'

    def __init__(
        self,
        obj,  #: BaseObj,
        fit_function: Callable,
        method: Optional[str] = None,
    ):  # todo after constraint changes, add type hint: obj: BaseObj  # noqa: E501
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
        self._p_0 = {}

    def available_methods(self) -> List[str]:
        return ['leastsq']

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        model: Optional[Callable] = None,
        parameters: Optional[List[Parameter]] = None,
        method: str = None,
        xtol: float = 1e-6,
        ftol: float = 1e-8,
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit using the DFO-ls engine.

        :param x: points to be calculated at
        :type x: np.ndarray
        :param y: measured points
        :type y: np.ndarray
        :param weights: Weights for supplied measured points * Not really optional*
        :type weights: np.ndarray
        :param model: Optional Model which is being fitted to
        :type model: lmModel
        :param parameters: Optional parameters for the fit
        :type parameters: List[bumpsParameter]
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

        if model is None:
            model_function = self._make_model(parameters=parameters)
            model = model_function(x, y, weights)
        self._cached_model = model
        self._cached_model.x = x
        self._cached_model.y = y

        ## TODO clean when full move to new_variable
        from easyscience.Objects.new_variable import Parameter

        if isinstance(self._cached_pars[list(self._cached_pars.keys())[0]], Parameter):
            self._p_0 = {f'p{key}': self._cached_pars[key].value for key in self._cached_pars.keys()}
        else:
            self._p_0 = {f'p{key}': self._cached_pars[key].raw_value for key in self._cached_pars.keys()}

        # Why do we do this? Because a fitting template has to have global_object instantiated outside pre-runtime
        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False

        try:
            model_results = self._dfo_fit(self._cached_pars, model, **kwargs)
            self._set_parameter_fit_result(model_results, stack_status)
            results = self._gen_fit_results(model_results, weights)
        except Exception as e:
            for key in self._cached_pars.keys():
                self._cached_pars[key].value = self._cached_pars_vals[key][0]
            raise FitError(e)
        return results

    def convert_to_pars_obj(self, par_list: Optional[list] = None):
        """
        Required by interface but not needed for DFO-LS
        """
        pass

    @staticmethod
    def convert_to_par_object(obj) -> None:
        """
        Required by interface but not needed for DFO-LS
        """
        pass

    def _make_model(self, parameters: Optional[List[Parameter]] = None) -> Callable:
        """
        Generate a model from the supplied `fit_function` and parameters in the base object.
        Note that this makes a callable as it needs to be initialized with *x*, *y*, *weights*

        :return: Callable model which returns residuals
        :rtype: Callable
        """
        fit_func = self._generate_fit_function()

        def _outer(obj: DFO):
            def _make_func(x, y, weights):
                ## TODO clean when full move to new_variable
                from easyscience.Objects.new_variable import Parameter as NewParameter

                dfo_pars = {}
                if not parameters:
                    for name, par in obj._cached_pars.items():
                        if isinstance(par, NewParameter):
                            dfo_pars[MINIMIZER_PARAMETER_PREFIX + str(name)] = par.value
                        else:
                            dfo_pars[MINIMIZER_PARAMETER_PREFIX + str(name)] = par.raw_value

                else:
                    for par in parameters:
                        if isinstance(par, NewParameter):
                            dfo_pars[MINIMIZER_PARAMETER_PREFIX + par.unique_name] = par.value
                        else:
                            dfo_pars[MINIMIZER_PARAMETER_PREFIX + par.unique_name] = par.raw_value

                def _residuals(pars_values: List[float]) -> np.ndarray:
                    for idx, par_name in enumerate(dfo_pars.keys()):
                        dfo_pars[par_name] = pars_values[idx]
                    return (y - fit_func(x, **dfo_pars)) / weights

                return _residuals

            return _make_func

        return _outer(self)

    def _set_parameter_fit_result(self, fit_result, stack_status, ci: float = 0.95) -> None:
        """
        Update parameters to their final values and assign a std error to them.

        :param fit_result: Fit object which contains info on the fit
        :param ci: Confidence interval for calculating errors. Default 95%
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

        error_matrix = self._error_from_jacobian(fit_result.jacobian, fit_result.resid, ci)
        for idx, par in enumerate(pars.values()):
            par.value = fit_result.x[idx]
            par.error = error_matrix[idx, idx]

        if stack_status:
            global_object.stack.endMacro()

    def _gen_fit_results(self, fit_results, weights, **kwargs) -> FitResults:
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
        results.success = not bool(fit_results.flag)

        pars = {}
        for p_name, par in self._cached_pars.items():
            ## TODO clean when full move to new_variable
            from easyscience.Objects.new_variable import Parameter

            if isinstance(par, Parameter):
                pars[f'p{p_name}'] = par.value
            else:
                pars[f'p{p_name}'] = par.raw_value
        results.p = pars

        results.p0 = self._p_0
        results.x = self._cached_model.x
        results.y_obs = self._cached_model.y
        results.y_calc = self.evaluate(results.x, minimizer_parameters=results.p)
        results.y_err = weights
        # results.residual = results.y_obs - results.y_calc
        # results.goodness_of_fit = fit_results.f

        results.minimizer_engine = self.__class__
        results.fit_args = None
        # results.check_sanity()

        return results

    @staticmethod
    def _dfo_fit(pars: Dict[str, Parameter], model: Callable, **kwargs):
        """
        Method to convert EasyScience styling to DFO-LS styling (yes, again)

        :param model: Model which accepts f(x[0])
        :type model: Callable
        :param kwargs: Any additional arguments for dfols.solver
        :type kwargs: dict
        :return: dfols fit results container
        """

        ## TODO clean when full move to new_variable
        from easyscience.Objects.new_variable import Parameter as NewParameter

        if isinstance(list(pars.values())[0], NewParameter):
            pars_values = np.array([par.value for par in pars.values()])
        else:
            pars_values = np.array([par.raw_value for par in pars.values()])

        bounds = (
            np.array([par.min for par in pars.values()]),
            np.array([par.max for par in pars.values()]),
        )
        # https://numericalalgorithmsgroup.github.io/dfols/build/html/userguide.html
        if np.isinf(bounds).any():
            results = dfols.solve(model, pars_values, bounds=bounds, **kwargs)
        else:
            # It is only possible to scale (normalize) variables if they are bound (different from inf)
            results = dfols.solve(model, pars_values, bounds=bounds, scaling_within_bounds=True, **kwargs)

        if 'Success' not in results.msg:
            raise FitError(f'Fit failed with message: {results.msg}')

        return results
