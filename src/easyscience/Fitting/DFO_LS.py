#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

from numbers import Number
from typing import List
from typing import Optional

# Import dfols specific objects
import dfols

from easyscience.Fitting.fitting_template import Callable
from easyscience.Fitting.fitting_template import FitError
from easyscience.Fitting.fitting_template import FitResults
from easyscience.Fitting.fitting_template import FittingTemplate
from easyscience.Fitting.fitting_template import NameConverter
from easyscience.Fitting.fitting_template import np


class DFO(FittingTemplate):  # noqa: S101
    """
    This is a wrapper to Derivative free optimisation: https://numericalalgorithmsgroup.github.io/dfols/
    """

    property_type = Number
    name = "DFO_LS"

    def __init__(self, obj, fit_function: Callable):
        """
        Initialize the fitting engine with a `BaseObj` and an arbitrary fitting function.

        :param obj: Object containing elements of the `Parameter` class
        :type obj: BaseObj
        :param fit_function: function that when called returns y values. 'x' must be the first
                            and only positional argument. Additional values can be supplied by
                            keyword/value pairs
        :type fit_function: Callable
        """
        super().__init__(obj, fit_function)
        self.p_0 = {}

    def make_model(self, pars: Optional[List] = None) -> Callable:
        """
        Generate a model from the supplied `fit_function` and parameters in the base object.
        Note that this makes a callable as it needs to be initialized with *x*, *y*, *weights*

        :return: Callable model which returns residuals
        :rtype: Callable
        """
        fit_func = self._generate_fit_function()

        def outer(obj):
            def make_func(x, y, weights):
                par = {}
                if not pars:
                    for name, item in obj._cached_pars.items():
                        par["p" + str(name)] = item.raw_value
                else:
                    for item in pars:
                        par["p" + str(NameConverter().get_key(item))] = item.raw_value

                def residuals(x0) -> np.ndarray:
                    for idx, par_name in enumerate(par.keys()):
                        par[par_name] = x0[idx]
                    return (y - fit_func(x, **par)) / weights

                setattr(residuals, "x", x)
                setattr(residuals, "y", y)
                return residuals

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
        self._cached_pars = {}
        self._cached_pars_vals = {}
        for parameter in self._object.get_fit_parameters():
            key = NameConverter().get_key(parameter)
            self._cached_pars[key] = parameter
            self._cached_pars_vals[key] = (parameter.value, parameter.error)

        # Make a new fit function
        def fit_function(x: np.ndarray, **kwargs):
            """
            Wrapped fit function which now has an EasyScience compatible form

            :param x: array of data points to be calculated
            :type x: np.ndarray
            :param kwargs: key word arguments
            :return: points calculated at `x`
            :rtype: np.ndarray
            """
            # Update the `Parameter` values and the callback if needed
            # TODO THIS IS NOT THREAD SAFE :-(
            for name, value in kwargs.items():
                par_name = int(name[1:])
                if par_name in self._cached_pars.keys():
                    # This will take into account constraints
                    if self._cached_pars[par_name].raw_value != value:
                        self._cached_pars[par_name].value = value
                    # Since we are calling the parameter fset will be called.
            # TODO Pre processing here
            for constraint in self.fit_constraints():
                constraint()
            return_data = func(x)
            # TODO Loading or manipulating data here
            return return_data

        self._fit_function = fit_function
        return fit_function

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        model=None,
        parameters=None,
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
        if method is not None and method in self.available_methods():
            default_method["method"] = method

        if weights is None:
            weights = np.sqrt(np.abs(y))

        if model is None:
            model = self.make_model(pars=parameters)
            model = model(x, y, weights)
        self._cached_model = model
        self.p_0 = {
            f"p{key}": self._cached_pars[key].raw_value
            for key in self._cached_pars.keys()
        }

        # Why do we do this? Because a fitting template has to have borg instantiated outside pre-runtime
        from easyscience import borg

        stack_status = borg.stack.enabled
        borg.stack.enabled = False

        try:
            model_results = self.dfols_fit(model, **kwargs)
            self._set_parameter_fit_result(model_results, stack_status)
            results = self._gen_fit_results(model_results, weights)
        except Exception as e:
            for key in self._cached_pars.keys():
                self._cached_pars[key].value = self._cached_pars_vals[key][0]
            raise FitError(e)
        return results

    def convert_to_pars_obj(self, par_list: Optional[list] = None):
        """
        NOTE THAT THIS IS NOT NEEDED FOR DFO-LS
        """

        pass

    @staticmethod
    def convert_to_par_object(obj) -> None:
        """
        Convert an `EasyScience.Objects.Base.Parameter` object to a new Parameter object
        NOTE THAT THIS IS NOT NEEDED FOR DFO-LS
        """
        pass

    def _set_parameter_fit_result(
        self, fit_result, stack_status, ci: float = 0.95
    ) -> None:
        """
        Update parameters to their final values and assign a std error to them.

        :param fit_result: Fit object which contains info on the fit
        :param ci: Confidence interval for calculating errors. Default 95%
        :return: None
        :rtype: noneType
        """
        from easyscience import borg

        pars = self._cached_pars
        if stack_status:
            for name in pars.keys():
                pars[name].value = self._cached_pars_vals[name][0]
                pars[name].error = self._cached_pars_vals[name][1]
            borg.stack.enabled = True
            borg.stack.beginMacro("Fitting routine")

        error_matrix = self._error_from_jacobian(
            fit_result.jacobian, fit_result.resid, ci
        )
        for idx, par in enumerate(pars.values()):
            par.value = fit_result.x[idx]
            par.error = error_matrix[idx, idx]

        if stack_status:
            borg.stack.endMacro()

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
        pars = self._cached_pars
        item = {}
        for p_name, par in pars.items():
            item[f"p{p_name}"] = par.raw_value
        results.p0 = self.p_0
        results.p = item
        results.x = self._cached_model.x
        results.y_obs = self._cached_model.y
        results.y_calc = self.evaluate(results.x, parameters=results.p)
        results.y_err = weights
        # results.residual = results.y_obs - results.y_calc
        # results.goodness_of_fit = fit_results.f

        results.fitting_engine = self.__class__
        results.fit_args = None
        # results.check_sanity()

        return results

    def available_methods(self) -> List[str]:
        return ["leastsq"]

    def dfols_fit(self, model: Callable, **kwargs):
        """
                Method to convert EasyScience styling to DFO-LS styling (yes, again)

                :param model: Model which accepts f(x[0])
                :type model: Callable
                :param kwargs: Any additional arguments for dfols.solver
                :type kwargs: dict
                :return: dfols fit results container
        ="""
        x0 = np.array([par.raw_value for par in iter(self._cached_pars.values())])
        bounds = (
            np.array([par.min for par in iter(self._cached_pars.values())]),
            np.array([par.max for par in iter(self._cached_pars.values())]),
        )
        results = dfols.solve(model, x0, bounds=bounds, **kwargs)
        return results
