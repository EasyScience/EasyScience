#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

import inspect
from typing import List
from typing import Optional

from lmfit import Model as lmModel

# Import lmfit specific objects
from lmfit import Parameter as lmParameter
from lmfit import Parameters as lmParameters
from lmfit.model import ModelResult

from easyscience.Fitting.fitting_template import Callable
from easyscience.Fitting.fitting_template import FitError
from easyscience.Fitting.fitting_template import FitResults
from easyscience.Fitting.fitting_template import FittingTemplate
from easyscience.Fitting.fitting_template import NameConverter
from easyscience.Fitting.fitting_template import np


class lmfit(FittingTemplate):  # noqa: S101
    """
    This is a wrapper to lmfit: https://lmfit.github.io/
    It allows for the lmfit fitting engine to use parameters declared in an `EasyScience.Objects.Base.BaseObj`.
    """

    property_type = lmParameter
    name = "lmfit"

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

    def make_model(self, pars: Optional[lmParameters] = None) -> lmModel:
        """
        Generate a lmfit model from the supplied `fit_function` and parameters in the base object.

        :return: Callable lmfit model
        :rtype: lmModel
        """
        # Generate the fitting function
        fit_func = self._generate_fit_function()
        if pars is None:
            pars = self._cached_pars
        # Create the model
        model = lmModel(
            fit_func,
            independent_vars=["x"],
            param_names=["p" + str(key) for key in pars.keys()],
        )
        # Assign values from the `Parameter` to the model
        for name, item in pars.items():
            if isinstance(item, lmParameter):
                value = item.value
            else:
                value = item.raw_value
            model.set_param_hint(
                "p" + str(name), value=value, min=item.min, max=item.max
            )

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
            Wrapped fit function which now has a lmfit compatible form.

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

        # Fake the function signature.
        # This is done as lmfit wants the function to be in the form:
        # f = (x, a=1, b=2)...
        # Where we need to be generic. Note that this won't hold for much outside of this scope.
        params = [
            inspect.Parameter(
                "x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=inspect._empty
            ),
            *[
                inspect.Parameter(
                    "p" + str(name),
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=inspect._empty,
                    default=parameter.raw_value,
                )
                for name, parameter in self._cached_pars.items()
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
        model: Optional[lmModel] = None,
        parameters: Optional[lmParameters] = None,
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
        :type model: lmModel
        :param parameters: Optional parameters for the fit
        :type parameters: lmParameters
        :param minimizer_kwargs: Arguments to be passed directly to the minimizer
        :type minimizer_kwargs: dict
        :param kwargs: Additional arguments for the fitting function.
        :return: Fit results
        :rtype: ModelResult
        """
        default_method = {}
        if method is not None and method in self.available_methods():
            default_method["method"] = method

        if weights is None:
            weights = 1 / np.sqrt(np.abs(y))

        if engine_kwargs is None:
            engine_kwargs = {}

        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        else:
            minimizer_kwargs = {"fit_kws": minimizer_kwargs}
        minimizer_kwargs.update(engine_kwargs)

        # Why do we do this? Because a fitting template has to have borg instantiated outside pre-runtime
        from easyscience import borg

        stack_status = borg.stack.enabled
        borg.stack.enabled = False

        try:
            if model is None:
                model = self.make_model()

            model_results = model.fit(
                y, x=x, weights=weights, **default_method, **minimizer_kwargs, **kwargs
            )
            self._set_parameter_fit_result(model_results, stack_status)
            results = self._gen_fit_results(model_results)
        except Exception as e:
            for key in self._cached_pars.keys():
                self._cached_pars[key].value = self._cached_pars_vals[key][0]
            raise FitError(e)
        return results

    def convert_to_pars_obj(self, par_list: Optional[List] = None) -> lmParameters:
        """
        Create an lmfit compatible container with the `Parameters` converted from the base object.

        :param par_list: If only a single/selection of parameter is required. Specify as a list
        :type par_list: List[str]
        :return: lmfit Parameters compatible object
        :rtype: lmParameters
        """
        if par_list is None:
            # Assume that we have a BaseObj for which we can obtain a list
            par_list = self._object.get_fit_parameters()
        pars_obj = lmParameters().add_many(
            [self.__class__.convert_to_par_object(obj) for obj in par_list]
        )
        return pars_obj

    @staticmethod
    def convert_to_par_object(obj) -> lmParameter:
        """
        Convert an `EasyScience.Objects.Base.Parameter` object to a lmfit Parameter object.

        :return: lmfit Parameter compatible object.
        :rtype: lmParameter
        """
        return lmParameter(
            "p" + str(NameConverter().get_key(obj)),
            value=obj.raw_value,
            vary=not obj.fixed,
            min=obj.min,
            max=obj.max,
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
        from easyscience import borg

        pars = self._cached_pars
        if stack_status:
            for name in pars.keys():
                pars[name].value = self._cached_pars_vals[name][0]
                pars[name].error = self._cached_pars_vals[name][1]
            borg.stack.enabled = True
            borg.stack.beginMacro("Fitting routine")
        for name in pars.keys():
            pars[name].value = fit_result.params["p" + str(name)].value
            if fit_result.errorbars:
                pars[name].error = fit_result.params["p" + str(name)].stderr
            else:
                pars[name].error = 0.0
        if stack_status:
            borg.stack.endMacro()

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
        results.x = fit_results.userkws["x"]
        results.p = fit_results.values
        results.p0 = fit_results.init_values
        # results.goodness_of_fit = fit_results.chisqr
        results.y_calc = fit_results.best_fit
        results.y_err = 1 / fit_results.weights
        results.fitting_engine = self.__class__
        results.fit_args = None

        results.engine_result = fit_results
        # results.check_sanity()
        return results

    def available_methods(self) -> List[str]:
        return [
            "least_squares",
            "leastsq",
            "differential_evolution",
            "basinhopping",
            "ampgo",
            "nelder",
            "lbfgsb",
            "powell",
            "cg",
            "newton",
            "cobyla",
            "bfgs",
        ]
