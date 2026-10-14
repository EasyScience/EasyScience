# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Callable
from typing import List

import iminuit
import numpy as np
from iminuit.cost import LeastSquares

from ..available_minimizers import AvailableMinimizers
from ..engine_base import PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults

# Minuit stops when the estimated distance to the minimum (EDM) is below 0.002 * tol * errordef.
# Its own default (tol = 0.1) stops short of the exact minimum, which is too coarse for a noise-free
# fit, so the default here is tighter.
DEFAULT_TOLERANCE = 1e-4
# The EDM goal is absolute, while the rounding noise of a chi-square grows with its size: for a
# chi-square near 1e12 a goal of 2e-7 cannot be reached. The default tolerance is therefore raised
# until the goal is at least this fraction of the chi-square at the start (1e-14 / 0.002).
RELATIVE_TOLERANCE_FLOOR = 5e-12


class Minuit(MinimizerBase):
    """
    This is a wrapper to the Minuit2 minimizer through ``iminuit``:
    https://scikit-hep.org/iminuit/ It allows the MIGRAD and simplex
    algorithms to use parameters declared in an
    ``easyscience.base_classes.ModelBase``.
    """

    package = 'minuit'

    def __init__(
        self,
        obj: object,  #: ModelBase,
        fit_function: Callable,
        minimizer_enum: AvailableMinimizers | None = None,
    ):  # todo after constraint changes, add type hint: obj: ModelBase  # noqa: E501
        """
        Initialize the minimizer.

        Parameters
        ----------
        obj : object
            Object containing the ``Parameter`` instances to fit.
        fit_function : Callable
            Callable returning model y values for the supplied x values.
        minimizer_enum : AvailableMinimizers | None, default=None
            Selected Minuit configuration. By default, None.
        """
        super().__init__(obj=obj, fit_function=fit_function, minimizer_enum=minimizer_enum)

    @staticmethod
    def all_methods() -> List[str]:
        return ['migrad', 'simplex']

    @staticmethod
    def supported_methods() -> List[str]:
        return ['migrad', 'simplex']

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray = None,
        method: str | None = None,
        tolerance: float | None = None,
        max_evaluations: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        scale_covar: bool = False,
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit using the Minuit engine.

        Parameters
        ----------
        x : np.ndarray
            Points to be calculated at.
        y : np.ndarray
            Measured points.
        weights : np.ndarray, default=None
            Weights for supplied measured points; they should be
            1/sigma, where sigma is the standard deviation of the
            measurement. By default, None.
        method : str | None, default=None
            Minuit algorithm, ``'migrad'`` or ``'simplex'``. By default,
            None, which uses the method of the selected minimizer.
        tolerance : float | None, default=None
            Requested tolerance, as the ``tol`` of Minuit: it scales the
            distance to the minimum at which the fit stops. By default,
            None, which uses 1e-4, raised for a very large chi-square.
        max_evaluations : int | None, default=None
            Maximum number of function evaluations. By default, None.
        progress_callback : Callable[[dict], None] | None, default=None
            Optional callback receiving normalized progress payloads.
            Its return value is ignored. By default, None.
        scale_covar : bool, default=False
            Multiply the variances written to the parameters by the
            reduced chi-square, as lmfit does. The ``errors`` and the
            ``covariance`` of the Minuit object in ``engine_result`` are
            never scaled. By default, False.
        **kwargs :
            Not used. Any additional keyword argument raises a
            ``TypeError``.

        Returns
        -------
        FitResults
            Fit results.

        Raises
        ------
        FitError
            If the Minuit optimization fails.
        TypeError
            If an unexpected keyword argument is given.

        Notes
        -----
        ``engine_result`` of the returned ``FitResults`` is the Minuit
        object, so ``hesse()``, ``minos()`` and ``covariance`` are
        available. Its cost function writes the values it evaluates into
        the ``Parameter`` objects, so after ``hesse()`` or ``minos()``
        the parameters sit at the last evaluated point. Restore the
        fitted values with
        ``result.engine_result.fcn(list(result.engine_result.values))``.
        """
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments for Minuit: {sorted(kwargs)}.')
        x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
        self.validate_arrays(x, y, weights)
        method = self._get_method_kwargs(method).get('method', 'migrad')

        # Why do we do this? Because a fitting template has to have global_object instantiated outside pre-runtime
        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False
        try:
            fit_function = self._generate_fit_function()
            self._fit_function = fit_function
            if not self._cached_pars:
                raise FitError('There are no free parameters to fit.')
            keys = list(self._cached_pars)
            names = [PARAMETER_PREFIX + str(key) for key in keys]
            n_degrees_of_freedom = len(y) - len(names)
            evaluations = 0

            def model(x_values: np.ndarray, *parameters: float) -> np.ndarray:
                # iminuit calls the model positionally; the wrapper takes keyword arguments.
                nonlocal evaluations
                prediction = fit_function(x_values, **dict(zip(names, parameters)))
                evaluations += 1
                if progress_callback is not None:
                    chi2 = float(np.square((y - prediction) * weights).sum())
                    progress_callback({
                        'iteration': evaluations,
                        'chi2': chi2,
                        'reduced_chi2': chi2 / n_degrees_of_freedom
                        if n_degrees_of_freedom > 0
                        else chi2,
                        'parameter_values': dict(zip(keys, map(float, parameters))),
                        'refresh_plots': False,
                        'finished': False,
                    })
                return prediction

            cost = LeastSquares(x, y, 1 / weights, model, name=names)
            start = {
                name: float(parameter.value)
                for name, parameter in zip(names, self._cached_pars.values())
            }
            minuit = iminuit.Minuit(cost, **start)
            for name, parameter in zip(names, self._cached_pars.values()):
                minuit.limits[name] = (parameter.min, parameter.max)
            if tolerance is None:
                start_chi2 = float(np.square((y - fit_function(x, **start)) * weights).sum())
                tolerance = max(DEFAULT_TOLERANCE, RELATIVE_TOLERANCE_FLOOR * start_chi2)
            minuit.tol = tolerance

            if method == 'simplex':
                minuit.simplex(ncall=max_evaluations)
            else:
                minuit.migrad(ncall=max_evaluations)

            # Evaluated before the write-back: a failure here must not leave entries on the undo stack.
            y_calc = fit_function(x, **minuit.values.to_dict())
            self._set_parameter_fit_result(
                minuit, names, stack_status, method, scale_covar, len(y)
            )
            results = self._gen_fit_results(minuit, names, start, y_calc, x, y, weights)
        except Exception as e:
            self._restore_parameter_values()
            raise FitError(e)
        finally:
            global_object.stack.enabled = stack_status
        return results

    def _set_parameter_fit_result(
        self,
        fit_result: iminuit.Minuit,
        names: List[str],
        stack_status: bool,
        method: str,
        scale_covar: bool,
        n_points: int,
    ) -> None:
        """
        Update parameters to their final values and assign a std error
        to them.

        Parameters
        ----------
        fit_result : iminuit.Minuit
            The Minuit object after the minimization.
        names : List[str]
            The Minuit names of the fitted parameters, in the order of
            ``self._cached_pars``.
        stack_status : bool
            Whether the undo stack was enabled.
        method : str
            The algorithm that was run. Only ``'migrad'`` provides a
            covariance, so only then do the parameters get an error.
        scale_covar : bool
            Multiply the variances by the reduced chi-square.
        n_points : int
            Number of fitted data points.
        """
        from easyscience import global_object

        variances = self._variances(fit_result, method, scale_covar, n_points, len(names))
        if stack_status:
            self._restore_parameter_values()
            global_object.stack.enabled = True
            global_object.stack.beginMacro('Fitting routine')
        for index, (name, parameter) in enumerate(zip(names, self._cached_pars.values())):
            parameter.value = fit_result.values[name]
            # None keeps "no uncertainty available" distinguishable from a genuine zero and
            # clears any stale error from a previous fit.
            parameter.error = None if variances is None else float(np.sqrt(variances[index]))

        if stack_status:
            global_object.stack.endMacro()
        covariance = None
        if method == 'migrad' and fit_result.valid and fit_result.covariance is not None:
            covariance = np.array(fit_result.covariance, dtype=float)
        self._store_fit_covariance(covariance, list(self._cached_pars.values()))

        if fit_result.fmin.has_parameters_at_limit:
            global_object.log.getLogger('fitting.minuit').warning(
                'At least one parameter is at its limit after the Minuit fit; '
                'the HESSE uncertainties are not reliable.'
            )

    @staticmethod
    def _variances(
        fit_result: iminuit.Minuit, method: str, scale_covar: bool, n_points: int, n_free: int
    ) -> np.ndarray | None:
        """
        Variances of the fitted parameters, or None if there are none.

        Parameters
        ----------
        fit_result : iminuit.Minuit
            The Minuit object after the minimization.
        method : str
            The algorithm that was run.
        scale_covar : bool
            Multiply the variances by the reduced chi-square, if there
            are degrees of freedom.
        n_points : int
            Number of fitted data points.
        n_free : int
            Number of fitted parameters.

        Returns
        -------
        np.ndarray | None
            The diagonal of the covariance matrix, or None for a method
            without a covariance (the errors of Minuit are then only
            step sizes) and for a minimum that is not valid (for example
            when ``max_evaluations`` was reached, where the covariance
            is only a rough starting estimate).
        """
        if method != 'migrad' or not fit_result.valid or fit_result.covariance is None:
            return None
        variances = np.diag(np.asarray(fit_result.covariance)).copy()
        degrees_of_freedom = n_points - n_free
        if scale_covar and degrees_of_freedom > 0:
            variances *= fit_result.fval / degrees_of_freedom
        return variances

    def _gen_fit_results(
        self,
        fit_result: iminuit.Minuit,
        names: List[str],
        start: dict[str, float],
        y_calc: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        **kwargs: Any,
    ) -> FitResults:
        """
        Convert the Minuit object into the unified ``FitResults``.

        Parameters
        ----------
        fit_result : iminuit.Minuit
            The Minuit object after the minimization. It is kept as
            ``engine_result``.
        names : List[str]
            The Minuit names of the fitted parameters.
        start : dict[str, float]
            The starting value of every fitted parameter.
        y_calc : np.ndarray
            The model evaluated at the minimum.
        x : np.ndarray
            Points that were calculated at.
        y : np.ndarray
            Measured points.
        weights : np.ndarray
            Weights of the measured points.
        **kwargs : Any
            Additional result attributes to copy onto ``FitResults``.

        Returns
        -------
        FitResults
            Fit results container.
        """
        results = FitResults()
        for name, value in kwargs.items():
            if getattr(results, name, False):
                setattr(results, name, value)

        results.success = bool(fit_result.valid)
        results.x = x
        results.y_obs = y
        results.p = {name: float(fit_result.values[name]) for name in names}
        results.p0 = dict(start)
        results.y_calc = y_calc
        results.y_err = 1 / weights
        results.n_evaluations = fit_result.nfcn
        # Minuit has no separate iteration count: every iteration is a function call.
        results.iterations = fit_result.nfcn
        results.message = self._message(fit_result)
        if not results.success:
            from easyscience import global_object

            global_object.log.getLogger('fitting.minuit').warning(results.message)
        results.covariance, results.covariance_parameters = self._covariance_results()
        results.minimizer_engine = self.__class__
        results.fit_args = None
        results.engine_result = fit_result
        return results

    @staticmethod
    def _message(fit_result: iminuit.Minuit) -> str:
        """
        A short description of how the minimization ended.

        Parameters
        ----------
        fit_result : iminuit.Minuit
            The Minuit object after the minimization.

        Returns
        -------
        str
            The message for ``FitResults``.
        """
        if fit_result.fmin.has_reached_call_limit:
            return 'Minuit stopped: max evaluations reached.'
        if not fit_result.valid:
            return 'Minuit did not find a valid minimum.'
        return 'Minuit found a valid minimum.'
