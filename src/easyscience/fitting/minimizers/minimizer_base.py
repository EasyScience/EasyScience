# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import List
from typing import Sequence

import numpy as np

from easyscience.variable import Parameter
from easyscience.variable.covariance import ParameterCovariance

from ..available_minimizers import AvailableMinimizers
from ..engine_base import PARAMETER_PREFIX
from ..engine_base import EngineBase
from .utils import FitError
from .utils import FitResults


class MinimizerBase(EngineBase):
    """
    This template class is the basis for all minimizer engines in
    ``EasyScience``.
    """

    def __init__(
        self,
        obj,  #: ModelBase,
        fit_function: Callable,
        minimizer_enum: AvailableMinimizers,
    ):  # todo after constraint changes, add type hint: obj: ModelBase  # noqa: E501
        if minimizer_enum.method not in self.supported_methods():
            raise FitError(f'Method {minimizer_enum.method} not available in {self.__class__}')
        super().__init__(obj=obj, fit_function=fit_function)
        self._minimizer_enum = minimizer_enum
        self._method = minimizer_enum.method
        self._cached_model = None
        self._covariance: ParameterCovariance | None = None

    @property
    def enum(self) -> AvailableMinimizers:
        return self._minimizer_enum

    @property
    def name(self) -> str:
        return self._minimizer_enum.name

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        method: str | None = None,
        tolerance: float | None = None,
        max_evaluations: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit using the  engine.

        Parameters
        ----------
        x : np.ndarray
            Points to be calculated at.
        y : np.ndarray
            Measured points.
        weights : np.ndarray
            Weights for supplied measured points.
        method : str | None, default=None
            Method for the minimizer to use. By default, None.
        tolerance : float | None, default=None
            Requested convergence tolerance. By default, None.
        max_evaluations : int | None, default=None
            Maximum number of objective evaluations. By default, None.
        progress_callback : Callable[[dict], None] | None, default=None
            Optional progress callback. Its return value is ignored by
            every backend; use ``abort_test`` to stop a running fit. By
            default, None.
        **kwargs :
            Additional arguments for the fitting function.

        Returns
        -------
        FitResults
            Fit results.
        """

    def _get_method_kwargs(self, passed_method: str | None = None) -> dict[str, str]:
        if passed_method is not None:
            if passed_method not in self.supported_methods():
                raise FitError(f'Method {passed_method} not available in {self.__class__}')
            return {'method': passed_method}

        if self._method is not None:
            return {'method': self._method}

        return {}

    @staticmethod
    @abstractmethod
    def supported_methods() -> List[str]:
        """
        Return a list of supported methods for the minimizer.

        Returns
        -------
        List[str]
            List of supported methods.
        """

    @staticmethod
    @abstractmethod
    def all_methods() -> List[str]:
        """
        Return a list of all available methods for the minimizer.

        Returns
        -------
        List[str]
            List of all available methods.
        """

    @staticmethod
    def _covariance_from_jacobian(jacobian: np.ndarray) -> np.ndarray | None:
        """
        Covariance matrix ``(J^T J)^-1`` of a least-squares fit.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian of the weighted residuals, shape
            ``(n_points, n_parameters)``. With weights of ``1 / sigma``
            the result is the unscaled covariance.

        Returns
        -------
        np.ndarray | None
            The covariance matrix, or None (with a logged warning) if
            ``J^T J`` is singular or the Jacobian is not finite.
        """
        from easyscience import global_object

        jacobian = np.asarray(jacobian, dtype=float)
        normal_matrix = jacobian.T @ jacobian
        if (
            np.all(np.isfinite(normal_matrix))
            and np.linalg.cond(normal_matrix) < 1 / np.finfo(float).eps
        ):
            covariance = np.linalg.inv(normal_matrix)
            return (covariance + covariance.T) / 2
        global_object.log.getLogger('fitting').warning(
            'The Jacobian is singular or not finite; no covariance or errors are available.'
        )
        return None

    def _store_fit_covariance(
        self,
        covariance: np.ndarray | None,
        parameters: Sequence[Parameter],
        source: str = 'hessian',
    ) -> ParameterCovariance | None:
        """
        Make the covariance of a finished fit the covariance of its
        parameters. Call it after the values and errors are written.

        Parameters
        ----------
        covariance : np.ndarray | None
            The unscaled covariance in the order of ``parameters``, or
            None if the fit has none.
        parameters : Sequence[Parameter]
            The fitted parameters, in the order of the matrix.
        source : str, default='hessian'
            Source of the covariance. By default, 'hessian'.

        Returns
        -------
        ParameterCovariance | None
            The attached record, or None if there is no valid
            covariance. A record that the parameters held before is
            then removed.
        """
        from easyscience import global_object

        for parameter in parameters:
            previous = getattr(parameter, '_covariance_record', None)
            if previous is not None:
                previous.detach()
        self._covariance = None
        if not isinstance(covariance, np.ndarray) or not len(parameters):
            return None
        try:
            record = ParameterCovariance(parameters, covariance, source=source)
        except (TypeError, ValueError) as error:
            global_object.log.getLogger('fitting').warning(f'No covariance stored: {error}')
            return None
        self._covariance = record.attach()
        return self._covariance

    def _covariance_results(self) -> tuple[ParameterCovariance | None, list[str]]:
        """
        The stored covariance and the labels of its parameters.

        Returns
        -------
        tuple[ParameterCovariance | None, list[str]]
            The record and the labels (the keys of ``FitResults.p``) in
            matrix order; ``(None, [])`` if there is none.
        """
        record = self._covariance
        if record is None:
            return None, []
        keys = {id(parameter): key for key, parameter in self._cached_pars.items()}
        labels = [keys.get(id(parameter)) for parameter in record.parameters]
        if None in labels:
            return None, []
        return record, [PARAMETER_PREFIX + str(label) for label in labels]
