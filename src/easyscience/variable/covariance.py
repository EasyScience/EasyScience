# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING
from typing import Sequence

import numpy as np

if TYPE_CHECKING:
    from .parameter import Parameter

SOURCES = ('hessian', 'posterior', 'user')

_SYMMETRY_TOLERANCE = 1e-8


def _refresh_dependents(parameters: Sequence[Parameter | None]) -> None:
    """
    Recalculate the dependent parameters of ``parameters``, which use the
    covariance of their sources for their variance.
    """
    seen = set()
    for parameter in parameters:
        if parameter is None:
            continue
        for observer in list(parameter._observers):
            if id(observer) not in seen:
                seen.add(id(observer))
                observer._update()


def _same(current: float | None, stamped: float | None) -> bool:
    """
    Whether a value or variance is still the one that was stamped; a
    round trip through ``error`` changes a variance by rounding only.
    """
    if current is None or stamped is None:
        return current is stamped
    return math.isclose(current, stamped, rel_tol=1e-12, abs_tol=0.0)


def _none() -> None:
    """Return None; the reconstruction function used when pickling."""
    return None


class ParameterCovariance:
    """
    The covariance of the parameters of one fit or sampling run.

    The record keeps weak references to its parameters, so it never
    keeps a parameter alive. The covariance is the unscaled one: it
    assumes that the weights of the fit are exactly ``1 / sigma``. Use
    :meth:`scaled` for a covariance scaled by, for example, the reduced
    chi-square. A record is immutable, is not serialized and is not
    copied: a copy of a parameter has no covariance.

    A record is attached to its parameters with :meth:`attach`. It is
    then only reported by the parameters while it is *current*: every
    participant still has the value and variance it had when the record
    was attached and none is fixed. Setting a value or an error, fixing
    a parameter or undoing the fit therefore makes the record stale
    without any bookkeeping in the setters.

    Parameters
    ----------
    parameters : Sequence[Parameter]
        The independent parameters, in the order of the matrix.
    covariance : np.ndarray
        The square covariance matrix.
    source : str, default='hessian'
        Where the covariance comes from: ``'hessian'``,
        ``'posterior'`` or ``'user'``. By default, 'hessian'.
    samples : np.ndarray | None, default=None
        Optional draws of shape ``(n_draws, n_parameters)``. By default,
        None.
    names : Sequence[str] | None, default=None
        Labels of the parameters. By default, the display names.

    Raises
    ------
    TypeError
        If a parameter is not a ``Parameter``.
    ValueError
        If the source is unknown, the matrix has the wrong shape, is not
        finite or not symmetric, has a negative variance, or the
        parameters contain duplicates or a dependent parameter.
    """

    __slots__ = ('_refs', '_names', '_cov', '_samples', '_source', '_stamp')

    def __init__(
        self,
        parameters: Sequence[Parameter],
        covariance: np.ndarray,
        *,
        source: str = 'hessian',
        samples: np.ndarray | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        from .parameter import Parameter

        if source not in SOURCES:
            raise ValueError(f'source must be one of {SOURCES}, got {source!r}')
        parameters = list(parameters)
        for parameter in parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError(f'Expected Parameter objects, got {type(parameter).__name__}')
            if not parameter.independent:
                raise ValueError('A dependent parameter can not be part of a covariance')
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError('The parameters of a covariance must be distinct')

        matrix = np.array(covariance, dtype=float)
        size = len(parameters)
        if matrix.shape != (size, size):
            raise ValueError(f'Expected a {size}x{size} covariance matrix, got {matrix.shape}')
        if not np.all(np.isfinite(matrix)):
            raise ValueError('The covariance matrix must be finite')
        scale = max(float(np.max(np.abs(matrix))), 1e-300) if size else 1.0
        if np.max(np.abs(matrix - matrix.T), initial=0.0) > _SYMMETRY_TOLERANCE * scale:
            raise ValueError('The covariance matrix must be symmetric')
        matrix = (matrix + matrix.T) / 2
        if np.any(np.diag(matrix) < 0):
            raise ValueError('The covariance matrix has a negative variance')
        matrix.flags.writeable = False

        if names is None:
            names = [parameter.display_name for parameter in parameters]
        if len(names) != size:
            raise ValueError('There must be one name per parameter')

        if samples is not None:
            samples = np.array(samples, dtype=float)
            if samples.ndim != 2 or samples.shape[1] != size:
                raise ValueError(
                    f'Expected samples of shape (n_draws, {size}), got {samples.shape}'
                )
            samples.flags.writeable = False

        self._refs = [weakref.ref(parameter) for parameter in parameters]
        self._names = tuple(str(name) for name in names)
        self._cov = matrix
        self._samples = samples
        self._source = source
        self._stamp = None

    @classmethod
    def from_samples(
        cls,
        parameters: Sequence[Parameter],
        samples: np.ndarray,
        *,
        source: str = 'posterior',
        names: Sequence[str] | None = None,
    ) -> ParameterCovariance:
        """
        Build a record from draws of the parameters.

        Parameters
        ----------
        parameters : Sequence[Parameter]
            The parameters, in the order of the columns of ``samples``.
        samples : np.ndarray
            Draws of shape ``(n_draws, n_parameters)``.
        source : str, default='posterior'
            Source of the record. By default, 'posterior'.
        names : Sequence[str] | None, default=None
            Labels of the parameters. By default, the display names.

        Returns
        -------
        ParameterCovariance
            The covariance of the draws, with the draws attached.

        Raises
        ------
        ValueError
            If there are fewer than two draws or they are not 2D.
        """
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[0] < 2:
            raise ValueError('At least two draws of shape (n_draws, n_parameters) are needed')
        covariance = np.atleast_2d(np.cov(samples, rowvar=False))
        return cls(parameters, covariance, source=source, samples=samples, names=names)

    @property
    def parameters(self) -> list[Parameter | None]:
        """The parameters in matrix order; ``None`` for a collected one."""
        return [ref() for ref in self._refs]

    @property
    def names(self) -> tuple[str, ...]:
        """The labels of the parameters, fixed when the record was made."""
        return self._names

    @property
    def covariance(self) -> np.ndarray:
        """The read-only covariance matrix."""
        return self._cov

    @property
    def samples(self) -> np.ndarray | None:
        """The read-only draws, or None."""
        return self._samples

    @property
    def source(self) -> str:
        """Where the covariance comes from."""
        return self._source

    @property
    def std(self) -> np.ndarray:
        """The standard deviations, the square root of the diagonal."""
        return np.sqrt(np.diag(self._cov))

    @property
    def correlation(self) -> np.ndarray:
        """
        The correlation matrix. A parameter without variance has a
        correlation of 0 with the others and 1 with itself.
        """
        std = self.std
        outer = np.outer(std, std)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.where(outer > 0, self._cov / np.where(outer > 0, outer, 1.0), 0.0)
        np.fill_diagonal(matrix, 1.0)
        return np.clip(matrix, -1.0, 1.0)

    @property
    def is_current(self) -> bool:
        """
        Whether every participant still exists, still has the value and
        variance it had when the record was attached (to rounding) and is
        not fixed.
        """
        if self._stamp is None:
            return False
        for parameter, stamp in zip(self.parameters, self._stamp):
            if parameter is None or stamp is None or parameter.fixed:
                return False
            if not _same(parameter.value, stamp[0]) or not _same(parameter.variance, stamp[1]):
                return False
        return True

    def index_of(self, parameter: Parameter) -> int | None:
        """
        Position of a parameter in the matrix.

        Parameters
        ----------
        parameter : Parameter
            The parameter, compared by identity.

        Returns
        -------
        int | None
            The index, or None if the parameter is not part of it.
        """
        for index, ref in enumerate(self._refs):
            if ref() is parameter:
                return index
        return None

    def covariance_of(self, first: Parameter, second: Parameter) -> float | None:
        """
        Covariance of two parameters.

        Parameters
        ----------
        first : Parameter
            The first parameter.
        second : Parameter
            The second parameter.

        Returns
        -------
        float | None
            The covariance, or None if either is not part of the record.
        """
        i, j = self.index_of(first), self.index_of(second)
        if i is None or j is None:
            return None
        return float(self._cov[i, j])

    def correlation_of(self, first: Parameter, second: Parameter) -> float:
        """
        Correlation coefficient of two parameters.

        Parameters
        ----------
        first : Parameter
            The first parameter.
        second : Parameter
            The second parameter.

        Returns
        -------
        float
            1.0 for the same object, 0.0 if either is not part of the
            record, otherwise the coefficient.
        """
        if first is second:
            return 1.0
        i, j = self.index_of(first), self.index_of(second)
        if i is None or j is None:
            return 0.0
        denominator = math.sqrt(self._cov[i, i] * self._cov[j, j])
        if not denominator > 0:
            return 0.0
        return float(np.clip(self._cov[i, j] / denominator, -1.0, 1.0))

    def submatrix(self, parameters: Sequence[Parameter]) -> np.ndarray:
        """
        The covariance of a subset of the parameters, in the given order.

        Parameters
        ----------
        parameters : Sequence[Parameter]
            Parameters that are all part of the record.

        Returns
        -------
        np.ndarray
            The square sub-matrix.

        Raises
        ------
        KeyError
            If a parameter is not part of the record.
        """
        indices = []
        for parameter in parameters:
            index = self.index_of(parameter)
            if index is None:
                raise KeyError('The parameter is not part of this covariance')
            indices.append(index)
        return self._cov[np.ix_(indices, indices)].copy()

    def scaled(self, factor: float) -> ParameterCovariance:
        """
        A new record with the covariance multiplied by a factor, for
        example the reduced chi-square. It is not attached and has no
        draws.

        Parameters
        ----------
        factor : float
            A non-negative, finite factor.

        Returns
        -------
        ParameterCovariance
            The scaled record.

        Raises
        ------
        ValueError
            If the factor is negative or not finite, or a parameter no
            longer exists.
        """
        if not np.isfinite(factor) or factor < 0:
            raise ValueError('The scale factor must be finite and not negative')
        parameters = self.parameters
        if any(parameter is None for parameter in parameters):
            raise ValueError('A parameter of this covariance no longer exists')
        # The draws are not scaled, so the scaled record has none.
        return ParameterCovariance(
            parameters, self._cov * factor, source=self._source, names=self._names
        )

    def attach(self) -> ParameterCovariance:
        """
        Make this the covariance of its parameters, replacing any other.
        The current value and variance of every parameter are recorded,
        so call this after the values and errors are written.

        Returns
        -------
        ParameterCovariance
            This record.
        """
        parameters = self.parameters
        self._stamp = tuple(
            None if parameter is None else (parameter.value, parameter.variance)
            for parameter in parameters
        )
        for parameter in parameters:
            if parameter is not None:
                parameter._covariance_record = self
        _refresh_dependents(parameters)
        return self

    def detach(self) -> None:
        """Remove this record from the parameters that still hold it."""
        parameters = self.parameters
        for parameter in parameters:
            if parameter is not None and parameter._covariance_record is self:
                parameter._covariance_record = None
        _refresh_dependents(parameters)

    def __len__(self) -> int:
        return len(self._refs)

    def __contains__(self, parameter: object) -> bool:
        return self.index_of(parameter) is not None

    def __copy__(self) -> ParameterCovariance:
        return self

    def __deepcopy__(self, memo: dict) -> ParameterCovariance:
        return self

    def __reduce__(self) -> tuple:
        return (_none, ())

    def __repr__(self) -> str:
        return f'ParameterCovariance({self._source}, {len(self)} parameters: {", ".join(self._names)})'
