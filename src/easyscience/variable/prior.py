# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Prior distributions of parameters.

A prior is stored on a ``Parameter`` as plain data, a dictionary such as
``{'distribution': 'norm', 'loc': 1.0, 'scale': 0.1}``. The key
``'distribution'`` names a continuous distribution of ``scipy.stats``
and every other key is a keyword argument of that distribution (shape
parameters, ``loc`` and ``scale``). A parameter without a prior has a
uniform prior between its ``min`` and ``max``.

The bounds of a parameter always truncate its prior: the prior of a
parameter is the distribution restricted to ``[min, max]`` and
normalised. ``Prior`` provides that truncated distribution, with the
``ppf`` that maps the unit interval to the parameter, which is what
nested samplers and unit-cube samplers need.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

DISTRIBUTION_KEY = 'distribution'


def validate_prior(spec: Mapping[str, Any]) -> dict[str, Any]:
    """
    Check a prior given as plain data.

    Parameters
    ----------
    spec : Mapping[str, Any]
        ``{'distribution': <name in scipy.stats>, <keyword>: <number>,
        ...}``.

    Returns
    -------
    dict[str, Any]
        A copy of ``spec``.

    Raises
    ------
    TypeError
        If ``spec`` is not a mapping, the name is not a string or a
        keyword is not a number.
    ValueError
        If the distribution is unknown or not continuous, or scipy
        rejects the arguments.
    """  # noqa: DOC503 -- raised by frozen_distribution
    if not isinstance(spec, Mapping):
        raise TypeError(f'A prior must be a dict or None, got {type(spec).__name__}')
    spec = dict(spec)
    name = spec.get(DISTRIBUTION_KEY)
    if not isinstance(name, str):
        raise TypeError(f"A prior needs a distribution name under the key '{DISTRIBUTION_KEY}'")
    for key, value in spec.items():
        if key != DISTRIBUTION_KEY and (
            not isinstance(value, numbers.Real) or isinstance(value, bool)
        ):
            raise TypeError(f'The prior argument {key!r} must be a number, got {value!r}')
    frozen_distribution(spec)
    return spec


def frozen_distribution(spec: Mapping[str, Any]) -> Any:
    """
    The ``scipy.stats`` distribution described by a prior.

    Parameters
    ----------
    spec : Mapping[str, Any]
        The prior as plain data.

    Returns
    -------
    Any
        The frozen distribution.

    Raises
    ------
    ValueError
        If the distribution is unknown or not continuous, or scipy
        rejects the arguments.
    """
    name = spec[DISTRIBUTION_KEY]
    distribution = getattr(stats, name, None) if not name.startswith('_') else None
    if not isinstance(distribution, stats.rv_continuous):
        raise ValueError(f'{name!r} is not a continuous distribution of scipy.stats')
    arguments = {key: value for key, value in spec.items() if key != DISTRIBUTION_KEY}
    try:
        frozen = distribution(**arguments)
        support = frozen.support()
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid arguments for the prior {name!r}: {error}') from error
    if np.isnan(support).any():
        raise ValueError(f'Invalid arguments for the prior {name!r}: {arguments}')
    return frozen


class Prior:
    """
    The prior of a parameter: a distribution truncated to its bounds.

    Parameters
    ----------
    spec : Mapping[str, Any] | None
        The prior as plain data, or None for a uniform prior.
    lower : float
        The lower bound of the parameter.
    upper : float
        The upper bound of the parameter.

    Raises
    ------
    ValueError
        If the prior has no probability inside the bounds.
    """

    def __init__(self, spec: Mapping[str, Any] | None, lower: float, upper: float) -> None:
        self._spec = None if spec is None else validate_prior(spec)
        self._lower = float(lower)
        self._upper = float(upper)
        self._dist = None if self._spec is None else frozen_distribution(self._spec)
        self._cdf_lower = 0.0
        self._mass = 1.0
        if self._dist is not None:
            with np.errstate(all='ignore'):
                self._cdf_lower = float(self._dist.cdf(self._lower))
                self._mass = float(self._dist.cdf(self._upper)) - self._cdf_lower
            if not np.isfinite(self._mass) or self._mass <= 0.0:
                raise ValueError(
                    f'The prior {self._spec} has no probability between '
                    f'{self._lower} and {self._upper}'
                )

    @property
    def spec(self) -> dict[str, Any] | None:
        """The prior as plain data, None for a uniform prior."""
        return None if self._spec is None else dict(self._spec)

    @property
    def lower(self) -> float:
        """The lower limit of the truncated prior."""
        return self._lower

    @property
    def upper(self) -> float:
        """The upper limit of the truncated prior."""
        return self._upper

    @property
    def is_uniform(self) -> bool:
        """Whether this is the default uniform prior."""
        return self._dist is None

    @property
    def is_proper(self) -> bool:
        """Whether the prior can be normalised and sampled with ``ppf``."""
        if self._dist is None:
            return bool(np.isfinite(self._lower) and np.isfinite(self._upper))
        return True

    def logpdf(self, x: Any) -> Any:
        """
        Logarithm of the probability density.

        Parameters
        ----------
        x : Any
            A value or an array of values.

        Returns
        -------
        Any
            ``-inf`` outside the bounds. For an improper uniform prior
            the constant 0.
        """
        x = np.asarray(x, dtype=float)
        inside = (x >= self._lower) & (x <= self._upper)
        with np.errstate(all='ignore'):
            if self._dist is None:
                width = self._upper - self._lower
                constant = -math.log(width) if np.isfinite(width) else 0.0
                result = np.where(inside, constant, -np.inf)
            else:
                result = np.where(inside, self._dist.logpdf(x) - math.log(self._mass), -np.inf)
        return result if result.ndim else float(result)

    def cdf(self, x: Any) -> Any:
        """
        Cumulative distribution function of the truncated prior.

        Parameters
        ----------
        x : Any
            A value or an array of values.

        Returns
        -------
        Any
            Probability between 0 and 1.

        Raises
        ------
        ValueError
            If the prior is not proper.
        """  # noqa: DOC502 -- raised by _require_proper
        self._require_proper()
        x = np.clip(np.asarray(x, dtype=float), self._lower, self._upper)
        if self._dist is None:
            result = (x - self._lower) / (self._upper - self._lower)
        else:
            result = (self._dist.cdf(x) - self._cdf_lower) / self._mass
        result = np.clip(result, 0.0, 1.0)
        return result if result.ndim else float(result)

    def ppf(self, u: Any) -> Any:
        """
        Inverse of ``cdf``, the prior transform from the unit interval
        to the parameter.

        Parameters
        ----------
        u : Any
            A probability or an array of probabilities in ``[0, 1]``.

        Returns
        -------
        Any
            The parameter value, inside the bounds.

        Raises
        ------
        ValueError
            If the prior is not proper.
        """  # noqa: DOC502 -- raised by _require_proper
        self._require_proper()
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        if self._dist is None:
            result = self._lower + u * (self._upper - self._lower)
        else:
            result = np.clip(
                self._dist.ppf(self._cdf_lower + u * self._mass), self._lower, self._upper
            )
        return result if result.ndim else float(result)

    def rvs(self, size: int | None = None, rng: np.random.Generator | None = None) -> Any:
        """
        Random values from the prior.

        Parameters
        ----------
        size : int | None, default=None
            Number of values. By default, None.
        rng : np.random.Generator | None, default=None
            Random number generator. By default, None.

        Returns
        -------
        Any
            One value or an array.
        """
        generator = np.random.default_rng() if rng is None else rng
        return self.ppf(generator.uniform(size=size))

    def _require_proper(self) -> None:
        if not self.is_proper:
            raise ValueError(
                'A uniform prior needs finite bounds: set min and max of the parameter or '
                'give it a prior'
            )

    def __repr__(self) -> str:
        what = 'uniform' if self._dist is None else self._spec
        return f'Prior({what}, [{self._lower}, {self._upper}])'
