# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Sequence

import numpy as np

from easyscience.variable import Parameter
from easyscience.variable.prior import Prior


class PriorTransform:
    """
    The prior transform of a set of parameters.

    It maps the unit hypercube to the parameter space through the
    inverse cumulative distribution (``ppf``) of the prior of each
    parameter, truncated to its bounds. This is what nested samplers
    (for example ``dynesty``) need, and it lets a sampler work in the
    unit cube instead of in parameter space, where a parameter without
    finite bounds is a problem.

    Parameters
    ----------
    parameters : Sequence[Parameter]
        The parameters, in the order of the coordinates.

    Raises
    ------
    ValueError
        If a parameter has no proper prior: a uniform prior needs finite
        ``min`` and ``max``.
    """

    def __init__(self, parameters: Sequence[Parameter]) -> None:
        self._parameters = list(parameters)
        self._priors: list[Prior] = []
        for parameter in self._parameters:
            prior = parameter.prior_distribution
            if not prior.is_proper:
                raise ValueError(
                    f"The parameter '{parameter.display_name}' has no proper prior: give it "
                    'finite min and max, or a prior distribution.'
                )
            self._priors.append(prior)

    @property
    def parameters(self) -> list[Parameter]:
        """The parameters, in the order of the coordinates."""
        return list(self._parameters)

    def __len__(self) -> int:
        return len(self._priors)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """
        Map points of the unit hypercube to parameter values.

        Parameters
        ----------
        u : np.ndarray
            Coordinates in ``[0, 1]``, shape ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.ndarray
            The parameter values, same shape as ``u``.
        """
        u = self._check(u)
        return np.stack([prior.ppf(u[..., i]) for i, prior in enumerate(self._priors)], axis=-1)

    def inverse(self, theta: np.ndarray) -> np.ndarray:
        """
        Map parameter values to the unit hypercube.

        Parameters
        ----------
        theta : np.ndarray
            Parameter values, shape ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.ndarray
            Coordinates in ``[0, 1]``, same shape as ``theta``.
        """
        theta = self._check(theta)
        return np.stack(
            [prior.cdf(theta[..., i]) for i, prior in enumerate(self._priors)], axis=-1
        )

    def log_prior(self, theta: np.ndarray) -> np.ndarray | float:
        """
        Logarithm of the joint prior density.

        Parameters
        ----------
        theta : np.ndarray
            Parameter values, shape ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.ndarray | float
            The sum over the parameters; ``-inf`` outside the bounds.
        """
        theta = self._check(theta)
        return np.sum(
            [prior.logpdf(theta[..., i]) for i, prior in enumerate(self._priors)], axis=0
        )

    def _check(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape[-1:] != (len(self._priors),):
            raise ValueError(
                f'Expected {len(self._priors)} coordinates in the last axis, got {values.shape}'
            )
        return values
