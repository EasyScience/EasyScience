# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from bumps.bounds import Bounds
from bumps.parameter import Parameter as BumpsParameter
from scipy import stats

from easyscience.variable.prior import Prior


class PriorBounds(Bounds):
    """
    The prior of an EasyScience ``Parameter`` as a BUMPS prior.

    The prior is truncated to the bounds of the parameter, so the
    limits that BUMPS uses as hard walls are the bounds of the
    parameter. BUMPS adds ``nllf`` (minus the log of the prior density)
    to the cost of every point, which makes a fit with a prior a MAP
    fit and a sampling run sample the posterior.

    Parameters
    ----------
    prior : Prior
        The truncated prior of the parameter.
    """

    def __init__(self, prior: Prior) -> None:
        self.prior = prior

    @property
    def limits(self) -> tuple[float, float]:
        """Hard limits: the bounds of the parameter."""
        return (self.prior.lower, self.prior.upper)

    @property
    def dof(self) -> int:
        """Degrees of freedom the prior adds, as BUMPS counts them."""
        return 1

    def get01(self, x: float) -> float:
        """Convert a value into the unit interval."""
        return self.prior.cdf(x)

    def put01(self, v: float) -> float:
        """Convert the unit interval into a value."""
        return self.prior.ppf(v)

    def getfull(self, x: float) -> float:
        """Convert a value into the whole real line."""
        probability = np.clip(self.prior.cdf(x), 1e-300, 1 - 1e-16)
        return float(stats.norm.ppf(probability))

    def putfull(self, v: float) -> float:
        """Convert the whole real line into a value."""
        return self.prior.ppf(stats.norm.cdf(v))

    def random(self, n: int = 1, target: float = 1.0) -> np.ndarray:
        """Random values from the prior."""
        return self.prior.rvs(n)

    def nllf(self, value: float) -> float:
        """Minus the log of the prior density; inf outside the bounds."""
        return float(-self.prior.logpdf(value))

    def residual(self, value: float) -> float:
        """The prior as a normally distributed residual."""
        probability = np.clip(self.prior.cdf(value), 1e-300, 1 - 1e-16)
        return float(stats.norm.ppf(probability))

    def to_dict(self) -> dict:
        """The prior as plain data."""
        return {'type': type(self).__name__, 'limits': self.limits, 'prior': self.prior.spec}

    def __str__(self) -> str:
        return str(self.prior)


class PriorParameter(BumpsParameter):
    """
    A BUMPS parameter that keeps the prior of an EasyScience parameter.

    BUMPS rebuilds the prior of every parameter from its bounds and
    distribution whenever it assembles a problem; this parameter puts
    its ``PriorBounds`` there instead.
    """

    easyscience_prior: PriorBounds | None = None

    def reset_prior(self) -> None:
        """Use the EasyScience prior if there is one."""
        if self.easyscience_prior is None:
            super().reset_prior()
        else:
            self.prior = self.easyscience_prior
