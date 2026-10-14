# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Mapping
from typing import Sequence

import numpy as np

from easyscience import global_object
from easyscience.variable import Parameter
from easyscience.variable.covariance import ParameterCovariance


def posterior_covariance(
    parameters_by_name: Mapping[str, Parameter],
    param_names: Sequence[str],
    draws: np.ndarray,
) -> ParameterCovariance | None:
    """
    The covariance of posterior draws as a ``ParameterCovariance``.

    The record is not attached to the parameters: a sampler leaves the
    parameters at the last evaluated point, which is not a result. Call
    ``attach()`` on the record to make it the covariance of the
    parameters.

    Parameters
    ----------
    parameters_by_name : Mapping[str, Parameter]
        The sampled parameters by their unique name.
    param_names : Sequence[str]
        The unique names of the columns of ``draws``.
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.

    Returns
    -------
    ParameterCovariance | None
        The record, with the draws attached and the source
        ``'posterior'``, or None if a name is unknown, there are fewer
        than two draws or the draws are not finite.
    """
    draws = np.asarray(draws, dtype=float)
    log = global_object.log.getLogger('fitting')
    if any(name not in parameters_by_name for name in param_names):
        log.warning('No posterior covariance: a sampled parameter is not known.')
        return None
    if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] != len(param_names):
        log.debug('No posterior covariance: the draws have an unexpected shape.')
        return None
    if not np.all(np.isfinite(draws)):
        log.debug('No posterior covariance: the draws are not finite.')
        return None
    parameters = [parameters_by_name[name] for name in param_names]
    try:
        return ParameterCovariance.from_samples(parameters, draws, names=list(param_names))
    except (TypeError, ValueError) as error:
        log.debug(f'No posterior covariance: {error}')
        return None
