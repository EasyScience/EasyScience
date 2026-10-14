# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Data reshaping and fit-function wrapping shared by ``Fitter`` and ``Sampler``.

Both fitting and sampling provide the engine witha flat 1-D ``y`` array
and a dummy 1-D ``x`` index array, while the user's fit function is called
with the real ``x``. The helpers here do the translation in one place:
``reshape_dataset``/``inject_x`` do one dataset,
``reshape_datasets``/``inject_x_multi`` do a list of datasets.
"""

from __future__ import annotations

import functools
from typing import Callable

import numpy as np


def reshape_dataset(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    vectorized: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[int, ...]]:
    """
    Check the dimensions of the inputs and reshape if necessary.

    Parameters
    ----------
    x : np.ndarray
        Independent points; 1-D, or ND with the coordinate components on
        the last axis.
    y : np.ndarray
        Dependent points, one per observation.
    weights : np.ndarray | None
        Optional weights for the fit.
    vectorized : bool
        Whether ``x`` already stores vectorized coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[int, ...]]
        Dummy x index array for the engine, reshaped x values, flattened
        y values, flattened weights, and the shape of ``y`` (the dependent
        dimensions; its product is the number of observations, which the
        multi-dataset helpers use to slice the combined output).

    Raises
    ------
    ValueError
        If the shapes of ``x`` and ``y`` are incompatible.
    """
    # Make sure that they are np arrays
    x_new = np.array(x)
    y_new = np.array(y)
    # Get the shapes
    x_shape = x_new.shape
    y_shape = y_new.shape
    # Check if the x data is 1D
    if len(x_shape) > 1:
        # It is ND data
        # Check if the data is vectorized. i.e. should x be [NxMx...x Ndims]
        if vectorized:
            # Assert that the shapes are the same
            if np.all(x_shape[:-1] != y_new.shape):
                raise ValueError('The shape of the x and y data must be the same')
            # If so do nothing but note that the data is vectorized
            # x_shape = (-1,) # Should this be done?
        else:
            # Assert that the shapes are the same
            if np.prod(x_new.shape[:-1]) != y_new.size:
                raise ValueError('The number of elements in x and y data must be the same')
            # Reshape the data to be [len(NxMx..), Ndims] i.e. flatten to columns
            x_new = x_new.reshape(-1, x_shape[-1], order='F')
    else:
        # Assert that the shapes are the same
        if np.all(x_shape != y_new.shape):
            raise ValueError('The shape of the x and y data must be the same')
        # It is 1D data
        x_new = x.flatten()
    # The optimizer needs a 1D array, flatten the y data
    y_new = y_new.flatten()
    if weights is not None:
        weights = np.array(weights).flatten()
    # Make a 'dummy' x array for the fit function
    x_for_fit = np.array(range(y_new.size))
    return x_for_fit, x_new, y_new, weights, y_shape


def reshape_datasets(
    x: list[np.ndarray],
    y: list[np.ndarray],
    weights: list[np.ndarray] | None,
    vectorized: bool,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray | None, list[tuple[int, ...]]]:
    """
    Convert a list of X's and Y's to an acceptable shape for fitting.

    Parameters
    ----------
    x : list[np.ndarray]
        List of independent variables.
    y : list[np.ndarray]
        List of dependent variables.
    weights : list[np.ndarray] | None
        Optional weights for each dataset.
    vectorized : bool
        When ``True``, each x array may be multi-dimensional (e.g. an
        ``(N, M, 2)`` grid for a 2D model) and is left as-is. When
        ``False`` (default), each x array is expected to be 1-D.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray | None, list[tuple[int, ...]]]
        Dummy x index array for the engine, per-dataset reshaped x values,
        concatenated y values, concatenated weights, and the per-dataset
        dependent dimensions.
    """
    if weights is None:
        weights = [None] * len(x)
    x_new, y_new, w_new, dims = [], [], [], []
    for _x, _y, _w in zip(x, y, weights):
        _, _x_new, _y_new, _w_new, _dims = reshape_dataset(_x, _y, _w, vectorized)
        x_new.append(_x_new)
        y_new.append(_y_new)
        w_new.append(_w_new)
        dims.append(_dims)
    y_new = np.hstack(y_new)
    w_new = None if w_new[0] is None else np.hstack(w_new)
    x_fit = np.linspace(0, y_new.size - 1, y_new.size)
    return x_fit, x_new, y_new, w_new, dims


def inject_x(
    fit_function: Callable,
    real_x: np.ndarray | None = None,
    flatten: bool = True,
) -> Callable:
    """
    Wrap a fit function so it ignores the engine's dummy x and evaluates
    on ``real_x`` instead, flattening the result if needed.

    Parameters
    ----------
    fit_function : Callable
        The user's fit function.
    real_x : np.ndarray | None, default=None
        Independent x values to be injected. By default, None.
    flatten : bool, default=True
        Should the result be a flat 1D array? By default, True.

    Returns
    -------
    Callable
        Wrapped optimizer function.
    """

    @functools.wraps(fit_function)
    def wrapped_fit_function(x, **kwargs):
        if real_x is not None:
            x = real_x
        dependent = fit_function(x, **kwargs)
        if flatten:
            dependent = dependent.flatten()
        return dependent

    return wrapped_fit_function


def inject_x_multi(
    fit_functions: list[Callable],
    real_x: list[np.ndarray],
    dims: list[tuple[int, ...]],
    flatten: bool = True,
) -> Callable:
    """
    Wrap one fit function per dataset into a single function whose output
    is the concatenation of the per-dataset outputs.

    Parameters
    ----------
    fit_functions : list[Callable]
        One fit function per dataset.
    real_x : list[np.ndarray]
        One independent x array per dataset, injected into the matching
        fit function.
    dims : list[tuple[int, ...]]
        Per-dataset dependent (``y``) shapes used to slice the combined
        output, as returned by ``reshape_datasets``.
    flatten : bool, default=True
        Should each per-dataset result be flattened? By default, True.

    Returns
    -------
    Callable
        Wrapped optimizer function.
    """
    wrapped_fns = [
        inject_x(this_fun, this_x, flatten=flatten)
        for this_x, this_fun in zip(real_x, fit_functions)
    ]

    def wrapped_fun(x, **kwargs):
        # Generate an empty Y based on x
        y = np.zeros_like(x)
        i = 0
        # Iterate through wrapped functions, passing the WRONG x, the correct
        # x was injected in the step above.
        for wrapped, dim in zip(wrapped_fns, dims):
            ep = i + np.prod(dim)
            y[i:ep] = wrapped(x, **kwargs)
            i = ep
        return y

    return wrapped_fun
