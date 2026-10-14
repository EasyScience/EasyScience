# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for ``reshaping.py`` — the data reshaping and fit-function
wrapping shared by ``Fitter``, ``MultiFitter`` and ``Sampler``."""

import numpy as np
import pytest

from easyscience.fitting.reshaping import inject_x
from easyscience.fitting.reshaping import inject_x_multi
from easyscience.fitting.reshaping import reshape_dataset
from easyscience.fitting.reshaping import reshape_datasets


def _grid(n: int, m: int):
    """A vectorized ``(n, m, 2)`` coordinate grid and its ``(n, m)`` dependent values."""
    X, Y = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, n))
    x = np.stack((X, Y), axis=2)
    return x, X + 2 * Y


def _sum_xy(x):
    return x[..., 0] + 2 * x[..., 1]


class TestReshapeDataset:
    def test_1d_dims_are_y_shape(self):
        x = np.linspace(0, 1, 5)
        x_fit, x_new, y_new, w_new, dims = reshape_dataset(x, 2 * x, np.ones(5), vectorized=False)
        assert dims == (5,)
        assert x_fit.shape == y_new.shape == w_new.shape == (5,)

    def test_vectorized_dims_are_y_shape_not_x_shape(self):
        """For multi-dimensional coordinates the dependent dims must exclude
        the coordinate-component axis: ``(2, 3, 2)`` x holds 6 observations."""
        x, y = _grid(2, 3)
        _, x_new, y_new, _, dims = reshape_dataset(x, y, None, vectorized=True)
        assert dims == (2, 3)
        assert y_new.shape == (6,)
        assert x_new.shape == (2, 3, 2)

    def test_non_vectorized_nd_dims_are_y_shape(self):
        x = np.random.default_rng(0).random((6, 2))
        y = np.arange(6.0)
        _, x_new, y_new, _, dims = reshape_dataset(x, y, None, vectorized=False)
        assert dims == (6,)
        assert x_new.shape == (6, 2)


class TestInjectX:
    def test_injects_real_x_and_flattens(self):
        x, y = _grid(2, 3)
        wrapped = inject_x(_sum_xy, x, flatten=True)
        np.testing.assert_allclose(wrapped(np.zeros(6)), y.flatten())


class TestInjectXMulti:
    def test_two_multidimensional_datasets_are_sliced_by_observation_count(self):
        """Regression: slicing by the product of the x shape allocated twice as
        many output positions as observations for vectorized grids, which
        raised on the first dataset and silently clipped on the last."""
        x1, y1 = _grid(2, 3)
        x2, y2 = _grid(3, 4)
        x_fit, x_new, y_new, _, dims = reshape_datasets([x1, x2], [y1, y2], None, vectorized=True)
        assert dims == [(2, 3), (3, 4)]
        assert y_new.shape == (18,)

        wrapped = inject_x_multi([_sum_xy, _sum_xy], x_new, dims)
        np.testing.assert_allclose(wrapped(x_fit), np.hstack([y1.flatten(), y2.flatten()]))

    def test_multidimensional_dataset_first_then_1d(self):
        """Dataset boundaries hold regardless of order; a 2D dataset that is
        not last used to raise a broadcast error."""
        x1, y1 = _grid(2, 3)
        x2 = np.linspace(0, 1, 4)
        y2 = 3 * x2
        x_fit, x_new, y_new, _, dims = reshape_datasets([x1, x2], [y1, y2], None, vectorized=True)
        assert dims == [(2, 3), (4,)]

        wrapped = inject_x_multi([_sum_xy, lambda x: 3 * x], x_new, dims)
        np.testing.assert_allclose(wrapped(x_fit), np.hstack([y1.flatten(), y2]))

    def test_weights_none_for_all_datasets(self):
        x = np.linspace(0, 1, 3)
        _, _, _, w_new, _ = reshape_datasets([x, x], [x, x], [None, None], vectorized=False)
        assert w_new is None

    def test_weights_are_concatenated(self):
        x = np.linspace(0, 1, 3)
        _, _, _, w_new, _ = reshape_datasets(
            [x, x], [x, x], [np.ones(3), 2 * np.ones(3)], vectorized=False
        )
        np.testing.assert_array_equal(w_new, [1, 1, 1, 2, 2, 2])


@pytest.mark.parametrize('vectorized', [False, True])
def test_shape_mismatch_raises(vectorized):
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError, match='shape of the x and y data must be the same'):
        reshape_dataset(x, np.zeros(3), None, vectorized=vectorized)
