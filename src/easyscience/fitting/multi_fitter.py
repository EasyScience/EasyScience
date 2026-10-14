# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Callable

import numpy as np

from ..base_classes import EasyList
from ..base_classes import ModelBase
from .fitter import Fitter
from .minimizers import FitResults
from .reshaping import inject_x_multi
from .reshaping import reshape_datasets


class MultiFitter(Fitter):
    """
    Extension of Fitter to enable multiple dataset/fit function fitting.

    We can fit these types of data simultaneously:
    - Multiple models on multiple datasets.

    The inherited ``fit`` wrapper from ``Fitter`` is used unchanged,
    including support for forwarding progress callbacks to the active
    minimizer.

    """

    def __init__(
        self,
        fit_objects: Sequence | None = None,
        fit_functions: Sequence[Callable] | None = None,
    ):
        # Both arguments default to None so the constructor can be called
        # empty; normalise to empty sequences so nothing below has to
        # special-case None.
        if fit_objects is None:
            fit_objects = []
        if fit_functions is None:
            fit_functions = []
        # Aggregate the fit objects so a single object can be sent to Fitter.
        # *-unpacking keeps any sequence (list, tuple, etc) working.
        self._fit_objects = EasyList(*fit_objects, protected_types=ModelBase)
        self._fit_functions = list(fit_functions)
        # Initialize with the first of the fit_functions, without this it is
        # not possible to change the fitting engine. With no functions given
        # the Fitter is created with ``None``; the minimizer only stores the
        # callable, so this is harmless until a fit is attempted.
        first_fit_function = self._fit_functions[0] if self._fit_functions else None
        super().__init__(self._fit_objects, first_fit_function)

    @property
    def fit_functions(self) -> list[Callable]:
        """
        Get the per-dataset fit functions, in dataset order.

        Returns
        -------
        list[Callable]
            One fit function per dataset.
        """
        return list(self._fit_functions)

    def _fit_function_wrapper(
        self,
        real_x: list[np.ndarray] | None = None,
        flatten: bool = True,
    ) -> Callable:
        """
        Wrap the per-dataset fit functions into one function that
        evaluates each on its real X (independent) values and
        concatenates the results, flattening if needed.

        Parameters
        ----------
        real_x : list[np.ndarray] | None, default=None
            List of independent x parameters to be injected. By default,
            None.
        flatten : bool, default=True
            Should the result be a flat 1D array? By default, True.

        Returns
        -------
        Callable
            Wrapped optimizer function.
        """
        return inject_x_multi(self._fit_functions, real_x, self._dependent_dims, flatten=flatten)

    _precompute_reshaping = staticmethod(reshape_datasets)

    def _post_compute_reshaping(
        self,
        fit_result_obj: FitResults,
        x: list[np.ndarray],
        y: list[np.ndarray],
    ) -> list[FitResults]:
        """
        Split a multi-fit result object back into per-dataset results.

        Parameters
        ----------
        fit_result_obj : FitResults
            Combined fit result returned by the minimizer.
        x : list[np.ndarray]
            Original x coordinates for each dataset.
        y : list[np.ndarray]
            Original y coordinates for each dataset.

        Returns
        -------
        list[FitResults]
            One fit result object per dataset.
        """

        cls = fit_result_obj.__class__
        sp = 0
        fit_results_list = []
        for idx, this_x in enumerate(x):
            # Create a new Results obj
            current_results = cls()
            ep = sp + int(np.array(self._dependent_dims[idx]).prod())

            #  Fill out the new result obj (see EasyScience.fitting.Fitting_template.FitResults)
            current_results.success = fit_result_obj.success
            current_results.minimizer_engine = fit_result_obj.minimizer_engine
            current_results.p = fit_result_obj.p
            current_results.p0 = fit_result_obj.p0
            current_results.n_evaluations = fit_result_obj.n_evaluations
            current_results.iterations = fit_result_obj.iterations
            current_results.message = fit_result_obj.message
            current_results.x = this_x
            current_results.y_obs = y[idx]
            current_results.y_calc = np.reshape(
                fit_result_obj.y_calc[sp:ep], current_results.y_obs.shape
            )
            current_results.y_err = np.reshape(
                fit_result_obj.y_err[sp:ep], current_results.y_obs.shape
            )
            current_results.engine_result = fit_result_obj.engine_result

            # Attach an additional field for the un-modified results
            current_results.total_results = fit_result_obj
            fit_results_list.append(current_results)
            sp = ep
        return fit_results_list
