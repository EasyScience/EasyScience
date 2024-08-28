from typing import Callable

from .. import available_minimizers
from ..available_minimizers import AvailableMinimizers
from .minimizer_base import MinimizerBase

if available_minimizers.lmfit_engine_available:
    from .minimizer_lmfit import LMFit
if available_minimizers.dfo_engine_available:
    from .minimizer_dfo import DFO
if available_minimizers.bumps_engine_available:
    from .minimizer_bumps import Bumps


def factory(minimizer_enum: AvailableMinimizers, fit_object, fit_function: Callable) -> MinimizerBase:
    if minimizer_enum == AvailableMinimizers.LMFit:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='leastsq')
    elif minimizer_enum == AvailableMinimizers.LMFit_leastsq:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='leastsq')
    elif minimizer_enum == AvailableMinimizers.LMFit_powell:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='powell')
    elif minimizer_enum == AvailableMinimizers.LMFit_cobyla:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='cobyla')
    elif minimizer_enum == AvailableMinimizers.LMFit_differential_evolution:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='differential_evolution')

    elif minimizer_enum == AvailableMinimizers.Bumps:
        minimizer = Bumps(obj=fit_object, fit_function=fit_function, method='amoeba')
    elif minimizer_enum == AvailableMinimizers.Bumps_simplex:
        minimizer = Bumps(obj=fit_object, fit_function=fit_function, method='amoeba')
    elif minimizer_enum == AvailableMinimizers.Bumps_newton:
        minimizer = Bumps(obj=fit_object, fit_function=fit_function, method='newton')
    elif minimizer_enum == AvailableMinimizers.Bumps_lm:
        minimizer = Bumps(obj=fit_object, fit_function=fit_function, method='lm')

    elif minimizer_enum == AvailableMinimizers.DFO:
        minimizer = DFO(obj=fit_object, fit_function=fit_function, method='leastsq')
    elif minimizer_enum == AvailableMinimizers.DFO_leastsq:
        minimizer = DFO(obj=fit_object, fit_function=fit_function, method='leastsq')

    return minimizer
