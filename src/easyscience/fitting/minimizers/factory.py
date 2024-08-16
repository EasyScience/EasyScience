import warnings
from enum import Enum
from enum import auto
from typing import Callable

from .minimizer_base import MinimizerBase

lmfit_engine_imported = False
try:
    from .minimizer_lmfit import LMFit

    lmfit_engine_imported = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('LMFit minimization is not available. Probably lmfit has not been installed.', ImportWarning, stacklevel=2)

bumps_engine_imported = False
try:
    from .minimizer_bumps import Bumps

    bumps_engine_imported = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('Bumps minimization is not available. Probably bumps has not been installed.', ImportWarning, stacklevel=2)

dfo_engine_imported = False
try:
    from .minimizer_dfo import DFO

    dfo_engine_imported = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('DFO minimization is not available. Probably dfols has not been installed.', ImportWarning, stacklevel=2)


class AvailableMinimizers(Enum):
    if lmfit_engine_imported:
        LMFit = auto()
        LMFit_leastsq = auto()
        LMFit_powell = auto()
        LMFit_cobyla = auto()

    if bumps_engine_imported:
        Bumps = auto()
        Bumps_simplex = auto()
        Bumps_newton = auto()
        Bumps_lm = auto()

    if dfo_engine_imported:
        DFO = auto()
        DFO_leastsq = auto()

# Temporary solution to convert string to enum
def from_string_to_enum(minimizer_name: str) -> AvailableMinimizers:
    if minimizer_name == 'LMFit':
        minmizer_enum = AvailableMinimizers.LMFit
    elif minimizer_name == 'LMFit_leastsq':
        minmizer_enum = AvailableMinimizers.LMFit_leastsq
    elif minimizer_name == 'LMFit_powell':
        minmizer_enum = AvailableMinimizers.LMFit_powell
    elif minimizer_name == 'LMFit_cobyla':
        minmizer_enum = AvailableMinimizers.LMFit_cobyla

    elif minimizer_name == 'Bumps':
        minmizer_enum = AvailableMinimizers.Bumps
    elif minimizer_name == 'Bumps_simplex':
        minmizer_enum = AvailableMinimizers.Bumps_simplex
    elif minimizer_name == 'Bumps_newton':
        minmizer_enum = AvailableMinimizers.Bumps_newton
    elif minimizer_name == 'Bumps_lm':
        minmizer_enum = AvailableMinimizers.Bumps_lm

    elif minimizer_name == 'DFO':
        minmizer_enum = AvailableMinimizers.DFO
    elif minimizer_name == 'DFO_leastsq':
        minmizer_enum = AvailableMinimizers.DFO_leastsq
    else:
        raise ValueError(f"Invalid minimizer name: {minimizer_name}. The following minimizers are available: {[minimize.name for minimize in AvailableMinimizers]}")          # noqa: E501

    return minmizer_enum


def factory(minimizer_enum: AvailableMinimizers, fit_object, fit_function: Callable) -> MinimizerBase:
    if minimizer_enum == AvailableMinimizers.LMFit:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='leastsq')
    elif minimizer_enum == AvailableMinimizers.LMFit_leastsq:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='leastsq')
    elif minimizer_enum == AvailableMinimizers.LMFit_powell:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='powell')
    elif minimizer_enum == AvailableMinimizers.LMFit_cobyla:
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, method='cobyla')

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
