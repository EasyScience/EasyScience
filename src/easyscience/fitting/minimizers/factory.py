import warnings
from enum import Enum
from enum import auto
from typing import Callable

from .minimizer_base import MinimizerBase

lmfit_minimizer_imported = False
try:
    from .minimizer_lmfit import LMFit

    lmfit_minimizer_imported = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('LMFit minimization is not available. Probably lmfit has not been installed.', ImportWarning, stacklevel=2)

bumps_minimizer_imported = False
try:
    from .minimizer_bumps import Bumps

    bumps_minimizer_imported = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('Bummps minimization is not available. Probably bumps has not been installed.', ImportWarning, stacklevel=2)

dfo_minimizer_imported = False
try:
    from .minimizer_dfo import DFO

    dfo_minimizer_imported = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn('DFO minimization is not available. Probably dfols has not been installed.', ImportWarning, stacklevel=2)


class AvailableMinimizers(Enum):
    if lmfit_minimizer_imported:
        LMFit = auto()
        LMFit_leastsq = auto()
        LMFit_powell = auto()
        LMFit_cobyla = auto()

    if bumps_minimizer_imported:
        Bumps = auto()
        Bumps_simplex = auto()
        Bumps_newton = auto()
        Bumps_lm = auto()

    if dfo_minimizer_imported:
        DFO = auto()


def from_string(engine_name: str) -> AvailableMinimizers:
    if engine_name == 'lmfit':
        engine_enum = AvailableMinimizers.LMFit
    elif engine_name == 'lmfit-leastsq':
        engine_enum = AvailableMinimizers.LMFit_leastsq
    elif engine_name == 'lmfit-powell':
        engine_enum = AvailableMinimizers.LMFit_powell
    elif engine_name == 'lmfit-cobyla':
        engine_enum = AvailableMinimizers.LMFit_cobyla

    elif engine_name == 'bumps':
        engine_enum = AvailableMinimizers.Bumps
    elif engine_name == 'bumps-simplex':
        engine_enum = AvailableMinimizers.Bumps_simplex
    elif engine_name == 'bumps-newton':
        engine_enum = AvailableMinimizers.Bumps_newton
    elif engine_name == 'bumps-lm':
        engine_enum = AvailableMinimizers.Bumps_lm

    elif engine_name == 'dfo_ls':
        engine_enum = AvailableMinimizers.DFO

    return engine_enum


def minimizer_class_factory(minimizer_enum: AvailableMinimizers, fit_object, fit_function: Callable) -> MinimizerBase:
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
        minimizer = DFO(obj=fit_object, fit_function=fit_function)

    return minimizer
