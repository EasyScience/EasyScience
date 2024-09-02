import warnings
from enum import Enum
from enum import auto

import pkg_resources

installed_packages = {pkg.key for pkg in pkg_resources.working_set}

# Change to importlib.metadata when Python 3.10 is the minimum version
# import importlib.metadata
# installed_packages = [x.name for x in importlib.metadata.distributions()]

lmfit_engine_available = False
if 'lmfit' in installed_packages:
    lmfit_engine_available = True
else:
    # TODO make this a proper message (use logging?)
    warnings.warn('LMFit minimization is not available. Probably lmfit has not been installed.', ImportWarning, stacklevel=2)

bumps_engine_available = False
if 'bumps' in installed_packages:
    bumps_engine_available = True
else:
    # TODO make this a proper message (use logging?)
    warnings.warn('Bumps minimization is not available. Probably bumps has not been installed.', ImportWarning, stacklevel=2)

dfo_engine_available = False
if 'dfo-ls' in installed_packages:
    dfo_engine_available = True
else:
    # TODO make this a proper message (use logging?)
    warnings.warn('DFO minimization is not available. Probably dfols has not been installed.', ImportWarning, stacklevel=2)


class AvailableMinimizers(Enum):
    if lmfit_engine_available:
        LMFit = auto()
        LMFit_leastsq = auto()
        LMFit_powell = auto()
        LMFit_cobyla = auto()
        LMFit_differential_evolution = auto()
        LMFit_scipy_least_squares = auto()

    if bumps_engine_available:
        Bumps = auto()
        Bumps_simplex = auto()
        Bumps_newton = auto()
        Bumps_lm = auto()

    if dfo_engine_available:
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
    elif minimizer_name == 'LMFit_differential_evolution':
        minmizer_enum = AvailableMinimizers.LMFit_differential_evolution
    elif minimizer_name == 'LMFit_scipy_least_squares':
        minmizer_enum = AvailableMinimizers.LMFit_scipy_least_squares

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
        raise ValueError(
            f'Invalid minimizer name: {minimizer_name}. The following minimizers are available: {[minimize.name for minimize in AvailableMinimizers]}'  # noqa: E501
        )

    return minmizer_enum
