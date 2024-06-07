from enum import Enum

from .minimizer_bumps import Bumps
from .minimizer_dfo import DFO
from .minimizer_lmfit import LMFit


class Minimizers(Enum):
    Bumps = 1
    DFO = 2
    LMFit = 3


def from_string(engine_name: str) -> Minimizers:
    if engine_name == 'bumps':
        engine_enum = Minimizers.Bumps
    if engine_name == 'lmfit':
        engine_enum = Minimizers.LMFit
    if engine_name == 'dfo_ls':
        engine_enum = Minimizers.DFO
    return engine_enum


def minimizer_class_factory(minimizer: Minimizers):
    if minimizer == Minimizers.Bumps:
        return Bumps
    elif minimizer == Minimizers.DFO:
        return DFO
    elif minimizer == Minimizers.LMFit:
        return LMFit
