import warnings

import pint

from .global_object import GlobalObject

# Must be executed before any other imports
ureg = pint.UnitRegistry()
global_object = GlobalObject()
global_object.instantiate_stack()
global_object.stack.enabled = False


from .__version__ import __version__ as __version__  # noqa: E402
from .fitting.available_minimizers import AvailableMinimizers  # noqa: E402

__all__ = [
    __version__,
    AvailableMinimizers,
    global_object,
]


# alias for global_object, remove later
def __getattr__(name):
    if name == 'borg':
        warnings.warn(
            "The 'borg' has been renamed to 'global_object', this alias will be deprecated in the future", DeprecationWarning
        )  # noqa: E501
        print("The 'borg' has been renamed to 'global_object', this alias will be deprecated in the future")
        return global_object
