# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

from .utilities import jit, guvectorize
from . import algebra, distance, utilities

algebra_ufuncs = type('algebra_gufuncs', (object,), dict())()
utilities.guvectorize_module_functions(algebra, algebra_ufuncs)

from .array import array
