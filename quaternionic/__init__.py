# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

__version__ = '0.0.1a'

from .utilities import jit, guvectorize
from . import algebra, distance, utilities
from .array import array


algebra_gufuncs = type('algebra_gufuncs', (object,), dict())()
utilities.guvectorize_module_functions(algebra, algebra_gufuncs)
