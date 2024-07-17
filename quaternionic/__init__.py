# Copyright (c) 2024, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:  # pragma: no cover
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

from .utilities import jit, guvectorize, float64, boolean
from . import algebra, properties, converters, distance, utilities

algebra_ufuncs = type('AlgebraUfuncs', (object,), dict())()
utilities.guvectorize_module_functions(algebra, algebra_ufuncs)

from .arrays import QuaternionicArray, array

from .alignment import align

one = array(1, 0, 0, 0)
one.flags.writeable = False

x = array(0, 1, 0, 0)
x.flags.writeable = False
i = x

y = array(0, 0, 1, 0)
y.flags.writeable = False
j = y

z = array(0, 0, 0, 1)
z.flags.writeable = False
k = z

from .interpolation import unflip_rotors, slerp, squad
