# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

"""Quaternionic arrays.

This module subclasses numpy's array type, interpreting the array as an array
of quaternions, and accelerating the algebra using numba.  This enables natural
manipulations, like multiplying quaternions as `a*b`, while also working with
standard numpy functions, as in `np.log(q)`.  There is also basic initial
support for symbolic manipulation of quaternions by creating quaternionic
arrays with sympy symbols as elements, though this is a work in progress.

"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:  # pragma: no cover
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

from .utilities import jit, guvectorize, float64, boolean
from . import algebra, properties, converters, distance, utilities

algebra_ufuncs = type('AlgebraUfuncs', (object,), dict())()
utilities.guvectorize_module_functions(algebra, algebra_ufuncs)

from .arrays import quaternionic_array_factory, array

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
