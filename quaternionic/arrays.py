# Copyright (c) 2024, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import numpy as np

from . import jit
from .properties import QuaternionPropertiesMixin
from .converters import QuaternionConvertersMixin

try:
    _ones_like = np._core.umath._ones_like
except AttributeError:
    _ones_like = np.core.umath._ones_like


def QuaternionicArray(jit=jit, dtype=float):
    """Construct a quaternionic array type.

    This factory returns a `class` encapsulating a quaternionic array type,
    where the jit function and dtype are passed to this factory function, and
    used when creating the class.  The returned class can then be used to
    instantiate actual arrays.  This allows us to, for example, skip jit
    compilation and construct numpy arrays of dtype `object`, so that we can
    use more general python types than the standard numeric types — such as
    sympy expressions.

    """
    class QArray(QuaternionPropertiesMixin(jit), QuaternionConvertersMixin(jit), np.ndarray):
        """Subclass of numpy arrays interpreted as quaternions.

        This class encapsulates quaternion algebra, with numpy's "ufunc"s
        overridden by quaternionic methods.  Standard algebraic operations can be
        performed naturally — as in `q1+q2`, `q1*q2`, etc.  Numpy functions can
        also be used as expected — as in `np.exp(q)`, `np.log(q)`, etc.

        Because this is a subclass of numpy's ndarray object, its constructor takes
        anything the ndarray constructor takes, or just an ndarray to be re-interpreted
        as a quaternion array:

            q1 = quaternionic.array([1, 2, 3, 4])  # explicit array
            q2 = quaternionic.array(np.random.normal(size=(10, 4))  # re-interpret ndarray

        In addition to the basic numpy array features, we also have a number of
        extra properties that are particularly useful for quaternions, including

          * Methods to extract and/or set components
            * w, x, y, z
            * i, j, k (equivalent to x, y, z)
            * scalar, vector (equivalent to w, [x, y, z])
            * real, imag (equivalent to scalar, vector)
          * Methods related to norms
            * abs (square-root of sum of squares of components)
            * norm (sum of squares of components)
            * modulus, magnitude (equal to abs)
            * absolute_square, abs2, mag2 (equal to norm)
            * normalized
            * conjugate, conj
            * inverse
          * Methods related to array infrastructure
            * ndarray
            * flattened
            * iterator

        There are also several converters to and from other representations of
        rotations, including

           * to/from_rotation_matrix
           * to_transformation_matrix
           * to/from_axis_angle
           * to/from_euler_angles
           * to/from_spherical_coordinates

        """

        # https://numpy.org/doc/1.18/user/basics.subclassing.html
        def __new__(cls, input_array, *args, **kwargs):
            from numbers import Real
            if isinstance(input_array, Real) and len(args) >= 3 and all(isinstance(a, Real) for a in args[:3]):
                input_array = [input_array,] + list(args[:3])
            input_array = np.asanyarray(input_array, dtype=dtype)
            if len(input_array.shape) == 0:
                raise ValueError("len(input_array.shape) == 0")
            if input_array.shape[-1] != 4:
                raise ValueError(
                    f"\nInput array has shape {input_array.shape} when viewed as a float array.\n"
                    "Its last dimension should have size 4, representing the components of a quaternion."
                )
            obj = input_array.view(cls)
            return obj

        def __getitem__(self, i):
            # Note that this simply assumes that if the returned array has last
            # dimension of size 4, it is a quaternionic array.  Obviously, this may
            # not always be true, but there's no simple way to decide more
            # correctly.
            r = super().__getitem__(i)
            return type(self)(r)

        def __array_finalize__(self, obj):
            if self.shape[-1] != 4:
                raise ValueError(
                    f"\nArray to finalize has shape {self.shape}; "
                    "last dimension should have size 4 to represent a quaternion.\n"
                    "If you are trying to slice the quaternions, you should append `.ndarray` before slicing.\n"
                    "For example, instead of `q[..., 2:]`, you must use `q.ndarray[..., 2:]` to return a\n"
                    "general (non-quaternion) numpy array.  This is a limitation of numpy.\n\n"
                    "Also note that quaternions have attributes like `q.w`, `q.x`, `q.y`, and `q.z` to return\n"
                    "arrays of individual components, and `q.vector` to return the \"vector\" part."
                )

        def __array_function__(self, func, types, args, kwargs):
            from . import algebra_ufuncs as algebra

            if func == np.angle:
                output = np.zeros(args[0].shape[:-1], dtype=dtype)
                algebra.angle(args[0].ndarray, output)
                if kwargs.get("deg", False):
                    output *= 180/np.pi
            else:
                output = super().__array_function__(func, types, args, kwargs)
                if func in [np.ones_like]:
                    # Want the last dimension to equal [1,0,0,0] not [1,1,1,1]
                    output.vector = 0
            return output

        def __array_ufunc__(self, ufunc, method, *args, **kwargs):
            from . import algebra_ufuncs as algebra

            out = kwargs.pop('out', None)

            # We will not be supporting any more ufunc keywords beyond `out`
            if kwargs:
                raise NotImplementedError(f"Unrecognized arguments to {type(self).__name__}.__array_ufunc__: {kwargs}")

            if method in ["reduce", "accumulate", "reduceat", "outer", "at"]:
                raise NotImplementedError(f"Only __call__ method works for quaternionic arrays; got {method}")

            this_type = lambda o: isinstance(o, type(self))

            if ufunc in [
                np.add, np.subtract, np.multiply, np.divide, np.true_divide,
                np.bitwise_or, np.bitwise_xor, np.right_shift, np.left_shift,
            ]:
                # float64[4](float64[4], float64[4])
                if this_type(args[0]) and this_type(args[1]):
                    a1, a2 = args[:2]
                    b1 = a1.ndarray[..., 0]
                    b2 = a2.ndarray[..., 0]
                    shape = np.broadcast(b1, b2).shape + (4,)
                    result = out or np.zeros(shape, dtype=dtype)
                    if isinstance(result, tuple):
                        result = result[0]
                    result = result.view(np.ndarray)
                    getattr(algebra, ufunc.__name__)(a1.ndarray, a2.ndarray, result)
                    result = type(self)(result)

                # float64[4](float64, float64[4])
                elif not this_type(args[0]) and this_type(args[1]) and ufunc in [np.multiply, np.divide, np.true_divide]:
                    a1, a2 = args[:2]
                    b1 = a1
                    b2 = a2.ndarray[..., 0]
                    shape = np.broadcast(b1, b2).shape + (4,)
                    result = out or np.zeros(shape, dtype=dtype)
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, type(self)):
                        result = result.view(np.ndarray)
                    getattr(algebra, f"{ufunc.__name__}_scalar")(a1, a2.ndarray, result)
                    result = type(self)(result)

                # float64[4](float64[4], float64)
                elif this_type(args[0]) and not this_type(args[1]) and ufunc in [np.multiply, np.divide, np.true_divide]:
                    a1, a2 = args[:2]
                    b1 = a1.ndarray[..., 0]
                    b2 = a2
                    shape = np.broadcast(b1, b2).shape + (4,)
                    result = out or np.zeros(shape, dtype=dtype)
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, type(self)):
                        result = result.view(np.ndarray)
                    getattr(algebra, f"scalar_{ufunc.__name__}")(a1.ndarray, a2, result)
                    result = type(self)(result)
                else:
                    return NotImplemented

            # float64[4](float64[4])
            elif ufunc in [
                np.negative, np.positive, np.conj, np.conjugate, np.invert,
                np.exp, np.log, np.sqrt, np.square, np.reciprocal,
                _ones_like,
            ]:
                if this_type(args[0]):
                    a1 = args[0]
                    result = out or np.zeros(a1.shape, dtype=dtype)
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, type(self)):
                        result = result.view(np.ndarray)
                    getattr(algebra, ufunc.__name__)(a1.ndarray, result)
                    result = type(self)(result)
                else:
                    return NotImplemented

            # float64[4](float64[4], float64)
            elif ufunc in [np.float_power, np.power]:
                if this_type(args[0]) and not this_type(args[1]):
                    a1, a2 = args[:2]
                    b1 = a1.ndarray[..., 0]
                    b2 = a2
                    shape = np.broadcast(b1, b2).shape + (4,)
                    result = out or np.zeros(shape, dtype=dtype)
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, type(self)):
                        result = result.view(np.ndarray)
                    algebra.float_power(a1.ndarray, a2, result)
                    result = type(self)(result)
                else:
                    return NotImplemented

            # float64(float64[4])
            elif ufunc in [np.absolute]:
                if this_type(args[0]):
                    a1 = args[0]
                    result = out or np.zeros(a1.shape[:-1], dtype=dtype)
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, type(self)):
                        result = result.view(np.ndarray)
                    getattr(algebra, ufunc.__name__)(a1.ndarray, result)
                else:
                    return NotImplemented

            # bool(float64[4], float64[4])
            elif ufunc in [np.not_equal, np.equal, np.logical_and, np.logical_or]:
                # Note that these ufuncs are used in numerous unexpected places
                # throughout numpy, so we really need them for basic things to work
                if this_type(args[0]) and this_type(args[1]):
                    a1, a2 = args[:2]
                    b1 = a1.ndarray[..., 0]
                    b2 = a2.ndarray[..., 0]
                    shape = np.broadcast(b1, b2).shape
                    result = out or np.zeros(shape, dtype=bool)
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, type(self)):
                        result = result.view(np.ndarray)
                    getattr(algebra, ufunc.__name__)(a1.ndarray, a2.ndarray, result)
                else:
                    return NotImplemented

            # bool(float64[4])
            elif ufunc in [np.isfinite, np.isinf, np.isnan]:
                if this_type(args[0]):
                    a1 = args[0]
                    result = out or np.zeros(a1.shape[:-1], dtype=bool)
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, type(self)):
                        result = result.view(np.ndarray)
                    getattr(algebra, ufunc.__name__)(a1.ndarray, result)
                else:
                    return NotImplemented

            else:
                return NotImplemented

            return result

        def __repr__(self):
            return 'quaternionic.' + repr(self.ndarray)

        def __str__(self):
            return str(self.ndarray)

    return QArray


array = QuaternionicArray()
