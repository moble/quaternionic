import numpy as np

from .properties import QuaternionPropertiesMixin
from .converters import QuaternionConvertersMixin


class Quaternion(QuaternionPropertiesMixin, QuaternionConvertersMixin, np.ndarray):

    # https://numpy.org/doc/1.18/user/basics.subclassing.html
    def __new__(cls, input_array, *args, **kwargs):
        input_array = np.asanyarray(input_array).view(float)
        if input_array.shape[-1] != 4:
            raise ValueError(
                f"\nInput array has shape {input_array.shape} when viewed as a float array.\n"
                "Its last dimension should have size 4, representing the components of a quaternion."
            )
        obj = input_array.view(cls)
        return obj

    def __getitem__(self, i):
        r = super().__getitem__(i)
        if hasattr(r, 'shape') and r.shape[-1] == 4:
            return type(self)(r)
        else:
            return r

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

    def __array_ufunc__(self, ufunc, method, *args, out=None, **kwargs):
        from . import algebra

        # We will not be supporting any more ufunc keywords beyond `out`
        if kwargs:
            raise NotImplementedError(f"Unrecognized arguments to {type(self).__name__}.__array_ufunc__: {kwargs}")

        this_type = lambda o: isinstance(o, type(self))

        # # These are required for basic support, but can be more-or-less passed through because they return bools
        # if ufunc in [np.not_equal, np.logical_or, np.isinf, np.isnan]:
        #     args = [arg.view(np.ndarray) if isinstance(arg, type(self)) else arg for arg in args]
        #     return np.any(self.view(np.ndarray).__array_ufunc__(ufunc, method, *args, **kwargs), axis=-1, out=out)
        # if ufunc in [np.equal, np.logical_and, np.logical_or, np.isfinite]:
        #     args = [arg.view(np.ndarray) if isinstance(arg, type(self)) else arg for arg in args]
        #     return np.all(self.view(np.ndarray).__array_ufunc__(ufunc, method, *args, **kwargs), axis=-1, out=out)

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
                result = out or np.zeros(shape)
                if isinstance(result, tuple):
                    result = result[0]
                if isinstance(result, type(self)):
                    result = result.view(np.ndarray)
                getattr(algebra, ufunc.__name__)(a1.ndarray, a2.ndarray, result)
                result = type(self)(result)

            # float64[4](float64, float64[4])
            elif not this_type(args[0]) and this_type(args[1]) and ufunc in [np.multiply, np.divide, np.true_divide]:
                a1, a2 = args[:2]
                b1 = a1
                b2 = a2.ndarray[..., 0]
                shape = np.broadcast(b1, b2).shape + (4,)
                result = out or np.zeros(shape)
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
                result = out or np.zeros(shape)
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
        ]:
            if this_type(args[0]):
                qout = np.empty_like(args[0]) if out is None else out[0]
                getattr(algebra, ufunc.__name__)(args[0], qout)
                result = type(self)(qout)
            else:
                return NotImplemented

        # float64[4](float64[4], float64)
        elif ufunc in [np.float_power]:
            a1, a2 = args[:2]
            b1 = a1.ndarray[..., 0]
            b2 = a2
            shape = np.broadcast(b1, b2).shape + (4,)
            result = out or np.zeros(shape)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, type(self)):
                result = result.view(np.ndarray)
            algebra.float_power(a1.ndarray, a2, result)
            result = type(self)(result)

        # float64(float64[4])
        elif ufunc in [np.absolute]:
            if this_type(args[0]):
                qout = np.empty(args[0].shape[:-1]) if out is None else out[0]
                algebra.absolute(args[0], qout)
                result = qout
            else:
                return NotImplemented

        # bool(float64[4], float64[4])
        elif ufunc in [np.not_equal, np.equal, np.logical_and, np.logical_or]:
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

        # bool(float64[4])
        elif ufunc in [np.isfinite, np.isinf, np.isnan]:
            if this_type(args[0]):
                qout = np.empty(args[0].shape[:-1], dtype=bool) if out is None else out[0]
                getattr(algebra, ufunc.__name__)(args[0], qout)
                result = qout
            else:
                return NotImplemented

        else:
            return NotImplemented

        if result is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        return result
