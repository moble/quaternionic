# Copyright (c) 2024, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import abc
import collections
import numpy as np
from . import jit
from .utilities import type_self_return, ndarray_args, ndarray_args_and_return


TwoSpinor = collections.namedtuple("TwoSpinor", ["a", "b"])


def QuaternionPropertiesMixin(jit=jit):
    class mixin(abc.ABC):
        """Basic properties for quaternionic array class.

        This abstract base class comprises the basic interface for quaternionic
        arrays, including the components (w, x, y, z), the parts (scalar,
        vector), the norm and absolute value, the normalized equivalent, and so
        on.  Also included are various

        The other main elements for a quaternionic array class are converters
        (to and from matrices, Euler angles, etc.)  and, of course, the
        algebraic functions (addition, multiplication, etc.).

        """

        @property
        def w(self):
            """The first (scalar) component of the quaternion"""
            return self.ndarray[..., 0]

        @w.setter
        def w(self, wprime):
            self.ndarray[..., 0] = wprime

        @property
        def x(self):
            """The second component of the quaternion (rotation about x)"""
            return self.ndarray[..., 1]

        @x.setter
        def x(self, xprime):
            self.ndarray[..., 1] = xprime

        @property
        def y(self):
            """The third component of the quaternion (rotation about y)"""
            return self.ndarray[..., 2]

        @y.setter
        def y(self, yprime):
            self.ndarray[..., 2] = yprime

        @property
        def z(self):
            """The fourth component of the quaternion (rotation about z)"""
            return self.ndarray[..., 3]

        @z.setter
        def z(self, zprime):
            self.ndarray[..., 3] = zprime

        @property
        def vector(self):
            """The "vector" part of the quaternion (final three components).

            Note that it is entirely standard to describe this part of the
            quaternion as the "vector" part.  It would be more correct to refer
            to it as the "bivector" part, as explained by geometric algebra.

            """
            return self.ndarray[..., 1:]

        @vector.setter
        def vector(self, v):
            self.ndarray[..., 1:] = v

        @property
        def two_spinor(self):
            """Return the two-spinor representation of the quaternion

            The standard mapping from quaternions to two-spinors is

                q ↦ (q.w + 1j * q.z, q.y + 1j * q.x)

            This function returns a namedtuple with those components (as numpy
            arrays), where the elements names are 'a' and 'b', respectively.

            """
            return TwoSpinor(
                self.ndarray[..., 0] + 1j * self.ndarray[..., 3],
                self.ndarray[..., 2] + 1j * self.ndarray[..., 1]
            )

        @property
        @ndarray_args
        @jit
        def norm(self):
            """The (Cayley) norm of the quaternion.

            This quantity is the sum of the squares of the components of the
            quaternion — equal to the square of the absolute value.

            Note that it may be surprising to find that this "norm" does not
            include the usual square root.  This is conventional.  For example,
            the Boost library's implementation of quaternions also uses this
            convention.  Similarly, the implementation of complex numbers in
            C++ defines the `abs` and `norm` functions in this way, while
            python and numpy only define `abs`.

            If you are uncomfortable with this choice of the meaning of `norm`,
            it may make more sense to use one of the aliases of this function,
            which include

              * absolute_square
              * abs2
              * mag2

            """
            s = self.reshape((-1, 4))
            n = np.empty(s.shape[0], dtype=self.dtype)
            for i in range(s.shape[0]):
                n[i] = s[i, 0]**2 + s[i, 1]**2 + s[i, 2]**2 + s[i, 3]**2
            return n.reshape(self.shape[:-1])

        @property
        @ndarray_args
        @jit
        def abs(self):
            """The absolute value of the quaternion.

            This quantity is the square-root of the sum of the squares of the
            components of each quaternion.

            See the note in `norm` for the difference between this function and
            that.  Basically, this is the square-root of that function.

            """
            s = self.reshape((-1, 4))
            n = np.empty(s.shape[0], dtype=self.dtype)
            for i in range(s.shape[0]):
                n[i] = np.sqrt(s[i, 0]**2 + s[i, 1]**2 + s[i, 2]**2 + s[i, 3]**2)
            return n.reshape(self.shape[:-1])

        def conjugate(self):
            """The quaternion conjugate of this quaternion"""
            c = self.copy()
            c.vector *= -1
            return c

        @property
        @ndarray_args_and_return
        @jit
        def inverse(self):
            """The multiplicative inverse of this quaternion"""
            s = self.reshape((-1, 4))
            inv = np.empty(s.shape, dtype=self.dtype)
            for i in range(s.shape[0]):
                n = s[i, 0]**2 + s[i, 1]**2 + s[i, 2]**2 + s[i, 3]**2
                inv[i, 0] = s[i, 0] / n
                inv[i, 1] = -s[i, 1] / n
                inv[i, 2] = -s[i, 2] / n
                inv[i, 3] = -s[i, 3] / n
            return inv.reshape(self.shape)

        # Aliases
        scalar = w
        real = w
        i = x
        j = y
        k = z
        imag = vector
        absolute_square = norm
        abs2 = norm
        mag2 = norm
        modulus = abs
        magnitude = abs
        conj = conjugate

        @property
        def normalized(self):
            """The normalized version of this quaternion"""
            return self / self.abs

        @property
        def ndarray(self):
            """View this array as a numpy ndarray"""
            return self.view(np.ndarray)

        @property
        def flattened(self):
            """A view of this array with all but the last dimension combined into one"""
            return self.reshape((-1, 4))

        @property
        def iterator(self):
            """Iterate over all but the last dimension of this quaternion array"""
            s = self.reshape((-1, 4))
            for i in range(s.shape[0]):
                yield(s[i])

        def nonzero(self):
            """Return the indices of all nonzero elements.

            This is essentially the same function as numpy.nonzero, except that
            the last dimension is treated as a single quaternion; if any
            component of the quaternion is nonzero, the quaternion is
            considered nonzero.

            """
            return np.nonzero(np.atleast_1d(np.any(self.ndarray, axis=-1)))

        def rotate(self, v, axis=-1):
            """Rotate vectors by quaternions in this array.

            For simplicity, this function simply converts the input
            quaternion(s) to a matrix, and rotates the input vector(s) by the
            usual matrix multiplication.  However, it should be noted that if
            each input quaternion is only used to rotate a single vector, it is
            more efficient (in terms of operation counts) to use the formula

              v' = v + 2 * r x (s * v + r x v) / m

            where x represents the cross product, s and r are the scalar and
            vector parts of the quaternion, respectively, and m is the sum of
            the squares of the components of the quaternion.  If you are
            looping over a very large number of quaternions, and just rotating
            a single vector each time, you might want to implement that
            alternative algorithm using numba (or something that doesn't use
            python).


            Parameters
            ----------
            v : float array
                Three-vectors to be rotated.
            axis : int, optional
                Axis of the `v` array to use as the vector dimension.  This axis of
                `v` must have length 3.  The default is the last axis.

            Returns
            -------
            vprime : float array
                The rotated vectors.  This array has shape self.shape+v.shape.

            """
            v = np.asarray(v, dtype=self.dtype)
            if v.ndim < 1 or 3 not in v.shape:
                raise ValueError("Input `v` does not have at least one dimension of length 3")
            if v.shape[axis] != 3:
                raise ValueError("Input `v` axis {0} has length {1}, not 3.".format(axis, v.shape[axis]))
            m = self.to_rotation_matrix
            tensordot_axis = m.ndim-2
            final_axis = tensordot_axis + (axis % v.ndim)
            return np.moveaxis(
                np.tensordot(m, v, axes=(-1, axis)),
                tensordot_axis, final_axis
            )

    return mixin
