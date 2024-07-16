# Copyright (c) 2024, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import abc
import numpy as np
import numba
from . import jit
from .utilities import ndarray_args


def ToEulerPhases(jit=jit):
    @jit
    def _to_euler_phases(R, z):
        """Helper function for `to_euler_phases`"""
        a = R[0]**2 + R[3]**2
        b = R[1]**2 + R[2]**2
        sqrta = np.sqrt(a)
        sqrtb = np.sqrt(b)
        z[1] = ((a - b) + 2j * sqrta * sqrtb) / (a + b)  # exp[iβ]
        if sqrta > 0.0:
            zp = (R[0] + 1j * R[3]) / sqrta  # exp[i(α+γ)/2]
        else:
            zp = 1.0 + 0.0j
        if abs(sqrtb) > 0.0:
            zm = (R[2] - 1j * R[1]) / sqrtb  # exp[i(α-γ)/2]
        else:
            zm = 1.0 +0.0j
        z[0] = zp * zm
        z[2] = zp * zm.conjugate()
    return _to_euler_phases

_to_euler_phases = ToEulerPhases(jit)


def FromEulerPhases(jit=jit):
    @jit
    def _from_euler_phases(R, z):
        """Helper function for `from_euler_phases`"""
        for i in range(R.shape[0]):
            zb = np.sqrt(z[i, 1])  # exp[iβ/2]
            zp = np.sqrt(z[i, 0] * z[i, 2])  # exp[i(α+γ)/2]
            zm = np.sqrt(z[i, 0] * z[i, 2].conjugate())  # exp[i(α-γ)/2]
            if abs(z[i, 0] - zp * zm) > abs(z[i, 0] + zp * zm):
                zp *= -1
            R[i, 0] = zb.real * zp.real
            R[i, 1] = -zb.imag * zm.imag
            R[i, 2] = zb.imag * zm.real
            R[i, 3] = zb.real * zp.imag
    return _from_euler_phases

_from_euler_phases = FromEulerPhases(jit)


def QuaternionConvertersMixin(jit=jit):
    _to_euler_phases = ToEulerPhases(jit)
    _from_euler_phases = FromEulerPhases(jit)

    class mixin(abc.ABC):
        """Converters for quaternionic array class.

        This abstract base class provides converters for quaternionic arrays, going
        to and from other representations, including the axis-angle representation,
        rotation matrices, Euler angles, spherical coordinates.

        """

        @property
        def to_scalar_part(self):
            """The "scalar" part of the quaternion (first component)."""
            return self.scalar

        @classmethod
        def from_scalar_part(cls, scalars):
            """Create a quaternionic array from its scalar part.

            Essentially, this just inserts three 0s after each scalar part, and
            re-interprets the result as a quaternion.

            Parameters
            ----------
            scalars : float array
                Array of scalar parts of quaternions.

            Returns
            -------
            q : array of quaternions
                Quaternions with scalar parts corresponding to input scalars.  Output shape
                is scalars.shape+(4,).

            """
            q = np.zeros(scalars.shape+(4,), dtype=scalars.dtype)
            q[..., 0] = scalars
            return cls(q)

        @property
        def to_vector_part(self):
            """The "vector" part of the quaternion (final three components).

            Note that it is entirely standard to describe this part of the
            quaternion as the "vector" part.  It would be more correct to refer
            to it as the "bivector" part, as explained by geometric algebra.

            """
            return self.vector

        @classmethod
        def from_vector_part(cls, vec):
            """Create a quaternionic array from its vector part.

            Essentially, this just inserts a 0 in front of each vector part, and
            re-interprets the result as a quaternion.

            Parameters
            ----------
            vec : (..., 3) float array

                Array of vector parts of quaternions. 

            Returns
            -------
            q : array of quaternions
                Quaternions with vector parts corresponding to input vectors.  Output shape
                is vec.shape[:-1]+(4,).

            """
            return cls(np.insert(vec, 0, 0.0, axis=-1))

        @property
        @ndarray_args
        @jit
        def to_rotation_matrix(self):
            """Convert quaternions to 3x3 rotation matrices.

            Assuming the quaternion R rotates a vector v according to

                v' = R * v * R⁻¹,

            we can also express this rotation in terms of a 3x3 matrix ℛ such that

                v' = ℛ * v.

            This function returns that matrix.

            Returns
            -------
            rot : float array
                Output shape is self.shape[:-1]+(3,3).  This matrix should multiply
                (from the left) a column vector to produce the rotated column
                vector.

            """
            s = self.reshape((-1, 4))
            m = np.empty(s.shape[:1] + (3, 3), dtype=self.dtype)
            for i in range(s.shape[0]):
                n = s[i, 0]**2 + s[i, 1]**2 + s[i, 2]**2 + s[i, 3]**2
                m[i, 0, 0] = 1.0 - 2*(s[i, 2]**2 + s[i, 3]**2) / n
                m[i, 0, 1] = 2*(s[i, 1]*s[i, 2] - s[i, 3]*s[i, 0]) / n
                m[i, 0, 2] = 2*(s[i, 1]*s[i, 3] + s[i, 2]*s[i, 0]) / n
                m[i, 1, 0] = 2*(s[i, 1]*s[i, 2] + s[i, 3]*s[i, 0]) / n
                m[i, 1, 1] = 1.0 - 2*(s[i, 1]**2 + s[i, 3]**2) / n
                m[i, 1, 2] = 2*(s[i, 2]*s[i, 3] - s[i, 1]*s[i, 0]) / n
                m[i, 2, 0] = 2*(s[i, 1]*s[i, 3] - s[i, 2]*s[i, 0]) / n
                m[i, 2, 1] = 2*(s[i, 2]*s[i, 3] + s[i, 1]*s[i, 0]) / n
                m[i, 2, 2] = 1.0 - 2*(s[i, 1]**2 + s[i, 2]**2) / n
            return m.reshape(self.shape[:-1] + (3, 3))

        @classmethod
        def from_rotation_matrix(cls, rot, nonorthogonal=True):
            """Convert input 3x3 rotation matrix to unit quaternion.

            Assuming an orthogonal 3x3 matrix ℛ rotates a vector v such that

                v' = ℛ * v,

            we can also express this rotation in terms of a unit quaternion R such that

                v' = R * v * R⁻¹,

            where v and v' are now considered pure-vector quaternions.  This function
            returns that quaternion.  If `rot` is not orthogonal, the "closest" orthogonal
            matrix is used; see Notes below.

            Parameters
            ----------
            rot : (...Nx3x3) float array
                Each 3x3 matrix represents a rotation by multiplying (from the left)
                a column vector to produce a rotated column vector.  Note that this
                input may actually have ndims>3; it is just assumed that the last
                two dimensions have size 3, representing the matrix.
            nonorthogonal : bool, optional
                If scipy.linalg is available, use the more robust algorithm of
                Bar-Itzhack.  Default value is True.

            Returns
            -------
            q : array of quaternions
                Unit quaternions resulting in rotations corresponding to input
                rotations.  Output shape is rot.shape[:-2].

            Raises
            ------
            LinAlgError
                If any of the eigenvalue solutions does not converge

            Notes
            -----
            By default this function uses Bar-Itzhack's algorithm to allow for
            non-orthogonal matrices.  [J. Guidance, Vol. 23, No. 6, p. 1085
            <http://dx.doi.org/10.2514/2.4654>]  This will almost certainly be quite a bit
            slower than simpler versions, though it will be more robust to numerical errors
            in the rotation matrix.  Also note that the Bar-Itzhack paper uses some pretty
            weird conventions.  The last component of the quaternion appears to represent
            the scalar, and the quaternion itself is conjugated relative to the convention
            used throughout this module.

            If the optional `nonorthogonal` parameter is set to `False`, this function
            falls back to the possibly faster, but less robust, algorithm of Markley
            [J. Guidance, Vol. 31, No. 2, p. 440 <http://dx.doi.org/10.2514/1.31730>].

            """
            from scipy import linalg

            rot = np.array(rot, copy=False)
            shape = rot.shape[:-2]

            if nonorthogonal:
                from operator import mul
                from functools import reduce

                K3 = np.empty(shape+(4, 4), dtype=rot.dtype)
                K3[..., 0, 0] = (rot[..., 0, 0] - rot[..., 1, 1] - rot[..., 2, 2])/3
                K3[..., 0, 1] = (rot[..., 1, 0] + rot[..., 0, 1])/3
                K3[..., 0, 2] = (rot[..., 2, 0] + rot[..., 0, 2])/3
                K3[..., 0, 3] = (rot[..., 1, 2] - rot[..., 2, 1])/3
                K3[..., 1, 0] = K3[..., 0, 1]
                K3[..., 1, 1] = (rot[..., 1, 1] - rot[..., 0, 0] - rot[..., 2, 2])/3
                K3[..., 1, 2] = (rot[..., 2, 1] + rot[..., 1, 2])/3
                K3[..., 1, 3] = (rot[..., 2, 0] - rot[..., 0, 2])/3
                K3[..., 2, 0] = K3[..., 0, 2]
                K3[..., 2, 1] = K3[..., 1, 2]
                K3[..., 2, 2] = (rot[..., 2, 2] - rot[..., 0, 0] - rot[..., 1, 1])/3
                K3[..., 2, 3] = (rot[..., 0, 1] - rot[..., 1, 0])/3
                K3[..., 3, 0] = K3[..., 0, 3]
                K3[..., 3, 1] = K3[..., 1, 3]
                K3[..., 3, 2] = K3[..., 2, 3]
                K3[..., 3, 3] = (rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2])/3

                if not shape:
                    q = np.empty((4,), dtype=rot.dtype)
                    eigvals, eigvecs = linalg.eigh(K3.T, subset_by_index=(3, 3))
                    q[0] = eigvecs[-1, 0]
                    q[1:] = -eigvecs[:-1].flatten()
                    return cls(q)
                else:
                    q = np.empty(shape+(4,), dtype=rot.dtype)
                    for flat_index in range(reduce(mul, shape)):
                        multi_index = np.unravel_index(flat_index, shape)
                        eigvals, eigvecs = linalg.eigh(K3[multi_index], subset_by_index=(3, 3))
                        q[multi_index+(0,)] = eigvecs[-1, 0]
                        q[multi_index+(slice(1,None),)] = -eigvecs[:-1].flatten()
                    return cls(q)

            else:  # Not `nonorthogonal`
                diagonals = np.empty(shape+(4,), dtype=rot.dtype)
                diagonals[..., 0] = rot[..., 0, 0]
                diagonals[..., 1] = rot[..., 1, 1]
                diagonals[..., 2] = rot[..., 2, 2]
                diagonals[..., 3] = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]

                indices = np.argmax(diagonals, axis=-1)

                q = diagonals  # reuse storage space
                indices_i = (indices == 0)
                if np.any(indices_i):
                    if indices_i.shape == ():
                        indices_i = Ellipsis
                    rot_i = rot[indices_i, :, :]
                    q[indices_i, 0] = rot_i[..., 2, 1] - rot_i[..., 1, 2]
                    q[indices_i, 1] = 1 + rot_i[..., 0, 0] - rot_i[..., 1, 1] - rot_i[..., 2, 2]
                    q[indices_i, 2] = rot_i[..., 0, 1] + rot_i[..., 1, 0]
                    q[indices_i, 3] = rot_i[..., 0, 2] + rot_i[..., 2, 0]
                indices_i = (indices == 1)
                if np.any(indices_i):
                    if indices_i.shape == ():
                        indices_i = Ellipsis
                    rot_i = rot[indices_i, :, :]
                    q[indices_i, 0] = rot_i[..., 0, 2] - rot_i[..., 2, 0]
                    q[indices_i, 1] = rot_i[..., 1, 0] + rot_i[..., 0, 1]
                    q[indices_i, 2] = 1 - rot_i[..., 0, 0] + rot_i[..., 1, 1] - rot_i[..., 2, 2]
                    q[indices_i, 3] = rot_i[..., 1, 2] + rot_i[..., 2, 1]
                indices_i = (indices == 2)
                if np.any(indices_i):
                    if indices_i.shape == ():
                        indices_i = Ellipsis
                    rot_i = rot[indices_i, :, :]
                    q[indices_i, 0] = rot_i[..., 1, 0] - rot_i[..., 0, 1]
                    q[indices_i, 1] = rot_i[..., 2, 0] + rot_i[..., 0, 2]
                    q[indices_i, 2] = rot_i[..., 2, 1] + rot_i[..., 1, 2]
                    q[indices_i, 3] = 1 - rot_i[..., 0, 0] - rot_i[..., 1, 1] + rot_i[..., 2, 2]
                indices_i = (indices == 3)
                if np.any(indices_i):
                    if indices_i.shape == ():
                        indices_i = Ellipsis
                    rot_i = rot[indices_i, :, :]
                    q[indices_i, 0] = 1 + rot_i[..., 0, 0] + rot_i[..., 1, 1] + rot_i[..., 2, 2]
                    q[indices_i, 1] = rot_i[..., 2, 1] - rot_i[..., 1, 2]
                    q[indices_i, 2] = rot_i[..., 0, 2] - rot_i[..., 2, 0]
                    q[indices_i, 3] = rot_i[..., 1, 0] - rot_i[..., 0, 1]

                q /= np.linalg.norm(q, axis=-1)[..., np.newaxis]

                return cls(q)

        @property
        @ndarray_args
        @jit
        def to_transformation_matrix(self):
            """Convert quaternions to 4x4 transformation matrices.

            Assuming the quaternion Q transforms another quaternion P according to

                P' = Q * P * Q̄,

            we can also express this rotation in terms of a 4x4 matrix ℳ such that

                P' = ℳ * P,

            where P is viewed as a 4-vector in the last line.  This function
            returns that matrix.


            Returns
            -------
            m : float array
                Output shape is self.shape[:-1]+(4,4).  This matrix should multiply
                (from the left) a column vector to produce the transformed column
                vector.

            See also
            --------
            to_rotation_matrix : assumes Q is a unit quaternion

            """
            s = self.reshape((-1, 4))
            m = np.zeros(s.shape[:1] + (4, 4), dtype=self.dtype)
            for i in range(s.shape[0]):
                m[i, 0, 0] = s[i, 0]**2 + s[i, 1]**2 + s[i, 2]**2 + s[i, 3]**2
                m[i, 1, 1] = s[i, 0]**2 + s[i, 1]**2 - s[i, 2]**2 - s[i, 3]**2
                m[i, 1, 2] = 2 * (s[i, 1]*s[i, 2] - s[i, 3]*s[i, 0])
                m[i, 1, 3] = 2 * (s[i, 1]*s[i, 3] + s[i, 2]*s[i, 0])
                m[i, 2, 1] = 2 * (s[i, 1]*s[i, 2] + s[i, 3]*s[i, 0])
                m[i, 2, 2] = s[i, 0]**2 - s[i, 1]**2 + s[i, 2]**2 - s[i, 3]**2
                m[i, 2, 3] = 2 * (s[i, 2]*s[i, 3] - s[i, 1]*s[i, 0])
                m[i, 3, 1] = 2 * (s[i, 1]*s[i, 3] - s[i, 2]*s[i, 0])
                m[i, 3, 2] = 2 * (s[i, 2]*s[i, 3] + s[i, 1]*s[i, 0])
                m[i, 3, 3] = s[i, 0]**2 - s[i, 1]**2 - s[i, 2]**2 + s[i, 3]**2
            return m.reshape(self.shape[:-1] + (4, 4))

        @property
        def to_axis_angle(self):
            """Convert input quaternion to the axis-angle representation.

            Note that if any of the input quaternions has norm zero, no error is
            raised, but NaNs will appear in the output.

            Returns
            -------
            rot : float array
                Output shape is q.shape[:-1]+(3,).  Each vector represents the axis of
                the rotation, with norm equal to the angle of the rotation in radians.

            """
            return 2 * np.log(self.normalized).vector

        to_rotation_vector = to_axis_angle

        @classmethod
        def from_axis_angle(cls, vec):
            """Convert 3-vector in axis-angle representation to unit quaternion.

            Parameters
            ----------
            vec : (...N, 3) float array
                Each vector represents the axis of the rotation, with norm
                proportional to the angle of the rotation in radians.

            Returns
            -------
            q : array of quaternions
                Unit quaternions resulting in rotations corresponding to input
                rotations.  Output shape is rot.shape[:-1].

            """
            vec = np.asarray(vec)
            dtype = np.result_type(0.5, vec)
            quats = np.zeros(vec.shape[:-1] + (4,), dtype=dtype)
            quats[..., 1:] = 0.5 * vec[...]
            return np.exp(cls(quats))

        from_rotation_vector = from_axis_angle

        @property
        @ndarray_args
        @jit
        def to_euler_angles(self):
            """Open Pandora's Box.

            If somebody is trying to make you use Euler angles, tell them no, and
            walk away, and go and tell your mum.

            You don't want to use Euler angles.  They are awful.  Stay away.  It's
            one thing to convert from Euler angles to quaternions; at least you're
            moving in the right direction.  But to go the other way?!  It's just not
            right.

            Assumes the Euler angles correspond to the quaternion R via

                R = exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)

            The angles are naturally in radians.

            NOTE: Before opening an issue reporting something "wrong" with this
            function, be sure to read all of the following page, *especially* the
            very last section about opening issues or pull requests.
            <https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible>

            Returns
            -------
            alpha_beta_gamma : float array
                Output shape is q.shape+(3,).  These represent the angles (alpha,
                beta, gamma) in radians, where the normalized input quaternion
                represents `exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)`.

            Raises
            ------
            AllHell
                ...if you try to actually use Euler angles, when you could have
                been using quaternions like a sensible person.

            """
            s = self.reshape((-1, 4))
            alpha_beta_gamma = np.empty((s.shape[0], 3), dtype=self.dtype)
            for i in range(s.shape[0]):
                n = s[i, 0]**2 + s[i, 1]**2 + s[i, 2]**2 + s[i, 3]**2
                alpha_beta_gamma[i, 0] = np.arctan2(s[i, 3], s[i, 0]) + np.arctan2(-s[i, 1], s[i, 2])
                alpha_beta_gamma[i, 1] = 2*np.arccos(np.sqrt((s[i, 0]**2 + s[i, 3]**2) / n))
                alpha_beta_gamma[i, 2] = np.arctan2(s[i, 3], s[i, 0]) - np.arctan2(-s[i, 1], s[i, 2])
            return alpha_beta_gamma.reshape(self.shape[:-1] + (3,))

        @classmethod
        def from_euler_angles(cls, alpha_beta_gamma, beta=None, gamma=None):
            """Improve your life drastically.

            Assumes the Euler angles correspond to the quaternion R via

                R = exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)

            The angles naturally must be in radians for this to make any sense.

            NOTE: Before opening an issue reporting something "wrong" with this
            function, be sure to read all of the following page, *especially* the
            very last section about opening issues or pull requests.
            <https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible>

            Parameters
            ----------
            alpha_beta_gamma : float or array of floats
                This argument may either contain an array with last dimension of
                size 3, where those three elements describe the (alpha, beta, gamma)
                radian values for each rotation; or it may contain just the alpha
                values, in which case the next two arguments must also be given.
            beta : None, float, or array of floats
                If this array is given, it must be able to broadcast against the
                first and third arguments.
            gamma : None, float, or array of floats
                If this array is given, it must be able to broadcast against the
                first and second arguments.

            Returns
            -------
            R : quaternionic.array
                The shape of this array will be the same as the input, except that
                the last dimension will be removed.

            """
            # Figure out the input angles from either type of input
            if gamma is None:
                alpha_beta_gamma = np.asarray(alpha_beta_gamma)
                alpha = alpha_beta_gamma[..., 0]
                beta  = alpha_beta_gamma[..., 1]
                gamma = alpha_beta_gamma[..., 2]
            else:
                alpha = np.asarray(alpha_beta_gamma)
                beta  = np.asarray(beta)
                gamma = np.asarray(gamma)

            # Pre-compute trig
            cosβover2 = np.cos(beta/2)
            sinβover2 = np.sin(beta/2)

            # Set up the output array
            R = np.empty(np.broadcast(alpha, beta, gamma).shape + (4,), dtype=cosβover2.dtype)

            # Compute the actual values of the quaternion components
            R[..., 0] =  cosβover2*np.cos((alpha+gamma)/2)  # scalar quaternion components
            R[..., 1] = -sinβover2*np.sin((alpha-gamma)/2)  # x quaternion components
            R[..., 2] =  sinβover2*np.cos((alpha-gamma)/2)  # y quaternion components
            R[..., 3] =  cosβover2*np.sin((alpha+gamma)/2)  # z quaternion components

            return cls(R)

        @property
        @ndarray_args
        @jit
        def to_euler_phases(self):
            """Convert input quaternion to complex phases of Euler angles

            Returns
            -------
            z : complex array
                For each quaternion in the input array, this array contains the complex
                phases (zₐ, zᵦ, zᵧ) in that order.  The shape of this output array is
                self.shape[:-1]+(3,).

            See Also
            --------
            from_euler_phases : Create quaternion from Euler phases
            to_euler_angles : Convert quaternion to Euler angles
            from_euler_angles : Create quaternion from Euler angles

            Notes
            -----
            We define the Euler phases from the Euler angles (α, β, γ) as

                zₐ ≔ exp(i*α)
                zᵦ ≔ exp(i*β)
                zᵧ ≔ exp(i*γ)

            These are more useful geometric quantites than the angles themselves — being
            involved in computing spherical harmonics and Wigner's 𝔇 matrices — and can be
            computed from the components of the corresponding quaternion algebraically
            (without the use of transcendental functions).

            """
            R = self.reshape(-1, 4)
            z = np.empty(R.shape[:-1] + (3,), dtype=np.complex128)
            for i in range(z.shape[0]):
                _to_euler_phases(R[i], z[i])
            return z.reshape(self.shape[:-1] + (3,))

        @classmethod
        def from_euler_phases(cls, z):
            """Return the quaternion corresponding to these Euler phases.

            Parameters
            ----------
            z : complex array_like
                This argument must be able to be interpreted as a complex array with last
                dimension of size 3, which represent the complex phases (zₐ, zᵦ, zᵧ) in
                that order.

            Returns
            -------
            R : quaternionic.array
                The shape of this array will be the same as the input, except that the last
                dimension will be removed and replaced with the quaternionic components.

            See Also
            --------
            to_euler_phases : Convert quaternion to Euler phases
            to_euler_angles : Convert quaternion to Euler angles
            from_euler_angles : Create quaternion from Euler angles

            Notes
            -----
            We define the Euler phases from the Euler angles (α, β, γ) as

                zₐ ≔ exp(i*α)
                zᵦ ≔ exp(i*β)
                zᵧ ≔ exp(i*γ)

            These are more useful geometric quantites than the angles themselves — being
            involved in computing spherical harmonics and Wigner's 𝔇 matrices — and can be
            used to compute the components of the corresponding quaternion algebraically
            (without the use of transcendental functions).

            """
            z = np.asarray(z, dtype=complex)
            shape = z.shape
            z = z.reshape(-1, 3)
            R = np.empty(z.shape[:-1]+(4,), dtype=float)
            _from_euler_phases(R, z)
            return cls(R.reshape(shape[:-1] + (4,)))

        @property
        def to_spherical_coordinates(self):
            """Return the spherical coordinates corresponding to this quaternion.

            Obviously, spherical coordinates do not contain as much information as a
            quaternion, so this function does lose some information.  However, the
            returned spherical coordinates will represent the point(s) on the sphere
            to which the input quaternion(s) rotate the z axis.

            Returns
            -------
            vartheta_varphi : float array
                Output shape is q.shape+(2,).  These represent the angles (vartheta,
                varphi) in radians, where the normalized input quaternion represents
                `exp(varphi*z/2) * exp(vartheta*y/2)`, up to an arbitrary inital
                rotation about `z`.

            """
            return self.to_euler_angles[..., 1::-1]

        @classmethod
        def from_spherical_coordinates(cls, theta_phi, phi=None):
            """Return the quaternion corresponding to these spherical coordinates.

            Assumes the spherical coordinates correspond to the quaternion R via

                R = exp(phi*z/2) * exp(theta*y/2)

            The angles naturally must be in radians for this to make any sense.

            Note that this quaternion rotates `z` onto the point with the given
            spherical coordinates, but also rotates `x` and `y` onto the usual basis
            vectors (theta and phi, respectively) at that point.

            Parameters
            ----------
            theta_phi : float or array of floats
                This argument may either contain an array with last dimension of
                size 2, where those two elements describe the (theta, phi) values in
                radians for each point; or it may contain just the theta values in
                radians, in which case the next argument must also be given.
            phi : None, float, or array of floats
                If this array is given, it must be able to broadcast against the
                first argument.

            Returns
            -------
            R : quaternion array
                If the second argument is not given to this function, the shape
                will be the same as the input shape except for the last dimension,
                which will be removed.  If the second argument is given, this
                output array will have the shape resulting from broadcasting the
                two input arrays against each other.

            """
            # Figure out the input angles from either type of input
            if phi is None:
                theta_phi = np.asarray(theta_phi)
                theta = theta_phi[..., 0]
                phi  = theta_phi[..., 1]
            else:
                theta = np.asarray(theta_phi)
                phi = np.asarray(phi)

            # Pre-compute trig
            cp = np.cos(phi/2)
            ct = np.cos(theta/2)
            sp = np.sin(phi/2)
            st = np.sin(theta/2)

            # Set up the output array
            R = np.empty(np.broadcast(theta, phi).shape + (4,), dtype=cp.dtype)

            # Compute the actual values of the quaternion components
            R[..., 0] =  cp*ct  # scalar quaternion components
            R[..., 1] = -sp*st  # x quaternion components
            R[..., 2] =  cp*st  # y quaternion components
            R[..., 3] =  sp*ct  # z quaternion components

            return cls(R)

        def to_angular_velocity(self, t, t_new=None, axis=0):
            """Compute the angular velocity of quaternion timeseries with respect to `t`

            Note that this is the angular velocity of a rotating frame given by the
            quaternionic array, assuming that the quaternions take inertial vectors in the
            current frame to vectors in the rotating frame.

            Parameters
            ----------
            t : array-like of float
                This array represents the times at which the quaternions are measured, and
                with respect to which the derivative will be taken.  Note that these times
                must be real, finite and in strictly increasing order.
            t_new : array-like of float, optional
                If present, the output is interpolated to this set of times.  Defaults to
                None, meaning that the original set of times will be used.
            axis : int, optional
                Axis along which this array is assumed to be varying with `t`. Meaning that
                for t[i] the corresponding quaternions are `np.take(self, i, axis=axis)`.
                Defaults to 0.

            Notes
            -----
            For both unit and non-unit quaternions `Q`, we define the angular velocity as

                ω = 2 * dQ/dt * Q⁻¹

            This agress with the standard definition when `Q` is a unit quaternion and we
            rotate a vector `v` according to

                v' = Q * v * Q̄ = Q * v * Q⁻¹,

            in which case ω is a "pure vector" quaternion, and we have the usual

                dv'/dt = ω × v'.

            It also generalizes this to the case where `Q` is not a unit quaternion, which
            means that it also rescales the vector by the amount Q*Q̄.  In this case, ω also
            has a scalar component encoding twice the logarithmic time-derivative of this
            rescaling, and we have

                dv'/dt = ω * v' + v' * ω̄.

            """
            from scipy.interpolate import CubicSpline
            spline = CubicSpline(t, self, axis=axis)
            if t_new is None:
                Q = self  # shortcut
            else:
                Q = type(self)(spline(t_new))
            t_new = t if t_new is None else t_new
            Q̇ = type(self)(spline.derivative()(t_new))
            return (2 * Q̇ / Q).vector

        @classmethod
        def from_angular_velocity(cls, omega, t, R0=None, tolerance=1e-12):
            """Create a quaternionic array corresponding to angular velocity

            Assuming that omega represents the angular velocity of a frame moving with
            respect to the current frame, this function returns a quaternionic array
            representing that rotating frame.

            Parameters
            ----------
            omega : {callable, (N, 3) array_like}
                If a callable, this must take a quaternion and a float representing the
                current orientation and time, respectively.  If array-like, this is
                assumed to represent the angular velocity at a series of times, given
                by `t`.
            t : (N,) array_like
                Times at which the output frame will be evaluated.
            R0 : quaternionic, optional
                Initial orientation of the frame.  If None, the default, this is set to
                `quaternionic.one`.
            tolerance : float, optional
                Absolute tolerance used in integration.  Defaults to 1e-12.

            See Also
            --------
            to_angular_velocity

            Notes
            -----
            We define the angular velocity as

                ω = 2 * dQ/dt * Q⁻¹

            This only defines the rotating frame up to a constant overall rotation.  In
            particular, if `Q` satisfies the above equation, then so does `Q * P` for any
            constant quaternion `P`.

            """
            import warnings
            from scipy.integrate import solve_ivp
            from scipy.interpolate import CubicSpline
            from . import one, array
            eps = 1e-14

            if R0 is None:
                R0 = one.ndarray
            else:
                R0 = array(R0).ndarray

            R = cls(np.empty((t.size, 4)))

            if callable(omega):
                ω = omega
            else:
                omega = np.asarray(omega)
                if not np.issubdtype(omega.dtype, np.floating):
                    raise ValueError(f"Input omega must have float dtype; it has dtype {omega.dtype}.")
                if omega.shape != (t.shape[0], 3):
                    raise ValueError(f"Input omega must have shape {(t.shape[0], 3)}; it has shape {omega.shape}.")
                ωspline = CubicSpline(t, omega)
                ω = lambda _, ti: ωspline(ti)

            def RHS(t, y):
                R = array(y)
                Ω = array(0.0, *ω(R, t))
                return (0.5 * Ω * R).ndarray

            solution = solve_ivp(
                RHS, [t[0], t[-1]], R0, method="DOP853",
                t_eval=t, dense_output=True, atol=tolerance, rtol=100*np.finfo(float).eps
            )
            # print("Number of function evaluations:", solution.nfev)
            return cls(solution.y.T)

        def to_minimal_rotation(self, t, t_new=None, axis=0, iterations=2):
            """Adjust frame so that there is no rotation about z' axis

            Parameters
            ----------
            t : array-like of float
                This array represents the times at which the quaternions are measured, and
                with respect to which the derivative will be taken.  Note that these times
                must be real, finite and in strictly increasing order.
            t_new : array-like of float, optional
                If present, the output is interpolated to this set of times.  Defaults to
                None, meaning that the original set of times will be used.
            axis : int, optional
                Axis along which this array is assumed to be varying with `t`. Meaning that
                for t[i] the corresponding quaternions are `np.take(self, i, axis=axis)`.
                Defaults to 0.
            iterations : int, optional
                Repeat the minimization to refine the result.  Defaults to 2.

            Notes
            -----
            The output of this function is a frame that rotates the z axis onto the same z'
            axis as the input frame, but with minimal rotation about that axis.  This is
            done by pre-composing the input rotation with a rotation about the z axis
            through an angle γ, where

                dγ/dt = 2*(dR/dt * z * R̄).w

            This ensures that the angular velocity has no component along the z' axis.

            Note that this condition becomes easier to impose the closer the input rotation
            is to a minimally rotating frame, which means that repeated application of this
            function improves its accuracy.  By default, this function is iterated twice,
            though a few more iterations may be called for.

            """
            from scipy.interpolate import CubicSpline
            if iterations == 0:
                return self
            z = type(self)([0.0, 0.0, 0.0, 1.0])
            t_new = t if t_new is None else t_new
            spline = CubicSpline(t, self.ndarray, axis=axis)
            R = type(self)(spline(t_new))
            Rdot = type(self)(spline.derivative()(t_new))
            γ̇over2 = (Rdot * z * np.conjugate(R)).w
            γover2 = CubicSpline(t_new, γ̇over2).antiderivative()(t_new)
            Rγ = np.exp(z * γover2)
            return (R * Rγ).to_minimal_rotation(t_new, t_new=None, axis=axis, iterations=iterations-1)

        @classmethod
        def random(cls, shape=(4,), normalize=False):
            """Construct random quaternions

            Parameters
            ----------
            shape : tuple, optional
                Shape of the array.  If this does not end with `4`, it will be appended.
                Default is `(4,)`.
            normalize : bool, optional
                If True, normalize the result, so that the returned array can be
                interpreted as rotors.  Defaults to False.

            Returns
            -------
            q : array of quaternions

            Notes
            -----
            This function constructs quaternions in which each component has a random value
            drawn from a normal (Gaussian) distribution centered at 0 with scale 1.  This
            has the nice property that the resulting distribution of quaternions is
            isotropic — it is spherically symmetric.  If the result is normalized, these
            are truly random rotors.

            """
            if isinstance(shape, int):
                shape = (shape,)
            if len(shape) == 0:
                shape = (4,)
            if shape[-1] != 4:
                shape = shape + (4,)
            q = np.random.normal(size=shape)  # Note the weird naming of this argument to `normal`
            if normalize:
                q /= np.linalg.norm(q, axis=-1)[..., np.newaxis]
            return cls(q)

    return mixin
