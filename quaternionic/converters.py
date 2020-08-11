import abc
import numpy as np
from . import jit


class QuaternionConvertersMixin(abc.ABC):
    """Converters for quaternionic array class

    This abstract base class provides converters for quaternionic arrays, going
    to and from other representations, including the axis-angle representation,
    rotation matrices, Euler angles, spherical coordinates.

    """

    @property
    @jit
    def as_rotation_matrix(self):
        """Convert quaternions to 3x3 rotation matrices

        Assuming the quaternion R rotates a vector v according to

            v' = R * v * R⁻¹,

        we can also express this rotation in terms of a 3x3 matrix ℛ such that

            v' = ℛ * v.

        This function returns that matrix.


        Returns
        -------
        rot: float array
            Output shape is self.shape[:-1]+(3,3).  This matrix should multiply
            (from the left) a column vector to produce the rotated column
            vector.

        Raises
        ------
        ZeroDivisionError
            If any of the input quaternions have norm 0.0.

        """
        n = self.norm
        if not np.all(n):
            raise ZeroDivisionError("Quaternion input to `as_rotation_matrix` has at least one element with zero norm")
        s = self.flattened
        m = np.empty(s.shape[0] + (3, 3))
        for i in range(s.shape[0]):
            m[i, 0, 0] = 1.0 - 2*(s[i, 2]**2 + s[i, 3]**2)/n
            m[i, 0, 1] = 2*(s[i, 1]*s[i, 2] - s[i, 3]*s[i, 0])/n
            m[i, 0, 2] = 2*(s[i, 1]*s[i, 3] + s[i, 2]*s[i, 0])/n
            m[i, 1, 0] = 2*(s[i, 1]*s[i, 2] + s[i, 3]*s[i, 0])/n
            m[i, 1, 1] = 1.0 - 2*(s[i, 1]**2 + s[i, 3]**2)/n
            m[i, 1, 2] = 2*(s[i, 2]*s[i, 3] - s[i, 1]*s[i, 0])/n
            m[i, 2, 0] = 2*(s[i, 1]*s[i, 3] - s[i, 2]*s[i, 0])/n
            m[i, 2, 1] = 2*(s[i, 2]*s[i, 3] + s[i, 1]*s[i, 0])/n
            m[i, 2, 2] = 1.0 - 2*(s[i, 1]**2 + s[i, 2]**2)/n
        return m.reshape(self.shape[:-1] + (3, 3))

    @classmethod
    def from_rotation_matrix(cls, rot, nonorthogonal=True):
        """Convert input 3x3 rotation matrix to unit quaternion

        By default, if scipy.linalg is available, this function uses
        Bar-Itzhack's algorithm to allow for non-orthogonal matrices.
        [J. Guidance, Vol. 23, No. 6, p. 1085 <http://dx.doi.org/10.2514/2.4654>]
        This will almost certainly be quite a bit slower than simpler versions,
        though it will be more robust to numerical errors in the rotation matrix.
        Also note that Bar-Itzhack uses some pretty weird conventions.  The last
        component of the quaternion appears to represent the scalar, and the
        quaternion itself is conjugated relative to the convention used
        throughout this module.

        If scipy.linalg is not available or if the optional
        `nonorthogonal` parameter is set to `False`, this function falls
        back to the possibly faster, but less robust, algorithm of Markley
        [J. Guidance, Vol. 31, No. 2, p. 440
        <http://dx.doi.org/10.2514/1.31730>].

        Parameters
        ----------
        rot: (...Nx3x3) float array
            Each 3x3 matrix represents a rotation by multiplying (from the left)
            a column vector to produce a rotated column vector.  Note that this
            input may actually have ndims>3; it is just assumed that the last
            two dimensions have size 3, representing the matrix.
        nonorthogonal: bool, optional
            If scipy.linalg is available, use the more robust algorithm of
            Bar-Itzhack.  Default value is True.

        Returns
        -------
        q: array of quaternions
            Unit quaternions resulting in rotations corresponding to input
            rotations.  Output shape is rot.shape[:-2].

        Raises
        ------
        LinAlgError
            If any of the eigenvalue solutions does not converge

        """
        try:
            from scipy import linalg
        except ImportError:
            linalg = False

        rot = np.array(rot, copy=False)
        shape = rot.shape[:-2]

        if linalg and nonorthogonal:
            from operator import mul
            from functools import reduce

            K3 = np.empty(shape+(4, 4))
            K3[..., 0, 0] = (rot[..., 0, 0] - rot[..., 1, 1] - rot[..., 2, 2])/3.0
            K3[..., 0, 1] = (rot[..., 1, 0] + rot[..., 0, 1])/3.0
            K3[..., 0, 2] = (rot[..., 2, 0] + rot[..., 0, 2])/3.0
            K3[..., 0, 3] = (rot[..., 1, 2] - rot[..., 2, 1])/3.0
            K3[..., 1, 0] = K3[..., 0, 1]
            K3[..., 1, 1] = (rot[..., 1, 1] - rot[..., 0, 0] - rot[..., 2, 2])/3.0
            K3[..., 1, 2] = (rot[..., 2, 1] + rot[..., 1, 2])/3.0
            K3[..., 1, 3] = (rot[..., 2, 0] - rot[..., 0, 2])/3.0
            K3[..., 2, 0] = K3[..., 0, 2]
            K3[..., 2, 1] = K3[..., 1, 2]
            K3[..., 2, 2] = (rot[..., 2, 2] - rot[..., 0, 0] - rot[..., 1, 1])/3.0
            K3[..., 2, 3] = (rot[..., 0, 1] - rot[..., 1, 0])/3.0
            K3[..., 3, 0] = K3[..., 0, 3]
            K3[..., 3, 1] = K3[..., 1, 3]
            K3[..., 3, 2] = K3[..., 2, 3]
            K3[..., 3, 3] = (rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2])/3.0

            if not shape:
                q = np.empty((4,))
                eigvals, eigvecs = linalg.eigh(K3.T, eigvals=(3, 3))
                q[0] = eigvecs[-1]
                q[1:] = -eigvecs[:-1].flatten()
                return cls(q)
            else:
                q = np.empty(shape+(4,))
                for flat_index in range(reduce(mul, shape)):
                    multi_index = np.unravel_index(flat_index, shape)
                    eigvals, eigvecs = linalg.eigh(K3[multi_index], eigvals=(3, 3))
                    q[multi_index+(0,)] = eigvecs[-1]
                    q[multi_index+(slice(1,None),)] = -eigvecs[:-1].flatten()
                return cls(q)

        else:  # No scipy.linalg or not `nonorthogonal`
            diagonals = np.empty(shape+(4,))
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
    @jit
    def as_transformation_matrix(self):
        """Convert quaternions to 4x4 transformation matrices

        Assuming the quaternion Q transforms another quaternion P according to

            P' = Q * P * Q̄,

        we can also express this rotation in terms of a 4x4 matrix ℳ such that

            P' = ℳ * P,

        where P is viewed as a 4-vector in the last line.  This function
        returns that matrix.


        Returns
        -------
        m: float array
            Output shape is self.shape[:-1]+(4,4).  This matrix should multiply
            (from the left) a column vector to produce the transformed column
            vector.

        See also
        --------
        as_rotation_matrix: assumes Q is a unit quaternion

        """
        s = self.flattened
        m = np.zeros(s.shape[0] + (4, 4))
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
    def as_axis_angle(self):
        """Convert input quaternion to the axis-angle representation

        Note that if any of the input quaternions has norm zero, no error is
        raised, but NaNs will appear in the output.

        Returns
        -------
        rot: float array
            Output shape is q.shape[:-1]+(3,).  Each vector represents the axis of
            the rotation, with norm proportional to the angle of the rotation in
            radians.

        """
        return 2 * np.log(self.normalized).vector

    as_rotation_vector = as_axis_angle

    @classmethod
    def from_axis_angle(cls, vec):
        """Convert 3-vector in axis-angle representation to unit quaternion

        Parameters
        ----------
        vec: (...N, 3) float array
            Each vector represents the axis of the rotation, with norm
            proportional to the angle of the rotation in radians.

        Returns
        -------
        q: array of quaternions
            Unit quaternions resulting in rotations corresponding to input
            rotations.  Output shape is rot.shape[:-1].

        """
        vec = np.asarray(vec)
        quats = np.zeros(vec.shape[:-1] + (4,))
        quats[..., 1:] = vec[...] / 2
        return np.exp(cls(quats))

    from_rotation_vector = from_axis_angle

    @property
    @jit
    def as_euler_angles(self):
        """Open Pandora's Box

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
        alpha_beta_gamma: float array
            Output shape is q.shape+(3,).  These represent the angles (alpha,
            beta, gamma) in radians, where the normalized input quaternion
            represents `exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)`.

        Raises
        ------
        AllHell
            ...if you try to actually use Euler angles, when you could have
            been using quaternions like a sensible person.

        """
        s = self.flattened
        alpha_beta_gamma = np.empty(s.shape[0] + (3,))
        for i in range(s.shape[0]):
            alpha_beta_gamma[i, 0] = np.arctan2(q[i, 3], q[i, 0]) + np.arctan2(-q[i, 1], q[i, 2])
            alpha_beta_gamma[i, 1] = 2*np.arccos(np.sqrt((q[i, 0]**2 + q[i, 3]**2)/n))
            alpha_beta_gamma[i, 2] = np.arctan2(q[i, 3], q[i, 0]) - np.arctan2(-q[i, 1], q[i, 2])
        return alpha_beta_gamma.reshape(self.shape[:-1] + (3,))

    @classmethod
    def from_euler_angles(cls, alpha_beta_gamma, beta=None, gamma=None):
        """Improve your life drastically

        Assumes the Euler angles correspond to the quaternion R via

            R = exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)

        The angles naturally must be in radians for this to make any sense.

        NOTE: Before opening an issue reporting something "wrong" with this
        function, be sure to read all of the following page, *especially* the
        very last section about opening issues or pull requests.
        <https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible>

        Parameters
        ----------
        alpha_beta_gamma: float or array of floats
            This argument may either contain an array with last dimension of
            size 3, where those three elements describe the (alpha, beta, gamma)
            radian values for each rotation; or it may contain just the alpha
            values, in which case the next two arguments must also be given.
        beta: None, float, or array of floats
            If this array is given, it must be able to broadcast against the
            first and third arguments.
        gamma: None, float, or array of floats
            If this array is given, it must be able to broadcast against the
            first and second arguments.

        Returns
        -------
        R: quaternion array
            The shape of this array will be the same as the input, except that
            the last dimension will be removed.

        """
        # Figure out the input angles from either type of input
        if gamma is None:
            alpha_beta_gamma = np.asarray(alpha_beta_gamma, dtype=float)
            alpha = alpha_beta_gamma[..., 0]
            beta  = alpha_beta_gamma[..., 1]
            gamma = alpha_beta_gamma[..., 2]
        else:
            alpha = np.asarray(alpha_beta_gamma, dtype=float)
            beta  = np.asarray(beta, dtype=float)
            gamma = np.asarray(gamma, dtype=float)

        # Set up the output array
        R = np.empty(np.broadcast(alpha, beta, gamma).shape + (4,))

        # Compute the actual values of the quaternion components
        R[..., 0] =  np.cos(beta/2)*np.cos((alpha+gamma)/2)  # scalar quaternion components
        R[..., 1] = -np.sin(beta/2)*np.sin((alpha-gamma)/2)  # x quaternion components
        R[..., 2] =  np.sin(beta/2)*np.cos((alpha-gamma)/2)  # y quaternion components
        R[..., 3] =  np.cos(beta/2)*np.sin((alpha+gamma)/2)  # z quaternion components

        return cls(R)

    @property
    def as_spherical_coords(self):
        """Return the spherical coordinates corresponding to this quaternion

        Obviously, spherical coordinates do not contain as much information as a
        quaternion, so this function does lose some information.  However, the
        returned spherical coordinates will represent the point(s) on the sphere
        to which the input quaternion(s) rotate the z axis.

        Returns
        -------
        vartheta_varphi: float array
            Output shape is q.shape+(2,).  These represent the angles (vartheta,
            varphi) in radians, where the normalized input quaternion represents
            `exp(varphi*z/2) * exp(vartheta*y/2)`, up to an arbitrary inital
            rotation about `z`.

        """
        return as_euler_angles(self)[..., 1::-1]

    @classmethod
    def from_spherical_coords(cls, theta_phi, phi=None):
        """Return the quaternion corresponding to these spherical coordinates

        Assumes the spherical coordinates correspond to the quaternion R via

            R = exp(phi*z/2) * exp(theta*y/2)

        The angles naturally must be in radians for this to make any sense.

        Note that this quaternion rotates `z` onto the point with the given
        spherical coordinates, but also rotates `x` and `y` onto the usual basis
        vectors (theta and phi, respectively) at that point.

        Parameters
        ----------
        theta_phi: float or array of floats
            This argument may either contain an array with last dimension of
            size 2, where those two elements describe the (theta, phi) values in
            radians for each point; or it may contain just the theta values in
            radians, in which case the next argument must also be given.
        phi: None, float, or array of floats
            If this array is given, it must be able to broadcast against the
            first argument.

        Returns
        -------
        R: quaternion array
            If the second argument is not given to this function, the shape
            will be the same as the input shape except for the last dimension,
            which will be removed.  If the second argument is given, this
            output array will have the shape resulting from broadcasting the
            two input arrays against each other.

        """
        # Figure out the input angles from either type of input
        if phi is None:
            theta_phi = np.asarray(theta_phi, dtype=float)
            theta = theta_phi[..., 0]
            phi  = theta_phi[..., 1]
        else:
            theta = np.asarray(theta_phi, dtype=float)
            phi = np.asarray(phi, dtype=float)

        # Set up the output array
        R = np.empty(np.broadcast(theta, phi).shape + (4,))

        # Compute the actual values of the quaternion components
        R[..., 0] =  np.cos(phi/2)*np.cos(theta/2)  # scalar quaternion components
        R[..., 1] = -np.sin(phi/2)*np.sin(theta/2)  # x quaternion components
        R[..., 2] =  np.cos(phi/2)*np.sin(theta/2)  # y quaternion components
        R[..., 3] =  np.sin(phi/2)*np.cos(theta/2)  # z quaternion components

        return cls(R)
