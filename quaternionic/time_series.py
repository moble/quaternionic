# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import abc
import numpy as np
from . import jit


def angular_velocity(qarray, t, t_new=None, axis=0):
    """Compute the angular velocity of quaternion timeseries with respect to `t`

    Note that, for both unit and non-unit quaternions `Q`, we define the angular
    velocity as

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

    Parameters
    ----------
    qarray : quaternionic.array
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

    """
    spline = CubicSpline(t, qarray, axis=axis)
    if t_new is None:
        Q = self  # shortcut
    else:
        Q = type(self)(spline(t_new))
    t_new = t_new or t
    Q̇ = type(self)(spline.derivative()(t_new))
    return 2 * Q̇ / Q


def integrate_angular_velocity(ω, t, axis=0):
    raise NotImplementedError()


def minimal_rotation(qarray, t, t_new=None, axis=0, iterations=2):
    """Adjust frame so that there is no rotation about z' axis

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
        for t[i] the corresponding quaternions are `np.take(qarray, i, axis=axis)`.
        Defaults to 0.
    iterations : int, optional
        Repeat the minimization to refine the result.  Defaults to 2.

    """
    from scipy.interpolate import CubicSpline
    if iterations == 0:
        return qarray
    z = type(qarray)([0.0, 0.0, 0.0, 1.0])
    t_new = t_new or t
    s = CubicSpline(t, qarray.ndarray, axis=axis)
    R = type(qarray)(s(t_new))
    Rdot = type(qarray)(s.derivative()(t_new))
    γ̇over2 = (Rdot * z * np.conjugate(R)).w
    γover2 = CubicSpline(t_new, γ̇over2).antiderivative()(t_new)
    Rγ = np.exp(z * γover2)
    return minimal_rotation(R * Rγ, t_new, t_new=None, axis=axis, iterations=iterations-1)


def slerp(self):
    """Spherical linear interpolation of quaternions

    See also
    --------
    squad : spherical quadratic interpolation
    slerp_pairwise : Transition between to quaternionic functions of time

    """
    raise NotImplementedError()


def slerp_pairwise(self):
    """Spherical linear interpolation of quaternions

    See also
    --------
    slerp : spherical linear interpolation
    squad : spherical quadratic interpolation

    """
    raise NotImplementedError()


def squad(self):
    """Spherical quadratic interpolation of quaternions

    See also
    --------
    slerp : spherical linear interpolation

    """
    raise NotImplementedError()
