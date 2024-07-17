# Copyright (c) 2024, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import numpy as np
from numba import float64

import quaternionic
from . import guvectorize


@guvectorize([(float64[:, :], float64[:, :])], '(i,j)->(i,j)')
def _unflip_gufunc(q_in, q_out):  # pragma: no cover
    q_out[0, :] = q_in[0, :]
    for i in range(1, q_in.shape[0]):
        inner_product = (
            q_out[i-1, 0] * q_in[i, 0]
            + q_out[i-1, 1] * q_in[i, 1]
            + q_out[i-1, 2] * q_in[i, 2]
            + q_out[i-1, 3] * q_in[i, 3]
        )
        if inner_product < 0.0:
            q_out[i, :] = -q_in[i, :]
        else:
            q_out[i, :] = q_in[i, :]


def unflip_rotors(R_in, axis=-2, inplace=False):
    """Flip signs of quaternions along axis to ensure continuity

    Parameters
    ----------
    R_in : array_like
        Quaternionic array to modify.
    axis : int, optional
        Axis along which successive quaternions will be compared.  Default value is
        the second-to-last last axis of the array (the last before the quaternion
        index).
    inplace : bool, optional
        If True, modify the data in place without creating a copy; if False (the
        default), a new array is created and returned.

    Returns
    -------
    R_out : array_like
        An array of precisely the same shape as the input array, differing only by
        factors of precisely -1 in some elements.

    Notes
    -----
    Quaternions form a "double cover" of the rotation group, meaning that if `q`
    represents a rotation, then `-q` represents the same rotation.  This is clear
    from the way a quaternion is used to rotate a vector `v`: the rotated vector is
    `q * v * q.conjugate()`, which is precisely the same as the vector resulting
    from `(-q) * v * (-q).conjugate()`.  Some ways of constructing quaternions
    (such as converting from rotation matrices or other representations) can result
    in unexpected sign choices.  For many applications, this will not be a problem.
    But if, for example, the quaternions need to be interpolated or differentiated,
    the results may be surprising.  This function flips the signs of successive
    quaternions (along some chosen axis, if relevant), so that successive
    quaternions are as close as possible while still representing the same
    rotations.

    This function works by taking the inner product between successive quaternions
    (along `axis`); whenever that product is negative, the sign of the second
    quaternion is flipped.  This does not depend on the quaternions being
    normalized.

    """
    R_in = quaternionic.array(R_in)
    if R_in.ndim == 1:
        if R_in.w < 0:
            if inplace:
                R_in *= -1
                return R_in
            else:
                return -R_in
        else:
            return R_in
    axis = axis % R_in.ndim
    if axis >= R_in.ndim - 1:
        raise ValueError(f"Requested axis {axis} is outside the input array's non-quaternionic shape {R_in.shape[:-1]}")
    f_in = R_in.ndarray
    axes = [(axis, -1), (axis, -1)]
    f_out = f_in if inplace else np.empty_like(f_in)
    _unflip_gufunc(f_in, out=f_out, axes=axes)
    return quaternionic.array(f_out)


def slerp(R1, R2, tau):
    """Spherical linear interpolation of quaternions

    The result of a "slerp" is given by

        (R2 / R1) ** tau * R1

    When `tau` is 0, this evaluates to `R1`; when `tau` is 1, this evaluates to
    `R2`; for any other values the result smoothly varies between the two.

    Parameters
    ----------
    R1 : quaternionic.array
        Value at `tau` == 0
    R2 : quaternionic.array
        Value at `tau` == 1
    tau : array_like
        Fractional contribution from `R2` relative to `R1`.  It is permissible to
        use values outside of the range [0, 1], but note that this corresponds to
        extrapolation.  This must broadcast against the result of `R2 / R1` — for
        example, it could be a single number, or it could have the same shape as
        `R1` and `R2`.

    See also
    --------
    squad : spherical quadratic interpolation

    """
    return np.float_power(R2 / R1, tau) * R1


def squad(R_in, t_in, t_out):
    """Spherical "quadrangular" interpolation of rotors with a cubic spline

    Parameters
    ----------
    R_in : array of quaternions
        A time-series of rotors (unit quaternions) to be interpolated
    t_in : array of float
        The times corresponding to R_in
    t_out : array of float
        The times to which R_in should be interpolated

    See Also
    --------
    slerp : Spherical linear interpolation

    Notes
    -----
    This is a smoother way to interpolate a rotation as a function of time than
    simple `slerp`.  It uses the analog of a cubic spline, except that the
    interpolant is confined to the rotor manifold in a natural way.  Alternative
    methods involving interpolation of other coordinates on the rotation group or
    normalization of interpolated values give bad results.  The results from this
    method are fairly natural, and are continuous in first derivatives.

    The input `R_in` rotors are assumed to be reasonably continuous (no sign
    flips), and the input `t` arrays are assumed to be sorted.  No checking is done
    for either case, and you may get silently bad results if these conditions are
    violated.  The first dimension of `R_in` must have the same size as `t_in`, but
    may have additional axes following.

    """
    from functools import partial

    roll = partial(np.roll, axis=0)

    if R_in.size == 0 or t_out.size == 0:
        return quaternionic.array(np.zeros((0, 4)))

    R_in = quaternionic.array(R_in)

    # This list contains an index for each `t_out` such that
    # t_in[i-1] <= t_out < t_in[i]
    # Note that `side='right'` is much faster in my tests
    # i_in_for_out = t_in.searchsorted(t_out, side='left')
    # np.clip(i_in_for_out, 0, len(t_in) - 1, out=i_in_for_out)
    i_in_for_out = t_in.searchsorted(t_out, side='right')-1

    # Compute shapes used to broadcast `t` arrays against `R` arrays
    t_in_broadcast_shape = t_in.shape + (1,)*len(R_in.shape[1:])
    t_out_broadcast_shape = t_out.shape + (1,)*len(R_in.shape[1:])

    # Now, for each index `i` in `i_in`, we need to compute the
    # interpolation "coefficients" (`A_i`, `B_ip1`).
    #
    # I previously tested an explicit version of the loops below,
    # comparing `stride_tricks.as_strided` with explicit
    # implementation via `roll` (as seen here).  I found that the
    # `roll` was significantly more efficient for simple calculations,
    # though the difference is probably totally washed out here.  In
    # any case, it might be useful to test again.
    #
    A = R_in * np.exp(
        (
            - np.log((~R_in) * roll(R_in, -1))
            + np.log((~roll(R_in, 1)) * R_in)
            * np.reshape((roll(t_in, -1) - t_in) / (t_in - roll(t_in, 1)), t_in_broadcast_shape)[..., 0]
         ) * 0.25
    )
    B = roll(R_in, -1) * np.exp(
        (
            np.log((~roll(R_in, -1)) * roll(R_in, -2))
            * np.reshape((roll(t_in, -1) - t_in) / (roll(t_in, -2) - roll(t_in, -1)), t_in_broadcast_shape)[..., 0]
            - np.log((~R_in) * roll(R_in, -1))
        ) * -0.25
    )

    # Correct the first and last A time steps, and last two B time steps.  We extend R_in with the following wrap-around
    # values:
    # R_in[0-1] = R_in[0]*(~R_in[1])*R_in[0]
    # R_in[n+0] = R_in[-1] * (~R_in[-2]) * R_in[-1]
    # R_in[n+1] = R_in[0] * (~R_in[-1]) * R_in[0]
    #           = R_in[-1] * (~R_in[-2]) * R_in[-1] * (~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1]
    #           = R_in[-1] * (~R_in[-2]) * R_in[-1] * (~R_in[-2]) * R_in[-1]
    # A[i] = R_in[i] * np.exp((- np.log((~R_in[i]) * R_in[i+1])
    #                          + np.log((~R_in[i-1]) * R_in[i]) * ((t_in[i+1] - t_in[i]) / (t_in[i] - t_in[i-1]))
    #                          ) * 0.25)
    # A[0] = R_in[0] * np.exp((- np.log((~R_in[0]) * R_in[1]) + np.log((~R_in[0])*R_in[1]*(~R_in[0])) * R_in[0]) * 0.25)
    #      = R_in[0]
    A[0] = R_in[0]
    # A[-1] = R_in[-1] * np.exp((- np.log((~R_in[-1]) * R_in[n+0])
    #                          + np.log((~R_in[-2]) * R_in[-1]) * ((t_in[n+0] - t_in[-1]) / (t_in[-1] - t_in[-2]))
    #                          ) * 0.25)
    #       = R_in[-1] * np.exp((- np.log((~R_in[-1]) * R_in[n+0]) + np.log((~R_in[-2]) * R_in[-1])) * 0.25)
    #       = R_in[-1] * np.exp((- np.log((~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1])
    #                           + np.log((~R_in[-2]) * R_in[-1])) * 0.25)
    #       = R_in[-1] * np.exp((- np.log((~R_in[-2]) * R_in[-1]) + np.log((~R_in[-2]) * R_in[-1])) * 0.25)
    #       = R_in[-1]
    A[-1] = R_in[-1]
    # B[i] = R_in[i+1] * np.exp((np.log((~R_in[i+1]) * R_in[i+2]) * ((t_in[i+1] - t_in[i]) / (t_in[i+2] - t_in[i+1]))
    #                            - np.log((~R_in[i]) * R_in[i+1])) * -0.25)
    # B[-2] = R_in[-1] * np.exp((np.log((~R_in[-1]) * R_in[0]) * ((t_in[-1] - t_in[-2]) / (t_in[0] - t_in[-1]))
    #                            - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1] * np.exp((np.log((~R_in[-1]) * R_in[0]) - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1] * np.exp((np.log((~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1])
    #                            - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1] * np.exp((np.log((~R_in[-2]) * R_in[-1]) - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1]
    B[-2] = R_in[-1]
    # B[-1] = R_in[0]
    # B[-1] = R_in[0] * np.exp((np.log((~R_in[0]) * R_in[1]) - np.log((~R_in[-1]) * R_in[0])) * -0.25)
    #       = R_in[-1] * (~R_in[-2]) * R_in[-1]
    #         * np.exp((np.log((~(R_in[-1] * (~R_in[-2]) * R_in[-1])) * R_in[-1] * (~R_in[-2]) * R_in[-1] * (~R_in[-2]) * R_in[-1])
    #                  - np.log((~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1])) * -0.25)
    #       = R_in[-1] * (~R_in[-2]) * R_in[-1]
    #         * np.exp((np.log(((~R_in[-1]) * R_in[-2] * (~R_in[-1])) * R_in[-1] * (~R_in[-2]) * R_in[-1] * (~R_in[-2]) * R_in[-1])
    #                  - np.log((~R_in[-1]) * R_in[-1] * (~R_in[-2]) * R_in[-1])) * -0.25)
    #         * np.exp((np.log((~R_in[-2]) * R_in[-1])
    #                  - np.log((~R_in[-2]) * R_in[-1])) * -0.25)
    B[-1] = R_in[-1] * (~R_in[-2]) * R_in[-1]

    # Use the coefficients at the corresponding t_out indices to
    # compute the squad interpolant
    # R_ip1 = np.array(roll(R_in, -1)[i_in_for_out])
    # R_ip1[-1] = R_in[-1]*(~R_in[-2])*R_in[-1]
    R_ip1 = roll(R_in, -1)
    R_ip1[-1] = R_in[-1]*(~R_in[-2])*R_in[-1]
    R_ip1 = quaternionic.array(R_ip1[i_in_for_out])
    t_inp1 = roll(t_in, -1)
    t_inp1[-1] = t_in[-1] + (t_in[-1] - t_in[-2])
    tau = np.reshape((t_out - t_in[i_in_for_out]) / ((t_inp1 - t_in)[i_in_for_out]), t_out_broadcast_shape)[..., 0]
    # tau = (t_out - t_in[i_in_for_out]) / ((roll(t_in, -1) - t_in)[i_in_for_out])

    R_out = slerp(
        slerp(R_in[i_in_for_out], R_ip1, tau),
        slerp(A[i_in_for_out], B[i_in_for_out], tau),
        2*tau*(1-tau)
    )

    # R_out = np.squad_vectorized(tau, R_in[i_in_for_out], A[i_in_for_out], B[i_in_for_out], R_ip1)

    return R_out
