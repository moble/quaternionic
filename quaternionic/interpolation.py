# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import numpy as np
import quaternionic


def slerp_pairwise(R1, R2, t, tau):
    """Spherical linear interpolation between quaternionic functions of time

    Parameters
    ----------
    R1 : quaternionic.array
    R2 : quaternionic.array
        Quaternionic functions of time representing "close" but different
        orientations that we smoothly transition between.
    t : array_like
        Times at which `R1` and `R2` are measured.  This must be the same shape as
        `R1.shape[0]` and `R2.shape[0]`.
    tau : array_like
        Fractional contribution from `R2` — meaning that when `tau` is 0, the
        output will be precisely the value of `R1` at that instant; when `tau` is
        1, the output will be precisely `R2`; and when `tau` is between those
        values, the result will be linearly interpolated between the values of `R1`
        and `R2` at that instant.  It is permissible to use values outside of this
        range, but note that this corresponds to extrapolation.

    See also
    --------
    slerp : spherical linear interpolation
    squad : spherical quadratic interpolation

    """
    # if any((tau < 0) | (tau > 1)):
    #     raise ValueError("All values of tau must be betwen 0 and 1, inclusive.")
    if t.ndim != 1 or t.shape[0] != R1.shape[0] or t.shape[0] != R2.shape[0]:
        raise ValueError(f"Shape of t {t.shape} is not consistent with shapes of R1 {R1.shape} and R2 {R2.shape}.")
    if R1.shape != R2.shape:
        raise ValueError("Shapes of R1 {R1.shape} and R2 {R2.shape} are inconsistent.")
    R1 = quaternionic.array(R1)
    R2 = quaternionic.array(R2)
    R3 = quaternionic.array(np.copy(R1))
    R3[tau == 1] = R2[tau == 1]
    work_indices = (tau != 0) & (tau != 1)
    R1w = R1[work_indices]
    R2w = R2[work_indices]
    tauw = tau[work_indices]
    R3[work_indices] = (R2w / R1w) ** tauw * R1w
    return R3


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
    return (R2 / R1) ** tau * R1


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
    slerp_pairwise : Transition between to quaternionic functions of time

    Notes
    -----
    This is the smoothest way to interpolate a rotation as a function of time.  It
    uses the analog of a cubic spline, except that the interpolant is confined to
    the rotor manifold in a natural way.  Alternative methods involving
    interpolation of other coordinates on the rotation group or normalization of
    interpolated values give bad results.  The results from this method are as
    natural as any, and are continuous in first and second derivatives.

    The input `R_in` rotors are assumed to be reasonably continuous (no sign
    flips), and the input `t` arrays are assumed to be sorted.  No checking is done
    for either case, and you may get silently bad results if these conditions are
    violated.  The first dimension of `R_in` must have the same size as `t_in`, but
    may have additional axes following.

    """
    from functools import partial

    roll = partial(np.roll, axis=0)

    if R_in.size == 0 or t_out.size == 0:
        return np.array((), dtype=np.quaternion)

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
    A = R_in * np.exp((- np.log((~R_in) * roll(R_in, -1))
                       + np.log((~roll(R_in, 1)) * R_in)
                       * np.reshape((roll(t_in, -1) - t_in) / (t_in - roll(t_in, 1)), t_in_broadcast_shape)
                       ) * 0.25)
    B = roll(R_in, -1) * np.exp((np.log((~roll(R_in, -1)) * roll(R_in, -2))
                                    * np.reshape((roll(t_in, -1) - t_in) / (roll(t_in, -2) - roll(t_in, -1)), t_in_broadcast_shape)
                                    - np.log((~R_in) * roll(R_in, -1))) * -0.25)

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
    R_ip1 = np.array(R_ip1[i_in_for_out])
    t_inp1 = roll(t_in, -1)
    t_inp1[-1] = t_in[-1] + (t_in[-1] - t_in[-2])
    tau = np.reshape((t_out - t_in[i_in_for_out]) / ((t_inp1 - t_in)[i_in_for_out]), t_out_broadcast_shape)
    # tau = (t_out - t_in[i_in_for_out]) / ((roll(t_in, -1) - t_in)[i_in_for_out])

    raise NotImplementedError()
    R_out = np.squad_vectorized(tau, R_in[i_in_for_out], A[i_in_for_out], B[i_in_for_out], R_ip1)

    return R_out
