# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

"""Distance functions of quaternions

This module contains four distance functions:

  * `rotor.intrinsic`
  * `rotor.chordal`
  * `rotation.intrinsic`
  * `rotation.chordal`

The "rotor" distances do not account for possible differences in signs, meaning
that rotor distances can be large even when they represent identical rotations;
the "rotation" functions just return the smaller of the distance between `q1`
and `q2` or the distance between `q1` and `-q2`.  So, for example, either
"rotation" distance between `q` and `-q` is always zero, whereas neither
"rotor" distance between `q` and `-q` will ever be zero (unless `q` is zero).
The "intrinsic" functions measure the geodesic distance within the manifold of
*unit* quaternions, and is somewhat slower but may be more meaningful; the
"chordal" functions measure the Euclidean distance in the (linear) space of all
quaternions, and is faster but its precise value is not necessarily as
meaningful.

These functions satisfy some important conditions.  For each of these functions
`d`, and for any nonzero quaternions `q1` and `q2`, and *unit* quaternions `q3`
and `q4`, we have

  * symmetry: `d(q1, q2) = d(q2, q1)`
  * invariance: `d(q3*q1, q3*q2) = d(q1, q2) = d(q1*q4, q2*q4)`
  * identity: `d(q1, q1) = 0`
  * positive-definiteness:
    * For rotor functions `d(q1, q2) > 0` whenever `q1 ≠ q2`
    * For rotation functions `d(q1, q2) > 0` whenever `q1 ≠ q2` and `q1 ≠ -q2`

Note that the rotation functions also satisfy both the usual identity property
`d(q1, q1) = 0` and the opposite-identity property `d(q1, -q1) = 0`.

See [Moakher (2002)](https://doi.org/10.1137/S0895479801383877) for a nice
general discussion.

"""

import numpy as np
from . import jit, guvectorize, algebra, float64
from .utilities import ndarray_args

_divide = jit(algebra.divide)
_log = jit(algebra.log)
_absolute = jit(algebra.absolute)


def create_rotor_metrics(jit=jit, guvectorize=guvectorize):
    class Rotor(object):
        @ndarray_args
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()')
        def intrinsic(q1, q2, out):
            """Geodesic distance between rotors within the Spin(3)=SU(2) manifold.

            This function is equivalent to

                np.absolute(np.log(q1 / q2))

            which is a measure of the "distance" between two quaternions.  Note
            that no normalization is performed, which means that if q1 and/or
            q2 do not have unit norm, this is a more general type of distance.
            If they are normalized, the result of this function is half the
            angle through which vectors rotated by q1 would need to be rotated
            to lie on the same vectors rotated by q2.

            Parameters
            ----------
            q1, q2 : QuaternionicArray
                Quaternionic arrays to be measured

            See also
            --------
            quaternionic.distance.rotor.chordal
            quaternionic.distance.rotation.intrinsic
            quaternionic.distance.rotation.chordal

            """
            qtemp = np.empty(4)
            _divide(q1, q2, qtemp)
            _log(qtemp, qtemp)
            _absolute(qtemp, out)
            out[0] *= 2

        @ndarray_args
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()')
        def chordal(q1, q2, out):
            """Euclidean distance between rotors.

            This function is equivalent to

                np.absolute(q1 - q2)

            Note that no normalization is performed.  If the quaternions are
            normalized, this represents the length of the chord joining these
            two points on the unit 3-sphere, considered as embedded in
            Euclidean 4-space.

            Parameters
            ----------
            q1, q2 : QuaternionicArray
                Quaternionic arrays to be measured

            See also
            --------
            quaternionic.distance.rotor.intrinsic
            quaternionic.distance.rotation.intrinsic
            quaternionic.distance.rotation.chordal

            """
            out[0] = np.sqrt((q1[0]-q2[0])**2 + (q1[1]-q2[1])**2 + (q1[2]-q2[2])**2 + (q1[3]-q2[3])**2)

    return Rotor


def create_rotation_metrics(jit=jit, guvectorize=guvectorize):
    class Rotation(object):
        @ndarray_args
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()')
        def intrinsic(q1, q2, out):
            """Geodesic distance between rotations within the SO(3) manifold.

            This function is equivalent to

                min(
                    np.absolute(np.log(q1 / q2)),
                    np.absolute(np.log(q1 / -q2))
                )

            which is a measure of the "distance" between two rotations.  Note
            that no normalization is performed, which means that if q1 and/or
            q2 do not have unit norm, this is a more general type of distance.
            If they are normalized, the result of this function is half the
            angle through which vectors rotated by q1 would need to be rotated
            to lie on the same vectors rotated by q2.

            Parameters
            ----------
            q1, q2 : QuaternionicArray
                Quaternionic arrays to be measured

            See also
            --------
            quaternionic.distance.rotor.chordal
            quaternionic.distance.rotor.intrinsic
            quaternionic.distance.rotation.chordal

            """
            qtemp = np.empty(4)
            a = (q1[0]-q2[0])**2 + (q1[1]-q2[1])**2 + (q1[2]-q2[2])**2 + (q1[3]-q2[3])**2
            b = (q1[0]+q2[0])**2 + (q1[1]+q2[1])**2 + (q1[2]+q2[2])**2 + (q1[3]+q2[3])**2
            if b > a:
                _divide(q1, q2, qtemp)
            else:
                _divide(q1, -q2, qtemp)
            _log(qtemp, qtemp)
            _absolute(qtemp, out)
            out[0] *= 2

        @ndarray_args
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()')
        def chordal(q1, q2, out):
            """Euclidean distance between rotations.

            This function is equivalent to

                min(
                    np.absolute(q1 - q2),
                    np.absolute(q1 + q2)
                )

            Note that no normalization is performed.  If the quaternions are
            normalized, this represents the length of the chord joining these
            two points on the unit 3-sphere, considered as embedded in
            Euclidean 4-space.

            Parameters
            ----------
            q1, q2 : QuaternionicArray
                Quaternionic arrays to be measured

            See also
            --------
            quaternionic.distance.rotor.intrinsic
            quaternionic.distance.rotor.chordal
            quaternionic.distance.rotation.intrinsic

            """
            a = (q1[0]-q2[0])**2 + (q1[1]-q2[1])**2 + (q1[2]-q2[2])**2 + (q1[3]-q2[3])**2
            b = (q1[0]+q2[0])**2 + (q1[1]+q2[1])**2 + (q1[2]+q2[2])**2 + (q1[3]+q2[3])**2
            if b > a:
                out[0] = np.sqrt(a)
            else:
                out[0] = np.sqrt(b)

    return Rotation


rotor = create_rotor_metrics()
rotation = create_rotation_metrics()
