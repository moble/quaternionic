# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import numpy as np
from numba import float64
from . import jit, guvectorize, algebra
from .utilities import ndarray_args

_divide = jit(algebra.divide)
_log = jit(algebra.log)
_absolute = jit(algebra.absolute)


def CreateMetrics(jit=jit, guvectorize=guvectorize):
    class Rotor(object):
        @ndarray_args
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()')
        def intrinsic(q1, q2, out):
            """Geodesic distance between rotors within the Spin(3)=SU(2) manifold

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
            q1, q2: QuaternionicArray
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
            """Euclidean distance between rotors

            This function is equivalent to

                np.absolute(q1 - q2)

            Note that no normalization is performed.  If the quaternions are
            normalized, this represents the length of the chord joining these
            two points on the unit 3-sphere, considered as embedded in
            Euclidean 4-space.

            Parameters
            ----------
            q1, q2: QuaternionicArray
                Quaternionic arrays to be measured

            See also
            --------
            quaternionic.distance.rotor.intrinsic
            quaternionic.distance.rotation.intrinsic
            quaternionic.distance.rotation.chordal

            """
            out[0] = np.sqrt((q1[0]-q2[0])**2 + (q1[1]-q2[1])**2 + (q1[2]-q2[2])**2 + (q1[3]-q2[3])**2)


    class Rotation(object):
        @ndarray_args
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()')
        def intrinsic(q1, q2, out):
            """Geodesic distance between rotations within the SO(3) manifold

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
            q1, q2: QuaternionicArray
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
            """Euclidean distance between rotations

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
            q1, q2: QuaternionicArray
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


    return Rotor, Rotation


rotor, rotation = CreateMetrics()
