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
            qtemp = np.empty(4)
            _divide(q1, q2, qtemp)
            _log(qtemp, qtemp)
            _absolute(qtemp, out)
            out[0] *= 2

        @ndarray_args
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()')
        def chordal(q1, q2, out):
            out[0] = np.sqrt((q1[0]-q2[0])**2 + (q1[1]-q2[1])**2 + (q1[2]-q2[2])**2 + (q1[3]-q2[3])**2)


    class Rotation(object):
        @ndarray_args
        @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()')
        def intrinsic(q1, q2, out):
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
            a = (q1[0]-q2[0])**2 + (q1[1]-q2[1])**2 + (q1[2]-q2[2])**2 + (q1[3]-q2[3])**2
            b = (q1[0]+q2[0])**2 + (q1[1]+q2[1])**2 + (q1[2]+q2[2])**2 + (q1[3]+q2[3])**2
            if b > a:
                out[0] = np.sqrt(a)
            else:
                out[0] = np.sqrt(b)


    return Rotor, Rotation


rotor, rotation = CreateMetrics()
