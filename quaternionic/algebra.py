import numpy as np
from . import guvectorize

_quaternion_resolution = 10 * np.finfo(float).resolution

unary_guvectorize = guvectorize([(nb.float64[:], nb.float64[:])], '(n)->(n)')
binary_guvectorize = guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], '(n),(n)->(n)')


@binary_guvectorize
def add(q1, q2, qout):
    qout[:] = q1[:] + q2[:]


@binary_guvectorize
def subtract(q1, q2, qout):
    qout[:] = q1[:] - q2[:]


@binary_guvectorize
def multiply(q1, q2, qout):
    qout[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    qout[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    qout[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    qout[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]


@binary_guvectorize
def divide(q1, q2, qout):
    q2norm = q2[0]**2 + q2[1]**2 + q2[2]**2 + q2[3]**2
    qout[0] = (+q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]) / q2norm
    qout[1] = (-q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2]) / q2norm
    qout[2] = (-q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0] - q1[3]*q2[1]) / q2norm
    qout[3] = (-q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] + q1[3]*q2[0]) / q2norm


true_divide = divide


@guvectorize([(nb.float64, nb.float64[:], nb.float64[:])], '(),(n)->(n)')
def multiply_scalar(s, q, qout):
    qout[:] = s * q[:]


@guvectorize([(nb.float64, nb.float64[:], nb.float64[:])], '(),(n)->(n)')
def divide_scalar(s, q, qout):
    qnorm = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    qout[0] = s * q[0] / qnorm
    qout[1:] = -s * q[1:] / qnorm


true_divide_scalar = divide_scalar


@guvectorize([(nb.float64[:], nb.float64, nb.float64[:])], '(n),()->(n)')
def scalar_multiply(q, s, qout):
    qout[:] = q[:] * s


@guvectorize([(nb.float64[:], nb.float64, nb.float64[:])], '(n),()->(n)')
def scalar_divide(q, s, qout):
    qout[:] = q[:] / s


scalar_true_divide = scalar_divide


@unary_guvectorize
def negative(qin, qout):
    qout[:] = -qin[:]


@unary_guvectorize
def positive(qin, qout):
    qout[:] = +qin[:]


@guvectorize([(nb.float64[:], nb.float64, nb.float64[:])], '(n),()->(n)')
def float_power(qin, s, qout):
    b = np.sqrt(qin[1]**2 + qin[2]**2 + qin[3]**2)
    if np.abs(b) <= _quaternion_resolution * np.abs(qin[0]):
        if qin[0] < 0.0:
            if np.abs(qin[0] + 1) > _quaternion_resolution:
                qout[0] = np.log(-qin[0])
                qout[1] = np.pi
                qout[2:] = 0.0
            else:
                qout[:] = 0.0
                qout[1] = np.pi
        else:
            qout[0] = np.log(qin[0])
            qout[1:] = 0.0
    else:
        v = np.atan2(b, qin[0])
        f = v / b
        qout[0] = np.log(qin[0] * qin[0] + b * b) / 2.0
        qout[1:] = f * qin[1:]
    qout *= s
    vnorm = np.sqrt(qout[1]**2 + qout[2]**2 + qout[3]**2)
    if vnorm > _quaternion_resolution:
        e = np.exp(qout[0])
        qout[0] = e * np.cos(vnorm)
        qout[1:] *= e * np.sin(vnorm) / vnorm
    else:
        qout[0] = np.exp(qout[0])
        qout[1:] = 0.0


@guvectorize([(nb.float64[:], nb.float64)], '(n)->()')
def absolute(qin, qout):
    qout[:] = np.sqrt(qin[0]**2 + qin[1]**2 + qin[2]**2 + qin[3]**2)


@unary_guvectorize
def conj(qin, qout):
    qout[0] = +qin[0]
    qout[1:] = -qin[1:]


conjugate = conj


@unary_guvectorize
def exp(qin, qout):
    vnorm = np.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
    if vnorm > _quaternion_resolution:
        s = np.sin(vnorm) / vnorm
        e = np.exp(q[0])
        qout[0] = e * cos(vnorm)
        qout[1] = e * s * q[1]
        qout[2] = e * s * q[2]
        qout[3] = e * s * q[3]
    else:
        qout[0] = np.exp(q[0])
        qout[1:] = 0.0


@unary_guvectorize
def log(qin, qout):
    b = np.sqrt(qin[1]**2 + qin[2]**2 + qin[3]**2)
    if np.abs(b) <= _quaternion_resolution * np.abs(qin[0]):
        if qin[0] < 0.0:
            if np.abs(qin[0] + 1) > _quaternion_resolution:
                qout[0] = np.log(-qin[0])
                qout[1] = np.pi
                qout[2:] = 0.0
            else:
                qout[:] = 0.0
                qout[1] = np.pi
        else:
            qout[0] = np.log(qin[0])
            qout[1:] = 0.0
    else:
        v = np.atan2(b, qin[0])
        f = v / b
        qout[0] = np.log(qin[0] * qin[0] + b * b) / 2.0
        qout[1:] = f * qin[1:]


@unary_guvectorize
def sqrt(qin, qout):
    # √Q = (n + Q) / √(2*n + 2*Q[0])
    n = np.sqrt(qin[0]**2 + qin[1]**2 + qin[2]**2 + qin[3]**2)
    if np.abs(n + qin[0]) < _quaternion_resolution * n:
        qout[:] = 0.0
        qout[1] = np.sqrt(n)
    else:
        c = np.sqrt(0.5 / (n + qin[0]))
        qout[:] = qin[:] * c
        qout[0] += n * c


@unary_guvectorize
def square(qin, qout):
    qout[0] = qin[0]**2 - qin[1]**2 - qin[2]**2 - qin[3]**2
    qout[1] = 2*qin[0]*qin[1]
    qout[2] = 2*qin[0]*qin[2]
    qout[3] = 2*qin[0]*qin[3]


@unary_guvectorize
def reciprocal(qin, qout):
    norm = qin[0]**2 + qin[1]**2 + qin[2]**2 + qin[3]**2
    qout[0] = qout[0] / norm
    qout[1:] = -qout[1:] / norm


@binary_guvectorize
def bitwise_or(q1, q2, qout):  # inner product
    raise NotImplementedError()


@binary_guvectorize
def bitwise_xor(q1, q2, qout):  # outer product
    raise NotImplementedError()


invert = conj  # reversion (= conjugate in quaternions)


@binary_guvectorize
def left_shift(q1, q2, qout):  # left contraction (= reverse of right-contraction of reverses)
    raise NotImplementedError()


@binary_guvectorize
def right_shift(q1, q2, qout):  # right contraction (= reverse of left-contraction of reverses)
    raise NotImplementedError()


@guvectorize([(nb.float64[:], nb.float64[:], nb.boolean)], '(n),(n)->()')
def not_equal(q1, q2, qout):
    qout[:] = np.any(q1[:] != q2[:])


@guvectorize([(nb.float64[:], nb.float64[:], nb.boolean)], '(n),(n)->()')
def equal(q1, q2, qout):
    qout[:] = np.all(q1[:] == q2[:])


@guvectorize([(nb.float64[:], nb.float64[:], nb.boolean)], '(n),(n)->()')
def logical_and(q1, q2, qout):
    qout[:] = np.any(q1[:]) and np.any(q2[:])


@guvectorize([(nb.float64[:], nb.float64[:], nb.boolean)], '(n),(n)->()')
def logical_or(q1, q2, qout):
    qout[:] = np.any(q1[:]) or np.any(q2[:])


@guvectorize([(nb.float64[:], nb.boolean)], '(n)->()')
def isfinite(qin, qout):
    qout[:] = np.isfinite(qin[0]) and np.isfinite(qin[1]) and np.isfinite(qin[2]) and np.isfinite(qin[3])


@guvectorize([(nb.float64[:], nb.boolean)], '(n)->()')
def isinf(qin, qout):
    qout[:] = np.isinf(qin[0]) or np.isinf(qin[1]) or np.isinf(qin[2]) or np.isinf(qin[3])


@guvectorize([(nb.float64[:], nb.boolean)], '(n)->()')
def isnan(qin, qout):
    qout[:] = np.isnan(qin[0]) or np.isnan(qin[1]) or np.isnan(qin[2]) or np.isnan(qin[3])
