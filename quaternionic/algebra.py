# Copyright (c) 2024, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

"""Essential functions for quaternion algebra.

These functions form the basic algebraic behavior of quaternions — addition,
multiplication, exp, log, etc.

Each function takes one or two array-like inputs — depending on whether it is
unary or binary — as the first parameter or two, and another array-like object
for output as the final parameter.  Even for functions that return a single
float or bool, the output must be array-like so that it can be modified inside
the function.  These functions are JIT-compiled by numba's `guvectorize`
function, meaning that they can also act on arbitrary arrays just like standard
numpy arrays, as long as the final dimension of any quaternion-valued input has
size 4 to represent the components of the quaternion.

These functions are generic, meaning that they can be used without the
`quaternionic.array` object.  However, these functions are implemented as the
"ufunc"s of those arrays, meaning that we can perform algebra directly on them
in natural ways — as in `q1+q2`, `q1*q2`, etc. — and using the standard numpy
functions — as in `np.exp(q)`, `np.log(q)`, etc.

For this purpose, we implement as many of [the standard
ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html) as make sense for
quaternions.  For the most part, this means ignoring operations related to
integers, remainders, ordering, or trigonometric functions.  The bit-twiddling
functions are re-interpreted as they usually are in geometric algebra to denote
geometric operations.

All functions in this module are magically compiled to ufuncs in `__init__.py`
and placed attached to the `algebra_ufuncs` object.

"""


import numpy as np
from . import float64, boolean
from .utilities import attach_typelist_and_signature


_quaternion_resolution = 10 * np.finfo(float).resolution


@attach_typelist_and_signature([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def add(q1, q2, qout):
    """Add two quaternions q1+q2"""
    qout[0] = q1[0] + q2[0]
    qout[1] = q1[1] + q2[1]
    qout[2] = q1[2] + q2[2]
    qout[3] = q1[3] + q2[3]


@attach_typelist_and_signature([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def subtract(q1, q2, qout):
    """Subtract quaternion q1-q2"""
    qout[0] = q1[0] - q2[0]
    qout[1] = q1[1] - q2[1]
    qout[2] = q1[2] - q2[2]
    qout[3] = q1[3] - q2[3]


@attach_typelist_and_signature([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def multiply(q1, q2, qout):
    """Multiply quaternions q1*q2"""
    a = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    b = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    c = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    d = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    qout[0] = a
    qout[1] = b
    qout[2] = c
    qout[3] = d


@attach_typelist_and_signature([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def divide(q1, q2, qout):
    """Divide quaternions q1/q2 = q1 * q2.inverse"""
    q2norm = q2[0]**2 + q2[1]**2 + q2[2]**2 + q2[3]**2
    a = (+q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]) / q2norm
    b = (-q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2]) / q2norm
    c = (-q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0] - q1[3]*q2[1]) / q2norm
    d = (-q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] + q1[3]*q2[0]) / q2norm
    qout[0] = a
    qout[1] = b
    qout[2] = c
    qout[3] = d


true_divide = divide


@attach_typelist_and_signature([(float64, float64[:], float64[:])], '(),(n)->(n)')
def multiply_scalar(s, q, qout):
    """Multiply scalar by quaternion s*q"""
    qout[0] = s * q[0]
    qout[1] = s * q[1]
    qout[2] = s * q[2]
    qout[3] = s * q[3]


@attach_typelist_and_signature([(float64, float64[:], float64[:])], '(),(n)->(n)')
def divide_scalar(s, q, qout):
    """Divide scalar by quaternion s/q = s * q.inverse"""
    qnorm = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    qout[0] = s * q[0] / qnorm
    qout[1] = -s * q[1] / qnorm
    qout[2] = -s * q[2] / qnorm
    qout[3] = -s * q[3] / qnorm


true_divide_scalar = divide_scalar


@attach_typelist_and_signature([(float64[:], float64, float64[:])], '(n),()->(n)')
def scalar_multiply(q, s, qout):
    """Multiply quaternion by scalar q*s"""
    qout[0] = q[0] * s
    qout[1] = q[1] * s
    qout[2] = q[2] * s
    qout[3] = q[3] * s


@attach_typelist_and_signature([(float64[:], float64, float64[:])], '(n),()->(n)')
def scalar_divide(q, s, qout):
    """Divide quaternion by scalar q/s"""
    qout[0] = q[0] / s
    qout[1] = q[1] / s
    qout[2] = q[2] / s
    qout[3] = q[3] / s


scalar_true_divide = scalar_divide


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def negative(q, qout):
    """Return negative quaternion -q"""
    qout[0] = -q[0]
    qout[1] = -q[1]
    qout[2] = -q[2]
    qout[3] = -q[3]


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def positive(q, qout):
    """Return input quaternion q"""
    qout[0] = q[0]
    qout[1] = q[1]
    qout[2] = q[2]
    qout[3] = q[3]


@attach_typelist_and_signature([(float64[:], float64, float64[:])], '(n),()->(n)')
def float_power(q, s, qout):
    """Raise quaternion to scalar power exp(log(q)*s)"""
    b = np.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
    if np.abs(b) <= _quaternion_resolution * np.abs(q[0]):
        if q[0] < 0.0:
            if np.abs(q[0] + 1) > _quaternion_resolution:
                qout[0] = np.log(-q[0])
                qout[1] = np.pi
                qout[2] = 0.0
                qout[3] = 0.0
            else:
                qout[0] = 0.0
                qout[1] = np.pi
                qout[2] = 0.0
                qout[3] = 0.0
        else:
            qout[0] = np.log(q[0])
            qout[1] = 0.0
            qout[2] = 0.0
            qout[3] = 0.0
    else:
        v = np.arctan2(b, q[0])
        f = v / b
        qout[0] = np.log(q[0] * q[0] + b * b) / 2.0
        qout[1] = f * q[1]
        qout[2] = f * q[2]
        qout[3] = f * q[3]
    qout *= s
    vnorm = np.sqrt(qout[1]**2 + qout[2]**2 + qout[3]**2)
    if vnorm > _quaternion_resolution:
        e = np.exp(qout[0])
        qout[0] = e * np.cos(vnorm)
        qout[1:] *= e * np.sin(vnorm) / vnorm
    else:
        qout[0] = np.exp(qout[0])
        qout[1] = 0.0
        qout[2] = 0.0
        qout[3] = 0.0


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->()')
def absolute(q, qout):
    """Return absolute value of quaternion |q|"""
    qout[0] = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def conj(q, qout):
    """Return quaternion-conjugate of quaternion q̄"""
    qout[0] = +q[0]
    qout[1] = -q[1]
    qout[2] = -q[2]
    qout[3] = -q[3]


conjugate = conj


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def exp(q, qout):
    """Return exponential of input quaternion exp(q)"""
    vnorm = np.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
    if vnorm > _quaternion_resolution:
        s = np.sin(vnorm) / vnorm
        e = np.exp(q[0])
        qout[0] = e * np.cos(vnorm)
        qout[1] = e * s * q[1]
        qout[2] = e * s * q[2]
        qout[3] = e * s * q[3]
    else:
        qout[0] = np.exp(q[0])
        qout[1] = 0.0
        qout[2] = 0.0
        qout[3] = 0.0


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def log(q, qout):
    """Return logarithm of input quaternion log(q)"""
    b = np.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
    if b <= _quaternion_resolution * np.abs(q[0]):
        if q[0] < 0.0:
            if np.abs(q[0] + 1) > _quaternion_resolution:
                qout[0] = np.log(-q[0])
                qout[1] = np.pi
                qout[2] = 0.0
                qout[3] = 0.0
            else:
                qout[0] = 0.0
                qout[1] = np.pi
                qout[2] = 0.0
                qout[3] = 0.0
        else:
            qout[0] = np.log(q[0])
            qout[1] = 0.0
            qout[2] = 0.0
            qout[3] = 0.0
    else:
        v = np.arctan2(b, q[0])
        f = v / b
        qout[0] = np.log(q[0] * q[0] + b * b) / 2.0
        qout[1] = f * q[1]
        qout[2] = f * q[2]
        qout[3] = f * q[3]


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->()')
def angle(q, qout):
    """Return angle (in radians) through which input quaternion rotates a vector

    If `q = np.exp(v̂ * θ/2)` for some unit vector `v̂` and an angle `θ` ∈[-2π,2π],
    then this function returns `abs(θ)`.  This equals 2*abs(log(q)), but is more
    efficient.

    """
    b = np.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
    if b <= _quaternion_resolution * np.abs(q[0]):
        if q[0] < 0.0:
            qout[0] = 2*np.pi
        else:
            qout[0] = 0.0
    else:
        qout[0] = 2*np.abs(np.arctan2(b, q[0]))


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def sqrt(q, qout):
    """Return square-root of input quaternion √q.

    The general formula whenever the denominator is nonzero is

    ```
        √q = (|q| + q) / √(2|q| + 2q.w)
    ```

    This can be proven by expanding `q` as `q.w + q.vec` and multiplying the
    expression above by itself.

    When the denominator is zero, the quaternion is a pure-real negative number.
    It is not clear what the appropriate square-root is in this case (because the
    quaternions come with infinitely many elements that square to -1), so we
    arbitrarily choose the result to be proportional to the `x` quaternion.

    """
    # √Q = (a + Q) / √(2*a + 2*Q[0])
    a = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if np.abs(a + q[0]) < _quaternion_resolution * a:
        qout[0] = 0.0
        qout[1] = np.sqrt(a)
        qout[2] = 0.0
        qout[3] = 0.0
    else:
        c = np.sqrt(0.5 / (a + q[0]))
        qout[0] = (a + q[0]) * c
        qout[1] = q[1] * c
        qout[2] = q[2] * c
        qout[3] = q[3] * c


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def square(q, qout):
    """Return square of quaternion q*q"""
    a = q[0]**2 - q[1]**2 - q[2]**2 - q[3]**2
    b = 2*q[0]*q[1]
    c = 2*q[0]*q[2]
    d = 2*q[0]*q[3]
    qout[0] = a
    qout[1] = b
    qout[2] = c
    qout[3] = d


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def reciprocal(q, qout):
    """Return reciprocal (inverse) of quaternion q.inverse"""
    norm = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    qout[0] = q[0] / norm
    qout[1] = -q[1] / norm
    qout[2] = -q[2] / norm
    qout[3] = -q[3] / norm


@attach_typelist_and_signature([(float64[:], float64[:])], '(n)->(n)')
def _ones_like(q, qout):
    """Return the identity quaternion

    Note that this is a helper function related to the ufunc
    `np.core.umath._ones_like`, which is different from the function
    `np.ones_like`.

    """
    qout[0] = 1
    qout[1:4] = 0


@attach_typelist_and_signature([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def bitwise_or(q1, q2, qout):
    """Return scalar product of quaternions q1|q2.

    If we denote by `⟨a⟩ₛ` the grade-s component of the general multivector `a`,
    we can express this product as

    ```
        a | b = Σₛ,ᵣ ⟨⟨a⟩ₛ ⟨b⟩ᵣ⟩₀
    ```

    Note that this is different from the "Hestenes dot" product where the sum
    runs over s≠0 and r≠0; that is the product returned by `galgebra` using
    this operator.

    """
    qout[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    qout[1] = 0.0
    qout[2] = 0.0
    qout[3] = 0.0


@attach_typelist_and_signature([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def bitwise_xor(q1, q2, qout):
    """Return outer product of quaternions q1^q2.

    This is the generalized outer product of geometric algebra.  If we denote
    by `⟨a⟩ₛ` the grade-s component of the general multivector `a`, we can
    express this product as

    ```
        a ^ b = Σₛ,ᵣ ⟨⟨a⟩ₛ ⟨b⟩ᵣ⟩ₛ₊ᵣ
    ```

    Note that the result may seem surprising because we sometimes think of quaternions as 

    """
    a = q1[0]*q2[0]
    b = q1[0]*q2[1] + q1[1]*q2[0]
    c = q1[0]*q2[2] + q1[2]*q2[0]
    d = q1[0]*q2[3] + q1[3]*q2[0]
    qout[0] = a
    qout[1] = b
    qout[2] = c
    qout[3] = d


invert = conj  # reversion (= conjugate for quaternion algebra)


@attach_typelist_and_signature([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def left_shift(q1, q2, qout):
    """Return left-contraction of quaternions q1<<q2 = q1⌋q1.

    For all quaternions `a`, `b`, `c`, we have

    ```
        (a ^ b) * c = a * (b ⌋ c)
    ```

    """
    a = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    b = q1[0]*q2[1]
    c = q1[0]*q2[2]
    d = q1[0]*q2[3]
    qout[0] = a
    qout[1] = b
    qout[2] = c
    qout[3] = d


@attach_typelist_and_signature([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def right_shift(q1, q2, qout):
    """Return right-contraction of quaternions q1>>q2 = q1⌊q2.

    For all quaternions `a`, `b`, `c`, we have

    ```
        c * (b ^ a) = (c ⌊ b) * a
    ```

    """
    a = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    b = q1[1]*q2[0]
    c = q1[2]*q2[0]
    d = q1[3]*q2[0]
    qout[0] = a
    qout[1] = b
    qout[2] = c
    qout[3] = d


@attach_typelist_and_signature([(float64[:], float64[:], boolean[:])], '(n),(n)->()')
def not_equal(q1, q2, bout):
    bout[0] = np.any(q1[:] != q2[:])


@attach_typelist_and_signature([(float64[:], float64[:], boolean[:])], '(n),(n)->()')
def equal(q1, q2, bout):
    bout[0] = np.all(q1[:] == q2[:])


@attach_typelist_and_signature([(float64[:], float64[:], boolean[:])], '(n),(n)->()')
def logical_and(q1, q2, bout):
    bout[0] = np.any(q1[:]) and np.any(q2[:])


@attach_typelist_and_signature([(float64[:], float64[:], boolean[:])], '(n),(n)->()')
def logical_or(q1, q2, bout):
    bout[0] = np.any(q1[:]) or np.any(q2[:])


@attach_typelist_and_signature([(float64[:], boolean[:])], '(n)->()')
def isfinite(qin, bout):
    bout[0] = np.isfinite(qin[0]) and np.isfinite(qin[1]) and np.isfinite(qin[2]) and np.isfinite(qin[3])


@attach_typelist_and_signature([(float64[:], boolean[:])], '(n)->()')
def isinf(qin, bout):
    bout[0] = np.isinf(qin[0]) or np.isinf(qin[1]) or np.isinf(qin[2]) or np.isinf(qin[3])


@attach_typelist_and_signature([(float64[:], boolean[:])], '(n)->()')
def isnan(qin, bout):
    bout[0] = np.isnan(qin[0]) or np.isnan(qin[1]) or np.isnan(qin[2]) or np.isnan(qin[3])
