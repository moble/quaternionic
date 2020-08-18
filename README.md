[![Build Status](https://travis-ci.org/moble/quaternionic.svg?branch=master)](https://travis-ci.org/moble/quaternionic)
[![Documentation Status](https://readthedocs.org/projects/quaternionic/badge/?version=latest)](https://quaternionic.readthedocs.io/en/latest/?badge=latest)
[![Test Coverage](https://codecov.io/gh/moble/quaternionic/branch/master/graph/badge.svg)](https://codecov.io/gh/moble/quaternionic)


# Quaternions by way of numpy arrays

This module subclasses numpy's array type, interpreting the array as an array of quaternions, and
accelerating the algebra using numba.  There is also basic initial support for symbolic manipulation
of quaternions by creating quaternionic arrays with sympy symbols as elements, though this is a work
in progress.

This package has evolved from the [quaternion](https://github.com/moble/quaternion) package, which
adds a quaternion dtype directly to numpy.  In many ways, that is a much better approach because
dtypes are built in to numpy, making it more robust than this package.  However, that approach has
its own limitations, including that it is harder to maintain, and requires much of the code to be
written in C, which also makes it harder to distribute.  This package is written entirely in python
code, but should actually have comparable performance because it is compiled by numba.  Moreover,
because the core code is written in pure python, it is reusable for purposes other than the core
purpose of this package, which is to provide the numeric array type.


# Installation

Because this package is pure python code, it can be installed with the simplest tools.  In
particular, you can just run

```bash
pip install quaternionic
```

For development work, the best current option is [poetry](https://python-poetry.org/).  From the
top-level directory, run `poetry install` or just `poetry run <some command>`.

# Usage

Any numpy array `a` with a last axis of size 4 (and float dtype) can be reinterpreted as a
quaternionic array with `quaternionic.array(a)`:

```python
import numpy as np
import quaternionic

a = 1.0 - np.random.rand(17, 3, 4)  # Just some random numbers; last dimension is 4
q1 = quaternionic.array(a)  # Reinterpret an existing array
q2 = quaternionic.array([1.2, 2.3, 3.4, 4.5])  # Create a new array
```

Here, the original array `a` will still exist just as it was, and will behave just as a normal numpy
array — including changing its values (which will change the values in `q1`), slicing, math, etc.
However, `q1` will be another
["view"](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html) into the same
data.  Operations on `q1` will be quaternionic.  For example, whereas `1/a` returns the element-wise
inverse of each float in the array, `1/q1` returns the *quaternionic* inverse of each quaternion.
Similarly, if you multiply two quaternionic arrays, their product will be computed with the usual
quaternion multiplication, rather than element-wise multiplication of floats as numpy usually
performs.

All the usual quaternion operations are available, including

  * Addition `q1 + q2`
  * Subtraction `q1 - q2`
  * Multiplication `q1 * q2`
  * Division `q1 / q2`
  * Scalar multiplication `q1 * s == s * q1`
  * Scalar division `q1 / s != s / q1`
  * Exponential `np.exp(q1)`
  * Logarithm `np.log(q1)`
  * Square-root `np.sqrt(q1)`

All numpy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html) that make sense for
quaternions are supported.  When the arrays have different shapes, the usual numpy
[broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) rules take effect.

In addition to the basic numpy array features, we also have a number of extra properties that are
particularly useful for quaternions, including

  * Methods to extract and/or set components
    * w, x, y, z
    * i, j, k (equivalent to x, y, z)
    * scalar, vector (equivalent to w, [x, y, z])
    * real, imag (equivalent to scalar, vector)
  * Methods related to norms
    * abs (square-root of sum of squares of components)
    * norm (sum of squares of components)
    * modulus, magnitude (equal to abs)
    * absolute\_square, abs2, mag2, squared_norm (equal to norm)
    * normalized
    * conjugate, conj
    * inverse
  * Methods related to array infrastructure
    * ndarray (the numpy array underlying the quaternionic array)
    * flattened (all dimensions but last are flattened into one)
    * iterator (iterate over all quaternions)
    
Note that this package makes a distinction between `abs` and `norm` — the latter being the square of
the former.  This choice agrees with the [Boost library's implementation of
quaternions](https://www.boost.org/doc/libs/1_74_0/libs/math/doc/html/math_toolkit/value_op.html),
as well as this package's forerunner
[quaternion](https://github.com/moble/quaternion/blob/99913120b1b2a8a5eb7769c29ee760a236d40880/quaternion.h#L115-L120).
This also agrees with the corresponding functions on the [C++ standard library's complex
numbers](http://www.cplusplus.com/reference/complex/norm/).  Because this may be confusing, a number
of aliases are also provided that may be less confusing.  For example, some people find the pair
`abs` and `abs2` to be more sensible.

This package does not specialize to *unit* quaternions, since it is usually better for numerical
purposes not to do so.  For example, whereas rotation of a vector `v` by a quaternion is usually
implemented as `R * v * np.conjugate(R)`, it is usually better to drop the assumption that the
quaternion has unit magnitude and implement rotation as `R * v * np.reciprocal(R)`.  That is what
this package does by default whenever rotations are involved.

Although this package does not specialize to unit quaternions, there are several converters to and
from other representations of rotations, including

   * to/from\_rotation\_matrix
   * to\_transformation\_matrix (for non-unit quaternions)
   * to/from\_axis\_angle representation
   * to/from\_euler\_angles (though using Euler angles is almost always a bad idea)
   * to/from\_spherical\_coordinates
   * to/from\_angular\_velocity

Note that the last item relates to quaternion-valued functions of time.  Converting to an angular
velocity requires differentiation, while converting from angular velocity requires integration (as
explored in [this paper](https://arxiv.org/abs/1604.08139)).

For these converters, the "to" functions are properties on the individual arrays, whereas the "from"
functions are "classmethod"s that take the corresponding objects as inputs.  For example, we could
write

```python
q1 = quaternionic.array(np.random.rand(100, 4)).normalized
m = q1.to_rotation_matrix
```

to obtain the matrix `m` *from* a quaternionic array `q1`.  (Here, `m` is actually a series of 100
3x3 matrices corresponding to the 100 quaternions in `q1`.)  On the other hand, to obtain a
quaternionic array from some matrix `m`, we would write

```python
q2 = quaternionic.array.from_rotation_matrix(m)
```

Also note that, because the quaternions form a "double cover" of the rotation group (meaning that
quaternions `R` and `-R` represent the same rotation), these functions are not perfect inverses of
each other.  In this case, for example, `q1` and `q2` may have opposite signs.  We can, however,
prove that these quaternions represent the same rotations by measuring the "distance" between the
quaternions as rotations:

```python
np.max(quaternionic.distance.rotation.intrinsic(q1, q2))  # Typically around 1e-15
```

There are four types of distance measures:

  * `quaternionic.distance.rotor.intrinsic`
  * `quaternionic.distance.rotor.chordal`
  * `quaternionic.distance.rotation.intrinsic`
  * `quaternionic.distance.rotation.chordal`

The "rotor" distances do not account for possible differences in signs, meaning that rotor distances
can be large even when they represent identical rotations; the "rotation" functions just return the
smaller of the distance between `q1` and `q2` or the distance between `q1` and `-q2`.  The
"intrinsic" functions measure the geodesic distance within the manifold of *unit* quaternions, and
is somewhat slower but may be more meaningful; the "chordal" functions measure the Euclidean
distance in the (linear) space of all quaternions, and is faster but its precise value is not
necessarily as meaningful.

Finally, there are also capabilities related to interpolation

  * slerp (spherical linear interpolation)
  * squad (spherical quadratic interpolation)



# Related packages

Packages with some quaternion features available on pypi include

  * numpy-quaternion (a.k.a. quaternion, but renamed on pypi)
  * clifford (very powerful; more general geometric algebras)
  * pyquaternion (many features; pure python; no acceleration)
  * quaternions (basic pure python package; no acceleration; specialized for rotations only)
  * rowan (similar approach to this package, but no acceleration or overloading)
  * Quaternion (minimal capabilities)
  * scipy.spatial.transform.Rotation.as_quat (quaternion output for Rotation object)
  * mathutils (a Blender package with python bindings)

Also note that there is some capability to do symbolic manipulations of quaternions in these packages:

  * galgebra
  * sympy.algebras.quaternion

