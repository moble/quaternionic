[![Test Status](https://github.com/moble/quaternionic/workflows/tests/badge.svg)](https://github.com/moble/quaternionic/actions)
[![Test Coverage](https://codecov.io/gh/moble/quaternionic/branch/main/graph/badge.svg)](https://codecov.io/gh/moble/quaternionic)
[![Documentation Status](https://readthedocs.org/projects/quaternionic/badge/?version=latest)](https://quaternionic.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/quaternionic?color=)](https://pypi.org/project/quaternionic/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/quaternionic.svg?color=)](https://anaconda.org/conda-forge/quaternionic)
[![DOI](https://zenodo.org/badge/286745519.svg)](https://zenodo.org/badge/latestdoi/286745519)


# Quaternionic arrays

This module subclasses numpy's array type, interpreting the array as an array of quaternions, and
accelerating the algebra using numba.  This enables natural manipulations, like multiplying
quaternions as `a*b`, while also working with standard numpy functions, as in `np.log(q)`.  There is
also basic initial support for symbolic manipulation of quaternions by creating quaternionic arrays
with sympy symbols as elements, though this is a work in progress.

This package has evolved from the [quaternion](https://github.com/moble/quaternion) package, which
adds a quaternion dtype directly to numpy.  In some ways, that is a better approach because dtypes
are built in to numpy, making it more robust than this package.  However, that approach has its own
limitations, including that it is harder to maintain, and requires much of the code to be written in
C, which also makes it harder to distribute.  This package is written entirely in python code, but
should actually have comparable performance because it is compiled by numba.  Moreover, because the
core code is written in pure python, it is reusable for purposes other than the core purpose of this
package, which is to provide the numeric array type.


# Installation

Because this package is pure python code, installation is very simple.  In particular, with
a reasonably modern installation, you can just run

```bash
conda install -c conda-forge quaternionic
```

or

```bash
python -m pip install quaternionic
```

These will download and install the package.  (Using `python -m pip` instead of just `pip` or `pip3`
helps avoid problems that new python users frequently run into; the reason is explained by a veteran
python core contributor [here](https://snarky.ca/why-you-should-use-python-m-pip/).)

You can also install the package from source if you have `pip` version 10.0 or greater by running
`python -m pip install .` — or if you have `poetry` by running `poetry install` — from the top-level
directory.

Note that only python 3.8 or greater is supported.  (I have also tried to support PyPy3, although
I cannot test this as `scipy` does not currently install.  Pull requests are welcome.)  In any case,
I strongly recommend installing by way of an environment manager — especially
[conda](https://docs.anaconda.com/anaconda/install/), though other managers like `virtualenv` or
`pipenv` should also work.

For development work, the best current option is [poetry](https://python-poetry.org/).  From the
top-level directory, you can run `poetry run <some command>` to run the command in an isolated
environment.


# Usage

## Basic construction

The key function is `quaternionic.array`, which takes nearly the same arguments as `numpy.array`,
except that whatever array will result must have a final axis of size 4 (and the `dtype` must be
`float`).  As long as these conditions are satisfied, we can create new arrays or just reinterpret
existing arrays:

```python
import numpy as np
import quaternionic

a = np.random.normal(size=(17, 11, 4))  # Just some random numbers; last dimension is 4
q1 = quaternionic.array(a)  # Reinterpret an existing array
q2 = quaternionic.array([1.2, 2.3, 3.4, 4.5])  # Create a new array
```

In this example, `q1` is an array of 187 (17*11) quaternions, just to demonstrate that any number of
dimensions may be used, as long as the final dimension has size 4.

Here, the original array `a` will still exist just as it was, and will behave just as a normal numpy
array — including changing its values (which will change the values in `q1`), slicing, math, etc.
However, `q1` will be another
["view"](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html) into the same
data.  Operations on `q1` will be quaternionic.  For example, whereas `1/a` returns the element-wise
inverse of each float in the array, `1/q1` returns the *quaternionic* inverse of each quaternion.
Similarly, if you multiply two quaternionic arrays, their product will be computed with the usual
quaternion multiplication, rather than element-wise multiplication of floats as numpy usually
performs.

| :warning: WARNING                                                                                |
|:-------------------------------------------------------------------------------------------------|
| Because of an unfortunate choice by the numpy developers, the `np.copy` function will not preserve the quaternionic nature of an array by default; the result will just be a plain array of floats.  You could pass the optional argument `subok=True`, as in `q3 = np.copy(q1, subok=True)`, but it's easier to just use the member function: `q3 = q1.copy()`. |


## Algebra

All the usual quaternion operations are available, including

  * Addition `q1 + q2`
  * Subtraction `q1 - q2`
  * Multiplication `q1 * q2`
  * Division `q1 / q2`
  * Scalar multiplication `q1 * s == s * q1`
  * Scalar division `q1 / s` and `s / q1`
  * Reciprocal `np.reciprocal(q1) == 1/q1`
  * Exponential `np.exp(q1)`
  * Logarithm `np.log(q1)`
  * Square-root `np.sqrt(q1)`
  * Conjugate `np.conjugate(q1) == np.conj(q1)`

All numpy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html) that make sense for
quaternions are supported.  When the arrays have different shapes, the usual numpy
[broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) rules take effect.


## Attributes

In addition to the basic numpy array features, we also have a number of extra properties that are
particularly useful for quaternions, including

  * Methods to extract and/or set components
    * `w`, `x`, `y`, `z`
    * `i`, `j`, `k` (equivalent to `x`, `y`, `z`)
    * `scalar`, `vector` (equivalent to `w`, [`x`, `y`, `z`])
    * `real`, `imag` (equivalent to `scalar`, `vector`)
  * Methods related to norms
    * `abs` (square-root of sum of squares of components)
    * `norm` (sum of squares of components)
    * `modulus`, `magnitude` (equal to `abs`)
    * `absolute_square`, `abs2`, `mag2` (equal to `norm`)
    * `normalized`
    * `inverse`
  * Methods related to array infrastructure
    * `ndarray` (the numpy array underlying the quaternionic array)
    * `flattened` (all dimensions but last are flattened into one)
    * `iterator` (iterate over all quaternions)

Note that this package makes a distinction between `abs` and `norm` — the latter being equal to the
square of the former.  This version of the norm is also known as the "Cayley" norm, commonly used
when emphasizing the properties of an object in an algebra, as opposed to the "Euclidean" norm more
common when emphasizing the properties of an object in a vector space — though of course, algebras
are vector spaces with additional structure.  This choice agrees with the [Boost library's
implementation of
quaternions](https://www.boost.org/doc/libs/1_74_0/libs/math/doc/html/math_toolkit/value_op.html),
as well as this package's forerunner
[quaternion](https://github.com/moble/quaternion/blob/99913120b1b2a8a5eb7769c29ee760a236d40880/quaternion.h#L115-L120).
This also agrees with the corresponding functions on the [C++ standard library's complex
numbers](http://www.cplusplus.com/reference/complex/norm/).  Because this may be confusing, a number
of aliases are also provided that may be less confusing.  For example, some people find the pair
`abs` and `abs2` (meaning the square of `abs`) to be more sensible.


## Rotations

The most common application of quaternions is to representing rotations by means of unit
quaternions.  Note that this package does not *restrict* quaternions to have unit norms, since it is
usually better for numerical purposes not to do so.  For example, whereas rotation of a vector $v$
by a quaternion is usually implemented as $R\, v\, \bar{R}$, it is generally better to drop the
assumption that the quaternion has unit magnitude and implement rotation as $R\, v\, R^{-1}$.  This
is almost always more efficient, and more accurate.  That is what this package does by default
whenever rotations are involved.

Although this package does not restrict to unit quaternions, there are several converters to and
from other representations of rotations.  First, we have

   * `to_vector_part`, `from_vector_part`

These convert between the standard 3-d vector representation and their equivalent quaternions, which
allows them to be manipulated as vectors — as in `R * from_vector_part(v) * R.conjugate()`.  However,
note that you may not need to convert to/from quaternions.  For example, to rotate vectors `v` by
`R`, you can use
```python
R.rotate(v)
```
It may also be relevant to consider a vector as a "generator" of
rotations, in which case the actual rotation is obtained by applying `exp` to the generator.  This
*does* require conversion to a quaternionic array.  We also have converters that deal with standard
representations of rotations:

   * `to_rotation_matrix`, `from_rotation_matrix`
   * `to_transformation_matrix` (for non-unit quaternions)
   * `to_axis_angle`, `from_axis_angle`
   * `to_euler_angles`, `from_euler_angles` (though using Euler angles is almost always a bad idea)
   * `to_euler_phases`, `from_euler_phases` (see above)
   * `to_spherical_coordinates`, `from_spherical_coordinates`
   * `to_angular_velocity`, `from_angular_velocity`
   * `to_minimal_rotation`

Note that the last two items relate to quaternion-valued functions of time.  Converting to an angular
velocity requires differentiation, while converting from angular velocity requires integration (as
explored in [this paper](https://arxiv.org/abs/1604.08139)).  The 
["minimal rotation"](https://arxiv.org/abs/1110.2965) modifies an input rotation-function-of-time to
have the same effect on the `z` axis, while minimizing the amount of rotation that actually happens.

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

Also note that, because the unit quaternions form a "double cover" of the rotation group (meaning
that quaternions `q` and `-q` represent the same rotation), these functions are not perfect inverses
of each other.  In this case, for example, `q1` and `q2` may have opposite signs.  We can, however,
prove that these quaternions represent the same rotations by measuring the "distance" between the
quaternions as rotations:

```python
np.max(quaternionic.distance.rotation.intrinsic(q1, q2))  # Typically around 1e-15
```

Also note the classmethod

   * `random`

This constructs a quaternionic array in which each component is randomly selected from a normal
(Gaussian) distribution centered at 0 with scale 1, which means that the result is isotropic
(spherically symmetric).  It is also possible to pass the `normalize` argument to this function,
which results in truly random unit quaternions.


## Distance functions

The `quaternionic.distance` contains four distance functions:

  * `rotor.intrinsic`
  * `rotor.chordal`
  * `rotation.intrinsic`
  * `rotation.chordal`

The "rotor" distances do not account for possible differences in signs, meaning that rotor distances
can be large even when they represent identical rotations; the "rotation" functions just return the
smaller of the distance between `q1` and `q2` or the distance between `q1` and `-q2`.  So, for
example, either "rotation" distance between `q` and `-q` is always zero, whereas neither "rotor"
distance between `q` and `-q` will ever be zero (unless `q` is zero).  The "intrinsic" functions
measure the geodesic distance within the manifold of *unit* quaternions, and is somewhat slower but
may be more meaningful; the "chordal" functions measure the Euclidean distance in the (linear) space
of all quaternions, and is faster but its precise value is not necessarily as meaningful.

These functions satisfy some important conditions.  For each of these functions `d`, and for any
nonzero quaternions `q1` and `q2`, and *unit* quaternions `q3` and `q4`, we have

  * symmetry: `d(q1, q2) = d(q2, q1)`
  * invariance: `d(q3*q1, q3*q2) = d(q1, q2) = d(q1*q4, q2*q4)`
  * identity: `d(q1, q1) = 0`
  * positive-definiteness:
    * For rotor functions `d(q1, q2) > 0` whenever `q1 ≠ q2`
    * For rotation functions `d(q1, q2) > 0` whenever `q1 ≠ q2` and `q1 ≠ -q2`

Note that the rotation functions also satisfy both the usual identity property `d(q1, q1) = 0` and
the opposite-identity property `d(q1, -q1) = 0`.

See [Moakher (2002)](https://doi.org/10.1137/S0895479801383877) for a nice general discussion.


## Interpolation

Finally, there are also capabilities related to interpolation, for example as functions of time:

  * slerp (spherical linear interpolation)
  * squad (spherical quadratic interpolation)


## Caching

By default, the compiled code generated by numba is cached so that the compilation only needs to
take place on the first import. If you want to disable this caching, for example in a
high-performance computing environment where it may be preferable to compile the code than try to
load a cache from disk, set the environment variable `QUATERNIONIC_DISABLE_CACHE` to `1` before
importing this package.


# Related packages

Other python packages with some quaternion features include

  * [quaternion](https://github.com/moble/quaternion/) (core written in C; very fast; adds
    quaternion `dtype` to numpy; named
    [numpy-quaternion](https://pypi.org/project/numpy-quaternion/) on pypi due to name conflict)
  * [clifford](https://github.com/pygae/clifford) (very powerful; more general geometric algebras)
  * [rowan](https://github.com/glotzerlab/rowan) (many features; similar approach to this package;
    no acceleration or overloading)
  * [pyquaternion](http://kieranwynn.github.io/pyquaternion/) (many features; pure python; no
    acceleration or overloading)
  * [quaternions](https://github.com/mjsobrep/quaternions) (basic pure python package; no
    acceleration; specialized for rotations only)
  * [scipy.spatial.transform.Rotation.as\_quat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html)
    (quaternion output for `Rotation` object)
  * [mathutils](https://gitlab.com/ideasman42/blender-mathutils) (a Blender package with python
    bindings)
  * [Quaternion](https://pypi.org/project/Quaternion/) (extremely limited capabilities; unmaintained)

Also note that there is some capability to do symbolic manipulations of quaternions in these
packages:

  * [galgebra](https://github.com/pygae/galgebra) (more general geometric algebras; analogous to
    `clifford`, but for symbolic calculations)
  * [sympy.algebras.quaternion](https://docs.sympy.org/latest/modules/algebras.html)
