# Quaternions by way of numpy arrays

This module subclasses numpy's array type, interpreting the array as an array of quaternions, and
accelerating the algebra using numba.

This package has evolved from the [quaternion](https://github.com/moble/quaternion) package, which
adds a quaternion dtype directly to numpy.  In many ways, that is a much better approach because
dtypes are built in to numpy, making it more robust than this package.  However, that approach has
its own limitations, including that it is harder to maintain, and requires much of the code to be
written in C, which also makes it harder to distribute.  This package is written entirely in python
code, but should actually have comparable performance.




# Similar packages

Packages with similar features available on pypi include
  * numpy-quaternion
  * Quaternion
  * quaternions
  * pyquaternion
  * rowan
  * clifford
  * scipy.spatial.transform.Rotation.as_quat
  * mathutils (a Blender package with python bindings)

Also note that there is some capability to do symbolic manipulations of quaternions in these packages:
  * galgebra
  * sympy.algebras.quaternion

