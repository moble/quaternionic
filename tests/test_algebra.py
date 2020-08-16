import warnings
import numpy as np
import quaternionic
import pytest


algebra_pyufuncs = type('AlgebraPyufuncs', (object,), dict())()
quaternionic.utilities.pyguvectorize_module_functions(quaternionic.algebra, algebra_pyufuncs)


def test_basis_multiplication():
    # Basis components
    one, i, j, k = tuple(quaternionic.array(np.eye(4)))

    # Full multiplication table
    assert one * one == one
    assert one * i == i
    assert one * j == j
    assert one * k == k
    assert i * one == i
    assert i * i == np.negative(one)
    assert i * j == k
    assert i * k == -j
    assert j * one == j
    assert j * i == -k
    assert j * j == -one
    assert j * k == i
    assert k * one == k
    assert k * i == j
    assert k * j == -i
    assert k * k == -one

    # Standard expressions
    assert one*one == one
    assert i*i == -one
    assert j*j == -one
    assert k*k == -one
    assert i*j*k == -one
