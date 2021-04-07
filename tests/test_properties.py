import warnings
import numpy as np
import quaternionic
import pytest


def test_getting_components():
    q = quaternionic.array([1, 2, 3, 4])  # Note the integer input
    assert q.w == 1.0
    assert q.x == 2.0
    assert q.y == 3.0
    assert q.z == 4.0

    assert q.scalar == 1.0
    assert np.array_equal(q.vector, [2.0, 3.0, 4.0])

    assert q.i == 2.0
    assert q.j == 3.0
    assert q.k == 4.0

    assert q.real == 1.0
    assert np.array_equal(q.imag, [2.0, 3.0, 4.0])


def test_setting_components():
    q = quaternionic.array([1, 2, 3, 4])  # Note the integer input
    q.w = 5
    q.x = 6
    q.y = 7
    q.z = 8
    assert np.array_equal(q.ndarray, [5.0, 6.0, 7.0, 8.0])

    q.scalar = 1
    q.vector = [2, 3, 4]
    assert np.array_equal(q.ndarray, [1.0, 2.0, 3.0, 4.0])

    q.w = 5
    q.i = 6
    q.j = 7
    q.k = 8
    assert np.array_equal(q.ndarray, [5.0, 6.0, 7.0, 8.0])

    q.real = 1
    q.imag = [2, 3, 4]
    assert np.array_equal(q.ndarray, [1.0, 2.0, 3.0, 4.0])


def test_two_spinor():
    np.random.seed(1234)
    q = quaternionic.array.random((17, 9, 4))
    s = q.two_spinor
    a = q.w + 1j * q.z
    b = q.y + 1j * q.x
    assert np.array_equal(s.a, a)
    assert np.array_equal(s.b, b)


def test_iterator():
    a = np.arange(17*3*4).reshape((17, 3, 4))
    q = quaternionic.array(a)
    for i, qi in enumerate(q.iterator):
        assert np.array_equal(qi, np.arange(4)+4.0*i)


def test_rotate_vectors(Rs):
    one, x, y, z = tuple(quaternionic.array(np.eye(4)))
    zero = 0.0 * one

    with pytest.raises(ValueError):
        one.rotate(np.array(3.14))
    with pytest.raises(ValueError):
        one.rotate(np.random.normal(size=(17, 9, 4)))
    with pytest.raises(ValueError):
        one.rotate(np.random.normal(size=(17, 9, 3)), axis=1)

    np.random.seed(1234)
    # Test (1)*(1)
    vecs = np.random.normal(size=(3,))
    quats = z
    vecsprime = quats.rotate(vecs)
    assert np.allclose(vecsprime,
                       (quats * quaternionic.array(0, *vecs) * quats.inverse).vector,
                       rtol=0.0, atol=0.0)
    assert quats.shape[:-1] + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (1)*(5)
    vecs = np.random.normal(size=(5, 3))
    quats = z
    vecsprime = quats.rotate(vecs)
    for i, vec in enumerate(vecs):
        assert np.allclose(vecsprime[i],
                           (quats * quaternionic.array(0, *vec) * quats.inverse).vector,
                           rtol=0.0, atol=0.0)
    assert quats.shape[:-1] + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (1)*(5) inner axis
    vecs = np.random.normal(size=(3, 5))
    quats = z
    vecsprime = quats.rotate(vecs, axis=-2)
    for i, vec in enumerate(vecs.T):
        assert np.allclose(vecsprime[:, i],
                           (quats * quaternionic.array(0, *vec) * quats.inverse).vector,
                           rtol=0.0, atol=0.0)
    assert quats.shape[:-1] + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(1)
    vecs = np.random.normal(size=(3))
    quats = Rs
    vecsprime = quats.rotate(vecs)
    assert np.allclose(vecsprime,
                       [vprime.vector for vprime in quats * quaternionic.array(0, *vecs) * ~quats],
                       rtol=1e-15, atol=1e-15)
    assert quats.shape[:-1] + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(5)
    vecs = np.random.normal(size=(5, 3))
    quats = Rs
    vecsprime = quats.rotate(vecs)
    for i, vec in enumerate(vecs):
        assert np.allclose(vecsprime[:, i],
                           [vprime.vector for vprime in quats * quaternionic.array(0, *vec) * ~quats],
                           rtol=1e-15, atol=1e-15)
    assert quats.shape[:-1] + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(5) inner axis
    vecs = np.random.normal(size=(3, 5))
    quats = Rs
    vecsprime = quats.rotate(vecs, axis=-2)
    for i, vec in enumerate(vecs.T):
        assert np.allclose(vecsprime[:, :, i],
                           [vprime.vector for vprime in quats * quaternionic.array(0, *vec) * ~quats],
                           rtol=1e-15, atol=1e-15)
    assert quats.shape[:-1] + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
