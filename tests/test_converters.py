import warnings
import math
import numpy as np
import quaternionic
import pytest


def test_to_rotation_matrix(Rs, eps, array):
    one, x, y, z = tuple(array(np.eye(4)))
    zero = 0.0 * one
    Rs = array(Rs.ndarray)

    def quat_mat(quat):
        return np.array([(quat * v * quat.inverse).vector for v in [x, y, z]]).T

    def quat_mat_vec(quats):
        mat_vec = np.array([(quats * v * np.reciprocal(quats)).vector for v in [x, y, z]])
        return np.transpose(mat_vec, tuple(range(mat_vec.ndim))[1:-1]+(-1, 0))

    with np.errstate(invalid='raise'):
        with pytest.raises((FloatingPointError, ZeroDivisionError)):
            zero.to_rotation_matrix

    for R in Rs:
        # Test correctly normalized rotors:
        assert np.allclose(quat_mat(R), R.to_rotation_matrix, atol=2*eps)
        # Test incorrectly normalized rotors:
        assert np.allclose(quat_mat(R), (1.1*R).to_rotation_matrix, atol=2*eps)

    Rs0 = Rs.copy()
    Rs0[Rs.shape[0]//2] = zero
    with np.errstate(invalid='raise'):
        with pytest.raises((FloatingPointError, ZeroDivisionError)):
            Rs0.to_rotation_matrix

    # Test correctly normalized rotors:
    assert np.allclose(quat_mat_vec(Rs), Rs.to_rotation_matrix, atol=2*eps)
    # Test incorrectly normalized rotors:
    assert np.allclose(quat_mat_vec(Rs), (1.1*Rs).to_rotation_matrix, atol=2*eps)

    # Simply test that this function succeeds and returns the right shape
    assert (Rs.reshape((2, 5, 10, 4))).to_rotation_matrix.shape == (2, 5, 10, 3, 3)


def test_from_rotation_matrix(Rs, eps):
    from scipy import linalg

    for nonorthogonal in [True, False]:
        if nonorthogonal:
            rot_mat_eps = 10*eps
        else:
            rot_mat_eps = 5*eps
        for i, R1 in enumerate(Rs):
            R2 = quaternionic.array.from_rotation_matrix(R1.to_rotation_matrix, nonorthogonal=nonorthogonal)
            d = quaternionic.distance.rotation.intrinsic(R1, R2)
            assert d < rot_mat_eps, (i, R1, R2, d)  # Can't use allclose here; we don't care about rotor sign

        Rs2 = quaternionic.array.from_rotation_matrix(Rs.to_rotation_matrix, nonorthogonal=nonorthogonal)
        for R1, R2 in zip(Rs, Rs2):
            d = quaternionic.distance.rotation.intrinsic(R1, R2)
            assert d < rot_mat_eps, (R1, R2, d)  # Can't use allclose here; we don't care about rotor sign

        Rs3 = Rs.reshape((2, 5, 10, 4))
        Rs4 = quaternionic.array.from_rotation_matrix(Rs3.to_rotation_matrix)
        for R3, R4 in zip(Rs3.flattened, Rs4.flattened):
            d = quaternionic.distance.rotation.intrinsic(R3, R4)
            assert d < rot_mat_eps, (R3, R4, d)  # Can't use allclose here; we don't care about rotor sign


def test_to_transformation_matrix(Rs, eps, array):
    one, x, y, z = tuple(array(np.eye(4)))
    zero = 0.0 * one
    Rs = array(Rs.ndarray)

    def quat_mat(quat):
        return np.array([(quat * v * np.conjugate(quat)).ndarray for v in [one, x, y, z]]).T

    def quat_mat_vec(quats):
        mat_vec = np.array([(quats * v * np.conjugate(quats)).ndarray for v in [one, x, y, z]])
        return np.transpose(mat_vec, tuple(range(mat_vec.ndim))[1:-1]+(-1, 0))

    # Test individual quaternions
    for R in Rs:
        # Test correctly normalized rotors:
        assert np.allclose(quat_mat(R), R.to_transformation_matrix, atol=2*eps)
        # Test incorrectly normalized rotors:
        for scale in [0.0, 0.123, 0.5, 1.1, 2.3]:
            assert np.allclose(scale**2*quat_mat(R), (scale*R).to_transformation_matrix, atol=2*eps)

    # Test vectorized quaternions
    # Test correctly normalized rotors:
    assert np.allclose(quat_mat_vec(Rs), Rs.to_transformation_matrix, atol=2*eps)
    # Test incorrectly normalized rotors:
    for scale in [0.0, 0.123, 0.5, 1.1, 2.3]:
        assert np.allclose(scale**2*quat_mat_vec(Rs), (scale*Rs).to_transformation_matrix, atol=2*eps)

    # Simply test that this function succeeds and returns the right shape
    assert (Rs.reshape((2, 5, 10, 4))).to_transformation_matrix.shape == (2, 5, 10, 4, 4)


def test_to_rotation_vector():
    np.random.seed(1234)
    n_tests = 1000
    vecs = np.random.uniform(high=math.pi/math.sqrt(3), size=n_tests*3).reshape((n_tests, 3))
    quats = np.zeros(vecs.shape[:-1]+(4,))
    quats[..., 1:] = vecs[...]
    quats = quaternionic.array(quats)
    quats = np.exp(quats/2)
    quat_vecs = quats.to_rotation_vector
    assert np.allclose(quat_vecs, vecs)


def test_from_rotation_vector():
    np.random.seed(1234)
    n_tests = 1000
    vecs = np.random.uniform(high=math.pi/math.sqrt(3), size=n_tests*3).reshape((n_tests, 3))
    quats = np.zeros(vecs.shape[:-1]+(4,))
    quats[..., 1:] = vecs[...]
    quats = quaternionic.array(quats)
    quats = np.exp(quats/2)
    quat_vecs = quats.to_rotation_vector
    quats2 = quaternionic.array.from_rotation_vector(quat_vecs)
    assert np.allclose(quats, quats2)


def test_from_spherical_coordinates():
    one, x, y, z = tuple(quaternionic.array(np.eye(4)))
    zero = 0.0 * one

    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for vartheta, varphi in random_angles:
        q = quaternionic.array.from_spherical_coordinates(vartheta, varphi)
        assert abs((np.exp(quaternionic.array(0, 0, 0, varphi / 2.)) * np.exp(quaternionic.array(0, 0, vartheta / 2., 0)))
                   - q) < 1.e-15
        xprime = q * x * q.inverse
        yprime = q * y * q.inverse
        zprime = q * z * q.inverse
        nhat = quaternionic.array(
            0.0,
            math.sin(vartheta)*math.cos(varphi),
            math.sin(vartheta)*math.sin(varphi),
            math.cos(vartheta)
        )
        thetahat = quaternionic.array(
            0.0,
            math.cos(vartheta)*math.cos(varphi),
            math.cos(vartheta)*math.sin(varphi),
            -math.sin(vartheta)
        )
        phihat = quaternionic.array(0.0, -math.sin(varphi), math.cos(varphi), 0.0)
        assert abs(xprime - thetahat) < 1.e-15
        assert abs(yprime - phihat) < 1.e-15
        assert abs(zprime - nhat) < 1.e-15
    assert np.max(np.abs(
        quaternionic.array.from_spherical_coordinates(random_angles)
        - quaternionic.array([quaternionic.array.from_spherical_coordinates(vartheta, varphi) for vartheta, varphi in random_angles])
    )) < 1.e-15


def test_to_spherical_coordinates(Rs):
    one, x, y, z = tuple(quaternionic.array(np.eye(4)))
    zero = 0.0 * one

    np.random.seed(1843)
    # First test on rotors that are precisely spherical-coordinate rotors
    random_angles = [[np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
                     for i in range(5000)]
    for vartheta, varphi in random_angles:
        vartheta2, varphi2 = (quaternionic.array.from_spherical_coordinates(vartheta, varphi)).to_spherical_coordinates
        varphi2 = (varphi2 + 2*np.pi) if varphi2 < 0 else varphi2
        assert abs(vartheta - vartheta2) < 1e-12, ((vartheta, varphi), (vartheta2, varphi2))
        assert abs(varphi - varphi2) < 1e-12, ((vartheta, varphi), (vartheta2, varphi2))
    # Now test that arbitrary rotors rotate z to the appropriate location
    for R in Rs:
        vartheta, varphi = R.to_spherical_coordinates
        R2 = quaternionic.array.from_spherical_coordinates(vartheta, varphi)
        assert (R*z*R.inverse - R2*z*R2.inverse).abs < 4e-15, (R, R2, (vartheta, varphi))


def test_from_euler_angles():
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for alpha, beta, gamma in random_angles:
        assert abs((np.exp(quaternionic.array(0, 0, 0, alpha / 2.))
                    * np.exp(quaternionic.array(0, 0, beta / 2., 0))
                    * np.exp(quaternionic.array(0, 0, 0, gamma / 2.))
                   )
                   - quaternionic.array.from_euler_angles(alpha, beta, gamma)) < 1.e-15
    assert np.max(np.abs(quaternionic.array.from_euler_angles(random_angles)
                         - quaternionic.array([quaternionic.array.from_euler_angles(alpha, beta, gamma)
                                               for alpha, beta, gamma in random_angles]))) < 1.e-15


def test_to_euler_angles(eps, array):
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for alpha, beta, gamma in random_angles:
        R1 = array.from_euler_angles(alpha, beta, gamma)
        R2 = array.from_euler_angles(*list(R1.to_euler_angles))
        d = quaternionic.distance.rotation.intrinsic(R1, R2)
        assert d < 6e3*eps, ((alpha, beta, gamma), R1, R2, d)  # Can't use allclose here; we don't care about rotor sign
    q0 = array(0, 0.6, 0.8, 0)
    assert q0.norm == 1.0
    assert abs(q0 - array.from_euler_angles(*list(q0.to_euler_angles))) < 1.e-15
