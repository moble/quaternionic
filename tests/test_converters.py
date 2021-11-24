import warnings
import math
import numpy as np
import quaternionic
import pytest


def test_to_scalar_part(Rs):
    assert np.array_equal(Rs.to_scalar_part, Rs.ndarray[..., 0])


def test_from_scalar_part():
    scalars = np.random.rand(17, 8, 3)
    q = quaternionic.array.from_scalar_part(scalars)
    assert q.shape[-1] == 4
    assert q.shape[:-1] == scalars.shape
    assert np.array_equal(q.to_scalar_part, scalars)


def test_to_vector_part(Rs):
    assert np.array_equal(Rs.to_vector_part, Rs.ndarray[..., 1:])


def test_from_vector_part():
    vec = np.random.rand(17, 8, 5, 3)
    q = quaternionic.array.from_vector_part(vec)
    assert q.shape[-1] == 4
    assert q.shape[:-1] == vec.shape[:-1]
    assert np.array_equal(q.to_vector_part, vec)


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

    assert (
        quaternionic.array.from_axis_angle([1, 2, 3])
        ==
        quaternionic.array.from_axis_angle([1.0, 2.0, 3.0])
    )


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

    # https://github.com/moble/quaternionic/issues/29
    assert (
        quaternionic.array.from_spherical_coordinates(1, 2)
        ==
        quaternionic.array.from_spherical_coordinates(1.0, 2.0)
    )


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

    assert (
        quaternionic.array.from_euler_angles(1, 2, 3)
        ==
        quaternionic.array.from_euler_angles(1.0, 2.0, 3.0)
    )


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


def test_from_euler_phases(eps, array):
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, 2 * np.pi),
                      np.random.uniform(0, np.pi),
                      np.random.uniform(-np.pi, 2 * np.pi)]
                     for i in range(5000)]
    for alpha, beta, gamma in random_angles:
        R1 = array.from_euler_angles(alpha, beta, gamma)
        R2 = array.from_euler_phases([np.exp(1j * alpha), np.exp(1j * beta), np.exp(1j * gamma)])
        d = quaternionic.distance.rotation.intrinsic(R1, R2)
        assert d < 8*eps, ((alpha, beta, gamma), R1, R2, d)  # Can't use allclose here; we don't care about rotor sign


def test_to_euler_phases(eps, array):
    np.random.seed(1843)
    random_angles = [
        [np.random.uniform(-np.pi, 2 * np.pi),
         np.random.uniform(0, np.pi),
         np.random.uniform(-np.pi, 2 * np.pi)]
        for _ in range(5000)
    ]
    for alpha, beta, gamma in random_angles:
        z1 = array.from_euler_angles(alpha, beta, gamma).to_euler_phases
        z2 = np.array([np.exp(1j * alpha), np.exp(1j * beta), np.exp(1j * gamma)])
        assert abs(z1[0] - z2[0]) < 5*eps, (alpha, beta, gamma, z1, z2)
        assert abs(z1[1] - z2[1]) < 5*eps, (alpha, beta, gamma, z1, z2)
        assert abs(z1[2] - z2[2]) < 5*eps, (alpha, beta, gamma, z1, z2)
    random_angles = tuple(
        [np.random.uniform(-np.pi, 2 * np.pi),
         0.0,
         np.random.uniform(-np.pi, 2 * np.pi)]
        for _ in range(50)
    )
    for alpha, beta, gamma in random_angles:
        R1 = array.from_euler_angles(alpha, beta, gamma)
        R1.x = 0.0
        R1.y = 0.0
        z1 = R1.to_euler_phases
        z2 = np.array([np.exp(1j * alpha), np.exp(1j * beta), np.exp(1j * gamma)])
        assert abs(z1[1] - 1) < 5*eps, (alpha, beta, gamma, z1, z2)
        assert abs(z1[1] - z2[1]) < 5*eps, (alpha, beta, gamma, z1, z2)
        assert abs(z1[0]*z1[2] - z2[0]*z2[2]) < 5*eps, (alpha, beta, gamma, z1, z2)
    random_angles = tuple(
        [np.random.uniform(-np.pi, 2 * np.pi),
         np.pi,
         np.random.uniform(-np.pi, 2 * np.pi)]
        for _ in range(50)
    )
    for alpha, beta, gamma in random_angles:
        R1 = array.from_euler_angles(alpha, beta, gamma)
        R1.w = 0.0
        R1.z = 0.0
        z1 = R1.to_euler_phases
        z2 = np.array([np.exp(1j * alpha), np.exp(1j * beta), np.exp(1j * gamma)])
        assert abs(z1[1] - -1) < 5*eps, (alpha, beta, gamma, z1, z2)
        assert abs(z1[1] - z2[1]) < 5*eps, (alpha, beta, gamma, z1, z2)
        assert abs(z1[0]*z1[2].conjugate() - z2[0]*z2[2].conjugate()) < 5*eps, (alpha, beta, gamma, z1, z2)


def test_to_angular_velocity():
    import math
    import numpy as np
    import quaternionic

    t0 = 0.0
    t2 = 10_000.0
    Omega_orb = 2 * math.pi * 100 / t2
    Omega_prec = 2 * math.pi * 10 / t2
    alpha = 0.125 * math.pi
    alphadot = 2 * alpha / t2
    nu = 0.2 * alpha
    Omega_nu = Omega_prec
    R0 = np.exp(-1.1 * alpha * quaternionic.x / 2)

    def R(t):
        return (R0
                * np.exp(Omega_prec * t * quaternionic.z / 2) * np.exp((alpha + alphadot * t) * quaternionic.x / 2)
                * np.exp(-Omega_prec * t * quaternionic.z / 2)
                * np.exp(Omega_orb * t * quaternionic.z / 2)
                * np.exp(nu * np.cos(Omega_nu * t) * quaternionic.y / 2))

    def Rdot(t):
        R_dynamic = R0.inverse * R(t)
        R_prec = np.exp(Omega_prec * t * quaternionic.z / 2)
        R_nu = np.exp(nu * np.cos(Omega_nu * t) * quaternionic.y / 2)
        return R0 * (0.5 * Omega_prec * quaternionic.z * R_dynamic
                     + 0.5 * alphadot * R_prec * quaternionic.x * R_prec.conj() * R_dynamic
                     + 0.5 * (Omega_orb - Omega_prec) * R_dynamic * R_nu.inverse * quaternionic.z * R_nu
                     + 0.5 * (-Omega_nu * nu * np.sin(Omega_nu * t)) * R_dynamic * quaternionic.y)

    def Omega_tot(_, t):
        Rotor = R(t)
        RotorDot = Rdot(t)
        return (2 * RotorDot * Rotor.inverse).vector

    t = np.linspace(t0, t2/100, num=10_000)
    R_approx = R(t).to_angular_velocity(t, t_new=None, axis=0)
    R_exact = Omega_tot(None, t)
    assert np.max(np.linalg.norm(R_approx - R_exact, axis=1)) < 5e-13

    t = np.linspace(t0, t2/100, num=10_000)
    t_new = np.linspace(t0, t2/100, num=103)
    R_approx = R(t).to_angular_velocity(t, t_new=t_new, axis=0)
    R_exact = Omega_tot(None, t_new)
    assert np.max(np.linalg.norm(R_approx - R_exact, axis=1)) < 5e-13


def test_from_angular_velocity():
    import math
    import numpy as np
    import quaternionic

    t0 = 0.0
    t2 = 10_000.0
    Omega_orb = 2 * math.pi * 100 / t2
    Omega_prec = 2 * math.pi * 10 / t2
    alpha = 0.125 * math.pi
    alphadot = 2 * alpha / t2
    nu = 0.2 * alpha
    Omega_nu = Omega_prec
    R0 = np.exp(-1.1 * alpha * quaternionic.x / 2)

    def R(t):
        return (R0
                * np.exp(Omega_prec * t * quaternionic.z / 2) * np.exp((alpha + alphadot * t) * quaternionic.x / 2)
                * np.exp(-Omega_prec * t * quaternionic.z / 2)
                * np.exp(Omega_orb * t * quaternionic.z / 2)
                * np.exp(nu * np.cos(Omega_nu * t) * quaternionic.y / 2))

    def Rdot(t):
        R_dynamic = R0.inverse * R(t)
        R_prec = np.exp(Omega_prec * t * quaternionic.z / 2)
        R_nu = np.exp(nu * np.cos(Omega_nu * t) * quaternionic.y / 2)
        return R0 * (0.5 * Omega_prec * quaternionic.z * R_dynamic
                     + 0.5 * alphadot * R_prec * quaternionic.x * R_prec.conj() * R_dynamic
                     + 0.5 * (Omega_orb - Omega_prec) * R_dynamic * R_nu.inverse * quaternionic.z * R_nu
                     + 0.5 * (-Omega_nu * nu * np.sin(Omega_nu * t)) * R_dynamic * quaternionic.y)

    def Omega_tot(_, t):
        Rotor = R(t)
        RotorDot = Rdot(t)
        return (2 * RotorDot * Rotor.inverse).vector

    t = np.linspace(t0, t2/10, num=1_000)

    # Test raisers
    with pytest.raises(ValueError):
        R_approx = quaternionic.array.from_angular_velocity([1+2j, 3+4j], t, R0=R(t0), tolerance=1e-6)
    with pytest.raises(ValueError):
        R_approx = quaternionic.array.from_angular_velocity(np.random.rand(17, 2), t, R0=R(t0), tolerance=1e-6)

    # Test with exact Omega function
    R_approx = quaternionic.array.from_angular_velocity(Omega_tot, t, R0=R(t0), tolerance=1e-6)
    R_exact = R(t)
    # phi_Delta = np.array([quaternionic.distance.rotation.intrinsic(e, a) for e, a in zip(R_exact, R_approx)])
    phi_Delta = quaternionic.distance.rotation.intrinsic(R_exact, R_approx)
    assert np.max(phi_Delta) < 1e-4, np.max(phi_Delta)

    # Test with exact Omega function
    R_approx = quaternionic.array.from_angular_velocity(Omega_tot, t, R0=None, tolerance=1e-6)
    R_exact = R(t) * R(t0).inverse
    # phi_Delta = np.array([quaternionic.distance.rotation.intrinsic(e, a) for e, a in zip(R_exact, R_approx)])
    phi_Delta = quaternionic.distance.rotation.intrinsic(R_exact, R_approx)
    assert np.max(phi_Delta) < 1e-4, np.max(phi_Delta)

    # Test with explicit values, given at the moments output above
    v = np.array([Omega_tot(None, ti) for ti in t])
    R_approx = quaternionic.array.from_angular_velocity(v, t, R0=R(t0), tolerance=1e-6)
    R_exact = R(t)
    phi_Delta = quaternionic.distance.rotation.intrinsic(R_exact, R_approx)
    assert np.max(phi_Delta) < 1e-4, np.max(phi_Delta)


def test_to_minimal_rotation():
    import math
    import numpy as np
    import quaternionic

    t = np.linspace(0.0, 100.0, num=1_000)
    ω = (5 * 2 * np.pi) / (t[-1] - t[0])

    # Test basic removal of rotation about z
    q = np.exp((ω * t / 2) * quaternionic.z)
    q_minimal_rotation = q.to_minimal_rotation(t, t_new=None, axis=0, iterations=2)
    qa = q * quaternionic.z * q.inverse
    qb = q_minimal_rotation * quaternionic.z * q_minimal_rotation.inverse
    assert np.max((qa - qb).norm) < 1e-16
    assert np.max((q_minimal_rotation - quaternionic.one).norm) < 1e-16

    # Test same with t_new
    t_new = np.linspace(0.0, 100.0, num=1_005)
    ω = (5 * 2 * np.pi) / (t[-1] - t[0])
    q = np.exp((ω * t / 2) * quaternionic.z)
    q_new = np.exp((ω * t_new / 2) * quaternionic.z)
    q_minimal_rotation = q.to_minimal_rotation(t, t_new=t_new, axis=0, iterations=2)
    qa = q_new * quaternionic.z * q_new.inverse
    qb = q_minimal_rotation * quaternionic.z * q_minimal_rotation.inverse
    assert t_new.shape[0] == q_minimal_rotation.shape[0]
    assert np.max((qa - qb).norm) < 1e-16
    assert np.max((q_minimal_rotation - quaternionic.one).norm) < 1e-16

    # Test rotation onto uniform rotation in x-y plane
    q = quaternionic.array(
        np.stack(
            (
                np.ones(t.size),
                np.cos(ω*t),
                np.sin(ω*t),
                np.zeros(t.size)
            ),
            axis=1
        )
        / np.sqrt(2)
    )
    q_minimal_rotation = q.to_minimal_rotation(t)
    qa = q * quaternionic.z * q.inverse
    qb = q_minimal_rotation * quaternionic.z * q_minimal_rotation.inverse
    assert np.max((qa - qb).norm) < 1e-16
    assert np.max(abs(ω - np.linalg.norm(q_minimal_rotation.to_angular_velocity(t), axis=1))) < 1e-8
    assert np.max(abs(q_minimal_rotation.to_angular_velocity(t)[:, :2])) < 1e-8


def test_random():
    q = quaternionic.array.random()
    assert isinstance(q, quaternionic.array)
    assert q.dtype == np.float64
    assert q.shape == (4,)

    q = quaternionic.array.random(tuple())
    assert isinstance(q, quaternionic.array)
    assert q.dtype == np.float64
    assert q.shape == (4,)

    q = quaternionic.array.random(17)
    assert isinstance(q, quaternionic.array)
    assert q.dtype == np.float64
    assert q.shape == (17, 4)

    q = quaternionic.array.random((17, 3))
    assert isinstance(q, quaternionic.array)
    assert q.dtype == np.float64
    assert q.shape == (17, 3, 4)

    q = quaternionic.array.random((17, 3, 4))
    assert isinstance(q, quaternionic.array)
    assert q.dtype == np.float64
    assert q.shape == (17, 3, 4)

    q = quaternionic.array.random((17, 3, 4), normalize=True)
    assert np.max(np.abs(1 - q.abs)) < 4 * np.finfo(float).eps
