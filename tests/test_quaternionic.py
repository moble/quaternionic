import warnings
import numpy as np
import quaternionic
from numpy import *
import pytest


from sys import platform
on_windows = ('win' in platform.lower() and not 'darwin' in platform.lower())


eps = np.finfo(float).eps



def test_version():
    from quaternionic import __version__
    assert __version__ == '0.0.1a'


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



# The following fixtures are used to establish some re-usable data
# for the tests; they need to be re-constructed because some of the
# tests will change the values, but we want the values to be constant
# on every entry into a test.

@pytest.fixture
def Qs():
    return quaternion_sampler()

def quaternion_sampler():
    return quaternionic.array([
        [np.nan, 0., 0., 0.],
        [np.inf, 0., 0., 0.],
        [-np.inf, 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1.1, 2.2, 3.3, 4.4],
        [-1.1, -2.2, -3.3, -4.4],
        [1.1, -2.2, -3.3, -4.4],
        [
            0.18257418583505537115232326093360,
            0.36514837167011074230464652186720,
            0.54772255750516611345696978280080,
            0.73029674334022148460929304373440
        ],
        [1.7959088706354, 0.515190292664085, 0.772785438996128, 1.03038058532817],
        [2.81211398529184, -0.392521193481878, -0.588781790222817, -0.785042386963756],
    ])

Qs_array = quaternion_sampler()

q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, Q, Qneg, Qbar, Qnormalized, Qlog, Qexp, = range(len(Qs_array))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    Qs_zero = np.arange(len(Qs_array))[Qs_array == Qs_array[q_0]]
    Qs_nonzero = np.arange(len(Qs_array))[np.nonzero(Qs_array)]
    Qs_nan = np.arange(len(Qs_array))[np.isnan(Qs_array)]
    Qs_nonnan = np.arange(len(Qs_array))[~np.isnan(Qs_array)]
    Qs_nonnannonzero = np.arange(len(Qs_array))[~np.isnan(Qs_array) & (Qs_array != Qs_array[q_0])]
    Qs_inf = np.arange(len(Qs_array))[np.isinf(Qs_array)]
    Qs_noninf = np.arange(len(Qs_array))[~np.isinf(Qs_array)]
    Qs_noninfnonzero = np.arange(len(Qs_array))[~np.isinf(Qs_array) & (Qs_array != Qs_array[q_0])]
    Qs_finite = np.arange(len(Qs_array))[np.isfinite(Qs_array)]
    Qs_nonfinite = np.arange(len(Qs_array))[~np.isfinite(Qs_array)]
    Qs_finitenonzero = np.arange(len(Qs_array))[np.isfinite(Qs_array) & (Qs_array != Qs_array[q_0])]


@pytest.fixture
def Rs():
    np.random.seed(1842)
    ones = [0, -1., 1.]
    rs = [[w, x, y, z] for w in ones for x in ones for y in ones for z in ones][1:]
    rs = rs + [r for r in [quaternionic.array(np.random.uniform(-1, 1, size=4)) for _ in range(20)]]
    return quaternionic.array(rs).normalized




# Unary bool returners
# def test_quaternion_nonzero(Qs):
#     assert not Qs[q_0].nonzero()  # Do this one explicitly, to not use circular logic
#     assert Qs[q_1].nonzero()  # Do this one explicitly, to not use circular logic
#     for q in Qs[Qs_zero]:
#         assert not q.nonzero()
#     for q in Qs[Qs_nonzero]:
#         assert q.nonzero()


def test_quaternion_isnan(Qs):
    assert not np.isnan(Qs[q_0])  # Do this one explicitly, to not use circular logic
    assert not np.isnan(Qs[q_1])  # Do this one explicitly, to not use circular logic
    assert np.isnan(Qs[q_nan1])  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nan]:
        assert np.isnan(q)
    for q in Qs[Qs_nonnan]:
        assert not np.isnan(q)


def test_quaternion_isinf(Qs):
    assert not np.isinf(Qs[q_0])  # Do this one explicitly, to not use circular logic
    assert not np.isinf(Qs[q_1])  # Do this one explicitly, to not use circular logic
    assert np.isinf(Qs[q_inf1])  # Do this one explicitly, to not use circular logic
    assert np.isinf(Qs[q_minf1])  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_inf]:
        assert np.isinf(q)
    for q in Qs[Qs_noninf]:
        assert not np.isinf(q)


def test_quaternion_isfinite(Qs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not np.isfinite(Qs[q_inf1])  # Do this one explicitly, to not use circular logic
        assert not np.isfinite(Qs[q_minf1])  # Do this one explicitly, to not use circular logic
    assert not np.isfinite(Qs[q_nan1])  # Do this one explicitly, to not use circular logic
    assert np.isfinite(Qs[q_0])  # Do this one explicitly, to not use circular logic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for q in Qs[Qs_nonfinite]:
            assert not np.isfinite(q)
    for q in Qs[Qs_finite]:
        assert np.isfinite(q)


# Binary bool returners
def test_quaternion_equal(Qs):
    for j in Qs_nonnan:
        assert Qs[j] == Qs[j]  # self equality
        for k in range(len(Qs)):  # non-self inequality
            assert (j == k) or (not (Qs[j] == Qs[k]))
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert not q == p  # nan should never equal anything


def test_quaternion_not_equal(Qs):
    for j in Qs_nonnan:
        assert not (Qs[j] != Qs[j])  # self non-not_equality
        for k in Qs_nonnan:  # non-self not_equality
            assert (j == k) or (Qs[j] != Qs[k])
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert q != p  # nan should never equal anything


# Unary float returners
def test_quaternion_absolute(Qs):
    for norm in [np.abs, lambda q: q.abs]:
        for q in Qs[Qs_nan]:
            assert np.isnan(norm(q))
        for q in Qs[Qs_inf]:
            if on_windows:
                assert np.isinf(norm(q)) or np.isnan(norm(q))
            else:
                assert np.isinf(norm(q))
        for q, a in [(Qs[q_0], 0.0), (Qs[q_1], 1.0), (Qs[x], 1.0), (Qs[y], 1.0), (Qs[z], 1.0),
                     (Qs[Q], np.sqrt(Qs[Q].w ** 2 + Qs[Q].x ** 2 + Qs[Q].y ** 2 + Qs[Q].z ** 2)),
                     (Qs[Qbar], np.sqrt(Qs[Q].w ** 2 + Qs[Q].x ** 2 + Qs[Q].y ** 2 + Qs[Q].z ** 2))]:
            assert np.allclose(norm(q), a)


def test_quaternion_norm(Qs):
    for q in Qs[Qs_nan]:
        assert np.isnan(q.norm)
    for q in Qs[Qs_inf]:
        if on_windows:
            assert np.isinf(q.norm) or np.isnan(q.norm)
        else:
            assert np.isinf(q.norm)
    for q, a in [(Qs[q_0], 0.0), (Qs[q_1], 1.0), (Qs[x], 1.0), (Qs[y], 1.0), (Qs[z], 1.0),
                 (Qs[Q], Qs[Q].w ** 2 + Qs[Q].x ** 2 + Qs[Q].y ** 2 + Qs[Q].z ** 2),
                 (Qs[Qbar], Qs[Q].w ** 2 + Qs[Q].x ** 2 + Qs[Q].y ** 2 + Qs[Q].z ** 2)]:
        assert np.allclose(q.norm, a)


# Unary quaternion returners
def test_quaternion_negative(Qs):
    assert -Qs[Q] == Qs[Qneg]
    for q in Qs[Qs_finite]:
        assert -q == -1.0 * q
    for q in Qs[Qs_nonnan]:
        assert -(-q) == q


def test_quaternion_conjugate(Qs):
    assert Qs[Q].conjugate() == Qs[Qbar]
    for q in Qs[Qs_nonnan]:
        assert q.conjugate() == q.conj()
        assert q.conjugate().conjugate() == q
        c = q.conjugate()
        assert c.w == q.w
        assert c.x == -q.x
        assert c.y == -q.y
        assert c.z == -q.z


def test_quaternion_npconjugate(Qs):
    assert np.conjugate(Qs[Q]) == Qs[Qbar]
    for q in Qs[Qs_nonnan]:
        assert np.conjugate(q) == np.conj(q)
        assert np.conjugate(np.conjugate(q)) == q
        c = np.conjugate(q)
        assert c.w == q.w
        assert c.x == -q.x
        assert c.y == -q.y
        assert c.z == -q.z


def test_quaternion_sqrt(Qs):
    sqrt_precision = 2.e-15
    # Test sqrt of basis elements
    sqrthalf = np.sqrt(0.5)
    assert np.array_equal(
        np.sqrt(quaternionic.array(np.eye(4))).ndarray,
        np.array([
            [1, 0, 0, 0],
            [sqrthalf, sqrthalf, 0, 0],
            [sqrthalf, 0, sqrthalf, 0],
            [sqrthalf, 0, 0, sqrthalf],
        ])
    )
    # Test all my samples
    one, i, j, k = tuple(quaternionic.array(np.eye(4)))
    for q in Qs[Qs_finitenonzero]:
        assert allclose(np.sqrt(q) * np.sqrt(q), q, rtol=sqrt_precision)
        # Ensure that non-unit quaternions are handled correctly
        for s in [1, -1, 2, -2, 3.4, -3.4]:
            for r in [one, i, j, k]:
                srq = s*r*q
                assert allclose(np.sqrt(srq) * np.sqrt(srq), srq, rtol=sqrt_precision)
    # Test a huge batch of random quaternions
    np.random.seed(1234)
    a = quaternionic.array(np.random.uniform(-10, 10, size=10_000*4).reshape((-1, 4)))
    assert np.allclose(a, np.square(np.sqrt(a)), rtol=10*sqrt_precision, atol=0)


def test_quaternion_square(Qs):
    square_precision = 1.e-15
    for q in Qs[Qs_finite]:
        assert (q*q - q**2).norm < square_precision
        a = quaternionic.array([q])
        assert (a**2 - quaternionic.array([q**2])).norm < square_precision


def test_quaternion_log_exp(Qs):
    qlogexp_precision = 4.e-15
    zero = quaternionic.array([0, 0, 0, 0])
    one, i, j, k = tuple(quaternionic.array(np.eye(4)))
    assert (np.log(Qs[Q]) - Qs[Qlog]).abs < qlogexp_precision
    assert (np.exp(Qs[Q]) - Qs[Qexp]).abs < qlogexp_precision
    assert (np.exp(np.log(Qs[Q])) - Qs[Q]).abs < qlogexp_precision
    assert (np.log(np.exp(Qs[Q])) - Qs[Q]).abs > qlogexp_precision  # Note order of operations!
    assert np.log(one) == zero
    assert np.log(i) == (np.pi / 2) * i
    assert np.log(j) == (np.pi / 2) * j
    assert np.log(k) == (np.pi / 2) * k
    assert np.log(-one) == (np.pi) * i



def test_xor():
    basis = one, i, j, k = tuple(quaternionic.array(np.eye(4)))
    zero = 0 * one
    assert one ^ one == one
    assert one ^ i == i
    assert one ^ j == j
    assert one ^ k == k
    assert i ^ one == i
    assert i ^ i == zero
    assert i ^ j == zero
    assert i ^ k == zero
    assert j ^ one == j
    assert j ^ i == zero
    assert j ^ j == zero
    assert j ^ k == zero
    assert k ^ one == k
    assert k ^ i == zero
    assert k ^ j == zero
    assert k ^ k == zero


def test_contractions():
    basis = one, i, j, k = tuple(quaternionic.array(np.eye(4)))
    zero = 0 * one

    assert one << one == one
    assert one << i == i
    assert one << j == j
    assert one << k == k
    assert i << one == zero
    assert i << i == -one
    assert i << j == zero
    assert i << k == zero
    assert j << one == zero
    assert j << i == zero
    assert j << j == -one
    assert j << k == zero
    assert k << one == zero
    assert k << i == zero
    assert k << j == zero
    assert k << k == -one

    assert one >> one == one
    assert one >> i == zero
    assert one >> j == zero
    assert one >> k == zero
    assert i >> one == i
    assert i >> i == -one
    assert i >> j == zero
    assert i >> k == zero
    assert j >> one == j
    assert j >> i == zero
    assert j >> j == -one
    assert j >> k == zero
    assert k >> one == k
    assert k >> i == zero
    assert k >> j == zero
    assert k >> k == -one

    for a in basis:
        for b in basis:
            for c in basis:
                assert np.allclose((a ^ b) | c, a | (b << c))
                assert np.allclose(c | (b ^ a), (c >> b) | a)


def test_metrics(Rs):
    metric_precision = 4.e-15
    intrinsic_funcs = (quaternionic.distance.rotor.intrinsic, quaternionic.distance.rotation.intrinsic)
    chordal_funcs = (quaternionic.distance.rotor.chordal, quaternionic.distance.rotation.chordal)
    metric_funcs = intrinsic_funcs + chordal_funcs
    rotor_funcs = (quaternionic.distance.rotor.intrinsic, quaternionic.distance.rotor.chordal)
    rotation_funcs = (quaternionic.distance.rotation.intrinsic, quaternionic.distance.rotation.chordal)

    # for R1 in Rs:
    #     for R2 in Rs:
    #         for f in metric_funcs:
    #             print()
    #             print(R1, R2, f)
    #             print(f(R1, R2))


    distance_dict = {func: func.outer(Rs, Rs) for func in metric_funcs}

    # Check non-negativity
    for mat in distance_dict.values():
        assert np.all(mat.ndarray >= 0.)

    # Check discernibility
    for func in metric_funcs:
        if func in chordal_funcs:
            eps = 0
        else:
            eps = 5.e-16
        if func in rotor_funcs:
            target = Rs != Rs[:, np.newaxis]
        else:
            target = np.logical_and(Rs != Rs[:, np.newaxis], Rs != - Rs[:, np.newaxis])
        assert ((distance_dict[func] > eps) == target).all()

    # Check symmetry
    for mat in distance_dict.values():
        assert np.allclose(mat, mat.T, atol=metric_precision, rtol=0)

    # Check triangle inequality
    for mat in distance_dict.values():
        assert ((mat - metric_precision)[:, np.newaxis, :] <= mat[:, :, np.newaxis] + mat).all()

    # Check distances from self or -self
    for func in metric_funcs:
        # All distances from self should be 0.0
        if func in chordal_funcs:
            eps = 0
        else:
            eps = 5.e-16
        assert (np.diag(distance_dict[func]) <= eps).all()

    # Chordal rotor distance from -self should be 2
    assert (abs(quaternionic.distance.rotor.chordal(Rs, -Rs) - 2.0) < metric_precision).all()
    # Intrinsic rotor distance from -self should be 2pi
    assert (abs(quaternionic.distance.rotor.intrinsic(Rs, -Rs) - 2.0 * np.pi) < metric_precision).all()
    # Rotation distances from -self should be 0
    assert (quaternionic.distance.rotation.chordal(Rs, -Rs) == 0.0).all()
    assert (quaternionic.distance.rotation.intrinsic(Rs, -Rs) < 5.e-16).all()

    # We expect the chordal distance to be smaller than the intrinsic distance (or equal, if the distance is zero)
    assert np.logical_or(quaternionic.distance.rotor.chordal(quaternion.one, Rs)
                           < quaternionic.distance.rotor.intrinsic(quaternion.one, Rs),
                         Rs == quaternion.one).all()
    # Check invariance under overall rotations: d(R1, R2) = d(R3*R1, R3*R2) = d(R1*R3, R2*R3)
    for func in quaternionic.distance.rotor.chordal, quaternionic.distance.rotation.intrinsic:
        rotations = Rs[:, np.newaxis] * Rs
        right_distances = func(rotations, rotations[:, np.newaxis])
        assert (abs(distance_dict[func][:, :, np.newaxis] - right_distances) < metric_precision).all()
        left_distances = func(rotations[:, :, np.newaxis], rotations[:, np.newaxis])
        assert (abs(distance_dict[func] - left_distances) < metric_precision).all()


def test_to_rotation_matrix(Rs):
    one, x, y, z = tuple(quaternionic.array(np.eye(4)))
    zero = 0.0 * one

    def quat_mat(quat):
        return np.array([(quat * v * quat.inverse).vector for v in [x, y, z]]).T

    def quat_mat_vec(quats):
        mat_vec = np.array([(quats * v * np.reciprocal(quats)).vector
                            for v in [x, y, z]])
        return np.transpose(mat_vec, tuple(range(mat_vec.ndim))[1:-1]+(-1, 0))

    with pytest.raises(ZeroDivisionError):
        zero.to_rotation_matrix

    for R in Rs:
        # Test correctly normalized rotors:
        assert allclose(quat_mat(R), R.to_rotation_matrix, atol=2*eps)
        # Test incorrectly normalized rotors:
        assert allclose(quat_mat(R), (1.1*R).to_rotation_matrix, atol=2*eps)

    Rs0 = Rs.copy()
    Rs0[Rs.shape[0]//2] = zero
    with pytest.raises(ZeroDivisionError):
        Rs0.to_rotation_matrix

    # Test correctly normalized rotors:

    assert allclose(quat_mat_vec(Rs), Rs.to_rotation_matrix, atol=2*eps)
    # Test incorrectly normalized rotors:
    assert allclose(quat_mat_vec(Rs), (1.1*Rs).to_rotation_matrix, atol=2*eps)

    # Simply test that this function succeeds and returns the right shape
    assert (Rs.reshape((2, 5, 10, 4))).to_rotation_matrix.shape == (2, 5, 10, 3, 3)


def test_from_rotation_matrix(Rs):
    try:
        from scipy import linalg
        have_linalg = True
    except ImportError:
        have_linalg = False

    for nonorthogonal in [True, False]:
        if nonorthogonal and have_linalg:
            rot_mat_eps = 10*eps
        else:
            rot_mat_eps = 5*eps
        for i, R1 in enumerate(Rs):
            R2 = quaternionic.array.from_rotation_matrix(R1.to_rotation_matrix, nonorthogonal=nonorthogonal)
            d = quaternionic.distance.c.rotation.intrinsic(R1, R2)
            assert d < rot_mat_eps, (i, R1, R2, d)  # Can't use allclose here; we don't care about rotor sign

        Rs2 = quaternionic.array.from_rotation_matrix(Rs.to_rotation_matrix, nonorthogonal=nonorthogonal)
        for R1, R2 in zip(Rs, Rs2):
            d = quaternionic.distance.c.rotation.intrinsic(R1, R2)
            assert d < rot_mat_eps, (R1, R2, d)  # Can't use allclose here; we don't care about rotor sign

        Rs3 = Rs.reshape((2, 5, 10))
        Rs4 = quaternionic.array.from_rotation_matrix(Rs3.to_rotation_matrix)
        for R3, R4 in zip(Rs3.flatten(), Rs4.flatten()):
            d = quaternionic.distance.c.rotation.intrinsic(R3, R4)
            assert d < rot_mat_eps, (R3, R4, d)  # Can't use allclose here; we don't care about rotor sign


def test_to_rotation_vector():
    np.random.seed(1234)
    n_tests = 1000
    vecs = np.random.uniform(high=math.pi/math.sqrt(3), size=n_tests*3).reshape((n_tests, 3))
    quats = np.zeros(vecs.shape[:-1]+(4,))
    quats[..., 1:] = vecs[...]
    quats = quaternionic.array(quats)
    quats = np.exp(quats/2)
    quat_vecs = quats.to_rotation_vector
    assert allclose(quat_vecs, vecs)


def test_from_rotation_vector():
    np.random.seed(1234)
    n_tests = 1000
    vecs = np.random.uniform(high=math.pi/math.sqrt(3), size=n_tests*3).reshape((n_tests, 3))
    quats = np.zeros(vecs.shape[:-1]+(4,))
    quats[..., 1:] = vecs[...]
    quats = quaternionic.array(quats)
    quats = np.exp(quats/2)
    quat_vecs = quaternionic.array.to_rotation_vector(quats)
    quats2 = quaternionic.array.from_rotation_vector(quat_vecs)
    assert allclose(quats, quats2)


def test_rotate_vectors(Rs):
    np.random.seed(1234)
    # Test (1)*(1)
    vecs = np.random.rand(3)
    quats = quaternion.z
    vecsprime = quaternion.rotate_vectors(quats, vecs)
    assert np.allclose(vecsprime,
                       (quats * quaternionic.array(*vecs) * quats.inverse).vector,
                       rtol=0.0, atol=0.0)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (1)*(5)
    vecs = np.random.rand(5, 3)
    quats = quaternion.z
    vecsprime = quaternion.rotate_vectors(quats, vecs)
    for i, vec in enumerate(vecs):
        assert np.allclose(vecsprime[i],
                           (quats * quaternionic.array(*vec) * quats.inverse).vector,
                           rtol=0.0, atol=0.0)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (1)*(5) inner axis
    vecs = np.random.rand(3, 5)
    quats = quaternion.z
    vecsprime = quaternion.rotate_vectors(quats, vecs, axis=-2)
    for i, vec in enumerate(vecs.T):
        assert np.allclose(vecsprime[:, i],
                           (quats * quaternionic.array(*vec) * quats.inverse).vector,
                           rtol=0.0, atol=0.0)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(1)
    vecs = np.random.rand(3)
    quats = Rs
    vecsprime = quaternion.rotate_vectors(quats, vecs)
    assert np.allclose(vecsprime,
                       [vprime.vector for vprime in quats * quaternionic.array(*vecs) * ~quats],
                       rtol=1e-15, atol=1e-15)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(5)
    vecs = np.random.rand(5, 3)
    quats = Rs
    vecsprime = quaternion.rotate_vectors(quats, vecs)
    for i, vec in enumerate(vecs):
        assert np.allclose(vecsprime[:, i],
                           [vprime.vector for vprime in quats * quaternionic.array(*vec) * ~quats],
                           rtol=1e-15, atol=1e-15)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)
    # Test (N)*(5) inner axis
    vecs = np.random.rand(3, 5)
    quats = Rs
    vecsprime = quaternion.rotate_vectors(quats, vecs, axis=-2)
    for i, vec in enumerate(vecs.T):
        assert np.allclose(vecsprime[:, :, i],
                           [vprime.vector for vprime in quats * quaternionic.array(*vec) * ~quats],
                           rtol=1e-15, atol=1e-15)
    assert quats.shape + vecs.shape == vecsprime.shape, ("Out of shape!", quats.shape, vecs.shape, vecsprime.shape)


def test_from_spherical_coords():
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for vartheta, varphi in random_angles:
        q = quaternionic.from_spherical_coords(vartheta, varphi)
        assert abs((np.quaternion(0, 0, 0, varphi / 2.).exp() * np.quaternion(0, 0, vartheta / 2., 0).exp())
                   - q) < 1.e-15
        xprime = q * quaternion.x * q.inverse
        yprime = q * quaternion.y * q.inverse
        zprime = q * quaternion.z * q.inverse
        nhat = np.quaternion(0.0, math.sin(vartheta)*math.cos(varphi), math.sin(vartheta)*math.sin(varphi),
                             math.cos(vartheta))
        thetahat = np.quaternion(0.0, math.cos(vartheta)*math.cos(varphi), math.cos(vartheta)*math.sin(varphi),
                                 -math.sin(vartheta))
        phihat = np.quaternion(0.0, -math.sin(varphi), math.cos(varphi), 0.0)
        assert abs(xprime - thetahat) < 1.e-15
        assert abs(yprime - phihat) < 1.e-15
        assert abs(zprime - nhat) < 1.e-15
    assert np.max(np.abs(quaternionic.from_spherical_coords(random_angles)
                         - np.array([quaternionic.from_spherical_coords(vartheta, varphi)
                                     for vartheta, varphi in random_angles]))) < 1.e-15


def test_to_spherical_coords(Rs):
    np.random.seed(1843)
    # First test on rotors that are precisely spherical-coordinate rotors
    random_angles = [[np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
                     for i in range(5000)]
    for vartheta, varphi in random_angles:
        vartheta2, varphi2 = quaternionic.to_spherical_coords(quaternionic.from_spherical_coords(vartheta, varphi))
        varphi2 = (varphi2 + 2*np.pi) if varphi2 < 0 else varphi2
        assert abs(vartheta - vartheta2) < 1e-12, ((vartheta, varphi), (vartheta2, varphi2))
        assert abs(varphi - varphi2) < 1e-12, ((vartheta, varphi), (vartheta2, varphi2))
    # Now test that arbitrary rotors rotate z to the appropriate location
    for R in Rs:
        vartheta, varphi = quaternionic.to_spherical_coords(R)
        R2 = quaternionic.from_spherical_coords(vartheta, varphi)
        assert (R*quaternion.z*R.inverse - R2*quaternion.z*R2.inverse).abs() < 4e-15, (R, R2, (vartheta, varphi))


def test_from_euler_angles():
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for alpha, beta, gamma in random_angles:
        assert abs((np.quaternion(0, 0, 0, alpha / 2.).exp()
                    * np.quaternion(0, 0, beta / 2., 0).exp()
                    * np.quaternion(0, 0, 0, gamma / 2.).exp()
                   )
                   - quaternionic.from_euler_angles(alpha, beta, gamma)) < 1.e-15
    assert np.max(np.abs(quaternionic.from_euler_angles(random_angles)
                         - np.array([quaternionic.from_euler_angles(alpha, beta, gamma)
                                     for alpha, beta, gamma in random_angles]))) < 1.e-15


def test_to_euler_angles():
    np.random.seed(1843)
    random_angles = [[np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi),
                      np.random.uniform(-np.pi, np.pi)]
                     for i in range(5000)]
    for alpha, beta, gamma in random_angles:
        R1 = quaternionic.from_euler_angles(alpha, beta, gamma)
        R2 = quaternionic.from_euler_angles(*list(quaternionic.to_euler_angles(R1)))
        d = quaternionic.distance.rotation.intrinsic(R1, R2)
        assert d < 6e3*eps, ((alpha, beta, gamma), R1, R2, d)  # Can't use allclose here; we don't care about rotor sign
    q0 = quaternionic.array(0, 0.6, 0.8, 0)
    assert q0.norm() == 1.0
    assert abs(q0 - quaternionic.from_euler_angles(*list(quaternionic.to_euler_angles(q0)))) < 1.e-15
