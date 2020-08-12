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


def test_basic_construction():
    q = quaternionic.Quaternion([1, 2, 3, 4])  # Note the integer input
    assert q.w == 1.0
    assert q.x == 2.0
    assert q.y == 3.0
    assert q.z == 4.0


def test_basis_multiplication():
    # Basis components
    one, i, j, k = tuple(quaternionic.Quaternion(np.eye(4)))

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

    # Standard criteria
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
    return quaternionic.Quaternion([
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
    ones = [0, -1., 1.]
    rs = [
        quaternionic.Quaternion([w, x, y, z]).normalized()
        for w in ones for x in ones for y in ones for z in ones
    ][1:]
    np.random.seed(1842)
    rs = rs + [r.normalized() for r in [quaternionic.Quaternion(np.random.uniform(-1, 1, size=4)) for _ in range(20)]]
    return quaternionic.Quaternion(rs)




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
    one, i, j, k = tuple(quaternionic.Quaternion(np.eye(4)))
    for q in Qs[Qs_finitenonzero]:
        assert allclose(np.sqrt(q) * np.sqrt(q), q, rtol=sqrt_precision)
        # Ensure that non-unit quaternions are handled correctly
        for s in [1, -1, 2, -2, 3.4, -3.4]:
            for r in [one, i, j, k]:
                srq = s*r*q
                assert allclose(np.sqrt(srq) * np.sqrt(srq), srq, rtol=sqrt_precision)


def test_quaternion_square(Qs):
    square_precision = 1.e-15
    for q in Qs[Qs_finite]:
        assert (q*q - q**2).norm < square_precision
        a = quaternionic.Quaternion([q])
        assert (a**2 - quaternionic.Quaternion([q**2])).norm < square_precision


def test_quaternion_log_exp(Qs):
    qlogexp_precision = 4.e-15
    zero = quaternionic.Quaternion([0, 0, 0, 0])
    one, i, j, k = tuple(quaternionic.Quaternion(np.eye(4)))
    assert (np.log(Qs[Q]) - Qs[Qlog]).abs < qlogexp_precision
    assert (np.exp(Qs[Q]) - Qs[Qexp]).abs < qlogexp_precision
    assert (np.exp(np.log(Qs[Q])) - Qs[Q]).abs < qlogexp_precision
    assert (np.log(np.exp(Qs[Q])) - Qs[Q]).abs > qlogexp_precision  # Note order of operations!
    assert np.log(one) == zero
    assert np.log(i) == (np.pi / 2) * i
    assert np.log(j) == (np.pi / 2) * j
    assert np.log(k) == (np.pi / 2) * k
    assert np.log(-one) == (np.pi) * i
