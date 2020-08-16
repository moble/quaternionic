import warnings
import numpy as np
import quaternionic
import pytest


# Unary bool returners
def test_quaternion_nonzero(Qs, Q_names, Q_conditions):
    assert np.nonzero(Qs[Q_names.q_0])[0].size == 0  # Do this one explicitly, to not use circular logic
    assert np.nonzero(Qs[Q_names.q_1])[0].size > 0  # Do this one explicitly, to not use circular logic
    for q in Qs[Q_conditions.zero]:
        assert np.nonzero(q)[0].size == 0
    for q in Qs[Q_conditions.nonzero]:
        assert np.nonzero(q)[0].size > 0


def test_quaternion_isnan(Qs, Q_names, Q_conditions):
    assert not np.isnan(Qs[Q_names.q_0])  # Do this one explicitly, to not use circular logic
    assert not np.isnan(Qs[Q_names.q_1])  # Do this one explicitly, to not use circular logic
    assert np.isnan(Qs[Q_names.q_nan1])  # Do this one explicitly, to not use circular logic
    for q in Qs[Q_conditions.nan]:
        assert np.isnan(q)
    for q in Qs[Q_conditions.nonnan]:
        assert not np.isnan(q)


def test_quaternion_isinf(Qs, Q_names, Q_conditions):
    assert not np.isinf(Qs[Q_names.q_0])  # Do this one explicitly, to not use circular logic
    assert not np.isinf(Qs[Q_names.q_1])  # Do this one explicitly, to not use circular logic
    assert np.isinf(Qs[Q_names.q_inf1])  # Do this one explicitly, to not use circular logic
    assert np.isinf(Qs[Q_names.q_minf1])  # Do this one explicitly, to not use circular logic
    for q in Qs[Q_conditions.inf]:
        assert np.isinf(q)
    for q in Qs[Q_conditions.noninf]:
        assert not np.isinf(q)


def test_quaternion_isfinite(Qs, Q_names, Q_conditions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not np.isfinite(Qs[Q_names.q_inf1])  # Do this one explicitly, to not use circular logic
        assert not np.isfinite(Qs[Q_names.q_minf1])  # Do this one explicitly, to not use circular logic
    assert not np.isfinite(Qs[Q_names.q_nan1])  # Do this one explicitly, to not use circular logic
    assert np.isfinite(Qs[Q_names.q_0])  # Do this one explicitly, to not use circular logic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for q in Qs[Q_conditions.nonfinite]:
            assert not np.isfinite(q)
    for q in Qs[Q_conditions.finite]:
        assert np.isfinite(q)


# Binary bool returners
def test_quaternion_equal(Qs, Q_names, Q_conditions):
    for j in Q_conditions.nonnan:
        assert Qs[j] == Qs[j]  # self equality
        for k in range(len(Qs)):  # non-self inequality
            assert (j == k) or (not (Qs[j] == Qs[k]))
    for q in Qs:
        for p in Qs[Q_conditions.nan]:
            assert not q == p  # nan should never equal anything


def test_quaternion_not_equal(Qs, Q_names, Q_conditions):
    for j in Q_conditions.nonnan:
        assert not (Qs[j] != Qs[j])  # self non-not_equality
        for k in Q_conditions.nonnan:  # non-self not_equality
            assert (j == k) or (Qs[j] != Qs[k])
    for q in Qs:
        for p in Qs[Q_conditions.nan]:
            assert q != p  # nan should never equal anything


# Unary float returners
def test_quaternion_absolute(Qs, Q_names, Q_conditions, on_windows):
    for abs_func in [np.abs, lambda q: q.abs]:
        for q in Qs[Q_conditions.nan]:
            assert np.isnan(abs_func(q))
        for q in Qs[Q_conditions.inf]:
            if on_windows:  # pragma: no cover
                assert np.isinf(abs_func(q)) or np.isnan(abs_func(q))
            else:
                assert np.isinf(abs_func(q))
        for q, a in [(Qs[Q_names.q_0], 0.0), (Qs[Q_names.q_1], 1.0), (Qs[Q_names.x], 1.0), (Qs[Q_names.y], 1.0), (Qs[Q_names.z], 1.0),
                     (Qs[Q_names.Q], np.sqrt(Qs[Q_names.Q].w ** 2 + Qs[Q_names.Q].x ** 2 + Qs[Q_names.Q].y ** 2 + Qs[Q_names.Q].z ** 2)),
                     (Qs[Q_names.Qbar], np.sqrt(Qs[Q_names.Q].w ** 2 + Qs[Q_names.Q].x ** 2 + Qs[Q_names.Q].y ** 2 + Qs[Q_names.Q].z ** 2))]:
            assert np.allclose(abs_func(q), a)


def test_quaternion_norm(Qs, Q_names, Q_conditions, on_windows):
    for q in Qs[Q_conditions.nan]:
        assert np.isnan(q.norm)
    for q in Qs[Q_conditions.inf]:
        if on_windows:  # pragma: no cover
            assert np.isinf(q.norm) or np.isnan(q.norm)
        else:
            assert np.isinf(q.norm)
    for q, a in [(Qs[Q_names.q_0], 0.0), (Qs[Q_names.q_1], 1.0), (Qs[Q_names.x], 1.0), (Qs[Q_names.y], 1.0), (Qs[Q_names.z], 1.0),
                 (Qs[Q_names.Q], Qs[Q_names.Q].w ** 2 + Qs[Q_names.Q].x ** 2 + Qs[Q_names.Q].y ** 2 + Qs[Q_names.Q].z ** 2),
                 (Qs[Q_names.Qbar], Qs[Q_names.Q].w ** 2 + Qs[Q_names.Q].x ** 2 + Qs[Q_names.Q].y ** 2 + Qs[Q_names.Q].z ** 2)]:
        assert np.allclose(q.norm, a)


# Unary quaternion returners
def test_quaternion_negative(Qs, Q_names, Q_conditions):
    assert -Qs[Q_names.Q] == Qs[Q_names.Qneg]
    for q in Qs[Q_conditions.finite]:
        assert -q == -1.0 * q
    for q in Qs[Q_conditions.nonnan]:
        assert -(-q) == q


def test_quaternion_npconjugate(Qs, Q_names, Q_conditions):
    assert np.conjugate(Qs[Q_names.Q]) == Qs[Q_names.Qbar]
    for q in Qs[Q_conditions.nonnan]:
        assert np.conjugate(q) == np.conj(q)
        assert np.conjugate(np.conjugate(q)) == q
        c = np.conjugate(q)
        assert c.w == q.w
        assert c.x == -q.x
        assert c.y == -q.y
        assert c.z == -q.z


def test_quaternion_sqrt(Qs, Q_names, Q_conditions):
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
    for q in Qs[Q_conditions.finitenonzero]:
        assert np.allclose(np.sqrt(q) * np.sqrt(q), q, rtol=sqrt_precision)
        # Ensure that non-unit quaternions are handled correctly
        for s in [1, -1, 2, -2, 3.4, -3.4]:
            for r in [one, i, j, k]:
                srq = s*r*q
                assert np.allclose(np.sqrt(srq) * np.sqrt(srq), srq, rtol=sqrt_precision)
    # Test a huge batch of random quaternions
    np.random.seed(1234)
    a = quaternionic.array(np.random.uniform(-10, 10, size=10_000*4).reshape((-1, 4)))
    assert np.allclose(a, np.square(np.sqrt(a)), rtol=10*sqrt_precision, atol=0)


def test_quaternion_square(Qs, Q_names, Q_conditions):
    square_precision = 1.e-15
    for q in Qs[Q_conditions.finite]:
        assert (q*q - q**2).norm < square_precision
        a = quaternionic.array([q])
        assert (a**2 - quaternionic.array([q**2])).norm < square_precision


def test_quaternion_log_exp(Qs, Q_names, Q_conditions):
    qlogexp_precision = 4.e-15
    zero = quaternionic.array([0, 0, 0, 0])
    one, i, j, k = tuple(quaternionic.array(np.eye(4)))
    assert (np.log(Qs[Q_names.Q]) - Qs[Q_names.Qlog]).abs < qlogexp_precision
    assert (np.exp(Qs[Q_names.Q]) - Qs[Q_names.Qexp]).abs < qlogexp_precision
    assert (np.exp(np.log(Qs[Q_names.Q])) - Qs[Q_names.Q]).abs < qlogexp_precision
    assert (np.log(np.exp(Qs[Q_names.Q])) - Qs[Q_names.Q]).abs > qlogexp_precision  # Note order of operations!
    assert np.log(one) == zero
    assert np.log(i) == (np.pi / 2) * i
    assert np.log(j) == (np.pi / 2) * j
    assert np.log(k) == (np.pi / 2) * k
    assert np.log(-one) == (np.pi) * i


# Binary quat returners
def test_quaternion_conjugate(Qs, Q_names, Q_conditions):
    for conj_func in [np.conj, np.conjugate, lambda q: q.conj(), lambda q: q.conjugate()]:
        assert conj_func(Qs[Q_names.Q]) == Qs[Q_names.Qbar]
        for q in Qs[Q_conditions.nonnan]:
            assert conj_func(conj_func(q)) == q
            c = conj_func(q)
            assert c.w == q.w
            assert c.x == -q.x
            assert c.y == -q.y
            assert c.z == -q.z
    for q in Qs[Q_conditions.nonnan]:
        assert q.conjugate() == q.conj()
        assert np.conjugate(q) == np.conj(q)


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
