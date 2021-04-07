import numpy as np
import quaternionic
import pytest


def test_unflip_rotors(Rs):
    np.random.seed(12345)
    unflip_precision = 4e-16
    q = quaternionic.array.random((17, 1_000, 4), normalize=True)
    ndim = q.ndim
    axis = -2
    inplace = False
    with pytest.raises(ValueError):
        quaternionic.unflip_rotors(q, axis=-1, inplace=inplace)
    with pytest.raises(ValueError):
        quaternionic.unflip_rotors(q, axis=ndim-1, inplace=inplace)
    q_out = quaternionic.unflip_rotors(q, axis=axis, inplace=inplace)
    diff = np.linalg.norm(np.diff(q_out, axis=(axis % ndim)), axis=-1)
    assert np.sum(diff > 1.4142135623730950488016887242097) == 0
    q_in = np.empty((2,)+Rs.shape)
    q_in[0, ...] = Rs
    q_in[1, ...] = -Rs
    q_out = quaternionic.unflip_rotors(q_in, axis=0, inplace=False)
    assert np.array_equal(q_out[1], Rs)
    q_in = np.empty((2,)+Rs.shape)
    q_in[0, ...] = Rs
    q_in[1, ...] = -Rs
    quaternionic.unflip_rotors(q_in, axis=0, inplace=True)
    assert np.array_equal(q_in[1], Rs)


def test_slerp(Rs):
    from quaternionic import slerp
    from numpy import allclose
    slerp_precision = 4.e-15
    ones = [
        quaternionic.one,  # Skip -1 because of ambiguity interpolating between 1 and -1
        quaternionic.x, quaternionic.y, quaternionic.z,
        -quaternionic.x, -quaternionic.y, -quaternionic.z
    ]
    # Check extremes
    for Q1 in ones:
        assert quaternionic.distance.rotation.chordal(slerp(Q1, Q1, 0.0), Q1) < slerp_precision
        assert quaternionic.distance.rotation.chordal(slerp(Q1, Q1, 1.0), Q1) < slerp_precision
        assert quaternionic.distance.rotation.chordal(slerp(Q1, -Q1, 0.0), Q1) < slerp_precision
        assert quaternionic.distance.rotation.chordal(slerp(Q1, -Q1, 1.0), Q1) < slerp_precision
        for Q2 in ones:
            assert quaternionic.distance.rotation.chordal(slerp(Q1, Q2, 0.0), Q1) < slerp_precision
            assert quaternionic.distance.rotation.chordal(slerp(Q1, Q2, 1.0), Q2) < slerp_precision
            assert quaternionic.distance.rotation.chordal(slerp(Q1, -Q2, 0.0), Q1) < slerp_precision
            assert quaternionic.distance.rotation.chordal(slerp(Q1, -Q2, 1.0), -Q2) < slerp_precision
            assert quaternionic.distance.rotation.chordal(slerp(Q2, Q1, 0.0), Q2) < slerp_precision
            assert quaternionic.distance.rotation.chordal(slerp(Q2, Q1, 1.0), Q1) < slerp_precision
    # Test simple increases in each dimension
    for Q2 in ones[1:]:
        for t in np.linspace(0.0, 1.0, num=100, endpoint=True):
            assert quaternionic.distance.rotation.chordal(slerp(quaternionic.one, Q2, t),
                                                        (np.cos(np.pi * t / 2) * quaternionic.one + np.sin(
                                                            np.pi * t / 2) * Q2)) < slerp_precision
        t = np.linspace(0.0, 1.0, num=100, endpoint=True)
        assert allclose(slerp(quaternionic.one, Q2, t),
                        np.cos(np.pi * t / 2) * quaternionic.one + np.sin(np.pi * t / 2) * Q2)
        assert allclose(slerp(quaternionic.one, Q2, t),  #, -10.0, 20.0, 30 * t - 10.0),
                        np.cos(np.pi * t / 2) * quaternionic.one + np.sin(np.pi * t / 2) * Q2)
        t = 1.5 * t - 0.125
        assert allclose(slerp(quaternionic.one, Q2, t),
                        np.cos(np.pi * t / 2) * quaternionic.one + np.sin(np.pi * t / 2) * Q2)
    # Test that (slerp of rotated rotors) is (rotated slerp of rotors)
    for R in Rs:
        for Q2 in ones[1:]:
            for t in np.linspace(0.0, 1.0, num=100, endpoint=True):
                assert quaternionic.distance.rotation.chordal(
                    R * slerp(quaternionic.one, Q2, t),
                    slerp(R * quaternionic.one, R * Q2, t)
                ) < slerp_precision
            t = np.linspace(0.0, 1.0, num=100, endpoint=True)
            assert allclose(R * slerp(quaternionic.one, Q2, t),
                            slerp(R * quaternionic.one, R * Q2, t))


def test_squad(Rs):
    from quaternionic import slerp, squad
    np.random.seed(1234)
    squad_precision = 4.e-15
    ones = [
        quaternionic.one,
        quaternionic.x, quaternionic.y, quaternionic.z,
        -quaternionic.x, -quaternionic.y, -quaternionic.z
    ]
    t_in = np.linspace(0.0, 1.0, num=13, endpoint=True)
    t_out = np.linspace(0.0, 1.0, num=37, endpoint=True)
    t_out2 = np.array(sorted([np.random.uniform(0.0, 1.0) for i in range(59)]))
    # Make sure we get empty output when asked
    empty = quaternionic.array(np.zeros((0, 4)))
    assert np.array_equal(squad(empty, t_in, t_out), empty)
    assert np.array_equal(squad(Rs, t_in, np.array(())), empty)
    # squad interpolated onto the inputs should be the identity
    for R1 in Rs:
        for R2 in Rs:
            R_in = quaternionic.array([slerp(R1, R2, t) for t in t_in])
            assert np.all(np.abs(squad(R_in, t_in, t_in) - R_in) < squad_precision)
    # squad should be the same as slerp for linear interpolation
    for R in ones:
        R_in = quaternionic.array([slerp(quaternionic.one, R, t) for t in t_in])
        R_out_squad = squad(R_in, t_in, t_out)
        R_out_slerp = quaternionic.array([slerp(quaternionic.one, R, t) for t in t_out])
        # print(
        #     R, "\n",
        #     np.argmax(np.abs(R_out_squad - R_out_slerp)),
        #     len(R_out_squad), "\n",
        #     np.max(np.abs(R_out_squad - R_out_slerp)), "\n",
        #     R_out_squad[-6:], "\n",
        #     R_out_slerp[-6:],
        # )
        assert np.all(np.abs(R_out_squad - R_out_slerp) < squad_precision), (
            R,
            np.argmax(np.abs(R_out_squad - R_out_slerp)),
            len(R_out_squad),
            R_out_squad[np.argmax(np.abs(R_out_squad - R_out_slerp))-2:np.argmax(np.abs(R_out_squad - R_out_slerp))+3],
            R_out_slerp[np.argmax(np.abs(R_out_squad - R_out_slerp))-2:np.argmax(np.abs(R_out_squad - R_out_slerp))+3],
        )
        R_out_squad = squad(R_in, t_in, t_out2)
        R_out_slerp = quaternionic.array([slerp(quaternionic.one, R, t) for t in t_out2])
        assert np.all(np.abs(R_out_squad - R_out_slerp) < squad_precision)
        # assert False # Test unequal input time steps, and correct squad output [0,-2,-1]

    for i in range(len(ones)):
        R3 = quaternionic.array(np.roll(ones, i, axis=0)[:3])
        R_in = quaternionic.array([[slerp(quaternionic.one, R, t) for R in R3] for t in t_in])
        R_out_squad = squad(R_in, t_in, t_out)
        R_out_slerp = quaternionic.array([[slerp(quaternionic.one, R, t) for R in R3] for t in t_out])
        assert np.all(np.abs(R_out_squad - R_out_slerp) < squad_precision), (
            R,
            np.argmax(np.abs(R_out_squad - R_out_slerp)),
            len(R_out_squad),
            R_out_squad[np.argmax(np.abs(R_out_squad - R_out_slerp))-2:np.argmax(np.abs(R_out_squad - R_out_slerp))+3],
            R_out_slerp[np.argmax(np.abs(R_out_squad - R_out_slerp))-2:np.argmax(np.abs(R_out_squad - R_out_slerp))+3],
        )
        R_out_squad = squad(R_in, t_in, t_out2)
        R_out_slerp = quaternionic.array([[slerp(quaternionic.one, R, t) for R in R3] for t in t_out2])
        assert np.all(np.abs(R_out_squad - R_out_slerp) < squad_precision)
