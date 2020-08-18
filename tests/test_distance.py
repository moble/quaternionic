import warnings
import numpy as np
import quaternionic
import pytest


@pytest.mark.parametrize("rotor,rotation,slow", [  # pragma: no branch
    (quaternionic.distance.rotor, quaternionic.distance.rotation, True),
    quaternionic.distance.CreateMetrics(lambda f: f, quaternionic.utilities.pyguvectorize) + (False,)
], ids=["jit metrics", "non-jit metrics"])
def test_metrics(Rs, array, rotor, rotation, slow):
    metric_precision = 4.e-15
    Rs = array(Rs.ndarray)
    one = array(1, 0, 0, 0)

    intrinsic_funcs = (rotor.intrinsic, rotation.intrinsic)
    chordal_funcs = (rotor.chordal, rotation.chordal)
    metric_funcs = intrinsic_funcs + chordal_funcs
    rotor_funcs = (rotor.intrinsic, rotor.chordal)
    rotation_funcs = (rotation.intrinsic, rotation.chordal)
    distance_dict = {func: func(Rs, Rs[:, np.newaxis]) for func in metric_funcs}

    # Check non-negativity
    for mat in distance_dict.values():
        assert np.all(mat >= 0.)

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
    assert (abs(rotor.chordal(Rs, -Rs) - 2.0) < metric_precision).all()
    # Intrinsic rotor distance from -self should be 2pi
    assert (abs(rotor.intrinsic(Rs, -Rs) - 2.0 * np.pi) < metric_precision).all()
    # Rotation distances from -self should be 0
    assert (rotation.chordal(Rs, -Rs) == 0.0).all()
    assert (rotation.intrinsic(Rs, -Rs) < 5.e-16).all()

    # We expect the chordal distance to be smaller than the intrinsic distance (or equal, if the distance is zero)
    assert np.logical_or(rotor.chordal(one, Rs) < rotor.intrinsic(one, Rs), Rs == one).all()

    if slow:
        # Check invariance under overall rotations: d(R1, R2) = d(R3*R1, R3*R2) = d(R1*R3, R2*R3)
        for func in rotor.chordal, rotation.intrinsic:
            rotations = Rs[:, np.newaxis] * Rs
            right_distances = func(rotations, rotations[:, np.newaxis])
            assert (abs(distance_dict[func][:, :, np.newaxis] - right_distances) < metric_precision).all()
            left_distances = func(rotations[:, :, np.newaxis], rotations[:, np.newaxis])
            assert (abs(distance_dict[func] - left_distances) < metric_precision).all()
