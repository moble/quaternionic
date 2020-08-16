import warnings
import numpy as np
import quaternionic
import pytest


def test_metrics(Rs):
    metric_precision = 4.e-15
    one = quaternionic.array(1, 0, 0, 0)
    intrinsic_funcs = (quaternionic.distance.rotor.intrinsic, quaternionic.distance.rotation.intrinsic)
    chordal_funcs = (quaternionic.distance.rotor.chordal, quaternionic.distance.rotation.chordal)
    metric_funcs = intrinsic_funcs + chordal_funcs
    rotor_funcs = (quaternionic.distance.rotor.intrinsic, quaternionic.distance.rotor.chordal)
    rotation_funcs = (quaternionic.distance.rotation.intrinsic, quaternionic.distance.rotation.chordal)
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
    assert (abs(quaternionic.distance.rotor.chordal(Rs, -Rs) - 2.0) < metric_precision).all()
    # Intrinsic rotor distance from -self should be 2pi
    assert (abs(quaternionic.distance.rotor.intrinsic(Rs, -Rs) - 2.0 * np.pi) < metric_precision).all()
    # Rotation distances from -self should be 0
    assert (quaternionic.distance.rotation.chordal(Rs, -Rs) == 0.0).all()
    assert (quaternionic.distance.rotation.intrinsic(Rs, -Rs) < 5.e-16).all()

    # We expect the chordal distance to be smaller than the intrinsic distance (or equal, if the distance is zero)
    assert np.logical_or(quaternionic.distance.rotor.chordal(one, Rs)
                           < quaternionic.distance.rotor.intrinsic(one, Rs),
                         Rs == one).all()
    # Check invariance under overall rotations: d(R1, R2) = d(R3*R1, R3*R2) = d(R1*R3, R2*R3)
    for func in quaternionic.distance.rotor.chordal, quaternionic.distance.rotation.intrinsic:
        rotations = Rs[:, np.newaxis] * Rs
        right_distances = func(rotations, rotations[:, np.newaxis])
        assert (abs(distance_dict[func][:, :, np.newaxis] - right_distances) < metric_precision).all()
        left_distances = func(rotations[:, :, np.newaxis], rotations[:, np.newaxis])
        assert (abs(distance_dict[func] - left_distances) < metric_precision).all()
