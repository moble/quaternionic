"""Wahba alignment is invariant to vector storage dtype and weight units."""

import numpy as np
import pytest
import quaternionic


def _weighted_kabsch(a, b, weights=None):
    # Independent SVD optimizer of sum_i w_i ||a_i - R b_i||^2.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    weights = (
        np.ones(len(a)) if weights is None else np.asarray(weights, dtype=np.float64)
    )
    weights = weights / weights.sum()
    left, _, right = np.linalg.svd((weights[:, None] * a).T @ b)
    orientation = np.eye(3)
    orientation[2, 2] = np.linalg.det(left @ right)
    return left @ orientation @ right


def _rotation_matrix():
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = 0.73
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("weighted", [False, True])
def test_alignment_integer_vectors_and_floating_correlations(dtype, weighted):
    a = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3], [1, -1, 2]], dtype=dtype)
    rotation = _rotation_matrix()
    b = a @ rotation.T
    weights = np.array([0.01, 0.02, 0.03, 0.04]) if weighted else None
    original_a, original_b = a.copy(), b.copy()
    actual = quaternionic.align(a, b, weights)
    np.testing.assert_allclose(
        actual.to_rotation_matrix, rotation.T, atol=2e-14, rtol=2e-14
    )
    np.testing.assert_allclose(
        actual.to_rotation_matrix,
        _weighted_kabsch(a, b, weights),
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(actual.rotate(b), a, atol=8e-14, rtol=2e-14)
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(b, original_b)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_alignment_fractional_weights_and_weight_scale_invariance(dtype):
    a = np.array([[1, 0, 1], [0, 2, 0], [1, -1, 2], [-2, 0, 1]], dtype=dtype)
    b = np.array([[0, 1, 1], [-1, 2, 1], [2, 0, 1], [1, -1, 2]], dtype=dtype)
    weights = np.array([0.2, 0.3, 0.4, 1.0])
    expected = _weighted_kabsch(a, b, weights)
    for scale in [1e-3, 1.0, 1e3]:
        observed = quaternionic.align(a, b, scale * weights).to_rotation_matrix
        np.testing.assert_allclose(observed, expected, atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(np.linalg.det(observed), 1.0, atol=3e-14, rtol=0.0)
        normalized = weights / weights.sum()
        expected_loss = np.sum(normalized[:, None] * (a - b @ expected.T) ** 2)
        observed_loss = np.sum(normalized[:, None] * (a - b @ observed.T) ** 2)
        np.testing.assert_allclose(observed_loss, expected_loss, atol=3e-14, rtol=3e-14)


def test_alignment_preserves_the_precision_of_b_and_weights():
    a = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3], [1, -1, 2]], dtype=np.float32)
    b = a.astype(np.float64) @ _rotation_matrix().T
    b += np.array(
        [
            [0.01, 0.03, -0.02],
            [0.02, -0.01, 0.04],
            [-0.03, 0.04, 0.01],
            [0.02, 0.01, -0.03],
        ]
    )
    weights = np.array([0.2, 0.3, 0.4, 1.0], dtype=np.float64)
    observed = quaternionic.align(a, b, weights).to_rotation_matrix
    expected = _weighted_kabsch(a, b, weights)
    np.testing.assert_allclose(observed, expected, atol=3e-14, rtol=3e-14)


def test_alignment_float64_control_and_swap_inverse():
    a = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, -1.0, 2.0]])
    b = a @ _rotation_matrix().T
    weights = np.array([0.2, 0.3, 0.4, 1.0])
    observed = quaternionic.align(a, b, weights).to_rotation_matrix
    inverse = quaternionic.align(b, a, weights).to_rotation_matrix
    np.testing.assert_allclose(
        observed, _weighted_kabsch(a, b, weights), atol=3e-14, rtol=3e-14
    )
    np.testing.assert_allclose(inverse, observed.T, atol=3e-14, rtol=3e-14)
