import numpy as np
import quaternionic
import pytest

from quaternionic import align

eps = np.finfo(float).eps

def test_alignment():
    N = 17
    a⃗ = np.random.normal(size=(N, 3))
    R = quaternionic.array.random().normalized

    # Test the exact result
    b⃗ = R.rotate(a⃗)
    Rprm = align(a⃗, b⃗)
    assert quaternionic.distance.rotation.intrinsic(R, Rprm.conj()) < 25*eps
    assert np.max(np.abs(a⃗ - Rprm.rotate(b⃗))) < 40*eps

    # Perturb b⃗ slightly
    δ = np.sqrt(eps)
    b⃗prm = b⃗ + (2*(np.random.rand(N,3) - 0.5) * δ/np.sqrt(3))
    Rprmprm = align(a⃗, b⃗prm)
    assert quaternionic.distance.rotation.intrinsic(R, Rprmprm.conj()) < 25*δ
    assert np.max(np.abs(a⃗ - Rprmprm.rotate(b⃗prm))) < 40*δ
