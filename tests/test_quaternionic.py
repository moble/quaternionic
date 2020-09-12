import warnings
import numpy as np
import quaternionic
import pytest


def test_constants():
    for const in ['one', 'x', 'y', 'z', 'i', 'j', 'k']:
        assert hasattr(quaternionic, const)
        c = getattr(quaternionic, const)
        with pytest.raises(ValueError):
            c *= 1.2
    assert np.equal(quaternionic.one, quaternionic.array(1, 0, 0, 0))
    assert np.equal(quaternionic.x, quaternionic.array(0, 1, 0, 0))
    assert np.equal(quaternionic.i, quaternionic.array(0, 1, 0, 0))
    assert np.equal(quaternionic.y, quaternionic.array(0, 0, 1, 0))
    assert np.equal(quaternionic.j, quaternionic.array(0, 0, 1, 0))
    assert np.equal(quaternionic.z, quaternionic.array(0, 0, 0, 1))
    assert np.equal(quaternionic.k, quaternionic.array(0, 0, 0, 1))
