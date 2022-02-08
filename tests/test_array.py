import warnings
import numpy as np
import quaternionic
import pytest


# Array methods
def test_new(array):
    q = array(1, 2, 3, 4)
    assert q.dtype == float
    assert q.shape == (4,)
    assert q.w == 1.0 and q.x == 2.0 and q.y == 3.0 and q.z == 4.0
    q = array([1, 2, 3, 4])
    assert q.dtype == float
    assert q.shape == (4,)
    assert q.w == 1.0 and q.x == 2.0 and q.y == 3.0 and q.z == 4.0
    q = array([[1, 2, 3, 4]])
    assert q.dtype == float
    assert q.shape == (1, 4)
    assert q.w == 1.0 and q.x == 2.0 and q.y == 3.0 and q.z == 4.0
    with pytest.raises(ValueError):
        array(np.array(3.14))
    with pytest.raises(ValueError):
        array(np.array([]))
    with pytest.raises(ValueError):
        array(np.random.rand(4, 3))


def test_getitem(array):
    q = array(np.random.normal(size=(17, 3, 4)))
    p = q[1:-1]
    assert isinstance(p, array)
    assert p.shape == (q.shape[0]-2,) + q.shape[1:]
    assert np.array_equal(p.ndarray, q.ndarray[1:-1])
    with pytest.raises(ValueError):
        q[..., 1:3]


def test_array_finalize(array):
    q = array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        q[1:3]


def test_repr(array):
    q = array(np.random.normal(size=(17, 3, 4)))
    assert repr(q) == 'quaternionic.' + repr(q.ndarray)


def test_str(array):
    q = array(np.random.normal(size=(17, 3, 4)))
    assert str(q) == str(q.ndarray)


def test_ones_like(Qs):
    z = np.ones_like(Qs)
    assert np.all(z.ndarray[:, 1:] == 0)
    assert np.all(z.ndarray[:, 0] == 1)
