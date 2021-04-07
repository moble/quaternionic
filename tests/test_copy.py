import warnings
import copy
import pickle
import numpy as np
import quaternionic
import pytest


def np_copy(q):
    return np.copy(q)

def np_array_copy(q):
    return np.array(q, copy=True)

def np_array_copy_subok(q):
    return np.array(q, copy=True, subok=True)

def ndarray_copy(q):
    return q.copy()

def pickle_roundtrip(q):
    return pickle.loads(pickle.dumps(q))

def copy_copy(q):
    return copy.copy(q)

def copy_deepcopy(q):
    return copy.deepcopy(q)

# Note that np.copy and np.array(..., copy=True) return ndarray's, and thus lose information
copy_xfail = lambda f: pytest.param(f, marks=pytest.mark.xfail(reason="Unexpected numpy defaults"))
local_xfail = lambda f: pytest.param(f, marks=pytest.mark.xfail(reason="Can't pickle local object"))

@pytest.mark.parametrize("copier", [
    copy_xfail(np_copy), copy_xfail(np_array_copy), np_array_copy_subok,
    ndarray_copy, local_xfail(pickle_roundtrip), copy_copy, copy_deepcopy
])
def test_modes_copying_and_pickling(copier):
    q = quaternionic.array(np.random.normal(size=(17, 4)))
    c = copier(q)
    assert q is not c
    assert isinstance(c, type(q))
    assert np.array_equal(c, q)
