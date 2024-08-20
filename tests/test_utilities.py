import subprocess
import platform
import os
import sys
import numpy as np
import quaternionic
import pytest


def test_self_return():
    def f1(a, b, c):
        d = np.asarray(a).copy()
        assert isinstance(a, np.ndarray) and isinstance(a, quaternionic.array)
        assert isinstance(b, np.ndarray) and isinstance(b, quaternionic.array)
        assert isinstance(c, np.ndarray) and isinstance(c, quaternionic.array)
        assert isinstance(d, np.ndarray) and not isinstance(d, quaternionic.array)
        return d
    a = quaternionic.array.random((17, 3, 4))
    b = quaternionic.array.random((13, 3, 4))
    c = quaternionic.array.random((11, 3, 4))
    d1 = f1(a, b, c)
    assert isinstance(d1, np.ndarray) and not isinstance(d1, quaternionic.array)
    f2 = quaternionic.utilities.type_self_return(f1)
    d2 = f2(a, b, c)
    assert isinstance(d2, np.ndarray) and isinstance(d2, quaternionic.array)
    f1.nin = 3
    f3 = quaternionic.utilities.type_self_return(f1)
    d3 = f3(a, b, c)
    assert isinstance(d3, np.ndarray) and isinstance(d3, quaternionic.array)


def test_ndarray_args():
    def f1(a, b, c):
        d = np.asarray(a).copy()
        assert isinstance(a, np.ndarray) and not isinstance(a, quaternionic.array)
        assert isinstance(b, np.ndarray) and not isinstance(b, quaternionic.array)
        assert isinstance(c, np.ndarray) and not isinstance(c, quaternionic.array)
        assert isinstance(d, np.ndarray) and not isinstance(d, quaternionic.array)
        return d
    a = quaternionic.array.random((17, 3, 4))
    b = quaternionic.array.random((13, 3, 4))
    c = quaternionic.array.random((11, 3, 4))
    f2 = quaternionic.utilities.ndarray_args(f1)
    d2 = f2(a, b, c)
    assert isinstance(d2, np.ndarray) and not isinstance(d2, quaternionic.array)
    f1.nin = 3
    f3 = quaternionic.utilities.ndarray_args(f1)
    d3 = f3(a, b, c)
    assert isinstance(d3, np.ndarray) and not isinstance(d3, quaternionic.array)


def test_ndarray_args_and_return():
    def f1(a, b, c):
        d = np.asarray(a).copy()
        assert isinstance(a, np.ndarray) and not isinstance(a, quaternionic.array)
        assert isinstance(b, np.ndarray) and not isinstance(b, quaternionic.array)
        assert isinstance(c, np.ndarray) and not isinstance(c, quaternionic.array)
        assert isinstance(d, np.ndarray) and not isinstance(d, quaternionic.array)
        return d
    a = quaternionic.array.random((17, 3, 4))
    b = quaternionic.array.random((13, 3, 4))
    c = quaternionic.array.random((11, 3, 4))
    f2 = quaternionic.utilities.ndarray_args_and_return(f1)
    d2 = f2(a, b, c)
    assert isinstance(d2, np.ndarray) and isinstance(d2, quaternionic.array)
    f1.nin = 3
    f3 = quaternionic.utilities.ndarray_args_and_return(f1)
    d3 = f3(a, b, c)
    assert isinstance(d3, np.ndarray) and isinstance(d3, quaternionic.array)


@pytest.mark.skipif(sys.implementation.name.lower() == 'pypy', reason="No numba on pypy")
def test_types_to_ftylist():
    import numba
    types_to_ftylist = quaternionic.utilities.convert_numpy_ufunc_type_to_numba_ftylist
    types = '?bhilqpBHILQPfgF->D'
    ftylist = numba.complex128(
        numba.boolean,
        numba.byte,
        numba.short,
        numba.intc,
        numba.int_,
        numba.longlong,
        numba.intp,
        numba.char,
        numba.ushort,
        numba.uintc,
        numba.uint,
        numba.ulonglong,
        numba.uintp,
        numba.float32,
        numba.double,
        numba.complex64,
    )
    assert types_to_ftylist([types]) == [ftylist]


def test_pyguvectorize():
    _quaternion_resolution = 10 * np.finfo(float).resolution
    np.random.seed(1234)
    one = quaternionic.array(1, 0, 0, 0)
    x = quaternionic.array.random((7, 13, 4))
    y = quaternionic.array.random((13, 4))
    z = np.random.rand(13)

    arg0s = [one, -(1+2*_quaternion_resolution)*one, -one, x]

    for k in dir(quaternionic.algebra_ufuncs):
        if not k.startswith('__'):
            f1 = getattr(quaternionic.algebra_ufuncs, k)
            f2 = getattr(quaternionic.algebra, k)
            sig = f2.signature
            inputs = sig.split('->')[0].split(',')
            for arg0 in arg0s:
                args = [arg0.ndarray] if inputs[0] == '(n)' else [z,]
                if len(inputs) > 1:
                    args.append(y.ndarray if inputs[1] == '(n)' else z)
                assert np.allclose(
                    f1(*args),
                    quaternionic.utilities.pyguvectorize(f2.types, f2.signature)(f2)(*args),
                    atol=0.0,
                    rtol=_quaternion_resolution
                )


@pytest.mark.skipif(sys.implementation.name.lower() == "pypy", reason="No numba on pypy")
def test_cache_disable(tmp_path):
    # Stolen from https://github.com/pypa/setuptools/blob/477f713450ff57de126153f3034d032542916d03/setuptools/tests/test_distutils_adoption.py#L13-L22
    def win_sr(env):
        """
        On Windows, SYSTEMROOT must be present to avoid
    
        > Fatal Python error: _Py_HashRandomization_Init: failed to get random numbers to initialize Python
        """
        if platform.system() == "Windows":
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        return env

    # First check caching works by default.
    cache_dir = tmp_path / "enabled"
    subprocess.run(
        [sys.executable, "-c", "import quaternionic"],
        env=win_sr({
            "NUMBA_CACHE_DIR": str(cache_dir),
        }),
    )

    # Numba uses a subdirectory named `quaternionic_<hashstr>`, with the hashstr computed
    # from the full path to the source. We should have 1 directory in our cache, and that
    # should contain many files.
    contents = list(cache_dir.iterdir())
    assert len(contents) == 1
    subdir = contents[0]
    assert subdir.name.startswith("quaternionic_")
    assert len(list(subdir.iterdir())) > 10

    # Change the cache location and check with the environment variable disabling caching.
    # This seems to not create the cache directory, but include an alternative check that
    # it is empty if it exists in case the directory does get created sometimes.
    cache_dir = tmp_path / "disabled"
    subprocess.run(
        [sys.executable, "-c", "import quaternionic"],
        env=win_sr({
            "QUATERNIONIC_DISABLE_CACHE": "1",
            "NUMBA_CACHE_DIR": str(cache_dir),
        }),
    )
    assert (not cache_dir.exists()) or len(list(cache_dir.iterdir())) == 0
