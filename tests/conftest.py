import warnings
import numpy as np
import quaternionic
import pytest


def pytest_generate_tests(metafunc):
    if "array" in metafunc.fixturenames:
        metafunc.parametrize(
            "array",
            [quaternionic.QuaternionicArray(), quaternionic.QuaternionicArray(jit=lambda f: f)],
            ids=["jit array", "non-jit array"],
        )


@pytest.fixture
def on_windows():
    from sys import platform
    return 'win' in platform.lower() and not 'darwin' in platform.lower()


@pytest.fixture
def eps():
    return np.finfo(float).eps


def quaternion_sampler():
    Qs_array = quaternionic.array([
        [np.nan, 0., 0., 0.],
        [np.inf, 0., 0., 0.],
        [-np.inf, 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1.1, 2.2, 3.3, 4.4],
        [-1.1, -2.2, -3.3, -4.4],
        [1.1, -2.2, -3.3, -4.4],
        [
            0.18257418583505537115232326093360,
            0.36514837167011074230464652186720,
            0.54772255750516611345696978280080,
            0.73029674334022148460929304373440
        ],
        [1.7959088706354, 0.515190292664085, 0.772785438996128, 1.03038058532817],
        [2.81211398529184, -0.392521193481878, -0.588781790222817, -0.785042386963756],
    ])
    names = type("QNames", (object,), dict())()
    names.q_nan1 = 0
    names.q_inf1 = 1
    names.q_minf1 = 2
    names.q_0 = 3
    names.q_1 = 4
    names.x = 5
    names.y = 6
    names.z = 7
    names.Q = 8
    names.Qneg = 9
    names.Qbar = 10
    names.Qnormalized = 11
    names.Qlog = 12
    names.Qexp = 13
    return Qs_array, names


@pytest.fixture
def Qs():
    return quaternion_sampler()[0]


@pytest.fixture
def Q_names():
    return quaternion_sampler()[1]


@pytest.fixture
def Q_conditions():
    Qs_array, names = quaternion_sampler()
    conditions = type("QConditions", (object,), dict())()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conditions.zero = np.arange(len(Qs_array))[Qs_array == Qs_array[names.q_0]]
        conditions.nonzero = np.arange(len(Qs_array))[np.nonzero(Qs_array)]
        conditions.nan = np.arange(len(Qs_array))[np.isnan(Qs_array)]
        conditions.nonnan = np.arange(len(Qs_array))[~np.isnan(Qs_array)]
        conditions.nonnannonzero = np.arange(len(Qs_array))[~np.isnan(Qs_array) & (Qs_array != Qs_array[names.q_0])]
        conditions.inf = np.arange(len(Qs_array))[np.isinf(Qs_array)]
        conditions.noninf = np.arange(len(Qs_array))[~np.isinf(Qs_array)]
        conditions.noninfnonzero = np.arange(len(Qs_array))[~np.isinf(Qs_array) & (Qs_array != Qs_array[names.q_0])]
        conditions.finite = np.arange(len(Qs_array))[np.isfinite(Qs_array)]
        conditions.nonfinite = np.arange(len(Qs_array))[~np.isfinite(Qs_array)]
        conditions.finitenonzero = np.arange(len(Qs_array))[np.isfinite(Qs_array) & (Qs_array != Qs_array[names.q_0])]
    return conditions


@pytest.fixture
def Rs():
    np.random.seed(1842)
    ones = [0, -1., 1.]
    rs = [[w, x, y, z] for w in ones for x in ones for y in ones for z in ones][1:]
    rs = rs + [r for r in [quaternionic.array(np.random.uniform(-1, 1, size=4)) for _ in range(20)]]
    return quaternionic.array(rs).normalized
