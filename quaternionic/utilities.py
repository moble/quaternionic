import functools
import numba

jit = functools.partial(numba.njit, cache=True)
guvectorize = functools.partial(numba.guvectorize, nopython=True, cache=True)
