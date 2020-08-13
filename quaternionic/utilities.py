import functools
import numba

jit = functools.partial(numba.njit, cache=True)
guvectorize = functools.partial(numba.guvectorize, nopython=True, cache=True)


def type_self_return(f):
    """Decorate jitted functions to return with type of first argument"""
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return_value = f(*args, **kwargs)
        return type(args[0])(return_value)
    return wrapped
