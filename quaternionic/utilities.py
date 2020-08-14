import functools
import numba

ufunc_attributes = [
    'nin', 'nout', 'nargs', 'ntypes', 'types', 'identity', 'signature',
    'reduce', 'accumulate', 'reduceat', 'outer', 'at'
]

jit = functools.partial(numba.njit, cache=True)
guvectorize = functools.partial(numba.guvectorize, nopython=True, cache=True)


def type_self_return(f):
    """Decorate jitted functions to return with type of first argument"""
    @functools.wraps(f)
    def f_wrapped(*args, **kwargs):
        return_value = f(*args, **kwargs)
        return type(args[0])(return_value)
    for attr in ufunc_attributes:
        if hasattr(f, attr):
            setattr(f_wrapped, attr, getattr(f, attr))
    return f_wrapped


def ndarray_args(f):
    """Decorate jitted functions to accept quaternionic arrays"""
    @functools.wraps(f)
    def f_wrapped(*args, **kwargs):
        args_ndarray = (arg if not hasattr(arg, 'ndarray') else arg.ndarray for arg in args)
        return f(*args_ndarray, **kwargs)
    for attr in ufunc_attributes:
        if hasattr(f, attr):
            setattr(f_wrapped, attr, getattr(f, attr))
    return f_wrapped


def ndarray_args_and_return(f):
    """Decorate jitted functions to accept and return quaternionic arrays"""
    @functools.wraps(f)
    def f_wrapped(*args, **kwargs):
        args_ndarray = (arg if not hasattr(arg, 'ndarray') else arg.ndarray for arg in args)
        return type(args[0])(f(*args_ndarray, **kwargs))
    for attr in ufunc_attributes:
        if hasattr(f, attr):
            setattr(f_wrapped, attr, getattr(f, attr))
    return f_wrapped
