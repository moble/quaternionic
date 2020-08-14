# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

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


def convert_numpy_ufunc_type_to_numba_ftylist(typelist):
    import numba
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars-character-codes
    # https://numba.pydata.org/numba-doc/latest/reference/types.html
    map_numpy_typecode_to_numba_type = {
        '?': numba.boolean,
        'b': numba.byte,
        'h': numba.short,
        'i': numba.intc,
        'l': numba.int_,
        'q': numba.longlong,
        'p': numba.intp,
        # 'B': numba.ubyte,  # numba.ubyte doesn't exist
        'B': numba.char,  # probably closest
        'H': numba.ushort,
        'I': numba.uintc,
        'L': numba.uint,
        'Q': numba.ulonglong,
        'P': numba.uintp,
        # 'e': numba.half,  # numba.half doesn't exist
        # 'f': numba.single,  # numba.single doesn't exist
        'f': numba.float32,  # probably closest
        'd': numba.float_,
        # 'g': numba.longfloat,  # numba.longfloat doesn't exist
        'g': numba.double,  # probably closest
        # 'F': numba.csingle,  # numba.csingle doesn't exist
        'F': numba.complex64,  # probably closest
        # 'D': numba.complex_,  # numba.complex_ doesn't exist
        'D': numba.complex128,  # probably closest
        # 'G': numba.clongfloat,  # numba.clongfloat doesn't exist
    }

    ftylist = []
    for types in typelist:
        in_types, out_type = types.split('->')
        inputs = (map_numpy_typecode_to_numba_type[c] for c in in_types)
        output = map_numpy_typecode_to_numba_type[out_type]
        ftylist.append(output(*inputs))
    return ftylist


def attach_typelist_and_signature(ftylist, signature):
    import functools
    def wrapper(f):
        f.types = ftylist
        f.signature = signature
        return f
    return wrapper


def guvectorize_module_functions(module, obj):
    """Compile all functions in module to ufuncs and attach to obj

    Note that the functions in module must have attributes `types` and
    `signature`, providing the necessary arguments to numba's guvectorize
    decorator; any function that does not have those attributes will be skipped
    silently.

    """
    import types
    for k, v in module.__dict__.items():
        if isinstance(v, types.FunctionType) and hasattr(v, 'types') and hasattr(v, 'signature'):
            v_ufunc = guvectorize(v.types, v.signature)(v)
            setattr(obj, k, v_ufunc)
