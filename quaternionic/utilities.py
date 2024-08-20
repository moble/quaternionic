# Copyright (c) 2024, Michael Boyle
# See LICENSE file for details:
# <https://github.com/moble/quaternionic/blob/master/LICENSE>

import os
import sys
import functools
import numpy as np

ufunc_attributes = [
    'nin', 'nout', 'nargs', 'ntypes', 'types', 'identity', 'signature',
    'reduce', 'accumulate', 'reduceat', 'outer', 'at'
]


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
        # 'g': numba.longfloat,  # numba.longfloat doesn't exist
        'g': numba.double,  # probably closest
        # 'F': numba.csingle,  # numba.csingle doesn't exist
        'F': numba.complex64,  # probably closest
        # 'D': numba.complex_,  # numba.complex_ doesn't exist
        'D': numba.complex128,  # probably closest
        # 'G': numba.clongfloat,  # numba.clongfloat doesn't exist
    }
    try:  # This is broken on numpy >= 2.0
        map_numpy_typecode_to_numba_type["d"] = numba.float_
    except:
        pass

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
    """Compile all functions in module to ufuncs and attach to obj.

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


def pyguvectorize(types, signature):
    """Test function to convert functions to general universal functions.

    Note that this is almost certainly only useful for functions defined in
    quaternionic.algebra — and specifically only if they have `type` and
    `signature` attributes.  Moreover, the result is a slow python function,
    meant only for testing.

    Nonetheless, the output of this decorator should be callable with the same
    signature as a gufunc.  In this way, tests designed for the compiled
    gufuncs can also be used to test the python-mode code and obtain coverage
    results.

    """
    import functools
    import numpy as np
    inputs, output = signature.split('->')
    inputs = inputs.split(',')
    slice_a = slice(None) if inputs[0]=='()' else 0
    pad_a = np.newaxis if inputs[0]=='()' else slice(None)
    last_dim_a = slice(None, 1) if inputs[0]=='()' else slice(None)
    if len(inputs) > 1:
        slice_b = slice(None) if inputs[1]=='()' else 0
        pad_b = np.newaxis if inputs[1]=='()' else slice(None)
        last_dim_b = slice(None, 1) if inputs[1]=='()' else  slice(None)
    #slice_c = slice(None) if output=='()' else 0
    pad_c = slice(None)# np.newaxis if output=='()' else slice(None)
    last_dim_c = slice(None, 1) if output=='()' else  slice(None)
    last_axis_c = tuple() if output=='()' else (4,)
    dtype_c = np.dtype(types[0][-1].dtype.name)
    if len(inputs) == 1:
        def wrapper(f):
            @functools.wraps(f)
            def f_wrapped(a):
                shape_c = a[..., slice_a].shape + last_axis_c
                c = np.empty(shape_c, dtype=dtype_c)
                if not last_axis_c:
                    c = c[..., np.newaxis]
                a, ctmp = np.broadcast_arrays(a[..., pad_a], c[..., pad_c])
                a = a.reshape((-1, 4))
                ctmp.flags.writeable = True
                ctmp = ctmp.reshape((-1, 4))
                for a_i, c_i in zip(a, ctmp):
                    f(a_i, c_i)
                return c.reshape(shape_c)
            return f_wrapped
    else:
        def wrapper(f):
            @functools.wraps(f)
            def f_wrapped(a, b):
                shape_c = np.broadcast(a[..., slice_a], b[..., slice_b]).shape + last_axis_c
                c = np.empty(shape_c, dtype=dtype_c)
                if not last_axis_c:
                    c = c[..., np.newaxis]
                a, b, ctmp = np.broadcast_arrays(a[..., pad_a], b[..., pad_b], c[..., pad_c])
                a = a.reshape((-1, 4))
                b = b.reshape((-1, 4))
                ctmp = ctmp.reshape((-1, 4))
                ctmp.flags.writeable = True
                for a_i, b_i, c_i in zip(a, b, ctmp):
                    f(a_i[last_dim_a], b_i[last_dim_b], c_i[last_dim_c])
                return c.reshape(shape_c)
            return f_wrapped
    return wrapper


def pyguvectorize_module_functions(module, obj):
    """Wrap all functions in module to ufunc-like python funcs and attach to obj.

    This function is almost exclusively useful for testing purposes.  See
    docstring of pyguvectorize function for details.

    """
    import types
    for k, v in module.__dict__.items():
        if isinstance(v, types.FunctionType) and hasattr(v, 'types') and hasattr(v, 'signature'):
            v_ufunc = pyguvectorize(v.types, v.signature)(v)
            setattr(obj, k, v_ufunc)


if sys.implementation.name.lower() == 'pypy':  # pragma: no cover
    class _FakeNumbaType(object):
        def __init__(self, name):
            self.name = name
        @property
        def dtype(self):
            return self
        def __getitem__(self, *args, **kwargs):
            return self
    float64 = _FakeNumbaType('float64')
    boolean = _FakeNumbaType('boolean')
    jit = lambda f: f
    guvectorize = pyguvectorize
else:
    import numba
    from numba import float64, boolean
    cache = os.environ.get("QUATERNIONIC_DISABLE_CACHE", "0") != "1"
    jit = functools.partial(numba.njit, cache=cache)
    guvectorize = functools.partial(numba.guvectorize, nopython=True, cache=cache)
