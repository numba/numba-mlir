# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from numba.core import types
from numba.core.typing import npydecl
from numba.core.extending import overload
from numba.core.types.npytypes import Array
from numba.np.numpy_support import is_nonelike
from numba.core.typing.templates import signature, AbstractTemplate

from ..target import typing_registry, infer_global


def _get_init_like_impl(init_func, dtype, shape):
    # NumPy uses 'a' as the arg name for the array-like
    if is_nonelike(dtype) and is_nonelike(shape):

        def impl(a, dtype=None, shape=None):
            return init_func(dtype=a.dtype, shape=a.shape)

    elif not is_nonelike(dtype) and is_nonelike(shape):

        def impl(a, dtype=None, shape=None):
            return init_func(dtype=dtype, shape=a.shape)

    elif is_nonelike(dtype) and not is_nonelike(shape):

        def impl(a, dtype=None, shape=None):
            return init_func(dtype=a.dtype, shape=shape)

    elif not is_nonelike(dtype) and not is_nonelike(shape):

        def impl(a, dtype=None, shape=None):
            return init_func(dtype=dtype, shape=shape)

    else:
        assert False, "Unreachable"

    return impl


@overload(np.empty_like)
def np_zeros_like(a, dtype=None, shape=None):
    return _get_init_like_impl(np.empty, dtype, shape)


@overload(np.zeros_like)
def np_zeros_like(a, dtype=None, shape=None):
    return _get_init_like_impl(np.zeros, dtype, shape)


@overload(np.ones_like)
def np_zeros_like(a, dtype=None, shape=None):
    return _get_init_like_impl(np.ones, dtype, shape)


def _remove_infer_global(registry, func):
    funcs = registry.globals
    registry.globals = list(filter(lambda a: a[0] is not func, funcs))


def _replace_global(registry, func, cls):
    # _remove_infer_global(registry, func)
    registry.register_global(func)(cls)


def get_reduction_id(prefer_float):
    class ReductionId(AbstractTemplate):
        prefer_literal = True

        def generic(self, args, kws):
            if len(args) != 1 or not isinstance(args[0], Array):
                return

            arr = args[0]
            axis = kws.get("axis", None)

            if "keepdims" in kws:
                keepdims = kws["keepdims"].literal_value
            else:
                keepdims = False

            if "dtype" in kws:
                dtype = npydecl.parse_dtype(kws["dtype"])
            else:
                dtype = arr.dtype
                if prefer_float and dtype in types.integer_domain:
                    dtype = types.float64

            res_args = args + tuple(kws.values())
            if axis is None:
                if keepdims:
                    ndim = arr.ndim
                else:
                    return signature(dtype, *res_args)
            else:
                if keepdims:
                    ndim = arr.ndim
                else:
                    ndim = arr.ndim - 1

            arr_type = Array(dtype=dtype, ndim=ndim, layout="C")
            return signature(arr_type, *res_args)

    return ReductionId


ReductionId = get_reduction_id(False)
ReductionFloatId = get_reduction_id(True)
for func in [np.sum, np.max, np.min, np.amax, np.amin, np.prod]:
    _replace_global(typing_registry, func, ReductionId)

_replace_global(typing_registry, np.mean, ReductionFloatId)


def get_abstract_template(pattern_func):
    class TemmplateId(AbstractTemplate):
        def generic(self, args, kwargs):
            try:
                a = pattern_func(*args, **kwargs)
            except:
                return

            if isinstance(a, tuple):
                return self.generic_impl(*a)
            else:
                return self.generic_impl(a)

    return TemmplateId


def is_none(arg):
    return arg is None or arg == types.none


def is_type_or_none(arg, typ):
    return is_none(arg) or isinstance(arg, typ)


def _transpose_pattern(a, axes=None):
    return a, axes


@infer_global(np.transpose)
class TransposeId(get_abstract_template(_transpose_pattern)):
    prefer_literal = True

    def generic_impl(self, arr, axes):
        if not isinstance(arr, Array):
            return

        if not is_type_or_none(axes, types.BaseTuple):
            return

        res_type = Array(dtype=arr.dtype, ndim=arr.ndim, layout="C")
        if is_none(axes):
            return signature(res_type, arr)
        else:
            return signature(res_type, arr, axes)


def _dot_pattern(a, b, out=None):
    return a, b, out


@infer_global(np.dot)
class DotId(get_abstract_template(_dot_pattern)):
    prefer_literal = True

    def generic_impl(self, a, b, out):
        if not isinstance(a, Array) or not isinstance(b, Array):
            return

        if not is_type_or_none(out, Array):
            return

        ndims = (a.ndim, b.ndim)

        if not is_none(out):
            dtype = out.dtype
        else:
            # TODO: coerce type
            dtype = a.dtype

        if ndims == (2, 2):
            return_type = Array(dtype, 2, "C")
        elif ndims == (2, 1) or ndims == (1, 2):
            return_type = Array(dtype, 1, "C")
        elif ndims == (1, 1):
            return_type = dtype
        else:
            return

        if out is None:
            return signature(return_type, a, b)
        else:
            return signature(return_type, a, b, out)
