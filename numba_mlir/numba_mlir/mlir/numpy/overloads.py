# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from numba.np.numpy_support import is_nonelike
from numba.core.extending import overload
from numba.core.typing.templates import signature, AbstractTemplate
from numba.core.typing import npydecl
from numba.core.types.npytypes import Array
from numba.core import types


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
    _remove_infer_global(registry, func)
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
    _replace_global(npydecl.registry, func, ReductionId)

_replace_global(npydecl.registry, np.mean, ReductionFloatId)
