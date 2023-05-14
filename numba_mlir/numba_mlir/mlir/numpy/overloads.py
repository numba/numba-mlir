# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from numba.np.numpy_support import is_nonelike
from numba.core.extending import overload


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
