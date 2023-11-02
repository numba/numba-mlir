# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numba_mlir import njit
import inspect
import pytest


def parametrize_function_variants(name, strings):
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    g = vars(caller_module)
    funcs = [eval(f, g) for f in strings]
    return pytest.mark.parametrize(name, funcs, ids=strings)


def code_or_obj(func):
    try:
        return func.__code__
    except:
        return func


def del_prev_items(cache, max_size):
    while len(cache) >= max_size:
        del cache[next(iter(cache))]


class JitfuncCache:
    def __init__(self, decorator):
        self._cached_funcs = {}
        self._decorator = decorator

    def cached_decorator(self, func, *args, **kwargs):
        if args or kwargs:
            return self._decorator(func, *args, **kwargs)

        cached = self._cached_funcs.get(func)
        if cached is not None:
            # Move to front
            del self._cached_funcs[func]
            self._cached_funcs[func] = cached
            return cached

        jitted = self._decorator(func)
        del_prev_items(self._cached_funcs, 64)

        self._cached_funcs[func] = jitted
        return jitted


njit_cache = JitfuncCache(njit)
njit_cached = njit_cache.cached_decorator
