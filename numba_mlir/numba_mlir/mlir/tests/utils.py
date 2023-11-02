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


class _Wrapper:
    def __init__(self, value):
        self._value = value
        self._hash = hash(value)

    def __eq__(self, obj):
        return self._value is obj

    def __hash__(self):
        return self._hash


def _get_closure_key(closure):
    if closure:
        return tuple(map(lambda a: _Wrapper(a.cell_contents), closure))

    return closure


def _get_key(func):
    try:
        return (func.__code__,) + _get_closure_key(func.__closure__)
    except:
        return func


def _del_prev_items(cache, max_size):
    while len(cache) >= max_size:
        del cache[next(iter(cache))]


class JitfuncCache:
    def __init__(self, decorator):
        self._cached_funcs = {}
        self._decorator = decorator

    def cached_decorator(self, func, *args, **kwargs):
        if args or kwargs:
            return self._decorator(func, *args, **kwargs)

        key = _get_key(func)
        cached = self._cached_funcs.get(key)
        if cached is not None:
            # Move to front
            del self._cached_funcs[key]
            self._cached_funcs[key] = cached
            return cached

        _del_prev_items(self._cached_funcs, 64)
        jitted = self._decorator(func)

        self._cached_funcs[key] = jitted
        return jitted


njit_cache = JitfuncCache(njit)
njit_cached = njit_cache.cached_decorator
