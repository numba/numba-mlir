# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from numpy import ufunc
from numpy.testing import assert_equal, assert_allclose

try:
    from hypothesis import given, assume, settings, example
except ImportError:

    def _dummy_dec(*args, **kwars):
        def _func(func):
            return func

        return _func

    given = assume = settings = example = _dummy_dec

from numba_mlir.mlir.utils import readenv
from numba_mlir.mlir.numpy.funcs import get_registered_funcs

from .utils import njit_cached as njit


RUN_FUZZ = readenv("NUMBA_MLIR_TESTS_FUZZ", int, 0)


def run_fuzz(func):
    return pytest.mark.skipif(not RUN_FUZZ, reason="Fuzz tests disabled")(func)


def _get_funcs():
    unary_funcs = []
    binary_funcs = []

    funcs = get_registered_funcs()
    for name, func, args in funcs:
        if not isinstance(func, ufunc):
            continue

        nargs = len(args)
        if nargs == 1:
            unary_funcs.append(pytest.param(func, id=name))
        elif nargs == 2:
            binary_funcs.append(pytest.param(func, id=name))

        # print(name, func, args)

    return unary_funcs, binary_funcs


_unary_funcs, _binary_funcs = _get_funcs()


def _array_strat():
    try:
        import hypothesis.strategies as st
        from hypothesis.extra.numpy import (
            arrays,
            scalar_dtypes,
            array_shapes,
            floating_dtypes,
            unsigned_integer_dtypes,
            integer_dtypes,
            complex_number_dtypes,
            boolean_dtypes,
        )
    except ImportError:
        return None

    # TODO: f16, datetime, timedelta
    dtype_strat = st.one_of(
        boolean_dtypes(),
        integer_dtypes(),
        unsigned_integer_dtypes(),
        floating_dtypes(sizes=(32, 64)),
        complex_number_dtypes(),
    )
    return arrays(dtype_strat, array_shapes())


def _get_tol(arr):
    dtype = arr.dtype
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.boolean):
        return 0, 0

    if dtype == np.float16:
        return 1e-3, 1e-3

    return 1e-5, 1e-5


@settings(deadline=None)
@given(arr=_array_strat())
@pytest.mark.parametrize("func", _unary_funcs)
@run_fuzz
def test_unary_ufunc(func, arr):
    def py_func(a):
        return func(a)

    try:
        expected = py_func(arr)
    except:
        assume(False)

    jit_func = njit(py_func)
    got = jit_func(arr)
    rtol, atol = _get_tol(expected)
    assert_allclose(got, expected, rtol=rtol, atol=atol)


@settings(deadline=None)
@given(arr1=_array_strat(), arr2=_array_strat())
@pytest.mark.parametrize("func", _binary_funcs)
@run_fuzz
def test_binary_ufunc(func, arr1, arr2):
    def py_func(a1, a2):
        return func(a1, a2)

    try:
        expected = py_func(arr1, arr2)
    except:
        assume(False)

    jit_func = njit(py_func)
    got = jit_func(arr1, arr2)
    rtol, atol = _get_tol(expected)
    assert_allclose(got, expected, rtol=rtol, atol=atol)
