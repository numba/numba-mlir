# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import copy
import numbers
import pytest
import math
import numpy as np
import types as pytypes
from numpy.testing import assert_equal, assert_allclose

from numba_mlir.mlir.utils import readenv
from numba_mlir import njit, jit, vectorize, njit_replace_parfors

from numba.core.registry import CPUDispatcher
from numba_mlir.mlir.passes import (
    print_pass_ir,
    get_print_buffer,
    is_print_buffer_empty,
)

import numba.tests.test_parfors


def _gen_tests():
    testcases = [
        numba.tests.test_parfors.TestParforBasic,
        numba.tests.test_parfors.TestParforNumericalMisc,
        numba.tests.test_parfors.TestParforNumPy,
        numba.tests.test_parfors.TestParfors,
        numba.tests.test_parfors.TestParforsBitMask,
        # numba.tests.test_parfors.TestParforsDiagnostics,
        numba.tests.test_parfors.TestParforsLeaks,
        numba.tests.test_parfors.TestParforsMisc,
        # numba.tests.test_parfors.TestParforsOptions,
        # numba.tests.test_parfors.TestParforsUnsupported,
        numba.tests.test_parfors.TestParforsSlice,
        numba.tests.test_parfors.TestParforsVectorizer,
        numba.tests.test_parfors.TestPrangeBasic,
        numba.tests.test_parfors.TestPrangeSpecific,
    ]

    xfail_tests = {
        "test_prange25",  # list support
        "test_list_setitem_hoisting",  # list support
        "test_list_comprehension_prange",  # list comprehension support
        "test_prange_raises_invalid_step_size",  # we actually support arbirary step in prange
        "test_parfor_race_1",  # cfg->scf conversion failure
        "test_record_array_setitem_yield_array",  # Record and string support
        "test_record_array_setitem",  # Record and string support
        "test_prange_conflicting_reduction_ops",  # Conflicting reduction reduction check
        "test_ssa_false_reduction",  # Frontend: object has no attribute 'name'
        "test_mutable_list_param",  # List support
        "test_kde_example",  # List suport
        "test_simple01",  # Empty shape not failed
        "test_kmeans",  # List suport
        "test_ndarray_fill",  # array.fill
        "test_fuse_argmin_argmax_max_min",  # numpy argmin, argmax
        "test_arange",  # select issue, complex
        "test_pi",  # np.random.ranf
        "test_simple24",  # getitem with array
        "test_0d_array",  # numpy prod
        "test_argmin",  # numpy.argmin
        "test_argmax",  # numpy.argmax
        "test_ndarray_fill2d",  # array.fill
        "test_simple18",  # np.linalg.svd
        "test_linspace",  # np.linspace
        "test_std",  # array.std
        "test_mvdot",  # np.dot unsupported args
        "test_namedtuple1",  # namedtuple support
        "test_0d_broadcast",  # np.array
        "test_var",  # array.var
        "test_reshape_with_too_many_neg_one",  # unsupported reshape
        "test_namedtuple2",  # namedtuple support
        "test_simple19",  # np.dot unsupported args
        "test_no_hoisting_with_member_function_call",  # set support
        "test_parfor_array_access3",  # TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
        "test_preparfor_canonicalize_kws",  # array.argsort
        "test_parfor_array_access4",  # np.dot unsupported args
        "test_tuple_concat_with_reverse_slice",  # enumerate
        "test_reduce",  # functools.reduce
        "test_tuple_concat",  # enumerate
        "test_two_d_array_reduction_with_float_sizes",  # np.array
        "test_parfor_array_access_lower_slice",  # plier.getitem
        "test_parfor_slice6",  # array.transpose
        "test_parfor_slice22",  # slice using array
        "test_simple13",  # complex128
        "test_issue3169",  # list support
        "test_issue5001",  # list suport
        "test_issue5167",  # np.full
        "test_issue5065",  # tuple unpack
        "test_no_state_change_in_gufunc_lowering_on_error",  # custom pipeline
        "test_namedtuple3",  # namedtuple
        "test_issue6102",  # list support
        "test_oversized_tuple_as_arg_to_kernel",  # UnsupportedParforsError not raised
        "test_parfor_generate_fuse",  # operand #0 does not dominate this use
        "test_parfor_slice7",  # array.transpose
        "test_one_d_array_reduction",  # np.array
        "test_setitem_2d_one_replaced",
        "test_tuple_arg",
        "test_tuple_arg_1d",
        "test_tuple_arg_1d_literal",
        "test_tuple_arg_literal",
        "test_real_imag_attr",
        "test_tuple_arg_not_whole_array",
        "test_tuple_for_pndindex",
        "test_int_arg_pndindex",
        "test_prange_optional",
        "test_1array_control_flow",
        "test_randoms",  # random functions are not supported
        "test_vectorizer_fastmath_asm",  # no asm generated
        "test_signed_vs_unsigned_vec_asm",  # no asm generated
        "test_unsigned_refusal_to_vectorize",  # no asm generated
        "test_prange_fastmath_check_works",  # no asm generated
        "test_issue9256_lower_sroa_conflict",  # using variable outside of parfor
        "test_issue9256_lower_sroa_conflict_variant1",  # using variable outside of parfor
        "test_issue9256_lower_sroa_conflict_variant2",  # using variable outside of parfor
    }

    skip_tests = {
        "test_no_warn_if_cache_set",  # Cache = True is not supported
        "test_three_d_array_reduction",  # We intentionally do not detect data races
        "test_two_d_array_reduction_reuse",  # We intentionally do not detect data races
        "test_two_d_array_reduction",  # We intentionally do not detect data races
        "test_two_d_array_reduction_prod",  # We intentionally do not detect data races
    }

    def countParfors(test_func, args, **kws):
        pytest.skip()

    def countArrays(test_func, args, **kws):
        pytest.skip()

    def countArrayAllocs(test_func, args, **kws):
        pytest.skip()

    def countNonParforArrayAccesses(test_func, args, **kws):
        pytest.skip()

    def get_optimized_numba_ir(test_func, args, **kws):
        pytest.skip()

    def _wrap_test_class(test_base):
        class _Wrapper(test_base):
            def _gen_normal(self, func):
                return njit()(func)

            def _gen_parallel(self, func):
                def wrapper(*args, **kwargs):
                    with print_pass_ir([], ["ParallelToTbbPass"]):
                        res = njit(parallel=True)(func)(*args, **kwargs)
                        ir = get_print_buffer()
                        # Check some parallel loops were actually generated
                        if ir.count("numba_util.parallel") == 0:
                            # In some cases we can canonicalize all loops away
                            # Make sure no loops are present
                            assert ir.count("scf.for") == 0, ir
                            assert ir.count("scf.parallel") == 0, ir
                    return res

                return wrapper

            def _gen_parallel_fastmath(self, func):
                ops = (
                    "fadd",
                    "fsub",
                    "fmul",
                    "fdiv",
                    "frem",
                    "fcmp",
                )

                def wrapper(*args, **kwargs):
                    with print_pass_ir([], ["PostLLVMLowering"]):
                        res = njit(parallel=True, fastmath=True)(func)(*args, **kwargs)
                        ir = get_print_buffer()
                        # Check some fastmath llvm flags were generated
                        opCount = 0
                        fastCount = 0
                        for line in ir.splitlines():
                            for op in ops:
                                if line.count("llvm." + op) > 0:
                                    opCount += 1
                                    if line.count("llvm.fastmath<fast>") > 0:
                                        fastCount += 1
                                    break
                        if opCount > 0:
                            assert fastCount > 0, it
                    return res

                return wrapper

            def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):
                pytest.skip()

            def assertRaises(self, a):
                pytest.skip()

            def prange_tester(self, pyfunc, *args, **kwargs):
                patch_instance = kwargs.pop("patch_instance", None)

                pyfunc = self.generate_prange_func(pyfunc, patch_instance)
                return self._check_impl(pyfunc, *args, **kwargs)

            def check(self, pyfunc, *args, **kwargs):
                if isinstance(pyfunc, CPUDispatcher):
                    pyfunc = pyfunc.py_func

                return self._check_impl(pyfunc, *args, **kwargs)

            def _check_impl(self, pyfunc, *args, **kwargs):
                scheduler_type = kwargs.pop("scheduler_type", None)
                check_fastmath = kwargs.pop("check_fastmath", False)
                check_fastmath_result = kwargs.pop("check_fastmath_result", False)
                check_scheduling = kwargs.pop("check_scheduling", True)
                check_args_for_equality = kwargs.pop("check_arg_equality", None)
                # assert not kwargs, "Unhandled kwargs: " + str(kwargs)

                cfunc = self._gen_normal(pyfunc)
                cpfunc = self._gen_parallel(pyfunc)

                if check_fastmath or check_fastmath_result:
                    fastmath_pcres = self._gen_parallel_fastmath(pyfunc)

                def copy_args(*args):
                    if not args:
                        return tuple()
                    new_args = []
                    for x in args:
                        if isinstance(x, np.ndarray):
                            new_args.append(x.copy("k"))
                        elif isinstance(x, np.number):
                            new_args.append(x.copy())
                        elif isinstance(x, numbers.Number):
                            new_args.append(x)
                        elif isinstance(x, tuple):
                            new_args.append(copy.deepcopy(x))
                        elif isinstance(x, list):
                            new_args.append(x[:])
                        else:
                            raise ValueError("Unsupported argument type encountered")
                    return tuple(new_args)

                # python result
                py_args = copy_args(*args)
                py_expected = pyfunc(*py_args)

                # njit result
                njit_args = copy_args(*args)
                njit_output = cfunc(*njit_args)

                # parfor result
                parfor_args = copy_args(*args)
                parfor_output = cpfunc(*parfor_args)

                if check_args_for_equality is None:
                    np.testing.assert_almost_equal(njit_output, py_expected, **kwargs)
                    np.testing.assert_almost_equal(parfor_output, py_expected, **kwargs)
                    self.assertEqual(type(njit_output), type(parfor_output))
                else:
                    assert len(py_args) == len(check_args_for_equality)
                    for pyarg, njitarg, parforarg, argcomp in zip(
                        py_args, njit_args, parfor_args, check_args_for_equality
                    ):
                        argcomp(njitarg, pyarg, **kwargs)
                        argcomp(parforarg, pyarg, **kwargs)

                # Ignore check_scheduling
                # if check_scheduling:
                #     self.check_scheduling(cpfunc, scheduler_type)

                # if requested check fastmath variant
                if check_fastmath or check_fastmath_result:
                    parfor_fastmath_output = fastmath_pcres(*copy_args(*args))
                    if check_fastmath_result:
                        np.testing.assert_almost_equal(
                            parfor_fastmath_output, py_expected, **kwargs
                        )

        return _Wrapper

    def _replace_global(func, name, newval):
        if name in func.__globals__:
            func.__globals__[name] = newval

    def _gen_test_func(func):
        _replace_global(func, "jit", jit)
        _replace_global(func, "njit", njit)
        _replace_global(func, "vectorize", vectorize)

        _replace_global(func, "countParfors", countParfors)
        _replace_global(func, "countArrays", countArrays)
        _replace_global(func, "countArrayAllocs", countArrayAllocs)
        _replace_global(
            func, "countNonParforArrayAccesses", countNonParforArrayAccesses
        )
        _replace_global(func, "get_optimized_numba_ir", get_optimized_numba_ir)

        def wrapper():
            return func()

        return wrapper

    this_module = sys.modules[__name__]
    for tc in testcases:
        inst = _wrap_test_class(tc)()
        for func_name in dir(tc):
            if func_name.startswith("test"):
                func = getattr(inst, func_name)
                if callable(func):
                    func = _gen_test_func(func)
                    func = pytest.mark.numba_parfor(func)
                    if func_name in xfail_tests:
                        func = pytest.mark.xfail(func)
                    elif func_name in skip_tests:
                        func = pytest.mark.skip(func)

                    setattr(this_module, func_name, func)


_gen_tests()
del _gen_tests


def _gen_replace_parfor_tests():
    testcases = [
        numba.tests.test_parfors.TestParforBasic,
        numba.tests.test_parfors.TestParforNumericalMisc,
        numba.tests.test_parfors.TestParforNumPy,
        numba.tests.test_parfors.TestParfors,
        numba.tests.test_parfors.TestParforsBitMask,
        numba.tests.test_parfors.TestParforsDiagnostics,
        numba.tests.test_parfors.TestParforsLeaks,
        numba.tests.test_parfors.TestParforsMisc,
        numba.tests.test_parfors.TestParforsOptions,
        numba.tests.test_parfors.TestParforsUnsupported,
        numba.tests.test_parfors.TestParforsSlice,
        numba.tests.test_parfors.TestParforsVectorizer,
        numba.tests.test_parfors.TestPrangeBasic,
        numba.tests.test_parfors.TestPrangeSpecific,
    ]

    xfail_tests = {
        "test_simple01",
        "test_argmax",
        "test_argmin",
        "test_simple13",
        "test_namedtuple1",
        "test_kmeans",
        "test_namedtuple2",
        "test_prange_optional",
        "test_std",
        "test_recursive_untraced_value_tuple",
        "test_simple20",
        "test_var",
        "test_pi",
        "test_prange_unknown_call1",
        "test_reduce",
        "test_parfor_bitmask6",
        "test_issue3169",
        "test_untraced_value_tuple",
        "test_max",
        "test_no_hoisting_with_member_function_call",
        "test_fuse_argmin_argmax_max_min",
        "test_parfor_array_access_lower_slice",
        "test_issue6102",
        "test_no_state_change_in_gufunc_lowering_on_error",
        "test_issue5001",
        "test_parfor_slice22",
        "test_issue5065",
        "test_prange25",
        "test_record_array_setitem",
        "test_list_setitem_hoisting",
        "test_list_comprehension_prange",
        "test_parfor_race_1",
        "test_record_array_setitem_yield_array",
        "test_mutable_list_param",
        "test_ssa_false_reduction",
        "test_min",
        "test_nd_parfor",
        "test_arange",
        "test_linspace",
        "test_randoms",  # random functions are not supported
        "test_vectorizer_fastmath_asm",  # no asm generated
        "test_signed_vs_unsigned_vec_asm",  # no asm generated
        "test_unsigned_refusal_to_vectorize",  # no asm generated
        "test_prange_fastmath_check_works",  # no asm generated
        "test_issue9256_lower_sroa_conflict",  # using variable outside of parfor
        "test_issue9256_lower_sroa_conflict_variant1",  # using variable outside of parfor
        "test_issue9256_lower_sroa_conflict_variant2",  # using variable outside of parfor
        "test_allocation_hoisting",  # TODO: investigate
    }
    skip_tests = {
        "test_copy_global_for_parfor",  # flaky test
        "test_prange04",  # flaky test
        "test_prange22",  # flaky test
        "test_no_warn_if_cache_set",  # caching is not supported
        "test_prange07",  # reverse iteration
        "test_prange12",  # reverse iteration
        "test_three_d_array_reduction",  # We intentionally do not detect data races
        "test_two_d_array_reduction_reuse",  # We intentionally do not detect data races
        "test_two_d_array_reduction",  # We intentionally do not detect data races
        "test_two_d_array_reduction_prod",  # We intentionally do not detect data races
        "test_two_d_array_reduction_with_float_sizes",  # We intentionally do not detect data races
    }

    def _wrap_test_class(test_base):
        class _Wrapper(test_base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.has_parallel = None

            def _gen_normal(self, func):
                return njit_replace_parfors(func)

            def _gen_parallel(self, func):
                def wrapper(*args, **kwargs):
                    with print_pass_ir([], ["ParallelToTbbPass"]):
                        res = njit_replace_parfors(parallel=True)(func)(*args, **kwargs)
                        # ir = get_print_buffer()
                        # # Check some parallel loops were actually generated
                        # self.has_parallel = "scf.parallel" in ir
                        # if ir.count("numba_util.parallel") == 0:
                        #     # In some cases we can canonicalize all loops away
                        #     # Make sure no loops are present
                        #     assert ir.count("scf.for") == 0, ir
                        #     assert ir.count("scf.parallel") == 0, ir
                    return res

                return wrapper

            def _gen_parallel_fastmath(self, func):
                ops = (
                    "fadd",
                    "fsub",
                    "fmul",
                    "fdiv",
                    "frem",
                    "fcmp",
                )

                def wrapper(*args, **kwargs):
                    with print_pass_ir([], ["PostLLVMLowering"]):
                        res = njit(parallel=True, fastmath=True)(func)(*args, **kwargs)
                        ir = get_print_buffer()
                        # Check some fastmath llvm flags were generated
                        opCount = 0
                        fastCount = 0
                        for line in ir.splitlines():
                            for op in ops:
                                if line.count("llvm." + op) > 0:
                                    opCount += 1
                                    if line.count("llvm.fastmath<fast>") > 0:
                                        fastCount += 1
                                    break
                        if opCount > 0:
                            assert fastCount > 0, it
                    return res

                return wrapper

            def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):
                pytest.skip()

            # def assertRaises(self, a):
            #     pytest.skip()

            def prange_tester(self, pyfunc, *args, **kwargs):
                patch_instance = kwargs.pop("patch_instance", None)

                pyfunc = self.generate_prange_func(pyfunc, patch_instance)
                return self._check_impl(pyfunc, *args, **kwargs)

            def check(self, pyfunc, *args, **kwargs):
                if isinstance(pyfunc, CPUDispatcher):
                    pyfunc = pyfunc.py_func

                return self._check_impl(pyfunc, *args, **kwargs)

            def _check_impl(self, pyfunc, *args, **kwargs):
                scheduler_type = kwargs.pop("scheduler_type", None)
                check_fastmath = kwargs.pop("check_fastmath", False)
                check_fastmath_result = kwargs.pop("check_fastmath_result", False)
                check_scheduling = kwargs.pop("check_scheduling", True)
                check_args_for_equality = kwargs.pop("check_arg_equality", None)
                # assert not kwargs, "Unhandled kwargs: " + str(kwargs)

                cfunc = self._gen_normal(pyfunc)
                cpfunc = self._gen_parallel(pyfunc)

                if check_fastmath or check_fastmath_result:
                    fastmath_pcres = self._gen_parallel_fastmath(pyfunc)

                def copy_args(*args):
                    if not args:
                        return tuple()
                    new_args = []
                    for x in args:
                        if isinstance(x, np.ndarray):
                            new_args.append(x.copy("k"))
                        elif isinstance(x, np.number):
                            new_args.append(x.copy())
                        elif isinstance(x, numbers.Number):
                            new_args.append(x)
                        elif isinstance(x, tuple):
                            new_args.append(copy.deepcopy(x))
                        elif isinstance(x, list):
                            new_args.append(x[:])
                        else:
                            raise ValueError("Unsupported argument type encountered")
                    return tuple(new_args)

                # python result
                py_args = copy_args(*args)
                py_expected = pyfunc(*py_args)

                # njit result
                njit_args = copy_args(*args)
                njit_output = cfunc(*njit_args)

                # parfor result
                parfor_args = copy_args(*args)
                parfor_output = cpfunc(*parfor_args)

                if check_args_for_equality is None:
                    np.testing.assert_almost_equal(njit_output, py_expected, **kwargs)
                    np.testing.assert_almost_equal(parfor_output, py_expected, **kwargs)
                    self.assertEqual(type(njit_output), type(parfor_output))
                else:
                    assert len(py_args) == len(check_args_for_equality)
                    for pyarg, njitarg, parforarg, argcomp in zip(
                        py_args, njit_args, parfor_args, check_args_for_equality
                    ):
                        argcomp(njitarg, pyarg, **kwargs)
                        argcomp(parforarg, pyarg, **kwargs)

                # Mimic original error
                # if check_scheduling:
                #     # self.check_scheduling(cpfunc, scheduler_type)
                #     assert self.has_parallel, "'@do_scheduling' not found"

                # if requested check fastmath variant
                if check_fastmath or check_fastmath_result:
                    parfor_fastmath_output = fastmath_pcres(*copy_args(*args))
                    if check_fastmath_result:
                        np.testing.assert_almost_equal(
                            parfor_fastmath_output, py_expected, **kwargs
                        )

        return _Wrapper

    def _replace_global(func, name, newval):
        if name in func.__globals__:
            func.__globals__[name] = newval

    def _njit_wrapper(*args, **kwargs):
        return njit_replace_parfors(*args, **kwargs)

    def _gen_test_func(func):
        _replace_global(func, "njit", _njit_wrapper)

        def wrapper():
            return func()

        return wrapper

    this_module = sys.modules[__name__]
    for tc in testcases:
        inst = _wrap_test_class(tc)()
        for func_name in dir(tc):
            if func_name.startswith("test"):
                func = getattr(inst, func_name)
                if callable(func):
                    func = _gen_test_func(func)
                    func = pytest.mark.numba_parfor(func)
                    if func_name in xfail_tests:
                        func = pytest.mark.xfail(func)
                    elif func_name in skip_tests:
                        func = pytest.mark.skip(func)

                    setattr(this_module, "test_replace_parfors_" + func_name, func)


_gen_replace_parfor_tests()
del _gen_replace_parfor_tests


def test_replace_parfor_numpy():
    def py_func(a, b):
        return np.add(a, b)

    a = np.arange(10)
    b = np.arange(10, 20)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a, b), jit_func(a, b))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_numpy_multidim():
    def py_func():
        return np.ones((2, 3, 4))

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(), jit_func())
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_numpy_tuple():
    def py_func():
        return np.ones((10, 10)) + 1.0

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(), jit_func())
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_numpy_reduction():
    def py_func(a, b):
        return np.sum(a + b)

    shape = (3, 4)
    count = math.prod(shape)

    a = np.arange(count).reshape(shape)
    b = np.arange(count, count + count).reshape(shape)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a, b), jit_func(a, b))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_dot():
    def py_func(a, b):
        return np.dot(a, b)

    a = np.linspace(0, 1, 20).reshape(2, 10)
    b = np.linspace(2, 1, 10)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_allclose(py_func(a, b), jit_func(a, b), rtol=1e-4, atol=1e-7)
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_numpy_operator():
    def py_func(a, b):
        return a + b

    a = np.arange(10)
    b = np.arange(10, 20)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a, b), jit_func(a, b))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_slice_capture():
    def py_func(a):
        a += 1
        a[:] = 3

    a = np.ones(1)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a), jit_func(a))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_prange():
    def py_func(c):
        for i in numba.prange(len(c)):
            c[i] = i * i

    a = np.empty(10)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a), jit_func(a))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_2prange():
    def py_func(a, b, c, d):
        for i in numba.prange(n):
            c[i] = a[i] + b[i]
        for i in numba.prange(n):
            d[i] = a[i] - b[i]
        return

    n = 10
    a = np.arange(n, dtype=np.float32) * 2
    b = np.arange(n, dtype=np.float32)
    c1 = np.zeros((n), dtype=np.float32)
    d1 = np.zeros((n), dtype=np.float32)
    c2 = np.zeros_like(c1)
    d2 = np.zeros_like(d1)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        py_func(a, b, c1, d1)
        jit_func(a, b, c2, d2)
        assert_equal(c1, c2)
        assert_equal(d1, d2)
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_2prange_red():
    def py_func(a, b, c, d):
        res = 0
        for i in numba.prange(n):
            c[i] = a[i] + b[i]
        for i in numba.prange(n):
            d[i] = a[i] - b[i]
        for i in numba.prange(n):
            res += a[i] * b[i]
        return

    n = 10
    a = np.arange(n, dtype=np.float32) * 2
    b = np.arange(n, dtype=np.float32)
    c1 = np.zeros((n), dtype=np.float32)
    d1 = np.zeros((n), dtype=np.float32)
    c2 = np.zeros_like(c1)
    d2 = np.zeros_like(d1)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a, b, c1, d1), jit_func(a, b, c2, d2))
        assert_equal(c1, c2)
        assert_equal(d1, d2)
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_prange_nested():
    def py_func(c):
        for i in numba.prange(c.shape[0]):
            for j in numba.prange(c.shape[1]):
                c[i, j] = i * j

    a = np.empty((3, 4))
    b = a.copy()

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        py_func(a)
        jit_func(b)
        assert_equal(a, b)
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_prange_nested_reduction():
    def py_func(c):
        acc = 0
        for i in numba.prange(c.shape[0]):
            for j in numba.prange(c.shape[1]):
                acc += c[i, j]
        return acc

    a = np.arange(3 * 4).reshape(3, 4)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a), jit_func(a))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_prange_reduction1():
    def py_func(c):
        res = 0
        for i in numba.prange(len(c)):
            # ind = 2 if i == 4 else i
            # res = res + c[ind]
            res = res + c[i]
        return res

    a = np.arange(10)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a), jit_func(a))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


def test_replace_parfor_prange_reduction2():
    def py_func(c):
        res1 = 0
        res2 = 1
        for i in numba.prange(len(c)):
            res1 = res1 + c[i]
            res2 = res2 * c[i]
        return res1, res2

    a = np.arange(10)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(a), jit_func(a))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


@pytest.mark.skip(reason="reverse iteration is not supported yet")
def test_replace_parfor_prange_reverse_iter():
    def py_func(A):
        s = 0
        for i in numba.prange(4, 1):
            s += A[i]
        return s

    jit_func = njit_replace_parfors(py_func, parallel=True)

    A = np.ones((4), dtype=np.float64)

    with print_pass_ir([], ["CFGToSCFPass"]):
        assert_equal(py_func(A), jit_func(A))
        ir = get_print_buffer()
        assert len(ir) > 0  # Check some code was actually generated


@pytest.mark.skip(reason="for debugging purposes only!")
def test_replace_parfor_dpnp():
    import numba_dpex
    import dpnp

    def py_func(c):
        res = 0
        for i in numba.prange(len(c)):
            # ind = 2 if i == 4 else i
            # res = res + c[ind]
            res = res + c[i]
        return res

    a = dpnp.arange(10)

    jit_func = njit_replace_parfors(py_func, parallel=True)
    assert_equal(py_func(a), jit_func(a))
