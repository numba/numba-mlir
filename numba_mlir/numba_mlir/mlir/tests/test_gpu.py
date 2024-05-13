# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from numpy.testing import assert_equal, assert_allclose
import numpy as np
import math
import numba
import itertools
import re

from numba_mlir.mlir.dpctl_interop import get_default_device
from numba_mlir.mlir.kernel_impl import Kernel
from numba_mlir.mlir.utils import readenv
from numba_mlir.kernel import *
from numba_mlir.mlir.passes import (
    print_pass_ir,
    get_print_buffer,
    is_print_buffer_empty,
)

from .utils import JitfuncCache, parametrize_function_variants
from numba_mlir import njit as njit_orig

_def_device = get_default_device().filter_string
_has_fp64 = get_default_device().has_fp64
_fp64_dtypes = {np.float64, np.complex128}

require_f64 = pytest.mark.skipif(not _has_fp64, reason="Need f64 support")


def skip_fp64_arg(arg):
    if _has_fp64:
        return arg

    mark = lambda a: pytest.param(
        a, marks=pytest.mark.skip(reason="fp64 doesn't supported")
    )
    if arg in _fp64_dtypes or (hasattr(arg, "dtype") and arg.dtype in _fp64_dtypes):
        return mark(arg)

    return arg


def skip_fp64_args(args):
    assert isinstance(args, list)
    return [skip_fp64_arg(a) for a in args]


def _to_device_kernel_args(args):
    import dpctl.tensor as dpt

    if isinstance(args, tuple):
        return tuple(_to_device_kernel_args(a) for a in args)

    elif isinstance(args, np.ndarray):
        if args.flags["C_CONTIGUOUS"]:
            order = "C"
        elif args.flags["F_CONTIGUOUS"]:
            order = "F"
        else:
            order = "K"
        return dpt.asarray(
            args,
            dtype=args.dtype,
            device=_def_device,
            copy=None,
            usm_type=None,
            sycl_queue=None,
            order=order,
        )

    return args


def _from_device_kernel_args(orig_args, args):
    import dpctl.tensor as dpt

    if isinstance(orig_args, tuple):
        assert isinstance(args, tuple)
        for a, b in zip(orig_args, args):
            _from_device_kernel_args(a, b)

    elif isinstance(orig_args, np.ndarray):
        np.copyto(orig_args, dpt.asnumpy(args))


class LegacyNumpyKernel(Kernel):
    def __call__(self, *args, **kwargs):
        res = self.check_call_args(args, kwargs)
        new_args = _to_device_kernel_args(args)
        res = super().__call__(*new_args, **kwargs)
        _from_device_kernel_args(args, new_args)
        return res


def kernel(func=None, **kwargs):
    if func is None:

        def wrapper(f):
            return LegacyNumpyKernel(f, kwargs)

        return wrapper
    return LegacyNumpyKernel(func, kwargs)


FP64_TRUNCATE = readenv("NUMBA_MLIR_TESTS_FP64_TRUNCATE", str, "")


def kernel_wrapper(*args, **kwargs):
    if FP64_TRUNCATE:
        if FP64_TRUNCATE == "auto":
            fp64_truncate = "auto"
        else:
            fp64_truncate = bool(FP64_TRUNCATE)

        kwargs["gpu_fp64_truncate"] = fp64_truncate

    return kernel(*args, **kwargs)


kernel_cache = JitfuncCache(kernel_wrapper)
kernel_cached = kernel_cache.cached_decorator


def njit_wrapper(*args, **kwargs):
    if FP64_TRUNCATE:
        if FP64_TRUNCATE == "auto":
            fp64_truncate = "auto"
        else:
            fp64_truncate = bool(FP64_TRUNCATE)

        kwargs["gpu_fp64_truncate"] = fp64_truncate

    return njit_orig(*args, **kwargs)


njit_cache = JitfuncCache(njit_wrapper)
njit = njit_cache.cached_decorator

GPU_TESTS_ENABLED = readenv("NUMBA_MLIR_ENABLE_GPU_TESTS", int, 0)

if GPU_TESTS_ENABLED:
    import dpctl
    import dpctl.tensor as dpt


def require_gpu(func):
    func = pytest.mark.test_gpu(func)
    return pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="GPU tests disabled")(func)


_test_values = [
    True,
    False,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    -2.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    2.5,
]


@pytest.mark.smoke
@require_gpu
def test_simple1():
    def func(a, b, c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    b = np.array([[[7, 8, 9], [10, 11, 12]]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_gpu
def test_simple2():
    get_id = get_global_id

    def func(a, b, c):
        i = get_id(0)
        j = get_id(1)
        k = get_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    b = np.array([[[7, 8, 9], [10, 11, 12]]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_simple3():
    def func(a, b):
        i = get_global_id(0)
        b[i, 0] = a[i, 0]
        b[i, 1] = a[i, 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[1, 2], [3, 4], [5, 6]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape[0], DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape[0], DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_gpu
@pytest.mark.skip(reason="Crashes SYCL OpenCL backend")
@pytest.mark.parametrize("dtype", skip_fp64_args([np.complex64, np.complex128]))
def test_complex(dtype):
    def func(c):
        i = get_global_id(0)
        j = get_global_id(1)
        c[i, j] = i + 1j * j

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    shape = (17, 27)

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, DEFAULT_LOCAL_SIZE](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, DEFAULT_LOCAL_SIZE](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("val", _test_values)
@pytest.mark.parametrize(
    "dtype", skip_fp64_args([np.int32, np.int64, np.float32, np.float64])
)
def test_scalar(val, dtype):
    get_id = get_global_id

    def func(a, b, c):
        i = get_id(0)
        c[i] = a[i] + b

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(-6, 6, dtype=dtype)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, val, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, val, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("val", _test_values)
@pytest.mark.parametrize(
    "dtype", skip_fp64_args([np.int32, np.int64, np.float32, np.float64])
)
def test_scalar_cature(val, dtype):
    get_id = get_global_id

    def func(a, c):
        i = get_id(0)
        c[i] = a[i] + val

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(-6, 6, dtype=dtype)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_empty_kernel():
    def func(a):
        pass

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 0, ir


@require_gpu
@require_f64
def test_f64_truncate():
    def func(a, b, c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func1 = kernel(gpu_fp64_truncate=True)(func)
    gpu_func2 = kernel(gpu_fp64_truncate=True)(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float64)
    b = np.array([[[7, 8, 9], [10, 11, 12]]], np.float64)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res1 = np.zeros(a.shape, a.dtype)
    gpu_res2 = np.zeros(a.shape, a.dtype)

    pattern = "arith.addf %[0-9a-zA-Z]+, %[0-9a-zA-Z]+ : f32"

    with print_pass_ir(["TruncateF64ForGPUPass"], []):
        gpu_func1[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res1)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    with print_pass_ir([], ["TruncateF64ForGPUPass"]):
        gpu_func2[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res2)
        ir = get_print_buffer()
        assert re.search(pattern, ir), ir

    assert_equal(gpu_res1, sim_res)
    assert_equal(gpu_res2, sim_res)


@require_gpu
def test_list_args():
    def func(a, b, c):
        i = get_global_id(0)
        c[i] = a[i] + b[i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1, 2, 3, 4, 5, 6], np.float32)
    b = np.array([7, 8, 9, 10, 11, 12], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)

    dims = [a.shape[0]]
    sim_func[dims, []](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[dims, []](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_slice():
    def func(a, b):
        i = get_global_id(0)
        b1 = b[i]
        j = get_global_id(1)
        b2 = b1[j]
        k = get_global_id(2)
        b2[k] = a[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(3 * 4 * 5).reshape((3, 4, 5))

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_inner_loop():
    def func(a, b, c):
        i = get_global_id(0)
        res = 0.0
        for j in range(a[i]):
            res = res + b[j]
        c[i] = res

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1, 2, 3, 4], np.int32)
    b = np.array([5, 6, 7, 8, 9], np.float32)

    sim_res = np.zeros(a.shape, b.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros(a.shape, b.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


def _test_unary(func, dtype, ir_pass, ir_check):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype)

    sim_res = np.zeros(a.shape, dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros(a.shape, dtype)

    with print_pass_ir([], [ir_pass]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert ir_check(ir), ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)


def _test_binary(func, dtype, ir_pass, ir_check):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([11, 12, 13, 14, 15], dtype)
    b = np.array([1, 2, 3, 4, 5], dtype)

    sim_res = np.zeros(a.shape, dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros(a.shape, dtype)

    with print_pass_ir([], [ir_pass]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir_check(ir), ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)


_math_unary_funcs = [
    "sqrt",
    "log",
    "exp",
    "sin",
    "cos",
    "erf",
    "tanh",
    "floor",
    "ceil",
    "acos",
]


@require_gpu
@pytest.mark.parametrize("op", _math_unary_funcs)
@pytest.mark.parametrize("dtype", skip_fp64_args([np.float32, np.float64]))
def test_math_funcs_unary(op, dtype):
    f = eval(f"math.{op}")

    def func(a, b):
        i = get_global_id(0)
        b[i] = f(a[i])

    _test_unary(func, dtype, "GPUToSpirvPass", lambda ir: ir.count(f"CL.{op}") == 1)


@require_gpu
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "//", "%", "**"])
@pytest.mark.parametrize("dtype", skip_fp64_args([np.int32, np.float32, np.float64]))
def test_gpu_ops_binary(op, dtype):
    f = eval(f"lambda a, b: a {op} b")
    inner = kernel_func(f)

    def func(a, b, c):
        i = get_global_id(0)
        c[i] = inner(a[i], b[i])

    _test_binary(
        func,
        dtype,
        "ConvertParallelLoopToGpu",
        lambda ir: ir.count(f"gpu.launch blocks") == 1,
    )


_test_shapes = [
    (1,),
    (7,),
    (1, 1),
    (7, 13),
    (1, 1, 1),
    (7, 13, 23),
]


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
def test_get_global_id(shape):
    def func1(c):
        i = get_global_id(0)
        c[i] = i

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        c[i, j] = i + j * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = i + j * 100 + k * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, DEFAULT_LOCAL_SIZE](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, DEFAULT_LOCAL_SIZE](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
@pytest.mark.parametrize("lsize", [(1, 1, 1), (2, 4, 8)])
def test_get_local_id(shape, lsize):
    def func1(c):
        i = get_global_id(0)
        li = get_local_id(0)
        c[i] = li

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        li = get_local_id(0)
        lj = get_local_id(1)
        c[i, j] = li + lj * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        li = get_local_id(0)
        lj = get_local_id(1)
        lk = get_local_id(2)
        c[i, j, k] = li + lj * 100 + lk * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    if len(lsize) > len(shape):
        lsize = tuple(lsize[: len(shape)])

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, lsize](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, lsize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
@pytest.mark.parametrize("lsize", [(1, 1, 1), (2, 4, 8)])
def test_get_group_id(shape, lsize):
    def func1(c):
        i = get_global_id(0)
        li = get_group_id(0)
        c[i] = li

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        li = get_group_id(0)
        lj = get_group_id(1)
        c[i, j] = li + lj * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        li = get_group_id(0)
        lj = get_group_id(1)
        lk = get_group_id(2)
        c[i, j, k] = li + lj * 100 + lk * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    if len(lsize) > len(shape):
        lsize = tuple(lsize[: len(shape)])

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, lsize](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, lsize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
def test_get_global_size(shape):
    def func1(c):
        i = get_global_id(0)
        w = get_global_size(0)
        c[i] = w

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        w = get_global_size(0)
        h = get_global_size(1)
        c[i, j] = w + h * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        w = get_global_size(0)
        h = get_global_size(1)
        d = get_global_size(2)
        c[i, j, k] = w + h * 100 + d * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, DEFAULT_LOCAL_SIZE](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, DEFAULT_LOCAL_SIZE](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
@pytest.mark.parametrize("lsize", [(1, 1, 1), (2, 4, 8)])
def test_get_local_size(shape, lsize):
    def func1(c):
        i = get_global_id(0)
        w = get_local_size(0)
        c[i] = w

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        w = get_local_size(0)
        h = get_local_size(1)
        c[i, j] = w + h * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        w = get_local_size(0)
        h = get_local_size(1)
        d = get_local_size(2)
        c[i, j, k] = w + h * 100 + d * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    if len(lsize) > len(shape):
        lsize = tuple(lsize[: len(shape)])

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, lsize](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, lsize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


_atomic_dtypes = skip_fp64_args(["int32", "int64", "float32"])
_atomic_funcs = [atomic.add, atomic.sub]


def _check_atomic_ir(ir):
    return (
        ir.count("spirv.AtomicIAdd") == 1
        or ir.count("spirv.AtomicISub") == 1
        or ir.count("spirv.EXT.AtomicFAdd") == 1
    )


def _test_atomic(func, dtype, ret_size):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype)

    sim_res = np.zeros([ret_size], dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros([ret_size], dtype)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert _check_atomic_ir(ir), ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("dtype", _atomic_dtypes)
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics(dtype, atomic_op):
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, a[i])

    _test_atomic(func, dtype, 1)


@require_gpu
@pytest.mark.xfail(reason="Only direct func calls work for now")
def test_atomics_modname():
    def func(a, b):
        i = get_global_id(0)
        atomic.add(b, 0, a[i])

    _test_atomic(func, "int32", 1)


@require_gpu
@pytest.mark.parametrize("dtype", _atomic_dtypes)
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_offset(dtype, atomic_op):
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, i % 2, a[i])

    _test_atomic(func, dtype, 2)


@require_gpu
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_different_types1(atomic_op):
    dtype = "int32"

    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, a[i] + 1)

    _test_atomic(func, dtype, 1)


@require_gpu
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_different_types2(atomic_op):
    dtype = "int32"

    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, 1)

    _test_atomic(func, dtype, 1)


@require_gpu
@pytest.mark.parametrize("funci", [1, 2])
def test_atomics_multidim(funci):
    atomic_op = atomic.add
    dtype = "int32"

    def func1(a, b):
        i = get_global_id(0)
        j = get_global_id(1)
        atomic_op(b, (i % 2, 0), a[i, j])

    def func2(a, b):
        i = get_global_id(0)
        j = get_global_id(1)
        atomic_op(b, (i % 2, j % 2), a[i, j])

    func = func1 if funci == 1 else func2

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype)

    sim_res = np.zeros((2, 2), dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros((2, 2), dtype)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert _check_atomic_ir(ir), ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_atomics_different_dims():
    atomic_op = atomic.add
    dtype = "int32"

    def func(a, b, c):
        i = get_global_id(0)
        j = get_global_id(1)
        atomic_op(a, (i, j), c[i, j])
        atomic_op(b, (i), c[i, j])

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    c = np.array([[1, 2, 3], [4, 5, 6]], dtype)
    a_sim = np.zeros(c.shape, dtype)
    a_gpu = np.zeros(c.shape, dtype)

    b_sim = np.zeros(c.shape[0], dtype)
    b_gpu = np.zeros(c.shape[0], dtype)

    sim_func[c.shape, DEFAULT_LOCAL_SIZE](a_sim, b_sim, c)
    gpu_func[c.shape, DEFAULT_LOCAL_SIZE](a_gpu, b_gpu, c)

    assert_equal(a_gpu, a_sim)
    assert_equal(b_gpu, b_sim)


@pytest.mark.skip(reason="Fails on CI, investigate")
@require_gpu
@pytest.mark.parametrize(
    "s", [slice(1, None, 3), slice(1, None, -2), slice(1, 8, None)]
)
def test_kernel_slice_arg(s):
    def func(s, res):
        i = get_global_id(0)
        res[s][i] = i

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    res = np.zeros(100, dtype=np.int32)
    N = len(res[s])

    sim_res = res.copy()
    gpu_res = res.copy()
    sim_func[N, DEFAULT_LOCAL_SIZE](s, sim_res)
    gpu_func[N, DEFAULT_LOCAL_SIZE](s, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_gpu
def test_fastmath():
    def func(a, b, c, res):
        i = get_global_id(0)
        res[i] = a[i] * b[i] + c[i]

    sim_func = kernel_sim(func)
    a = np.array([1, 2, 3, 4], np.float32)
    b = np.array([5, 6, 7, 8], np.float32)
    c = np.array([9, 10, 11, 12], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, c, sim_res)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_res = np.zeros(a.shape, a.dtype)
        gpu_func = kernel(fastmath=False)(func)
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, c, gpu_res)
        ir = get_print_buffer()
        assert ir.count("spirv.CL.fma") == 0, ir
        assert_equal(gpu_res, sim_res)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_res = np.zeros(a.shape, a.dtype)
        gpu_func = kernel(fastmath=True)(func)
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, c, gpu_res)
        ir = get_print_buffer()
        assert ir.count("spirv.CL.fma") == 1, ir
        assert_equal(gpu_res, sim_res)


@require_gpu
def test_fastmath_nested_range():
    def func(a, b, res):
        i = get_global_id(0)
        d = np.float32(0)
        for j in range(10):
            d += a[i] * b[i]
        res[i] = d

    sim_func = kernel_sim(func)
    a = np.array([1, 2, 3, 4], np.float32)
    b = np.array([5, 6, 7, 8], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_res = np.zeros(a.shape, a.dtype)
        gpu_func = kernel(fastmath=False)(func)
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("spirv.CL.fma") == 0, ir
        assert_equal(gpu_res, sim_res)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_res = np.zeros(a.shape, a.dtype)
        gpu_func = kernel(fastmath=True)(func)
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("spirv.CL.fma") == 1, ir
        assert_equal(gpu_res, sim_res)


@require_gpu
def test_input_load_cse():
    def func(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = i + 10 * j + 100 * k

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir(["SerializeSPIRVPass"], []):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](gpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                'spirv.Load "Input" %__builtin__GlobalInvocationId___addr : vector<3xi64>'
            )
            == 1
        ), ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("op", [barrier, mem_fence])
@pytest.mark.parametrize("flags", [LOCAL_MEM_FENCE, GLOBAL_MEM_FENCE])
@pytest.mark.parametrize("global_size", [1, 2, 27, 67, 101])
@pytest.mark.parametrize("local_size", [1, 2, 7, 17, 33])
def test_barrier_ops(op, flags, global_size, local_size):
    atomic_add = atomic.add

    def func(a, b):
        i = get_global_id(0)
        v = a[i]
        op(flags)
        b[i] = a[i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(global_size, dtype=np.int64)

    sim_res = np.zeros(global_size, a.dtype)
    sim_func[global_size, local_size](a, sim_res)

    gpu_res = np.zeros(global_size, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[global_size, local_size](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("global_size", [1, 2, 4, 27, 67, 101])
@pytest.mark.parametrize("local_size", [1, 2, 7, 17, 33])
def test_barrier1(global_size, local_size):
    atomic_add = atomic.add

    def func(a, b):
        i = get_global_id(0)
        off = i // local_size
        atomic_add(a, off, i)
        barrier(GLOBAL_MEM_FENCE)
        b[i] = a[off]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    count = (global_size + local_size - 1) // local_size
    a = np.array([0] * count, np.int64)

    sim_res = np.zeros(global_size, a.dtype)
    sim_func[global_size, local_size](a.copy(), sim_res)

    gpu_res = np.zeros(global_size, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[global_size, local_size](a.copy(), gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
def test_local_memory1(blocksize):
    local_array = local.array

    def func(A):
        lm = local_array(shape=blocksize, dtype=np.float32)
        i = get_global_id(0)

        # preload
        lm[i] = A[i]
        # barrier local or global will both work as we only have one work group
        barrier(LOCAL_MEM_FENCE)  # local mem fence
        # write
        A[i] += lm[blocksize - 1 - i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.arange(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
def test_local_memory2(blocksize):
    local_array = local.array

    def func(A):
        lm = local_array(shape=(1, blocksize), dtype=np.float32)
        i = get_global_id(0)

        # preload
        lm[0, i] = A[i]
        # barrier local or global will both work as we only have one work group
        barrier(LOCAL_MEM_FENCE)  # local mem fence
        # write
        A[i] += lm[0, blocksize - 1 - i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.arange(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
def test_private_memory1(blocksize):
    private_array = private.array

    def func(A):
        i = get_global_id(0)
        prvt_mem = private_array(shape=1, dtype=np.float32)
        prvt_mem[0] = i
        barrier(LOCAL_MEM_FENCE)  # local mem fence
        A[i] = prvt_mem[0] * 2

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.zeros(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
def test_private_memory2(blocksize):
    private_array = private.array

    def func(A):
        i = get_global_id(0)
        prvt_mem = private_array(shape=(1, 1), dtype=np.float32)
        prvt_mem[0, 0] = i
        barrier(LOCAL_MEM_FENCE)  # local mem fence
        A[i] = prvt_mem[0, 0] * 2

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.zeros(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
@pytest.mark.parametrize("priv_size", [1, 11, 27])
def test_private_memory3(blocksize, priv_size):
    private_array = private.array

    def func(A):
        i = get_global_id(0)
        prvt_mem = private_array(shape=priv_size, dtype=np.float32)
        for j in range(priv_size):
            prvt_mem[j] = j + i

        res = 0
        for j in range(priv_size):
            res += prvt_mem[(j + i) % priv_size]

        A[i] = res

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.zeros(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
def test_private_memory4(blocksize):
    private_array = private.array

    def func(A):
        i = get_global_id(0)
        S = 10
        prvt_mem = private_array(shape=(S, S), dtype=np.float32)
        for j in range(S):
            for k in range(S):
                prvt_mem[j, k] = i

        barrier(LOCAL_MEM_FENCE)  # local mem fence
        A[i] = prvt_mem[i % S, i % S] * 2

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.zeros(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
@pytest.mark.xfail(reason="Type inference issue for private mem")
def test_private_memory5(blocksize):
    private_array = private.array

    def func(A):
        i = get_global_id(0)
        S = 10
        prvt_mem = private_array(shape=(S, S), dtype=np.float32)
        prvt_mem[:] = i
        barrier(LOCAL_MEM_FENCE)  # local mem fence
        A[i] = prvt_mem[i % S, i % S] * 2

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.zeros(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize(
    "group_op", [group.reduce_add, group.reduce_mul, group.reduce_min, group.reduce_max]
)
@pytest.mark.parametrize("global_size", [1, 2, 4, 27, 67, 101])
@pytest.mark.parametrize("local_size", [1, 2, 7, 17, 33])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
def test_group_func(group_op, global_size, local_size, dtype):
    if global_size > 25 and group_op is group.reduce_mul and dtype is np.int64:
        # TODO: investigate overflow handling by spirv group ops
        pytest.skip()

    def func(a, b):
        i = get_global_id(0)
        v = group_op(a[i])
        b[i] = v

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(global_size, dtype=dtype)

    sim_res = np.zeros(global_size, a.dtype)
    sim_func[(global_size,), (local_size,)](a, sim_res)

    gpu_res = np.zeros(global_size, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[(global_size,), (local_size,)](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)


@require_gpu
def test_pairwise1():
    def func(X1, X2, D):
        i = get_global_id(0)
        j = get_global_id(1)

        X1_cols = X1.shape[1]

        d = 0.0
        for k in range(X1_cols):
            tmp = X1[i, k] - X2[j, k]
            d += tmp * tmp
        D[i, j] = np.sqrt(d)

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dims = 3
    npoints = 128
    dtype = np.float32

    a = np.arange(npoints * dims, dtype=dtype).reshape(npoints, dims)
    b = a + 5

    sim_res = np.zeros((npoints, npoints), dtype=dtype)
    sim_func[(a.shape[0], b.shape[0]), DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros((npoints, npoints), dtype=dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[(a.shape[0], b.shape[0]), DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)


def _from_host(arr, buffer):
    if arr.flags["C_CONTIGUOUS"]:
        order = "C"
    elif arr.flags["F_CONTIGUOUS"]:
        order = "F"
    else:
        order = "K"
    return dpt.asarray(
        arr,
        dtype=arr.dtype,
        device=_def_device,
        copy=True,
        usm_type=buffer,
        sycl_queue=None,
        order=order,
    )


def _to_host(src, dst):
    np.copyto(dst, dpt.asnumpy(src))


def _check_filter_string(array, ir):
    filter_string = array.device.sycl_device.filter_string
    assert (
        ir.count(
            f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}"'
        )
        > 0
    ), ir


@require_gpu
@pytest.mark.parametrize(
    "s", [slice(1, None, 3), slice(1, None, -2), slice(1, 8, None)]
)
def test_kernel_slice_arg_dpctl(s):
    def func(s, res):
        i = get_global_id(0)
        res[s][i] = i

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    res = np.zeros(100, dtype=np.int32)
    N = len(res[s])

    sim_res = res.copy()
    gpu_res = res.copy()
    dgpu_res = _from_host(gpu_res, buffer="device")
    sim_func[N, DEFAULT_LOCAL_SIZE](s, sim_res)
    gpu_func[N, DEFAULT_LOCAL_SIZE](s, dgpu_res)

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_gpu
@pytest.mark.parametrize("size", [1, 13, 127, 1024])
def test_dpctl_simple1(size):
    def func(a, b, c):
        i = get_global_id(0)
        c[i] = a[i] + b[i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(size, dtype=np.float32)
    b = np.arange(size, dtype=np.float32) * 3

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="shared")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](da, db, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_gpu
def test_parfor_simple1():
    def py_func(a, b, c):
        for i in numba.prange(len(a)):
            c[i] = a[i] + b[i]

    gpu_func = njit(py_func)

    a = np.arange(1024, dtype=np.float32)
    b = np.arange(1024, dtype=np.float32) * 3

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, b, sim_res)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="shared")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func(da, db, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("val", _test_values)
def test_parfor_scalar(val):
    def py_func(a, b, c, d):
        for i in numba.prange(len(a)):
            c[i] = a[i] + b[i] + d

    gpu_func = njit(py_func)

    a = np.arange(1024, dtype=np.float32)
    b = np.arange(1024, dtype=np.float32) * 3

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, b, sim_res, val)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="shared")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func(da, db, dgpu_res, val)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("val", _test_values)
def test_parfor_scalar_capture(val):
    def py_func(a, b, c):
        for i in numba.prange(len(a)):
            c[i] = a[i] + b[i] + val

    gpu_func = njit(py_func)

    a = np.arange(1024, dtype=np.float32)
    b = np.arange(1024, dtype=np.float32) * 3

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, b, sim_res)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="shared")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func(da, db, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_gpu
def test_cfd_simple1():
    def py_func(a, b):
        b[:] = a * 2

    jit_func = njit(py_func)

    a = np.arange(1024, dtype=np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, sim_res)

    da = _from_host(a, buffer="device")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func(da, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_gpu
def test_cfd_simple2():
    def py_func(a, b, c):
        c[:] = a + b

    jit_func = njit(py_func)

    a = np.arange(1024, dtype=np.float32)
    b = np.arange(1024, dtype=np.float32) * 3

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, b, sim_res)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="shared")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func(da, db, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") > 0, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_gpu
def test_cfd_indirect():
    def py_func1(a, b):
        b[:] = a * 2

    jit_func1 = njit(py_func1)

    def py_func2(a, b):
        jit_func1(a, b)

    jit_func2 = njit(py_func2)

    a = np.arange(1024, dtype=np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    py_func2(a, sim_res)

    da = _from_host(a, buffer="device")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func2(da, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") > 0, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_gpu
def test_cfd_f64_truncate():
    def py_func(a, b, c):
        c[:] = a + b

    jit_func1 = njit(py_func, gpu_fp64_truncate=True)
    jit_func2 = njit(py_func, gpu_fp64_truncate=True)

    a = np.arange(1024, dtype=np.float32)
    b = 2.5

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, b, sim_res)

    da = _from_host(a, buffer="device")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    pattern = "%arg2: f64"

    with print_pass_ir(["TruncateF64ForGPUPass"], []):
        jit_func1(da, b, dgpu_res)
        ir = get_print_buffer()
        assert re.search(pattern, ir), ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)

    with print_pass_ir([], ["TruncateF64ForGPUPass"]):
        jit_func2(da, b, dgpu_res)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_gpu
def test_cfd_f64_truncate_indirect():
    def py_func_inner(a, b, c):
        c[:] = a + b

    jit_func_inner = njit_orig(py_func_inner, gpu_fp64_truncate=True)

    def py_func(a, b, c):
        jit_func_inner(a, b, c)

    jit_func1 = njit_orig(py_func, gpu_fp64_truncate=True)
    jit_func2 = njit_orig(py_func, gpu_fp64_truncate=True)

    a = np.arange(1024, dtype=np.float32)
    b = 2.5

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, b, sim_res)

    da = _from_host(a, buffer="device")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    pattern = "%arg2: f64"

    with print_pass_ir(["TruncateF64ForGPUPass"], []):
        jit_func1(da, b, dgpu_res)
        ir = get_print_buffer()
        assert re.search(pattern, ir), ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)

    with print_pass_ir([], ["TruncateF64ForGPUPass"]):
        jit_func2(da, b, dgpu_res)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_gpu
def test_cfd_use_64bit_index_prange_default():
    def py_func(a):
        ah, aw = a.shape

        for h in numba.prange(ah):
            for w in numba.prange(aw):
                a[h, w] = w + h * aw

    jit_func_defualt = njit(py_func)
    jit_func_64 = njit(py_func, gpu_use_64bit_index=True)

    a = np.zeros((2, 3), dtype=np.float32)

    da_default = _from_host(a, buffer="device")
    da64 = _from_host(a, buffer="device")

    da_default_host = np.zeros_like(a)
    da64_host = np.zeros_like(a)

    pattern = "spirv.SConvert %[0-9a-zA-Z]+ : i64 to i32"

    with print_pass_ir([], ["GPUToSpirvPass"]):
        jit_func_defualt(da_default)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    with print_pass_ir([], ["GPUToSpirvPass"]):
        jit_func_64(da64)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    _to_host(da_default, da_default_host)
    _to_host(da64, da64_host)
    assert_equal(da_default_host, da64_host)


@require_gpu
def test_cfd_use_64bit_index_prange():
    def py_func(a):
        ah, aw = a.shape

        for h in numba.prange(ah):
            for w in numba.prange(aw):
                a[h, w] = w + h * aw

    jit_func64 = njit(py_func, gpu_use_64bit_index=True)
    jit_func32 = njit(py_func, gpu_use_64bit_index=False)

    a = np.zeros((6, 8), dtype=np.float32)

    da64 = _from_host(a, buffer="device")
    da32 = _from_host(a, buffer="device")

    da64_host = np.zeros_like(a)
    da32_host = np.zeros_like(a)

    pattern = "spirv.SConvert %[0-9a-zA-Z]+ : i64 to i32"

    with print_pass_ir([], ["GPUToSpirvPass"]):
        jit_func64(da64)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    with print_pass_ir([], ["GPUToSpirvPass"]):
        jit_func32(da32)
        ir = get_print_buffer()
        assert re.search(pattern, ir), ir

    _to_host(da64, da64_host)
    _to_host(da32, da32_host)
    assert_equal(da64_host, da32_host)


@require_gpu
def test_cfd_use_64bit_index_kernel():
    def py_func(a):
        aw = a.shape[1]

        h = get_global_id(0)
        w = get_global_id(1)

        a[h, w] = w + h * aw

    jit_kern64 = kernel_cached(py_func, gpu_use_64bit_index=True)
    jit_kern32 = kernel_cached(py_func, gpu_use_64bit_index=False)

    a = np.zeros((6, 8), dtype=np.float32)

    da64 = _from_host(a, buffer="device")
    da32 = _from_host(a, buffer="device")

    da64_host = np.zeros_like(a)
    da32_host = np.zeros_like(a)

    pattern = "spirv.SConvert %[0-9a-zA-Z]+ : i64 to i32"

    with print_pass_ir([], ["GPUToSpirvPass"]):
        jit_kern64[da64.shape,](da64)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    with print_pass_ir([], ["GPUToSpirvPass"]):
        jit_kern32[da32.shape,](da32)
        ir = get_print_buffer()
        assert re.search(pattern, ir), ir

    _to_host(da64, da64_host)
    _to_host(da32, da32_host)
    assert_equal(da64_host, da32_host)


@require_gpu
def test_spirv_fastmath_default():
    def py_func(a):
        ah, aw = a.shape

        for h in numba.prange(ah):
            for w in numba.prange(aw):
                a[h, w] = w + np.float32(h) * aw

    jit_func_defualt = njit(py_func)
    jit_func_fm = njit(py_func, fastmath=True)

    a = np.zeros((2, 3), dtype=np.float32)

    da_default = _from_host(a, buffer="device")
    da64 = _from_host(a, buffer="device")

    da_default_host = np.zeros_like(a)
    da64_host = np.zeros_like(a)

    pattern = "fp_fast_math_mode = #spirv.fastmath_mode<Fast>"

    with print_pass_ir(["SerializeSPIRVPass"], []):
        jit_func_defualt(da_default)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    with print_pass_ir(["SerializeSPIRVPass"], []):
        jit_func_fm(da64)
        ir = get_print_buffer()
        assert re.search(pattern, ir), ir

    _to_host(da_default, da_default_host)
    _to_host(da64, da64_host)
    assert_equal(da_default_host, da64_host)


@require_gpu
def test_spirv_fastmath_prange():
    def py_func(a):
        ah, aw = a.shape

        for h in numba.prange(ah):
            for w in numba.prange(aw):
                a[h, w] = w + np.float32(h) * aw

    jit_func64 = njit(py_func, fastmath=False)
    jit_func32 = njit(py_func, fastmath=True)

    a = np.zeros((6, 8), dtype=np.float32)

    da64 = _from_host(a, buffer="device")
    da32 = _from_host(a, buffer="device")

    da64_host = np.zeros_like(a)
    da32_host = np.zeros_like(a)

    pattern = "fp_fast_math_mode = #spirv.fastmath_mode<Fast>"

    with print_pass_ir(["SerializeSPIRVPass"], []):
        jit_func64(da64)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    with print_pass_ir(["SerializeSPIRVPass"], []):
        jit_func32(da32)
        ir = get_print_buffer()
        assert re.search(pattern, ir), ir

    _to_host(da64, da64_host)
    _to_host(da32, da32_host)
    assert_equal(da64_host, da32_host)


@require_gpu
def test_spirv_fastmath_kernel():
    def py_func(a):
        aw = a.shape[1]

        h = get_global_id(0)
        w = get_global_id(1)

        a[h, w] = w + np.float32(h) * aw

    jit_kern64 = kernel_cached(py_func, fastmath=False)
    jit_kern32 = kernel_cached(py_func, fastmath=True)

    a = np.zeros((6, 8), dtype=np.float32)

    da64 = _from_host(a, buffer="device")
    da32 = _from_host(a, buffer="device")

    da64_host = np.zeros_like(a)
    da32_host = np.zeros_like(a)

    pattern = "fp_fast_math_mode = #spirv.fastmath_mode<Fast>"

    with print_pass_ir(["SerializeSPIRVPass"], []):
        jit_kern64[da64.shape,](da64)
        ir = get_print_buffer()
        assert re.search(pattern, ir) is None, ir

    with print_pass_ir(["SerializeSPIRVPass"], []):
        jit_kern32[da32.shape,](da32)
        ir = get_print_buffer()
        assert re.search(pattern, ir), ir

    _to_host(da64, da64_host)
    _to_host(da32, da32_host)
    assert_equal(da64_host, da32_host)


@require_gpu
def test_cfd_reshape():
    def py_func1(a):
        b = a.reshape(17, 23, 56)
        b[:] = 42

    jit_func1 = njit(py_func1)

    def py_func2(a):
        jit_func1(a)

    jit_func2 = njit(py_func2)

    a = np.arange(17 * 23 * 56, dtype=np.float32).reshape(23, 56, 17).copy()

    sim_res = np.zeros(a.shape, a.dtype)
    py_func2(sim_res)

    da = _from_host(a, buffer="device")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func2(dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") > 0, ir
    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_gpu
@pytest.mark.parametrize("size", [1, 7, 16, 64, 65, 256, 512, 1024 * 1024])
def test_cfd_reduce1(size):
    if size == 1:
        # TODO: Handle gpu array access outside the loops
        pytest.xfail()

    py_func = lambda a: a.sum()
    jit_func = njit(py_func)

    a = np.arange(size, dtype=np.float32)

    da = _from_host(a, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        assert_allclose(jit_func(da), py_func(a), rtol=1e-5)
        ir = get_print_buffer()
        _check_filter_string(da, ir)
        assert ir.count("gpu.launch blocks") == 1, ir


_shapes = (1, 7, 16, 25, 64, 65)


@require_gpu
@parametrize_function_variants(
    "py_func",
    [
        "lambda a: a.sum()",
        "lambda a: a.sum(axis=0)",
        "lambda a: a.sum(axis=1)",
        "lambda a: np.prod(a)",
        "lambda a: np.prod(a, axis=0)",
        "lambda a: np.prod(a, axis=1)",
        "lambda a: np.amin(a)",
        "lambda a: np.amax(a)",
    ],
)
@pytest.mark.parametrize("shape", itertools.product(_shapes, _shapes))
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
def test_cfd_reduce2(py_func, shape, dtype, request):
    if shape[0] == 1 or shape[1] == 1:
        # TODO: Handle gpu array access outside the loops
        pytest.xfail()

    count = math.prod(shape)
    if count > 30 and "np.prod" in str(request.node.callspec.id):
        # TODO: investigate overflow handling by spirv group ops
        pytest.skip()

    jit_func = njit(py_func)
    a = np.arange(1, count + 1, dtype=dtype).reshape(shape).copy()

    da = _from_host(a, buffer="shared")
    assert_allclose(jit_func(da), py_func(a), rtol=1e-5)


@require_gpu
@pytest.mark.parametrize(
    "a,b",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6]], np.float32),
            np.array([[1, 2], [3, 4], [5, 6]], np.float32),
        ),
    ],
)
@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b, c: np.dot(a, b, c)",
        "lambda a, b, c: np.dot(a, b, out=c)",
    ],
)
def test_cfd_dot(a, b, py_func):
    jit_func = njit(py_func)

    tmp = np.dot(a, b)
    res_py = np.zeros_like(tmp)
    gpu_res = np.zeros_like(tmp)

    py_func(a, b, res_py)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="device")
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func(da, db, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") > 0, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(res_py, gpu_res)


gemm_array_square = [
    np.zeros((4, 4), np.float32),
    np.ones((4, 4), np.float32),
    np.arange(4 * 4).reshape((4, 4)).astype(np.float32),
]
gemm_array = [
    np.zeros((4, 6), np.float32),
    np.ones((4, 6), np.float32),
    np.arange(4 * 6).reshape((4, 6)).astype(np.float32),
]
gemm_array_t = [a.T for a in gemm_array]
alpha_beta = [0, 1, 0.5]


@require_gpu
@pytest.mark.parametrize(
    "a,b,c,alpha,beta",
    list(
        itertools.product(
            gemm_array_square,
            gemm_array_square,
            gemm_array_square,
            alpha_beta,
            alpha_beta,
        )
    )
    + list(
        itertools.product(
            gemm_array, gemm_array_t, gemm_array_square, alpha_beta, alpha_beta
        )
    ),
)
def test_internal_gemm(a, b, c, alpha, beta):
    from numba_mlir.mlir.numpy.funcs import __internal_gemm

    def py_func(a, b, c, alpha, beta):
        __internal_gemm(a, b, c, alpha, beta)

    jit_func = njit(py_func)

    py_c = np.copy(c)
    cc = np.copy(c)

    py_func(a, b, py_c, alpha, beta)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="device")
    dc = _from_host(cc, buffer="device")

    jit_func(da, db, dc, alpha, beta)

    _to_host(dc, cc)
    assert_equal(py_c, cc)


@pytest.mark.smoke
@require_gpu
def test_l2_norm():
    def py_func(a, d):
        sq = np.square(a)
        sum = sq.sum(axis=1)
        d[:] = np.sqrt(sum)

    jit_func = njit(py_func)

    dtype = np.float32
    src = np.arange(10 * 10, dtype=dtype).reshape(10, 10)
    res = np.zeros(10, dtype=dtype)
    gpu_res = np.zeros_like(res)

    py_func(src, res)

    gpu_src = _from_host(src, buffer="device")
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func(gpu_src, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") > 0, ir

    _to_host(dgpu_res, gpu_res)
    assert_allclose(res, gpu_res, rtol=1e-5)


@require_gpu
def test_pairwise2():
    def py_func(X1, X2, D):
        X1_rows = X1.shape[0]
        X2_rows = X2.shape[0]
        X1_cols = X1.shape[1]

        for i in numba.prange(X1_rows):
            for j in numba.prange(X2_rows):
                d = 0.0
                for k in range(X1_cols):
                    tmp = X1[i, k] - X2[j, k]
                    d += tmp * tmp
                D[i, j] = np.sqrt(d)

    jit_func = njit(py_func)

    dims = 3
    npoints = 128
    dtype = np.float32

    a = np.arange(npoints * dims, dtype=dtype).reshape(npoints, dims)
    b = a + 5

    res = np.zeros((npoints, npoints), dtype=dtype)
    gpu_res = np.zeros_like(res)

    py_func(a, b, res)

    gpu_a = _from_host(a, buffer="device")
    gpu_b = _from_host(b, buffer="device")
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func(gpu_a, gpu_b, dgpu_res)
        ir = get_print_buffer()
        _check_filter_string(dgpu_res, ir)
        assert ir.count("gpu.launch blocks") > 0, ir

    _to_host(dgpu_res, gpu_res)
    assert_allclose(res, gpu_res, rtol=1e-5)


@require_gpu
def test_sycl_id_fit_in_int():
    @kernel_cached
    def func(a, b):
        i = get_global_id(0)
        if i == b:
            a[0] = b

    arr = np.zeros(1).astype(np.float32)
    gpu_arr = _from_host(arr, buffer="device")
    b = 10

    func[2**31 + 1,](gpu_arr, b)
    _to_host(gpu_arr, arr)
    assert arr[0] == b
