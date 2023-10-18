# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import atexit
from .utils import load_lib, mlir_func_name, register_cfunc
from .settings import MKL_AVAILABLE, SYCL_MKL_AVAILABLE

runtime_lib = load_lib("numba-mlir-math-runtime")
runtime_sycl_lib = load_lib("numba-mlir-math-sycl-runtime")

_init_func = runtime_lib.nmrtMathRuntimeInit
_init_func()

_init_sycl_func = runtime_sycl_lib.nmrtMathRuntimeInit
_init_sycl_func()


def load_function_variants(runtime_lib, func_name, suffixes):
    for s in suffixes:
        name = func_name % s
        mlir_name = mlir_func_name(name)
        func = getattr(runtime_lib, name)
        register_cfunc(mlir_name, func)


load_function_variants(runtime_lib, "dpnp_linalg_eig_%s", ["float32", "float64"])
if MKL_AVAILABLE:
    load_function_variants(runtime_lib, "mkl_gemm_%s", ["float32", "float64"])
    load_function_variants(runtime_lib, "mkl_inv_%s", ["float32", "float64"])
if SYCL_MKL_AVAILABLE:
    load_function_variants(
        runtime_sycl_lib, "mkl_gemm_%s_device", ["float32", "float64"]
    )

_finalize_func = runtime_lib.nmrtMathRuntimeFinalize
_finalize_sycl_func = runtime_sycl_lib.nmrtMathRuntimeFinalize


@atexit.register
def _cleanup():
    _finalize_func()
    _finalize_sycl_func()
