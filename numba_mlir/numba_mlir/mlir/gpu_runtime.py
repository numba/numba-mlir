# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import atexit
import logging
from .utils import load_lib, mlir_func_name, register_cfunc, readenv
from . import python_rt

from numba.core.runtime import _nrt_python as _nrt


GPU_RUNTIME = readenv("NUMBA_MLIR_GPU_RUNTIME", str, "sycl")


def _get_gpu_runtime_name():
    if GPU_RUNTIME == "sycl":
        return "numba-mlir-gpu-runtime-sycl"

    raise ValueError(f'Invalid GPU runtime type: "{GPU_RUNTIME}", expected "sycl"')


try:
    runtime_lib = load_lib(_get_gpu_runtime_name())
    IS_GPU_RUNTIME_AVAILABLE = True
except Exception:
    logging.exception("GPU runtime loading failed")
    IS_GPU_RUNTIME_AVAILABLE = False


if IS_GPU_RUNTIME_AVAILABLE:

    def _register_funcs():
        _funcs = [
            "gpuxAlloc",
            "gpuxDeAlloc",
            "gpuxKernelDestroy",
            "gpuxKernelGet",
            "gpuxLaunchKernel",
            "gpuxModuleDestroy",
            "gpuxModuleLoad",
            "gpuxStreamCreate",
            "gpuxStreamDestroy",
            "gpuxSuggestBlockSize",
            "gpuxWait",
            "gpuxDestroyEvent",
            mlir_func_name("get_global_id"),
            mlir_func_name("get_global_size"),
            mlir_func_name("get_group_id"),
            mlir_func_name("get_local_id"),
            mlir_func_name("get_local_size"),
            mlir_func_name("kernel_barrier"),
            mlir_func_name("kernel_mem_fence"),
            "gpuxDuplicateQueue",
        ]

        from itertools import product

        _types = ["int32", "int64", "float32", "float64"]

        _atomic_ops = ["add", "sub"]
        for o, t in product(_atomic_ops, _types):
            _funcs.append(mlir_func_name(f"atomic_{o}_{t}"))

        for n, t in product(range(8), _types):
            _funcs.append(mlir_func_name(f"local_array_{t}_{n}"))
            _funcs.append(mlir_func_name(f"private_array_{t}_{n}"))

        _group_ops = ["reduce_add", "reduce_mul", "reduce_min", "reduce_max"]
        for o, t in product(_group_ops, _types):
            _funcs.append(mlir_func_name(f"group_{o}_{t}"))

        for name in _funcs:
            if hasattr(runtime_lib, name):
                func = getattr(runtime_lib, name)
            else:
                func = 1
            register_cfunc(name, func)

    _register_funcs()
    del _register_funcs
