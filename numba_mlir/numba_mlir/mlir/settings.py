# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from .utils import readenv
from ..mlir_compiler import is_dpnp_supported, is_mkl_supported


USE_MLIR = readenv("NUMBA_MLIR_ENABLE", int, 1)
DUMP_PLIER = readenv("NUMBA_MLIR_DUMP_PLIER", int, 0)
DUMP_IR = readenv("NUMBA_MLIR_DUMP_IR", int, 0)
DUMP_DIAGNOSTICS = readenv("NUMBA_MLIR_DUMP_DIAGNOSTICS", int, 0)
DUMP_LLVM = readenv("NUMBA_MLIR_DUMP_LLVM", int, 0)
DUMP_OPTIMIZED = readenv("NUMBA_MLIR_DUMP_OPTIMIZED", int, 0)
DUMP_ASSEMBLY = readenv("NUMBA_MLIR_DUMP_ASSEMBLY", int, 0)
DEBUG_TYPE = list(filter(None, readenv("NUMBA_MLIR_DEBUG_TYPE", str, "").split(",")))
DPNP_AVAILABLE = (
    is_dpnp_supported()
)  # TODO: check if dpnp library is available at runtime
MKL_AVAILABLE = is_mkl_supported()
OPT_LEVEL = readenv("NUMBA_MLIR_OPT_LEVEL", int, 3)
