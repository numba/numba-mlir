# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from os import environ
import warnings

from ..mlir_compiler import is_dpnp_supported, is_mkl_supported


def _readenv(name, ctor, default):
    value = environ.get(name)
    if value is None:
        return default() if callable(default) else default
    try:
        return ctor(value)
    except Exception:
        warnings.warn(
            "environ %s defined but failed to parse '%s'" % (name, value),
            RuntimeWarning,
        )
        return default


USE_MLIR = _readenv("NUMBA_MLIR_ENABLE", int, 1)
DUMP_PLIER = _readenv("NUMBA_MLIR_DUMP_PLIER", int, 0)
DUMP_IR = _readenv("NUMBA_MLIR_DUMP_IR", int, 0)
DUMP_DIAGNOSTICS = _readenv("NUMBA_MLIR_DUMP_DIAGNOSTICS", int, 0)
DUMP_LLVM = _readenv("NUMBA_MLIR_DUMP_LLVM", int, 0)
DUMP_OPTIMIZED = _readenv("NUMBA_MLIR_DUMP_OPTIMIZED", int, 0)
DUMP_ASSEMBLY = _readenv("NUMBA_MLIR_DUMP_ASSEMBLY", int, 0)
DEBUG_TYPE = list(filter(None, _readenv("NUMBA_MLIR_DEBUG_TYPE", str, "").split(",")))
DPNP_AVAILABLE = (
    is_dpnp_supported()
)  # TODO: check if dpnp library is available at runtime
MKL_AVAILABLE = is_mkl_supported()
OPT_LEVEL = _readenv("NUMBA_MLIR_OPT_LEVEL", int, 3)
