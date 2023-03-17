# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numba.core import types
from numba.core.typing.templates import (
    ConcreteTemplate,
    signature,
    infer_global,
)


def _stub_error():
    raise NotImplementedError("This is a stub")


# Python defines separate modules math/cmath, but we want to have single
# function which dispatch either to normal or complex function


def exp():
    _stub_error()


def sqrt():
    _stub_error()


@infer_global(exp)
@infer_global(sqrt)
class _UnaryFuncId(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int8),
        signature(types.float64, types.uint8),
        signature(types.float64, types.int16),
        signature(types.float64, types.uint16),
        signature(types.float64, types.int32),
        signature(types.float64, types.uint32),
        signature(types.float64, types.int64),
        signature(types.float64, types.uint64),
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
        signature(types.complex64, types.complex64),
        signature(types.complex128, types.complex128),
    ]
