// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Common.hpp"
#include "numba-mlir-math-runtime_export.h"

extern "C" {
NUMBA_MLIR_MATH_RUNTIME_EXPORT void nmrtMathRuntimeInit() {
  // Nothing
}

NUMBA_MLIR_MATH_RUNTIME_EXPORT void nmrtMathRuntimeFinalize() {
  // Nothing
}
}
