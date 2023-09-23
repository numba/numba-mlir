// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba-mlir-runtime_export.h"

using AllocToken = void *;

extern "C" NUMBA_MLIR_RUNTIME_EXPORT AllocToken *nmrtCreateAllocToken() {
  return new AllocToken;
}

extern "C" NUMBA_MLIR_RUNTIME_EXPORT void
nmrtDestroyAllocToken(AllocToken *token) {
  delete token;
}
