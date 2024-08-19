// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>

namespace numba {
mlir::LogicalResult naivelyFuseParallelOps(mlir::Region &region);
mlir::LogicalResult
prepareForFusion(mlir::Region &region,
                 llvm::function_ref<bool(mlir::Operation &)> needPrepare);
} // namespace numba
