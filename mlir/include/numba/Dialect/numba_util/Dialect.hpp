// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include "numba/Dialect/numba_util/NumbaUtilOpsDialect.h.inc"
#include "numba/Dialect/numba_util/NumbaUtilOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "numba/Dialect/numba_util/NumbaUtilOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "numba/Dialect/numba_util/NumbaUtilOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "numba/Dialect/numba_util/NumbaUtilOps.h.inc"

namespace numba {
namespace util {
namespace attributes {
llvm::StringRef getFastmathName();
llvm::StringRef getJumpMarkersName();
llvm::StringRef getParallelName();
llvm::StringRef getMaxConcurrencyName();
llvm::StringRef getForceInlineName();
llvm::StringRef getOptLevelName();
llvm::StringRef getShapeRangeName();
} // namespace attributes
} // namespace util
} // namespace numba
