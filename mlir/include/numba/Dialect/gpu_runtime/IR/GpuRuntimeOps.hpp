// SPDX-FileCopyrightText: 2022 Intel Corporation
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

#include <mlir/Dialect/GPU/IR/GPUDialect.h>

#include "numba/Dialect/numba_util/Dialect.hpp"

#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.h.inc"

namespace gpu_runtime {
mlir::StringRef getFenceFlagsAttrName();
mlir::StringRef getFp64TruncateAttrName();
mlir::StringRef getUse64BitIndexAttrName();
mlir::StringRef getDeviceFuncAttrName();
mlir::StringRef getHostAllocAttrName();

enum class FenceFlags : int64_t {
  local = 1,
  global = 2,
};
} // namespace gpu_runtime
