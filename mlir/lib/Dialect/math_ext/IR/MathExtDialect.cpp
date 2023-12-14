// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Dialect/math_ext/IR/MathExt.hpp"

#include "mlir/Transforms/InliningUtils.h"

#include "numba/Dialect/math_ext/IR/MathExtOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with math
/// operations.
struct MathExtInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }
};
} // namespace

void numba::math_ext::MathExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "numba/Dialect/math_ext/IR/MathExtOps.cpp.inc"
      >();
  addInterfaces<MathExtInlinerInterface>();
}
