// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Dialect/math_ext/IR/MathExt.hpp"

#include <cmath>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/CommonFolders.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Builders.h>
#include <optional>

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "numba/Dialect/math_ext/IR/MathExtOps.cpp.inc"

/// Materialize an integer or floating point constant.
mlir::Operation *numba::math_ext::MathExtDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  if (auto poison = mlir::dyn_cast<mlir::ub::PoisonAttr>(value))
    return builder.create<mlir::ub::PoisonOp>(loc, type, poison);

  return mlir::arith::ConstantOp::materialize(builder, value, type, loc);
}

template <typename T>
static mlir::Attribute foldUnary(mlir::ArrayRef<mlir::Attribute> args,
                                 T &&folder) {
  return mlir::constFoldUnaryOpConditional<mlir::FloatAttr>(
      args, [&](const mlir::APFloat &a) -> std::optional<mlir::APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return mlir::APFloat(folder(a.convertToDouble()));
        case 32:
          return mlir::APFloat(folder(a.convertToFloat()));
        default:
          return {};
        }
      });
}

#define MAKE_UNARY_FOLDER(func)                                                \
  auto folder = [](auto arg) { return func(arg); };                            \
  return foldUnary(adaptor.getOperands(), folder);

mlir::OpFoldResult
numba::math_ext::AcosOp::fold(numba::math_ext::AcosOp::FoldAdaptor adaptor) {
  MAKE_UNARY_FOLDER(std::acos);
}

#undef MAKE_UNARY_FOLDER
