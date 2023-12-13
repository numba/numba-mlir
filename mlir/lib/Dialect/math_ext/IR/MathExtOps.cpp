// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Dialect/math_ext/IR/MathExt.hpp"

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
mlir::Operation *numba::math_ext::MathExtDialect::materializeConstant(mlir::OpBuilder &builder,
                                                  mlir::Attribute value, mlir::Type type,
                                                  mlir::Location loc) {
  if (auto poison = mlir::dyn_cast<mlir::ub::PoisonAttr>(value))
    return builder.create<mlir::ub::PoisonOp>(loc, type, poison);

  return mlir::arith::ConstantOp::materialize(builder, value, type, loc);
}

template<typename Folder>
static std::optional<mlir::APFloat> foldUnary(mlir::ValueRange args) {
  return mlir::constFoldUnaryOpConditional<mlir::FloatAttr>(
      args, [](const mlir::APFloat &a) -> std::optional<mlir::APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return mlir::APFloat(Folder()(a.convertToDouble()));
        case 32:
          return mlir::APFloat(Folder()(a.convertToFloat()));
        default:
          return {};
        }
      });
}

#define MAKE_FOLDER(func) struct Folder { auto operator()(auto arg) { return func(arg); } }; return foldUnary<Folder>(adaptor.getOperands());

mlir::OpFoldResult numba::math_ext::AcosOp::fold(numba::math_ext::AcosOp::FoldAdaptor adaptor) {
  return mlir::constFoldUnaryOpConditional<mlir::FloatAttr>(
      adaptor.getOperands(), [](const mlir::APFloat &a) -> std::optional<mlir::APFloat> {
        switch (a.getSizeInBits(a.getSemantics())) {
        case 64:
          return mlir::APFloat(atan(a.convertToDouble()));
        case 32:
          return mlir::APFloat(atanf(a.convertToFloat()));
        default:
          return {};
        }
      });
}
