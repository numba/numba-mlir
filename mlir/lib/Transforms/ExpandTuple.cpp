// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/ExpandTuple.hpp"

#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Transforms/TypeConversion.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

namespace {
static void flattenTuple(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ValueRange values,
                         llvm::SmallVectorImpl<mlir::Value> &ret) {
  for (auto arg : values) {
    if (auto tupleType = arg.getType().dyn_cast<mlir::TupleType>()) {
      for (auto &&[i, argType] : llvm::enumerate(tupleType.getTypes())) {
        auto ind = builder.createOrFold<mlir::arith::ConstantIndexOp>(loc, i);
        auto res = builder.createOrFold<numba::util::TupleExtractOp>(
            loc, argType, arg, ind);
        flattenTuple(builder, loc, res, ret);
      }
    } else {
      ret.emplace_back(arg);
    }
  }
}

struct ExpandTupleReturn
    : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op,
                  mlir::func::ReturnOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> newOperands;
    auto loc = op.getLoc();
    flattenTuple(rewriter, loc, adaptor.getOperands(), newOperands);
    auto *operation = op.getOperation();
    rewriter.modifyOpInPlace(op,
                             [&]() { operation->setOperands(newOperands); });
    return mlir::success();
  }
};

class ExpandEnvRegionYield
    : public mlir::OpConversionPattern<numba::util::EnvironmentRegionYieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionYieldOp op,
                  numba::util::EnvironmentRegionYieldOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> newOperands;
    auto loc = op.getLoc();
    flattenTuple(rewriter, loc, adaptor.getResults(), newOperands);

    rewriter.replaceOpWithNewOp<numba::util::EnvironmentRegionYieldOp>(
        op, newOperands);
    return mlir::success();
  }
};

static std::optional<mlir::Value> reconstructTuple(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::TupleType tupleType,
                                                   mlir::ValueRange values) {
  llvm::SmallVector<mlir::Value, 4> vals(tupleType.size());
  for (auto &&[i, type] : llvm::enumerate(tupleType.getTypes())) {
    if (auto innerTuple = type.dyn_cast<mlir::TupleType>()) {
      auto val = reconstructTuple(builder, loc, innerTuple, values);
      if (!val)
        return std::nullopt;

      vals[i] = *val;
      values = values.drop_front(innerTuple.size());
    } else {
      if (values.empty())
        return std::nullopt;

      vals[i] = values.front();
      values = values.drop_front();
    }
  }
  return builder.create<numba::util::BuildTupleOp>(loc, tupleType, vals);
}

static std::optional<mlir::Value> tupleToElem(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Type type,
                                              mlir::ValueRange values) {
  if (values.size() != 1)
    return std::nullopt;

  mlir::Value value = values.front();
  auto tupleType = value.getType().dyn_cast<mlir::TupleType>();
  if (!tupleType || tupleType.size() != 1 || tupleType.getType(0) != type)
    return std::nullopt;

  mlir::Value index = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value result =
      builder.create<numba::util::TupleExtractOp>(loc, type, value, index);
  return result;
}

struct ExpandTuplePass
    : public mlir::PassWrapper<ExpandTuplePass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandTuplePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addConversion(
        [&typeConverter](mlir::TupleType type,
                         llvm::SmallVectorImpl<mlir::Type> &ret)
            -> std::optional<mlir::LogicalResult> {
          if (mlir::failed(typeConverter.convertTypes(type.getTypes(), ret)))
            return std::nullopt;

          return mlir::success();
        });

    auto materializeTupleCast =
        [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
           mlir::Location loc) -> std::optional<mlir::Value> {
      if (auto tupleType = type.dyn_cast<mlir::TupleType>())
        return reconstructTuple(builder, loc, tupleType, inputs);

      if (auto elem = tupleToElem(builder, loc, type, inputs))
        return *elem;

      auto cast =
          builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);
      return cast.getResult(0);
    };
    typeConverter.addArgumentMaterialization(materializeTupleCast);
    typeConverter.addSourceMaterialization(materializeTupleCast);
    typeConverter.addTargetMaterialization(materializeTupleCast);

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    numba::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                              patterns, target);

    patterns.insert<ExpandTupleReturn, ExpandEnvRegionYield>(typeConverter,
                                                             context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createExpandTuplePass() {
  return std::make_unique<ExpandTuplePass>();
}
