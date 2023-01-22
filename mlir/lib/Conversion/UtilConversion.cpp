// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Conversion/UtilConversion.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Conversion/Passes.h>

namespace {
struct ConvertTakeContext
    : public mlir::OpConversionPattern<numba::util::TakeContextOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::TakeContextOp op,
                  numba::util::TakeContextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcTypes = op.getResultTypes();
    auto count = static_cast<unsigned>(srcTypes.size());
    llvm::SmallVector<mlir::Type> newTypes(count);
    auto converter = getTypeConverter();
    assert(converter);
    for (auto i : llvm::seq(0u, count)) {
      auto oldType = srcTypes[i];
      auto newType = converter->convertType(oldType);
      newTypes[i] = (newType ? newType : oldType);
    }

    auto initFunc = adaptor.getInitFunc().value_or(mlir::SymbolRefAttr());
    auto releaseFunc = adaptor.getReleaseFunc().value_or(mlir::SymbolRefAttr());
    rewriter.replaceOpWithNewOp<numba::util::TakeContextOp>(
        op, newTypes, initFunc, releaseFunc);
    return mlir::success();
  }
};

struct ConvertEnvRegion
    : public mlir::OpConversionPattern<numba::util::EnvironmentRegionOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionOp op,
                  numba::util::EnvironmentRegionOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(
            converter->convertTypes(op->getResultTypes(), resultTypes)))
      return mlir::failure();

    auto loc = op.getLoc();
    auto newOp = rewriter.create<numba::util::EnvironmentRegionOp>(
        loc, adaptor.getEnvironment(), adaptor.getArgs(), resultTypes);
    auto &newRegion = newOp.getRegion();

    // Erase block created by builder.
    rewriter.eraseBlock(&newRegion.front());

    auto &oldRegion = op.getRegion();
    rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
    rewriter.replaceOp(op, newOp.getResults());
    return mlir::success();
  }
};

struct ConvertEnvRegionYield
    : public mlir::OpConversionPattern<numba::util::EnvironmentRegionYieldOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionYieldOp op,
                  numba::util::EnvironmentRegionYieldOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<numba::util::EnvironmentRegionYieldOp>(
        op, adaptor.getOperands());
    return mlir::success();
  }
};
} // namespace

void numba::populateUtilConversionPatterns(mlir::TypeConverter &converter,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::ConversionTarget &target) {
  patterns.insert<ConvertTakeContext>(converter, patterns.getContext());

  target.addDynamicallyLegalOp<numba::util::TakeContextOp>(
      [&](mlir::Operation *op) -> llvm::Optional<bool> {
        for (auto range : {mlir::TypeRange(op->getOperandTypes()),
                           mlir::TypeRange(op->getResultTypes())})
          for (auto type : range)
            if (converter.isLegal(type))
              return true;

        return std::nullopt;
      });

  patterns.insert<ConvertEnvRegion, ConvertEnvRegionYield>(
      converter, patterns.getContext());

  target.addDynamicallyLegalOp<numba::util::EnvironmentRegionOp,
                               numba::util::EnvironmentRegionYieldOp>(
      [&](mlir::Operation *op) -> llvm::Optional<bool> {
        if (converter.isLegal(op))
          return true;

        return std::nullopt;
      });
}
