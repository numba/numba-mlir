// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/NtensorToMemref.hpp"

#include "numba/Conversion/UtilConversion.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Dialect/numba_util/Utils.hpp"
#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"
#include "numba/Transforms/TypeConversion.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
struct DimOpLowering : public mlir::OpConversionPattern<numba::ntensor::DimOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::DimOp op,
                  numba::ntensor::DimOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType =
        op.getSource().getType().cast<numba::ntensor::NTensorType>();
    auto src = adaptor.getSource();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto indexType = rewriter.getIndexType();
    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), indexType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::memref::DimOp>(loc, src, adaptor.getIndex())
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct CreateOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::CreateArrayOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CreateArrayOp op,
                  numba::ntensor::CreateArrayOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto dstType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!dstType)
      return mlir::failure();

    auto elemType = dstType.getElementType();
    auto initValue = adaptor.getInitValue();
    if (initValue && initValue.getType() != elemType)
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), srcType.getEnvironment(), dstType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          mlir::Value result = builder.create<mlir::memref::AllocOp>(
              loc, dstType, adaptor.getDynamicSizes());
          if (initValue)
            builder.create<mlir::linalg::FillOp>(loc, initValue, result);
          return result;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct SubviewOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::SubviewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::SubviewOp op,
                  numba::ntensor::SubviewOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType =
        op.getSource().getType().cast<numba::ntensor::NTensorType>();
    auto src = adaptor.getSource();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto dstType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!dstType)
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), dstType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto offsets = mlir::getMixedValues(adaptor.getStaticOffsets(),
                                              adaptor.getOffsets(), rewriter);
          auto sizes = mlir::getMixedValues(adaptor.getStaticSizes(),
                                            adaptor.getSizes(), rewriter);
          auto strides = mlir::getMixedValues(adaptor.getStaticStrides(),
                                              adaptor.getStrides(), rewriter);

          auto resType =
              mlir::memref::SubViewOp::inferRankReducedResultType(
                  dstType.getShape(), srcType, offsets, sizes, strides)
                  .cast<mlir::MemRefType>();

          mlir::Value res = builder.create<mlir::memref::SubViewOp>(
              loc, resType, src, offsets, sizes, strides);

          if (resType != dstType)
            res =
                builder.create<numba::util::ChangeLayoutOp>(loc, dstType, res);

          return res;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct LoadOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::LoadOp op,
                  numba::ntensor::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getArray().getType().cast<numba::ntensor::NTensorType>();
    auto src = adaptor.getArray();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto dstType = converter->convertType(op.getType());
    if (!dstType || dstType != origType.getElementType())
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), dstType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::memref::LoadOp>(loc, src, adaptor.getIndices())
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct StoreOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::StoreOp op,
                  numba::ntensor::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getArray().getType().cast<numba::ntensor::NTensorType>();
    auto src = adaptor.getArray();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), std::nullopt,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto val = adaptor.getValue();
          builder.create<mlir::memref::StoreOp>(loc, val, src,
                                                adaptor.getIndices());
          return std::nullopt;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ToTensorOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::ToTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::ToTensorOp op,
                  numba::ntensor::ToTensorOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto array = adaptor.getArray();
    if (!array.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto retType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::TensorType>();
    if (!retType)
      return mlir::failure();

    auto origType = op.getArray().getType().cast<numba::ntensor::NTensorType>();
    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), retType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::bufferization::ToTensorOp>(loc, retType, array)
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct FromTensorOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::FromTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::FromTensorOp op,
                  numba::ntensor::FromTensorOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tensor = adaptor.getTensor();
    if (!tensor.getType().isa<mlir::RankedTensorType>())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto origType = op.getType().cast<numba::ntensor::NTensorType>();
    auto retType =
        converter->convertType(origType).dyn_cast_or_null<mlir::MemRefType>();
    if (!retType)
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), retType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::bufferization::ToMemrefOp>(loc, retType, tensor)
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ToMemrefOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::ToMemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::ToMemrefOp op,
                  numba::ntensor::ToMemrefOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getArray();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto retType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!retType)
      return mlir::failure();

    if (srcType != retType)
      src = rewriter.create<mlir::memref::CastOp>(op.getLoc(), retType, src);

    rewriter.replaceOp(op, src);
    return mlir::success();
  }
};

struct FromMemrefOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::FromMemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::FromMemrefOp op,
                  numba::ntensor::FromMemrefOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getMemref();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto retType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!retType)
      return mlir::failure();

    if (srcType != retType)
      return mlir::failure();

    rewriter.replaceOp(op, src);
    return mlir::success();
  }
};

struct CastOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CastOp op,
                  numba::ntensor::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSource();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto origSrcType =
        op.getSource().getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!origSrcType)
      return mlir::failure();

    auto origDstType = op.getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!origDstType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto retType = converter->convertType(origDstType)
                       .dyn_cast_or_null<mlir::MemRefType>();

    if (!retType)
      return mlir::failure();

    if (srcType == retType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    if (origSrcType.getEnvironment() != origDstType.getEnvironment())
      return mlir::failure();

    if (!mlir::memref::CastOp::areCastCompatible(srcType, retType))
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), origSrcType.getEnvironment(), retType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder.create<mlir::memref::CastOp>(loc, retType, src)
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct CopyOpLowering
    : public mlir::OpConversionPattern<numba::ntensor::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CopyOp op,
                  numba::ntensor::CopyOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSource();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto dst = adaptor.getTarget();
    if (!dst.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto origSrcType =
        op.getSource().getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!origSrcType)
      return mlir::failure();

    auto origDstType =
        op.getTarget().getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!origDstType)
      return mlir::failure();

    if (origSrcType.getEnvironment() != origDstType.getEnvironment())
      return mlir::failure();

    numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), origSrcType.getEnvironment(), std::nullopt,
        [&](mlir::ConversionPatternRewriter &builder, mlir::Location /*loc*/) {
          builder.replaceOpWithNewOp<mlir::memref::CopyOp>(op, src, dst);
          return std::nullopt;
        });
    return mlir::success();
  }
};
} // namespace

void numba::populateNtensorToMemrefRewritesAndTarget(
    mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  converter.addConversion(
      [](numba::ntensor::NTensorType type) -> llvm::Optional<mlir::Type> {
        auto elemType = type.getElementType();
        if (mlir::MemRefType::isValidElementType(elemType))
          return mlir::MemRefType::get(type.getShape(), elemType);

        return std::nullopt;
      });

  patterns.insert<DimOpLowering, CreateOpLowering, SubviewOpLowering,
                  LoadOpLowering, StoreOpLowering, ToTensorOpLowering,
                  FromTensorOpLowering, ToMemrefOpLowering,
                  FromMemrefOpLowering, CastOpLowering, CopyOpLowering>(
      converter, patterns.getContext());

  target.addIllegalOp<numba::ntensor::DimOp, numba::ntensor::CreateArrayOp,
                      numba::ntensor::SubviewOp, numba::ntensor::LoadOp,
                      numba::ntensor::StoreOp, numba::ntensor::ToTensorOp,
                      numba::ntensor::FromTensorOp, numba::ntensor::ToMemrefOp,
                      numba::ntensor::FromMemrefOp, numba::ntensor::CastOp,
                      numba::ntensor::CopyOp>();
}

namespace {
struct NtensorToMemrefPass
    : public mlir::PassWrapper<NtensorToMemrefPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorToMemrefPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::TypeConverter converter;
    mlir::RewritePatternSet patterns(&context);
    mlir::ConversionTarget target(context);

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });

    numba::populateTupleTypeConverter(converter);

    auto addUnrealizedCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs, mlir::Location loc) {
      auto cast =
          builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);
      return llvm::Optional<mlir::Value>(cast.getResult(0));
    };
    converter.addArgumentMaterialization(addUnrealizedCast);
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    numba::populateTupleTypeConversionRewritesAndTarget(converter, patterns,
                                                        target);
    numba::populateControlFlowTypeConversionRewritesAndTarget(converter,
                                                              patterns, target);
    numba::populateNtensorToMemrefRewritesAndTarget(converter, patterns,
                                                    target);
    numba::populateUtilConversionPatterns(converter, patterns, target);

    auto op = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createNtensorToMemrefPass() {
  return std::make_unique<NtensorToMemrefPass>();
}
