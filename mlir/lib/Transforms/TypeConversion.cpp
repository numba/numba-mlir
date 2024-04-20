// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/TypeConversion.hpp"

#include "numba/Dialect/numba_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
static void
unpackUnrealizedConversionCast(mlir::Value v,
                               mlir::SmallVectorImpl<mlir::Value> &unpacked);

static llvm::SmallVector<mlir::Value>
unpackUnrealizedConversionCast(mlir::ValueRange values) {
  llvm::SmallVector<mlir::Value> ret;
  for (auto value : values)
    unpackUnrealizedConversionCast(value, ret);

  return ret;
}

static std::optional<llvm::SmallVector<mlir::Value>>
packResults(mlir::OpBuilder &rewriter, mlir::Location loc,
            const mlir::TypeConverter &typeConverter, mlir::TypeRange resTypes,
            mlir::ValueRange newResults) {
  llvm::SmallVector<mlir::Type> newResultTypes;
  llvm::SmallVector<unsigned> offsets;
  offsets.push_back(0);
  // Do the type conversion and record the offsets.
  for (auto type : resTypes) {
    if (mlir::failed(typeConverter.convertTypes(type, newResultTypes)))
      return std::nullopt;

    offsets.push_back(newResultTypes.size());
  }

  llvm::SmallVector<mlir::Value> packedRets;
  for (unsigned i = 1, e = offsets.size(); i < e; i++) {
    unsigned start = offsets[i - 1], end = offsets[i];
    unsigned len = end - start;
    mlir::ValueRange mappedValue = newResults.slice(start, len);
    if (len != 1) {
      // 1 : N type conversion.
      auto origType = resTypes[i - 1];
      auto mat = typeConverter.materializeSourceConversion(
          rewriter, loc, origType, mappedValue);
      if (!mat)
        return std::nullopt;

      packedRets.push_back(mat);
    } else {
      // 1 : 1 type conversion.
      packedRets.push_back(mappedValue.front());
    }
  }

  return packedRets;
}

class ConvertSelectOp
    : public mlir::OpConversionPattern<mlir::arith::SelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::arith::SelectOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> retTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op.getType(), retTypes)))
      return mlir::failure();

    if (retTypes.size() == 1) {
      rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
          op, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return mlir::success();
    }

    auto trueVals = unpackUnrealizedConversionCast(adaptor.getTrueValue());
    auto falseVals = unpackUnrealizedConversionCast(adaptor.getFalseValue());

    if (trueVals.size() != falseVals.size())
      return mlir::failure();

    auto loc = op.getLoc();
    auto cond = adaptor.getCondition();
    mlir::SmallVector<mlir::Value> results;
    for (auto &&[trueVal, falseVal] : llvm::zip(trueVals, falseVals)) {
      mlir::Value res =
          rewriter.create<mlir::arith::SelectOp>(loc, cond, trueVal, falseVal);
      results.emplace_back(res);
    }

    auto newResults =
        packResults(rewriter, loc, *typeConverter, op.getType(), results);
    if (!newResults)
      return mlir::failure();

    rewriter.replaceOp(op, *newResults);
    return mlir::success();
  }
};

class ConvertEnvRegionYield
    : public mlir::OpConversionPattern<numba::util::EnvironmentRegionYieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionYieldOp op,
                  numba::util::EnvironmentRegionYieldOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<numba::util::EnvironmentRegionYieldOp>(
        op, unpackUnrealizedConversionCast(adaptor.getResults()));
    return mlir::success();
  }
};

class ConvertEnvRegion
    : public mlir::OpConversionPattern<numba::util::EnvironmentRegionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionOp op,
                  numba::util::EnvironmentRegionOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(converter->convertTypes(op.getResultTypes(), resultTypes)))
      return mlir::failure();

    auto args = unpackUnrealizedConversionCast(adaptor.getArgs());

    auto loc = op.getLoc();
    auto newRegOp = rewriter.create<numba::util::EnvironmentRegionOp>(
        loc, adaptor.getEnvironment(), args, resultTypes);

    auto &newRegion = newRegOp.getRegion();
    rewriter.eraseBlock(&newRegion.front());

    auto &oldRegion = op.getRegion();
    rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());

    auto newResults = packResults(rewriter, loc, *typeConverter,
                                  op.getResultTypes(), newRegOp.getResults());
    if (!newResults)
      return mlir::failure();

    rewriter.replaceOp(op, *newResults);
    return mlir::success();
  }
};

using namespace mlir;
// Unpacks the single unrealized_conversion_cast using the list of inputs
// e.g., return [%b, %c, %d] for %a = unrealized_conversion_cast(%b, %c, %d)
static void unpackUnrealizedConversionCast(Value v,
                                           SmallVectorImpl<Value> &unpacked) {
  if (auto cast =
          dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp())) {
    auto inputs = cast.getInputs();
    if (inputs.size() != 1) {
      // 1 : N type conversion.
      unpacked.append(inputs.begin(), inputs.end());
      return;
    }
  }

  // TODO: hack for tuple, need proper 1 : N conversion support upstream
  if (auto cast =
          dyn_cast_or_null<numba::util::BuildTupleOp>(v.getDefiningOp())) {
    auto inputs = cast.getArgs();
    if (inputs.size() != 1) {
      // 1 : N type conversion.
      unpacked.append(inputs.begin(), inputs.end());
      return;
    }
  }
  // 1 : 1 type conversion.
  unpacked.push_back(v);
}

// Need our own copy to workaround a bug in upstream
// https://github.com/llvm/llvm-project/issues/58742
class ConvertForOpTypes : public OpConversionPattern<scf::ForOp> {
public:
  ConvertForOpTypes(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<scf::ForOp>(typeConverter, context,
                                        /*benefit*/ 10) {}

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    SmallVector<unsigned> offsets;
    offsets.push_back(0);
    // Do the type conversion and record the offsets.
    for (Type type : op.getResultTypes()) {
      if (failed(typeConverter->convertTypes(type, newResultTypes)))
        return rewriter.notifyMatchFailure(op, "could not convert result");
      offsets.push_back(newResultTypes.size());
    }

    // Create a empty new op and inline the regions from the old op.
    //
    // This is a little bit tricky. We have two concerns here:
    //
    // 1. We cannot update the op in place because the dialect conversion
    // framework does not track type changes for ops updated in place, so it
    // won't insert appropriate materializations on the changed result types.
    // PR47938 tracks this issue, but it seems hard to fix. Instead, we need
    // to clone the op.
    //
    // 2. We need to resue the original region instead of cloning it, otherwise
    // the dialect conversion framework thinks that we just inserted all the
    // cloned child ops. But what we want is to "take" the child regions and let
    // the dialect conversion framework continue recursively into ops inside
    // those regions (which are already in its worklist; inlining them into the
    // new op's regions doesn't remove the child ops from the worklist).

    auto indexType = rewriter.getIndexType();
    TypeRange origBlockArgs = op.getBody()->getArgumentTypes();
    TypeConverter::SignatureConversion newSig(origBlockArgs.size());
    newSig.addInputs(0, indexType);
    if (failed(typeConverter->convertSignatureArgs(origBlockArgs.drop_front(),
                                                   newSig, 1)))
      return failure();

    // convertRegionTypes already takes care of 1:N conversion.
    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter,
                                           &newSig)))
      return failure();

    auto loc = op.getLoc();
    auto lBound = typeConverter->materializeSourceConversion(
        rewriter, loc, indexType, adaptor.getLowerBound());
    auto uBound = typeConverter->materializeSourceConversion(
        rewriter, loc, indexType, adaptor.getUpperBound());
    auto step = typeConverter->materializeSourceConversion(
        rewriter, loc, indexType, adaptor.getStep());
    if (!lBound || !uBound || !step)
      return failure();

    // Unpacked the iteration arguments.
    SmallVector<Value> flatArgs;
    for (Value arg : adaptor.getInitArgs())
      unpackUnrealizedConversionCast(arg, flatArgs);

    // We can not do clone as the number of result types after conversion might
    // be different.
    scf::ForOp newOp =
        rewriter.create<scf::ForOp>(loc, lBound, uBound, step, flatArgs);

    // Reserve whatever attributes in the original op.
    newOp->setAttrs(op->getAttrs());

    // We do not need the empty block created by rewriter.
    rewriter.eraseBlock(newOp.getBody(0));
    // Inline the type converted region from the original operation.
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Pack the return value.
    SmallVector<Value, 6> packedRets;
    for (unsigned i = 1, e = offsets.size(); i < e; i++) {
      unsigned start = offsets[i - 1], end = offsets[i];
      unsigned len = end - start;
      ValueRange mappedValue = newOp.getResults().slice(start, len);
      if (len != 1) {
        // 1 : N type conversion.
        Type origType = op.getResultTypes()[i - 1];
        Value mat = typeConverter->materializeSourceConversion(
            rewriter, loc, origType, mappedValue);
        if (!mat)
          return rewriter.notifyMatchFailure(
              op, "Failed to materialize 1:N type conversion");
        packedRets.push_back(mat);
      } else {
        // 1 : 1 type conversion.
        packedRets.push_back(mappedValue.front());
      }
    }

    rewriter.replaceOp(op, packedRets);
    return success();
  }
};

// CRTP
// A base class that takes care of 1:N type conversion, which maps the converted
// op results (computed by the derived class) and materializes 1:N conversion.
template <typename SourceOp, typename ConcretePattern>
class Structural1ToNConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::typeConverter;
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<SourceOp>::OpAdaptor;

  //
  // Derived classes should provide the following method which performs the
  // actual conversion. It should return std::nullopt upon conversion failure
  // and return the converted operation upon success.
  //
  // std::optional<SourceOp> convertSourceOp(SourceOp op, OpAdaptor adaptor,
  //                                    ConversionPatternRewriter &rewriter,
  //                                    TypeRange dstTypes) const;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> dstTypes;
    SmallVector<unsigned> offsets;
    offsets.push_back(0);
    // Do the type conversion and record the offsets.
    for (Type type : op.getResultTypes()) {
      if (failed(typeConverter->convertTypes(type, dstTypes)))
        return rewriter.notifyMatchFailure(op, "could not convert result type");
      offsets.push_back(dstTypes.size());
    }

    // Calls the actual converter implementation to convert the operation.
    std::optional<SourceOp> newOp =
        static_cast<const ConcretePattern *>(this)->convertSourceOp(
            op, adaptor, rewriter, dstTypes);

    if (!newOp)
      return rewriter.notifyMatchFailure(op, "could not convert operation");

    // Packs the return value.
    SmallVector<Value> packedRets;
    for (unsigned i = 1, e = offsets.size(); i < e; i++) {
      unsigned start = offsets[i - 1], end = offsets[i];
      unsigned len = end - start;
      ValueRange mappedValue = newOp->getResults().slice(start, len);
      if (len != 1) {
        // 1 : N type conversion.
        Type origType = op.getResultTypes()[i - 1];
        Value mat = typeConverter->materializeSourceConversion(
            rewriter, op.getLoc(), origType, mappedValue);
        if (!mat) {
          return rewriter.notifyMatchFailure(
              op, "Failed to materialize 1:N type conversion");
        }
        packedRets.push_back(mat);
      } else {
        // 1 : 1 type conversion.
        packedRets.push_back(mappedValue.front());
      }
    }

    rewriter.replaceOp(op, packedRets);
    return success();
  }
};

class ConvertWhileOpTypes
    : public Structural1ToNConversionPattern<scf::WhileOp,
                                             ConvertWhileOpTypes> {
public:
  ConvertWhileOpTypes(TypeConverter &typeConverter, MLIRContext *context)
      : Structural1ToNConversionPattern<scf::WhileOp, ConvertWhileOpTypes>(
            typeConverter, context,
            /*benefit*/ 10) {}

  std::optional<scf::WhileOp>
  convertSourceOp(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter,
                  TypeRange dstTypes) const {
    // Unpacked the iteration arguments.
    SmallVector<Value> flatArgs;
    for (Value arg : adaptor.getOperands())
      unpackUnrealizedConversionCast(arg, flatArgs);

    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), dstTypes, flatArgs);

    for (auto i : {0u, 1u}) {
      if (failed(rewriter.convertRegionTypes(&op.getRegion(i), *typeConverter)))
        return std::nullopt;
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
    }
    return newOp;
  }
};

class ConvertYieldOpTypes : public OpConversionPattern<scf::YieldOp> {
public:
  ConvertYieldOpTypes(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<scf::YieldOp>(typeConverter, context,
                                          /*benefit*/ 10) {}

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> unpackedYield;
    for (Value operand : adaptor.getOperands())
      unpackUnrealizedConversionCast(operand, unpackedYield);

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, unpackedYield);
    return success();
  }
};

class ConvertConditionOpTypes : public OpConversionPattern<scf::ConditionOp> {
public:
  ConvertConditionOpTypes(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<scf::ConditionOp>(typeConverter, context,
                                              /*benefit*/ 10) {}

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> unpackedYield;
    for (Value operand : adaptor.getOperands())
      unpackUnrealizedConversionCast(operand, unpackedYield);

    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(unpackedYield); });
    return success();
  }
};
} // namespace

namespace {
// TODO: upstream
struct CallOpSignatureConversion
    : public OpConversionPattern<mlir::func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(converter->convertTypes(op.getResultTypes(), resultTypes)))
      return mlir::failure();

    llvm::SmallVector<mlir::Type> unpackedTypes;
    llvm::SmallVector<mlir::Value> unpackedArgs;
    for (auto arg : adaptor.getOperands()) {
      auto origType = arg.getType();
      unpackedTypes.clear();
      if (mlir::failed(converter->convertTypes(origType, unpackedTypes)))
        return mlir::failure();

      if (unpackedTypes.empty())
        continue;

      if (unpackedTypes.size() == 1 && unpackedTypes.front() == origType) {
        unpackedArgs.emplace_back(arg);
        continue;
      }

      unpackUnrealizedConversionCast(arg, unpackedArgs);
    }

    auto loc = op.getLoc();
    auto newOp = rewriter.create<mlir::func::CallOp>(loc, op.getCallee(),
                                                     resultTypes, unpackedArgs);

    auto newResults = packResults(rewriter, loc, *converter,
                                  op.getResultTypes(), newOp.getResults());
    if (!newResults)
      return mlir::failure();

    rewriter.replaceOp(op, *newResults);
    return success();
  }
};
} // namespace

void numba::populateControlFlowTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                            typeConverter);
  target.markUnknownOpDynamicallyLegal(
      [&](mlir::Operation *op) -> std::optional<bool> {
        if (auto func = mlir::dyn_cast<mlir::FunctionOpInterface>(op)) {
          if (typeConverter.isSignatureLegal(
                  func.getFunctionType().cast<mlir::FunctionType>()) &&
              typeConverter.isLegal(&func.getFunctionBody()))
            return true;
        } else if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
          if (typeConverter.isLegal(call))
            return true;
        } else if (mlir::isNotBranchOpInterfaceOrReturnLikeOp(op) ||
                   mlir::isLegalForBranchOpInterfaceTypeConversionPattern(
                       op, typeConverter) ||
                   mlir::isLegalForReturnOpTypeConversionPattern(op,
                                                                 typeConverter))
          return true;

        return std::nullopt;
      });

  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
  patterns.insert<CallOpSignatureConversion>(typeConverter,
                                             patterns.getContext());

  target.addDynamicallyLegalOp<mlir::arith::SelectOp>(
      [&](mlir::Operation *op) -> std::optional<bool> {
        if (typeConverter.isLegal(op))
          return true;

        return std::nullopt;
      });

  target.addDynamicallyLegalOp<numba::util::EnvironmentRegionOp>(
      [&](numba::util::EnvironmentRegionOp op) -> std::optional<bool> {
        if (typeConverter.isLegal(op.getArgs().getTypes()) &&
            typeConverter.isLegal(op.getResults().getTypes()))
          return true;

        return std::nullopt;
      });

  target.addDynamicallyLegalOp<numba::util::EnvironmentRegionYieldOp>(
      [&](numba::util::EnvironmentRegionYieldOp op) -> std::optional<bool> {
        if (typeConverter.isLegal(op.getResults().getTypes()))
          return true;

        return std::nullopt;
      });

  mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);

  patterns.insert<ConvertSelectOp, ConvertEnvRegionYield, ConvertEnvRegion,
                  ConvertForOpTypes, ConvertWhileOpTypes, ConvertYieldOpTypes,
                  ConvertConditionOpTypes>(typeConverter,
                                           patterns.getContext());
}

namespace {
struct BuildTupleConversionPattern
    : public mlir::OpConversionPattern<numba::util::BuildTupleOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::BuildTupleOp op,
                  numba::util::BuildTupleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto retType =
        mlir::TupleType::get(op.getContext(), adaptor.getArgs().getTypes());
    rewriter.replaceOpWithNewOp<numba::util::BuildTupleOp>(op, retType,
                                                           adaptor.getArgs());
    return mlir::success();
  }
};

static bool isUniTuple(mlir::TupleType type) {
  auto count = type.size();
  if (count == 0)
    return false;

  auto elemType = type.getType(0);
  for (auto i : llvm::seq<size_t>(1, count)) {
    if (type.getType(i) != elemType)
      return false;
  }
  return true;
}

struct GetItemTupleConversionPattern
    : public mlir::OpConversionPattern<numba::util::TupleExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::TupleExtractOp op,
                  numba::util::TupleExtractOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto container = adaptor.getSource();
    auto containerType = container.getType().dyn_cast<mlir::TupleType>();
    if (!containerType || containerType.size() == 0)
      return mlir::failure();

    auto &converter = *getTypeConverter();

    auto retType = converter.convertType(op.getType());
    if (!retType)
      return mlir::failure();

    auto index = adaptor.getIndex();
    if (isUniTuple(containerType)) {
      if (retType != containerType.getType(0))
        return mlir::failure();
    } else {
      auto constIndex = mlir::getConstantIntValue(index);
      if (!constIndex)
        return mlir::failure();

      auto i = *constIndex;
      if (i < 0 || i >= static_cast<int64_t>(containerType.size()) ||
          containerType.getType(i) != retType)
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<numba::util::TupleExtractOp>(op, retType,
                                                             container, index);
    return mlir::success();
  }
};
} // namespace

void numba::populateTupleTypeConverter(mlir::TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](mlir::TupleType type) -> std::optional<mlir::Type> {
        auto count = static_cast<unsigned>(type.size());
        llvm::SmallVector<mlir::Type> newTypes(count);
        bool changed = false;
        for (auto i : llvm::seq(0u, count)) {
          auto oldType = type.getType(i);
          auto newType = typeConverter.convertType(oldType);
          if (!newType)
            return std::nullopt;

          changed = changed || (newType != oldType);
          newTypes[i] = newType;
        }
        if (!changed)
          return std::nullopt;

        auto ret = mlir::TupleType::get(type.getContext(), newTypes);
        assert(ret != type);
        return ret;
      });
}

void numba::populateTupleTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  patterns.insert<BuildTupleConversionPattern, GetItemTupleConversionPattern>(
      typeConverter, patterns.getContext());

  target.addDynamicallyLegalOp<numba::util::BuildTupleOp>(
      [&typeConverter](numba::util::BuildTupleOp op) {
        return typeConverter.isLegal(op.getResult().getType());
      });

  target.addDynamicallyLegalOp<numba::util::TupleExtractOp>(
      [&typeConverter](numba::util::TupleExtractOp op) -> std::optional<bool> {
        auto inputType = op.getSource().getType();
        auto tupleType = typeConverter.convertType(inputType)
                             .dyn_cast_or_null<mlir::TupleType>();
        if (!tupleType)
          return std::nullopt;

        auto dstType = op.getType();
        auto srcType = [&]() -> mlir::Type {
          if (auto index = mlir::getConstantIntValue(op.getIndex())) {
            auto i = *index;
            auto size = static_cast<unsigned>(tupleType.size());
            if (i >= 0 && i < size)
              return tupleType.getType(static_cast<size_t>(i));
          } else if (isUniTuple(tupleType)) {
            return tupleType.getType(0);
          }
          return dstType;
        }();
        if (!srcType)
          return std::nullopt;

        return inputType == tupleType && srcType == dstType &&
               dstType == typeConverter.convertType(dstType);
      });
}
