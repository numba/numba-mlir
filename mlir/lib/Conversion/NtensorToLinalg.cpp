// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/NtensorToLinalg.hpp"

#include "numba/Analysis/AliasAnalysis.hpp"
#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Dialect/numba_util/Utils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

static const constexpr llvm::StringLiteral kReadonly("ntensor_readonly");

static mlir::RankedTensorType toTensorType(mlir::ShapedType type) {
  return mlir::RankedTensorType::get(type.getShape(), type.getElementType());
}

static mlir::MemRefType toMemrefType(mlir::ShapedType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

namespace {
template <typename Op>
static numba::ntensor::NTensorType getNTensorType(Op op) {
  return mlir::dyn_cast<numba::ntensor::NTensorType>(op.getType());
}

struct ConvertCreateOp
    : public mlir::OpRewritePattern<numba::ntensor::CreateArrayOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CreateArrayOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasAttr(kReadonly))
      return mlir::failure();

    auto dstType = getNTensorType(op);
    if (!dstType)
      return mlir::failure();

    auto elemType = dstType.getElementType();
    auto initValue = op.getInitValue();
    if (initValue && initValue.getType() != elemType)
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op.getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto tensorType = toTensorType(dstType);
          mlir::Value result = builder.create<mlir::tensor::EmptyOp>(
              loc, tensorType, op.getDynamicSizes());
          if (initValue)
            result =
                builder.create<mlir::linalg::FillOp>(loc, initValue, result)
                    .getResult(0);

          result = builder.create<numba::ntensor::FromTensorOp>(loc, dstType,
                                                                result);
          return result;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertCopyOp : public mlir::OpRewritePattern<numba::ntensor::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = getNTensorType(src);
    if (!srcType)
      return mlir::failure();

    auto dst = op.getTarget();
    auto dstType = getNTensorType(dst);
    if (!dstType)
      return mlir::failure();

    if (srcType.getRank() != dstType.getRank() ||
        srcType.getElementType() != dstType.getElementType() ||
        srcType.getEnvironment() != dstType.getEnvironment())
      return mlir::failure();

    numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), std::nullopt,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto rank = static_cast<unsigned>(srcType.getRank());

          auto srcTensorType = toTensorType(srcType);
          mlir::Value srcTensor =
              builder.create<numba::ntensor::ToTensorCopyOp>(loc, srcTensorType,
                                                             src);

          auto dstMemrefType = mlir::MemRefType::get(dstType.getShape(),
                                                     dstType.getElementType());
          mlir::Value dstMemref = builder.create<numba::ntensor::ToMemrefOp>(
              loc, dstMemrefType, dst);

          auto affineMap = mlir::AffineMap::getMultiDimIdentityMap(
              rank, builder.getContext());
          const mlir::AffineMap maps[] = {
              affineMap,
              affineMap,
          };

          llvm::SmallVector<mlir::utils::IteratorType> iterators(
              rank, mlir::utils::IteratorType::parallel);
          auto bodyBuilder = [](mlir::OpBuilder &b, mlir::Location l,
                                mlir::ValueRange args) {
            assert(args.size() == 2);
            b.create<mlir::linalg::YieldOp>(l, args.front());
          };
          auto memref = builder.create<mlir::bufferization::ToMemrefOp>(
              loc, toMemrefType(srcType), srcTensor);
          builder.create<mlir::linalg::GenericOp>(
              loc, memref.getResult(), dstMemref, maps, iterators, bodyBuilder);
          return std::nullopt;
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static bool isAllTensor(mlir::TypeRange types) {
  return llvm::all_of(types, [](mlir::Type type) {
    return type.isa<numba::ntensor::NTensorType>();
  });
}

struct ConvertElementwiseOp
    : public mlir::OpRewritePattern<numba::ntensor::ElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::ElementwiseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::ValueRange src = op.getInputs();
    mlir::TypeRange srcType = src.getTypes();
    if (srcType.empty() || !isAllTensor(srcType))
      return mlir::failure();

    mlir::TypeRange dstType = op.getResultTypes();
    if (dstType.empty() || !isAllTensor(dstType))
      return mlir::failure();

    auto type = srcType.front().cast<numba::ntensor::NTensorType>();

    for (auto range : {srcType.drop_front(), dstType}) {
      for (auto t : range) {
        auto nt = t.cast<numba::ntensor::NTensorType>();
        if (nt.getRank() != type.getRank() ||
            nt.getEnvironment() != type.getEnvironment())
          return mlir::failure();
      }
    }

    auto results = numba::util::wrapEnvRegion(
        rewriter, op.getLoc(), type.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto rank = static_cast<unsigned>(type.getRank());

          llvm::SmallVector<mlir::Value> inputs(src.size());
          for (auto &&[i, arg] : llvm::enumerate(src)) {
            auto srcTensorType =
                toTensorType(arg.getType().cast<numba::ntensor::NTensorType>());
            inputs[i] = builder.create<numba::ntensor::ToTensorCopyOp>(
                loc, srcTensorType, arg);
          }

          llvm::SmallVector<mlir::Value> results(dstType.size());
          llvm::SmallVector<mlir::Type> resultTypes(dstType.size());
          llvm::SmallVector<mlir::Value> dynSizes(rank);
          for (auto &&[i, argType] : llvm::enumerate(dstType)) {
            auto dstTensorType =
                toTensorType(argType.cast<numba::ntensor::NTensorType>());

            dynSizes.clear();
            for (auto &&[i, dim] : llvm::enumerate(dstTensorType.getShape()))
              if (mlir::ShapedType::isDynamic(dim))
                dynSizes.emplace_back(builder.create<mlir::tensor::DimOp>(
                    loc, inputs.front(), i));

            results[i] = builder.create<mlir::tensor::EmptyOp>(
                loc, dstTensorType, dynSizes);
            resultTypes[i] = dstTensorType;
          }

          auto affineMap = mlir::AffineMap::getMultiDimIdentityMap(
              rank, builder.getContext());
          llvm::SmallVector<mlir::AffineMap> maps(
              srcType.size() + dstType.size(), affineMap);

          llvm::SmallVector<mlir::utils::IteratorType> iterators(
              rank, mlir::utils::IteratorType::parallel);

          auto generic = builder.create<mlir::linalg::GenericOp>(
              loc, resultTypes, inputs, results, maps, iterators);

          mlir::Region &newRegion = generic.getRegion();
          builder.inlineRegionBefore(op.getRegion(), newRegion,
                                     newRegion.end());

          mlir::Block *block = &newRegion.front();

          for (auto type : resultTypes)
            block->addArgument(type.cast<mlir::ShapedType>().getElementType(),
                               loc);

          {
            auto term = mlir::cast<numba::ntensor::ElementwiseYieldOp>(
                block->getTerminator());
            mlir::OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPoint(term);
            auto args = term.getValues();
            builder.replaceOpWithNewOp<mlir::linalg::YieldOp>(term, args);
          }

          llvm::SmallVector<mlir::Value> res(generic->getNumResults());
          for (auto &&[i, arg] : llvm::enumerate(generic->getResults()))
            res[i] = builder.create<numba::ntensor::FromTensorOp>(
                loc, dstType[i], arg);

          return res;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertCastOp : public mlir::OpRewritePattern<numba::ntensor::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasAttr(kReadonly))
      return mlir::failure();

    auto src = op.getSource();
    auto srcType = getNTensorType(src);
    if (!srcType)
      return mlir::failure();

    auto dstType = getNTensorType(op);
    if (!dstType)
      return mlir::failure();

    if (srcType.getEnvironment() != dstType.getEnvironment())
      return mlir::failure();

    auto srcTensorType = toTensorType(srcType);
    auto dstTensorType = toTensorType(dstType);

    if (!mlir::tensor::CastOp::areCastCompatible(srcTensorType, dstTensorType))
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto srcTensor = builder.create<numba::ntensor::ToTensorCopyOp>(
              loc, srcTensorType, src);
          auto cast = builder.create<mlir::tensor::CastOp>(loc, dstTensorType,
                                                           srcTensor);
          return builder
              .create<numba::ntensor::FromTensorOp>(loc, dstType, cast)
              .getResult();
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertFromElementsOp
    : public mlir::OpRewritePattern<numba::ntensor::FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::FromElementsOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstType = getNTensorType(op);
    if (!dstType)
      return mlir::failure();

    auto elements = op.getElements();
    if (llvm::any_of(elements.getTypes(), [&](mlir::Type t) {
          return t != dstType.getElementType();
        }))
      return mlir::failure();

    auto dstTensorType = toTensorType(dstType);

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto res = builder.create<mlir::tensor::FromElementsOp>(
              loc, dstTensorType, elements);
          return builder.create<numba::ntensor::FromTensorOp>(loc, dstType, res)
              .getResult();
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertSubviewOp
    : public mlir::OpRewritePattern<numba::ntensor::SubviewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::SubviewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasAttr(kReadonly))
      return mlir::failure();

    auto src = op.getSource();
    auto srcType = getNTensorType(src);
    if (!srcType)
      return mlir::failure();

    auto dstType = getNTensorType(op);
    if (!dstType)
      return mlir::failure();

    if (srcType.getEnvironment() != dstType.getEnvironment())
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op.getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto srcTensorType = toTensorType(srcType);
          mlir::Value srcTensor = builder.create<numba::ntensor::ToTensorOp>(
              loc, srcTensorType, src);

          auto offsets = op.getMixedOffsets();
          auto sizes = op.getMixedSizes();
          auto strides = op.getMixedStrides();

          auto dstRank = static_cast<unsigned>(dstType.getRank());
          auto viewTensorType =
              mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                  dstRank, srcTensorType, offsets, sizes, strides);

          mlir::Value view = builder.create<mlir::tensor::ExtractSliceOp>(
              loc, viewTensorType, srcTensor, offsets, sizes, strides);
          mlir::Value result =
              builder.create<numba::ntensor::FromTensorOp>(loc, dstType, view);
          return result;
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertReshapeOp
    : public mlir::OpRewritePattern<numba::util::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasAttr(kReadonly))
      return mlir::failure();

    auto src = op.getSource();
    auto srcType = getNTensorType(src);
    if (!srcType)
      return mlir::failure();

    auto dstType = getNTensorType(op);
    if (!dstType)
      return mlir::failure();

    if (srcType.getEnvironment() != dstType.getEnvironment())
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op.getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto srcTensorType = toTensorType(srcType);
          auto dstTensorType = toTensorType(dstType);
          mlir::Value srcTensor = builder.create<numba::ntensor::ToTensorOp>(
              loc, srcTensorType, src);

          mlir::Value reshaped = builder.create<numba::util::ReshapeOp>(
              loc, dstTensorType, srcTensor, op.getShape());
          mlir::Value result = builder.create<numba::ntensor::FromTensorOp>(
              loc, dstType, reshaped);
          return result;
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertLoadOp : public mlir::OpRewritePattern<numba::ntensor::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getArray();
    auto srcType = getNTensorType(src);
    if (!srcType || op.getType() != srcType.getElementType())
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), srcType.getEnvironment(),
        srcType.getElementType(),
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto srcTensorType = toTensorType(srcType);
          mlir::Value srcTensor =
              builder.create<numba::ntensor::ToTensorCopyOp>(loc, srcTensorType,
                                                             src);

          mlir::Value result = builder.create<mlir::tensor::ExtractOp>(
              loc, srcTensor, op.getIndices());
          return result;
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertDimOp : public mlir::OpRewritePattern<numba::ntensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = getNTensorType(src);
    if (!srcType)
      return mlir::failure();

    auto results = numba::util::wrapEnvRegion(
        rewriter, op->getLoc(), srcType.getEnvironment(),
        rewriter.getIndexType(),
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto tensorType = toTensorType(srcType);
          mlir::Value tensor =
              builder.create<numba::ntensor::ToTensorOp>(loc, tensorType, src);
          mlir::Value result =
              builder.create<mlir::tensor::DimOp>(loc, tensor, op.getIndex());
          return result;
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

/// Get broadcasted dimension value from 2 values, if v1 value is equal to 1
/// or dims are equal then select val2 otherwise val1.
static mlir::Value broadcastDim(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value val1, mlir::Value val2) {
  assert(val1.getType().isa<mlir::IndexType>());
  assert(val2.getType().isa<mlir::IndexType>());
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto isOne = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, one, val1);
  auto isSame = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, val1, val2);
  auto tmp = builder.create<mlir::arith::AndIOp>(loc, isOne, isSame);
  return builder.create<mlir::arith::SelectOp>(loc, tmp, val2, val1);
}

/// Generate code for expanding specified dim of value src to corresponding
/// value in targetShape. Assume src dimension is either 1 or equal to the
/// target shape.
static mlir::Value expandDim(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value initial, mlir::Value src, unsigned dim,
                             mlir::ValueRange targetShape) {
  assert(initial.getType().isa<mlir::RankedTensorType>());
  assert(src.getType().isa<mlir::RankedTensorType>());
  auto context = builder.getContext();
  auto srcType = src.getType().cast<mlir::ShapedType>();
  auto numDims = static_cast<unsigned>(srcType.getRank());
  auto shape = llvm::to_vector(srcType.getShape());
  shape[dim] = mlir::ShapedType::kDynamic;
  mlir::Type targetType =
      mlir::RankedTensorType::get(shape, srcType.getElementType());
  auto dimVal = builder.create<mlir::tensor::DimOp>(loc, initial, dim);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value cond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, one, dimVal);
  mlir::Value cond2 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, targetShape[dim], dimVal);
  cond = builder.create<mlir::arith::AndIOp>(loc, cond, cond2);
  llvm::SmallVector<mlir::OpFoldResult> newShape(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    if (i == dim) {
      newShape[i] = targetShape[i];
    } else {
      newShape[i] =
          builder.create<mlir::tensor::DimOp>(loc, src, i).getResult();
    }
  }
  auto trueBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
    assert(dim < shape.size());
    shape[dim] = 1;
    auto init = builder
                    .create<mlir::tensor::EmptyOp>(loc, newShape,
                                                   srcType.getElementType())
                    .getResult();
    llvm::SmallVector<mlir::AffineExpr> exprs(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      if (i == dim) {
        exprs[i] = mlir::getAffineConstantExpr(0, context);
      } else {
        exprs[i] = mlir::getAffineDimExpr(i, context);
      }
    }
    const mlir::AffineMap maps[] = {
        mlir::AffineMap::get(numDims, 0, exprs, context),
        mlir::AffineMap::getMultiDimIdentityMap(numDims, context),
    };
    llvm::SmallVector<mlir::utils::IteratorType> iterators(
        numDims, mlir::utils::IteratorType::parallel);

    auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::ValueRange values) {
      assert(values.size() == 2);
      builder.create<mlir::linalg::YieldOp>(loc, values[0]);
    };

    auto expanded = builder.create<mlir::linalg::GenericOp>(
        loc, init.getType(), src, init, maps, iterators, body);
    auto res = builder.createOrFold<mlir::tensor::CastOp>(
        loc, targetType, expanded.getResult(0));
    builder.create<mlir::scf::YieldOp>(loc, res);
  };
  auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
    mlir::Value res = src;
    if (res.getType() != targetType)
      res = builder.create<mlir::tensor::CastOp>(loc, targetType, src);
    builder.create<mlir::scf::YieldOp>(loc, res);
  };
  return builder.create<mlir::scf::IfOp>(loc, cond, trueBody, falseBody)
      .getResult(0);
}

/// Expand all dims of val to targetShape.
static mlir::Value expandDims(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value val, unsigned numDims,
                              mlir::ValueRange targetShape) {
  assert(numDims <= targetShape.size());
  if (numDims < targetShape.size())
    targetShape = targetShape.drop_front(targetShape.size() - numDims);

  mlir::Value current = val;
  for (unsigned i = 0; i < numDims; ++i)
    current = expandDim(builder, loc, val, current, i, targetShape);

  if (!targetShape.empty())
    current =
        builder.create<numba::util::EnforceShapeOp>(loc, current, targetShape);
  return current;
}

template <typename C> static auto getTempShape(const C &container) {
  return llvm::SmallVector<mlir::OpFoldResult>(std::begin(container),
                                               std::end(container));
}

struct ConvertBroadcastOp
    : public mlir::OpRewritePattern<numba::ntensor::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::BroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::ValueRange inputs = op.getInputs();
    if (inputs.empty())
      return mlir::failure();

    mlir::ValueRange results = op.getResults();
    assert(inputs.size() == results.size());

    for (auto &&[src, dst] : llvm::zip(inputs, results))
      if (mlir::cast<mlir::ShapedType>(src.getType()).getElementType() !=
          mlir::cast<mlir::ShapedType>(dst.getType()).getElementType())
        return mlir::failure();

    auto env = mlir::cast<numba::ntensor::NTensorType>(inputs.front().getType())
                   .getEnvironment();
    for (auto args : {inputs.drop_front(), results})
      for (auto arg : args)
        if (mlir::cast<numba::ntensor::NTensorType>(arg.getType())
                .getEnvironment() != env)
          return mlir::failure();

    mlir::TypeRange resultTypes = op->getResultTypes();

    auto newResults = numba::util::wrapEnvRegion(
        rewriter, op.getLoc(), env, resultTypes,
        [&](mlir::OpBuilder &rewriter, mlir::Location loc) {
          llvm::SmallVector<mlir::Value> tensorInputs(inputs.size());
          for (auto &&[i, input] : llvm::enumerate(inputs)) {
            auto tensorType = toTensorType(
                input.getType().cast<numba::ntensor::NTensorType>());
            tensorInputs[i] = rewriter.create<numba::ntensor::ToTensorCopyOp>(
                loc, tensorType, input);
          }

          using ShapeT = llvm::SmallVector<mlir::Value>;
          auto getShape = [&](mlir::Value val) -> ShapeT {
            auto tensorType = val.getType().cast<mlir::RankedTensorType>();

            auto rank = static_cast<unsigned>(tensorType.getRank());
            ShapeT retShape(rank);
            for (auto i : llvm::seq(0u, rank))
              retShape[i] = rewriter.create<mlir::tensor::DimOp>(loc, val, i);

            return retShape;
          };

          // Compute resulting size
          auto retShape = getShape(tensorInputs.front());

          for (auto input : llvm::ArrayRef(tensorInputs).drop_front()) {
            auto newShape = getShape(input);

            for (auto &&[dim, newDim] :
                 llvm::zip(llvm::reverse(retShape), llvm::reverse(newShape))) {
              dim = broadcastDim(rewriter, loc, dim, newDim);
            }
            if (newShape.size() > retShape.size()) {
              auto front = llvm::ArrayRef(newShape).drop_back(retShape.size());
              assert(!front.empty());
              retShape.insert(retShape.begin(), front.begin(), front.end());
            }
          }

          auto context = getContext();
          auto dstRank = static_cast<unsigned>(retShape.size());

          // Broadcast individual arrays
          llvm::SmallVector<mlir::Value> newResults(tensorInputs.size());
          for (auto &&[i, input] : llvm::enumerate(tensorInputs)) {
            auto srcType = input.getType().cast<mlir::ShapedType>();
            auto srcRank = static_cast<unsigned>(srcType.getRank());
            auto result = expandDims(rewriter, loc, input, srcRank, retShape);

            auto resultType =
                mlir::cast<mlir::ShapedType>(results[i].getType());
            if (srcRank != dstRank) {
              auto elementType = srcType.getElementType();
              auto resultTensorType = toTensorType(resultType);
              auto init = rewriter
                              .create<mlir::tensor::EmptyOp>(
                                  loc, getTempShape(retShape), elementType)
                              .getResult();

              const mlir::AffineMap maps[] = {
                  mlir::AffineMap::getMinorIdentityMap(dstRank, srcRank,
                                                       context),
                  mlir::AffineMap::getMultiDimIdentityMap(dstRank, context),
              };
              llvm::SmallVector<mlir::utils::IteratorType> iterators(
                  dstRank, mlir::utils::IteratorType::parallel);
              auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::ValueRange values) {
                assert(values.size() == 2);
                auto res = values[0];
                builder.create<mlir::linalg::YieldOp>(loc, res);
              };
              result = rewriter
                           .create<mlir::linalg::GenericOp>(loc, init.getType(),
                                                            result, init, maps,
                                                            iterators, body)
                           .getResult(0);
              if (result.getType() != resultTensorType)
                result = rewriter.create<mlir::tensor::CastOp>(
                    loc, resultTensorType, result);
            }
            auto tempResultType =
                mlir::cast<mlir::ShapedType>(result.getType());
            if (tempResultType.getShape() != resultType.getShape()) {
              auto tempTensorType = tempResultType.clone(resultType.getShape());
              result = rewriter.create<mlir::tensor::CastOp>(
                  loc, tempTensorType, result);
            }

            result = rewriter.create<numba::ntensor::FromTensorOp>(
                loc, resultType, result);
            newResults[i] = result;
          }
          return newResults;
        });

    rewriter.replaceOp(op, newResults);
    return mlir::success();
  }
};
} // namespace

void numba::populateNtensorToLinalgPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertCreateOp, ConvertCopyOp, ConvertElementwiseOp,
                  ConvertCastOp, ConvertFromElementsOp, ConvertSubviewOp,
                  ConvertReshapeOp, ConvertLoadOp, ConvertDimOp,
                  ConvertBroadcastOp>(patterns.getContext());
}

namespace {
struct NtensorAliasAnalysisPass
    : public mlir::PassWrapper<NtensorAliasAnalysisPass,
                               mlir::InterfacePass<mlir::FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorAliasAnalysisPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<numba::ntensor::NTensorDialect>();
  }

  void runOnOperation() override {
    auto &context = getContext();
    auto func = getOperation();

    llvm::SmallVector<mlir::Operation *, 0> writers;
    func->walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::CallOpInterface>(op)) {
        writers.emplace_back(op);
        return;
      }

      if (!mlir::isa<mlir::memref::MemRefDialect, mlir::linalg::LinalgDialect,
                     numba::ntensor::NTensorDialect>(op->getDialect()))
        return;

      auto memInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
      if (!memInterface || memInterface.hasEffect<mlir::MemoryEffects::Write>())
        writers.emplace_back(op);
    });

    bool hasWriters = !writers.empty();
    auto *analysis = [&]() -> mlir::AliasAnalysis * {
      if (!hasWriters)
        return nullptr;

      return &getAnalysis<numba::AliasAnalysis>();
    }();

    auto getTensor = [](mlir::Operation *op) -> mlir::Value {
      assert(op);
      if (auto subview = mlir::dyn_cast<numba::ntensor::SubviewOp>(op))
        return subview.getResult();

      if (auto create = mlir::dyn_cast<numba::ntensor::CreateArrayOp>(op))
        return create.getResult();

      if (auto cast = mlir::dyn_cast<numba::ntensor::CastOp>(op))
        return cast.getDest();

      if (auto reshape = mlir::dyn_cast<numba::util::ReshapeOp>(op))
        return reshape.getSource();

      return {};
    };

    auto attrName = mlir::StringAttr::get(&context, kReadonly);
    auto unitAttr = mlir::UnitAttr::get(&context);
    func->walk([&](mlir::Operation *op) {
      if (auto tens = getTensor(op)) {
        if (hasWriters) {
          op->removeAttr(attrName);
          assert(analysis);
          for (auto writer : writers) {
            assert(writer);
            if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(writer)) {
              for (auto arg : call.getArgOperands())
                if (!analysis->alias(tens, arg).isNo())
                  return;

            } else if (analysis->getModRef(writer, tens).isMod())
              return;
          }
        }
        op->setAttr(attrName, unitAttr);
      }
    });
    markAllAnalysesPreserved();
  }
};

struct NtensorToLinalgPass
    : public mlir::PassWrapper<NtensorToLinalgPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorToLinalgPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<numba::ntensor::NTensorDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    numba::populateNtensorToLinalgPatterns(patterns);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

struct NtensorLowerToTensorCopyPass
    : public mlir::PassWrapper<NtensorLowerToTensorCopyPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorLowerToTensorCopyPass)

  void runOnOperation() override {
    llvm::SmallVector<numba::ntensor::ToTensorCopyOp> toProcess;
    llvm::SmallSetVector<mlir::Value, 8> writers;

    llvm::SmallVector<mlir::Value> tmp;
    getOperation()->walk([&](mlir::Operation *op) {
      if (auto toTensor = mlir::dyn_cast<numba::ntensor::ToTensorCopyOp>(op)) {
        toProcess.emplace_back(toTensor);
        return;
      }

      tmp.clear();
      if (numba::isWriter(*op, tmp))
        writers.insert(tmp.begin(), tmp.end());
    });

    if (toProcess.empty())
      return markAllAnalysesPreserved();

    auto &&AA = getAnalysis<numba::AliasAnalysis>();
    auto hasWrite = [&](mlir::Value val) -> bool {
      for (auto &&writer : writers)
        if (!AA.alias(val, writer).isNo())
          return true;

      return false;
    };

    mlir::OpBuilder builder(&getContext());
    for (auto toTensor : toProcess) {
      auto loc = toTensor.getLoc();
      auto resType = mlir::cast<mlir::TensorType>(toTensor.getType());
      auto src = toTensor.getArray();

      builder.setInsertionPoint(toTensor);
      if (!hasWrite(toTensor.getArray())) {
        auto res =
            builder.create<numba::ntensor::ToTensorOp>(loc, resType, src);
        toTensor->replaceAllUsesWith(res->getResults());
        toTensor->erase();
      }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createNtensorAliasAnalysisPass() {
  return std::make_unique<NtensorAliasAnalysisPass>();
}

std::unique_ptr<mlir::Pass> numba::createNtensorToLinalgPass() {
  return std::make_unique<NtensorToLinalgPass>();
}

std::unique_ptr<mlir::Pass> numba::createNtensorLowerToTensorCopyPass() {
  return std::make_unique<NtensorLowerToTensorCopyPass>();
}
