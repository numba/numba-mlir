// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"

#include "numba/Dialect/numba_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/TypeSwitch.h>

namespace {
struct NTensorInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final override {
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *, bool,
                       mlir::IRMapping &) const final override {
    return true;
  }
};
} // namespace

void numba::ntensor::NTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "numba/Dialect/ntensor/IR/NTensorOps.cpp.inc"
      >();

  addInterfaces<NTensorInlinerInterface>();

  addTypes<
#define GET_TYPEDEF_LIST
#include "numba/Dialect/ntensor/IR/NTensorOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "numba/Dialect/ntensor/IR/NTensorOpsAttributes.cpp.inc"
      >();
}

mlir::Operation *numba::ntensor::NTensorDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(
        loc, type, mlir::cast<mlir::TypedAttr>(value));

  if (type.isa<mlir::IndexType>())
    if (auto val = mlir::getConstantIntValue(value))
      return builder.create<mlir::arith::ConstantIndexOp>(loc, *val);

  return nullptr;
}

namespace {
struct NtensorReshapeSimplify
    : public mlir::OpRewritePattern<numba::util::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value src = op.getSource();
    auto srcType = mlir::dyn_cast<numba::ntensor::NTensorType>(src.getType());
    if (!srcType || srcType.getRank() != 1)
      return mlir::failure();

    auto dstType =
        mlir::dyn_cast<numba::ntensor::NTensorType>(op.getResult().getType());
    if (!dstType)
      return mlir::failure();

    auto srcRank = static_cast<unsigned>(srcType.getRank());
    auto dstRank = static_cast<unsigned>(dstType.getRank());
    auto newShape = op.getShape();
    if (newShape.size() != dstRank)
      return mlir::failure();

    if (srcRank == 1 && dstRank == 1) {
      mlir::OpFoldResult offset = rewriter.getIndexAttr(0);
      mlir::OpFoldResult size = newShape.front();
      mlir::OpFoldResult stride = rewriter.getIndexAttr(1);
      auto loc = op.getLoc();
      mlir::Value res = rewriter.create<numba::ntensor::SubviewOp>(
          loc, src, offset, size, stride);
      if (res.getType() != dstType)
        res = rewriter.create<numba::ntensor::CastOp>(loc, dstType, res);

      rewriter.replaceOp(op, res);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

void numba::ntensor::NTensorDialect::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results) const {
  results.add<NtensorReshapeSimplify>(getContext());
}

bool numba::ntensor::NTensorBase::hasRank() const { return true; }

llvm::ArrayRef<int64_t> numba::ntensor::NTensorBase::getShape() const {
  return cast<NTensorType>().getShape();
}

numba::ntensor::NTensorBase numba::ntensor::NTensorBase::cloneWith(
    std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
  auto t = cast<NTensorType>();
  return NTensorType::get(shape.value_or(getShape()), elementType,
                          t.getEnvironment(), t.getLayout());
}

bool numba::ntensor::NTensorBase::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat() || type.isa<mlir::ComplexType>();
}

static mlir::Value handleElemIndexVars(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value source,
                                       mlir::Value size) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto isNeg = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, source, zero);
  auto negIndex = builder.create<mlir::arith::AddIOp>(loc, size, source);
  auto posIndex =
      builder.create<mlir::arith::SelectOp>(loc, isNeg, negIndex, source);
  return posIndex;
}

static mlir::Value handleSliceIndexVars(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value source,
                                        mlir::Value size) {
  auto posIndex = handleElemIndexVars(builder, loc, source, size);
  auto isOutOfRange = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, posIndex, size);
  return builder.create<mlir::arith::SelectOp>(loc, isOutOfRange, size,
                                               posIndex);
}

static mlir::Value computeCount(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value begin, mlir::Value end,
                                mlir::Value step) {
  auto size = builder.createOrFold<mlir::arith::SubIOp>(loc, end, begin);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  size = builder.createOrFold<mlir::arith::SubIOp>(loc, size, one);
  size = builder.createOrFold<mlir::arith::AddIOp>(loc, size, step);
  size = builder.createOrFold<mlir::arith::DivUIOp>(loc, size, step);
  return size;
}

namespace {
struct ResolveSlicePropagate
    : public mlir::OpRewritePattern<numba::ntensor::ResolveSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::ResolveSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto buildSlice =
        op.getSlice().getDefiningOp<numba::ntensor::BuildSliceOp>();
    if (!buildSlice)
      return mlir::failure();

    auto loc = op.getLoc();
    auto size = op.getSize();
    std::array<mlir::Value, 4> results;
    if (auto begin = buildSlice.getBegin()) {
      results[0] = handleElemIndexVars(rewriter, loc, begin, size);
    } else {
      results[0] = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    }

    if (auto end = buildSlice.getEnd()) {
      results[1] = handleSliceIndexVars(rewriter, loc, end, size);
    } else {
      results[1] = size;
    }

    if (auto step = buildSlice.getStep()) {
      results[2] = step;
    } else {
      results[2] = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    }

    results[3] =
        computeCount(rewriter, loc, results[0], results[1], results[2]);

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::ResolveSliceOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<ResolveSlicePropagate>(context);
}

namespace {
struct LoadCastFold : public mlir::OpRewritePattern<numba::ntensor::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto array = op.getArray();
    auto src = array.getDefiningOp<numba::ntensor::CastOp>();
    if (!src)
      return mlir::failure();

    auto srcArray = src.getSource();
    auto srcArrayType = srcArray.getType().cast<numba::ntensor::NTensorType>();
    auto dstArrayType = array.getType().cast<numba::ntensor::NTensorType>();
    if (srcArrayType.getElementType() != dstArrayType.getElementType() ||
        srcArrayType.getEnvironment() != dstArrayType.getEnvironment())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::ntensor::LoadOp>(op, srcArray,
                                                        op.getIndices());
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::LoadOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<LoadCastFold>(context);
}

static std::optional<mlir::Value>
foldLoadFromElements(mlir::Value src, mlir::ValueRange indices) {
  auto fromElements = src.getDefiningOp<numba::ntensor::FromElementsOp>();
  if (!fromElements)
    return std::nullopt;

  if (indices.size() != 1)
    return std::nullopt;

  auto idxVal = mlir::getConstantIntValue(indices.front());
  if (!idxVal || *idxVal < 0)
    return std::nullopt;

  auto idx = static_cast<size_t>(*idxVal);
  auto args = fromElements.getElements();
  if (idx >= args.size())
    return std::nullopt;

  for (auto user : fromElements.getResult().getUsers())
    if (!mlir::isa<numba::ntensor::LoadOp, numba::ntensor::DimOp>(user))
      return std::nullopt;

  return args[idx];
}

mlir::OpFoldResult numba::ntensor::LoadOp::fold(FoldAdaptor) {
  if (auto result = foldLoadFromElements(getArray(), getIndices()))
    return *result;

  return nullptr;
}

namespace {
struct StoreCastFold : public mlir::OpRewritePattern<numba::ntensor::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto array = op.getArray();
    auto src = array.getDefiningOp<numba::ntensor::CastOp>();
    if (!src)
      return mlir::failure();

    auto srcArray = src.getSource();
    auto srcArrayType = srcArray.getType().cast<numba::ntensor::NTensorType>();
    auto dstArrayType = array.getType().cast<numba::ntensor::NTensorType>();
    if (srcArrayType.getElementType() != dstArrayType.getElementType() ||
        srcArrayType.getEnvironment() != dstArrayType.getEnvironment())
      return mlir::failure();

    auto val = op.getValue();
    rewriter.replaceOpWithNewOp<numba::ntensor::StoreOp>(op, val, srcArray,
                                                         op.getIndices());
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::StoreOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<StoreCastFold>(context);
}

namespace {
struct ResolveIndexPropagate
    : public mlir::OpRewritePattern<numba::ntensor::ResolveIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::ResolveIndexOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto res =
        handleElemIndexVars(rewriter, op.getLoc(), op.getIndex(), op.getSize());
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::DimOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  setNameFn(getResult(), "dim");
}

void numba::ntensor::DimOp::build(mlir::OpBuilder &builder,
                                  mlir::OperationState &result,
                                  mlir::Value source, int64_t index) {
  auto loc = result.location;
  auto indexValue = builder.create<mlir::arith::ConstantIndexOp>(loc, index);
  build(builder, result, source, indexValue);
}

std::optional<int64_t> numba::ntensor::DimOp::getConstantIndex() {
  if (auto val = mlir::getConstantIntValue(getIndex()))
    return *val;

  return {};
}

mlir::Speculation::Speculatability numba::ntensor::DimOp::getSpeculatability() {
  auto constantIndex = getConstantIndex();
  if (!constantIndex)
    return mlir::Speculation::NotSpeculatable;

  auto rankedType =
      mlir::dyn_cast<numba::ntensor::NTensorType>(getSource().getType());
  if (!rankedType)
    return mlir::Speculation::NotSpeculatable;

  // The verifier rejects operations that violate this assertion.
  assert(constantIndex < rankedType.getRank());
  return mlir::Speculation::Speculatable;
}

mlir::LogicalResult numba::ntensor::DimOp::verify() {
  // Assume unknown index to be in range.
  std::optional<int64_t> index = getConstantIndex();
  if (!index)
    return mlir::success();

  // Check that constant index is not knowingly out of range.
  auto type = getSource().getType();
  if (auto tensorType = type.dyn_cast<numba::ntensor::NTensorType>()) {
    if (*index >= tensorType.getRank())
      return emitOpError("index is out of range");
  } else {
    llvm_unreachable("expected operand with array type");
  }
  return mlir::success();
}

namespace {
struct FromTensorDimPropagate
    : public mlir::OpRewritePattern<numba::ntensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<numba::ntensor::FromTensorOp>();
    if (!src)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(op, src.getTensor(),
                                                     op.getIndex());
    return mlir::success();
  }
};

struct ToTensorDimPropagate
    : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<numba::ntensor::ToTensorOp>();
    if (!src)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::ntensor::DimOp>(op, src.getArray(),
                                                       op.getIndex());
    return mlir::success();
  }
};

struct ToTensorCopyDimPropagate
    : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<numba::ntensor::ToTensorCopyOp>();
    if (!src)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::ntensor::DimOp>(op, src.getArray(),
                                                       op.getIndex());
    return mlir::success();
  }
};

// TODO: upstream
struct LinalgGenericDimPropagate
    : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto generic = src.getDefiningOp<mlir::linalg::GenericOp>();
    if (!generic)
      return mlir::failure();

    assert(generic.getOutputs().size() == generic.getResults().size());
    auto outIndex = [&]() -> size_t {
      for (auto &&[i, out] : llvm::enumerate(generic.getResults())) {
        if (out == src)
          return i;
      }
      llvm_unreachable("Invalid result");
    }();

    auto out = generic.getOutputs()[outIndex];

    rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(op, out, op.getIndex());
    return mlir::success();
  }
};

// TODO: upstream
struct ExtractSliceDimPropagate
    : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto extract = src.getDefiningOp<mlir::tensor::ExtractSliceOp>();
    if (!extract)
      return mlir::failure();

    auto idx = op.getConstantIndex();
    if (!idx || *idx < 0)
      return mlir::failure();

    auto droppedDims = extract.getDroppedDims();
    auto srcDims = extract.getMixedSizes();
    llvm::SmallVector<mlir::OpFoldResult> dims;
    for (auto &&[i, dim] : llvm::enumerate(srcDims))
      if (!droppedDims[i])
        dims.emplace_back(dim);

    if (*idx >= static_cast<int64_t>(dims.size()))
      return mlir::failure();

    auto srcDim = dims[static_cast<size_t>(*idx)];
    if (auto constIdx = mlir::getConstantIntValue(srcDim)) {
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantIndexOp>(op, *constIdx);
    } else {
      rewriter.replaceOp(op, srcDim.get<mlir::Value>());
    }
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::DimOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<FromTensorDimPropagate, ToTensorDimPropagate,
                 ToTensorCopyDimPropagate, LinalgGenericDimPropagate,
                 ExtractSliceDimPropagate>(context);
}

mlir::OpFoldResult numba::ntensor::DimOp::fold(FoldAdaptor) {
  auto idxVal = getConstantIndex();
  if (!idxVal || *idxVal < 0)
    return nullptr;

  auto idx = static_cast<size_t>(*idxVal);

  auto getIndexVal = [&](int64_t val) {
    return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), val);
  };

  mlir::Value src = getSource();
  auto shape = mlir::cast<mlir::ShapedType>(src.getType()).getShape();
  if (idx < shape.size() && !mlir::ShapedType::isDynamic(shape[idx]))
    return getIndexVal(shape[idx]);

  if (auto create = src.getDefiningOp<numba::ntensor::CreateArrayOp>()) {
    auto sizes = create.getMixedSizes();
    if (idx >= sizes.size())
      return nullptr;

    return sizes[idx];
  }

  if (auto subview = src.getDefiningOp<numba::ntensor::SubviewOp>()) {
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    auto dropped = subview.getDroppedDims();
    for (auto &&[i, s] : llvm::enumerate(subview.getMixedSizes())) {
      if (!dropped[i])
        sizes.emplace_back(s);
    }

    if (idx >= sizes.size())
      return nullptr;

    return sizes[idx];
  }

  if (auto cast = src.getDefiningOp<numba::ntensor::CastOp>()) {
    getSourceMutable().assign(cast.getSource());
    return getResult();
  }

  return nullptr;
}

namespace {
struct FoldSelfCopy : public mlir::OpRewritePattern<numba::ntensor::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CopyOp copyOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (copyOp.getSource() != copyOp.getTarget())
      return mlir::failure();

    rewriter.eraseOp(copyOp);
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::CopyOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<FoldSelfCopy>(context);
}

numba::ntensor::NTensorType numba::ntensor::SubviewOp::inferResultType(
    numba::ntensor::NTensorType sourceType,
    mlir::ArrayRef<int64_t> staticOffsets, mlir::ArrayRef<int64_t> staticSizes,
    mlir::ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank && "staticOffsets length mismatch");
  assert(staticSizes.size() == rank && "staticSizes length mismatch");
  assert(staticStrides.size() == rank && "staticStrides length mismatch");
  return numba::ntensor::NTensorType::get(
      staticSizes, sourceType.getElementType(), sourceType.getEnvironment(),
      sourceType.getLayout());
}

numba::ntensor::NTensorType numba::ntensor::SubviewOp::inferResultType(
    numba::ntensor::NTensorType sourceShapedTensorType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return SubviewOp::inferResultType(sourceShapedTensorType, staticOffsets,
                                    staticSizes, staticStrides);
}

numba::ntensor::NTensorType
numba::ntensor::SubviewOp::inferRankReducedResultType(
    unsigned resultRank, numba::ntensor::NTensorType sourceType,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides) {
  // Type inferred in the absence of rank-reducing behavior.
  auto inferredType = mlir::cast<NTensorType>(
      inferResultType(sourceType, offsets, sizes, strides));
  int rankDiff = inferredType.getRank() - resultRank;
  if (rankDiff > 0) {
    auto shape = inferredType.getShape();
    llvm::SmallBitVector dimsToProject =
        mlir::getPositionsOfShapeOne(rankDiff, shape);
    llvm::SmallVector<int64_t> projectedShape;
    // Best effort rank-reducing: drop 1s in order.
    for (auto pos : llvm::seq<size_t>(0, shape.size()))
      if (!dimsToProject.test(pos))
        projectedShape.push_back(shape[pos]);

    inferredType = mlir::cast<NTensorType>(
        mlir::cast<mlir::ShapedType>(inferredType).clone(projectedShape));
  }
  return inferredType;
}

numba::ntensor::NTensorType
numba::ntensor::SubviewOp::inferRankReducedResultType(
    unsigned resultRank, numba::ntensor::NTensorType sourceType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return SubviewOp::inferRankReducedResultType(
      resultRank, sourceType, staticOffsets, staticSizes, staticStrides);
}

numba::ntensor::NTensorType
numba::ntensor::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, numba::ntensor::NTensorType sourceType,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides) {
  auto inferredType = inferResultType(sourceType, offsets, sizes, strides);
  assert(inferredType.getRank() >= static_cast<int64_t>(resultShape.size()) &&
         "expected ");
  if (inferredType.getRank() == static_cast<int64_t>(resultShape.size()))
    return inferredType;

  assert(mlir::computeRankReductionMask(inferredType.getShape(), resultShape)
             .has_value() &&
         "invalid rank reduction");

  return numba::ntensor::NTensorType::get(
      resultShape, sourceType.getElementType(), sourceType.getEnvironment(),
      sourceType.getLayout());
}

numba::ntensor::NTensorType
numba::ntensor::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, numba::ntensor::NTensorType sourceType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return SubviewOp::inferRankReducedResultType(
      resultShape, sourceType, staticOffsets, staticSizes, staticStrides);
}

// Build a SubViewOp with mixed static and dynamic entries and custom result
// type. If the type passed is nullptr, it is inferred.
void numba::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    numba::ntensor::NTensorType resultType, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceType = source.getType().cast<numba::ntensor::NTensorType>();
  // Structuring implementation this way avoids duplication between builders.
  if (!resultType) {
    resultType = numba::ntensor::SubviewOp::inferResultType(
        sourceType, staticOffsets, staticSizes, staticStrides);
  }
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build a SubViewOp with mixed static and dynamic entries and inferred result
// type.
void numba::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, numba::ntensor::NTensorType(), source, offsets, sizes,
        strides, attrs);
}

// Build a SubViewOp with static entries and inferred result type.
void numba::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, source, offsetValues, sizeValues, strideValues, attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the
// type passed is nullptr, it is inferred.
void numba::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    numba::ntensor::NTensorType resultType, mlir::Value source,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues,
        attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the type
// passed is nullptr, it is inferred.
void numba::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    numba::ntensor::NTensorType resultType, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues =
      llvm::to_vector<4>(llvm::map_range(
          offsets, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(
          sizes, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues =
      llvm::to_vector<4>(llvm::map_range(
          strides, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues);
}

// Build a SubViewOp with dynamic entries and inferred result type.
void numba::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, numba::ntensor::NTensorType(), source, offsets, sizes,
        strides, attrs);
}

void numba::ntensor::ResolveIndexOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<ResolveIndexPropagate>(context);
}

// Copypasted from upstream tensor.
llvm::SmallBitVector numba::ntensor::SubviewOp::getDroppedDims() {
  mlir::ArrayRef<int64_t> resultShape = getType().getShape();
  mlir::SmallVector<mlir::OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  if (resultShape.size() == mixedSizes.size())
    return droppedDims;

  unsigned shapePos = 0;
  for (const auto &size : enumerate(mixedSizes)) {
    std::optional<int64_t> sizeVal = getConstantIntValue(size.value());
    // If the size is not 1, or if the current matched dimension of the result
    // is the same static shape as the size value (which is 1), then the
    // dimension is preserved.
    if (!sizeVal || *sizeVal != 1 ||
        (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
      shapePos++;
      continue;
    }
    droppedDims.set(size.index());
  }
  return droppedDims;
}

static bool isIdentitySubview(numba::ntensor::SubviewOp op) {
  auto srcType = op.getSource().getType().cast<numba::ntensor::NTensorType>();
  if (srcType != op.getResult().getType())
    return false;

  for (auto val : op.getMixedOffsets())
    if (!mlir::isConstantIntValue(val, 0))
      return false;

  auto srcShape = srcType.getShape();
  for (auto &&[i, val] : llvm::enumerate(op.getMixedSizes())) {
    assert(i < srcShape.size());
    auto shapeVal = srcShape[i];
    if (mlir::ShapedType::isDynamic(shapeVal)) {
      auto dim = val.dyn_cast<mlir::Value>();
      if (!dim)
        return false;

      auto dimOp = dim.getDefiningOp<numba::ntensor::DimOp>();
      if (!dimOp)
        return false;

      auto dimInd = dimOp.getConstantIndex();
      if (!dimInd || *dimInd != static_cast<int64_t>(i))
        return false;
    } else {
      if (!mlir::isConstantIntValue(val, shapeVal))
        return false;
    }
  }

  for (auto val : op.getMixedStrides())
    if (!mlir::isConstantIntValue(val, 1))
      return false;

  return true;
}

mlir::OpFoldResult numba::ntensor::SubviewOp::fold(FoldAdaptor) {
  if (isIdentitySubview(*this))
    return getSource();

  return nullptr;
}

mlir::OpFoldResult numba::ntensor::FromTensorOp::fold(FoldAdaptor) {
  if (auto to = getTensor().getDefiningOp<numba::ntensor::ToTensorOp>()) {
    auto array = to.getArray();
    if (getType() == array.getType())
      return array;
  }
  return nullptr;
}

namespace {
struct FromTensorChain
    : public mlir::OpRewritePattern<numba::ntensor::FromTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::FromTensorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto srcOp = op.getTensor().getDefiningOp<numba::ntensor::ToTensorOp>();
    if (!srcOp)
      return mlir::failure();

    auto src = srcOp.getArray();
    auto srcType = mlir::cast<numba::ntensor::NTensorType>(src.getType());
    auto dstType = mlir::cast<numba::ntensor::NTensorType>(op.getType());
    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    if (!numba::ntensor::CastOp::areCastCompatible(srcType, dstType))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::ntensor::CastOp>(op, dstType, src);
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::FromTensorOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<FromTensorChain>(context);
}

mlir::OpFoldResult numba::ntensor::ToTensorOp::fold(FoldAdaptor) {
  auto arr = getArray();
  if (auto cast = arr.getDefiningOp<numba::ntensor::CastOp>()) {
    auto src = cast.getSource();
    auto srcType = mlir::cast<numba::ntensor::NTensorType>(src.getType());
    auto dstType = mlir::cast<numba::ntensor::NTensorType>(arr.getType());
    if (srcType.getShape() != dstType.getShape() ||
        srcType.getElementType() != dstType.getElementType() ||
        srcType.getEnvironment() != dstType.getEnvironment())
      return nullptr;

    this->setOperand(src);
    return this->getResult();
  }
  if (auto from = arr.getDefiningOp<numba::ntensor::FromTensorOp>()) {
    auto val = from.getTensor();
    if (getType() == val.getType())
      return val;
  }
  return nullptr;
}

mlir::OpFoldResult numba::ntensor::ToTensorCopyOp::fold(FoldAdaptor) {
  auto arr = getArray();
  if (auto from = arr.getDefiningOp<numba::ntensor::FromTensorOp>()) {
    if (!arr.hasOneUse())
      return nullptr;

    auto val = from.getTensor();
    if (getType() == val.getType())
      return val;
  }
  return nullptr;
}

mlir::OpFoldResult numba::ntensor::FromMemrefOp::fold(FoldAdaptor) {
  if (auto to = getMemref().getDefiningOp<numba::ntensor::ToMemrefOp>()) {
    auto array = to.getArray();
    if (getType() == array.getType())
      return array;
  }
  return nullptr;
}

mlir::OpFoldResult numba::ntensor::ToMemrefOp::fold(FoldAdaptor) {
  if (auto from = getArray().getDefiningOp<numba::ntensor::FromMemrefOp>()) {
    auto val = from.getMemref();
    if (getType() == val.getType())
      return val;
  }
  return nullptr;
}

// Copypasted from upstream tensor.
mlir::LogicalResult numba::ntensor::SubviewOp::reifyResultShapes(
    mlir::OpBuilder &builder,
    mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0].reserve(getType().getRank());
  mlir::SmallVector<mlir::OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims = getDroppedDims();
  mlir::Location loc = getLoc();
  for (const auto &size : enumerate(mixedSizes)) {
    if (droppedDims.test(size.index()))
      continue;
    if (auto attr = size.value().dyn_cast<mlir::Attribute>()) {
      reifiedReturnShapes[0].push_back(
          builder
              .create<mlir::arith::ConstantIndexOp>(
                  loc, attr.cast<mlir::IntegerAttr>().getInt())
              .getResult());
      continue;
    }
    reifiedReturnShapes[0].push_back(size.value().get<mlir::Value>());
  }
  return mlir::success();
}

mlir::OpFoldResult numba::ntensor::CastOp::fold(FoldAdaptor) {
  mlir::Value current = getSource();
  while (auto parent = current.getDefiningOp<CastOp>()) {
    auto parentSource = parent.getSource();
    if (parentSource.getType() == getType())
      return parentSource;

    current = parentSource;
  }
  return nullptr;
}

namespace {
struct NTensorChainCast
    : public mlir::OpRewritePattern<numba::ntensor::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prev = op.getSource().getDefiningOp<numba::ntensor::CastOp>();
    if (!prev)
      return mlir::failure();

    auto src = prev.getSource();
    auto srcType = mlir::cast<numba::ntensor::NTensorType>(src.getType());
    auto dstType = mlir::cast<numba::ntensor::NTensorType>(op.getType());
    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }
    if (!numba::ntensor::CastOp::areCastCompatible(srcType, dstType))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::ntensor::CastOp>(op, dstType, src);
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::CastOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<NTensorChainCast>(context);
}

bool numba::ntensor::CastOp::areCastCompatible(mlir::TypeRange inputs,
                                               mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  mlir::Type a = inputs.front(), b = outputs.front();
  auto aT = a.dyn_cast<numba::ntensor::NTensorType>();
  auto bT = b.dyn_cast<numba::ntensor::NTensorType>();
  if (!aT || !bT)
    return false;

  if (aT.getElementType() != bT.getElementType())
    return false;

  return succeeded(mlir::verifyCompatibleShape(aT, bT));
}

void numba::ntensor::ElementwiseOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs,
    ::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location,
                              ::mlir::ValueRange)>
        bodyBuilder) {
  build(odsBuilder, odsState, resultTypes, inputs);
  if (bodyBuilder) {
    mlir::Region *bodyRegion = odsState.regions.back().get();
    bodyRegion->push_back(new mlir::Block);
    mlir::Block &bodyBlock = bodyRegion->front();

    for (auto input : inputs) {
      auto srcType = input.getType().cast<numba::ntensor::NTensorType>();
      bodyBlock.addArgument(srcType.getElementType(), odsState.location);
    }

    mlir::OpBuilder::InsertionGuard guard(odsBuilder);
    odsBuilder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(odsBuilder, odsState.location, bodyBlock.getArguments());
  }
}

void numba::ntensor::BroadcastOp::build(::mlir::OpBuilder &odsBuilder,
                                        ::mlir::OperationState &odsState,
                                        ::mlir::ValueRange inputs) {
  if (inputs.empty())
    return build(odsBuilder, odsState, /*types*/ {}, /*inputs*/ {});

  if (inputs.size() == 1)
    return build(odsBuilder, odsState, inputs.front().getType(), inputs);

  auto isDynamicOrUnit = [](int64_t v) {
    return mlir::ShapedType::isDynamic(v) || v == 1;
  };

  auto areDimsBroadcastable = [&](int64_t a, int64_t b) {
    return isDynamicOrUnit(a) || isDynamicOrUnit(b) || a == b;
  };

  auto broadcastDim = [&](int64_t a, int64_t b) {
    if (!areDimsBroadcastable(a, b))
      return mlir::ShapedType::kDynamic; // Will be caught later

    if (mlir::ShapedType::isDynamic(a) || mlir::ShapedType::isDynamic(b))
      return mlir::ShapedType::kDynamic;

    return a == 1 ? b : a;
  };

  llvm::SmallVector<int64_t> newShape;
  for (auto &&arg : inputs) {
    auto type = mlir::cast<mlir::ShapedType>(arg.getType());
    auto shape = type.getShape();
    if (shape.size() > newShape.size()) {
      size_t diff = shape.size() - newShape.size();
      newShape.insert(newShape.begin(), diff, mlir::ShapedType::kDynamic);
    }

    for (auto &&[i, dim] : llvm::enumerate(shape)) {
      auto diff =
          shape.size() < newShape.size() ? newShape.size() - shape.size() : 0;
      auto oldDim = newShape[diff + i];
      auto newDim = broadcastDim(dim, oldDim);
      newShape[diff + i] = newDim;
    }
  }

  mlir::SmallVector<mlir::Type> resultTypes(inputs.size());
  for (auto &&[i, arg] : llvm::enumerate(inputs)) {
    auto type = mlir::cast<mlir::ShapedType>(arg.getType());
    resultTypes[i] = type.clone(newShape);
  }
  build(odsBuilder, odsState, resultTypes, inputs);
}

mlir::LogicalResult numba::ntensor::BroadcastOp::fold(
    FoldAdaptor, llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
  mlir::ValueRange inputs = getInputs();
  mlir::TypeRange resultTypes = getResultTypes();
  assert(inputs.size() == resultTypes.size());
  if (!inputs.empty()) {
    auto getShape = [](mlir::Type type) {
      return mlir::cast<mlir::ShapedType>(type).getShape();
    };

    mlir::Value first = inputs.front();
    auto firstShape = getShape(first.getType());
    if (firstShape != getShape(resultTypes.front()))
      return mlir::failure();

    results.emplace_back(first);
    for (auto &&[arg, res] :
         llvm::zip(inputs.drop_front(), resultTypes.drop_front())) {
      if (arg != first || firstShape != getShape(res)) {
        results.clear();
        return mlir::failure();
      }
      results.emplace_back(arg);
    }
    return mlir::success();
  }

  return mlir::failure();
}

namespace {
struct BroadcastSameStaticShape
    : public mlir::OpRewritePattern<numba::ntensor::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::BroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    if (inputs.size() < 2)
      return mlir::failure();

    auto firstType = inputs.front().getType().cast<mlir::ShapedType>();
    if (!firstType.hasStaticShape())
      return mlir::failure();

    auto shape = firstType.getShape();
    for (auto arg : inputs.drop_front())
      if (arg.getType().cast<mlir::ShapedType>().getShape() != shape)
        return mlir::failure();

    auto loc = op.getLoc();
    llvm::SmallVector<mlir::Value> newResults(inputs.size());
    for (auto &&[i, it] :
         llvm::enumerate(llvm::zip(op.getInputs(), op.getResults()))) {
      auto &&[src, res] = it;
      auto srcType = src.getType();
      auto dstType = res.getType();

      if (srcType != dstType) {
        newResults[i] =
            rewriter.create<numba::ntensor::CastOp>(loc, dstType, src);
      } else {
        newResults[i] = src;
      }
    }

    rewriter.replaceOp(op, newResults);
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::BroadcastOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<BroadcastSameStaticShape>(context);
}

mlir::Value numba::ntensor::CreateArrayOp::getDynamicSize(unsigned idx) {
  assert(getType().isDynamicDim(idx) && "expected dynamic dim");
  unsigned ctr = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(idx); ++i)
    if (getType().isDynamicDim(i))
      ++ctr;
  return getDynamicSizes()[ctr];
}

llvm::SmallVector<mlir::OpFoldResult>
numba::ntensor::CreateArrayOp::getMixedSizes() {
  llvm::SmallVector<mlir::OpFoldResult> result;
  unsigned ctr = 0;
  mlir::OpBuilder b(getContext());
  for (int64_t i = 0; i < getType().getRank(); ++i) {
    if (getType().isDynamicDim(i)) {
      result.push_back(getDynamicSizes()[ctr++]);
    } else {
      result.push_back(b.getIndexAttr(getType().getShape()[i]));
    }
  }
  return result;
}

namespace {
struct FoldCreateCast
    : public mlir::OpRewritePattern<numba::ntensor::CreateArrayOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CreateArrayOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::hasSingleElement(op->getUses()))
      return mlir::failure();

    auto cast = mlir::dyn_cast<mlir::CastOpInterface>(*op->getUsers().begin());
    if (!cast || cast->getNumResults() != 1)
      return mlir::failure();

    auto thisType = op.getType();
    auto resType = mlir::dyn_cast<numba::ntensor::NTensorType>(
        cast->getResult(0).getType());
    if (!resType || resType.getShape() != thisType.getShape() ||
        resType.getElementType() != thisType.getElementType())
      return mlir::failure();

    rewriter.modifyOpInPlace(op, [&]() { op.getResult().setType(resType); });
    rewriter.replaceOp(cast, op);
    return mlir::success();
  }
};
} // namespace

void numba::ntensor::CreateArrayOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<FoldCreateCast>(context);
}

static mlir::LogicalResult parseShape(mlir::AsmParser &parser,
                                      llvm::SmallVector<int64_t> &shape,
                                      mlir::Type &type) {
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();

  shape = std::move(dimensions);
  type = std::move(t);
  return mlir::success();
}

static void printShape(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
                       mlir::Type type) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;
}

static bool parseArgList(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &argsOperands,
    mlir::ArrayAttr &args_namesAttr) {
  if (parser.parseLParen())
    return true;

  auto *context = parser.getContext();
  llvm::SmallVector<mlir::Attribute> names;
  if (parser.parseOptionalRParen()) {
    std::string name;
    while (true) {
      name.clear();
      if (!parser.parseOptionalKeywordOrString(&name)) {
        if (parser.parseColon())
          return true;
      }
      names.push_back(mlir::StringAttr::get(context, name));

      argsOperands.push_back({});
      if (parser.parseOperand(argsOperands.back()))
        return true;

      if (!parser.parseOptionalRParen())
        break;

      if (parser.parseComma())
        return true;
    }
  }

  assert(names.size() == argsOperands.size());
  args_namesAttr = mlir::ArrayAttr::get(context, names);
  return false;
}

static void printArgList(mlir::OpAsmPrinter &printer,
                         numba::ntensor::CallOp call, mlir::ValueRange args,
                         mlir::ArrayAttr argsNames) {
  assert(args.size() == argsNames.size());
  printer << '(';
  bool first = true;
  for (auto &&[arg, name] : llvm::zip(args, argsNames)) {
    if (first) {
      first = false;
    } else {
      printer << ", ";
    }
    auto nameStr =
        (name ? name.cast<mlir::StringAttr>().getValue() : llvm::StringRef());
    if (!nameStr.empty())
      printer << nameStr << ':';
    printer.printOperand(arg);
  }
  printer << ')';
}

#include "numba/Dialect/ntensor/IR/NTensorOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "numba/Dialect/ntensor/IR/NTensorOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "numba/Dialect/ntensor/IR/NTensorOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "numba/Dialect/ntensor/IR/NTensorOpsTypes.cpp.inc"

#include "numba/Dialect/ntensor/IR/NTensorOpsEnums.cpp.inc"
