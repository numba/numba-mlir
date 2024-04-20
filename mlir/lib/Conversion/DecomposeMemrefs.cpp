// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/GpuToGpuRuntime.hpp"

#include "numba/Dialect/numba_util/Dialect.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

static void setInsertionPointToStart(OpBuilder &builder, Value val) {
  if (auto parentOp = val.getDefiningOp()) {
    builder.setInsertionPointAfter(parentOp);
  } else {
    builder.setInsertionPointToStart(val.getParentBlock());
  }
}

static bool isInsideLaunch(Operation *op) {
  return op->getParentOfType<gpu::LaunchOp>();
}

static std::tuple<Value, OpFoldResult, SmallVector<OpFoldResult>>
getFlatOffsetAndStrides(OpBuilder &rewriter, Location loc, Value source,
                        ArrayRef<OpFoldResult> subOffsets,
                        ArrayRef<OpFoldResult> subStrides = std::nullopt) {
  auto sourceType = cast<MemRefType>(source.getType());
  auto sourceRank = static_cast<unsigned>(sourceType.getRank());

  memref::ExtractStridedMetadataOp newExtractStridedMetadata;
  {
    OpBuilder::InsertionGuard g(rewriter);
    setInsertionPointToStart(rewriter, source);
    newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, source);
  }

  auto &&[sourceStrides, sourceOffset] = getStridesAndOffset(sourceType);

  auto getDim = [&](int64_t dim, Value dimVal) -> OpFoldResult {
    return ShapedType::isDynamic(dim) ? getAsOpFoldResult(dimVal)
                                      : rewriter.getIndexAttr(dim);
  };

  OpFoldResult origOffset =
      getDim(sourceOffset, newExtractStridedMetadata.getOffset());
  ValueRange sourceStridesVals = newExtractStridedMetadata.getStrides();

  SmallVector<OpFoldResult> origStrides;
  origStrides.reserve(sourceRank);

  SmallVector<OpFoldResult> strides;
  strides.reserve(sourceRank);

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
  for (auto i : llvm::seq(0u, sourceRank)) {
    OpFoldResult origStride = getDim(sourceStrides[i], sourceStridesVals[i]);

    if (!subStrides.empty()) {
      strides.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 * s1, {subStrides[i], origStride}));
    }

    origStrides.emplace_back(origStride);
  }

  auto &&[expr, values] =
      computeLinearIndex(origOffset, origStrides, subOffsets);
  OpFoldResult finalOffset =
      affine::makeComposedFoldedAffineApply(rewriter, loc, expr, values);
  return {newExtractStridedMetadata.getBaseBuffer(), finalOffset, strides};
}

static Value getFlatMemref(OpBuilder &rewriter, Location loc, Value source,
                           ValueRange offsets) {
  SmallVector<OpFoldResult> offsetsTemp = getAsOpFoldResult(offsets);
  auto &&[base, offset, ignore] =
      getFlatOffsetAndStrides(rewriter, loc, source, offsetsTemp);
  auto srcType = cast<MemRefType>(base.getType());
  auto layout = StridedLayoutAttr::get(rewriter.getContext(),
                                       ShapedType::kDynamic, std::nullopt);
  auto retType = MemRefType::get(srcType.getShape(), srcType.getElementType(),
                                 layout, srcType.getMemorySpace());
  mlir::Value ret = rewriter.create<memref::ReinterpretCastOp>(
      loc, retType, base, offset, std::nullopt, std::nullopt);
  if (srcType != retType)
    ret = rewriter.create<numba::util::MemrefApplyOffsetOp>(loc, srcType, ret);

  return ret;
}

static bool needFlatten(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getRank() != 0;
}

static bool checkLayout(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getLayout().isIdentity() ||
         isa<StridedLayoutAttr>(type.getLayout());
}

namespace {
struct FlattenLoad : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLaunch(op))
      return rewriter.notifyMatchFailure(op, "not inside gpu.launch");

    Value memref = op.getMemref();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    Location loc = op.getLoc();
    Value flatMemref = getFlatMemref(rewriter, loc, memref, op.getIndices());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, flatMemref);
    return success();
  }
};

struct FlattenStore : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLaunch(op))
      return rewriter.notifyMatchFailure(op, "not inside gpu.launch");

    Value memref = op.getMemref();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    Location loc = op.getLoc();
    Value flatMemref = getFlatMemref(rewriter, loc, memref, op.getIndices());
    Value value = op.getValue();
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, value, flatMemref);
    return success();
  }
};

struct FlattenSubview : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLaunch(op))
      return rewriter.notifyMatchFailure(op, "not inside gpu.launch");

    Value memref = op.getSource();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    Location loc = op.getLoc();
    SmallVector<OpFoldResult> subOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> subSizes = op.getMixedSizes();
    SmallVector<OpFoldResult> subStrides = op.getMixedStrides();
    auto &&[base, finalOffset, strides] =
        getFlatOffsetAndStrides(rewriter, loc, memref, subOffsets, subStrides);

    auto srcType = cast<MemRefType>(memref.getType());
    auto resultType = cast<MemRefType>(op.getType());
    unsigned subRank = static_cast<unsigned>(resultType.getRank());

    llvm::SmallBitVector droppedDims = op.getDroppedDims();

    SmallVector<OpFoldResult> finalSizes;
    finalSizes.reserve(subRank);

    SmallVector<OpFoldResult> finalStrides;
    finalStrides.reserve(subRank);

    for (auto i : llvm::seq(0u, static_cast<unsigned>(srcType.getRank()))) {
      if (droppedDims.test(i))
        continue;

      finalSizes.push_back(subSizes[i]);
      finalStrides.push_back(strides[i]);
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resultType, base, finalOffset, finalSizes, finalStrides);
    return success();
  }
};

// TODO: upstream
class DecomposeAtomicRMWOp
    : public mlir::OpRewritePattern<mlir::memref::AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AtomicRMWOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (llvm::all_of(op.getIndices(),
                     [](auto v) { return mlir::isConstantIntValue(v, 0); }))
      return mlir::failure();

    auto memref = op.getMemref();
    auto memrefType = memref.getType();
    auto rank = memrefType.getShape().size();
    llvm::SmallVector<mlir::OpFoldResult> offsets(rank);
    llvm::copy(op.getIndices(), offsets.begin());

    llvm::SmallVector<mlir::OpFoldResult> sizes(rank, rewriter.getIndexAttr(1));

    auto loc = op.getLoc();
    mlir::Value view = rewriter.create<mlir::memref::SubViewOp>(
        loc, memref, offsets, sizes, sizes);

    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    llvm::SmallVector<mlir::Value> newIndices(rank, zero);
    rewriter.modifyOpInPlace(op, [&]() {
      op.getIndicesMutable().assign(newIndices);
      op.getMemrefMutable().assign(view);
    });
    return mlir::success();
  }
};

static void
populateGpuDecomposeMemrefsPatternsImpl(RewritePatternSet &patterns) {
  patterns
      .insert<FlattenLoad, FlattenStore, FlattenSubview, DecomposeAtomicRMWOp>(
          patterns.getContext());
}

struct GpuDecomposeMemrefsPass
    : public mlir::PassWrapper<GpuDecomposeMemrefsPass, mlir::OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuDecomposeMemrefsPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    ::populateGpuDecomposeMemrefsPatternsImpl(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> gpu_runtime::createGpuDecomposeMemrefsPass() {
  return std::make_unique<GpuDecomposeMemrefsPass>();
}
