// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/CommonOpts.hpp"

#include "numba/Transforms/IfRewrites.hpp"
#include "numba/Transforms/IndexTypePropagation.hpp"
#include "numba/Transforms/LoopRewrites.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace {
static bool isSameRank(mlir::Type type1, mlir::Type type2) {
  auto shaped1 = type1.dyn_cast<mlir::ShapedType>();
  if (!shaped1)
    return false;

  auto shaped2 = type2.dyn_cast<mlir::ShapedType>();
  if (!shaped2)
    return false;

  if (!shaped1.hasRank() || !shaped2.hasRank())
    return false;

  return shaped1.getRank() == shaped2.getRank();
}

static bool isMixedValuesEqual(llvm::ArrayRef<mlir::OpFoldResult> values,
                               int64_t expectedVal) {
  for (auto val : values) {
    auto intVal = mlir::getConstantIntValue(val);
    if (!intVal || *intVal != expectedVal)
      return false;
  }
  return true;
}

struct SubviewLoadPropagate
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getMemref().getDefiningOp<mlir::memref::SubViewOp>();
    if (!src)
      return mlir::failure();

    if (!isSameRank(src.getSource().getType(), src.getType()))
      return mlir::failure();

    if (!isMixedValuesEqual(src.getMixedOffsets(), 0) ||
        !isMixedValuesEqual(src.getMixedStrides(), 1))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, src.getSource(),
                                                      op.getIndices());
    return mlir::success();
  }
};

struct SubviewStorePropagate
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getMemref().getDefiningOp<mlir::memref::SubViewOp>();
    if (!src)
      return mlir::failure();

    if (!isSameRank(src.getSource().getType(), src.getType()))
      return mlir::failure();

    if (!isMixedValuesEqual(src.getMixedOffsets(), 0) ||
        !isMixedValuesEqual(src.getMixedStrides(), 1))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, op.getValue(), src.getSource(), op.getIndices());
    return mlir::success();
  }
};

struct PowSimplify : public mlir::OpRewritePattern<mlir::math::PowFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::math::PowFOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    mlir::FloatAttr constValue;
    if (!mlir::matchPattern(rhs, mlir::m_Constant(&constValue)))
      return mlir::failure();

    assert(constValue);
    auto val = constValue.getValueAsDouble();
    if (val == 1.0) {
      rewriter.replaceOp(op, lhs);
      return mlir::success();
    }
    if (val == 2.0) {
      rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(op, lhs, lhs);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct AndConflictSimplify
    : public mlir::OpRewritePattern<mlir::arith::AndIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AndIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<mlir::arith::CmpIOp>();
    if (!lhs)
      return mlir::failure();

    auto rhs = op.getRhs().getDefiningOp<mlir::arith::CmpIOp>();
    if (!rhs)
      return mlir::failure();

    if (lhs.getLhs() != rhs.getLhs() || lhs.getRhs() != rhs.getRhs())
      return mlir::failure();

    using Pred = mlir::arith::CmpIPredicate;
    std::array<Pred, mlir::arith::getMaxEnumValForCmpIPredicate() + 1>
        handlers{};
    handlers[static_cast<size_t>(Pred::eq)] = Pred::ne;
    handlers[static_cast<size_t>(Pred::ne)] = Pred::eq;
    handlers[static_cast<size_t>(Pred::slt)] = Pred::sge;
    handlers[static_cast<size_t>(Pred::sle)] = Pred::sgt;
    handlers[static_cast<size_t>(Pred::sgt)] = Pred::sle;
    handlers[static_cast<size_t>(Pred::sge)] = Pred::slt;
    handlers[static_cast<size_t>(Pred::ult)] = Pred::uge;
    handlers[static_cast<size_t>(Pred::ule)] = Pred::ugt;
    handlers[static_cast<size_t>(Pred::ugt)] = Pred::ule;
    handlers[static_cast<size_t>(Pred::uge)] = Pred::ult;
    if (handlers[static_cast<size_t>(lhs.getPredicate())] != rhs.getPredicate())
      return mlir::failure();

    auto val = rewriter.getIntegerAttr(op.getType(), 0);
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, val);
    return mlir::success();
  }
};

// TODO: upstream
struct XorOfCmpF : public mlir::OpRewritePattern<mlir::arith::XOrIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::XOrIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cmp = op.getLhs().getDefiningOp<mlir::arith::CmpFOp>();
    if (!cmp || !llvm::hasSingleElement(cmp->getUses()) ||
        !mlir::isConstantIntValue(op.getRhs(), -1))
      return mlir::failure();

    // TODO: properly handle NaNs
    using Pred = mlir::arith::CmpFPredicate;
    const std::pair<Pred, Pred> mapping[] = {
        // clang-format off
      {Pred::OEQ, Pred::ONE},
      {Pred::ONE, Pred::OEQ},
      {Pred::OGE, Pred::OLT},
      {Pred::OGT, Pred::OLE},
      {Pred::OLE, Pred::OGT},
      {Pred::OLT, Pred::OGE},
        // clang-format on
    };

    auto pred = cmp.getPredicate();
    for (auto &&[oldPred, newPred] : mapping) {
      if (pred != oldPred)
        continue;

      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
          op, newPred, cmp.getLhs(), cmp.getRhs());
      return mlir::success();
    }

    return mlir::failure();
  }
};

// TODO: upstream
struct ExtractStridedMetadataUnused
    : public mlir::OpRewritePattern<mlir::memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExtractStridedMetadataOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto buffer = op.getBaseBuffer();
    if (buffer.use_empty())
      return mlir::failure();

    if (!op.getOffset().use_empty() ||
        llvm::any_of(op.getStrides(), [](auto s) { return !s.use_empty(); }))
      return mlir::failure();

    auto loc = op.getLoc();
    auto src = op.getSource();
    auto dstType = buffer.getType().cast<mlir::MemRefType>();
    mlir::Value offset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value newRes = rewriter.create<mlir::memref::ReinterpretCastOp>(
        loc, dstType, src, offset, std::nullopt, std::nullopt);
    rewriter.replaceAllUsesWith(buffer, newRes);
    for (auto &&[i, size] : llvm::enumerate(op.getSizes())) {
      if (size.use_empty())
        continue;

      mlir::Value newSize = rewriter.create<mlir::memref::DimOp>(
          loc, src, static_cast<int64_t>(i));
      rewriter.replaceAllUsesWith(size, newSize);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// TODO: upstream
struct ExtractStridedMetadataConstStrides
    : public mlir::OpRewritePattern<mlir::memref::ExtractStridedMetadataOp> {
  // Set benefit higher than ExtractStridedMetadataCast
  ExtractStridedMetadataConstStrides(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::memref::ExtractStridedMetadataOp>(
            context, /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExtractStridedMetadataOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    mlir::MemRefType srcType = src.getType();

    int64_t offset;
    llvm::SmallVector<int64_t> strides;
    if (mlir::failed(mlir::getStridesAndOffset(srcType, strides, offset)))
      return mlir::failure();

    bool changed = false;
    auto loc = op.getLoc();
    auto replaceUses = [&](mlir::Value res, int64_t val) {
      if (mlir::ShapedType::isDynamic(val) || res.use_empty())
        return;

      changed = true;
      mlir::Value constVal =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, val);
      rewriter.replaceAllUsesWith(res, constVal);
    };

    auto origStrides = op.getStrides();
    replaceUses(op.getOffset(), offset);
    for (auto &&[strideRes, strideVal] : llvm::zip(origStrides, strides))
      replaceUses(strideRes, strideVal);

    bool isIdentity = srcType.getLayout().isIdentity();
    if (isIdentity &&
        llvm::any_of(origStrides, [](auto s) { return !s.use_empty(); })) {
      auto rank = srcType.getRank();
      mlir::Value stride =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      rewriter.replaceAllUsesWith(origStrides[rank - 1], stride);
      for (auto i : llvm::seq<int64_t>(0, rank - 1)) {
        mlir::Value size =
            rewriter.create<mlir::memref::DimOp>(loc, src, rank - i - 1);
        if (i == 0) {
          stride = size;
        } else {
          stride = rewriter.create<mlir::arith::MulIOp>(loc, stride, size);
        }
        rewriter.replaceAllUsesWith(origStrides[rank - i - 2], stride);
      }
      changed = true;
    }

    return mlir::success(changed);
  }
};

// TODO: upstream
struct ExtractStridedMetadataCast
    : public mlir::OpRewritePattern<mlir::memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExtractStridedMetadataOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cast = op.getSource().getDefiningOp<mlir::memref::CastOp>();
    if (!cast)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::ExtractStridedMetadataOp>(
        op, cast.getSource());
    return mlir::success();
  }
};

// TODO: upstream
struct IndexCastOfIndexCast
    : public mlir::OpRewritePattern<mlir::arith::IndexCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::IndexCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mid = op.getIn();
    if (!mlir::isa<mlir::IndexType>(mid.getType()))
      return mlir::failure();

    auto prevCast = mid.getDefiningOp<mlir::arith::IndexCastOp>();
    if (!prevCast)
      return mlir::failure();

    auto src = prevCast.getIn();

    auto srcType = mlir::dyn_cast<mlir::IntegerType>(src.getType());
    if (!srcType)
      return mlir::failure();

    auto dstType = mlir::dyn_cast<mlir::IntegerType>(op.getResult().getType());
    if (!dstType)
      return mlir::failure();

    if (srcType.getWidth() < dstType.getWidth()) {
      rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(op, dstType, src);
    } else if (srcType.getWidth() > dstType.getWidth()) {
      rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(op, dstType, src);
    } else {
      rewriter.replaceOp(op, src);
    }
    return mlir::success();
  }
};

struct GPUGenGlobalId : public mlir::OpRewritePattern<mlir::arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AddIOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto getArg = [](auto op, bool rev) -> mlir::Value {
      return rev ? op.getLhs() : op.getRhs();
    };

    mlir::gpu::Dimension dim;
    mlir::arith::MulIOp other;
    for (auto rev : {false, true}) {
      auto arg1 = getArg(op, rev);
      auto arg2 = getArg(op, !rev);
      if (auto tid = arg1.getDefiningOp<mlir::gpu::ThreadIdOp>()) {
        dim = tid.getDimension();
        other = arg2.getDefiningOp<mlir::arith::MulIOp>();
        break;
      }
    }

    if (!other)
      return mlir::failure();

    for (auto rev : {false, true}) {
      auto arg1 = getArg(other, rev).getDefiningOp<mlir::gpu::BlockIdOp>();
      auto arg2 = getArg(other, !rev).getDefiningOp<mlir::gpu::BlockDimOp>();
      if (arg1 && arg2) {
        if (arg1.getDimension() != dim || arg2.getDimension() != dim)
          return mlir::failure();

        rewriter.replaceOpWithNewOp<mlir::gpu::GlobalIdOp>(op, dim);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

struct CommonOptsPass
    : public mlir::PassWrapper<CommonOptsPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommonOptsPass)

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    numba::populateCommonOptsPatterns(patterns);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void numba::populateCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (auto op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, context);
}

void numba::populateCommonOptsPatterns(mlir::RewritePatternSet &patterns) {
  populateCanonicalizationPatterns(patterns);

  patterns.insert<
      // clang-format off
      SubviewLoadPropagate,
      SubviewStorePropagate,
      PowSimplify,
      AndConflictSimplify,
      XorOfCmpF,
      ExtractStridedMetadataUnused,
      ExtractStridedMetadataConstStrides,
      ExtractStridedMetadataCast,
      IndexCastOfIndexCast,
      GPUGenGlobalId
      // clang-format on
      >(patterns.getContext());

  numba::populateIfRewritesPatterns(patterns);
  numba::populateLoopRewritesPatterns(patterns);
  numba::populateIndexPropagatePatterns(patterns);
}

std::unique_ptr<mlir::Pass> numba::createCommonOptsPass() {
  return std::make_unique<CommonOptsPass>();
}
