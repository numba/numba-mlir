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
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
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

static std::optional<std::pair<bool, int64_t>>
getCmpArgArithConst(mlir::arith::CmpIOp op) {
  for (auto reverse : {false, true}) {
    auto getOptionalPair = [&](const std::optional<int64_t> &val2)
        -> std::optional<std::pair<bool, int64_t>> {
      if (!val2)
        return std::nullopt;

      return std::pair(reverse, *val2);
    };
    auto getOptionalPairInv = [&](const std::optional<int64_t> &val2)
        -> std::optional<std::pair<bool, int64_t>> {
      if (!val2)
        return std::nullopt;

      return std::pair(reverse, -*val2);
    };

    auto val1 = (reverse ? op.getRhs() : op.getLhs());
    auto val2 = (reverse ? op.getLhs() : op.getRhs());
    if (auto add = val1.getDefiningOp<mlir::arith::AddIOp>()) {
      if (val2 == add.getLhs())
        return getOptionalPair(mlir::getConstantIntValue(add.getRhs()));

      if (val2 == add.getRhs())
        return getOptionalPair(mlir::getConstantIntValue(add.getRhs()));

      return std::nullopt;
    }
    if (auto sub = val1.getDefiningOp<mlir::arith::SubIOp>()) {
      if (val2 == sub.getLhs())
        return getOptionalPairInv(mlir::getConstantIntValue(sub.getRhs()));

      return std::nullopt;
    }
  }
  return std::nullopt;
}

static mlir::arith::CmpIPredicate
reversePredicate(mlir::arith::CmpIPredicate pred) {
  using namespace mlir;
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return arith::CmpIPredicate::eq;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::ne;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

// TODO: upstream
struct CmpOfArithConst : public mlir::OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto v = getCmpArgArithConst(op);
    if (!v)
      return mlir::failure();

    auto &&[inv, val] = *v;

    // TODO: ignore overflow for now
    using Pred = mlir::arith::CmpIPredicate;
    auto pred = op.getPredicate();
    if (inv)
      pred = reversePredicate(pred);

    auto replaceWithConst = [&](bool flag) {
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(op, int64_t(flag),
                                                              1);
    };

    if (pred == Pred::sge && val < 0) {
      replaceWithConst(false);
      return mlir::success();
    }

    return mlir::failure();
  }
};

// TODO: upstream
struct CanonConstSubI : public mlir::OpRewritePattern<mlir::arith::SubIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SubIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto val = mlir::getConstantIntValue(op.getRhs());
    if (!val)
      return mlir::failure();

    mlir::Value newVal;
    auto loc = op.getLoc();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(op.getType())) {
      newVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, -*val,
                                                           intType.getWidth());
    } else {
      assert(mlir::isa<mlir::IndexType>(op.getType()));
      newVal = rewriter.create<mlir::arith::ConstantIndexOp>(loc, -*val);
    }
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, op.getLhs(), newVal);
    return mlir::success();
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

// TODO: Upstream
struct TruncfOfExtf : public mlir::OpRewritePattern<mlir::arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::TruncFOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ext = op.getIn().getDefiningOp<mlir::arith::ExtFOp>();
    if (!ext)
      return mlir::failure();

    auto in = ext.getIn();
    if (in.getType() != op.getType())
      return mlir::failure();

    rewriter.replaceOp(op, in);
    return mlir::success();
  }
};

// TODO: Upstream
template <typename Op>
struct ResTruncFUnary : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto res = op.getResult();
    if (!llvm::hasSingleElement(res.getUsers()))
      return mlir::failure();

    auto trunc = mlir::dyn_cast<mlir::arith::TruncFOp>(*res.getUsers().begin());
    if (!trunc)
      return mlir::failure();

    auto resType = trunc.getType();

    auto loc = op.getLoc();
    mlir::Value arg =
        rewriter.create<mlir::arith::TruncFOp>(loc, resType, op.getOperand());

    mlir::Value newRes = rewriter.create<Op>(loc, arg, op.getFastmath());
    rewriter.replaceOp(trunc, newRes);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// TODO: Upstream
template <typename Op>
struct ResTruncIUnary : public mlir::OpRewritePattern<mlir::arith::TruncIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::TruncIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prev = op.getIn().getDefiningOp<Op>();
    if (!prev)
      return mlir::failure();

    auto src = prev.getOperand();
    auto resType = op.getType();
    rewriter.replaceOpWithNewOp<Op>(op, resType, src);
    return mlir::success();
  }
};

// TODO: Upstream
template <typename Op>
struct ResTruncIBinary : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto res = op.getResult();
    if (!llvm::hasSingleElement(res.getUsers()))
      return mlir::failure();

    auto trunc = mlir::dyn_cast<mlir::arith::TruncIOp>(*res.getUsers().begin());
    if (!trunc)
      return mlir::failure();

    auto resType = trunc.getType();

    auto loc = op.getLoc();
    mlir::Value lhs =
        rewriter.create<mlir::arith::TruncIOp>(loc, resType, op.getLhs());
    mlir::Value rhs =
        rewriter.create<mlir::arith::TruncIOp>(loc, resType, op.getRhs());

    mlir::Value newRes = rewriter.create<Op>(loc, lhs, rhs);
    rewriter.replaceOp(trunc, newRes);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// TODO: Upstream
template <typename Op>
struct ResTruncFBinary : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto res = op.getResult();
    if (!llvm::hasSingleElement(res.getUsers()))
      return mlir::failure();

    auto trunc = mlir::dyn_cast<mlir::arith::TruncFOp>(*res.getUsers().begin());
    if (!trunc)
      return mlir::failure();

    auto resType = trunc.getType();

    auto loc = op.getLoc();
    mlir::Value lhs =
        rewriter.create<mlir::arith::TruncFOp>(loc, resType, op.getLhs());
    mlir::Value rhs =
        rewriter.create<mlir::arith::TruncFOp>(loc, resType, op.getRhs());

    mlir::Value newRes = rewriter.create<Op>(loc, lhs, rhs, op.getFastmath());
    rewriter.replaceOp(trunc, newRes);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static llvm::SmallVector<mlir::arith::CmpIOp> getCondCmps(mlir::Value cond) {
  llvm::SmallVector<mlir::Operation *> worklist;
  auto addToWorklist = [&](mlir::Value val) {
    if (auto op = val.getDefiningOp())
      worklist.emplace_back(op);
  };

  addToWorklist(cond);

  llvm::SmallVector<mlir::arith::CmpIOp> ret;
  while (!worklist.empty()) {
    auto op = worklist.pop_back_val();
    if (auto cmpOp = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
      ret.emplace_back(cmpOp);
    } else if (auto andOp = mlir::dyn_cast<mlir::arith::AndIOp>(op)) {
      addToWorklist(andOp.getLhs());
      addToWorklist(andOp.getRhs());
    }
  }

  return ret;
}

struct CmpOfIndexCast : public mlir::OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    for (auto reverse : {false, true}) {
      auto cast = (reverse ? op.getRhs() : op.getLhs())
                      .getDefiningOp<mlir::arith::IndexCastOp>();
      if (!cast)
        continue;

      mlir::Value other = (reverse ? op.getLhs() : op.getRhs());
      auto constVal = mlir::getConstantIntValue(other);
      if (!constVal)
        continue;

      mlir::Operation *current = op;
      while (auto parent = current->getParentOfType<mlir::scf::IfOp>()) {
        current = parent;

        auto candidates = getCondCmps(parent.getCondition());
        for (auto cond : candidates) {
          mlir::Value lhs = cast.getIn();
          if (lhs != cond.getLhs() && lhs != cond.getRhs())
            continue;

          auto newType = lhs.getType();
          auto newVal = rewriter.getIntegerAttr(newType, *constVal);
          auto loc = op.getLoc();
          mlir::Value rhs =
              rewriter.create<mlir::arith::ConstantOp>(loc, newType, newVal);
          if (reverse)
            std::swap(lhs, rhs);

          rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
              op, op.getPredicate(), lhs, rhs);
          return mlir::success();
        }
      }
    }
    return mlir::failure();
  }
};

struct CmpInvIf : public mlir::OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Operation *current = op;
    while (auto parent = current->getParentOfType<mlir::scf::IfOp>()) {
      current = parent;

      auto candidates = getCondCmps(parent.getCondition());
      for (auto cond : candidates) {
        if (cond.getLhs() != op.getLhs() || cond.getRhs() != op.getRhs())
          continue;

        auto pred = op.getPredicate();
        auto otherPred = cond.getPredicate();

        bool inverted;
        if (pred == otherPred) {
          inverted = false;
        } else if (pred == mlir::arith::invertPredicate(otherPred)) {
          inverted = true;
        } else {
          continue;
        }

        int64_t value = inverted != parent.getThenRegion().isAncestor(
                                        op->getParentRegion());
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(op, value, 1);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

struct MoveArithOutOfIf : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    for (auto body : {op.thenBlock(), op.elseBlock()}) {
      if (!body)
        continue;

      auto ops = body->without_terminator();
      auto onlyArithOps = [&]() -> bool {
        for (auto &bodyOp : ops)
          if (!mlir::isa<mlir::arith::ArithDialect>(bodyOp.getDialect()))
            return false;

        return true;
      }();
      if (!onlyArithOps)
        continue;

      for (auto &bodyOp : llvm::make_early_inc_range(ops)) {

        rewriter.modifyOpInPlace(&bodyOp, [&]() { bodyOp.moveBefore(op); });
        changed = true;
      }
    }

    return mlir::success(changed);
  }
};

struct CanonicalizeLoopMemrefIndex
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp loadOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(loadOp->getParentOp());
    if (!loop || loadOp->getBlock() != loop.getBeforeBody())
      return rewriter.notifyMatchFailure(loadOp, "Not inside the loop");

    auto memref = loadOp.getMemref();
    if (!mlir::isa_and_present<mlir::memref::AllocOp, mlir::memref::AllocaOp>(
            memref.getDefiningOp()))
      return rewriter.notifyMatchFailure(loadOp, "Not result of alloc");

    auto isAncestor = [&](mlir::Operation *op) -> bool {
      auto reg = op->getParentRegion();
      return loop.getBefore().isAncestor(reg) ||
             loop.getAfter().isAncestor(reg);
    };

    mlir::memref::StoreOp storeOp;
    for (auto user : memref.getUsers()) {
      if (user == loadOp)
        continue;

      if (mlir::isa<mlir::memref::DeallocOp>(user))
        continue;

      if (mlir::isa<mlir::memref::LoadOp>(user)) {
        if (isAncestor(user)) {
          return rewriter.notifyMatchFailure(
              loadOp, [&](mlir::Diagnostic &diag) {
                diag << "Unsupported nested load: " << *user;
              });
        } else {
          continue;
        }
      }

      if (auto op = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
        if (op->getBlock() == loop.getBeforeBody()) {
          if (storeOp) {
            return rewriter.notifyMatchFailure(
                loadOp, [&](mlir::Diagnostic &diag) {
                  diag << "Unsupported Multiple stores: " << *storeOp << " and "
                       << *op;
                });
          } else {
            storeOp = op;
            continue;
          }
        } else {
          if (isAncestor(user)) {
            return rewriter.notifyMatchFailure(
                loadOp, [&](mlir::Diagnostic &diag) {
                  diag << "Unsupported nested store: " << *user;
                });
          } else {
            continue;
          }
        }
      }

      return rewriter.notifyMatchFailure(loadOp, [&](mlir::Diagnostic &diag) {
        diag << "Unsupported user: " << *user;
      });
    }

    if (!storeOp || storeOp.getIndices() != loadOp.getIndices())
      return rewriter.notifyMatchFailure(loadOp, "invalid store op");

    mlir::DominanceInfo dom;
    if (!dom.properlyDominates(loadOp.getOperation(), storeOp.getOperation()))
      return rewriter.notifyMatchFailure(loadOp,
                                         "Store op doesn't dominate load");

    auto indices = storeOp.getIndices();
    for (auto idx : indices) {
      if (!dom.properlyDominates(idx, loop))
        return rewriter.notifyMatchFailure(loadOp, [&](mlir::Diagnostic &diag) {
          diag << "Index doesnt dominate the loop: " << idx;
        });
    }

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(loop);
    auto loc = loop.getLoc();
    mlir::Value init =
        rewriter.create<mlir::memref::LoadOp>(loc, memref, indices);

    auto newInits = llvm::to_vector(loop.getInits());
    newInits.emplace_back(init);

    auto newResults = llvm::to_vector(loop->getResultTypes());
    newResults.emplace_back(init.getType());
    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        loc, newResults, newInits, nullptr, nullptr);

    auto oldBefore = loop.getBeforeBody();
    auto oldAfter = loop.getAfterBody();
    auto newBefore = newLoop.getBeforeBody();
    auto newAfter = newLoop.getAfterBody();

    rewriter.inlineBlockBefore(oldBefore, newBefore, newBefore->begin(),
                               newBefore->getArguments().drop_back());
    rewriter.inlineBlockBefore(oldAfter, newAfter, newAfter->begin(),
                               newAfter->getArguments().drop_back());

    auto beforeTerm =
        mlir::cast<mlir::scf::ConditionOp>(newBefore->getTerminator());
    rewriter.setInsertionPoint(beforeTerm);
    auto newCondArgs = llvm::to_vector(beforeTerm.getArgs());
    newCondArgs.emplace_back(storeOp.getValueToStore());
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        beforeTerm, beforeTerm.getCondition(), newCondArgs);

    rewriter.eraseOp(storeOp);
    rewriter.replaceOp(loadOp, newBefore->getArguments().back());

    auto afterTerm = mlir::cast<mlir::scf::YieldOp>(newAfter->getTerminator());
    rewriter.setInsertionPoint(afterTerm);
    auto newYieldArgs = llvm::to_vector(afterTerm.getResults());
    newYieldArgs.emplace_back(newAfter->getArguments().back());
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(afterTerm, newYieldArgs);

    rewriter.setInsertionPointAfter(newLoop);
    rewriter.create<mlir::memref::StoreOp>(loc, newLoop.getResults().back(),
                                           memref, indices);
    rewriter.replaceOp(loop, newLoop.getResults().drop_back());
    return mlir::success();
  }
};

static bool canMoveOpToBefore(mlir::Operation *op) {
  if (op->getNumResults() != 1)
    return false;

  return mlir::isPure(op);
}

struct MoveOpsFromBefore : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldBefore = op.getBeforeBody();
    auto oldAfter = op.getAfterBody();
    auto oldTerm =
        mlir::cast<mlir::scf::ConditionOp>(oldBefore->getTerminator());

    mlir::Operation *opToMove = nullptr;
    size_t idx = 0;
    for (auto &&[i, args] : llvm::enumerate(llvm::zip(
             oldTerm.getArgs(), oldAfter->getArguments(), op.getResults()))) {
      auto &&[arg, afterArg, res] = args;
      if (afterArg.use_empty() && res.use_empty())
        continue;

      auto argOp = arg.getDefiningOp();
      if (argOp && canMoveOpToBefore(argOp)) {
        opToMove = argOp;
        idx = i;
        break;
      }
    }

    if (!opToMove)
      return rewriter.notifyMatchFailure(op, "No ops to move");

    mlir::OpBuilder::InsertionGuard g(rewriter);

    auto newResults = llvm::to_vector(op->getResultTypes());
    llvm::append_range(newResults, opToMove->getOperandTypes());

    auto newTermArgs = llvm::to_vector(oldTerm.getArgs());
    llvm::append_range(newTermArgs, opToMove->getOperands());

    rewriter.setInsertionPoint(oldTerm);
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        oldTerm, oldTerm.getCondition(), newTermArgs);

    rewriter.setInsertionPoint(op);
    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        op.getLoc(), newResults, op.getInits(), nullptr, nullptr);

    auto newBefore = newLoop.getBeforeBody();
    auto newAfter = newLoop.getAfterBody();

    auto numArgs = opToMove->getNumOperands();
    auto newAfterArgs = newAfter->getArguments();
    rewriter.inlineBlockBefore(oldBefore, newBefore, newBefore->begin(),
                               newBefore->getArguments());
    rewriter.inlineBlockBefore(oldAfter, newAfter, newAfter->begin(),
                               newAfterArgs.drop_back(numArgs));

    mlir::IRMapping mapping;
    mapping.map(opToMove->getOperands(), newAfterArgs.take_back(numArgs));

    rewriter.setInsertionPointToStart(newAfter);
    auto newOp = rewriter.clone(*opToMove, mapping);
    rewriter.replaceAllUsesWith(newAfterArgs[idx], newOp->getResult(0));

    mapping.map(opToMove->getOperands(),
                newLoop.getResults().take_back(numArgs));

    rewriter.setInsertionPointAfter(newLoop);
    newOp = rewriter.clone(*opToMove, mapping);
    rewriter.replaceAllUsesWith(op.getResult(idx), newOp->getResult(0));

    rewriter.replaceOp(op, newLoop.getResults().drop_back(numArgs));
    return mlir::success();
  }
};

struct WhileOpLICM : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;

    mlir::DominanceInfo dom;
    for (mlir::Block *body : {loop.getBeforeBody(), loop.getAfterBody()}) {
      for (mlir::Operation &op :
           llvm::make_early_inc_range(body->without_terminator())) {
        if (!mlir::isPure(&op))
          continue;

        if (llvm::any_of(op.getOperands(), [&](auto &&arg) {
              return !dom.properlyDominates(arg, loop);
            }))
          continue;

        rewriter.modifyOpInPlace(&op, [&]() { op.moveBefore(loop); });
        changed = true;
      }
    }
    return mlir::success(changed);
  }
};

// struct WhileOpExpandTuple : public mlir::OpRewritePattern<mlir::scf::WhileOp>
// {
//   using OpRewritePattern::OpRewritePattern;

//  mlir::LogicalResult
//  matchAndRewrite(mlir::scf::WhileOp loop,
//                  mlir::PatternRewriter &rewriter) const override {
//    mlir::Block *beforeBody = loop.getBeforeBody();
//    auto beforeTerm =
//        mlir::cast<mlir::scf::ConditionOp>(beforeBody->getTerminator());

//    mlir::Block *afterBody = loop.getAfterBody();

//    size_t argIndex = 0;
//    mlir::TypedValue<mlir::TupleType> arg;
//    for (auto &&[i, it] :
//         llvm::enumerate(llvm::zip(loop.getResults(), beforeTerm.getArgs(),
//                                   afterBody->getArguments()))) {
//      auto &&[loopRes, beforeArg, afterArg] = it;
//      if (loopRes.use_empty() && afterArg.use_empty())
//        continue;

//      if (!mlir::isa<mlir::TupleType>(beforeArg.getType()))
//        continue;

//      argIndex = i;
//      arg = mlir::cast<mlir::TypedValue<mlir::TupleType>>(beforeArg);
//      break;
//    }

//    if (!arg)
//      return mlir::failure();

//    auto type = arg.getType();
//    auto count = type.size();
//    auto loc = beforeTerm.getLoc();
//    mlir::OpBuilder::InsertionGuard g(rewriter);
//    rewriter.setInsertionPoint(beforeTerm);

//    auto newCondArgs = llvm::to_vector(beforeTerm.getArgs());
//    for (auto i : llvm::seq<size_t>(0, count)) {
//      auto val = rewriter.create<numba::util::TupleExtractOp>(loc, arg, i);
//      newCondArgs.emplace_back(val);
//    }
//    rewriter.setInsertionPoint(beforeTerm);
//    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
//        beforeTerm, beforeTerm.getCondition(), newCondArgs);

//    mlir::ValueRange newArgsRange(newCondArgs);
//    auto newResTypes = newArgsRange.getTypes();

//    rewriter.setInsertionPoint(loop);
//    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
//        loc, newResTypes, loop.getInits(), nullptr, nullptr);

//    auto newBefore = newLoop.getBeforeBody();
//    auto newAfter = newLoop.getAfterBody();

//    rewriter.inlineBlockBefore(beforeBody, newBefore, newBefore->begin(),
//                               newBefore->getArguments());

//    mlir::ValueRange newAfterArgs = newAfter->getArguments();
//    rewriter.setInsertionPointToStart(newAfter);

//    auto newTuple = rewriter.create<numba::util::BuildTupleOp>(
//        loc, newAfterArgs.take_back(count));
//    auto mappedAfterArgs = llvm::to_vector(newAfterArgs.drop_back(count));
//    mappedAfterArgs[argIndex] = newTuple;

//    rewriter.inlineBlockBefore(afterBody, newAfter, newAfter->end(),
//                               mappedAfterArgs);

//    mlir::ValueRange newLoopResults = newLoop.getResults();
//    rewriter.setInsertionPointAfter(newLoop);
//    newTuple = rewriter.create<numba::util::BuildTupleOp>(
//        loc, newLoopResults.take_back(count));
//    auto mappedLoopResults = llvm::to_vector(newLoopResults.drop_back(count));
//    mappedLoopResults[argIndex] = newTuple;
//    rewriter.replaceOp(loop, mappedLoopResults);
//    return mlir::success();
//  }
//};

struct WhileOpMoveIfCond : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(op->getParentOp());
    if (!loop || op->getBlock() != loop.getBeforeBody())
      return mlir::failure();

    mlir::Block *beforeBody = loop.getBeforeBody();
    auto beforeTerm =
        mlir::cast<mlir::scf::ConditionOp>(beforeBody->getTerminator());
    if (op.getCondition() != beforeTerm.getCondition())
      return mlir::failure();

    for (auto result : op.getResults())
      for (auto user : result.getUsers())
        if (user != beforeTerm)
          return mlir::failure();

    for (auto &nextOp : llvm::make_range(std::next(op->getIterator()),
                                         beforeTerm->getIterator()))
      if (!mlir::isPure(&nextOp))
        return mlir::failure();

    mlir::DominanceInfo dom;
    llvm::SmallSetVector<mlir::Value, 8> capturedValues;
    for (auto body : {op.thenBlock(), op.elseBlock()}) {
      if (!body)
        continue;

      body->walk([&](mlir::Operation *blockOp) {
        for (auto arg : blockOp->getOperands()) {
          if (dom.properlyDominates(arg, loop) ||
              !dom.properlyDominates(arg, op))
            continue;

          capturedValues.insert(arg);
        }
      });
    }

    auto newResTypes = llvm::to_vector(loop.getResultTypes());
    llvm::append_range(
        newResTypes, mlir::ValueRange(capturedValues.getArrayRef()).getTypes());

    mlir::OpBuilder::InsertionGuard g(rewriter);

    auto newBeforeArgs = llvm::to_vector(beforeTerm.getArgs());
    llvm::append_range(newBeforeArgs, capturedValues);
    rewriter.setInsertionPoint(beforeTerm);
    auto newTerm = rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        beforeTerm, beforeTerm.getCondition(), newBeforeArgs);

    rewriter.setInsertionPoint(loop);
    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        loop.getLoc(), newResTypes, loop.getInits(), nullptr, nullptr);

    auto newAfter = newLoop.getAfterBody();
    mlir::ValueRange newAfterArgs = newAfter->getArguments();

    {
      auto replaceChecker = [&](mlir::OpOperand &operand) -> bool {
        auto owner = operand.getOwner();
        return op.getThenRegion().isAncestor(owner->getParentRegion());
      };
      rewriter.replaceUsesWithIf(capturedValues.getArrayRef(),
                                 newAfterArgs.take_back(capturedValues.size()),
                                 replaceChecker);
    }
    if (op.elseBlock()) {
      auto replaceChecker = [&](mlir::OpOperand &operand) -> bool {
        auto owner = operand.getOwner();
        return op.getElseRegion().isAncestor(owner->getParentRegion());
      };
      rewriter.replaceUsesWithIf(
          capturedValues.getArrayRef(),
          newLoop.getResults().take_back(capturedValues.size()),
          replaceChecker);
    }

    auto newBefore = newLoop.getBeforeBody();

    rewriter.inlineBlockBefore(beforeBody, newBefore, newBefore->begin(),
                               newBefore->getArguments());

    auto afterMapping =
        llvm::to_vector(newAfterArgs.drop_back(capturedValues.size()));

    auto thenYield = op.thenYield();
    for (auto &&[res, yieldArg] :
         llvm::zip(op.getResults(), thenYield.getResults())) {
      for (auto &use : res.getUses()) {
        assert(use.getOwner() == newTerm && "Invalid user");
        afterMapping[use.getOperandNumber() - 1] = yieldArg;
      }
    }
    rewriter.eraseOp(thenYield);

    rewriter.inlineBlockBefore(op.thenBlock(), newAfter, newAfter->end());

    auto afterBody = loop.getAfterBody();
    rewriter.inlineBlockBefore(afterBody, newAfter, newAfter->end(),
                               afterMapping);

    afterMapping.clear();
    llvm::append_range(afterMapping,
                       newLoop.getResults().drop_back(capturedValues.size()));

    if (op.elseBlock()) {
      auto elseYield = op.elseYield();
      for (auto &&[res, yieldArg] :
           llvm::zip(op.getResults(), elseYield.getResults())) {
        for (auto &use : res.getUses()) {
          assert(use.getOwner() == newTerm && "Invalid user");
          afterMapping[use.getOperandNumber() - 1] = yieldArg;
        }
      }
      rewriter.eraseOp(elseYield);

      rewriter.inlineBlockBefore(op.elseBlock(), newLoop->getBlock(),
                                 std::next(newLoop->getIterator()));
    }
    rewriter.replaceOp(loop, afterMapping);

    auto termLoc = newTerm.getLoc();
    rewriter.setInsertionPoint(newTerm);
    for (auto res : op.getResults()) {
      mlir::Value newRes =
          rewriter.create<mlir::ub::PoisonOp>(termLoc, res.getType(), nullptr);
      rewriter.replaceAllUsesWith(res, newRes);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static std::optional<llvm::SmallVector<unsigned>>
getArgsMapping(mlir::ValueRange args1, mlir::ValueRange args2) {
  if (args1.size() != args2.size())
    return std::nullopt;

  llvm::SmallVector<unsigned> ret(args1.size());
  for (auto &&[i, arg1] : llvm::enumerate(args1)) {
    auto it = llvm::find(args2, arg1);
    if (it == args2.end())
      return std::nullopt;

    auto j = it - args2.begin();
    ret[j] = static_cast<unsigned>(i);
  }

  return ret;
}

struct WhileOpAlignBeforeArgs
    : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldBefore = loop.getBeforeBody();
    auto oldTerm =
        mlir::cast<mlir::scf::ConditionOp>(oldBefore->getTerminator());
    mlir::ValueRange beforeArgs = oldBefore->getArguments();
    mlir::ValueRange termArgs = oldTerm.getArgs();
    if (beforeArgs == termArgs)
      return mlir::failure();

    auto mapping = getArgsMapping(beforeArgs, termArgs);
    if (!mapping)
      return mlir::failure();

    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(oldTerm);
      rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
          oldTerm, oldTerm.getCondition(), beforeArgs);
    }

    auto oldAfter = loop.getAfterBody();

    llvm::SmallVector<mlir::Type> newResultTypes(beforeArgs.size());
    for (auto &&[i, j] : llvm::enumerate(*mapping))
      newResultTypes[j] = loop.getResult(i).getType();

    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        loop.getLoc(), newResultTypes, loop.getInits(), nullptr, nullptr);
    auto newBefore = newLoop.getBeforeBody();
    auto newAfter = newLoop.getAfterBody();

    llvm::SmallVector<mlir::Value> newResults(beforeArgs.size());
    llvm::SmallVector<mlir::Value> newAfterArgs(beforeArgs.size());
    for (auto &&[i, j] : llvm::enumerate(*mapping)) {
      newResults[i] = newLoop.getResult(j);
      newAfterArgs[i] = newAfter->getArgument(j);
    }

    rewriter.inlineBlockBefore(oldBefore, newBefore, newBefore->begin(),
                               newBefore->getArguments());
    rewriter.inlineBlockBefore(oldAfter, newAfter, newAfter->begin(),
                               newAfterArgs);

    rewriter.replaceOp(loop, newResults);
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

struct ReplacePoisonMath : public mlir::OpRewritePattern<mlir::ub::PoisonOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::ub::PoisonOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    for (auto &use : llvm::make_early_inc_range(op->getUses())) {
      auto owner = use.getOwner();
      if (!mlir::isa<mlir::arith::ArithDialect, mlir::math::MathDialect,
                     mlir::index::IndexDialect>(owner->getDialect()))
        continue;

      if (owner->getNumOperands() != 1 || owner->getNumResults() != 1)
        continue;

      auto resType = owner->getResult(0).getType();
      rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(owner, resType, nullptr);
      changed = true;
    }
    return mlir::success(changed);
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

void numba::populatePoisonOptsPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<ReplacePoisonMath>(patterns.getContext());
}

void numba::populateLoopOptsPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<CanonicalizeLoopMemrefIndex, MoveOpsFromBefore, WhileOpLICM,
                  /*WhileOpExpandTuple,*/ WhileOpMoveIfCond,
                  WhileOpAlignBeforeArgs>(patterns.getContext());
}

void numba::populateCommonOptsPatterns(mlir::RewritePatternSet &patterns) {
  populateCanonicalizationPatterns(patterns);
  populatePoisonOptsPatterns(patterns);
  populateLoopOptsPatterns(patterns);

  patterns.insert<
      // clang-format off
      SubviewLoadPropagate,
      SubviewStorePropagate,
      PowSimplify,
      AndConflictSimplify,
      XorOfCmpF,
      CmpOfArithConst,
      CanonConstSubI,
      ExtractStridedMetadataUnused,
      ExtractStridedMetadataConstStrides,
      ExtractStridedMetadataCast,
      IndexCastOfIndexCast,
      TruncfOfExtf,
      ResTruncFUnary<mlir::arith::NegFOp>,
      ResTruncFBinary<mlir::arith::AddFOp>,
      ResTruncFBinary<mlir::arith::SubFOp>,
      ResTruncFBinary<mlir::arith::MulFOp>,
      ResTruncFBinary<mlir::arith::DivFOp>,
      ResTruncIUnary<mlir::arith::IndexCastOp>,
      ResTruncIUnary<mlir::arith::FPToSIOp>,
      ResTruncIBinary<mlir::arith::AddIOp>,
      ResTruncIBinary<mlir::arith::SubIOp>,
      ResTruncIBinary<mlir::arith::MulIOp>,
      CmpOfIndexCast,
      CmpInvIf,
      MoveArithOutOfIf,
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

static void moveOpsIntoParallel(mlir::scf::ParallelOp outer, int maxDepth,
                                int depth = 0) {
  auto outerBody = outer.getBody();
  auto parallelIt = llvm::find_if(*outerBody, [](auto &op) {
    return mlir::isa<mlir::scf::ParallelOp>(op);
  });
  if (outerBody->end() == parallelIt)
    return;

  auto parallelOp = mlir::cast<mlir::scf::ParallelOp>(*parallelIt);
  auto parallelOpBody = parallelOp.getBody();
  if (parallelIt != outerBody->begin()) {
    auto it = std::prev(parallelIt);
    auto begin = outerBody->begin();
    while (true) {
      bool first = (it == begin);
      auto &op = *it;
      auto isParallelOpOperand = [&](mlir::Operation &op) -> bool {
        auto operands = parallelOp->getOperands();
        for (auto r : op.getResults())
          if (llvm::is_contained(operands, r))
            return true;

        return false;
      };

      auto isUsedOutside = [&](mlir::Operation &op) -> bool {
        auto &region = parallelOp.getRegion();
        for (auto user : op.getUsers())
          if (!region.isAncestor(user->getParentRegion()))
            return true;

        return false;
      };

      if (!mlir::isMemoryEffectFree(&op) || isParallelOpOperand(op) ||
          isUsedOutside(op))
        break;

      if (first) {
        op.moveBefore(&parallelOpBody->front());
        break;
      }

      --it;
      op.moveBefore(&parallelOpBody->front());
    }
  }
  depth += outer.getStep().size();
  if (depth >= maxDepth)
    return;

  moveOpsIntoParallel(parallelOp, maxDepth, depth);
}

namespace {
struct MoveIntoParallel
    : public mlir::PassWrapper<MoveIntoParallel, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MoveIntoParallel)

  void runOnOperation() override {
    getOperation()->walk([&](mlir::scf::ParallelOp op) -> mlir::WalkResult {
      moveOpsIntoParallel(op, 3);
      return mlir::WalkResult::skip();
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createMoveIntoParallelPass() {
  return std::make_unique<MoveIntoParallel>();
}
