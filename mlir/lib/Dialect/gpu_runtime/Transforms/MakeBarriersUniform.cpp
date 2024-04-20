// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

static std::optional<mlir::Attribute> getNeutralValue(mlir::Region &region) {
  if (!llvm::hasSingleElement(region))
    return std::nullopt;

  mlir::Block &block = region.front();
  auto body = block.without_terminator();
  if (!llvm::hasSingleElement(body))
    return std::nullopt;

  return mlir::arith::getNeutralElement(&(*body.begin()));
}

static int64_t getMinVal(unsigned bits) {
  assert(bits > 0 && bits <= 64);
  return static_cast<int64_t>(1) << (bits - 1);
}

static int64_t getMaxVal(unsigned bits) { return getMinVal(bits) - 1; }

// TODO: Upstream
static std::optional<mlir::Attribute>
getNeutralValue(mlir::Type resultType, mlir::gpu::AllReduceOperation op) {
  using Op = mlir::gpu::AllReduceOperation;
  // Builder only used as helper for attribute creation.
  mlir::OpBuilder b(resultType.getContext());
  if (auto floatType = resultType.dyn_cast<mlir::FloatType>()) {
    const llvm::fltSemantics &semantic = floatType.getFloatSemantics();
    if (op == Op::ADD)
      return b.getFloatAttr(resultType, llvm::APFloat::getZero(semantic));
    if (op == Op::MUL)
      return b.getFloatAttr(resultType, llvm::APFloat(semantic, 1));
    if (op == Op::MAXIMUMF || op == Op::MAXNUMF)
      return b.getFloatAttr(resultType,
                            llvm::APFloat::getInf(semantic, /*Negative=*/true));
    if (op == Op::MINIMUMF || op == Op::MINNUMF)
      return b.getFloatAttr(
          resultType, llvm::APFloat::getInf(semantic, /*Negative=*/false));
    return std::nullopt;
  }
  if (op == Op::ADD || op == Op::OR || op == Op::XOR)
    return b.getIntegerAttr(resultType, 0);
  if (op == Op::AND)
    return b.getIntegerAttr(resultType, -1);
  if (op == Op::MAXSI)
    return b.getIntegerAttr(resultType,
                            getMinVal(resultType.getIntOrFloatBitWidth()));
  if (op == Op::MINSI)
    return b.getIntegerAttr(resultType,
                            getMaxVal(resultType.getIntOrFloatBitWidth()));
  if (op == Op::MUL)
    return b.getIntegerAttr(resultType, 1);
  return std::nullopt;
}

static std::optional<mlir::Attribute>
getNeutralValue(mlir::gpu::AllReduceOp reduceOp) {
  if (auto res = getNeutralValue(reduceOp.getBody()))
    return res;

  if (auto op = reduceOp.getOp())
    if (auto res = getNeutralValue(reduceOp.getResult().getType(), *op))
      return res;

  return std::nullopt;
}

static std::optional<mlir::Attribute>
getNeutralValue(mlir::gpu::SubgroupReduceOp reduceOp) {
  return getNeutralValue(reduceOp.getResult().getType(), reduceOp.getOp());
}

static mlir::LogicalResult convertBlockingOp(mlir::Operation *op,
                                             mlir::PatternRewriter &rewriter) {
  // IfOp must be an immediate parent
  auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op->getParentOp());
  if (!ifOp)
    return mlir::failure();

  // IfOp with else block is not yet supported;
  if (ifOp.elseBlock())
    return mlir::failure();

  mlir::TypedAttr neutralValAttr;
  if (auto reduceOp = mlir::dyn_cast<mlir::gpu::AllReduceOp>(op)) {
    auto nval = getNeutralValue(reduceOp);
    if (!nval)
      return mlir::failure();

    neutralValAttr = mlir::cast<mlir::TypedAttr>(*nval);
  } else if (auto reduceOp = mlir::dyn_cast<mlir::gpu::SubgroupReduceOp>(op)) {
    auto nval = getNeutralValue(reduceOp);
    if (!nval)
      return mlir::failure();

    neutralValAttr = mlir::cast<mlir::TypedAttr>(*nval);
  }

  mlir::Block *ifBody = ifOp.thenBlock();
  assert(ifBody);

  mlir::DominanceInfo dom;
  llvm::SmallMapVector<mlir::Value, unsigned, 8> yieldArgsMap;

  auto barrierIt = op->getIterator();
  for (auto &beforeOp : llvm::make_range(ifBody->begin(), barrierIt)) {
    for (auto result : beforeOp.getResults()) {
      for (mlir::OpOperand &&user : result.getUsers()) {
        auto owner = user.getOwner();
        if (dom.dominates(op, owner)) {
          auto idx = static_cast<unsigned>(yieldArgsMap.size());
          yieldArgsMap.insert({result, idx});
        }
      }
    }
  }

  auto yieldArgs = llvm::to_vector(llvm::make_first_range(yieldArgsMap));

  auto afterBlock = rewriter.splitBlock(ifBody, std::next(barrierIt));
  auto beforeBlock = rewriter.splitBlock(ifBody, ifBody->begin());

  rewriter.setInsertionPointToEnd(beforeBlock);
  rewriter.create<mlir::scf::YieldOp>(rewriter.getUnknownLoc(), yieldArgs);

  rewriter.setInsertionPoint(ifOp);
  auto ifLoc = ifOp->getLoc();
  auto cond = ifOp.getCondition();

  auto emptyBodyBuilder = [&](mlir::OpBuilder & /*builder*/,
                              mlir::Location /*loc*/) {};

  auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
    llvm::SmallVector<mlir::Value> results;
    results.reserve(yieldArgs.size());
    for (auto arg : yieldArgs) {
      auto val =
          builder.create<mlir::ub::PoisonOp>(loc, arg.getType(), nullptr);
      results.emplace_back(val);
    }

    builder.create<mlir::scf::YieldOp>(loc, results);
  };

  mlir::ValueRange yieldArgsRange(yieldArgs);
  auto beforeIf =
      rewriter.create<mlir::scf::IfOp>(ifLoc, cond, bodyBuilder, bodyBuilder);
  for (auto &op :
       llvm::make_early_inc_range(llvm::reverse(*beforeIf.thenBlock())))
    rewriter.eraseOp(&op);

  rewriter.mergeBlocks(beforeBlock, beforeIf.thenBlock());
  auto beforeIfResults = beforeIf.getResults();

  auto getBeforeIfResult = [&](unsigned i) {
    assert(i < beforeIfResults.size() && "Invalid result index.");
    return beforeIfResults[i];
  };

  if (auto barrierOp = mlir::dyn_cast<mlir::gpu::BarrierOp>(op)) {
    // Use clone to preserve user-defined attrs.
    auto newOp = rewriter.clone(*barrierOp);
    rewriter.replaceOp(barrierOp, newOp->getResults());
  } else if (auto reduceOp = mlir::dyn_cast<mlir::gpu::AllReduceOp>(op)) {
    auto loc = reduceOp.getLoc();
    auto reduceArg = reduceOp.getValue();
    assert(yieldArgsMap.count(reduceArg));
    auto mappedReduceArg =
        getBeforeIfResult(yieldArgsMap.find(reduceArg)->second);

    assert(neutralValAttr);
    auto neutralVal =
        rewriter.create<mlir::arith::ConstantOp>(loc, neutralValAttr);
    auto val = rewriter.create<mlir::arith::SelectOp>(
        loc, cond, mappedReduceArg, neutralVal);

    mlir::IRMapping mapping;
    mapping.map(reduceArg, val);
    auto newOp = rewriter.clone(*reduceOp, mapping);
    rewriter.replaceOp(op, newOp->getResults());
  } else if (auto reduceOp = mlir::dyn_cast<mlir::gpu::SubgroupReduceOp>(op)) {
    auto loc = reduceOp.getLoc();
    auto reduceArg = reduceOp.getValue();
    assert(yieldArgsMap.count(reduceArg));
    auto mappedReduceArg =
        getBeforeIfResult(yieldArgsMap.find(reduceArg)->second);

    assert(neutralValAttr);
    auto neutralVal =
        rewriter.create<mlir::arith::ConstantOp>(loc, neutralValAttr);
    auto val = rewriter.create<mlir::arith::SelectOp>(
        loc, cond, mappedReduceArg, neutralVal);

    mlir::IRMapping mapping;
    mapping.map(reduceArg, val);
    auto newOp = rewriter.clone(*reduceOp, mapping);
    rewriter.replaceOp(op, newOp->getResults());
  } else {
    llvm_unreachable("Unsupported barrier op");
  }

  auto afterIf =
      rewriter.create<mlir::scf::IfOp>(ifLoc, cond, emptyBodyBuilder);
  rewriter.mergeBlocks(afterBlock, afterIf.thenBlock());

  afterIf.thenBlock()->walk([&](mlir::Operation *innerOp) {
    for (mlir::OpOperand &arg : innerOp->getOpOperands()) {
      auto val = arg.get();
      auto it = yieldArgsMap.find(val);
      if (it != yieldArgsMap.end()) {
        auto newVal = getBeforeIfResult(it->second);
        rewriter.modifyOpInPlace(innerOp, [&]() { arg.set(newVal); });
      }
    }
  });

  rewriter.eraseOp(ifOp);
  return mlir::success();
}

namespace {
struct ConvertBarrierOp : public mlir::OpRewritePattern<mlir::gpu::BarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op,
                  mlir::PatternRewriter &rewriter) const override {
    return convertBlockingOp(op, rewriter);
  }
};

struct ConvertAllReduceOp
    : public mlir::OpRewritePattern<mlir::gpu::AllReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllReduceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    return convertBlockingOp(op, rewriter);
  }
};

struct ConvertSubgroupReduceOp
    : public mlir::OpRewritePattern<mlir::gpu::SubgroupReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::SubgroupReduceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    return convertBlockingOp(op, rewriter);
  }
};
} // namespace

void gpu_runtime::populateMakeBarriersUniformPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns
      .insert<ConvertBarrierOp, ConvertAllReduceOp, ConvertSubgroupReduceOp>(
          patterns.getContext());
}

namespace {
struct MakeBarriersUniformPass
    : public mlir::PassWrapper<MakeBarriersUniformPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MakeBarriersUniformPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::ub::UBDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    gpu_runtime::populateMakeBarriersUniformPatterns(patterns);

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true; // We need to visit top barriers first
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> gpu_runtime::createMakeBarriersUniformPass() {
  return std::make_unique<MakeBarriersUniformPass>();
}
