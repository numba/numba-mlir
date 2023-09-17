// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/PromoteToParallel.hpp"

#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Transforms/CommonOpts.hpp"
#include "numba/Transforms/ConstUtils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

static bool hasSideEffects(mlir::Operation *op) {
  assert(op);
  for (auto &region : op->getRegions()) {
    auto visitor = [](mlir::Operation *bodyOp) -> mlir::WalkResult {
      if (mlir::isa<mlir::scf::ReduceOp>(bodyOp) ||
          bodyOp->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>() ||
          bodyOp->hasTrait<mlir::OpTrait::IsTerminator>())
        return mlir::WalkResult::advance();

      if (mlir::isa<mlir::CallOpInterface>(bodyOp))
        return mlir::WalkResult::interrupt();

      auto memEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(bodyOp);
      if (!memEffects || memEffects.hasEffect<mlir::MemoryEffects::Write>())
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::advance();
    };
    if (region.walk(visitor).wasInterrupted())
      return true;
  }
  return false;
}

static bool canParallelizeLoop(mlir::Operation *op, bool hasParallelAttr) {
  return hasParallelAttr || !hasSideEffects(op);
}

using CheckFunc = bool (*)(mlir::Operation *, mlir::Value);
using LowerFunc = void (*)(mlir::OpBuilder &, mlir::Location, mlir::Value,
                           mlir::Operation *);

template <typename Op>
static bool simpleCheck(mlir::Operation *op, mlir::Value /*iterVar*/) {
  return mlir::isa<Op>(op);
}

template <typename Op>
static bool lhsArgCheck(mlir::Operation *op, mlir::Value iterVar) {
  auto casted = mlir::dyn_cast<Op>(op);
  if (!casted)
    return false;

  return casted.getLhs() == iterVar;
}

template <typename Op>
static void simpleLower(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value val, mlir::Operation *origOp) {
  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value lhs,
                         mlir::Value rhs) {
    auto casted = mlir::cast<Op>(origOp);
    mlir::IRMapping mapper;
    mapper.map(casted.getLhs(), lhs);
    mapper.map(casted.getRhs(), rhs);
    mlir::Value res = b.clone(*origOp, mapper)->getResult(0);
    b.create<mlir::scf::ReduceReturnOp>(l, res);
  };
  builder.create<mlir::scf::ReduceOp>(loc, val, bodyBuilder);
}

template <typename SubOp, typename AddOp>
static void subLower(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value val, mlir::Operation *origOp) {
  auto type = val.getType();
  auto zeroAttr = numba::getConstAttr(type, 0.0);
  auto zero = builder.create<mlir::arith::ConstantOp>(loc, type, zeroAttr);
  val = builder.create<SubOp>(loc, zero, val);
  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value lhs,
                         mlir::Value rhs) {
    mlir::Value res = b.create<AddOp>(l, lhs, rhs);
    b.create<mlir::scf::ReduceReturnOp>(l, res);
  };
  builder.create<mlir::scf::ReduceOp>(loc, val, bodyBuilder);
}

template <typename DivOp, typename MulOp>
static void divLower(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value val, mlir::Operation *origOp) {
  auto type = val.getType();
  auto oneAttr = numba::getConstAttr(type, 1.0);
  auto one = builder.create<mlir::arith::ConstantOp>(loc, type, oneAttr);
  val = builder.create<DivOp>(loc, one, val);
  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value lhs,
                         mlir::Value rhs) {
    mlir::Value res = b.create<MulOp>(l, lhs, rhs);
    b.create<mlir::scf::ReduceReturnOp>(l, res);
  };
  builder.create<mlir::scf::ReduceOp>(loc, val, bodyBuilder);
}

template <typename Op>
static constexpr std::pair<CheckFunc, LowerFunc> getSimpleHandler() {
  return {&simpleCheck<Op>, &simpleLower<Op>};
}

namespace arith = mlir::arith;
static const constexpr std::pair<CheckFunc, LowerFunc> promoteHandlers[] = {
    // clang-format off
    getSimpleHandler<arith::AddIOp>(),
    getSimpleHandler<arith::AddFOp>(),

    getSimpleHandler<arith::MulIOp>(),
    getSimpleHandler<arith::MulFOp>(),

    getSimpleHandler<arith::MinSIOp>(),
    getSimpleHandler<arith::MinUIOp>(),
    getSimpleHandler<arith::MinimumFOp>(),

    getSimpleHandler<arith::MaxSIOp>(),
    getSimpleHandler<arith::MaxUIOp>(),
    getSimpleHandler<arith::MaximumFOp>(),

    {&lhsArgCheck<arith::SubIOp>, &subLower<arith::SubIOp, arith::AddIOp>},
    {&lhsArgCheck<arith::SubFOp>, &subLower<arith::SubFOp, arith::AddFOp>},

    {&lhsArgCheck<arith::DivFOp>, &divLower<arith::DivFOp, arith::MulFOp>},
    // clang-format on
};

static LowerFunc getLowerer(mlir::Operation *op, mlir::Value iterVar) {
  assert(op);
  for (auto &&[checker, lowerer] : promoteHandlers)
    if (checker(op, iterVar))
      return lowerer;

  return nullptr;
}

static bool isInsideParallelRegion(mlir::Operation *op) {
  assert(op && "Invalid op");
  while (true) {
    auto region = op->getParentOfType<numba::util::EnvironmentRegionOp>();
    if (!region)
      return false;

    if (mlir::isa<numba::util::ParallelAttr>(region.getEnvironment()))
      return true;

    op = region;
  }
}

static bool checkIndexType(mlir::Operation *op) {
  auto type = op->getResult(0).getType();
  if (mlir::isa<mlir::IndexType>(type))
    return true;

  // TODO: check datalayout
  if (type.isSignlessInteger(64))
    return true;

  return false;
}

namespace {
struct PromoteWhileOp : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Block *beforeBody = loop.getBeforeBody();
    if (!llvm::hasSingleElement(beforeBody->without_terminator()))
      return rewriter.notifyMatchFailure(loop, "Loop body must have single op");

    auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(beforeBody->front());
    if (!cmp)
      return rewriter.notifyMatchFailure(loop,
                                         "Loop body must have single cmp op");

    auto beforeTerm =
        mlir::cast<mlir::scf::ConditionOp>(beforeBody->getTerminator());
    if (!llvm::hasSingleElement(cmp->getUses()) &&
        beforeTerm.getCondition() == cmp.getResult())
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Expected single condiditon use: " << *cmp;
      });

    if (mlir::ValueRange(beforeBody->getArguments()) != beforeTerm.getArgs())
      return rewriter.notifyMatchFailure(loop, "Invalid args order");

    using Pred = mlir::arith::CmpIPredicate;
    auto predicate = cmp.getPredicate();
    if (predicate != Pred::slt && predicate != Pred::sgt)
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Expected 'slt' or 'sgt' predicate: " << *cmp;
      });

    if (!checkIndexType(cmp))
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Expected index like type: " << *cmp;
      });

    mlir::BlockArgument iterVar;
    mlir::Value end;
    mlir::DominanceInfo dom;
    for (bool reverse : {false, true}) {
      auto expectedPred = reverse ? Pred::sgt : Pred::slt;
      if (cmp.getPredicate() != expectedPred)
        continue;

      auto arg1 = reverse ? cmp.getRhs() : cmp.getLhs();
      auto arg2 = reverse ? cmp.getLhs() : cmp.getRhs();
      if (!mlir::isa<mlir::BlockArgument>(arg1))
        continue;

      if (!dom.properlyDominates(arg2, loop))
        continue;

      iterVar = mlir::cast<mlir::BlockArgument>(arg1);
      end = arg2;
    }

    if (!iterVar)
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Unrecognized cmp form: " << *cmp;
      });

    if (!llvm::hasNItems(iterVar.getUses(), 2))
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Unrecognized iter var: " << iterVar;
      });

    auto &iterVarOperand = [&]() -> mlir::OpOperand & {
      for (auto &use : iterVar.getUses()) {
        if (use.getOwner() == beforeTerm)
          return use;
      }
      llvm_unreachable("Invalid IR");
    }();

    mlir::Block *afterBody = loop.getAfterBody();
    auto afterTerm = mlir::cast<mlir::scf::YieldOp>(afterBody->getTerminator());
    auto argNumber = iterVar.getArgNumber();
    auto afterTermIterArg = afterTerm.getResults()[argNumber];

    auto iterVarAfter =
        afterBody->getArgument(iterVarOperand.getOperandNumber());

    mlir::Value step;
    for (auto user : iterVarAfter.getUsers()) {
      auto owner = mlir::dyn_cast<mlir::arith::AddIOp>(user);
      if (!owner)
        continue;

      auto other =
          (iterVarAfter == owner.getLhs() ? owner.getRhs() : owner.getLhs());
      if (!dom.properlyDominates(other, loop))
        continue;

      if (afterTermIterArg != owner.getResult())
        continue;

      step = other;
    }

    if (!step)
      return rewriter.notifyMatchFailure(loop, "Didn't found suitable add op");

    auto begin = loop.getInits()[iterVarOperand.getOperandNumber()];

    auto loc = loop.getLoc();
    auto indexType = rewriter.getIndexType();
    auto toIndex = [&](mlir::Value val) -> mlir::Value {
      if (val.getType() != indexType)
        return rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, val);

      return val;
    };
    begin = toIndex(begin);
    end = toIndex(end);
    step = toIndex(step);

    llvm::SmallVector<mlir::Value> mapping;
    for (auto &&[i, init] : llvm::enumerate(loop.getInits())) {
      if (i == argNumber)
        continue;

      mapping.emplace_back(init);
    }
    auto emptyBuidler = [](mlir::OpBuilder &, mlir::Location, mlir::Value,
                           mlir::ValueRange) {};
    auto newLoop = rewriter.create<mlir::scf::ForOp>(loc, begin, end, step,
                                                     mapping, emptyBuidler);
    mlir::Block &newBody = newLoop.getLoopBody().front();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&newBody);
    mlir::Value newIterVar = newBody.getArgument(0);
    if (newIterVar.getType() != iterVar.getType())
      newIterVar = rewriter.create<mlir::arith::IndexCastOp>(
          loc, iterVar.getType(), newIterVar);

    mapping.resize(newBody.getNumArguments());
    for (auto &&[i, arg] : llvm::enumerate(newBody.getArguments())) {
      if (i < argNumber) {
        mapping[i + 1] = arg;
      } else if (i == argNumber) {
        mapping[0] = arg;
      } else {
        mapping[i] = arg;
      }
    }

    rewriter.inlineBlockBefore(beforeBody, &newBody, newBody.begin(), mapping);

    auto term = mlir::cast<mlir::scf::YieldOp>(newBody.getTerminator());

    mapping.clear();
    for (auto &&[i, arg] : llvm::enumerate(term.getResults())) {
      if (i == argNumber)
        continue;

      mapping.emplace_back(arg);
    }

    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(term, mapping);

    rewriter.setInsertionPointAfter(newLoop);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value stepDec = rewriter.create<mlir::arith::SubIOp>(loc, step, one);
    mlir::Value len = rewriter.create<mlir::arith::SubIOp>(loc, end, begin);
    len = rewriter.create<mlir::arith::AddIOp>(loc, len, stepDec);
    len = rewriter.create<mlir::arith::DivSIOp>(loc, len, step);
    mlir::Value res = rewriter.create<mlir::arith::MulIOp>(loc, len, step);
    res = rewriter.create<mlir::arith::AddIOp>(loc, begin, res);
    if (res.getType() != iterVar.getType())
      res = rewriter.create<mlir::arith::IndexCastOp>(loc, iterVar.getType(),
                                                      res);

    mapping.clear();
    llvm::append_range(mapping, newLoop.getResults());
    mapping.insert(mapping.begin() + argNumber, res);
    rewriter.replaceOp(loop, mapping);
    return mlir::failure();
  }
};

struct PromoteToParallel : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!canParallelizeLoop(op, isInsideParallelRegion(op)))
      return mlir::failure();

    mlir::Block &loopBody = op.getLoopBody().front();
    auto term = mlir::cast<mlir::scf::YieldOp>(loopBody.getTerminator());
    auto iterVars = op.getRegionIterArgs();
    assert(iterVars.size() == term.getResults().size());

    using ReductionDesc = std::tuple<mlir::Operation *, LowerFunc, mlir::Value>;
    llvm::SmallVector<ReductionDesc> reductionOps;
    llvm::SmallDenseSet<mlir::Operation *> reductionOpsSet;
    for (auto &&[iterVar, result] : llvm::zip(iterVars, term.getResults())) {
      auto reductionOp = result.getDefiningOp();
      if (!reductionOp || reductionOp->getNumResults() != 1 ||
          reductionOp->getNumOperands() != 2 ||
          !llvm::hasSingleElement(reductionOp->getUses()))
        return mlir::failure();

      mlir::Value reductionArg;
      if (reductionOp->getOperand(0) == iterVar) {
        reductionArg = reductionOp->getOperand(1);
      } else if (reductionOp->getOperand(1) == iterVar) {
        reductionArg = reductionOp->getOperand(0);
      } else {
        return mlir::failure();
      }

      auto lowerer = getLowerer(reductionOp, iterVar);
      if (!lowerer)
        return mlir::failure();

      reductionOps.emplace_back(reductionOp, lowerer, reductionArg);
      reductionOpsSet.insert(reductionOp);
    }

    auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange iterVals, mlir::ValueRange) {
      assert(1 == iterVals.size());
      mlir::IRMapping mapping;
      mapping.map(op.getInductionVar(), iterVals.front());
      for (auto &oldOp : loopBody.without_terminator())
        if (0 == reductionOpsSet.count(&oldOp))
          builder.clone(oldOp, mapping);

      for (auto &&[reductionOp, lowerer, reductionArg] : reductionOps) {
        auto arg = mapping.lookupOrDefault(reductionArg);
        lowerer(builder, loc, arg, reductionOp);
      }
      builder.create<mlir::scf::YieldOp>(loc);
    };

    rewriter.replaceOpWithNewOp<mlir::scf::ParallelOp>(
        op, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        op.getInitArgs(), bodyBuilder);

    return mlir::success();
  }
};

struct MergeNestedForIntoParallel
    : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parent = mlir::dyn_cast<mlir::scf::ForOp>(op->getParentOp());
    if (!parent)
      return mlir::failure();

    auto &block = parent.getLoopBody().front();
    if (!llvm::hasSingleElement(block.without_terminator()))
      return mlir::failure();

    if (parent.getInitArgs().size() != op.getInitVals().size())
      return mlir::failure();

    auto yield = mlir::cast<mlir::scf::YieldOp>(block.getTerminator());
    assert(yield.getNumOperands() == op.getNumResults());
    for (auto &&[arg, initVal, result, yieldOp] :
         llvm::zip(block.getArguments().drop_front(), op.getInitVals(),
                   op.getResults(), yield.getOperands())) {
      if (!arg.hasOneUse() || arg != initVal || result != yieldOp)
        return mlir::failure();
    }
    auto checkVals = [&](auto vals) {
      for (auto val : vals)
        if (val.getParentBlock() == &block)
          return true;

      return false;
    };
    if (checkVals(op.getLowerBound()) || checkVals(op.getUpperBound()) ||
        checkVals(op.getStep()))
      return mlir::failure();

    if (!canParallelizeLoop(op, isInsideParallelRegion(op)))
      return mlir::failure();

    auto makeValueList = [](auto op, auto ops) {
      llvm::SmallVector<mlir::Value> ret;
      ret.reserve(ops.size() + 1);
      ret.emplace_back(op);
      ret.append(ops.begin(), ops.end());
      return ret;
    };

    auto lowerBounds =
        makeValueList(parent.getLowerBound(), op.getLowerBound());
    auto upperBounds =
        makeValueList(parent.getUpperBound(), op.getUpperBound());
    auto steps = makeValueList(parent.getStep(), op.getStep());

    auto &oldBody = op.getLoopBody().front();
    auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location /*loc*/,
                           mlir::ValueRange iter_vals, mlir::ValueRange temp) {
      assert(iter_vals.size() == lowerBounds.size());
      assert(temp.empty());
      mlir::IRMapping mapping;
      assert((oldBody.getNumArguments() + 1) == iter_vals.size());
      mapping.map(block.getArgument(0), iter_vals.front());
      mapping.map(oldBody.getArguments(), iter_vals.drop_front());
      for (auto &op : oldBody.without_terminator())
        builder.clone(op, mapping);
    };

    rewriter.setInsertionPoint(parent);
    rewriter.replaceOpWithNewOp<mlir::scf::ParallelOp>(
        parent, lowerBounds, upperBounds, steps, parent.getInitArgs(),
        bodyBuilder);

    return mlir::success();
  }
};

struct PromoteWhilePass
    : public mlir::PassWrapper<PromoteWhilePass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteWhilePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto context = &getContext();

    mlir::RewritePatternSet patterns(context);
    numba::populatePromoteWhilePatterns(patterns);
    numba::populateLoopOptsPatterns(patterns);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      signalPassFailure();
  }
};

struct PromoteToParallelPass
    : public mlir::PassWrapper<PromoteToParallelPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteToParallelPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto context = &getContext();

    mlir::RewritePatternSet patterns(context);
    numba::populatePromoteToParallelPatterns(patterns);
    numba::populateLoopOptsPatterns(patterns);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void numba::populatePromoteWhilePatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<PromoteWhileOp>(patterns.getContext());
}

void numba::populatePromoteToParallelPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.insert<PromoteToParallel, MergeNestedForIntoParallel>(
      patterns.getContext());
}

std::unique_ptr<mlir::Pass> numba::createPromoteWhilePass() {
  return std::make_unique<PromoteWhilePass>();
}

std::unique_ptr<mlir::Pass> numba::createPromoteToParallelPass() {
  return std::make_unique<PromoteToParallelPass>();
}
