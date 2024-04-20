// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/CfgToScf.hpp"
#include "numba/Transforms/CommonOpts.hpp"

#include <mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/CFGToSCF.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/RegionUtils.h>

namespace {
static bool hasSideEffects(mlir::Operation *op) {
  assert(op);
  if (mlir::isa<mlir::CallOpInterface>(op))
    return true;

  return !mlir::isPure(op);
}

static bool isInverse(mlir::Value cond1, mlir::Value cond2) {
  auto op = cond1.getDefiningOp<mlir::arith::XOrIOp>();
  if (!op)
    return false;

  if ((op.getLhs() == cond2 && mlir::isConstantIntValue(op.getRhs(), -1)) ||
      (op.getRhs() == cond2 && mlir::isConstantIntValue(op.getLhs(), -1)))
    return true;

  return false;
}

struct WhileMoveToAfter : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto &beforeBlock = op.getBefore().front();
    auto it = beforeBlock.end();
    auto begin = beforeBlock.begin();
    if (it == begin)
      return mlir::failure();

    mlir::scf::IfOp ifOp;
    while (true) {
      --it;
      if (it == begin)
        return mlir::failure();

      auto &itOp = *it;
      if (auto i = mlir::dyn_cast<mlir::scf::IfOp>(itOp)) {
        ifOp = i;
        break;
      }

      if (hasSideEffects(&itOp))
        return mlir::failure();
    }
    auto condOp =
        mlir::cast<mlir::scf::ConditionOp>(beforeBlock.getTerminator());

    for (auto user : ifOp->getUsers()) {
      if (user != condOp)
        return mlir::failure();
    }

    bool inverse = false;
    auto condCond = condOp.getCondition();
    auto cond = ifOp.getCondition();
    if (condCond == cond) {
      // Nothing
    } else if (isInverse(condCond, cond)) {
      inverse = true;
    } else {
      return mlir::failure();
    }

    auto otherBlock = inverse ? ifOp.thenBlock() : ifOp.elseBlock();
    mlir::scf::YieldOp otherTerm;
    if (otherBlock) {
      auto otherBlockRegion = otherBlock->getParent();
      otherTerm = mlir::cast<mlir::scf::YieldOp>(otherBlock->getTerminator());

      for (auto arg : otherTerm.getResults()) {
        if (arg.getDefiningOp<mlir::ub::PoisonOp>())
          continue;

        if (!otherBlockRegion->isAncestor(arg.getParentRegion()))
          continue;

        return mlir::failure();
      }
    }

    auto ifBlock = inverse ? ifOp.elseBlock() : ifOp.thenBlock();
    if (!ifBlock)
      return mlir::failure();

    auto ifBlockRegion = ifBlock->getParent();
    auto term = mlir::cast<mlir::scf::YieldOp>(ifBlock->getTerminator());

    llvm::SmallVector<mlir::Value> definedOutside;
    ifBlock->walk([&](mlir::Operation *innerOp) {
      for (auto arg : innerOp->getOperands())
        if (!ifBlockRegion->isAncestor(arg.getParentRegion()))
          definedOutside.emplace_back(arg);
    });

    auto loc = rewriter.getUnknownLoc();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    auto &afterBlock = op.getAfter().front();
    for (auto val : definedOutside)
      afterBlock.addArgument(val.getType(), loc);

    mlir::ValueRange oldArgs =
        afterBlock.getArguments().drop_back(definedOutside.size());
    mlir::ValueRange newArgs =
        afterBlock.getArguments().take_back(definedOutside.size());

    mlir::IRMapping mapping;
    for (auto &&[oldVal, newVal] : llvm::zip(definedOutside, newArgs))
      mapping.map(oldVal, newVal);

    rewriter.setInsertionPointToStart(&afterBlock);
    for (auto &op : ifBlock->without_terminator())
      rewriter.clone(op, mapping);

    for (auto &&[res, yieldRes] :
         llvm::zip(ifOp.getResults(), term.getResults())) {
      for (auto &use : res.getUses()) {
        assert(use.getOwner() == condOp);
        assert(use.getOperandNumber() > 0);
        auto idx = use.getOperandNumber() - 1; // skip condition
        assert(idx < oldArgs.size());
        mlir::Value newVal = mapping.lookupOrDefault(yieldRes);
        rewriter.replaceAllUsesWith(oldArgs[idx], newVal);
      }
    }

    rewriter.setInsertionPoint(condOp);
    if (otherTerm) {
      for (auto &&[res, otherRes] :
           llvm::zip(ifOp.getResults(), otherTerm.getResults())) {
        if (otherRes.getDefiningOp<mlir::ub::PoisonOp>()) {
          mlir::Value undef =
              rewriter.create<mlir::ub::PoisonOp>(loc, res.getType(), nullptr);
          rewriter.replaceAllUsesWith(res, undef);
        } else {
          rewriter.replaceAllUsesWith(res, otherRes);
        }
      }
    }

    rewriter.eraseOp(ifOp);

    auto condArgs = condOp.getArgs();
    llvm::SmallVector<mlir::Value> newCondArgs(condArgs.begin(),
                                               condArgs.end());
    newCondArgs.append(definedOutside.begin(), definedOutside.end());

    assert(newCondArgs.size() == afterBlock.getNumArguments());
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(condOp, condCond,
                                                        newCondArgs);

    mlir::ValueRange newCondArgsRange(newCondArgs);

    auto emptyBuilder = [](mlir::OpBuilder &, mlir::Location,
                           mlir::ValueRange) {
      // Nothing
    };

    rewriter.setInsertionPoint(op);
    auto newWhile = rewriter.create<mlir::scf::WhileOp>(
        op.getLoc(), newCondArgsRange.getTypes(), op.getInits(), emptyBuilder,
        emptyBuilder);
    auto &newBeforeBlock = newWhile.getBefore().front();
    auto &newAfterBlock = newWhile.getAfter().front();
    rewriter.mergeBlocks(&beforeBlock, &newBeforeBlock,
                         newBeforeBlock.getArguments());
    rewriter.mergeBlocks(&afterBlock, &newAfterBlock,
                         newAfterBlock.getArguments());
    rewriter.replaceOp(op, newWhile.getResults().take_front(condArgs.size()));

    return mlir::success();
  }
};

struct WhileReductionSelect
    : public mlir::OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto condArg = mlir::dyn_cast<mlir::BlockArgument>(op.getCondition());
    if (!condArg)
      return mlir::failure();

    auto region = condArg.getParentRegion();
    auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(region->getParentOp());
    if (!whileOp)
      return mlir::failure();

    auto &afterBlock = whileOp.getAfter().front();
    auto yield = mlir::cast<mlir::scf::YieldOp>(afterBlock.getTerminator());

    auto inits = whileOp.getInits();
    auto initArg = inits[condArg.getArgNumber()];
    auto yieldArg = yield.getResults()[condArg.getArgNumber()];
    if (!mlir::isConstantIntValue(initArg, -1) ||
        !mlir::isConstantIntValue(yieldArg, 0))
      return mlir::failure();

    auto trueArg = op.getTrueValue();
    if (!mlir::DominanceInfo().dominates(trueArg, whileOp))
      return mlir::failure();

    auto falseArg = mlir::dyn_cast<mlir::BlockArgument>(op.getFalseValue());
    if (!falseArg || falseArg.getParentRegion() != region)
      return mlir::failure();

    if (!inits[falseArg.getArgNumber()].getDefiningOp<mlir::ub::PoisonOp>())
      return mlir::failure();

    mlir::SmallVector<mlir::Value> newInits(inits.begin(), inits.end());
    newInits[falseArg.getArgNumber()] = trueArg;
    rewriter.modifyOpInPlace(
        whileOp, [&]() { whileOp.getInitsMutable().assign(newInits); });
    rewriter.replaceOp(op, falseArg);
    return mlir::success();
  }
};

struct WhileUndefArgs : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    auto yield = op.getYieldOp();

    llvm::SmallVector<mlir::Value> newArgs;
    for (auto &&[yieldArg, init] :
         llvm::zip(yield.getResults(), op.getInits())) {
      if (yieldArg.getDefiningOp<mlir::ub::PoisonOp>() && yieldArg != init) {
        changed = true;
        newArgs.emplace_back(init);
        continue;
      }

      newArgs.emplace_back(yieldArg);
    }

    if (!changed)
      return mlir::failure();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(yield, newArgs);
    return mlir::success();
  }
};

// TODO: upstream
struct CondBrSameTarget
    : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto target = op.getTrueDest();
    if (op.getFalseDest() != target)
      return mlir::failure();

    llvm::SmallVector<mlir::Value> args;
    args.reserve(target->getNumArguments());

    auto loc = op.getLoc();
    auto cond = op.getCondition();
    for (auto &&[trueArg, falseArg] :
         llvm::zip(op.getTrueDestOperands(), op.getFalseDestOperands())) {
      if (trueArg == falseArg) {
        args.emplace_back(trueArg);
        continue;
      }

      mlir::Value arg =
          rewriter.create<mlir::arith::SelectOp>(loc, cond, trueArg, falseArg);
      args.emplace_back(arg);
    }

    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, target, args);
    return mlir::success();
  }
};

// TODO: upstream
struct OrOfXor : public mlir::OpRewritePattern<mlir::arith::OrIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::OrIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    for (bool reverse : {false, true}) {
      auto xorOp = (reverse ? op.getLhs() : op.getRhs())
                       .getDefiningOp<mlir::arith::XOrIOp>();
      if (!xorOp)
        continue;

      auto arg = reverse ? op.getRhs() : op.getLhs();
      for (bool reverseXor : {false, true}) {
        auto arg1 = reverseXor ? xorOp.getLhs() : xorOp.getRhs();
        auto arg2 = reverseXor ? xorOp.getRhs() : xorOp.getLhs();
        if (arg1 != arg || !mlir::isConstantIntValue(arg2, -1))
          continue;

        rewriter.replaceOp(op, arg2);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

struct HoistSelects : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    rewriter.startOpModification(op);

    mlir::DominanceInfo dom;
    for (auto block : {op.thenBlock(), op.elseBlock()}) {
      if (!block)
        continue;

      for (auto blockOp :
           llvm::make_early_inc_range(block->getOps<mlir::arith::SelectOp>())) {
        if (!dom.properlyDominates(blockOp.getCondition(), op) ||
            !dom.properlyDominates(blockOp.getTrueValue(), op) ||
            !dom.properlyDominates(blockOp.getFalseValue(), op))
          continue;

        rewriter.modifyOpInPlace(blockOp, [&]() { blockOp->moveBefore(op); });
        changed = true;
      }
    }

    if (changed) {
      rewriter.finalizeOpModification(op);
    } else {
      rewriter.cancelOpModification(op);
    }
    return mlir::success(changed);
  }
};

// TODO: upstream?
struct WhileHoistFromBefore
    : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Block &beforeBlock = op.getBefore().front();
    auto beforeTerm =
        mlir::cast<mlir::scf::ConditionOp>(beforeBlock.getTerminator());

    mlir::DominanceInfo dom;
    auto checkArg = [&](mlir::Value arg) -> bool {
      if (dom.properlyDominates(arg, op))
        return true;

      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(arg))
        return blockArg.getParentBlock() == &beforeBlock;

      return false;
    };
    auto canBeHoisted = [&](mlir::Operation &blockOp) -> bool {
      // TODO: hack, need to rework scf.while->scf.for conversion
      if (!mlir::isa_and_present<mlir::arith::ArithDialect>(
              blockOp.getDialect()))
        return false;

      for (auto res : blockOp.getResults()) {
        if (res == beforeTerm.getCondition())
          return false;
      }

      if (beforeBlock.begin() != blockOp.getIterator() &&
          !mlir::isPure(&blockOp))
        return false;

      if (blockOp.getNumRegions() != 0)
        return false;

      return llvm::all_of(blockOp.getOperands(), checkArg);
    };

    mlir::Operation *opToHoist = nullptr;
    for (auto &beforeOp : beforeBlock.without_terminator()) {
      if (!canBeHoisted(beforeOp))
        continue;

      opToHoist = &beforeOp;
      break;
    }

    if (!opToHoist)
      return mlir::failure();

    auto oldInits = op.getInits();
    mlir::IRMapping mapping;
    mapping.map(op.getBeforeArguments(), oldInits);
    auto newResults = rewriter.clone(*opToHoist, mapping)->getResults();

    llvm::SmallVector<mlir::Value> newInits(oldInits.begin(), oldInits.end());
    newInits.append(newResults.begin(), newResults.end());

    llvm::SmallVector<mlir::Location> newLocs(newResults.size(),
                                              rewriter.getUnknownLoc());
    beforeBlock.addArguments(newResults.getTypes(), newLocs);

    mlir::OpBuilder::InsertionGuard g(rewriter);
    auto yield = op.getYieldOp();
    auto yieldArgs = yield.getResults();
    mapping.map(op.getBeforeArguments(), yieldArgs);
    rewriter.setInsertionPoint(yield);
    newResults = rewriter.clone(*opToHoist, mapping)->getResults();
    llvm::SmallVector<mlir::Value> newYieldArgs(yieldArgs.begin(),
                                                yieldArgs.end());
    newYieldArgs.append(newResults.begin(), newResults.end());
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(yield, newYieldArgs);
    rewriter.replaceOp(opToHoist,
                       beforeBlock.getArguments().take_back(newResults.size()));

    rewriter.setInsertionPoint(op);

    mlir::Block &afterBlock = op.getAfter().front();

    mlir::Location loc = op.getLoc();
    auto newWhileOp = rewriter.create<mlir::scf::WhileOp>(
        loc, op.getResultTypes(), newInits,
        /*beforeBody*/ nullptr, /*afterBody*/ nullptr);
    mlir::Block &newBeforeBlock = newWhileOp.getBefore().front();
    mlir::Block &newAfterBlock = newWhileOp.getAfter().front();

    rewriter.mergeBlocks(&beforeBlock, &newBeforeBlock,
                         newBeforeBlock.getArguments());
    rewriter.mergeBlocks(&afterBlock, &newAfterBlock,
                         newAfterBlock.getArguments());

    rewriter.replaceOp(op, newWhileOp.getResults());
    return mlir::success();
  }
};

struct LowerIndexSwitch
    : public mlir::OpRewritePattern<mlir::scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IndexSwitchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cases = op.getCases();
    if (cases.empty())
      return mlir::failure();

    mlir::TypeRange resTypes = op.getResultTypes();
    auto arg = op.getArg();
    auto loc = op.getLoc();

    mlir::ValueRange results;
    mlir::scf::IfOp newIf;
    for (auto &&[caseVal, reg] : llvm::zip(cases, op.getCaseRegions())) {
      bool first = !newIf;
      if (!first) {
        auto elseBlock = rewriter.createBlock(&newIf.getElseRegion());
        rewriter.setInsertionPointToStart(elseBlock);
      }
      mlir::Value cst =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, caseVal);
      mlir::Value cond = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, arg, cst);
      newIf = rewriter.create<mlir::scf::IfOp>(loc, resTypes, cond);
      if (first) {
        results = newIf.getResults();
      } else {
        rewriter.create<mlir::scf::YieldOp>(loc, newIf.getResults());
      }

      newIf.getThenRegion().takeBody(reg);
    }

    assert(newIf);
    newIf.getElseRegion().takeBody(op.getDefaultRegion());

    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct CFGToSCFPass
    : public mlir::PassWrapper<CFGToSCFPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CFGToSCFPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::ub::UBDialect>();
  }

  void runOnOperation() override {
    auto &dom = getAnalysis<mlir::DominanceInfo>();

    mlir::ControlFlowToSCFTransformation transformation;
    auto visitor = [&](mlir::Operation *op) -> mlir::WalkResult {
      for (mlir::Region &reg : op->getRegions()) {
        if (mlir::failed(mlir::transformCFGToSCF(reg, transformation, dom)))
          return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    };

    auto op = getOperation();
    if (op->walk<mlir::WalkOrder::PostOrder>(visitor).wasInterrupted())
      return signalPassFailure();

    auto context = &getContext();

    mlir::RewritePatternSet patterns(context);

    patterns.insert<
        // clang-format off
        WhileReductionSelect,
        WhileUndefArgs,
        WhileMoveToAfter,
        CondBrSameTarget,
        OrOfXor,
        HoistSelects,
        WhileHoistFromBefore,
        LowerIndexSwitch
        // clang-format on
        >(context);

    mlir::scf::ExecuteRegionOp::getCanonicalizationPatterns(patterns, context);
    mlir::scf::IfOp::getCanonicalizationPatterns(patterns, context);
    mlir::scf::IndexSwitchOp::getCanonicalizationPatterns(patterns, context);
    mlir::scf::WhileOp::getCanonicalizationPatterns(patterns, context);
    mlir::arith::SelectOp::getCanonicalizationPatterns(patterns, context);

    numba::populatePoisonOptsPatterns(patterns);

    if (mlir::failed(
            mlir::applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();

    op->walk([&](mlir::Operation *o) -> mlir::WalkResult {
      if (mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp>(o)) {
        o->emitError("Unable to convert CFG to SCF");
        signalPassFailure();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createCFGToSCFPass() {
  return std::make_unique<CFGToSCFPass>();
}
