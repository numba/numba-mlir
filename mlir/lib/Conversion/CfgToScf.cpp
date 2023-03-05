// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/CfgToScf.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

namespace {
static mlir::Block *getNextBlock(mlir::Block *block) {
  assert(nullptr != block);
  if (auto br =
          mlir::dyn_cast_or_null<mlir::cf::BranchOp>(block->getTerminator()))
    return br.getDest();

  return nullptr;
};

static void eraseBlocks(mlir::PatternRewriter &rewriter,
                        llvm::ArrayRef<mlir::Block *> blocks) {
  for (auto block : blocks) {
    assert(nullptr != block);
    block->dropAllDefinedValueUses();
  }
  for (auto block : blocks)
    rewriter.eraseBlock(block);
}

static bool isBlocksDifferent(llvm::ArrayRef<mlir::Block *> blocks) {
  for (auto [i, block1] : llvm::enumerate(blocks)) {
    assert(nullptr != block1);
    for (auto block2 : blocks.drop_front(i + 1)) {
      assert(nullptr != block2);
      if (block1 == block2)
        return false;
    }
  }
  return true;
}

/// Convert
///
///  ```
///    BB1       BB1
///   /   \      |  \
/// BB2  BB3     |  BB2
///   \   /      |  /
///    BB4       BB3
/// ```
///
///  to `scf.if`
///
struct ScfIfRewriteOneExit
    : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getTrueDest() || !op.getFalseDest())
      return mlir::failure();

    auto getDest = [&](bool trueDest) {
      return trueDest ? op.getTrueDest() : op.getFalseDest();
    };
    auto getOperands = [&](bool trueDest) {
      return trueDest ? op.getTrueOperands() : op.getFalseOperands();
    };
    auto loc = op.getLoc();
    auto returnBlock = reinterpret_cast<mlir::Block *>(1); // Fake block
    for (bool reverse : {false, true}) {
      auto trueBlock = getDest(!reverse);
      auto getNextBlock = [&](mlir::Block *block) -> mlir::Block * {
        assert(nullptr != block);
        auto term = block->getTerminator();
        if (auto br = mlir::dyn_cast_or_null<mlir::cf::BranchOp>(term))
          return br.getDest();

        if (auto ret = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(term))
          return returnBlock;

        return nullptr;
      };
      auto postBlock = getNextBlock(trueBlock);
      if (nullptr == postBlock)
        continue;

      auto falseBlock = getDest(reverse);
      if (falseBlock != postBlock && getNextBlock(falseBlock) != postBlock)
        continue;

      auto startBlock = op.getOperation()->getBlock();
      if (!isBlocksDifferent({startBlock, trueBlock, postBlock}))
        continue;

      mlir::Value cond = op.getCondition();
      if (reverse) {
        auto i1 = mlir::IntegerType::get(op.getContext(), 1);
        auto one = rewriter.create<mlir::arith::ConstantOp>(
            loc, mlir::IntegerAttr::get(i1, 1));
        cond = rewriter.create<mlir::arith::XOrIOp>(loc, cond, one);
      }

      mlir::IRMapping mapper;
      llvm::SmallVector<mlir::Value> yieldVals;
      auto copyBlock = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Block &block, mlir::ValueRange args) {
        assert(args.size() == block.getNumArguments());
        mapper.clear();
        mapper.map(block.getArguments(), args);
        for (auto &op : block.without_terminator())
          builder.clone(op, mapper);

        auto operands = [&]() {
          auto term = block.getTerminator();
          if (postBlock == returnBlock) {
            return mlir::cast<mlir::func::ReturnOp>(term).getOperands();
          } else {
            return mlir::cast<mlir::cf::BranchOp>(term).getDestOperands();
          }
        }();
        yieldVals.clear();
        yieldVals.reserve(operands.size());
        for (auto op : operands)
          yieldVals.emplace_back(mapper.lookupOrDefault(op));

        builder.create<mlir::scf::YieldOp>(loc, yieldVals);
      };

      auto trueBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
        copyBlock(builder, loc, *trueBlock, getOperands(!reverse));
      };

      bool hasElse = (falseBlock != postBlock);
      auto resTypes = [&]() {
        auto term = trueBlock->getTerminator();
        if (postBlock == returnBlock) {
          return mlir::cast<mlir::func::ReturnOp>(term)
              .getOperands()
              .getTypes();
        } else {
          return mlir::cast<mlir::cf::BranchOp>(term)
              .getDestOperands()
              .getTypes();
        }
      }();
      mlir::scf::IfOp ifOp;
      if (hasElse) {
        auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
          copyBlock(builder, loc, *falseBlock, getOperands(reverse));
        };
        ifOp = rewriter.create<mlir::scf::IfOp>(loc, cond, trueBody, falseBody);
      } else {
        if (resTypes.empty()) {
          ifOp = rewriter.create<mlir::scf::IfOp>(loc, cond, trueBody);
        } else {
          auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
            auto res = getOperands(reverse);
            yieldVals.clear();
            yieldVals.reserve(res.size());
            for (auto op : res) {
              yieldVals.emplace_back(mapper.lookupOrDefault(op));
            }
            builder.create<mlir::scf::YieldOp>(loc, yieldVals);
          };
          ifOp =
              rewriter.create<mlir::scf::IfOp>(loc, cond, trueBody, falseBody);
        }
      }

      if (postBlock == returnBlock) {
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                          ifOp.getResults());
      } else {
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, postBlock,
                                                        ifOp.getResults());
      }

      if (trueBlock->use_empty())
        eraseBlocks(rewriter, trueBlock);

      if (falseBlock->use_empty())
        eraseBlocks(rewriter, falseBlock);

      return mlir::success();
    }
    return mlir::failure();
  }
};

/// Convert
///
/// ```
///    BB1
///    / |
/// BB2  |
///  | \ |
///  |  \|
/// BB3 BB4
/// ```
///
/// To
///
/// ```
///    |
/// scf.if
///    |
///   BB1
///  /   \
/// BB3 BB4
/// ```
///
/// To open more opportunities for `scf.while` conversion
struct ScfIfRewriteTwoExits
    : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert(op.getTrueDest());
    assert(op.getFalseDest());

    auto thisBlock = op->getBlock();
    for (bool reverse : {false, true}) {
      auto thenBlock = reverse ? op.getFalseDest() : op.getTrueDest();
      auto exitBlock = reverse ? op.getTrueDest() : op.getFalseDest();
      auto exitOps = (reverse ? op.getTrueOperands() : op.getFalseOperands());
      if (thenBlock == thisBlock || exitBlock == thisBlock)
        continue;

      auto thenBr =
          mlir::dyn_cast<mlir::cf::CondBranchOp>(thenBlock->getTerminator());
      if (!thenBr)
        continue;

      auto exitBlock1 = thenBr.getTrueDest();
      auto exitBlock2 = thenBr.getFalseDest();
      auto ops1 = thenBr.getTrueOperands();
      auto ops2 = thenBr.getFalseOperands();
      bool reverseExitCond = false;
      if (exitBlock2 == exitBlock) {
        // nothing
      } else if (exitBlock1 == exitBlock) {
        std::swap(exitBlock1, exitBlock2);
        std::swap(ops1, ops2);
        reverseExitCond = true;
      } else {
        continue;
      }

      if (exitBlock1 == thenBlock)
        continue;

      if (exitBlock1->getNumArguments() != 0)
        continue;

      if (thenBlock->getNumArguments() != 0)
        continue;

      llvm::SmallVector<mlir::Value> thenValsUsers;
      for (auto &op : thenBlock->without_terminator())
        for (auto res : op.getResults())
          if (res.isUsedOutsideOfBlock(thenBlock))
            thenValsUsers.emplace_back(res);

      auto trueBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
        mlir::IRMapping mapper;
        for (auto &op : thenBlock->without_terminator())
          builder.clone(op, mapper);

        auto cond = mapper.lookupOrDefault(thenBr.getCondition());
        if (reverseExitCond) {
          auto one =
              builder.create<mlir::arith::ConstantIntOp>(loc, /*value*/ 1,
                                                         /*width*/ 1);
          cond = builder.create<mlir::arith::XOrIOp>(loc, one, cond);
        }

        llvm::SmallVector<mlir::Value> ret;
        ret.emplace_back(cond);
        for (auto op : ops2)
          ret.emplace_back(mapper.lookupOrDefault(op));

        for (auto user : thenValsUsers)
          ret.emplace_back(mapper.lookupOrDefault(user));

        builder.create<mlir::scf::YieldOp>(loc, ret);
      };

      auto falseBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
        mlir::Value cond = rewriter.create<mlir::arith::ConstantIntOp>(
            loc, /*value*/ 0, /*width*/ 1);
        llvm::SmallVector<mlir::Value> ret;
        ret.emplace_back(cond);
        llvm::copy(exitOps, std::back_inserter(ret));
        for (auto user : thenValsUsers) {
          auto val = builder.create<numba::util::UndefOp>(loc, user.getType());
          ret.emplace_back(val);
        }
        builder.create<mlir::scf::YieldOp>(loc, ret);
      };

      mlir::Value cond = op.getCondition();
      auto loc = op.getLoc();
      if (reverse) {
        auto one = rewriter.create<mlir::arith::ConstantIntOp>(loc, /*value*/ 1,
                                                               /*width*/ 1);
        cond = rewriter.create<mlir::arith::XOrIOp>(loc, one, cond);
      }

      auto ifRetType = rewriter.getIntegerType(1);

      llvm::SmallVector<mlir::Type> retTypes;
      retTypes.emplace_back(ifRetType);
      llvm::copy(exitOps.getTypes(), std::back_inserter(retTypes));
      for (auto user : thenValsUsers)
        retTypes.emplace_back(user.getType());

      auto ifResults =
          rewriter.create<mlir::scf::IfOp>(loc, cond, trueBuilder, falseBuilder)
              .getResults();
      cond = rewriter.create<mlir::arith::AndIOp>(loc, cond, ifResults[0]);
      ifResults = ifResults.drop_front();

      assert(exitBlock1->getNumArguments() == ops1.size());
      assert(exitBlock2->getNumArguments() == exitOps.size());
      rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
          op, cond, exitBlock1, ops1, exitBlock2,
          ifResults.take_front(exitOps.size()));
      for (auto it : llvm::zip(thenValsUsers,
                               ifResults.take_back(thenValsUsers.size()))) {
        auto oldUser = std::get<0>(it);
        auto newUser = std::get<1>(it);
        for (auto &use : llvm::make_early_inc_range(oldUser.getUses())) {
          auto *owner = use.getOwner();
          rewriter.updateRootInPlace(owner, [&]() { use.set(newUser); });
        }
      }
      return mlir::success();
    }
    return mlir::failure();
  }
};

static mlir::scf::WhileOp
createWhile(mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange iterArgs,
            llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                    mlir::ValueRange)>
                beforeBuilder,
            llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                    mlir::ValueRange)>
                afterBuilder) {
  mlir::OperationState state(loc, mlir::scf::WhileOp::getOperationName());
  state.addOperands(iterArgs);

  {
    mlir::OpBuilder::InsertionGuard g(builder);
    auto addRegion = [&](mlir::ValueRange args) -> mlir::Block * {
      auto reg = state.addRegion();
      auto block = builder.createBlock(reg);
      auto loc = builder.getUnknownLoc();
      for (auto arg : args)
        block->addArgument(arg.getType(), loc);

      return block;
    };

    auto beforeBlock = addRegion(iterArgs);
    beforeBuilder(builder, state.location, beforeBlock->getArguments());
    auto cond =
        mlir::cast<mlir::scf::ConditionOp>(beforeBlock->getTerminator());
    state.addTypes(cond.getArgs().getTypes());

    auto afterblock = addRegion(cond.getArgs());
    afterBuilder(builder, state.location, afterblock->getArguments());
  }
  return mlir::cast<mlir::scf::WhileOp>(builder.create(state));
}

static bool isInsideBlock(mlir::Operation *op, mlir::Block *block) {
  assert(nullptr != op);
  assert(nullptr != block);
  do {
    if (op->getBlock() == block)
      return true;
  } while ((op = op->getParentOp()));
  return false;
}

/// Convert
/// ```
///  BB1
///   |
///  BB2
/// / | \
/// | V ^
/// | | /
/// | BB3
/// |
/// BB4
/// ```
/// To `scf.while`
struct ScfWhileRewrite : public mlir::OpRewritePattern<mlir::cf::BranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto beforeBlock = op.getDest();
    auto beforeTerm =
        mlir::dyn_cast<mlir::cf::CondBranchOp>(beforeBlock->getTerminator());
    if (!beforeTerm)
      return mlir::failure();

    mlir::DominanceInfo dom;
    auto startBlock = op.getOperation()->getBlock();
    for (bool reverse : {false, true}) {
      auto afterBlock =
          reverse ? beforeTerm.getFalseDest() : beforeTerm.getTrueDest();
      auto postBlock =
          reverse ? beforeTerm.getTrueDest() : beforeTerm.getFalseDest();
      auto falseArgs = reverse ? beforeTerm.getTrueDestOperands()
                               : beforeTerm.getFalseDestOperands();
      if (getNextBlock(afterBlock) != beforeBlock ||
          !isBlocksDifferent({startBlock, beforeBlock, afterBlock, postBlock}))
        continue;

      auto checkOutsideVals = [&](mlir::Operation *op) -> mlir::WalkResult {
        for (auto user : op->getUsers())
          if (!isInsideBlock(user, beforeBlock) &&
              !isInsideBlock(user, afterBlock))
            return mlir::WalkResult::interrupt();

        return mlir::WalkResult::advance();
      };

      if (afterBlock->walk(checkOutsideVals).wasInterrupted())
        continue;

      mlir::IRMapping mapper;
      llvm::SmallVector<mlir::Value> yieldVars;
      auto beforeBlockArgs = beforeBlock->getArguments();
      llvm::SmallVector<mlir::Value> origVars(beforeBlockArgs.begin(),
                                              beforeBlockArgs.end());

      auto beforeBody = [&](mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::ValueRange iterargs) {
        mapper.map(beforeBlockArgs, iterargs);
        yieldVars.resize(beforeBlockArgs.size());
        for (auto &op : beforeBlock->without_terminator()) {
          auto newOp = builder.clone(op, mapper);
          for (auto user : op.getUsers()) {
            if (!isInsideBlock(user, beforeBlock)) {
              for (auto [orig, yield] :
                   llvm::zip(op.getResults(), newOp->getResults())) {
                origVars.emplace_back(orig);
                yieldVars.emplace_back(yield);
              }
              break;
            }
          }
        }

        llvm::transform(
            beforeBlockArgs, yieldVars.begin(),
            [&](mlir::Value val) { return mapper.lookupOrDefault(val); });

        for (auto arg : falseArgs) {
          origVars.emplace_back(arg);
          yieldVars.emplace_back(mapper.lookupOrDefault(arg));
        }

        auto cond = mapper.lookupOrDefault(beforeTerm.getCondition());
        if (reverse) {
          auto condVal = rewriter.getIntegerAttr(cond.getType(), 1);
          auto one = rewriter.create<mlir::arith::ConstantOp>(loc, condVal);
          cond = rewriter.create<mlir::arith::XOrIOp>(loc, one, cond);
        }
        builder.create<mlir::scf::ConditionOp>(loc, cond, yieldVars);
      };
      auto afterBody = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange iterargs) {
        mapper.clear();
        assert(origVars.size() == iterargs.size());
        mapper.map(origVars, iterargs);
        auto afterArgs = afterBlock->getArguments();
        mapper.map(afterArgs, iterargs.take_back(afterArgs.size()));
        for (auto &op : afterBlock->without_terminator())
          builder.clone(op, mapper);

        yieldVars.clear();
        auto term = mlir::cast<mlir::cf::BranchOp>(afterBlock->getTerminator());
        for (auto arg : term.getOperands())
          yieldVars.emplace_back(mapper.lookupOrDefault(arg));

        builder.create<mlir::scf::YieldOp>(loc, yieldVars);
      };

      auto whileOp = createWhile(rewriter, op.getLoc(), op.getOperands(),
                                 beforeBody, afterBody);

      assert(origVars.size() == whileOp.getNumResults());
      for (auto arg : llvm::zip(origVars, whileOp.getResults())) {
        auto origVal = std::get<0>(arg);
        for (auto &use : llvm::make_early_inc_range(origVal.getUses())) {
          auto *owner = use.getOwner();
          auto *block = owner->getBlock();
          if (block != &whileOp.getBefore().front() &&
              block != &whileOp.getAfter().front()) {
            auto newVal = std::get<1>(arg);
            if (dom.properlyDominates(newVal, owner))
              rewriter.updateRootInPlace(owner, [&]() { use.set(newVal); });
          }
        }
      }

      auto results = whileOp.getResults().take_back(falseArgs.size());
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, postBlock, results);
      return mlir::success();
    }
    return mlir::failure();
  }
};

namespace {
struct SCC {
  struct Node {
    llvm::SmallVector<mlir::Block *, 4> blocks;
  };
  llvm::SmallVector<Node> nodes;

  void dump() const {
    for (auto [i, node] : llvm::enumerate(nodes)) {
      llvm::errs() << "scc node " << i << "\n";
      for (auto b : node.blocks)
        b->dump();
    }
  }
};

struct BlockDesc {
  enum { UndefinedIndex = -1 };
  int index = UndefinedIndex;
  int lowLink = UndefinedIndex;
  bool onStack = false;
};

using Edge = std::pair<mlir::Block *, mlir::Block *>;
} // namespace

static void strongconnect(mlir::Block *block,
                          llvm::SmallDenseMap<mlir::Block *, BlockDesc> &blocks,
                          llvm::SmallVectorImpl<mlir::Block *> &stack,
                          int &index, SCC &scc) {
  assert(block);
  auto &desc = blocks[block];
  if (desc.index != BlockDesc::UndefinedIndex)
    return;

  desc.index = index;
  desc.lowLink = index;
  ++index;

  desc.onStack = true;
  stack.push_back(block);

  auto region = block->getParent();
  for (mlir::Block *successor : block->getSuccessors()) {
    if (region != successor->getParent())
      continue;

    auto &successorDesc = blocks[successor];
    if (successorDesc.index == BlockDesc::UndefinedIndex) {
      strongconnect(successor, blocks, stack, index, scc);

      // Do not use cached values as underlying map may have been reallocated.
      auto &successorDesc1 = blocks[successor];
      auto &desc1 = blocks[block];
      desc1.lowLink = std::min(desc1.lowLink, successorDesc1.lowLink);
    } else if (successorDesc.onStack) {
      // Do not use cached values as underlying map may have been reallocated.
      auto &successorDesc1 = blocks[successor];
      auto &desc1 = blocks[block];
      desc1.lowLink = std::min(desc1.lowLink, successorDesc1.index);
    }
  }

  auto &desc1 = blocks[block];
  if (desc1.lowLink != desc1.index)
    return;

  auto &sccNode = scc.nodes.emplace_back();
  mlir::Block *currentBlock = nullptr;
  do {
    assert(!stack.empty());
    currentBlock = stack.pop_back_val();
    blocks[currentBlock].onStack = false;
    sccNode.blocks.emplace_back(currentBlock);
  } while (currentBlock != block);
}

/// SCC construction algorithm from
/// https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
static std::optional<SCC> buildSCC(mlir::Region &region) {
  SCC scc;

  llvm::SmallDenseMap<mlir::Block *, BlockDesc> blocks;
  llvm::SmallVector<mlir::Block *> stack;
  int index = 0;
  for (auto &block : region)
    strongconnect(&block, blocks, stack, index, scc);

  return scc;
}

static mlir::ValueRange getTerminatorArgs(mlir::Operation *term,
                                          mlir::Block *target) {
  assert(term);
  assert(target);
  if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(term))
    return br.getDestOperands();

  if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(term))
    return target == condBr.getTrueDest() ? condBr.getTrueDestOperands()
                                          : condBr.getFalseDestOperands();

  llvm_unreachable("getTerminatorArgs: unsupported terminator");
}

static mlir::ValueRange getEdgeArgs(Edge edge) {
  auto term = edge.first->getTerminator();
  return getTerminatorArgs(term, edge.second);
}

static void replaceEdgeDest(mlir::PatternRewriter &rewriter, Edge edge,
                            mlir::Block *newDest, mlir::ValueRange newArgs) {
  auto term = edge.first->getTerminator();
  mlir::OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(term);
  if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(term)) {
    assert(edge.second == br.getDest());
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(br, newDest, newArgs);
    return;
  }

  if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(term)) {
    mlir::Block *trueDest = condBr.getTrueDest();
    mlir::ValueRange trueArgs = condBr.getTrueDestOperands();
    mlir::Block *falseDest = condBr.getFalseDest();
    mlir::ValueRange falseArgs = condBr.getFalseDestOperands();
    if (edge.second == trueDest) {
      trueDest = newDest;
      trueArgs = newArgs;
    }
    if (edge.second == falseDest) {
      falseDest = newDest;
      falseArgs = newArgs;
    }

    auto cond = condBr.getCondition();
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        condBr, cond, trueDest, trueArgs, falseDest, falseArgs);
    return;
  }

  llvm_unreachable("replaceEdgeDest: unsupported terminator");
}

static void addTypesFromEdges(llvm::ArrayRef<Edge> edges,
                              llvm::SmallVectorImpl<mlir::Type> &ret) {
  for (auto edge : edges) {
    auto edgeArgs = edge.second->getArgumentTypes();
    ret.append(edgeArgs.begin(), edgeArgs.end());
  }
}

static void generateMultiplexedBranches(mlir::PatternRewriter &rewriter,
                                        mlir::Location loc,
                                        mlir::Block *srcBlock,
                                        mlir::ValueRange multiplexArgs,
                                        mlir::ValueRange srcArgs,
                                        llvm::ArrayRef<Edge> edges) {
  assert(srcBlock);
  mlir::OpBuilder::InsertionGuard g(rewriter);
  mlir::Block *currentBlock = srcBlock;
  auto region = srcBlock->getParent();
  auto numMultiplexVars = edges.size() - 1;
  assert(multiplexArgs.size() == numMultiplexVars);
  for (auto [i, edge] : llvm::enumerate(edges.drop_back())) {
    auto dst = edge.second;
    auto numArgs = dst->getNumArguments();
    auto args = srcArgs.take_front(numArgs);
    rewriter.setInsertionPointToStart(currentBlock);
    auto cond = multiplexArgs[i];
    if (i == numMultiplexVars - 1) {
      auto lastEdge = edges.back();
      auto lastDst = lastEdge.second;
      auto falseArgs = srcArgs.drop_front(numArgs);
      assert(falseArgs.size() == lastDst->getNumArguments());
      rewriter.create<mlir::cf::CondBranchOp>(loc, cond, dst, args, lastDst,
                                              falseArgs);
    } else {
      mlir::Block *nextBlock = rewriter.createBlock(region);
      rewriter.create<mlir::cf::CondBranchOp>(loc, cond, dst, args, nextBlock,
                                              mlir::ValueRange{});
      currentBlock = nextBlock;
    }
    srcArgs = srcArgs.drop_front(numArgs);
  }
}

static void initMultiplexConds(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, size_t currentBlock,
                               size_t numBlocks,
                               llvm::SmallVectorImpl<mlir::Value> &res) {
  assert(numBlocks > 0);
  assert(currentBlock < numBlocks);
  auto boolType = rewriter.getI1Type();
  for (auto j : llvm::seq<size_t>(0, numBlocks - 1)) {
    auto val = static_cast<int64_t>(j == currentBlock);
    mlir::Value cond =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, val, boolType);
    res.emplace_back(cond);
  }
}

static void initMultiplexVars(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, size_t currentBlock,
                              llvm::ArrayRef<Edge> edges,
                              llvm::SmallVectorImpl<mlir::Value> &res) {
  assert(currentBlock < edges.size());
  for (auto [j, edge] : llvm::enumerate(edges)) {
    auto args = getEdgeArgs(edge);
    if (j == currentBlock) {
      res.append(args.begin(), args.end());
    } else {
      for (auto arg : args) {
        auto type = arg.getType();
        mlir::Value init = rewriter.create<numba::util::UndefOp>(loc, type);
        res.emplace_back(init);
      }
    }
  }
}

/// Restructure loop into tail-controlled form according to algorithm described
/// in https://dl.acm.org/doi/pdf/10.1145/2693261
///
/// Returns true if any modifications to IR were made.
static bool restructureLoop(mlir::PatternRewriter &rewriter, SCC::Node &node) {
  assert(!node.blocks.empty());

  if (node.blocks.size() == 1)
    return false;

  auto &blocks = node.blocks;
  auto region = blocks.front()->getParent();

  llvm::SmallDenseSet<mlir::Block *> blocksSet(blocks.begin(), blocks.end());

  auto isInSCC = [&](mlir::Block *block) {
    assert(block);
    return blocksSet.count(block) != 0;
  };

  llvm::SmallVector<Edge> inEdges;
  llvm::SmallVector<Edge> outEdges;
  llvm::SmallVector<Edge> repetitionEdges;

  for (auto block : blocks) {
    bool isInput = false;
    for (auto predecessor : block->getPredecessors()) {
      if (predecessor->getParent() != region)
        continue;

      if (!isInSCC(predecessor)) {
        inEdges.emplace_back(predecessor, block);
        isInput = true;
      }
    }

    for (auto succesor : block->getSuccessors()) {
      if (succesor->getParent() != region)
        continue;

      if (!isInSCC(succesor))
        outEdges.emplace_back(block, succesor);
    }

    if (isInput) {
      for (auto predecessor : block->getPredecessors()) {
        if (predecessor->getParent() != region)
          continue;

        if (isInSCC(predecessor)) {
          repetitionEdges.emplace_back(predecessor, block);
          isInput = true;
        }
      }
    }
  }

  // Check if we are already in structured form.
  if (inEdges.size() == 1 && outEdges.size() == 1) {
    if (outEdges.front().first->getNumSuccessors() == 2) {
      auto inBlock = inEdges.front().second;
      auto successors = outEdges.front().first->getSuccessors();
      if (successors[0] == inBlock || successors[1] == inBlock)
        return false;
    }
  }

  auto boolType = rewriter.getI1Type();
  auto numInMultiplexVars = inEdges.size() - 1;
  mlir::Block *multiplexEntryBlock = nullptr;
  auto loc = rewriter.getUnknownLoc();
  auto createBlock = [&](mlir::TypeRange types =
                             std::nullopt) -> mlir::Block * {
    llvm::SmallVector<mlir::Location> locs(types.size(), loc);
    return rewriter.createBlock(region, {}, types, locs);
  };

  {
    llvm::SmallVector<mlir::Type> entryBlockTypes(numInMultiplexVars, boolType);
    addTypesFromEdges(inEdges, entryBlockTypes);
    multiplexEntryBlock = createBlock(entryBlockTypes);
    mlir::ValueRange blockArgs = multiplexEntryBlock->getArguments();
    generateMultiplexedBranches(rewriter, loc, multiplexEntryBlock,
                                blockArgs.take_front(numInMultiplexVars),
                                blockArgs.drop_front(numInMultiplexVars),
                                inEdges);
  }

  llvm::SmallVector<mlir::Value> branchArgs;
  for (auto [i, inEdge] : llvm::enumerate(inEdges)) {
    auto entryBlock = createBlock();
    rewriter.setInsertionPointToStart(entryBlock);
    branchArgs.clear();
    initMultiplexConds(rewriter, loc, i, inEdges.size(), branchArgs);
    initMultiplexVars(rewriter, loc, i, inEdges, branchArgs);

    rewriter.create<mlir::cf::BranchOp>(loc, multiplexEntryBlock, branchArgs);
    replaceEdgeDest(rewriter, inEdge, entryBlock, {});
  }

  mlir::ValueRange repMultiplexVars;
  mlir::ValueRange exitArgs;
  mlir::Block *exitBlock = nullptr;
  auto numOutMultiplexVars = outEdges.size() - 1;
  {
    llvm::SmallVector<mlir::Type> repBlockTypes(numOutMultiplexVars + 1,
                                                boolType);
    addTypesFromEdges(repetitionEdges, repBlockTypes);
    auto numRepArgs = repBlockTypes.size() - numOutMultiplexVars - 1;

    addTypesFromEdges(outEdges, repBlockTypes);

    auto repBlock = createBlock(repBlockTypes);
    exitBlock = createBlock();
    rewriter.setInsertionPointToStart(repBlock);
    mlir::Value cond = repBlock->getArgument(0);
    auto repBlockArgs =
        repBlock->getArguments().drop_front(numOutMultiplexVars + 1);
    repMultiplexVars =
        repBlock->getArguments().drop_front().take_front(numOutMultiplexVars);

    mlir::ValueRange repetitionArgs = repBlockArgs.take_front(numRepArgs);
    exitArgs = repBlockArgs.drop_front(numRepArgs);
    assert(multiplexEntryBlock->getNumArguments() == repetitionArgs.size());
    rewriter.create<mlir::cf::CondBranchOp>(loc, cond, multiplexEntryBlock,
                                            repetitionArgs, exitBlock,
                                            mlir::ValueRange{});

    llvm::SmallVector<mlir::Value> branchArgs;
    for (auto [i, repEdge] : llvm::enumerate(repetitionEdges)) {
      auto preRepBlock = createBlock();
      rewriter.setInsertionPointToStart(preRepBlock);
      mlir::Value trueVal =
          rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, boolType);
      mlir::Value undefVal =
          rewriter.create<numba::util::UndefOp>(loc, boolType);

      branchArgs.clear();
      branchArgs.emplace_back(trueVal);
      for (auto j : llvm::seq<size_t>(0, numOutMultiplexVars)) {
        (void)j;
        branchArgs.emplace_back(undefVal);
      }

      initMultiplexVars(rewriter, loc, i, repetitionEdges, branchArgs);

      for (auto type : llvm::ArrayRef(repBlockTypes)
                           .drop_front(numRepArgs + numOutMultiplexVars + 1)) {
        mlir::Value val = rewriter.create<numba::util::UndefOp>(loc, type);
        branchArgs.emplace_back(val);
      }
      assert(branchArgs.size() == repBlock->getNumArguments());
      rewriter.create<mlir::cf::BranchOp>(loc, repBlock, branchArgs);
      replaceEdgeDest(rewriter, repEdge, preRepBlock, {});
    }

    for (auto [i, outEdge] : llvm::enumerate(outEdges)) {
      auto preRepBlock = createBlock();
      rewriter.setInsertionPointToStart(preRepBlock);
      mlir::Value falseVal =
          rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, boolType);

      branchArgs.clear();
      branchArgs.emplace_back(falseVal);
      initMultiplexConds(rewriter, loc, i, outEdges.size(), branchArgs);

      for (auto type :
           llvm::ArrayRef(repBlockTypes).drop_front(1).take_front(numRepArgs)) {
        mlir::Value val = rewriter.create<numba::util::UndefOp>(loc, type);
        branchArgs.emplace_back(val);
      }

      initMultiplexVars(rewriter, loc, i, outEdges, branchArgs);

      assert(branchArgs.size() == repBlock->getNumArguments());
      rewriter.create<mlir::cf::BranchOp>(loc, repBlock, branchArgs);
      replaceEdgeDest(rewriter, outEdge, preRepBlock, {});
    }
  }

  generateMultiplexedBranches(rewriter, loc, exitBlock, repMultiplexVars,
                              exitArgs, outEdges);

  return true;
}

static bool isEntryBlock(mlir::Block &block) {
  auto region = block.getParent();
  return &(region->front()) == &block;
}

struct LoopRestructuringBr : public mlir::OpRewritePattern<mlir::cf::BranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto block = op->getBlock();
    if (!isEntryBlock(*block))
      return mlir::failure();

    llvm::errs() << "Build scc\n";
    auto scc = buildSCC(*block->getParent());
    if (!scc)
      return mlir::failure();

    scc->dump();

    bool changed = false;
    for (auto &node : scc->nodes)
      changed = restructureLoop(rewriter, node) || changed;

    return mlir::success(changed);
  }
};

/// Changes conditional branch on the end of loop body block to unconditiona to
/// open opportunities for scf.while rewrites.
///
/// ```
/// func.func @test() {
///   "test.test1"() : () -> ()
///   cf.br ^bb1
/// ^bb1:
///   %cond = "test.test2"() : () -> i1
///   cf.cond_br %cond, ^bb3, ^bb2
/// ^bb2:
///   %cond2 = "test.test3"() : () -> i1
///   cf.cond_br %cond2, ^bb3, ^bb1
/// ^bb3:
///   "test.test4"() : () -> ()
///   return
/// }
/// ```
///
/// Tranformed into
///
/// ```
/// func.func @test() {
///   "test.test1"() : () -> ()
///   %true = arith.constant true
///   cf.br ^bb1(%true : i1)
/// ^bb1(%0: i1):  // 2 preds: ^bb0, ^bb2
///   %1 = "test.test2"() : () -> i1
///   %2 = arith.andi %0, %1 : i1
///   cf.cond_br %2, ^bb3, ^bb2
/// ^bb2:  // pred: ^bb1
///   %3 = "test.test3"() : () -> i1
///   %4 = arith.xori %true, %3 : i1
///   cf.br ^bb1(%4 : i1)
/// ^bb3:  // pred: ^bb1
///   "test.test4"() : () -> ()
///   return
/// }
/// ```
struct BreakRewrite : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    for (bool reverse : {false, true}) {
      auto bodyBlock = op->getBlock();
      auto exitBlock = reverse ? op.getFalseDest() : op.getTrueDest();
      auto conditionBlock = reverse ? op.getTrueDest() : op.getFalseDest();
      assert(exitBlock);
      assert(conditionBlock);
      if (conditionBlock == bodyBlock)
        continue;

      auto conditionBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(
          conditionBlock->getTerminator());
      if (!conditionBr)
        continue;

      mlir::ValueRange bodyArgs = conditionBr.getTrueDestOperands();
      mlir::ValueRange exitArgs = conditionBr.getFalseDestOperands();
      if (conditionBr.getTrueDest() == bodyBlock &&
          conditionBr.getFalseDest() == exitBlock) {
        // Nothing
      } else if (conditionBr.getTrueDest() == exitBlock &&
                 conditionBr.getFalseDest() == bodyBlock) {
        std::swap(exitBlock, bodyBlock);
        //        std::swap(exitArgs, bodyArgs);
      } else {
        continue;
      }

      auto check = [&]() {
        for (auto user :
             llvm::make_early_inc_range(conditionBlock->getUsers())) {
          if (user == op)
            continue;

          if (mlir::isa<mlir::cf::CondBranchOp>(user))
            return false;
        }
        return true;
      }();
      if (!check)
        continue;

      auto loc = rewriter.getUnknownLoc();

      auto type = rewriter.getIntegerType(1);
      auto condVal = rewriter.getIntegerAttr(type, 1);

      conditionBlock->addArgument(op.getCondition().getType(),
                                  rewriter.getUnknownLoc());
      mlir::OpBuilder::InsertionGuard g(rewriter);
      for (auto user : llvm::make_early_inc_range(conditionBlock->getUsers())) {
        if (user != op) {
          rewriter.setInsertionPoint(user);
          auto condConst =
              rewriter.create<mlir::arith::ConstantOp>(loc, condVal);
          if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(user)) {
            llvm::SmallVector<mlir::Value> params(br.getDestOperands());
            params.emplace_back(condConst);
            rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(br, conditionBlock,
                                                            params);
          } else if (auto condBr =
                         mlir::dyn_cast<mlir::cf::CondBranchOp>(user)) {
            llvm_unreachable("not implemented");
          } else {
            llvm_unreachable("Unknown terminator type");
          }
        }
      }

      rewriter.setInsertionPoint(op);
      llvm::SmallVector<mlir::Value> params(op.getFalseOperands());
      auto one = rewriter.create<mlir::arith::ConstantOp>(loc, condVal);
      mlir::Value cond = op.getCondition();
      if (!reverse)
        cond = rewriter.create<mlir::arith::XOrIOp>(loc, one, cond);

      params.push_back(cond);

      assert(conditionBlock->getNumArguments() == params.size());
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, conditionBlock,
                                                      params);

      rewriter.setInsertionPoint(conditionBr);
      auto oldCond = conditionBr.getCondition();
      mlir::Value newCond = conditionBlock->getArguments().back();
      newCond = rewriter.create<mlir::arith::AndIOp>(loc, newCond, oldCond);

      assert(bodyBlock->getNumArguments() == bodyArgs.size());
      assert(exitBlock->getNumArguments() == exitArgs.size());
      rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
          conditionBr, newCond, bodyBlock, bodyArgs, exitBlock, exitArgs);
      return mlir::success();
    }
    return mlir::failure();
  }
};

/// Convert
/// ```
/// cf.cond_br %cond, ^bb1(%1: index), ^bb1(%2: index)
/// ```
/// to
/// ```
/// %3 = arith.select %cond, %1, %2 : index
/// cf.br ^bb1(%3: index)
/// ```
struct CondBranchSameTargetRewrite
    : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  // Set higher benefit than if rewrites
  CondBranchSameTargetRewrite(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::cf::CondBranchOp>(context,
                                                       /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto trueDest = op.getTrueDest();
    assert(trueDest);
    auto falseDest = op.getFalseDest();
    assert(falseDest);
    if (trueDest != falseDest)
      return mlir::failure();

    assert(op.getTrueOperands().size() == op.getFalseOperands().size());

    auto loc = op.getLoc();
    auto condition = op.getCondition();
    auto count = static_cast<unsigned>(op.getTrueOperands().size());
    llvm::SmallVector<mlir::Value> newOperands(count);
    for (auto i : llvm::seq(0u, count)) {
      auto trueArg = op.getTrueOperand(i);
      assert(trueArg);
      auto falseArg = op.getFalseOperand(i);
      assert(falseArg);
      if (trueArg == falseArg) {
        newOperands[i] = trueArg;
      } else {
        newOperands[i] = rewriter.create<mlir::arith::SelectOp>(
            loc, condition, trueArg, falseArg);
      }
    }

    assert(trueDest->getNumArguments() == newOperands.size());
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, trueDest, newOperands);
    return mlir::success();
  }
};

struct CFGToSCFPass
    : public mlir::PassWrapper<CFGToSCFPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CFGToSCFPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto context = &getContext();

    mlir::RewritePatternSet patterns(context);

    patterns.insert<
        // clang-format off
//        BreakRewrite,
        ScfIfRewriteOneExit,
        LoopRestructuringBr
//        ScfIfRewriteTwoExits,
//        ScfWhileRewrite,
//        CondBranchSameTargetRewrite
        // clang-format on
        >(context);

    context->getLoadedDialect<mlir::cf::ControlFlowDialect>()
        ->getCanonicalizationPatterns(patterns);
    mlir::cf::BranchOp::getCanonicalizationPatterns(patterns, context);
    mlir::cf::CondBranchOp::getCanonicalizationPatterns(patterns, context);

    mlir::scf::ExecuteRegionOp::getCanonicalizationPatterns(patterns, context);

    auto op = getOperation();
    (void)mlir::applyPatternsAndFoldGreedily(op, std::move(patterns));

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
