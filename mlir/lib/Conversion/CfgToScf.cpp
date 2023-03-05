// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/CfgToScf.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
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

      if (trueBlock == falseBlock)
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
static SCC buildSCC(mlir::Region &region) {
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
  assert(!edges.empty());
  mlir::OpBuilder::InsertionGuard g(rewriter);
  auto region = srcBlock->getParent();
  auto numMultiplexVars = edges.size() - 1;
  assert(multiplexArgs.size() == numMultiplexVars);
  if (edges.size() == 1) {
    rewriter.setInsertionPointToEnd(srcBlock);
    auto dst = edges.front().second;
    rewriter.create<mlir::cf::BranchOp>(loc, dst, srcArgs);
    return;
  }

  mlir::Block *currentBlock = srcBlock;
  for (auto [i, edge] : llvm::enumerate(edges.drop_back())) {
    auto dst = edge.second;
    auto numArgs = dst->getNumArguments();
    auto args = srcArgs.take_front(numArgs);
    rewriter.setInsertionPointToEnd(currentBlock);
    auto cond = multiplexArgs[i];
    if (i == numMultiplexVars - 1) {
      auto lastEdge = edges.back();
      auto lastDst = lastEdge.second;
      auto falseArgs = srcArgs.drop_front(numArgs);
      assert(falseArgs.size() == lastDst->getNumArguments());
      rewriter.create<mlir::cf::CondBranchOp>(loc, cond, dst, args, lastDst,
                                              falseArgs);
    } else {
      auto nextBlock = [&]() -> mlir::Block * {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        return rewriter.createBlock(region);
      }();
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
  auto trueVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, boolType);
  auto falseVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, boolType);
  for (auto j : llvm::seq<size_t>(0, numBlocks - 1)) {
    auto val = (j == currentBlock ? trueVal : falseVal);
    res.emplace_back(val);
  }
}

static void initUndefMultiplexConds(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, size_t numBlocks,
                                    llvm::SmallVectorImpl<mlir::Value> &res) {
  assert(numBlocks > 0);
  auto boolType = rewriter.getI1Type();
  auto undefVal = rewriter.create<numba::util::UndefOp>(loc, boolType);
  for (auto j : llvm::seq<size_t>(0, numBlocks - 1)) {
    (void)j;
    res.emplace_back(undefVal);
  }
}

static void initMultiplexVars(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, size_t currentBlock,
                              llvm::ArrayRef<Edge> edges,
                              llvm::SmallVectorImpl<mlir::Value> &res) {
  assert(currentBlock < edges.size());
  for (auto [j, edge] : llvm::enumerate(edges)) {
    mlir::ValueRange args = getEdgeArgs(edge);
    if (j == currentBlock) {
      res.append(args.begin(), args.end());
    } else {
      for (auto type : args.getTypes()) {
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

    mlir::Value cond = repBlock->getArgument(0);
    auto repBlockArgs =
        repBlock->getArguments().drop_front(numOutMultiplexVars + 1);
    repMultiplexVars =
        repBlock->getArguments().drop_front().take_front(numOutMultiplexVars);

    mlir::ValueRange repetitionArgs = repBlockArgs.take_front(numRepArgs);
    exitArgs = repBlockArgs.drop_front(numRepArgs);
    assert(multiplexEntryBlock->getNumArguments() == repetitionArgs.size());
    rewriter.setInsertionPointToStart(repBlock);
    rewriter.create<mlir::cf::CondBranchOp>(loc, cond, multiplexEntryBlock,
                                            repetitionArgs, exitBlock,
                                            mlir::ValueRange{});

    llvm::SmallVector<mlir::Value> branchArgs;
    for (auto [i, repEdge] : llvm::enumerate(repetitionEdges)) {
      auto preRepBlock = createBlock();
      rewriter.setInsertionPointToStart(preRepBlock);
      mlir::Value trueVal =
          rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, boolType);

      branchArgs.clear();
      branchArgs.emplace_back(trueVal);

      initUndefMultiplexConds(rewriter, loc, outEdges.size(), branchArgs);
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

      for (auto type : repetitionArgs.getTypes()) {
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

static mlir::LogicalResult runLoopRestructuring(mlir::PatternRewriter &rewriter,
                                                mlir::Region &region) {
  llvm::errs() << "Build scc\n";
  auto scc = buildSCC(region);

  scc.dump();

  bool changed = false;
  for (auto &node : scc.nodes)
    changed = restructureLoop(rewriter, node) || changed;

  return mlir::success(changed);
}

struct LoopRestructuringBr : public mlir::OpRewritePattern<mlir::cf::BranchOp> {
  // Set low benefit, so all if simplifications will run first.
  LoopRestructuringBr(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::cf::BranchOp>(context,
                                                   /*benefit*/ 0) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto block = op->getBlock();
    if (!isEntryBlock(*block))
      return mlir::failure();

    return runLoopRestructuring(rewriter, *block->getParent());
  }
};

struct LoopRestructuringCondBr
    : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  // Set low benefit, so all if simplifications will run first.
  LoopRestructuringCondBr(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::cf::CondBranchOp>(context,
                                                       /*benefit*/ 0) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto block = op->getBlock();
    if (!isEntryBlock(*block))
      return mlir::failure();

    return runLoopRestructuring(rewriter, *block->getParent());
  }
};

static llvm::SmallVector<mlir::Value> getDefinedValues(mlir::Block &block) {
  llvm::SmallVector<mlir::Value> ret;
  auto checkVal = [&](mlir::Value val) {
    if (val.isUsedOutsideOfBlock(&block))
      ret.emplace_back(val);
  };

  for (auto arg : block.getArguments())
    checkVal(arg);

  for (auto &op : block.without_terminator())
    for (auto res : op.getResults())
      checkVal(res);

  return ret;
}

struct TailLoopToWhile : public mlir::OpRewritePattern<mlir::cf::BranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto bodyBlock = op.getDest();
    auto bodyBr =
        mlir::dyn_cast<mlir::cf::CondBranchOp>(bodyBlock->getTerminator());
    if (!bodyBr)
      return mlir::failure();

    if (bodyBr.getTrueDest() == bodyBr.getFalseDest())
      return mlir::failure();

    for (bool reverse : {false, true}) {
      auto bodyBlock1 = reverse ? bodyBr.getFalseDest() : bodyBr.getTrueDest();
      auto exitBlock = reverse ? bodyBr.getTrueDest() : bodyBr.getFalseDest();

      if (bodyBlock1 != bodyBlock)
        continue;

      mlir::ValueRange bodyArgs = reverse ? bodyBr.getFalseDestOperands()
                                          : bodyBr.getTrueDestOperands();
      mlir::ValueRange exitArgs = reverse ? bodyBr.getTrueDestOperands()
                                          : bodyBr.getFalseDestOperands();

      llvm::SmallVector<mlir::Value> toReplace = getDefinedValues(*bodyBlock);

      mlir::IRMapping mapping;
      auto beforeBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::ValueRange args) {
        assert(bodyArgs.size() + exitArgs.size() + toReplace.size() ==
               args.size());
        mapping.map(bodyArgs, args.take_front(bodyArgs.size()));
        mapping.map(exitArgs, args.take_back(exitArgs.size()));
        mapping.map(bodyBlock->getArguments(),
                    args.take_front(bodyArgs.size()));

        for (auto &op : bodyBlock->without_terminator())
          builder.clone(op, mapping);

        llvm::SmallVector<mlir::Value> results;
        results.reserve(bodyArgs.size() + exitArgs.size());
        for (mlir::ValueRange ranges : {bodyArgs, exitArgs})
          for (mlir::Value val : ranges)
            results.emplace_back(mapping.lookupOrDefault(val));

        for (auto val : toReplace)
          results.emplace_back(mapping.lookupOrDefault(val));

        mlir::Value cond = mapping.lookupOrDefault(bodyBr.getCondition());
        if (reverse) {
          mlir::Value one =
              builder.create<mlir::arith::ConstantIntOp>(loc, 1, /*width*/ 1);
          cond = builder.create<mlir::arith::XOrIOp>(loc, cond, one);
        }
        builder.create<mlir::scf::ConditionOp>(loc, cond, results);
      };

      auto afterBuilder = [](mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::ValueRange args) {
        builder.create<mlir::scf::YieldOp>(loc, args);
      };

      auto bodyArgsTypes = bodyArgs.getTypes();
      auto exitTypes = exitArgs.getTypes();
      mlir::ValueRange toReplaceRange(toReplace);
      auto definedTypes = toReplaceRange.getTypes();
      llvm::SmallVector<mlir::Type> whileTypes(bodyArgsTypes.begin(),
                                               bodyArgsTypes.end());
      whileTypes.append(exitTypes.begin(), exitTypes.end());
      whileTypes.append(definedTypes.begin(), definedTypes.end());

      auto loc = op.getLoc();
      auto initArgs = op.getDestOperands();
      llvm::SmallVector<mlir::Value> whileArgs(initArgs.begin(),
                                               initArgs.end());

      for (auto types : {exitTypes, definedTypes}) {
        for (auto type : types) {
          mlir::Value val = rewriter.create<numba::util::UndefOp>(loc, type);
          whileArgs.emplace_back(val);
        }
      }

      auto whileOp = rewriter.create<mlir::scf::WhileOp>(
          loc, whileTypes, whileArgs, beforeBuilder, afterBuilder);

      auto results = whileOp.getResults();
      auto bodyResults = results.take_front(bodyArgsTypes.size());
      auto exitResults =
          results.drop_front(bodyArgsTypes.size()).take_front(exitTypes.size());
      auto definedResults = results.take_back(toReplace.size());
      assert(bodyResults.size() == bodyArgs.size());
      assert(exitResults.size() == exitBlock->getNumArguments());
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, exitBlock,
                                                      exitResults);

      for (auto [oldVal, newVal] : llvm::zip(toReplace, definedResults))
        rewriter.replaceAllUsesWith(oldVal, newVal);

      if (llvm::hasSingleElement(bodyBlock->getUses()))
        eraseBlocks(rewriter, bodyBlock);

      return mlir::success();
    }

    return mlir::failure();
  }
};

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

  if ((op.getLhs() == cond2 && mlir::isConstantIntValue(op.getRhs(), 1)) ||
      (op.getRhs() == cond2 && mlir::isConstantIntValue(op.getLhs(), 1)))
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

    auto ifBlock = inverse ? ifOp.elseBlock() : ifOp.thenBlock();
    auto ifBlockRegion = ifBlock->getParent();

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
    for (auto [oldVal, newVal] : llvm::zip(definedOutside, newArgs))
      mapping.map(oldVal, newVal);

    rewriter.setInsertionPointToStart(&afterBlock);
    for (auto &op : ifBlock->without_terminator())
      rewriter.clone(op, mapping);

    auto term = mlir::cast<mlir::scf::YieldOp>(ifBlock->getTerminator());
    for (auto [termVal, ifVal] :
         llvm::zip(term.getResults(), ifOp.getResults())) {
      for (auto &use : ifVal.getUses()) {
        assert(use.getOwner() == condOp);
        assert(use.getOperandNumber() < oldArgs.size());
        rewriter.replaceAllUsesWith(oldArgs[use.getOperandNumber()],
                                    mapping.lookupOrDefault(termVal));
      }
    }

    rewriter.setInsertionPoint(condOp);
    for (auto res : ifOp.getResults()) {
      mlir::Value undef =
          rewriter.create<numba::util::UndefOp>(loc, res.getType());
      rewriter.replaceAllUsesWith(res, undef);
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
        ScfIfRewriteOneExit,
        LoopRestructuringBr,
        LoopRestructuringCondBr,
        TailLoopToWhile,
        WhileMoveToAfter
        // clang-format on
        >(context);

    context->getLoadedDialect<mlir::cf::ControlFlowDialect>()
        ->getCanonicalizationPatterns(patterns);

    mlir::cf::BranchOp::getCanonicalizationPatterns(patterns, context);
    mlir::cf::CondBranchOp::getCanonicalizationPatterns(patterns, context);
    mlir::scf::ExecuteRegionOp::getCanonicalizationPatterns(patterns, context);
    mlir::scf::IfOp::getCanonicalizationPatterns(patterns, context);
    mlir::scf::WhileOp::getCanonicalizationPatterns(patterns, context);

    auto op = getOperation();
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
