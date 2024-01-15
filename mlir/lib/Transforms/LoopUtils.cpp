// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/LoopUtils.hpp"

#include "numba/Analysis/AliasAnalysis.hpp"

#include <llvm/ADT/SmallVector.h>

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LogicalResult.h>

// TODO: Copypasted from mlir
namespace {
using namespace mlir;

/// Verify there are no nested ParallelOps.
static bool hasNestedParallelOp(scf::ParallelOp ploop) {
  auto walkResult = ploop.getBody()->walk(
      [](scf::ParallelOp) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Verify equal iteration spaces.
static bool equalIterationSpaces(scf::ParallelOp firstPloop,
                                 scf::ParallelOp secondPloop) {
  if (firstPloop.getNumLoops() != secondPloop.getNumLoops())
    return false;

  auto matchOperands = [&](const OperandRange &lhs,
                           const OperandRange &rhs) -> bool {
    // TODO: Extend this to support aliases and equal constants.
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
  };
  return matchOperands(firstPloop.getLowerBound(),
                       secondPloop.getLowerBound()) &&
         matchOperands(firstPloop.getUpperBound(),
                       secondPloop.getUpperBound()) &&
         matchOperands(firstPloop.getStep(), secondPloop.getStep());
}

/// Checks if the parallel loops have mixed access to the same buffers. Returns
/// `true` if the first parallel loop writes to the same indices that the second
/// loop reads.
static bool haveNoReadsAfterWriteExceptSameIndex(
    scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
    const IRMapping &firstToSecondPloopIndices,
    llvm::function_ref<mlir::AliasAnalysis &()> getAnalysis) {
  DenseMap<Value, SmallVector<ValueRange, 1>> bufferStores;
  SmallVector<Value> bufferStoresVec;
  firstPloop.getBody()->walk([&](memref::StoreOp store) {
    bufferStores[store.getMemRef()].push_back(store.getIndices());
    bufferStoresVec.emplace_back(store.getMemRef());
  });
  auto walkResult = secondPloop.getBody()->walk([&](memref::LoadOp load) {
    // Stop if the memref is defined in secondPloop body. Careful alias analysis
    // is needed.
    auto *memrefDef = load.getMemRef().getDefiningOp();
    if (memrefDef && memrefDef->getBlock() == load->getBlock())
      return WalkResult::interrupt();

    if (bufferStores.empty())
      return WalkResult::advance();

    auto write = bufferStores.find(load.getMemRef());
    if (write == bufferStores.end()) {
      auto &analysis = getAnalysis();
      for (auto store : bufferStoresVec)
        if (!analysis.alias(store, load.getMemRef()).isNo())
          return WalkResult::interrupt();

      return WalkResult::advance();
    }

    // Allow only single write access per buffer.
    if (write->second.size() != 1)
      return WalkResult::interrupt();

    // Check that the load indices of secondPloop coincide with store indices of
    // firstPloop for the same memrefs.
    auto storeIndices = write->second.front();
    auto loadIndices = load.getIndices();
    if (storeIndices.size() != loadIndices.size())
      return WalkResult::interrupt();
    for (size_t i = 0, e = storeIndices.size(); i < e; ++i) {
      if (firstToSecondPloopIndices.lookupOrDefault(storeIndices[i]) !=
          loadIndices[i])
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !walkResult.wasInterrupted();
}

/// Analyzes dependencies in the most primitive way by checking simple read and
/// write patterns.
static LogicalResult
verifyDependencies(scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
                   const IRMapping &firstToSecondPloopIndices,
                   llvm::function_ref<mlir::AliasAnalysis &()> getAnalysis) {
  for (auto res : firstPloop.getResults()) {
    for (auto user : res.getUsers()) {
      if (secondPloop->isAncestor(user))
        return mlir::failure();
    }
  }

  if (!haveNoReadsAfterWriteExceptSameIndex(
          firstPloop, secondPloop, firstToSecondPloopIndices, getAnalysis))
    return failure();

  IRMapping secondToFirstPloopIndices;
  secondToFirstPloopIndices.map(secondPloop.getBody()->getArguments(),
                                firstPloop.getBody()->getArguments());
  return success(haveNoReadsAfterWriteExceptSameIndex(
      secondPloop, firstPloop, secondToFirstPloopIndices, getAnalysis));
}

static bool
isFusionLegal(scf::ParallelOp firstPloop, scf::ParallelOp secondPloop,
              const IRMapping &firstToSecondPloopIndices,
              llvm::function_ref<mlir::AliasAnalysis &()> getAnalysis) {
  return !hasNestedParallelOp(firstPloop) &&
         !hasNestedParallelOp(secondPloop) &&
         equalIterationSpaces(firstPloop, secondPloop) &&
         succeeded(verifyDependencies(firstPloop, secondPloop,
                                      firstToSecondPloopIndices, getAnalysis));
}

/// Prepends operations of firstPloop's body into secondPloop's body.
static bool
fuseIfLegal(scf::ParallelOp firstPloop, scf::ParallelOp &secondPloop,
            OpBuilder &builder,
            llvm::function_ref<mlir::AliasAnalysis &()> getAnalysis) {
  Block *block1 = firstPloop.getBody();
  Block *block2 = secondPloop.getBody();
  IRMapping firstToSecondPloopIndices;
  firstToSecondPloopIndices.map(block1->getArguments(), block2->getArguments());

  if (!isFusionLegal(firstPloop, secondPloop, firstToSecondPloopIndices,
                     getAnalysis))
    return false;

  mlir::DominanceInfo dom;
  for (auto user : firstPloop->getUsers())
    if (!dom.properlyDominates(secondPloop, user))
      return false;

  ValueRange inits1 = firstPloop.getInitVals();
  ValueRange inits2 = secondPloop.getInitVals();

  SmallVector<Value> newInitVars(inits1.begin(), inits1.end());
  newInitVars.append(inits2.begin(), inits2.end());

  IRRewriter b(builder);
  b.setInsertionPoint(secondPloop);
  auto newSecondPloop = b.create<scf::ParallelOp>(
      secondPloop.getLoc(), secondPloop.getLowerBound(),
      secondPloop.getUpperBound(), secondPloop.getStep(), newInitVars);

  Block *newBlock = newSecondPloop.getBody();
  auto term1 = cast<scf::ReduceOp>(block1->getTerminator());
  auto term2 = cast<scf::ReduceOp>(block2->getTerminator());

  b.inlineBlockBefore(block2, newBlock, newBlock->begin(),
                      newBlock->getArguments());
  b.inlineBlockBefore(block1, newBlock, newBlock->begin(),
                      newBlock->getArguments());

  ValueRange results = newSecondPloop.getResults();
  if (!results.empty()) {
    b.setInsertionPointToEnd(newBlock);

    ValueRange reduceArgs1 = term1.getOperands();
    ValueRange reduceArgs2 = term2.getOperands();
    SmallVector<Value> newReduceArgs(reduceArgs1.begin(), reduceArgs1.end());
    newReduceArgs.append(reduceArgs2.begin(), reduceArgs2.end());

    auto newReduceOp = b.create<scf::ReduceOp>(term2.getLoc(), newReduceArgs);

    for (auto &&[i, reg] : llvm::enumerate(llvm::concat<Region>(
             term1.getReductions(), term2.getReductions()))) {
      Block &oldRedBlock = reg.front();
      Block &newRedBlock = newReduceOp.getReductions()[i].front();
      b.inlineBlockBefore(&oldRedBlock, &newRedBlock, newRedBlock.begin(),
                          newRedBlock.getArguments());
    }

    firstPloop.replaceAllUsesWith(results.take_front(inits1.size()));
    secondPloop.replaceAllUsesWith(results.take_back(inits2.size()));
  }
  term1->erase();
  term2->erase();
  firstPloop.erase();
  secondPloop.erase();
  secondPloop = newSecondPloop;
  return true;
}

bool hasNoEffect(mlir::Operation *op) {
  if (op->getNumRegions() != 0)
    return false;

  if (mlir::isa<mlir::CallOpInterface>(op))
    return false;

  if (auto interface = dyn_cast<MemoryEffectOpInterface>(op))
    return !interface.hasEffect<mlir::MemoryEffects::Read>() &&
           !interface.hasEffect<mlir::MemoryEffects::Write>();

  return !op->hasTrait<::mlir::OpTrait::HasRecursiveMemoryEffects>();
}

bool hasNoEffect(mlir::scf::ParallelOp currentPloop, mlir::Operation *op) {
  if (currentPloop && currentPloop->getNumResults() != 0) {
    for (auto arg : op->getOperands()) {
      if (llvm::is_contained(currentPloop.getResults(), arg))
        return false;
    }
  }

  return hasNoEffect(op);
}
} // namespace

mlir::LogicalResult numba::naivelyFuseParallelOps(Region &region) {
  std::unique_ptr<mlir::AliasAnalysis> analysis;
  auto getAnalysis = [&]() -> mlir::AliasAnalysis & {
    if (!analysis) {
      auto parent = region.getParentOfType<mlir::FunctionOpInterface>();
      analysis = std::make_unique<mlir::AliasAnalysis>(parent);
      analysis->addAnalysisImplementation(numba::LocalAliasAnalysis());
    }

    return *analysis;
  };
  OpBuilder b(region);
  // Consider every single block and attempt to fuse adjacent loops.
  bool changed = false;
  SmallVector<SmallVector<scf::ParallelOp, 8>, 1> ploopChains;
  for (auto &block : region) {
    for (auto &op : block)
      for (auto &innerReg : op.getRegions())
        if (succeeded(naivelyFuseParallelOps(innerReg)))
          changed = true;

    ploopChains.clear();
    ploopChains.push_back({});
    // Not using `walk()` to traverse only top-level parallel loops and also
    // make sure that there are no side-effecting ops between the parallel
    // loops.
    scf::ParallelOp currentPloop;
    bool noSideEffects = true;
    for (auto &op : block) {
      if (auto ploop = dyn_cast<scf::ParallelOp>(op)) {
        currentPloop = ploop;
        if (noSideEffects) {
          ploopChains.back().push_back(ploop);
        } else {
          ploopChains.push_back({ploop});
          noSideEffects = true;
        }
        continue;
      }
      // TODO: Handle region side effects properly.
      noSideEffects &= hasNoEffect(currentPloop, &op);
    }
    for (llvm::MutableArrayRef<scf::ParallelOp> ploops : ploopChains) {
      for (size_t i = 0, e = ploops.size(); i + 1 < e; ++i)
        if (fuseIfLegal(ploops[i], ploops[i + 1], b, getAnalysis))
          changed = true;
    }
  }
  return mlir::success(changed);
}

LogicalResult numba::prepareForFusion(
    Region &region, llvm::function_ref<bool(mlir::Operation &)> needPrepare) {
  DominanceInfo dom(region.getParentOp());
  bool changed = false;
  for (auto &block : region) {
    for (auto &parallelOp : llvm::make_early_inc_range(block)) {
      for (auto &innerReg : parallelOp.getRegions())
        if (succeeded(prepareForFusion(innerReg, needPrepare)))
          changed = true;

      if (!needPrepare(parallelOp))
        continue;

      auto it = Block::iterator(parallelOp);
      if (it == block.begin())
        continue;

      --it;

      auto terminate = false;
      while (!terminate) {
        auto &currentOp = *it;
        if (needPrepare(currentOp))
          break;

        if (it == block.begin()) {
          terminate = true;
        } else {
          --it;
        }

        bool canMove = [&]() {
          if (currentOp.hasTrait<mlir::OpTrait::ConstantLike>())
            return false;

          if (!hasNoEffect(&currentOp))
            return false;

          for (auto arg : currentOp.getOperands())
            if (!dom.properlyDominates(arg, &parallelOp))
              return false;

          for (auto user : currentOp.getUsers())
            if (parallelOp.isAncestor(user) ||
                !dom.properlyDominates(&parallelOp, user))
              return false;

          return true;
        }();

        if (canMove) {
          currentOp.moveAfter(&parallelOp);
          changed = true;
        }
      }
    }
  }
  return mlir::success(changed);
}
