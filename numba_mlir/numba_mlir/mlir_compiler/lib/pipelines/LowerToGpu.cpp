// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pipelines/LowerToGpu.hpp"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/ComplexToStandard/ComplexToStandard.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/GPU/Transforms/Utils.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Math/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include "BasePipeline.hpp"
#include "CheckGpuCaps.hpp"
#include "PyLinalgResolver.hpp"
#include "pipelines/LowerToLlvm.hpp"
#include "pipelines/PlierToLinalg.hpp"
#include "pipelines/PlierToScf.hpp"
#include "pipelines/PlierToStd.hpp"
#include "pipelines/PreLowSimplifications.hpp"

#include "numba/Compiler/PipelineRegistry.hpp"
#include "numba/Conversion/GpuRuntimeToLlvm.hpp"
#include "numba/Conversion/GpuToGpuRuntime.hpp"
#include "numba/Conversion/UtilConversion.hpp"
#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "numba/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.hpp"
#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Dialect/plier/Dialect.hpp" // TODO: for slice slice type
#include "numba/Transforms/CallLowering.hpp"
#include "numba/Transforms/CastUtils.hpp"
#include "numba/Transforms/CommonOpts.hpp"
#include "numba/Transforms/CompositePass.hpp"
#include "numba/Transforms/PipelineUtils.hpp"
#include "numba/Transforms/RewriteWrapper.hpp"
#include "numba/Transforms/TypeConversion.hpp"

namespace {
static void moveOpsIntoParallel(mlir::scf::ParallelOp outer, int depth = 0) {
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
  if (depth >= 6)
    return;

  moveOpsIntoParallel(parallelOp, depth);
}

static gpu_runtime::GPURegionDescAttr getGpuRegionEnv(mlir::Operation *op) {
  assert(op && "Invalid op");
  while (auto region =
             op->getParentOfType<numba::util::EnvironmentRegionOp>()) {
    if (auto env = mlir::dyn_cast<gpu_runtime::GPURegionDescAttr>(
            region.getEnvironment()))
      return env;

    op = region;
  }
  return {};
}

static bool isGpuRegion(numba::util::EnvironmentRegionOp op) {
  return op.getEnvironment().isa<gpu_runtime::GPURegionDescAttr>();
}

struct PrepareForGPUPass
    : public mlir::PassWrapper<PrepareForGPUPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareForGPUPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([](numba::util::EnvironmentRegionOp envOp) {
      if (!isGpuRegion(envOp))
        return;

      for (auto &op : envOp.getRegion().front()) {
        if (auto parallel = mlir::dyn_cast<mlir::scf::ParallelOp>(op))
          moveOpsIntoParallel(parallel);
      }
    });
  }
};

static mlir::LogicalResult
convertParallelToFor(mlir::scf::ParallelOp op,
                     mlir::PatternRewriter &rewriter) {
  mlir::OpBuilder::InsertionGuard g(rewriter);
  auto lowerBounds = op.getLowerBound();
  auto upperBounds = op.getUpperBound();
  auto steps = op.getStep();
  mlir::ValueRange initVals = op.getInitVals();
  assert(!steps.empty());

  auto oldBody = op.getBody();
  assert(oldBody->getNumArguments() == steps.size());
  auto loopCount = steps.size();

  auto emptyBuilder = [](mlir::OpBuilder &, mlir::Location, mlir::Value,
                         mlir::ValueRange) {};

  auto loc = op.getLoc();
  mlir::scf::ForOp forOp;
  mlir::ValueRange results;
  llvm::SmallVector<mlir::Value> newIterVars(loopCount);
  for (auto &&[i, it] : llvm::enumerate(llvm::zip(
           lowerBounds, upperBounds, steps, oldBody->getArguments()))) {
    auto &&[begin, end, step, iter] = it;
    if (i != 0) {
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
    forOp = rewriter.create<mlir::scf::ForOp>(loc, begin, end, step, initVals,
                                              emptyBuilder);
    if (i == 0) {
      results = forOp.getResults();
    } else {
      rewriter.create<mlir::scf::YieldOp>(loc, forOp.getResults());
    }

    initVals = forOp.getRegionIterArgs();
    newIterVars[i] = forOp.getInductionVar();
  }

  auto newBody = forOp.getBody();

  rewriter.inlineBlockBefore(oldBody, newBody, newBody->begin(), newIterVars);

  llvm::SmallVector<mlir::Value> reduceResults;
  reduceResults.reserve(initVals.size());

  auto reduceOp = mlir::cast<mlir::scf::ReduceOp>(newBody->getTerminator());
  rewriter.setInsertionPoint(reduceOp);
  for (auto &&[reduceRegion, reduceArg] :
       llvm::zip(reduceOp.getReductions(), reduceOp.getOperands())) {
    auto &reduceBody = reduceRegion.front();
    assert(reduceBody.getNumArguments() == 2);
    auto term =
        mlir::cast<mlir::scf::ReduceReturnOp>(reduceBody.getTerminator());
    auto reduceIdx = reduceResults.size();
    reduceResults.emplace_back(term.getResult());
    rewriter.eraseOp(term);
    mlir::Value reduceArgs[2] = {forOp.getRegionIterArgs()[reduceIdx],
                                 reduceArg};
    rewriter.inlineBlockBefore(&reduceBody, reduceOp, reduceArgs);
  }
  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(reduceOp, reduceResults);
  rewriter.replaceOp(op, results);
  return mlir::success();
}

struct RemoveNestedParallel
    : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::scf::ParallelOp>())
      return mlir::failure();

    return convertParallelToFor(op, rewriter);
  }
};

// TODO: fix ParallelLoopToGpuPass
struct RemoveNestedParallelPass
    : public numba::RewriteWrapperPass<RemoveNestedParallelPass, void, void,
                                       RemoveNestedParallel> {};

struct CheckParallelToGpu
    : public mlir::PassWrapper<CheckParallelToGpu, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckParallelToGpu)

  void runOnOperation() override {
    auto visitor = [](mlir::scf::ParallelOp op) -> mlir::WalkResult {
      if (getGpuRegionEnv(op)) {
        op->emitError("scf.parallel op wasn't converted inside GPU region");
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    };
    if (getOperation()->walk(visitor).wasInterrupted())
      return signalPassFailure();
  };
};

struct RemoveGpuRegion
    : public mlir::OpRewritePattern<numba::util::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isGpuRegion(op))
      return mlir::failure();

    numba::util::EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return mlir::success();
  }
};

struct RemoveGpuRegionPass
    : public numba::RewriteWrapperPass<RemoveGpuRegionPass, void, void,
                                       RemoveGpuRegion> {};

struct KernelMemrefOpsMovementPass
    : public mlir::PassWrapper<KernelMemrefOpsMovementPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelMemrefOpsMovementPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto &body = func.getBody();
    if (body.empty())
      return;

    mlir::DominanceInfo dom(func);
    body.walk([&](mlir::gpu::LaunchOp launch) {
      launch.getBody().walk([&](mlir::Operation *op) {
        if (!mlir::isa<mlir::memref::DimOp,
                       mlir::memref::ExtractStridedMetadataOp>(op))
          return;

        for (auto &arg : op->getOpOperands()) {
          auto argOp = [&]() -> mlir::Operation * {
            auto val = arg.get();
            auto defOp = val.getDefiningOp();
            if (defOp)
              return defOp;

            return val.getParentBlock()->getParentOp();
          }();

          if (!dom.dominates(argOp, launch))
            return;
        }

        op->moveBefore(launch);
      });
    });
  }
};

static bool isMathOp(mlir::Operation *op) {
  assert(op);
  return mlir::isa<mlir::arith::ArithDialect, mlir::math::MathDialect,
                   mlir::complex::ComplexDialect>(op->getDialect());
}

template <typename C>
static void collectLaunchOps(mlir::Operation *op, C &res) {
  assert(op);

  llvm::SmallVector<mlir::Operation *> stack;
  auto users = op->getUsers();
  stack.append(users.begin(), users.end());

  while (!stack.empty()) {
    auto current = stack.pop_back_val();
    if (isMathOp(current)) {
      auto users = current->getUsers();
      stack.append(users.begin(), users.end());
      continue;
    }

    if (auto launch = mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(current))
      res.insert(launch);
  }
}

static void copySuggestBlockTree(mlir::Operation *op, mlir::OpBuilder &builder,
                                 mlir::IRMapping &mapping) {
  assert(op);

  llvm::SmallVector<mlir::Operation *> stack;
  auto users = op->getUsers();
  stack.append(users.begin(), users.end());

  while (!stack.empty()) {
    auto current = stack.pop_back_val();
    if (isMathOp(current)) {
      builder.setInsertionPoint(current);
      builder.clone(*current, mapping);
      auto users = current->getUsers();
      stack.append(users.begin(), users.end());
    }
  }
}

struct GPULowerDefaultLocalSize
    : public mlir::PassWrapper<GPULowerDefaultLocalSize,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPULowerDefaultLocalSize)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto &region = func.getBody();
    if (region.empty())
      return;

    if (!llvm::hasSingleElement(region)) {
      func.emitError("Only strucutred control flow is supported");
      signalPassFailure();
      return;
    }

    llvm::StringRef funcName("set_default_local_size");
    mlir::func::CallOp setDefSize;
    for (auto op : region.front().getOps<mlir::func::CallOp>()) {
      if (op.getCallee() == funcName && op->getNumOperands() == 3) {
        setDefSize = op;
        break;
      }
    }

    mlir::IRMapping mapping;
    llvm::SmallSetVector<mlir::gpu::LaunchFuncOp, 8> launchOps;

    mlir::DominanceInfo dom;
    mlir::OpBuilder builder(&getContext());
    func.walk([&](gpu_runtime::GPUSuggestBlockSizeOp op) {
      auto loc = op.getLoc();
      if (setDefSize && dom.properlyDominates(setDefSize, op)) {
        auto localSizes = setDefSize.getOperands();
        builder.setInsertionPoint(op);
        for (auto i : llvm::seq(0u, 3u)) {
          auto castedRes = numba::indexCast(builder, loc, localSizes[i]);
          op.getResult(i).replaceAllUsesWith(castedRes);
        }
        op.erase();
        return;
      }

      launchOps.clear();
      collectLaunchOps(op, launchOps);

      if (launchOps.empty())
        return;

      if (launchOps.size() == 1) {
        auto launch = launchOps.front();
        builder.setInsertionPoint(op);
        auto newOp = builder.create<gpu_runtime::GPUSuggestBlockSizeOp>(
            loc, /*queue*/ std::nullopt, op.getGridSize(), launch.getKernel());
        op->replaceAllUsesWith(newOp.getResults());
        op.erase();
        return;
      }

      for (auto launch : launchOps) {
        mapping.clear();

        builder.setInsertionPoint(op);
        auto newOp = builder.create<gpu_runtime::GPUSuggestBlockSizeOp>(
            loc, /*queue*/ std::nullopt, op.getGridSize(), launch.getKernel());

        mapping.map(op.getResults(), newOp.getResults());
        copySuggestBlockTree(op, builder, mapping);

        for (auto &arg : launch->getOpOperands()) {
          auto val = arg.get();
          auto newVal = mapping.lookupOrDefault(val);
          if (newVal != val)
            arg.set(newVal);
        }
      }
    });

    func.walk([&](mlir::func::CallOp op) {
      if (op.getCallee() == funcName) {
        if (!op->use_empty()) {
          op.emitError() << funcName << " call wasn't removed";
          signalPassFailure();
          return;
        }
        op.erase();
      }
    });
  }
};

struct FlattenScfIf : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return mlir::failure();

    auto arithDialect =
        getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
    auto canFlatten = [&](mlir::Operation *op) {
      return op->getDialect() == arithDialect;
    };

    auto &trueBody = op.getThenRegion().front();
    auto &falseBody = op.getElseRegion().front();
    for (auto *block : {&trueBody, &falseBody})
      for (auto &op : block->without_terminator())
        if (!canFlatten(&op))
          return mlir::failure();

    mlir::IRMapping mapper;
    for (auto *block : {&trueBody, &falseBody})
      for (auto &op : block->without_terminator())
        rewriter.clone(op, mapper);

    auto trueYield = mlir::cast<mlir::scf::YieldOp>(trueBody.getTerminator());
    auto falseYield = mlir::cast<mlir::scf::YieldOp>(falseBody.getTerminator());

    llvm::SmallVector<mlir::Value> results;
    results.reserve(op->getNumResults());

    auto loc = op.getLoc();
    auto cond = op.getCondition();
    for (auto &&[origTrueVal, origFalseVal] :
         llvm::zip(trueYield.getResults(), falseYield.getResults())) {
      auto trueVal = mapper.lookupOrDefault(origTrueVal);
      auto falseVal = mapper.lookupOrDefault(origFalseVal);
      auto res =
          rewriter.create<mlir::arith::SelectOp>(loc, cond, trueVal, falseVal);
      results.emplace_back(res);
    }

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct HoistScfIfChecks : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;

    llvm::SmallVector<mlir::Operation *> opsToMove;
    const size_t MAX_OPS_TO_MOVE = 16;
    auto copyOps = [&](auto &&ops) {
      opsToMove.clear();
      for (auto &innerOp : ops) {
        if (opsToMove.size() >= MAX_OPS_TO_MOVE)
          return;

        if (!mlir::isa<mlir::arith::ArithDialect>(innerOp.getDialect()))
          return;

        opsToMove.emplace_back(&innerOp);
      }

      for (auto innerOp : opsToMove) {
        rewriter.modifyOpInPlace(innerOp, [&] { innerOp->moveBefore(op); });
      }
      changed = true;
    };

    for (mlir::Block *body : {op.thenBlock(), op.elseBlock()}) {
      if (!body)
        continue;

      auto ops = body->without_terminator();
      // Must have at nested scf.if and al least one additional op.
      if (!llvm::hasNItemsOrMore(ops, 2))
        continue;

      ops = {ops.begin(), std::prev(ops.end())};
      if (!mlir::isa<mlir::scf::IfOp>(*ops.end()))
        continue;

      copyOps(ops);
    }

    return mlir::success(changed);
  }
};

struct MergeNestedScfIf : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.elseBlock())
      return mlir::failure();

    for (bool inverse : {false, true}) {
      auto body1 = inverse ? op.elseBlock() : op.thenBlock();
      auto body2 = inverse ? op.thenBlock() : op.elseBlock();

      if (!llvm::hasSingleElement(body1->without_terminator()) ||
          !body2->without_terminator().empty())
        continue;

      auto nestedIf = mlir::dyn_cast<mlir::scf::IfOp>(body1->front());
      if (!nestedIf || !nestedIf.elseBlock())
        continue;

      auto yield1 = mlir::cast<mlir::scf::YieldOp>(body1->getTerminator());
      if (yield1.getResults() != nestedIf.getResults())
        continue;

      auto yield2 = mlir::cast<mlir::scf::YieldOp>(body2->getTerminator());
      for (bool inverseInner : {false, true}) {
        auto nestedBody2 =
            inverseInner ? nestedIf.thenBlock() : nestedIf.elseBlock();

        if (!nestedBody2->without_terminator().empty())
          continue;

        auto nestedYield2 =
            mlir::cast<mlir::scf::YieldOp>(nestedBody2->getTerminator());
        if (nestedYield2.getResults() != yield2.getResults())
          continue;

        auto loc = op.getLoc();
        mlir::Value one;
        auto getInverse = [&](mlir::Value val) -> mlir::Value {
          if (!one)
            one = rewriter.create<mlir::arith::ConstantIntOp>(loc, /*value*/ 1,
                                                              /*width*/ 1);

          return rewriter.create<mlir::arith::XOrIOp>(loc, val, one);
        };

        mlir::Value cond1 = op.getCondition();
        if (inverse)
          cond1 = getInverse(cond1);

        mlir::Value cond2 = nestedIf.getCondition();
        if (inverseInner)
          cond2 = getInverse(cond2);

        mlir::Value newCond =
            rewriter.create<mlir::arith::AndIOp>(loc, cond1, cond2);
        auto newIf = rewriter.create<mlir::scf::IfOp>(loc, op->getResultTypes(),
                                                      newCond);
        auto &oldRegion1 =
            inverseInner ? nestedIf.getElseRegion() : nestedIf.getThenRegion();
        auto &oldRegion2 =
            inverseInner ? nestedIf.getThenRegion() : nestedIf.getElseRegion();
        auto &newRegion1 = newIf.getThenRegion();
        auto &newRegion2 = newIf.getElseRegion();
        rewriter.inlineRegionBefore(oldRegion1, newRegion1, newRegion1.end());
        rewriter.inlineRegionBefore(oldRegion2, newRegion2, newRegion2.end());
        rewriter.replaceOp(op, newIf.getResults());
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

struct FlattenScfPass
    : public numba::RewriteWrapperPass<FlattenScfPass, void, void, FlattenScfIf,
                                       HoistScfIfChecks, MergeNestedScfIf> {};

static mlir::LogicalResult processAllocUser(mlir::Operation *user,
                                            mlir::Operation *allocParent,
                                            mlir::DominanceInfo &dom,
                                            mlir::Operation *&lastUser) {
  auto origUser = user;
  if (user->hasTrait<mlir::OpTrait::IsTerminator>())
    return mlir::failure();

  auto parent = user->getParentOp();
  while (parent != allocParent) {
    user = parent;
    parent = user->getParentOp();
    if (parent == nullptr)
      return mlir::failure();
  }

  if (dom.properlyDominates(lastUser, user))
    lastUser = user;

  for (auto resUser : origUser->getUsers())
    if (mlir::failed(processAllocUser(resUser, allocParent, dom, lastUser)))
      return mlir::failure();

  return mlir::success();
}

template <typename AllocOp, typename DeallocOp>
struct CreateDeallocOp : public mlir::OpRewritePattern<AllocOp> {
  using mlir::OpRewritePattern<AllocOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AllocOp op, mlir::PatternRewriter &rewriter) const override {
    auto allocParent = op->getParentOp();
    mlir::Operation *lastUser = op;
    mlir::DominanceInfo dom;
    for (auto user : op->getUsers())
      if (mlir::isa<DeallocOp>(user) ||
          mlir::failed(processAllocUser(user, allocParent, dom, lastUser)))
        return mlir::failure();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(lastUser);
    rewriter.create<DeallocOp>(lastUser->getLoc(), op);
    return mlir::success();
  }
};

struct GPUExDeallocPass
    : public mlir::PassWrapper<GPUExDeallocPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUExDeallocPass)

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<CreateDeallocOp<gpu_runtime::LoadGpuModuleOp,
                                    gpu_runtime::DestroyGpuModuleOp>,
                    CreateDeallocOp<gpu_runtime::GetGpuKernelOp,
                                    gpu_runtime::DestroyGpuKernelOp>>(ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

template <typename Op, typename ReleaseOp>
static bool outlineOp(mlir::Operation &op,
                      llvm::SmallVectorImpl<mlir::Operation *> &deinit) {
  if (!mlir::isa<Op>(op))
    return false;

  auto opParent = op.getParentOp();
  auto origSize = deinit.size();
  for (auto user : op.getUsers()) {
    if (!mlir::isa<ReleaseOp>(user) || llvm::is_contained(deinit, user))
      continue;

    if (user->getParentOp() != opParent || user->getNumResults() != 0) {
      deinit.resize(origSize);
      return false;
    }
    deinit.emplace_back(user);
  }
  return true;
}

constexpr static llvm::StringLiteral kOutlinedInitAttr("plier.outlined_init");
constexpr static llvm::StringLiteral
    kOutlinedDeinitAttr("plier.outlined_deinit");

struct OutlineInitPass
    : public mlir::PassWrapper<OutlineInitPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineInitPass)

  void runOnOperation() override {
    auto mod = getOperation();

    using outline_func_t =
        bool (*)(mlir::Operation &, llvm::SmallVectorImpl<mlir::Operation *> &);
    const outline_func_t outlineHandlers[] = {
        &outlineOp<gpu_runtime::CreateGpuQueueOp,
                   gpu_runtime::DestroyGpuQueueOp>,
        &outlineOp<gpu_runtime::LoadGpuModuleOp,
                   gpu_runtime::DestroyGpuModuleOp>,
        &outlineOp<gpu_runtime::GetGpuKernelOp,
                   gpu_runtime::DestroyGpuKernelOp>,
    };

    llvm::SmallVector<mlir::Operation *> initOps;
    llvm::SmallVector<mlir::Operation *> deinitOps;
    llvm::SmallVector<mlir::Type> types;
    llvm::SmallVector<mlir::Value> values;
    mlir::IRMapping mapper;
    auto tryOutlineOp = [&](mlir::Operation &op) {
      for (auto arg : op.getOperands()) {
        auto argOp = arg.getDefiningOp();
        if (!argOp || !llvm::is_contained(initOps, argOp))
          return;
      }

      for (auto handler : outlineHandlers) {
        if (handler(op, deinitOps)) {
          initOps.emplace_back(&op);
          return;
        }
      }
    };

    mlir::OpBuilder builder(&getContext());
    auto unknownLoc = builder.getUnknownLoc();
    for (auto func : mod.getOps<mlir::func::FuncOp>()) {
      auto &body = func.getBody();
      if (!llvm::hasSingleElement(body))
        continue;

      auto funcName = func.getName();
      initOps.clear();
      deinitOps.clear();
      func->walk([&](mlir::Operation *op) { tryOutlineOp(*op); });

      if (!initOps.empty()) {
        builder.setInsertionPointToStart(mod.getBody());
        types.clear();
        for (auto *op : initOps)
          for (auto type : op->getResultTypes())
            types.emplace_back(type);

        auto funcType = builder.getFunctionType(std::nullopt, types);
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), (funcName + "outlined_init").str(),
            funcType);
        func.setPrivate();
        func->setAttr(kOutlinedInitAttr, builder.getUnitAttr());
        auto block = func.addEntryBlock();
        builder.setInsertionPointToStart(block);
        mapper.clear();
        values.clear();
        for (auto *op : initOps) {
          auto *newOp = builder.clone(*op, mapper);
          for (auto res : newOp->getResults())
            values.emplace_back(res);
        }
        builder.create<mlir::func::ReturnOp>(unknownLoc, values);

        builder.setInsertionPoint(initOps.front());
        auto call = builder.create<mlir::func::CallOp>(unknownLoc, func);
        call->setAttr(kOutlinedInitAttr, builder.getUnitAttr());
        auto results = call.getResults();
        values.assign(results.begin(), results.end());
        for (auto *op : llvm::reverse(initOps)) {
          auto numRes = op->getNumResults();
          assert(results.size() >= numRes);
          auto newRes = results.take_back(numRes);
          op->replaceAllUsesWith(newRes);
          results = results.drop_back(numRes);
          op->erase();
        }
      }

      if (!deinitOps.empty()) {
        assert(!initOps.empty());
        builder.setInsertionPointToStart(mod.getBody());
        assert(!types.empty());
        auto funcType = builder.getFunctionType(types, std::nullopt);
        auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), (funcName + "outlined_deinit").str(),
            funcType);
        func.setPrivate();
        func->setAttr(kOutlinedDeinitAttr, builder.getUnitAttr());

        auto block = func.addEntryBlock();
        builder.setInsertionPointToStart(block);
        mapper.clear();
        mapper.map(values, block->getArguments());
        for (auto *op : llvm::reverse(deinitOps))
          builder.clone(*op, mapper);

        builder.create<mlir::func::ReturnOp>(unknownLoc);

        builder.setInsertionPoint(deinitOps.front());
        auto call =
            builder.create<mlir::func::CallOp>(unknownLoc, func, values);
        call->setAttr(kOutlinedDeinitAttr, builder.getUnitAttr());
        for (auto *op : deinitOps)
          op->erase();
      }
    }
  }
};

struct GenerateOutlineContextPass
    : public mlir::PassWrapper<GenerateOutlineContextPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateOutlineContextPass)

  void runOnOperation() override {
    auto func = getOperation();
    auto &body = func.getBody();
    if (body.empty())
      return;

    if (!llvm::hasSingleElement(body)) {
      func.emitError("Only strucutred control flow is supported");
      signalPassFailure();
      return;
    }

    mlir::OpBuilder builder(&getContext());
    auto initAttr = builder.getStringAttr(kOutlinedInitAttr);
    auto deinitAttr = builder.getStringAttr(kOutlinedDeinitAttr);

    mlir::func::CallOp init;
    mlir::func::CallOp deinit;
    for (auto &op : body.front()) {
      auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
      if (!call)
        continue;

      if (call->hasAttr(initAttr)) {
        if (init) {
          call.emitError("More than one init function");
          signalPassFailure();
          return;
        }
        init = call;
      }

      if (call->hasAttr(deinitAttr)) {
        if (call->getNumResults() != 0) {
          call.emitError("deinit function mus have zero results");
          signalPassFailure();
          return;
        }

        if (deinit) {
          call.emitError("More than one deinit function");
          signalPassFailure();
          return;
        }
        deinit = call;
      }
    }

    if (!init)
      return;

    mlir::SymbolRefAttr initSym = init.getCalleeAttr();
    mlir::SymbolRefAttr deinitSym = (deinit ? deinit.getCalleeAttr() : nullptr);

    builder.setInsertionPoint(init);
    auto takeCtx = builder.create<numba::util::TakeContextOp>(
        init->getLoc(), initSym, deinitSym, init.getResultTypes());
    auto ctx = takeCtx.getContext();
    auto resValues = takeCtx.getResults();
    init->replaceAllUsesWith(resValues);
    init->erase();

    if (deinit) {
      builder.setInsertionPoint(deinit);
      builder.create<numba::util::ReleaseContextOp>(deinit->getLoc(), ctx);
      deinit->erase();
    } else {
      builder.setInsertionPoint(body.front().getTerminator());
      builder.create<numba::util::ReleaseContextOp>(builder.getUnknownLoc(),
                                                    ctx);
    }
  }
};

static void rerunStdPipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToStdPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  numba::addPipelineJumpMarker(mod, marker);
}

static mlir::FailureOr<mlir::Attribute>
getDeviceDescFromArgs(mlir::MLIRContext *context, mlir::TypeRange argTypes) {
  mlir::Attribute res;
  for (auto arg : argTypes) {
    if (auto tupleType = mlir::dyn_cast<mlir::TupleType>(arg)) {
      auto tupleRes = getDeviceDescFromArgs(context, tupleType.getTypes());
      if (mlir::failed(tupleRes))
        return mlir::failure();

      auto newRes = *tupleRes;
      if (!newRes)
        continue;

      if (!res) {
        res = newRes;
        continue;
      }

      if (newRes != res)
        return mlir::failure();

      continue;
    }

    auto tensor = arg.dyn_cast<numba::ntensor::NTensorType>();
    if (!tensor)
      continue;

    auto env = tensor.getEnvironment()
                   .dyn_cast_or_null<gpu_runtime::GPURegionDescAttr>();
    if (!env)
      continue;

    if (!res) {
      res = env;
    } else if (res != env) {
      return mlir::failure();
    }
  }

  return res;
}

static mlir::FailureOr<mlir::Attribute>
getDeviceDescFromFunc(mlir::MLIRContext *context, mlir::TypeRange argTypes) {
  auto res = getDeviceDescFromArgs(context, argTypes);
  if (mlir::failed(res) || !*res)
    return mlir::failure();

  return *res;
}

struct InsertGpuRegionPass
    : public mlir::PassWrapper<InsertGpuRegionPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGpuRegionPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    llvm::SmallSetVector<mlir::scf::WhileOp, 8> loops;
    getOperation()->walk([&](plier::IternextOp op) {
      auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(op->getParentOp());
      if (!loop)
        return;

      auto getiter = op.getValue().getDefiningOp<plier::GetiterOp>();
      if (!getiter)
        return;

      auto call = getiter.getValue().getDefiningOp<plier::PyCallOp>();
      if (!call)
        return;

      auto name = call.getFuncName();
      if (name != "_gpu_range")
        return;

      loops.insert(loop);
    });

    if (loops.empty())
      return markAllAnalysesPreserved();

    auto *ctx = &getContext();
    auto env = numba::util::ParallelAttr::get(ctx);
    mlir::OpBuilder builder(ctx);
    for (auto loop : loops) {
      auto loc = loop.getLoc();
      builder.setInsertionPoint(loop);
      mlir::Operation *nestedRegion;
      {
        auto region = builder.create<numba::util::EnvironmentRegionOp>(
            loc, env, /*args*/ std::nullopt, loop->getResultTypes());
        mlir::Block &body = region.getRegion().front();
        body.getTerminator()->erase();
        loop.getResults().replaceAllUsesWith(region.getResults());
        builder.setInsertionPointToEnd(&body);
        auto term = builder.create<numba::util::EnvironmentRegionYieldOp>(
            loc, loop.getResults());
        loop->moveBefore(term);
        nestedRegion = region;
      }

      if (getGpuRegionEnv(loop))
        continue;

      auto parent = loop->getParentOfType<mlir::FunctionOpInterface>();
      if (!parent)
        continue;

      auto env = getDeviceDescFromFunc(ctx, parent.getArgumentTypes());
      if (mlir::failed(env))
        continue;

      builder.setInsertionPoint(nestedRegion);
      auto region = builder.create<numba::util::EnvironmentRegionOp>(
          loc, *env, /*args*/ std::nullopt, loop->getResultTypes());
      mlir::Block &body = region.getRegion().front();
      body.getTerminator()->erase();
      loop.getResults().replaceAllUsesWith(region.getResults());
      builder.setInsertionPointToEnd(&body);
      auto term = builder.create<numba::util::EnvironmentRegionYieldOp>(
          loc, loop.getResults());
      nestedRegion->moveBefore(term);
    }
  }
};

struct LowerPlierCalls final : public numba::CallOpLowering {
  LowerPlierCalls(mlir::MLIRContext *context)
      : CallOpLowering(context),
        resolver("numba_mlir.mlir.kernel_impl", "registry") {}

protected:
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location loc,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const override {
    auto res = resolver.rewriteFunc(name, loc, rewriter, args, kwargs);
    if (!res)
      return mlir::failure();

    auto results = std::move(res).value();
    assert(results.size() == op->getNumResults());
    for (auto &&[i, r] : llvm::enumerate(results)) {
      auto dstType = op->getResultTypes()[i];
      if (dstType != r.getType())
        results[i] = rewriter.create<plier::CastOp>(loc, dstType, r);
    }

    rerunStdPipeline(op);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

static mlir::LogicalResult
lowerGetGlobalId(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                 mlir::ValueRange localSizes, mlir::ValueRange gridArgs,
                 mlir::ValueRange blockArgs, mlir::PatternRewriter &builder,
                 unsigned index) {
  auto loc = op.getLoc();
  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    if (val.getType() != indexType)
      return builder.createOrFold<plier::CastOp>(loc, indexType, val);
    return val;
  };
  auto localSize = indexCast(localSizes[index]);
  auto gridArg = indexCast(gridArgs[index]);
  auto blockArg = indexCast(blockArgs[index]);
  mlir::Value res =
      builder.create<mlir::arith::MulIOp>(loc, gridArg, localSize);
  res = builder.create<mlir::arith::AddIOp>(loc, res, blockArg);
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetLocallId(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                 mlir::ValueRange /*localSizes*/, mlir::ValueRange /*gridArgs*/,
                 mlir::ValueRange blockArgs, mlir::PatternRewriter &builder,
                 unsigned index) {
  auto loc = op.getLoc();
  auto res = blockArgs[index];
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetGroupId(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                mlir::ValueRange /*localSizes*/, mlir::ValueRange gridArgs,
                mlir::ValueRange /*blockArgs*/, mlir::PatternRewriter &builder,
                unsigned index) {
  auto loc = op.getLoc();
  auto res = gridArgs[index];
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetGlobalSize(mlir::func::CallOp op, mlir::ValueRange globalSizes,
                   mlir::ValueRange localSizes, mlir::ValueRange /*gridArgs*/,
                   mlir::ValueRange /*blockArgs*/,
                   mlir::PatternRewriter &builder, unsigned index) {
  auto loc = op.getLoc();
  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    if (val.getType() != indexType)
      return builder.createOrFold<mlir::arith::IndexCastOp>(loc, indexType,
                                                            val);
    return val;
  };
  mlir::Value global = indexCast(globalSizes[index]);
  mlir::Value local = indexCast(localSizes[index]);
  mlir::Value res = builder.create<mlir::arith::MulIOp>(loc, global, local);
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetLocalSize(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                  mlir::ValueRange localSizes, mlir::ValueRange /*gridArgs*/,
                  mlir::ValueRange /*blockArgs*/,
                  mlir::PatternRewriter &builder, unsigned index) {
  auto loc = op.getLoc();
  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    if (val.getType() != indexType)
      return builder.createOrFold<mlir::arith::IndexCastOp>(loc, indexType,
                                                            val);
    return val;
  };
  mlir::Value res = indexCast(localSizes[index]);
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static std::array<mlir::Value, 3>
dim3ToArray(const mlir::gpu::KernelDim3 &val) {
  return {val.x, val.y, val.z};
}

struct LowerBuiltinCalls : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    using handler_func_t = mlir::LogicalResult (*)(
        mlir::func::CallOp, mlir::ValueRange, mlir::ValueRange,
        mlir::ValueRange, mlir::ValueRange, mlir::PatternRewriter &, unsigned);
    auto launch = op->getParentOfType<mlir::gpu::LaunchOp>();
    if (!launch)
      return mlir::failure();

    auto handler = [&]() -> handler_func_t {
      static const std::pair<mlir::StringRef, handler_func_t> handlers[] = {
          {"get_global_id", &lowerGetGlobalId},
          {"get_local_id", &lowerGetLocallId},
          {"get_group_id", &lowerGetGroupId},
          {"get_global_size", &lowerGetGlobalSize},
          {"get_local_size", &lowerGetLocalSize},
      };
      auto name = op.getCallee();
      for (auto h : handlers)
        if (h.first == name)
          return h.second;

      return nullptr;
    }();

    if (!handler)
      return mlir::failure();

    if (op.getNumOperands() != 1 || op.getNumResults() != 1 ||
        !op.getOperand(0).getType().isa<mlir::IntegerType>() ||
        !op.getResult(0).getType().isa<mlir::IntegerType>())
      return mlir::failure();

    auto indAttr = mlir::getConstantIntValue(op.getOperands()[0]);
    if (!indAttr)
      return mlir::failure();

    auto ind = *indAttr;
    if (ind < 0 || ind >= 3)
      return mlir::failure();

    auto globalSize = dim3ToArray(launch.getGridSize());
    auto localSize = dim3ToArray(launch.getBlockSize());

    auto globalArgs = dim3ToArray(launch.getBlockIds());
    auto localArgs = dim3ToArray(launch.getThreadIds());

    auto uind = static_cast<unsigned>(ind);
    return handler(op, globalSize, localSize, globalArgs, localArgs, rewriter,
                   uind);
  }
};

static bool isValidArraySig(mlir::func::CallOp op) {
  if (op.getNumResults() != 1)
    return false;

  auto res = op.getResult(0);
  auto resType = mlir::dyn_cast<numba::ntensor::NTensorType>(res.getType());
  if (!resType)
    return false;

  mlir::TypeRange types = op.getOperandTypes();
  if (types.size() == 1 && llvm::isa<mlir::TupleType>(types.front()))
    types = llvm::cast<mlir::TupleType>(types.front()).getTypes();

  if (types.size() != resType.getShape().size())
    return false;

  return llvm::all_of(types, [](auto t) { return t.isIntOrIndex(); });
}

struct LowerKernelAllocCalls
    : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isValidArraySig(op))
      return mlir::failure();

    auto res = op.getResult(0);
    auto resType = mlir::cast<numba::ntensor::NTensorType>(res.getType());
    bool isLocal; // Othewise private;
    auto name = op.getCallee();
    if (name.starts_with("local_array_")) {
      isLocal = true;
    } else if (name.starts_with("private_array_")) {
      isLocal = false;
    } else {
      return mlir::failure();
    }

    auto origResType = resType;
    auto shape = resType.getShape();
    auto type = resType.getElementType();
    if (!resType.getEnvironment()) {
      if (auto env = getGpuRegionEnv(op)) {
        auto layout = resType.getLayout();
        resType = numba::ntensor::NTensorType::get(shape, type, env, layout);
      }
    }

    auto memSpace = mlir::gpu::AddressSpaceAttr::get(
        rewriter.getContext(),
        isLocal ? mlir::gpu::GPUDialect::getWorkgroupAddressSpace()
                : mlir::gpu::GPUDialect::getPrivateAddressSpace());
    auto memrefType = mlir::MemRefType::get(
        shape, type, mlir::MemRefLayoutAttrInterface{}, memSpace);
    auto memrefGenericType = mlir::MemRefType::get(shape, type);

    auto loc = op.getLoc();
    auto indexType = rewriter.getIndexType();
    auto indexCast = [&](mlir::Value val) -> mlir::Value {
      auto type = val.getType();
      if (mlir::isa<mlir::IndexType>(type))
        return val;

      auto intType = mlir::cast<mlir::IntegerType>(type);
      if (!intType.isSignless()) {
        intType =
            mlir::IntegerType::get(intType.getContext(), intType.getWidth());
        val = rewriter.create<numba::util::SignCastOp>(loc, intType, val);
      }

      return rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, val);
    };

    llvm::SmallVector<mlir::Value> args;
    if (op->getNumOperands() == 1 &&
        llvm::isa<mlir::TupleType>(op.getOperand(0).getType())) {
      auto arg = op.getOperand(0);
      auto tupleType = mlir::cast<mlir::TupleType>(arg.getType());
      args.resize(tupleType.size());
      for (auto &&[ind, type] : llvm::enumerate(tupleType)) {
        auto i = static_cast<int64_t>(ind);
        mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
        mlir::Value val =
            rewriter.create<numba::util::TupleExtractOp>(loc, type, arg, idx);
        args[i] = val;
      }
    } else {
      auto a = op.getOperands();
      args.assign(a.begin(), a.end());
    }

    for (auto &&[i, arg] : llvm::enumerate(args))
      args[i] = indexCast(arg);

    mlir::Value alloc =
        rewriter.create<mlir::memref::AllocaOp>(loc, memrefType, args);

    alloc =
        rewriter.create<numba::util::SignCastOp>(loc, memrefGenericType, alloc);
    alloc = rewriter.create<numba::ntensor::FromMemrefOp>(loc, resType, alloc);
    if (resType != origResType)
      alloc = rewriter.create<numba::ntensor::CastOp>(loc, origResType, alloc);

    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};

static bool isValidAtomicSig(mlir::func::CallOp op) {
  if (op.getNumResults() != 1 || op.getNumOperands() != 2)
    return false;

  auto res = op.getResult(0);
  auto arr = op.getOperand(0);
  auto val = op.getOperand(1);

  auto arrType = mlir::dyn_cast<numba::ntensor::NTensorType>(arr.getType());
  if (!arrType)
    return false;

  auto elemType = arrType.getElementType();
  return res.getType() == elemType && val.getType() == elemType;
}

struct LowerKernelAtomicCalls
    : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isValidAtomicSig(op))
      return mlir::failure();

    using RMWK = mlir::arith::AtomicRMWKind;

    const std::tuple<mlir::StringRef, RMWK, RMWK> handlers[] = {
        {"atomic_add_", RMWK::addi, RMWK::addf}};

    auto val = op.getOperand(1);
    auto kind = [&]() -> std::optional<RMWK> {
      auto name = op.getCallee();
      for (auto &&[funcName, iKind, fKind] : handlers) {
        if (name.starts_with(funcName)) {
          bool isFloat = mlir::isa<mlir::FloatType>(val.getType());
          return isFloat ? fKind : iKind;
        }
      }
      return std::nullopt;
    }();
    if (!kind)
      return mlir::failure();

    auto arr = op.getOperand(0);

    auto arrType = mlir::cast<numba::ntensor::NTensorType>(arr.getType());
    auto memrefType =
        mlir::MemRefType::get(arrType.getShape(), arrType.getElementType());
    auto signlessMemerefType = numba::makeSignlessType(memrefType);

    auto loc = op.getLoc();
    mlir::Value memref =
        rewriter.create<numba::ntensor::ToMemrefOp>(loc, memrefType, arr);
    if (memrefType != signlessMemerefType)
      memref = rewriter.create<numba::util::SignCastOp>(
          loc, signlessMemerefType, memref);

    auto signelessElemType = signlessMemerefType.getElementType();
    if (val.getType() != signelessElemType)
      val =
          rewriter.create<numba::util::SignCastOp>(loc, signelessElemType, val);

    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    llvm::SmallVector<mlir::Value> indices(memrefType.getRank(), zero);

    mlir::Value newRes = rewriter.create<mlir::memref::AtomicRMWOp>(
        loc, *kind, val, memref, indices);

    auto resType = op.getResult(0).getType();
    if (newRes.getType() != resType)
      newRes = rewriter.create<numba::util::SignCastOp>(loc, resType, newRes);

    rewriter.replaceOp(op, newRes);
    return mlir::success();
  }
};

struct LowerGpuBuiltinsPass
    : public numba::RewriteWrapperPass<
          LowerGpuBuiltinsPass, void,
          numba::DependentDialectsList<mlir::gpu::GPUDialect>, LowerPlierCalls,
          LowerKernelAllocCalls, LowerKernelAtomicCalls> {};

static std::optional<gpu_runtime::FenceFlags>
getFenceFlags(mlir::OpFoldResult arg) {
  auto val = mlir::getConstantIntValue(arg);
  if (!val)
    return std::nullopt;

  auto v = *val;
  if (v == 1)
    return gpu_runtime::FenceFlags::local;

  if (v == 2)
    return gpu_runtime::FenceFlags::global;

  return std::nullopt;
}

template <typename Op>
static void genBarrierOp(mlir::Operation *srcOp,
                         mlir::PatternRewriter &rewriter,
                         gpu_runtime::FenceFlags flags) {
  auto newOp = rewriter.create<Op>(srcOp->getLoc());
  auto attr = rewriter.getI64IntegerAttr(static_cast<int64_t>(flags));
  newOp->setAttr(gpu_runtime::getFenceFlagsAttrName(), attr);

  // TODO: remove
  assert(srcOp->getNumResults() == 1);
  auto retType = srcOp->getResult(0).getType();
  rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(srcOp, retType, nullptr);
}

template <typename Op>
static void genCustomBarrierOp(mlir::Operation *srcOp,
                               mlir::PatternRewriter &rewriter,
                               gpu_runtime::FenceFlags flags) {
  rewriter.create<Op>(srcOp->getLoc(), static_cast<int64_t>(flags));

  // TODO: remove
  assert(srcOp->getNumResults() == 1);
  auto retType = srcOp->getResult(0).getType();
  rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(srcOp, retType, nullptr);
}

class ConvertBarrierOps : public mlir::OpRewritePattern<mlir::func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto operands = op.getOperands();
    if (operands.size() != 1)
      return mlir::failure();

    if (op.getNumResults() != 1)
      return mlir::failure();

    auto fenceFlags = getFenceFlags(operands[0]);
    if (!fenceFlags)
      return mlir::failure();

    using funcptr_t = void (*)(mlir::Operation *, mlir::PatternRewriter &,
                               gpu_runtime::FenceFlags);
    const std::pair<llvm::StringRef, funcptr_t> handlers[] = {
        {"kernel_barrier", &genBarrierOp<mlir::gpu::BarrierOp>},
        {"kernel_mem_fence", &genCustomBarrierOp<gpu_runtime::GPUMemFenceOp>},
    };

    auto funcName = op.getCallee();
    for (auto &h : handlers) {
      if (h.first == funcName) {
        h.second(op, rewriter, *fenceFlags);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

template <mlir::gpu::AllReduceOperation ReduceType>
static void genGroupOp(mlir::Operation *srcOp, mlir::PatternRewriter &rewriter,
                       mlir::Value arg) {
  auto ctx = srcOp->getContext();
  auto reduceAttr = mlir::gpu::AllReduceOperationAttr::get(ctx, ReduceType);
  rewriter.replaceOpWithNewOp<mlir::gpu::AllReduceOp>(srcOp, arg, reduceAttr,
                                                      /*uniform*/ false);
}

class ConvertGroupOps : public mlir::OpRewritePattern<mlir::func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto operands = op.getOperands();
    if (operands.size() != 1)
      return mlir::failure();

    if (op.getNumResults() != 1)
      return mlir::failure();

    auto src = operands[0];
    auto srcType = src.getType();

    if (srcType != op.getResult(0).getType())
      return mlir::failure();

    auto funcName = op.getCallee();
    if (!funcName.consume_front("group_"))
      return mlir::failure();

    using funcptr_t =
        void (*)(mlir::Operation *, mlir::PatternRewriter &, mlir::Value);
    const std::tuple<llvm::StringRef, funcptr_t, funcptr_t> handlers[] = {
        {"reduce_add", &genGroupOp<mlir::gpu::AllReduceOperation::ADD>,
         &genGroupOp<mlir::gpu::AllReduceOperation::ADD>},
        {"reduce_mul", &genGroupOp<mlir::gpu::AllReduceOperation::MUL>,
         &genGroupOp<mlir::gpu::AllReduceOperation::MUL>},
        {"reduce_min", &genGroupOp<mlir::gpu::AllReduceOperation::MINSI>,
         &genGroupOp<mlir::gpu::AllReduceOperation::MINIMUMF>},
        {"reduce_max", &genGroupOp<mlir::gpu::AllReduceOperation::MAXSI>,
         &genGroupOp<mlir::gpu::AllReduceOperation::MAXIMUMF>},
    };

    for (auto &&[name, intFunc, floatFunc] : handlers) {
      if (funcName.starts_with(name)) {
        if (mlir::isa<mlir::IntegerType>(srcType)) {
          intFunc(op, rewriter, src);
        } else {
          floatFunc(op, rewriter, src);
        }
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

template <typename SpvOp>
static mlir::Value reduceOp(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value val1, mlir::Value val2) {
  return builder.create<SpvOp>(loc, val1, val2);
}

using ReduceFuncType = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                       mlir::Value, mlir::Value);
static ReduceFuncType getReduceFunc(mlir::gpu::AllReduceOperation op,
                                    bool isFloat) {
  using ReduceOp = mlir::gpu::AllReduceOperation;
  using HandlerType = std::tuple<ReduceOp, ReduceFuncType, ReduceFuncType>;
  const HandlerType handers[] = {{ReduceOp::ADD, &reduceOp<mlir::arith::AddIOp>,
                                  &reduceOp<mlir::arith::AddFOp>}};
  for (auto handler : handers) {
    if (std::get<0>(handler) == op)
      return isFloat ? std::get<2>(handler) : std::get<1>(handler);
  }
  return nullptr;
}

class ConvertGroupOpsToSubgroup
    : public mlir::OpRewritePattern<mlir::gpu::AllReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllReduceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto launchOp = op->getParentOfType<mlir::gpu::LaunchOp>();
    if (!launchOp)
      return mlir::failure();

    // This case will be handled by MakeBarriersUniformPass.
    if (mlir::isa<mlir::scf::IfOp>(op->getParentOp()) &&
        op->getParentOp()->getParentOp() == launchOp)
      return mlir::failure();

    if (!op.getOp())
      return mlir::failure();

    if (op.getUniform())
      return mlir::failure();

    if (!op.getType().isIntOrFloat())
      return mlir::failure();

    auto isFloat = op.getType().isa<mlir::FloatType>();
    auto reduceFunc = getReduceFunc(*op.getOp(), isFloat);

    if (!reduceFunc)
      return mlir::failure();

    mlir::Value groupBuffer;
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(launchOp);
      auto loc = launchOp->getLoc();
      mlir::Value size = launchOp.getBlockSizeX();
      size = rewriter.create<mlir::arith::MulIOp>(loc, size,
                                                  launchOp.getBlockSizeY());
      size = rewriter.create<mlir::arith::MulIOp>(loc, size,
                                                  launchOp.getBlockSizeZ());

      // TODO: Subgroup size is hardcoded for now.
      mlir::Value subgroupSize =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 8);

      mlir::Value numSubgroups =
          rewriter.create<mlir::arith::CeilDivSIOp>(loc, size, subgroupSize);

      auto elemType = op.getType();

      auto addrSpace = mlir::gpu::GPUDialect::getWorkgroupAddressSpace();
      auto storageClass =
          mlir::gpu::AddressSpaceAttr::get(rewriter.getContext(), addrSpace);
      auto memrefType = mlir::MemRefType::get(mlir::ShapedType::kDynamic,
                                              elemType, nullptr, storageClass);
      groupBuffer = rewriter
                        .create<mlir::gpu::AllocOp>(
                            loc, memrefType, /*asyncToken*/ mlir::Type(),
                            /*asyncDependencies*/ std::nullopt, numSubgroups,
                            /*symbolOperands*/ std::nullopt)
                        .getMemref();
      rewriter.setInsertionPointAfter(launchOp);
      rewriter.create<mlir::gpu::DeallocOp>(loc, /*asyncToken*/ mlir::Type(),
                                            /*asyncDependencies*/ std::nullopt,
                                            groupBuffer);
    }

    mlir::Value subgroupId = [&]() {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&launchOp.getBody().front());
      return rewriter.create<mlir::gpu::SubgroupIdOp>(rewriter.getUnknownLoc());
    }();

    auto loc = op->getLoc();
    auto reduceType = *op.getOp();
    mlir::Value sgResult = rewriter.create<mlir::gpu::SubgroupReduceOp>(
        loc, op.getValue(), reduceType);
    rewriter.create<mlir::memref::StoreOp>(loc, sgResult, groupBuffer,
                                           subgroupId);

    auto barrierOp = rewriter.create<mlir::gpu::BarrierOp>(loc);
    auto barrierFlagAttr = rewriter.getI64IntegerAttr(
        static_cast<int64_t>(gpu_runtime::FenceFlags::local));
    barrierOp->setAttr(gpu_runtime::getFenceFlagsAttrName(), barrierFlagAttr);

    mlir::Value numSubgroups = [&]() {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&launchOp.getBody().front());
      return rewriter.create<mlir::gpu::NumSubgroupsOp>(
          rewriter.getUnknownLoc());
    }();

    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value isFirstSg = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, subgroupId, zero);

    auto ifBodyBuilder = [&](mlir::OpBuilder &ifBuilder, mlir::Location ifLoc) {
      mlir::Value init =
          ifBuilder.create<mlir::memref::LoadOp>(ifLoc, groupBuffer, zero);

      auto forBodyBuilder = [&](mlir::OpBuilder &forBuilder,
                                mlir::Location forLoc, mlir::Value i,
                                mlir::ValueRange args) {
        assert(args.size() == 1);
        auto prev = args.front();
        mlir::Value val =
            forBuilder.create<mlir::memref::LoadOp>(forLoc, groupBuffer, i);

        mlir::Value res = reduceFunc(forBuilder, forLoc, prev, val);
        forBuilder.create<mlir::scf::YieldOp>(forLoc, res);
      };

      mlir::Value res = ifBuilder
                            .create<mlir::scf::ForOp>(ifLoc, one, numSubgroups,
                                                      one, init, forBodyBuilder)
                            .getResult(0);
      mlir::Value isSingleSg = ifBuilder.create<mlir::arith::CmpIOp>(
          ifLoc, mlir::arith::CmpIPredicate::eq, numSubgroups, one);
      res =
          ifBuilder.create<mlir::arith::SelectOp>(ifLoc, isSingleSg, init, res);
      ifBuilder.create<mlir::memref::StoreOp>(ifLoc, res, groupBuffer, zero);
      ifBuilder.create<mlir::scf::YieldOp>(ifLoc);
    };

    rewriter.create<mlir::scf::IfOp>(loc, isFirstSg, ifBodyBuilder);

    barrierOp = rewriter.create<mlir::gpu::BarrierOp>(loc);
    barrierOp->setAttr(gpu_runtime::getFenceFlagsAttrName(), barrierFlagAttr);

    mlir::Value result =
        rewriter.create<mlir::memref::LoadOp>(loc, groupBuffer, zero);
    rewriter.replaceOp(op, result);
    return mlir::failure();
  }
};

struct LowerGpuBuiltins2Pass
    : public numba::RewriteWrapperPass<
          LowerGpuBuiltins2Pass, void, void, ConvertBarrierOps, ConvertGroupOps,
          ConvertGroupOpsToSubgroup, LowerBuiltinCalls> {};

class ConvertLocalArrayAllocOps
    : public mlir::OpRewritePattern<mlir::memref::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocaOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto mod = op->getParentOfType<mlir::gpu::GPUModuleOp>();
    if (!mod)
      return mlir::failure();

    auto oldType = mlir::dyn_cast<mlir::MemRefType>(op.getType());
    if (!oldType || !oldType.hasStaticShape())
      return mlir::failure();

    auto addrSpace = mlir::dyn_cast_if_present<mlir::gpu::AddressSpaceAttr>(
        oldType.getMemorySpace());
    if (!addrSpace || addrSpace.getValue() !=
                          mlir::gpu::GPUDialect::getWorkgroupAddressSpace())
      return mlir::failure();

    auto global = [&]() -> mlir::StringRef {
      auto *block = mod.getBody();
      llvm::SmallString<64> name;
      for (unsigned i = 0;; ++i) {
        if (i == 0) {
          name = "__local_array";
        } else {
          name.clear();
          (llvm::Twine("__local_array") + llvm::Twine(i)).toVector(name);
        }
        if (!mod.lookupSymbol(name))
          break;
      }
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(block);
      auto loc = rewriter.getUnknownLoc();
      auto global = rewriter.create<mlir::memref::GlobalOp>(
          loc, name,
          /*sym_visibility=*/rewriter.getStringAttr("private"),
          /*type=*/oldType,
          /*initial_value=*/nullptr,
          /*constant=*/false,
          /*alignment=*/nullptr);
      return global.getSymName();
    }();

    auto loc = op->getLoc();
    mlir::Value newArray =
        rewriter.create<mlir::memref::GetGlobalOp>(loc, oldType, global);

    rewriter.replaceOp(op, newArray);
    return mlir::success();
  }
};

struct LowerGpuBuiltins3Pass
    : public numba::RewriteWrapperPass<LowerGpuBuiltins3Pass, void, void,
                                       ConvertLocalArrayAllocOps> {};

class GpuLaunchSinkOpsPass
    : public mlir::PassWrapper<GpuLaunchSinkOpsPass,
                               mlir::OperationPass<void>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuLaunchSinkOpsPass)

  void runOnOperation() override {
    using namespace mlir;

    Operation *op = getOperation();
    if (op->walk([](gpu::LaunchOp launch) {
            auto isSinkingBenefial = [](mlir::Operation *op) -> bool {
              if (mlir::isa<mlir::arith::ArithDialect, mlir::math::MathDialect,
                            mlir::complex::ComplexDialect, mlir::ub::UBDialect>(
                      op->getDialect()))
                return true;

              return isa<func::ConstantOp>(op);
            };

            // Pull in instructions that can be sunk
            if (failed(sinkOperationsIntoLaunchOp(launch, isSinkingBenefial)))
              return WalkResult::interrupt();

            return WalkResult::advance();
          }).wasInterrupted())
      signalPassFailure();
  }
};

class GpuPropagateKernelFlagsPass
    : public mlir::PassWrapper<GpuPropagateKernelFlagsPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuPropagateKernelFlagsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto gpuMod = getOperation();
    auto mod = gpuMod->getParentOfType<mlir::ModuleOp>();
    if (!mod) {
      gpuMod->emitError("No module parent");
      return signalPassFailure();
    }

    auto funcs = gpuMod.getOps<mlir::gpu::GPUFuncOp>();
    if (!llvm::hasSingleElement(funcs)) {
      gpuMod->emitError("GPU module must have exactly one func");
      return signalPassFailure();
    }

    auto gpuFunc = *funcs.begin();

    auto funcUses = mlir::SymbolTable::getSymbolUses(gpuFunc, mod);
    if (!funcUses || !llvm::hasSingleElement(*funcUses)) {
      gpuMod->emitError("GPU func must have exactly one use");
      return signalPassFailure();
    }

    auto use =
        mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(funcUses->begin()->getUser());
    if (!use) {
      gpuMod->emitError("Invalid func use");
      return signalPassFailure();
    }

    auto parent = use->getParentOfType<mlir::FunctionOpInterface>();
    if (!parent) {
      gpuMod->emitError("Invalid func parent");
      return signalPassFailure();
    }

    for (auto &&attr : parent->getDiscardableAttrs())
      gpuMod->setAttr(attr.getName(), attr.getValue());
  }
};

static const constexpr llvm::StringLiteral
    kGpuModuleDeviceName("gpu_module_device");

class NameGpuModulesPass
    : public mlir::PassWrapper<NameGpuModulesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NameGpuModulesPass)

  void runOnOperation() override {
    auto mod = getOperation();
    mod->walk([&](mlir::gpu::LaunchFuncOp launch) {
      auto env = getGpuRegionEnv(launch);
      if (!env)
        return;

      auto kernel = launch.getKernel();
      auto gpuModName = kernel.getRootReference();
      auto gpuMod = mod.lookupSymbol<mlir::gpu::GPUModuleOp>(gpuModName);
      if (!gpuMod)
        return;

      auto gpuModAttr = gpuMod->getAttrOfType<gpu_runtime::GPURegionDescAttr>(
          kGpuModuleDeviceName);
      if (gpuModAttr && gpuModAttr != env) {
        gpuMod->emitError("Incompatible gpu module devices: ")
            << gpuModAttr << " and " << env;
        return signalPassFailure();
      }
      gpuMod->setAttr(kGpuModuleDeviceName, env);
    });
  }
};

struct SinkGpuDims : public mlir::OpRewritePattern<mlir::gpu::LaunchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    const mlir::Value dimArgs[] = {op.getGridSizeX(),  op.getGridSizeY(),
                                   op.getGridSizeZ(),  op.getBlockSizeX(),
                                   op.getBlockSizeY(), op.getBlockSizeZ()};
    llvm::SmallVector<std::pair<mlir::OpOperand *, unsigned>> uses;
    for (auto &&[ind, val] : llvm::enumerate(dimArgs)) {
      if (mlir::getConstantIntValue(val))
        continue;

      auto i = static_cast<unsigned>(ind);
      auto addUse = [&](mlir::OpOperand &use) {
        if (op->isProperAncestor(use.getOwner()))
          uses.emplace_back(&use, i);
      };

      for (auto &use : val.getUses())
        addUse(use);

      if (auto cast = val.getDefiningOp<mlir::arith::IndexCastOp>())
        for (auto &use : cast.getIn().getUses())
          addUse(use);
    }

    if (uses.empty())
      return mlir::failure();

    std::array<mlir::Value, 6> dims = {}; // TODO: static vector

    auto loc = op.getLoc();
    rewriter.setInsertionPointToStart(&op.getBody().front());
    auto getDim = [&](unsigned i, mlir::Type type) -> mlir::Value {
      assert(i < dims.size());
      auto dim = dims[i];
      if (!dim) {
        if (i < 3) {
          dim = rewriter.create<mlir::gpu::GridDimOp>(
              loc, static_cast<mlir::gpu::Dimension>(i));
        } else {
          dim = rewriter.create<mlir::gpu::BlockDimOp>(
              loc, static_cast<mlir::gpu::Dimension>(i - 3));
        }
        dims[i] = dim;
      }

      if (type != dim.getType())
        dim = rewriter.create<mlir::arith::IndexCastOp>(loc, type, dim);

      return dim;
    };

    for (auto it : uses) {
      auto *use = it.first;
      auto dim = it.second;
      auto owner = use->getOwner();
      rewriter.modifyOpInPlace(owner, [&]() {
        auto type = use->get().getType();
        auto newVal = getDim(dim, type);
        use->set(newVal);
      });
    }

    return mlir::success();
  }
};

struct SinkGpuDimsPass : public numba::RewriteWrapperPass<SinkGpuDimsPass, void,
                                                          void, SinkGpuDims> {};

struct GPUToLLVMPass
    : public mlir::PassWrapper<GPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToLLVMPass)

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter converter(&context);
    mlir::RewritePatternSet patterns(&context);
    mlir::LLVMConversionTarget target(context);

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);
    mlir::populateGpuToLLVMConversionPatterns(
        converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

    numba::populateControlFlowTypeConversionRewritesAndTarget(converter,
                                                              patterns, target);

    gpu_runtime::populateGpuToLLVMPatternsAndLegality(converter, patterns,
                                                      target);
    numba::populateUtilConversionPatterns(converter, patterns, target);

    // TODO: There were some issues with structural conversion, investigate.
    target.addLegalOp<mlir::arith::SelectOp>();

    auto mod = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

static std::optional<mlir::spirv::Version> mapSpirvVersion(uint16_t major,
                                                           uint16_t minor) {
  if (major == 1) {
    const mlir::spirv::Version mapping[] = {
        mlir::spirv::Version::V_1_0, mlir::spirv::Version::V_1_1,
        mlir::spirv::Version::V_1_2, mlir::spirv::Version::V_1_3,
        mlir::spirv::Version::V_1_4, mlir::spirv::Version::V_1_5,
        mlir::spirv::Version::V_1_6,
    };
    if (minor < std::size(mapping))
      return mapping[minor];
  }
  return std::nullopt;
}

static mlir::spirv::TargetEnvAttr deviceCapsMapper(mlir::gpu::GPUModuleOp op) {
  auto deviceAttr =
      op->getAttrOfType<gpu_runtime::GPURegionDescAttr>(kGpuModuleDeviceName);
  if (!deviceAttr)
    return {};

  auto spirvVersionRet = mapSpirvVersion(deviceAttr.getSpirvMajorVersion(),
                                         deviceAttr.getSpirvMinorVersion());
  if (!spirvVersionRet)
    return nullptr;

  auto spirvVersion = *spirvVersionRet;

  auto context = op.getContext();
  namespace spirv = mlir::spirv;
  spirv::Capability fixedCaps[] = {
      // clang-format off
      spirv::Capability::Addresses,
      spirv::Capability::AtomicFloat32AddEXT,
      spirv::Capability::ExpectAssumeKHR,
      spirv::Capability::GenericPointer,
      spirv::Capability::GroupUniformArithmeticKHR,
      spirv::Capability::Groups,
      spirv::Capability::Int16,
      spirv::Capability::Int64,
      spirv::Capability::Int8,
      spirv::Capability::Kernel,
      spirv::Capability::Linkage,
      spirv::Capability::Vector16,
      // clang-format on
  };
  spirv::Extension exts[] = {spirv::Extension::SPV_EXT_shader_atomic_float_add,
                             spirv::Extension::SPV_KHR_expect_assume};

  llvm::SmallVector<spirv::Capability, 0> caps(std::begin(fixedCaps),
                                               std::end(fixedCaps));

  if (deviceAttr.getHasFp16()) {
    caps.emplace_back(spirv::Capability::Float16);
    caps.emplace_back(spirv::Capability::Float16Buffer);
  }

  if (deviceAttr.getHasFp64())
    caps.emplace_back(spirv::Capability::Float64);

  llvm::sort(caps);
  llvm::sort(exts);

  auto triple = spirv::VerCapExtAttr::get(spirvVersion, caps, exts, context);
  auto attr = spirv::TargetEnvAttr::get(
      triple, spirv::getDefaultResourceLimits(context),
      spirv::ClientAPI::OpenCL, spirv::Vendor::Unknown,
      spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);
  return attr;
}

static void commonOptPasses(mlir::OpPassManager &pm) {
  pm.addPass(numba::createCompositePass(
      "LowerGpuCommonOptPass", [](mlir::OpPassManager &p) {
        p.addPass(mlir::createCSEPass());
        p.addPass(numba::createCommonOptsPass());
      }));
}

static void populateLowerToGPUPipelineRegion(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<InsertGpuRegionPass>());
}

static void populateLowerToGPUPipelineHigh(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<LowerGpuBuiltinsPass>());
  commonOptPasses(pm);
  pm.addPass(mlir::createSymbolDCEPass());
}

static void populateLowerToGPUPipelineMed(mlir::OpPassManager &pm) {
  auto &funcPM = pm.nest<mlir::func::FuncOp>();
  funcPM.addPass(std::make_unique<PrepareForGPUPass>());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(std::make_unique<RemoveNestedParallelPass>());
  funcPM.addPass(mlir::math::createMathUpliftToFMA());
  funcPM.addPass(gpu_runtime::createSortParallelLoopsForGPU());
  funcPM.addPass(gpu_runtime::createTileParallelLoopsForGPUPass());
  funcPM.addPass(gpu_runtime::createInsertGPUGlobalReducePass());
  funcPM.addPass(gpu_runtime::createParallelLoopGPUMappingPass());
  funcPM.addPass(mlir::createParallelLoopToGpuPass());
  funcPM.addPass(std::make_unique<CheckParallelToGpu>());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(gpu_runtime::createLowerGPUGlobalReducePass());
  commonOptPasses(funcPM);
  funcPM.addPass(gpu_runtime::createCreateGPUAllocPass());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(std::make_unique<LowerGpuBuiltins2Pass>());
  funcPM.addPass(gpu_runtime::createGpuDecomposeMemrefsPass());
  funcPM.addPass(mlir::memref::createExpandStridedMetadataPass());
  funcPM.addPass(mlir::createLowerAffinePass());

  funcPM.addPass(gpu_runtime::createMakeBarriersUniformPass());
  commonOptPasses(funcPM);
  funcPM.addPass(std::make_unique<KernelMemrefOpsMovementPass>());
  funcPM.addPass(std::make_unique<SinkGpuDimsPass>());
  funcPM.addPass(std::make_unique<GpuLaunchSinkOpsPass>());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      std::make_unique<GpuPropagateKernelFlagsPass>());
  pm.addPass(std::make_unique<NameGpuModulesPass>());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<GPULowerDefaultLocalSize>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());

  auto &gpuFuncPM =
      pm.nest<mlir::gpu::GPUModuleOp>().nest<mlir::gpu::GPUFuncOp>();
  gpuFuncPM.addPass(mlir::createConvertComplexToStandardPass());
  gpuFuncPM.addPass(mlir::arith::createArithExpandOpsPass());
  gpuFuncPM.addPass(std::make_unique<FlattenScfPass>());
  gpuFuncPM.addPass(std::make_unique<LowerGpuBuiltins3Pass>());
  commonOptPasses(gpuFuncPM);

  pm.addNestedPass<mlir::gpu::GPUModuleOp>(gpu_runtime::createAbiAttrsPass());
  pm.addPass(gpu_runtime::createSetSPIRVCapabilitiesPass(&deviceCapsMapper));
  pm.addPass(gpu_runtime::createTruncateF64ForGPUPass());
  commonOptPasses(pm);
  pm.addPass(gpu_runtime::createGPUToSpirvPass());
  pm.addPass(gpu_runtime::createGpuIndexCastPass());
  commonOptPasses(pm);

  auto &modulePM = pm.nest<mlir::spirv::ModuleOp>();
  modulePM.addNestedPass<mlir::spirv::FuncOp>(
      gpu_runtime::createApplySPIRVFastmathFlags());
  modulePM.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());
  modulePM.addPass(mlir::spirv::createSPIRVUpdateVCEPass());
  pm.addPass(gpu_runtime::createSerializeSPIRVPass());
  pm.addPass(gpu_runtime::createGenDeviceFuncsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      gpu_runtime::createConvertGPUDeallocsPass());
  pm.addNestedPass<mlir::func::FuncOp>(gpu_runtime::createGPUExPass());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<RemoveGpuRegionPass>());
  commonOptPasses(pm);
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<GPUExDeallocPass>());
  pm.addPass(std::make_unique<OutlineInitPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<GenerateOutlineContextPass>());
  commonOptPasses(pm);
}
static void populateLowerToGPUPipelineLow(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<GPUToLLVMPass>());
  commonOptPasses(pm);
}
} // namespace

void registerLowerToGPUPipeline(numba::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto highStage = getHighLoweringStage();
    sink(lowerToGPUPipelineNameRegion(),
         {highStage.begin, plierToScfPipelineName()},
         {highStage.end, plierToLinalgGenPipelineName(),
          plierToStdPipelineName(), untuplePipelineName()},
         {}, &populateLowerToGPUPipelineRegion);

    sink(lowerToGPUPipelineNameHigh(),
         {highStage.begin, plierToStdPipelineName()},
         {highStage.end, plierToLinalgGenPipelineName(), untuplePipelineName()},
         {plierToStdPipelineName()}, &populateLowerToGPUPipelineHigh);

    auto lowStage = getLowerLoweringStage();
    sink(lowerToGPUPipelineNameMed(), {lowStage.begin, untuplePipelineName()},
         {lowStage.end, lowerToGPUPipelineNameLow(),
          preLowerToLLVMPipelineName()},
         {}, &populateLowerToGPUPipelineMed);

    sink(lowerToGPUPipelineNameLow(),
         {lowStage.begin, lowerToGPUPipelineNameMed(),
          preLowerToLLVMPipelineName()},
         {lowStage.end, lowerToLLVMPipelineName()}, {},
         &populateLowerToGPUPipelineLow);
  });
}

llvm::StringRef lowerToGPUPipelineNameRegion() { return "lower_to_gpu_region"; }
llvm::StringRef lowerToGPUPipelineNameHigh() { return "lower_to_gpu_high"; }
llvm::StringRef lowerToGPUPipelineNameMed() { return "lower_to_gpu_med"; }
llvm::StringRef lowerToGPUPipelineNameLow() { return "lower_to_gpu_low"; }
