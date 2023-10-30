// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/SCFVectorize.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>

static unsigned getTypeBitWidth(mlir::Type type) {
  if (mlir::isa<mlir::IndexType>(type))
    return 64; // TODO: unhardcode

  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();

  return 0;
}

static unsigned getArgsTypeWidth(mlir::Operation &op) {
  unsigned ret = 0;
  for (auto arg : op.getOperands())
    ret = std::max(ret, getTypeBitWidth(arg.getType()));

  for (auto res : op.getResults())
    ret = std::max(ret, getTypeBitWidth(res.getType()));

  return ret;
}

static bool isSupportedVectorOp(mlir::Operation &op) {
  return mlir::isa<mlir::arith::ArithDialect, mlir::math::MathDialect>(
      op.getDialect());
}

std::optional<numba::SCFVectorizeInfo>
numba::getLoopVectorizeInfo(mlir::scf::ParallelOp loop, unsigned dim,
                            unsigned vectorBitwidth) {
  assert(dim < loop.getStep().size());
  assert(vectorBitwidth > 0);
  unsigned factor = vectorBitwidth / 8;
  if (factor <= 1)
    return std::nullopt;

  if (!mlir::isConstantIntValue(loop.getStep()[dim], 1))
    return std::nullopt;

  unsigned count = 0;

  for (mlir::Operation &op : loop.getBody()->without_terminator()) {
    if (mlir::isa<mlir::scf::ReduceOp>(op))
      continue;

    if (op.getNumRegions() > 0)
      return std::nullopt;

    if (!isSupportedVectorOp(op))
      continue;

    auto width = getArgsTypeWidth(op);
    if (width == 0)
      continue;

    factor = std::min(factor, vectorBitwidth / width);
    if (factor <= 1)
      return std::nullopt;

    ++count;
  }

  if (count == 0)
    return std::nullopt;

  return SCFVectorizeInfo{factor, count};
}

mlir::LogicalResult
numba::vectorizeLoop(mlir::OpBuilder &builder, mlir::scf::ParallelOp loop,
                     const numba::SCFVectorizeParams &params) {
  auto dim = params.dim;
  auto factor = params.factor;
  assert(dim < loop.getStep().size());
  assert(factor > 1);
  assert(mlir::isConstantIntValue(loop.getStep()[dim], 1));

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(loop);

  auto lower = llvm::to_vector(loop.getLowerBound());
  auto upper = llvm::to_vector(loop.getUpperBound());
  auto step = llvm::to_vector(loop.getStep());

  auto loc = loop.getLoc();

  auto origIndexVar = loop.getInductionVars()[dim];

  mlir::Value factorVal =
      builder.create<mlir::arith::ConstantIndexOp>(loc, factor);

  auto origLower = lower[dim];
  auto origUpper = upper[dim];
  mlir::Value count =
      builder.create<mlir::arith::SubIOp>(loc, origUpper, origLower);
  mlir::Value newCount =
      builder.create<mlir::arith::DivSIOp>(loc, count, factorVal);
  upper[dim] = builder.create<mlir::arith::AddIOp>(loc, origLower, newCount);

  auto newLoop = builder.create<mlir::scf::ParallelOp>(
      loc, lower, upper, step, loop.getInits(), nullptr);
  auto newIndexVar = newLoop.getInductionVars()[dim];

  auto toVectorType = [&](mlir::Type elemType) -> mlir::VectorType {
    int64_t f = factor;
    return mlir::VectorType::get(f, elemType);
  };

  mlir::IRMapping mapping;
  mlir::IRMapping scalarMapping;

  auto getVar = [&](mlir::Value orig) -> mlir::Value {
    if (auto mapped = mapping.lookupOrNull(orig))
      return mapped;

    if (orig == origIndexVar) {
      auto vecType = toVectorType(builder.getIndexType());
      mlir::Value vec =
          builder.create<mlir::ub::PoisonOp>(loc, vecType, nullptr);
      for (auto i : llvm::seq(0u, factor)) {
        mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
        mlir::Value off =
            builder.create<mlir::arith::AddIOp>(loc, origIndexVar, idx);
        vec = builder.create<mlir::vector::InsertElementOp>(loc, off, vec, idx);
      }

      mlir::Value idx = builder.create<mlir::arith::MulIOp>(
          loc, newLoop.getInductionVars()[dim], factorVal);
      idx = builder.create<mlir::vector::SplatOp>(loc, idx, vecType);
      vec = builder.create<mlir::arith::AddIOp>(loc, idx, vec);
      mapping.map(orig, vec);
      return vec;
    }

    auto vecType = toVectorType(orig.getType());
    mlir::Value vec = builder.create<mlir::vector::SplatOp>(loc, orig, vecType);
    mapping.map(orig, vec);
    return vec;
  };

  mlir::Value zeroIndex;
  auto getZeroIndex = [&]() -> mlir::Value {
    if (zeroIndex)
      return zeroIndex;

    zeroIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    return zeroIndex;
  };

  mlir::Value mask;
  auto getMask = [&]() -> mlir::Value {
    if (mask)
      return mask;

    auto vecType = toVectorType(builder.getI1Type());
    mlir::OpFoldResult attr = builder.getIndexAttr(factor);
    mask = builder.create<mlir::vector::CreateMaskOp>(loc, vecType, attr);
    return mask;
  };

  mlir::DominanceInfo dom;
  auto canVectorizeMemOp = [&](auto op) -> bool {
    auto memref = op.getMemRef();
    auto memrefType = mlir::cast<mlir::MemRefType>(memref.getType());
    return dom.properlyDominates(memref, loop) && op.getIndices().size() == 1 &&
           memrefType.getLayout().isIdentity();
  };

  auto canTriviallyVectorizeMemOp = [&](auto op) -> bool {
    return op.getIndices()[0] == origIndexVar;
  };

  builder.setInsertionPointToStart(newLoop.getBody());
  for (mlir::Operation &op : loop.getBody()->without_terminator()) {
    auto extractElem = [&](mlir::Value arg, unsigned i) -> mlir::Value {
      auto loc = op.getLoc();
      mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
      return builder.create<mlir::vector::ExtractElementOp>(loc, arg, idx);
    };

    auto insertElem = [&](mlir::Value arg, mlir::Value val,
                          unsigned i) -> mlir::Value {
      auto loc = op.getLoc();
      mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
      return builder.create<mlir::vector::InsertElementOp>(loc, val, arg, idx);
    };

    if (isSupportedVectorOp(op)) {
      for (auto arg : op.getOperands())
        getVar(arg); // init mapper for op args

      auto newOp = builder.clone(op, mapping);
      for (auto res : newOp->getResults())
        res.setType(toVectorType(res.getType()));

      continue;
    }

    if (auto reduceOp = mlir::dyn_cast<mlir::scf::ReduceOp>(op)) {
      scalarMapping.clear();
      auto &reduceBody = reduceOp.getReductionOperator().front();
      auto reduceTerm =
          mlir::cast<mlir::scf::ReduceReturnOp>(reduceBody.getTerminator());
      assert(reduceBody.getNumArguments() == 2);
      auto lhs = reduceBody.getArgument(0);
      auto rhs = reduceBody.getArgument(1);
      mlir::Value arg = getVar(reduceOp.getOperand());
      mlir::Value reduceVal = extractElem(arg, 0);
      for (auto i : llvm::seq(1u, factor)) {
        mlir::Value val = extractElem(arg, i);
        scalarMapping.map(lhs, reduceVal);
        scalarMapping.map(rhs, val);
        for (auto &redOp : reduceBody.without_terminator()) {
          builder.clone(redOp, scalarMapping);
        }
        reduceVal = scalarMapping.lookupOrDefault(reduceTerm.getResult());
      }
      scalarMapping.clear();
      scalarMapping.map(reduceOp.getOperand(), reduceVal);
      builder.clone(op, scalarMapping);
      continue;
    }

    if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      if (canVectorizeMemOp(loadOp)) {
        auto resType = toVectorType(loadOp.getResult().getType());
        auto memref = loadOp.getMemRef();
        if (canTriviallyVectorizeMemOp(loadOp)) {
          auto vecLoad = builder.create<mlir::vector::LoadOp>(
              op.getLoc(), resType, memref, newIndexVar);
          mapping.map(loadOp.getResult(), vecLoad.getResult());
        } else {
          auto mask = getMask();
          auto indexVec = getVar(loadOp.getIndices()[0]);
          auto init =
              builder.create<mlir::ub::PoisonOp>(op.getLoc(), resType, nullptr);

          auto gather = builder.create<mlir::vector::GatherOp>(
              op.getLoc(), resType, memref, getZeroIndex(), indexVec, mask,
              init);
          mapping.map(loadOp.getResult(), gather.getResult());
        }
        continue;
      }
    }

    if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
      if (canVectorizeMemOp(storeOp)) {
        auto memref = storeOp.getMemRef();
        auto value = getVar(storeOp.getValueToStore());
        if (canTriviallyVectorizeMemOp(storeOp)) {
          builder.create<mlir::vector::StoreOp>(op.getLoc(), value, memref,
                                                newIndexVar);
        } else {
          auto mask = getMask();
          auto indexVec = getVar(storeOp.getIndices()[0]);

          builder.create<mlir::vector::ScatterOp>(
              op.getLoc(), memref, getZeroIndex(), indexVec, mask, value);
        }
        continue;
      }
    }

    // Fallback: Failed to vectorize op, just duplicate it `factor` times
    scalarMapping.clear();
    llvm::SmallVector<mlir::Value> args;
    for (auto arg : op.getOperands())
      args.emplace_back(getVar(arg));

    llvm::SmallVector<mlir::Value> results;
    for (auto res : op.getResultTypes())
      results.emplace_back(builder.create<mlir::ub::PoisonOp>(
          op.getLoc(), toVectorType(res), nullptr));

    for (auto i : llvm::seq(0u, factor)) {
      for (auto &&[j, arg] : llvm::enumerate(op.getOperands()))
        scalarMapping.map(arg, extractElem(args[j], i));

      auto newOp = builder.clone(op, scalarMapping);
      for (auto &&[j, res] : llvm::enumerate(newOp->getResults()))
        results[j] = insertElem(results[j], res, i);
    }

    for (auto &&[i, res] : llvm::enumerate(op.getResults()))
      mapping.map(res, results[i]);
  }

  builder.setInsertionPoint(loop);
  mlir::Value newLower =
      builder.create<mlir::arith::MulIOp>(loc, newCount, factorVal);
  newLower = builder.create<mlir::arith::AddIOp>(loc, origLower, newLower);

  auto lowerCopy = llvm::to_vector(loop.getLowerBound());
  lowerCopy[dim] = newLower;
  loop.getLowerBoundMutable().assign(lowerCopy);
  loop.getInitValsMutable().assign(newLoop.getResults());
  return mlir::success();
}

namespace {
struct SCFVectorizePass
    : public mlir::PassWrapper<SCFVectorizePass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFVectorizePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::ub::UBDialect>();
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    llvm::SmallVector<
        std::pair<mlir::scf::ParallelOp, numba::SCFVectorizeParams>>
        toVectorize;

    getOperation()->walk([&](mlir::scf::ParallelOp loop) {
      unsigned dim = 0;
      auto info = numba::getLoopVectorizeInfo(loop, dim, 256);
      if (!info)
        return;

      toVectorize.emplace_back(loop,
                               numba::SCFVectorizeParams{dim, info->factor});
    });

    if (toVectorize.empty())
      return markAllAnalysesPreserved();

    mlir::OpBuilder builder(&getContext());
    for (auto &&[loop, params] : toVectorize) {
      builder.setInsertionPoint(loop);
      if (mlir::failed(numba::vectorizeLoop(builder, loop, params)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createSCFVectorizePass() {
  return std::make_unique<SCFVectorizePass>();
}
