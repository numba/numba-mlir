// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Analysis/AliasAnalysis.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>

#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"

#include <functional>

static bool hasCopyBody(mlir::linalg::GenericOp &Op) {
  auto *body = Op.getBody();
  if (llvm::hasSingleElement(*body)) {
    if (auto yield =
            mlir::dyn_cast_or_null<mlir::linalg::YieldOp>(body->front())) {
      auto operands = yield.getOperands();
      if (operands[0] == body->getArgument(0))
        return true;
    }
  }

  return false;
};

static bool hasIdentityMap(mlir::linalg::GenericOp &Op) {
  auto maps = Op.getIndexingMaps();
  if (maps.size() != 2)
    return false;

  bool trivialAffineMaps = true;
  for (auto &&map : maps)
    trivialAffineMaps &=
        map.cast<mlir::AffineMapAttr>().getValue().isIdentity();

  return trivialAffineMaps;
};

static bool isLinalgGenericCopy(mlir::linalg::GenericOp &Op) {
  auto srcType = Op.getOperands()[0].getType();
  auto dstType = Op.getOutputs()[0].getType();
  // need to be sure it is not type cast/strides elimination
  return hasCopyBody(Op) && hasIdentityMap(Op) &&
         mlir::memref::CastOp::areCastCompatible(srcType, dstType);
}

static bool isCopy(mlir::Operation &op) {
  if (mlir::isa<numba::ntensor::CopyOp>(op))
    return true;

  if (auto lageneric = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
    return isLinalgGenericCopy(lageneric);

  return false;
}

static bool isTensor(mlir::Value val) {
  return mlir::isa_and_nonnull<mlir::MemRefType, numba::ntensor::NTensorType>(
      val.getType());
}

static bool isLocallyAllocated(mlir::Value value) {
  return mlir::isa_and_nonnull<numba::ntensor::CreateArrayOp,
                               mlir::memref::AllocOp, mlir::memref::AllocaOp>(
      value.getDefiningOp());
}

static mlir::Value CastTensor(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value src, mlir::Value dst) {
  if (mlir::isa_and_nonnull<numba::ntensor::NTensorType>(src.getType()))
    return builder.create<numba::ntensor::CastOp>(loc, dst.getType(), src);
  if (mlir::isa_and_nonnull<mlir::MemRefType>(src.getType()))
    return builder.create<mlir::memref::CastOp>(loc, dst.getType(), src);

  llvm_unreachable("Unknown tensor type");
}

static mlir::Value getCopyOpSource(mlir::Operation &op) {
  if (auto copy = mlir::dyn_cast<numba::ntensor::CopyOp>(op))
    return copy.getSource();

  if (auto lageneric = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
    assert(isLinalgGenericCopy(lageneric));
    return lageneric.getOperands()[0];
  }

  llvm_unreachable("Unknown tensor type");
}

static mlir::Value getCopyOpTarget(mlir::Operation &op) {
  if (auto copy = mlir::dyn_cast<numba::ntensor::CopyOp>(op))
    return copy.getTarget();

  if (auto lageneric = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
    assert(isLinalgGenericCopy(lageneric));
    return lageneric.getOutputs()[0];
  }

  llvm_unreachable("Unknown tensor type");
}

namespace numba {
struct CopyRemovalPass
    : public mlir::PassWrapper<CopyRemovalPass,
                               mlir::InterfacePass<mlir::FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyRemovalPass);

  void runOnOperation() override {
    auto root = this->getOperation();

    llvm::SmallVector<mlir::Operation *> reads;
    llvm::SmallVector<mlir::Operation *> writes;
    llvm::SmallVector<mlir::Operation *> copies;

    root->walk([&](mlir::Operation *op) {
      if (isCopy(*op)) {
        copies.emplace_back(op);
        reads.emplace_back(op);
        writes.emplace_back(op);
        return;
      }

      // Only process ops operation on operands known to be tensor arrays.
      if (!llvm::any_of(op->getOperands(),
                        [](mlir::Value arg) { return isTensor(arg); }))
        return;

      auto memEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
      if (!memEffects) {
        // Conservatively assume them both readers and writers.
        reads.emplace_back(op);
        writes.emplace_back(op);
        return;
      }

      if (memEffects.hasEffect<mlir::MemoryEffects::Read>())
        reads.emplace_back(op);

      if (memEffects.hasEffect<mlir::MemoryEffects::Write>())
        writes.emplace_back(op);
    });

    if (copies.empty())
      return this->markAllAnalysesPreserved();

    auto &dom = this->template getAnalysis<mlir::DominanceInfo>();
    auto &postDom = this->template getAnalysis<mlir::PostDominanceInfo>();
    auto &aa = this->template getAnalysis<numba::LocalAliasAnalysis>();

    auto inBetween = [&](mlir::Operation *op, mlir::Operation *begin,
                         mlir::Operation *end) -> bool {
      if (postDom.postDominates(begin, op))
        return false;

      if (dom.dominates(end, op))
        return false;

      return true;
    };

    auto hasAliasingWrites = [&](mlir::Value array, mlir::Operation *begin,
                                 mlir::Operation *end) -> bool {
      for (auto write : writes) {
        if (!inBetween(write, begin, end))
          continue;

        for (auto arg : write->getOperands()) {
          if (!isTensor(arg))
            continue;

          if (!aa.alias(array, arg).isNo())
            return true;
        }
      }
      return false;
    };

    mlir::OpBuilder builder(&this->getContext());

    // Propagate copy src.
    for (auto copy : copies) {
      auto src = getCopyOpSource(*copy);
      auto dst = getCopyOpTarget(*copy);
      for (auto &use : llvm::make_early_inc_range(dst.getUses())) {
        auto owner = use.getOwner();
        if (owner == copy || !dom.properlyDominates(copy, owner))
          continue;

        if (hasAliasingWrites(dst, copy, owner))
          continue;

        auto memInterface =
            mlir::dyn_cast<mlir::MemoryEffectOpInterface>(owner);
        if (!memInterface ||
            !memInterface.template getEffectOnValue<mlir::MemoryEffects::Read>(
                dst) ||
            memInterface.template getEffectOnValue<mlir::MemoryEffects::Write>(
                dst)) {
          continue;
        }

        mlir::Value newArg = src;
        if (src.getType() != dst.getType()) {
          auto loc = owner->getLoc();
          builder.setInsertionPoint(owner);
          newArg = CastTensor(builder, loc, newArg, dst);
        }

        use.set(newArg);
      }
    }

    auto getNextCopy = [&](mlir::Operation *src) -> mlir::Operation * {
      for (auto copy : copies) {
        if (src == copy)
          continue;

        if (!dom.properlyDominates(src, copy))
          continue;

        if (getCopyOpTarget(*src) == getCopyOpTarget(*copy))
          return copy;
      }

      return {};
    };

    auto hasAliasingReads = [&](mlir::Value array, mlir::Operation *begin,
                                mlir::Operation *end) -> bool {
      for (auto read : reads) {
        if (!inBetween(read, begin, end))
          continue;

        for (auto arg : read->getOperands()) {
          if (!isTensor(arg))
            continue;

          if (!aa.alias(array, arg).isNo())
            return true;
        }
      }
      return false;
    };

    auto isResultUnused = [&](mlir::Value array) {
      if (!array.hasOneUse())
        return false;

      return isLocallyAllocated(array);
    };

    llvm::SmallVector<mlir::Operation *> toErase;

    // Remove redundant copies.
    for (auto copy : copies) {
      auto dst = getCopyOpTarget(*copy);
      if (isResultUnused(dst)) {
        toErase.emplace_back(copy);
        toErase.emplace_back(dst.getDefiningOp());

        continue;
      }

      auto nextCopy = getNextCopy(copy);
      if (!nextCopy)
        continue;

      if (hasAliasingReads(dst, copy, nextCopy))
        continue;

      toErase.emplace_back(copy);
    }

    for (auto op : toErase)
      op->erase();
  }
};

std::unique_ptr<mlir::Pass> createCopyRemovalPass() {
  return std::unique_ptr<mlir::Pass>(new CopyRemovalPass());
}
} // namespace numba
