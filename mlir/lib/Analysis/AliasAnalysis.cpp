// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Analysis/AliasAnalysis.hpp"

#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

/// Check if value is function argument.
static bool isFuncArg(mlir::Value val) {
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val);
  if (!blockArg)
    return false;

  return mlir::isa_and_nonnull<mlir::FunctionOpInterface>(
      blockArg.getOwner()->getParentOp());
}

/// Check if value has "restrict" attribute. Value must be a function argument.
static bool isRestrict(mlir::Value val) {
  auto blockArg = val.cast<mlir::BlockArgument>();
  auto func =
      mlir::cast<mlir::FunctionOpInterface>(blockArg.getOwner()->getParentOp());
  return !!func.getArgAttr(blockArg.getArgNumber(),
                           numba::getRestrictArgName());
}

static bool isReferenceType(mlir::Type type) {
  return mlir::isa<mlir::MemRefType, mlir::TensorType,
                   numba::ntensor::NTensorType>(type);
}

static bool isReferenceType(mlir::Value val) {
  return isReferenceType(val.getType());
}

static std::optional<mlir::AliasResult> checkLinalgImpl(mlir::Value val) {
  auto op = val.getDefiningOp();
  if (mlir::isa_and_nonnull<mlir::linalg::GenericOp, mlir::tensor::EmptyOp>(op))
    return mlir::AliasResult::NoAlias;

  return std::nullopt;
}

static std::optional<mlir::AliasResult> checkLinalg(mlir::Value val1,
                                                    mlir::Value val2) {
  if (auto res = checkLinalgImpl(val1))
    return res;

  if (auto res = checkLinalgImpl(val2))
    return res;

  return std::nullopt;
}

mlir::AliasResult numba::LocalAliasAnalysis::aliasImpl(mlir::Value lhs,
                                                       mlir::Value rhs) {
  if (lhs == rhs)
    return mlir::AliasResult::MustAlias;

  if (!isReferenceType(lhs) || !isReferenceType(rhs))
    return mlir::AliasResult::NoAlias;

  // TODO: unhardcode
  if (auto res = checkLinalg(rhs, lhs))
    return *res;

  // Assume no aliasing if both values are function arguments and any of them
  // have restrict attr.
  if (isFuncArg(lhs) && isFuncArg(rhs))
    if (isRestrict(lhs) || isRestrict(rhs))
      return mlir::AliasResult::NoAlias;

  return mlir::LocalAliasAnalysis::aliasImpl(lhs, rhs);
}

numba::AliasAnalysis::AliasAnalysis(mlir::Operation *op)
    : mlir::AliasAnalysis(op) {
  addAnalysisImplementation(numba::LocalAliasAnalysis());
}

llvm::StringRef numba::getRestrictArgName() { return "numba.restrict"; }

bool numba::isWriter(mlir::Operation &op,
                     llvm::SmallVectorImpl<mlir::Value> &args) {
  if (auto func = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
    llvm::append_range(args, func.getArgOperands());
    return true;
  }

  auto memInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!memInterface) {
    llvm::append_range(args, op.getOperands());
    return true;
  }
  if (memInterface.hasEffect<mlir::MemoryEffects::Write>()) {
    for (auto arg : op.getOperands()) {
      if (memInterface.getEffectOnValue<mlir::MemoryEffects::Write>(arg))
        args.emplace_back(arg);
    }
    return true;
  }
  return false;
}
