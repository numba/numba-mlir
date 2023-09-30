// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/FuncUtils.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

mlir::func::FuncOp numba::addFunction(mlir::OpBuilder &builder,
                                      mlir::ModuleOp module,
                                      llvm::StringRef name,
                                      mlir::FunctionType type) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Insert before module terminator.
  builder.setInsertionPoint(module.getBody(),
                            std::prev(module.getBody()->end()));
  auto func =
      builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), name, type);
  func.setPrivate();
  return func;
}

numba::AllocaInsertionPoint::AllocaInsertionPoint(mlir::Operation *inst) {
  assert(nullptr != inst);
  auto parent = inst->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();
  assert(parent->getNumRegions() == 1);
  assert(!parent->getRegions().front().empty());
  auto &block = parent->getRegions().front().front();
  assert(!block.empty());
  insertionPoint = &block.front();
}

std::string numba::getUniqueLLVMGlobalName(mlir::ModuleOp mod,
                                           llvm::Twine srcName) {
  for (int i = 0;; ++i) {
    auto name =
        (i == 0 ? srcName.str() : (srcName + "_" + llvm::Twine(i)).str());
    if (!mod.lookupSymbol(name))
      return name;
  }
}

mlir::LLVM::LLVMFuncOp numba::getOrInserLLVMFunc(mlir::OpBuilder &builder,
                                                 mlir::ModuleOp mod,
                                                 llvm::StringRef name,
                                                 mlir::Type type) {
  auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name);
  if (!func) {
    mlir::OpBuilder::InsertionGuard g(builder);
    auto body = mod.getBody();
    builder.setInsertionPoint(body, body->end());
    auto loc = builder.getUnknownLoc();
    return builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, type);
  }
  return func;
}
