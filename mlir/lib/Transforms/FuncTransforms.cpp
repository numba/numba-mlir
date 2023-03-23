// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/FuncTransforms.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace {
static bool isArgUsed(mlir::BlockArgument val, mlir::func::FuncOp parentFunc) {
  for (auto &use : val.getUses()) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getOwner());
    if (!call)
      return true;

    if (call.getCallee() != parentFunc.getName() ||
        val.getArgNumber() != use.getOperandNumber())
      return true;
  }

  return false;
}

struct RemoveUnusedArgsPass
    : public mlir::PassWrapper<RemoveUnusedArgsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveUnusedArgsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    bool changed = false;

    llvm::BitVector removeArgs;
    llvm::SmallVector<mlir::Value> newArgs;
    llvm::SmallVector<mlir::Type> newArgTypes;

    bool repeat = true;

    while (repeat) {
      repeat = false;
      module->walk([&](mlir::func::FuncOp func) {
        if (func.isDeclaration() || func.isPublic())
          return;

        assert(!func.getBody().empty());
        auto &block = func.getBody().front();
        removeArgs.resize(block.getNumArguments());

        newArgTypes.clear();
        bool hasUnused = false;
        for (auto [i, arg] : llvm::enumerate(block.getArguments())) {
          auto isUsed = isArgUsed(arg, func);
          removeArgs[i] = !isUsed;
          hasUnused = hasUnused || !isUsed;
          if (isUsed)
            newArgTypes.emplace_back(arg.getType());
        }

        if (!hasUnused)
          return;

        auto funcUses = mlir::SymbolTable::getSymbolUses(func, module);
        if (!funcUses)
          return;

        for (auto use : *funcUses) {
          if (!mlir::isa<mlir::func::CallOp>(use.getUser()))
            return;
        }

        for (auto use : llvm::make_early_inc_range(*funcUses)) {
          auto call = mlir::cast<mlir::func::CallOp>(use.getUser());
          newArgs.clear();
          assert(call.getOperands().size() == removeArgs.size());
          for (auto [i, arg] : llvm::enumerate(call.getOperands())) {
            if (!removeArgs[i])
              newArgs.emplace_back(arg);
          }

          call->setOperands(newArgs);
        }

        block.eraseArguments(removeArgs);
        func.eraseArguments(removeArgs);
        changed = true;
        repeat = true;
      });
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createRemoveUnusedArgsPass() {
  return std::make_unique<RemoveUnusedArgsPass>();
}
