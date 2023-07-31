// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir {
class Location;
class ModuleOp;
class PatternRewriter;
class ValueRange;
} // namespace mlir

class PyFuncResolver {
public:
  PyFuncResolver();
  ~PyFuncResolver();

  struct Result {
    mlir::func::FuncOp func;
    llvm::SmallVector<mlir::Value> mappedArgs;
  };

  std::optional<Result> getFunc(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::ModuleOp module,
                                llvm::StringRef name, mlir::ValueRange args,
                                llvm::ArrayRef<llvm::StringRef> kwnames,
                                mlir::ValueRange kwargs) const;

private:
  struct Context;
  std::unique_ptr<Context> context;
};
