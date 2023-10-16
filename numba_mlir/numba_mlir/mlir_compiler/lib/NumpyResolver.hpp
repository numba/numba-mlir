// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <optional>
#include <string>

#include <llvm/ADT/SmallVector.h>

namespace llvm {
class StringRef;
}

namespace mlir {
class ArrayAttr;
class Location;
class OpBuilder;
class Value;
class ValueRange;
struct LogicalResult;
} // namespace mlir

enum class PrimitiveType { Default = 0, View = 1, SideEffect = 2 };

class NumpyResolver {
public:
  NumpyResolver(const char *modName, const char *mapName);
  ~NumpyResolver();

  bool hasFunc(llvm::StringRef name) const;

  mlir::LogicalResult
  resolveFuncArgs(mlir::OpBuilder &builder, mlir::Location loc,
                  llvm::StringRef name, mlir::ValueRange args,
                  mlir::ArrayAttr argsNames,
                  llvm::SmallVectorImpl<mlir::Value> &resultArgs,
                  llvm::SmallVectorImpl<mlir::Value> &outArgs,
                  PrimitiveType &primitive_type);

private:
  class Impl;

  std::unique_ptr<Impl> impl;
};
