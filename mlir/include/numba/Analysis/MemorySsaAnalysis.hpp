// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "numba/Analysis/MemorySsa.hpp"

#include <mlir/Pass/AnalysisManager.h>

namespace mlir {
class Operation;
class AliasAnalysis;
} // namespace mlir

namespace numba {
class MemorySSAAnalysis {
public:
  MemorySSAAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);
  MemorySSAAnalysis(const MemorySSAAnalysis &) = delete;

  mlir::LogicalResult optimizeUses();

  static bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa);

  llvm::Optional<numba::MemorySSA> memssa;
  mlir::AliasAnalysis *aliasAnalysis = nullptr;
};
} // namespace numba
