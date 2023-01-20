// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

#include <llvm/ADT/Optional.h>

namespace mlir {
class AnalysisManager;
class Pass;
struct LogicalResult;
} // namespace mlir

namespace imex {
llvm::Optional<mlir::LogicalResult>
optimizeMemoryOps(mlir::AnalysisManager &am);

std::unique_ptr<mlir::Pass> createMemoryOptPass();
} // namespace imex
