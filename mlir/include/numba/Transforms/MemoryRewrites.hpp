// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <mlir/IR/BuiltinTypes.h>
#include <optional>

namespace mlir {
class AnalysisManager;
class Pass;
} // namespace mlir

namespace numba {
std::optional<mlir::LogicalResult> optimizeMemoryOps(mlir::AnalysisManager &am);

std::unique_ptr<mlir::Pass> createMemoryOptPass();

/// Normalizes memref types shape and layout to most static one across func
/// call boudaries.
std::unique_ptr<mlir::Pass> createNormalizeMemrefArgsPass();
} // namespace numba
