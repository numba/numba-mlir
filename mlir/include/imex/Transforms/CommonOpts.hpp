// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class RewritePatternSet;
class MLIRContext;
class Pass;
} // namespace mlir

namespace imex {
void populateCanonicalizationPatterns(mlir::RewritePatternSet &patterns);

void populateCommonOptsPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createCommonOptsPass();
} // namespace imex
