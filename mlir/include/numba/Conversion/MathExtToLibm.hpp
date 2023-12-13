// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace numba {
void populateMathExtToLibmPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createMathExtToLibmPass();
} // namespace numba
