// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class MLIRContext;
class RewritePatternSet;
} // namespace mlir

namespace imex {
void populateIfRewritesPatterns(mlir::RewritePatternSet &patterns);

} // namespace imex
