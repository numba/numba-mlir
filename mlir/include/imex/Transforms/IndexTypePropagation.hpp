// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class RewritePatternSet;
class MLIRContext;
} // namespace mlir

namespace imex {
void populateIndexPropagatePatterns(mlir::RewritePatternSet &patterns);
}
