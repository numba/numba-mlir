// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace imex {
/// Tries to promote loads/stores in scf.for to loop-carried variables.
std::unique_ptr<mlir::Pass> createCanonicalizeReductionsPass();
} // namespace imex
