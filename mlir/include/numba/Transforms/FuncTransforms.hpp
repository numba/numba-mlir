// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace numba {
/// Remove unused functions arguments.
std::unique_ptr<mlir::Pass> createRemoveUnusedArgsPass();
} // namespace numba
