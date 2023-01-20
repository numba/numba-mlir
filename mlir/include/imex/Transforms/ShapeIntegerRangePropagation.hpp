// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace imex {
/// Propagate integer range info through the IR and optimize ops based on this
/// info.
std::unique_ptr<mlir::Pass> createShapeIntegerRangePropagationPass();
} // namespace imex
