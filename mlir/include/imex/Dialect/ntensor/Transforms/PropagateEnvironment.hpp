// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace imex {
namespace ntensor {
/// Propagate enviroment attribute through the ntensor ops using dataflow
/// analysis.
std::unique_ptr<mlir::Pass> createPropagateEnvironmentPass();
} // namespace ntensor
} // namespace imex
