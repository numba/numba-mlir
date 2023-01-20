// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {

class Pass;

/// Uplifts scf to affine. Supports:
/// 1. scf.parallel to affine.parallel.
std::unique_ptr<Pass> createSCFToAffinePass();

} // namespace mlir
