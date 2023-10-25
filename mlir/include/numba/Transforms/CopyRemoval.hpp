// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "numba/Analysis/AliasAnalysis.hpp"

#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>

#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"

#include <functional>

namespace numba {
std::unique_ptr<mlir::Pass> createCopyRemovalPass();
} // namespace numba
