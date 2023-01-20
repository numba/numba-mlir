// SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class ConversionTarget;
class MLIRContext;
class Pass;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace imex {
void populateNtensorToMemrefRewritesAndTarget(mlir::TypeConverter &converter,
                                              mlir::RewritePatternSet &patterns,
                                              mlir::ConversionTarget &target);

std::unique_ptr<mlir::Pass> createNtensorToMemrefPass();
} // namespace imex
