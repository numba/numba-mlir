// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace numba {
/// Convert arith ops according to provided type converter.
void populateArithConversionRewritesAndTarget(mlir::TypeConverter &converter,
                                              mlir::RewritePatternSet &patterns,
                                              mlir::ConversionTarget &target);

/// Convert math ops according to provided type converter.
void populateMathConversionRewritesAndTarget(mlir::TypeConverter &converter,
                                             mlir::RewritePatternSet &patterns,
                                             mlir::ConversionTarget &target);

/// Convert complex ops according to provided type converter.
void populateComplexConversionRewritesAndTarget(
    mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target);
} // namespace numba
