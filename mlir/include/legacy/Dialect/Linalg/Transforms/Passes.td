//===-- Passes.td - Linalg pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_LEGACY_PASSES
#define MLIR_DIALECT_LINALG_LEGACY_PASSES

include "mlir/Pass/PassBase.td"

def LinalgBufferizePass : Pass<"linalg-bufferize"> {
  let summary = "Bufferize the linalg dialect";
  let dependentDialects = [
    "affine::AffineDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
  ];
}

#endif // MLIR_DIALECT_LINALG_LEGACY_PASSES
