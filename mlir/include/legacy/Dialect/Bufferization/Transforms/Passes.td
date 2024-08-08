//===-- Passes.td - Bufferization pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_LEGACY_PASSES
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_LEGACY_PASSES

include "mlir/Pass/PassBase.td"

def BufferizationBufferize : Pass<"bufferization-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the `bufferization` dialect";
  let constructor = "mlir::bufferization::createBufferizationBufferizePass()";
}

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_LEGACY_PASSES
