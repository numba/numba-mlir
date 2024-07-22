//===- Bufferize.cpp - Bufferization for `tensor` dialect ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of `tensor` dialect ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

#include "legacy/Dialect/Bufferization/Transforms/Passes.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_BUFFERIZATIONBUFFERIZE
#include "legacy/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir::bufferization;

namespace {
struct BufferizationBufferizePass
    : public mlir::bufferization::impl::BufferizationBufferizeBase<
          BufferizationBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<BufferizationDialect>();

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::bufferization::BufferizationDialect,
                    mlir::memref::MemRefDialect>();
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
mlir::bufferization::legacy::createBufferizationBufferizePass() {
  return std::make_unique<BufferizationBufferizePass>();
}
