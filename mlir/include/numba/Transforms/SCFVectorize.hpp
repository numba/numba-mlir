// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <optional>

namespace mlir {
class OpBuilder;
class Pass;
struct LogicalResult;
namespace scf {
class ParallelOp;
}
} // namespace mlir

namespace numba {
struct SCFVectorizeInfo {
  unsigned factor = 0;
  unsigned count = 0;
};

std::optional<SCFVectorizeInfo> getLoopVectorizeInfo(mlir::scf::ParallelOp loop,
                                                     unsigned dim,
                                                     unsigned vectorBitwidth);

struct SCFVectorizeParams {
  unsigned dim = 0;
  unsigned factor = 0;
};

mlir::LogicalResult vectorizeLoop(mlir::OpBuilder &builder,
                                  mlir::scf::ParallelOp loop,
                                  const SCFVectorizeParams &params);

std::unique_ptr<mlir::Pass> createSCFVectorizePass();
} // namespace numba
