// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <mlir/IR/BuiltinTypes.h>
#include <optional>

namespace mlir {
class OpBuilder;
class Pass;
namespace scf {
class ParallelOp;
}
} // namespace mlir

namespace numba {

/// Loop vectorization info
struct SCFVectorizeInfo {
  /// Loop dimension on which to vectorize.
  unsigned dim = 0;

  /// Biggest vector width, in elements.
  unsigned factor = 0;

  /// Number of ops, which will be vectorized.
  unsigned count = 0;

  /// Can use masked vector ops for our of bounds memory accesses.
  bool masked = false;
};

/// Collect vectorization statistics on specified `scf.parallel` dimension.
/// Return `SCFVectorizeInfo` or `std::nullopt` if loop cannot be vectorized on
/// specified dimension.
///
/// `vectorBitwidth` - maximum vector size, in bits.
std::optional<SCFVectorizeInfo> getLoopVectorizeInfo(mlir::scf::ParallelOp loop,
                                                     unsigned dim,
                                                     unsigned vectorBitwidth);

/// Vectorization params
struct SCFVectorizeParams {
  /// Loop dimension on which to vectorize.
  unsigned dim = 0;

  /// Desired vector length, in elements
  unsigned factor = 0;

  /// Use masked vector ops for memory access outside loop bounds.
  bool masked = false;
};

/// Vectorize loop on specified dimension with specified factor.
///
/// If `masked` is `true` and loop bound is not divisible by `factor`, instead
/// of generating second loop to process remainig iterations, extend loop count
/// and generate masked vector ops to handle out-of bounds memory accesses.
mlir::LogicalResult vectorizeLoop(mlir::OpBuilder &builder,
                                  mlir::scf::ParallelOp loop,
                                  const SCFVectorizeParams &params);

std::unique_ptr<mlir::Pass> createSCFVectorizePass();
} // namespace numba
