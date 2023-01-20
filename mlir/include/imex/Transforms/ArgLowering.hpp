// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/PatternMatch.h>

namespace plier {
class ArgOp;
}

namespace imex {
struct ArgOpLowering : public mlir::OpRewritePattern<plier::ArgOp> {
  ArgOpLowering(mlir::MLIRContext *context);

  mlir::LogicalResult
  matchAndRewrite(plier::ArgOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace imex
